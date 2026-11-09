// H3: host CPU per-op profile for one decode step at a given context length.
//
// Times every non-trivial decode op of a Qwen3-layout model (default
// mini-coder-4b shapes) on this CPU, the way llama.cpp's CPU backend would run
// it: fp32 activations, fp16 KV cache, int8 weights, OpenMP threads.
// KV and weights are allocated per layer and visited layer by layer so they
// come from DRAM, not cache, as they do in a real decode step.
//
// Output (stdout, TSV): kind  ctx  threads  seconds_per_op_instance
//
// Build: gcc -O3 -march=native -fopenmp cpu_ops.c -o cpu_ops -lm
// Run:   OMP_PROC_BIND=close ./cpu_ops <threads> <ctx>...

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { D = 2560, F = 9728, H = 32, HKV = 8, DH = 128, LAYERS = 36, V = 151936 };
#define QD (H * DH)
#define KVD (HKV * DH)

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// Filled with 0x2C 0x21 bytes: as fp16 that is 0.0103, as fp32 ~2e-9 (normal), as
// int8 small values. Never denormal, which would make FMAs ~100x slower.
static void *xalloc(size_t n) {
    void *p = aligned_alloc(64, (n + 63) / 64 * 64);
    if (!p) { fprintf(stderr, "alloc %zu failed\n", n); exit(1); }
    uint16_t *h = p;
    for (size_t i = 0; i < n / 2; i++) h[i] = 0x211F;
    return p;
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float s = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) o[i] = x[i] * s * w[i];
}

static void rope(float *v, int heads, int pos) {
    for (int h = 0; h < heads; h++)
        for (int i = 0; i < DH / 2; i++) {
            float th = pos * powf(5e6f, -2.0f * i / DH);
            float c = cosf(th), s = sinf(th);
            float a = v[h * DH + i], b = v[h * DH + i + DH / 2];
            v[h * DH + i] = a * c - b * s;
            v[h * DH + i + DH / 2] = a * s + b * c;
        }
}

static inline float dot_f16(const float *q, const uint16_t *k) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < DH; i += 8) {
        __m256 kv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(k + i)));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), kv, acc);
    }
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// scores[h][t] = q_h . K[t][h/group]
static void attn_score(float *sc, const float *q, const uint16_t *K, int L) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H; h++)
        for (int t = 0; t < L; t++)
            sc[(size_t)h * L + t] = dot_f16(q + h * DH, K + (size_t)t * KVD + (h / (H / HKV)) * DH);
}

static void softmax(float *sc, int L) {
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        float *r = sc + (size_t)h * L, m = -1e30f, s = 0;
        for (int t = 0; t < L; t++) m = r[t] > m ? r[t] : m;
        for (int t = 0; t < L; t++) { r[t] = expf(r[t] - m); s += r[t]; }
        for (int t = 0; t < L; t++) r[t] /= s;
    }
}

// o[h][:] = sum_t p[h][t] * V[t][h/group][:]
static void attn_value(float *o, const float *p, const uint16_t *Vc, int L) {
#pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        __m256 acc[DH / 8];
        for (int j = 0; j < DH / 8; j++) acc[j] = _mm256_setzero_ps();
        const uint16_t *vb = Vc + (h / (H / HKV)) * DH;
        for (int t = 0; t < L; t++) {
            __m256 pt = _mm256_set1_ps(p[(size_t)h * L + t]);
            const uint16_t *vr = vb + (size_t)t * KVD;
            for (int j = 0; j < DH / 8; j++)
                acc[j] = _mm256_fmadd_ps(pt, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vr + 8 * j))), acc[j]);
        }
        for (int j = 0; j < DH / 8; j++) _mm256_storeu_ps(o + h * DH + 8 * j, acc[j]);
    }
}

// int8 GEMV: y[M] = W[M][N] x[N]
static void gemv_i8(int32_t *y, const int8_t *W, const int8_t *x, int M, int N) {
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const int8_t *w = W + (size_t)m * N;
        int32_t s = 0;
        for (int n = 0; n < N; n++) s += (int32_t)w[n] * x[n];
        y[m] = s;
    }
}

static void triad(float *A, const float *B, const float *C, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) A[i] = B[i] + 0.5f * C[i];
}

#define TIME(label, reps, body)                                           \
    do {                                                                  \
        double best = 1e30;                                               \
        for (int trial = 0; trial < 5; trial++) {                         \
            double t0 = now();                                            \
            for (int r = 0; r < (reps); r++) { body; }                    \
            double dt = (now() - t0) / (reps);                            \
            if (dt < best) best = dt;                                     \
        }                                                                 \
        printf("%s\t%d\t%d\t%.9e\n", label, L, nth, best);                \
        fflush(stdout);                                                   \
    } while (0)

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s threads ctx...\n", argv[0]); return 1; }
    int nth = atoi(argv[1]);
    omp_set_num_threads(nth);
    _mm_setcsr(_mm_getcsr() | 0x8040);   // flush-to-zero and denormals-are-zero, as llama.cpp
#pragma omp parallel
    _mm_setcsr(_mm_getcsr() | 0x8040);

    float *x = xalloc(sizeof(float) * F), *y = xalloc(sizeof(float) * F), *w = xalloc(sizeof(float) * F);
    float *q = xalloc(sizeof(float) * QD), *k = xalloc(sizeof(float) * KVD), *o = xalloc(sizeof(float) * QD);
    for (int i = 0; i < F; i++) { x[i] = 0.01f * (i % 97); w[i] = 1.0f; y[i] = 0.5f; }
    for (int i = 0; i < QD; i++) { q[i] = 0.01f * (i % 13); o[i] = 0.0f; }
    for (int i = 0; i < KVD; i++) k[i] = 0.02f * (i % 7);

    for (int a = 2; a < argc; a++) {
        int L = atoi(argv[a]);
        size_t kv_layer = (size_t)L * KVD;
        // One K and one V buffer per layer, so a step streams them from DRAM.
        int nl = L <= 4096 ? LAYERS : 4;   // cycling 4 layers already exceeds the LLC for big L
        uint16_t **K = malloc(nl * sizeof *K), **Vc = malloc(nl * sizeof *Vc);
        for (int l = 0; l < nl; l++) { K[l] = xalloc(2 * kv_layer); Vc[l] = xalloc(2 * kv_layer); }
        float *sc = xalloc(sizeof(float) * (size_t)H * L);
        int reps_attn = L < 4096 ? 200 : (L < 32768 ? 20 : 4);
        int li = 0;

        TIME("attn_score", reps_attn, { attn_score(sc, q, K[li], L); li = (li + 1) % nl; });
        TIME("softmax", reps_attn, { softmax(sc, L); });
        TIME("attn_value", reps_attn, { attn_value(o, sc, Vc[li], L); li = (li + 1) % nl; });

        if (a == 2) {   // context-independent ops: time once
            TIME("rmsnorm", 20000, { rmsnorm(y, x, w, D); });
            TIME("qk_norm", 20000, { for (int h = 0; h < H; h++) rmsnorm(q + h * DH, q + h * DH, w, DH);
                                     for (int h = 0; h < HKV; h++) rmsnorm(k + h * DH, k + h * DH, w, DH); });
            TIME("rope", 2000, { rope(q, H, L); rope(k, HKV, L); });
            TIME("act", 20000, { for (int i = 0; i < F; i++) y[i] = x[i] / (1.0f + expf(-x[i])) * w[i]; });
            TIME("residual", 20000, { for (int i = 0; i < D; i++) y[i] += x[i]; });
            TIME("kv_append", 20000, { for (int i = 0; i < KVD; i++) ((uint16_t *)sc)[i] = (uint16_t)k[i];
                                       for (int i = 0; i < KVD; i++) ((uint16_t *)sc)[KVD + i] = (uint16_t)k[i]; });
            // Static-weight GEMVs from DRAM, int8 weights.
            struct { const char *n; int M, N; } g[] = {
                {"q_proj", QD, D}, {"k_proj", KVD, D}, {"v_proj", KVD, D}, {"o_proj", D, QD},
                {"gate_proj", F, D}, {"up_proj", F, D}, {"down_proj", D, F}, {"lm_head", V, D}};
            int8_t *xi = xalloc(F);
            int32_t *yi = xalloc(sizeof(int32_t) * V);
            for (int i = 0; i < 8; i++) {
                size_t bytes = (size_t)g[i].M * g[i].N;
                int copies = bytes > (64u << 20) ? 1 : (int)((256u << 20) / bytes);
                int8_t *Wb = xalloc(bytes * copies);
                int ci = 0;
                TIME(g[i].n, copies * 2, { gemv_i8(yi, Wb + bytes * ci, xi, g[i].M, g[i].N); ci = (ci + 1) % copies; });
                free(Wb);
            }
            // Stream bandwidth (triad), for the roofline fit.
            size_t n = 64u << 20;
            float *A = xalloc(4 * n), *B = xalloc(4 * n), *C = xalloc(4 * n);
            TIME("triad_64M", 5, { triad(A, B, C, n); });
            free(A); free(B); free(C); free(xi); free(yi);
        }
        for (int l = 0; l < nl; l++) { free(K[l]); free(Vc[l]); }
        free(K); free(Vc); free(sc);
    }
    return 0;
}
