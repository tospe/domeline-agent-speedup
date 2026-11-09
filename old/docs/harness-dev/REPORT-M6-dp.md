# REPORT M6 (capped DP ground truth) — Agent C

Branch `h/m6-dp` (worktree `.worktrees/m6-dp`, base autopilot 4ed392d).

Files:
- new: `harness/dp_caps.py`, `experiments/dp_caps.py`, `tests/test_dp_caps.py` (9 tests);
- changed: `harness/large_space.py` (`mapping_dp(lam=...)`, `xbar_arrays`, `DPResult.arrays/objective/lam`);
- results: `results/large/dp_caps_main.json`, `dp_caps_main_plain.json`, `dp_caps_hulls_main.json`,
  and the edge-on-VFU cross-check `dp_caps_edge.json` / `dp_caps_hulls_edge.json` (see §5).

No model equation changed. `mapping_dp` adds `lam * arrays` to the DP objective only. Times, readings and areas
all come from the unchanged evaluator.

## 1. Plan (as executed)

1. Add a Lagrangian penalty to the M3 mapping DP: minimize LLM time + λ·arrays with hardware fixed. Test it
   against brute force on 2 layers.
2. Per hardware setting, sweep λ parametrically to get the exact vertices of the lower convex hull of
   (arrays, time).
3. Per (area cap, effort cap):
   - lower bound = the hull at the cap, minimized over settings;
   - upper bound = the best exactly evaluated, verifier-legal design;
   - gap = their difference.
4. If the gap is material, add an exact area dimension. **It is not (max 0.046%); see §3.**
5. Tighten the certificate: per-hardware DP minimum over all settings.

## 2. Method

**Area is separable.** With the hardware fixed, crossbar arrays are a sum over ops (`Integrator._resident` at
ref_ctx). Area = ceil(xbar_scale·arrays)·area_per_array(tech, tile, adc, cell) + unit areas of the techs used.
So min time + λ·arrays is the M3 DP with a per-op penalty. Its frontier carries (cost, time, arrays).

**Hull.** Eisner–Severance recursion on λ, between λ=0 (fastest) and λ=1 s/array (fewest arrays). Each new λ is
the slope of the segment between two known vertices. This yields every vertex of the lower hull.

**Which knobs the DP sweeps.**
- Hardware settings: 4,095 time-relevant settings × tech pattern:
  - tech pattern = the subset of {xbar, vfu, star} a mapping may use. This is what makes effort exact: effort
    depends only on the techs used and their knobs.
  - The VFU-attention run is separate for each vfu_bw.
- Knobs swept after the DP instead:
  - xbar_tile and xbar_scale change only area, so they are applied to the hull afterwards
    (4 combinations: isaac/lean × 1.0/1.25).
  - vfu_mem is fixed at 64 GB; it costs neither area nor effort.

**Bound validity.** The bound (hull interpolated at the cap) is a valid lower bound because:
- DP time ≤ true time (see the scale-1.0 term below);
- the relaxed area scale·arrays·apa without the unit areas ≤ the true area;
- the hull lies below every mapping.

**Upper bound.** I evaluate exactly (FastEvaluator, plus the M5 `point_verifier`; illegal = not counted):
- hull vertices that fit the cap, fastest first, continuing through ties so the smallest-area design among equal
  times wins;
- then, by branch and bound in order of each setting's bound (up to 400 settings), layer-wise mixes of the two
  vertices that bracket the cap: the first k layers come from the larger one, with k found by bisection.

Mixes never won; every best design is a vertex.

**Runtime.** Hulls take 4,979 s wall on 18 workers (89,534 CPU-s); the cap phase takes 35 s.

## 3. Results (host-only sampling and embed, verifier on; `dp_caps_main.json`)

Time = LLM + tools (227.02 s). GPU 3,782.5 s. "Bound" is the certified lower bound on time at the cap.

| area cap | effort cap | best time | speedup | bound speedup | gap | area mm² | effort mo | design |
|---|---|---|---|---|---|---|---|---|
| none | none | 399.49 s | **9.468x** | 9.468x | 0 | 7,291 | 18 | transpose, lean, s1.25, adc8, cell2, star64, die |
| 8000 | none | 399.49 s | 9.468x | 9.468x | 0 | 7,291 | 18 | same |
| 6000 | none | 399.51 s | 9.468x | 9.468x | 4.3e-5 | 5,833 | 18 | same at s1.0 |
| 5000 | none | 420.29 s | 9.000x | 9.000x | 0 | 4,615 | 21 | transpose, lean, s1.25, adc8, **cell4** |
| 4000 | none | 420.49 s | 8.996x | 9.000x | 4.6e-4 | 3,692 | 21 | same at s1.0 |
| 3000 | none | 525.93 s | 7.192x | 7.195x | 3.7e-4 | 2,872 | 21 | transpose, lean, s1.0, adc4, cell4 |
| none | 7 | 401.68 s | 9.417x | 9.417x | 0 | 12,161 | 7 | the walk's design (transpose, isaac, cxl) |
| 8000 | 7 | 472.50 s | 8.005x | 8.005x | 0 | 5,616 | 6 | isaac tech, lean, s1.25, adc8, star64, cxl |
| 6000 | 7 | 472.50 s | 8.005x | 8.005x | 0 | 5,616 | 6 | same |
| 5000 | 7 | 472.52 s | 8.005x | 8.005x | 3.6e-5 | 4,493 | 6 | same at s1.0 |
| 4000 | 7 | 577.95 s | 6.545x | 6.545x | 0 | 3,995 | 6 | isaac, lean, s1.25, adc4 |
| 3000 | 7 | 788.86 s | 4.795x | 4.795x | 2.2e-5 | 2,548 | 6 | isaac, lean, s1.0, adc2 |
| none | 5 | 472.50 s | 8.005x | 8.005x | 0 | 10,485 | 4 | isaac tech, isaac tile, s1.25, star64, cxl |
| 8000 | 5 | 577.97 s | 6.544x | 6.545x | 3.0e-5 | 7,091 | 4 | isaac, isaac tile, s1.0, adc4 |
| 6000 | 5 | 802.96 s | 4.711x | 4.711x | 0 | 5,531 | 4 | isaac, lean, s1.25, **VFU256 + one attention layer on VFU** |
| 5000 | 5 | 802.98 s | 4.711x | 4.711x | 2.1e-5 | 4,425 | 4 | same at s1.0 |
| 4000 | 5 | 906.46 s | 4.173x | 4.173x | 0 | 3,934 | 4 | same, adc4 |
| 3000 | 5 | 1113.48 s | 3.397x | 3.397x | 1.5e-5 | 2,509 | 4 | same, s1.0, adc2 |

"Same at s1.0" means the same mapping and hardware with xbar_scale 1.0 instead of 1.25. Except for the four
effort-5 rows with VFU attention, every design maps all layers the same way:
- weight MVMs, attention and lm_head on the crossbar;
- vector ops and softmax on STAR;
- embed and sample on the host;
- KV 8-bit.

### Duality gap: not material, and fully explained

The largest gap is 0.046%. **Every nonzero gap is one named term:** the xbar_scale 1.0 grid-top overflow
reprogram, which the DP does not model (the M3 caveat).
- The cost is 0.0172 s on most settings and 0.193 s on the cell4 setting.
- Each such row's best design runs at s1.0. The same design at s1.25 takes exactly the bound time but is over
  the cap. I checked all 8 rows.

So the capped optimum lies within that one term of the bound. An exact area dimension would not change any row,
so I did not add one.

If an exact zero gap is wanted, the DP would need that overflow term. That changes what the DP models, not the
evaluator. I left it out and named it instead.

### Cap rows barely use the large space

Under caps, the optimum moves on hardware knobs: tile (lean), cell bits (4), ADCs per IMA, xbar_scale and tech.
It does not move on per-layer splits. My M3 expectation that "caps ≤ 5,000 force per-layer splits" was wrong:
lean tiles and 4-bit cells cut area per array instead.

The only untied optima are the four effort-5 rows, and the split there is the issue below.

### Flag: one attention layer on the VFU unlocks vfu_bw for every VFU op

The rule (`space.py:83`, kept by `LargeSpace.fixed_hardware`): vfu_bw and vfu_mem apply iff attention is on the
VFU, and `mem_bw_Bps` then applies to *all* VFU ops.

- In the tied space, attention on the VFU means all 36 layers there (6,115 s), so the rule never pays off.
- In the large space, one token layer (L12 or L35) on the VFU switches the 253 vector ops to 256 GB/s.
- The cost is 1 effort month (vfu_bw 256G), which still fits under 5.

| effort 5 | with token layer | the same design, all attention on xbar | best without VFU attention (`dp_caps_main_plain.json`) |
|---|---|---|---|
| 6000 mm² | 4.711x | 3.07x (1,231 s) | 3.072x |
| 5000 mm² | 4.711x | 3.07x | 3.072x |
| 4000 mm² | 4.173x | 2.83x (1,337 s) | 2.830x |
| 3000 mm² | 3.397x | 2.44x (1,548 s) | 2.444x |

The other 14 rows are identical in both files. I did not change the rule (model equations are out of scope); see Q1.

### Certificate

**DP floor over all hardware (host edge): 399.49 s = 9.468x.**
- It equals the best design found, so the uncapped gap is 0.
- This replaces the M3 per-slot floor (349.31 s, 10.83x), which was 14.2% below the optimum.
- With caps, `bound_s` per row is the certificate. It matches the best design within the named term above.

The M3 sweep optimum (398.80 s, 9.485x) is below this floor because it puts embed and sample on the VFU, which the
verifier rejects (R2.transfer / R1.sampling). Its time is therefore 0.17% better than the best legal design.
The edge-allowed run (§5) checks that the DP reproduces it.

## 4. Decisions (flag if you disagree)

1. **Sampling and embed are host-only by default** (`DEFAULT_EXCLUDE`, matching the verifier). `--allow-edge-vfu`
   lifts it. The pending sampling-rule decision is not baked in: both runs exist.
2. **Effort via tech patterns.** Each pattern's DP allows only those techs. `pattern_effort` is the effort of a
   design that uses all of them. The selected design's effort is re-evaluated exactly.
3. **vfu_mem is fixed at 64 GB** in the grid; it costs no area or effort, so more memory never hurts.
4. **xbar_scale ∈ {1.0, 1.25} and tile are applied after the DP** (area only). At s1.0 the true time includes the
   reprogram term above, so the DP is optimistic there. That keeps the bound valid; the gap shows the term.
5. **Ties go to smaller area, then smaller effort.** E.g. the uncapped optimum is reported at 7,291 mm², not
   12,161.

## 5. Edge-on-VFU cross-check (`--allow-edge-vfu`, tag `edge`)

The hulls also allow embed and sample on the VFU; the verifier stays on for the designs picked. The hulls took
the same ~83 min.

- **The DP floor reproduces the M3 sweep optimum exactly: 398.7996 s = 9.4847x** (M3: 398.79959 s). So the DP
  and the sweep agree on the space they share, and the 0.17% between 9.485x and 9.468x is the edge-on-VFU term.
- The bounds in this run come from designs the verifier rejects. Their extra 0.2–0.6% over the host-edge rows is
  that same edge term, not a duality gap.
- Every legal best design is edge-on-host and identical to `dp_caps_main.json`, except effort 5 at ≤ 6000 mm².
  There the legal fallback is a poor mix (0.63–0.69x). The vertices of the effort-5 VFU hulls put the edge on the
  VFU, so every candidate the search builds from them is illegal.

**So use `dp_caps_main.json` as the capped ground truth.** `dp_caps_edge.json` only checks the floor and prices
the edge term. If the sampling rule ends up allowing edge on the VFU, the bounds in `dp_caps_edge.json` become the
targets. The best designs would then need the cap phase rerun with the verifier off.

## 6. Tests

- `tests/test_dp_caps.py`: 9 passed, run on 2 layers with 10 free slots against brute force. They check:
  - the λ-DP objective equals the brute-force minimum for 5 values of λ;
  - the DP's arrays equal `xbar_need`;
  - every hull vertex lies on the brute-force front;
  - the bound ≤ the capped brute-force optimum at every distinct cap;
  - the area and effort formulas match the evaluator;
  - the edge exclusion;
  - mix endpoints.
- `tests/test_m0_repro.py`: passes.
- Full suite: 91 passed, 3 failed (460 s). The 3 failures are `test_claims.py` (`main.tex` vs
  `paper/claims.yaml`). They also fail on autopilot (known since M3) and none comes from this branch.

## 7. Questions

1. **The vfu_bw rule (above).** Should vfu_bw/vfu_mem apply whenever the VFU runs any op, or only to attention's
   KV traffic? Either fix changes the evaluator, so it is your call. Until then the effort-5 rows have two values
   (token layer / plain), and E5/H5 should say which one is used. I'd use the plain one, or fix the rule first.
2. With host-only edge, the uncapped legal optimum is 9.468x, not 9.48x. Should the "9.48x" figure in the paper
   become 9.47x once the sampling rule is decided?
3. Do you want the s1.0 reprogram term in the DP (exact zero gap), or is the named 0.05% gap enough?
