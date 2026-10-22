#!/bin/bash
# H3: run the per-op CPU profile on this host. Output: results/cpu/cpu_ops.tsv
set -eu
cd "$(dirname "$0")"
OUT=../results/cpu; mkdir -p $OUT
C=/localhome/tsousape/tools/cuda
$C/bin/x86_64-conda-linux-gnu-gcc -O3 -march=native -fopenmp cpu_ops.c -o cpu_ops -lm -Wl,-rpath,$C/lib
CTX="128 512 1024 2048 4096 8192 16384 32768 65536 131072"
{ echo -e "kind\tctx\tthreads\tseconds"
  for T in 1 8 16; do OMP_PROC_BIND=close OMP_PLACES=cores ./cpu_ops $T $CTX; done; } > $OUT/cpu_ops.tsv
lscpu > $OUT/lscpu.txt
