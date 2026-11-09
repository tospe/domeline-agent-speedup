#!/bin/bash
# G1: RTX 3080 decode throughput vs context depth, per KV precision (flash attention on).
# One llama-bench run per (weights, kv type, depth) so an OOM at one depth does not kill the rest.
# Output: results/gpu/g1.jsonl (one JSON record per successful run) and g1_status.tsv.
set -u
B=/localhome/tsousape/tools/llama.cpp/build/bin
G=/localhome/tsousape/models/gguf
OUT=$(dirname "$0")/../results/gpu
mkdir -p "$OUT"
: > "$OUT/g1.jsonl"; echo -e "weights\tkv\tdepth\tstatus" > "$OUT/g1_status.tsv"
DEPTHS="0 1024 2048 4096 8192 16384 24576 32768 36864 40960 49152 65536 81920 98304 131072 163840 196608"
for W in Q8_0 Q4_K_M; do
  for KV in f16 q8_0 q4_0; do
    for D in $DEPTHS; do
      if $B/llama-bench -m $G/mini-coder-4b-$W.gguf -ngl 99 -fa 1 -ctk $KV -ctv $KV \
           -p 0 -n 64 -d $D -r 3 -o jsonl 2>>"$OUT/g1_stderr.log" >> "$OUT/g1.jsonl"; then
        echo -e "$W\t$KV\t$D\tok" >> "$OUT/g1_status.tsv"
      else
        echo -e "$W\t$KV\t$D\tfail" >> "$OUT/g1_status.tsv"
        break   # deeper contexts will not fit either
      fi
    done
  done
done
