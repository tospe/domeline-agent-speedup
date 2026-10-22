#!/bin/bash
# G2: what the RTX 3080 does past its VRAM wall. With -nkvo 1 the KV cache stays
# in host RAM and attention runs on the CPU, so long contexts fit but each token
# pays host attention. Measured at fp16 KV on both weight formats, including
# depths below the wall, so the step against G1 is visible.
# Run with the CPU otherwise idle (host attention is CPU-bound).
# Output: results/gpu/g2.jsonl (records tagged "_mode": "nkvo") and g2_status.tsv.
set -u
B=/localhome/tsousape/tools/llama.cpp/build/bin
G=/localhome/tsousape/models/gguf
OUT=$(dirname "$0")/../results/gpu
mkdir -p "$OUT"
: > "$OUT/g2.jsonl"; echo -e "weights\tkv\tdepth\tmode\tstatus" > "$OUT/g2_status.tsv"
DEPTHS="0 4096 16384 32768 49152 65536 98304"
for W in Q8_0 Q4_K_M; do
  for D in $DEPTHS; do
    if R=$($B/llama-bench -m $G/mini-coder-4b-$W.gguf -ngl 99 -fa 1 -nkvo 1 -ctk f16 -ctv f16 \
             -p 0 -n 32 -d $D -r 2 -o jsonl 2>>"$OUT/g2_stderr.log"); then
      echo "$R" | sed 's/^{/{"_mode": "nkvo", /' >> "$OUT/g2.jsonl"
      echo -e "$W\tf16\t$D\tnkvo\tok" >> "$OUT/g2_status.tsv"
    else
      echo -e "$W\tf16\t$D\tnkvo\tfail" >> "$OUT/g2_status.tsv"
      break
    fi
  done
done
