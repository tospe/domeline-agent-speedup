#!/bin/bash
# Serve mini-coder-4b on the RTX 3080 for trace collection.
# Q8_0 weights, f16 KV, flash attention, one slot, 32K context (the f16 wall is 32K-36K, G1).
# Sessions that outgrow it are re-run with KV=q8_0 CTX=65536, what a 3080 user would have to do.
B=/localhome/tsousape/tools/llama.cpp/build/bin
exec $B/llama-server -m /localhome/tsousape/models/gguf/mini-coder-4b-Q8_0.gguf \
  -ngl 99 -fa on -ctk ${KV:-f16} -ctv ${KV:-f16} -c ${CTX:-32768} -np 1 --jinja \
  --host 127.0.0.1 --port ${PORT:-8080} --no-webui "$@"
