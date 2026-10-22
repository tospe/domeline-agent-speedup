#!/bin/bash
# Serve mini-coder-4b on the RTX 3080 for trace collection.
# Q8_0 weights, f16 KV, flash attention, one slot, 32K context (the f16 wall is 32K-36K, G1).
B=/localhome/tsousape/tools/llama.cpp/build/bin
exec $B/llama-server -m /localhome/tsousape/models/gguf/mini-coder-4b-Q8_0.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -c ${CTX:-32768} -np 1 --jinja \
  --host 127.0.0.1 --port ${PORT:-8080} --no-webui "$@"
