#!/bin/bash
# Run after the main trace collection finishes:
#  1. re-run sessions that hit the 3080's fp16 KV wall with int8 KV at 64K context;
#  2. start the exhaustive sweep on the frozen workload.
set -u
cd "$(dirname "$0")/.."
PY=/localhome/tsousape/tools/venv/bin/python
APY=/localhome/tsousape/tools/agentvenv/bin/python
pkill -f "llama-server .*mini-coder-4b" ; sleep 3
KV=q8_0 CTX=65536 nohup ./traces/serve.sh > results/traces/llama_server_kv8.log 2>&1 &
for i in $(seq 90); do curl -s 127.0.0.1:8080/health | grep -q ok && break; sleep 2; done
$PY - <<'PY'
import json, pathlib
root = pathlib.Path("results/traces/swebench")
ids = [d.name for d in sorted(root.iterdir()) if (d / "result.json").exists()
       and json.loads((d / "result.json").read_text())["exit_status"] == "ContextExceeded"]
rows = {l.split("\t")[0]: l for l in open("traces/swebench_sample100.tsv").read().splitlines()[1:]}
with open("traces/swebench_wall.tsv", "w") as f:
    f.write("instance_id\trepo\timage\n")
    for i in ids:
        f.write(rows[i] + "\n")
print(len(ids), "sessions hit the fp16 wall:", ids)
PY
$APY traces/run_swebench.py --instances traces/swebench_wall.tsv --out results/traces/swebench_kv8 \
    >> results/traces/swebench_kv8_run.log 2>&1
pkill -f "llama-server .*mini-coder-4b"
nohup ./traces/serve.sh > results/traces/llama_server.log 2>&1 &
cd experiments && nice -n 5 $PY sweep_all.py --workers 20 --tag main > ../results/search/sweep_main.log 2>&1
