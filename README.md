# Domeline agent speedup

A harness that designs a processing-in-memory (PIM) coprocessor for an agentic workload, guided by Domeline. The coprocessor sits next to a CPU, VIPER's task-offloading mode. The case study is a coding agent (mini-swe-agent + mini-coder-4b on SWE-bench Verified), and the bar is an RTX 3080 running the same sessions.

Design notes: `docs/harness.md`. Run log, decisions and status: `docs/autopilot.md`.

## Layout

| Path | What |
|---|---|
| `harness/model.py` | Model spec from a HF `config.json` (mini-coder-4b / Qwen3, BitNet) |
| `harness/ops.py` | Op graph for one forward step (decode or prefill) with the activation tensors each op reads and writes |
| `harness/tech.py` | Technology cost models (crossbar, tiled GEMV, digital unit, DRAM-PIM, CPU with measured profile) and the YAML loader |
| `harness/design.py` | A design: op -> technology mapping, units, link, precisions |
| `harness/integrator.py` | One step's time, split into Domeline terms (host, per-tech compute, transfer, launch, write), plus capacity |
| `harness/session.py` | Session replay over a trace; GPU baseline from measurements |
| `harness/space.py` | The design space (17 knobs: 14 mapping/sizing knobs plus 3 sourced area knobs), area and effort |
| `harness/evaluate.py` | Budgeted evaluator; Domeline readings for each evaluated design |
| `harness/search.py` | Domeline walk, hill climbing, random search, TPE, sweeps |
| `harness/agent.py` | LLM agent searcher (`claude -p` or a local llama-server), with or without the readings |
| `harness/figstyle.py` | One figure style (IEEE column/page widths, validated palette) |
| `lib/tech/*.yaml` | Technology library; every entry cites its source, and uncertain numbers are marked VERIFY |
| `gpu/g1_sweep.sh`, `gpu/g2_sweep.sh` | RTX 3080 decode time vs context and KV precision; past the VRAM wall with KV in host memory |
| `profile/cpu_ops.c` | Host CPU per-op profile at mini-coder-4b shapes |
| `traces/run_swebench.py` | Agent runner that logs per-turn traces (context, prefill/decode times, tool times) |
| `experiments/` | Search comparison, effort staircase, sensitivity, exhaustive sweep and its analysis, iso-area study, second held-out set |
| `figs/` | One script per figure; figures land in `figures/` |
| `tests/` | Reproduction of TERINT-GEMV's estimator; fast replay vs exact replay |

## Reproduce

Environment: Python 3.12 venv (`numpy scipy matplotlib pandas pyyaml optuna pytest`), llama.cpp built with CUDA, mini-coder-4b converted to GGUF Q8_0, mini-swe-agent 1.17.5 in a separate venv, rootless podman.

```bash
pytest -q tests                                   # integrator == TERINT estimator; fast replay within 0.3%
gpu/g1_sweep.sh                                   # GPU decode vs context      -> results/gpu/g1.jsonl
profile/run_cpu_profile.sh                        # host per-op profile        -> results/cpu/cpu_ops.tsv
traces/serve.sh &                                 # llama-server on the 3080
python traces/run_swebench.py --instances traces/swebench_sample100.tsv --out results/traces/swebench
cd experiments
python search_compare.py --budget 100 --seeds 0 1 2 --tag main
python effort_sweep.py --tag main
python sensitivity.py --tag main
python sweep_all.py --workers 20 --tag main       # all 575,400 designs, ~11 h on 20 cores
python sweep_analyze.py --tag main --runs main effort5 effort5_haiku
python area_sweep.py --budget 150 --workers 20 --tag main
python search_compare.py --budget 100 --seeds 0 1 2 --agent-seeds 2 --area-cap 3000 --tag area3000
python heldout_extra.py --runs main effort5
cd .. && for f in figs/fig_*.py; do python $f main; done; python figs/fig_search.py effort5; python figs/fig_search.py area3000
```
