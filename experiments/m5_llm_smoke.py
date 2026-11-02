"""M5 smoke runs of the LLM agents and LLM baselines, one per backend, small budget.

On the 45 search sessions, LargeSpace + FastEvaluator, verifier on. For each backend:
  route    the three role agents behind llm_agents.route_search (a stand-in until team.team_search)
  single   single_agent_search without readings (E5 baseline)
  eat      eat_team_search, supervisor team without readings or routing (E5 baseline)
Writes results/m5_llm/<backend>_<method>.json (RunRecord) and results/m5_llm/smoke.csv.

Usage: python experiments/m5_llm_smoke.py [--budget 10] [--backends sonnet haiku local] [--methods ...]
The local backend needs llama-server on :8080 (traces/serve.sh).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from common import gpu, library, model, sessions, split  # noqa: E402
from harness import llm_agents as A  # noqa: E402
from harness.large_space import FastEvaluator, LargeSpace  # noqa: E402
from harness.tools import Toolbox  # noqa: E402
from harness.verify import verify_point  # noqa: E402

BACKENDS = {"sonnet": lambda: A.ClaudeCLI(A.SONNET), "haiku": lambda: A.ClaudeCLI(A.HAIKU),
            "local": lambda: A.LocalLlama()}
OUT = ROOT / "results/m5_llm"


def setup(budget):
    train, test = split(sessions())
    space = LargeSpace(library(), model(), ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    ev = FastEvaluator(space, train, gpu(), budget=budget)
    return ev, Toolbox(ev, verifier=lambda p: verify_point(space, p, train))


def run_one(name, method, budget, seed):
    ev, tb = setup(budget)
    be = BACKENDS[name]()
    t0 = time.time()
    if method == "route":
        rec = A.route_search(ev, {r: A.LLMAgent(r, be) for r in A.ROLES}, tb=tb, seed=seed)
    elif method == "single":
        rec = A.single_agent_search(ev, be, tb=tb, seed=seed)
    else:
        rec = A.eat_team_search(ev, be, tb=tb, seed=seed, stall_k=4)
    rec.config.update(backend=name, model_id=be.model)
    path = OUT / f"{name}_{method}.json"
    rec.write(path, ev, tb)
    d = json.loads(path.read_text())
    outs = [e["outcome"] for e in d["log"]]
    best = d["final"]
    return {"backend": name, "model_id": be.model, "method": method, "seed": seed, "budget": budget,
            "evaluations": d["cost"]["evaluations"], "proposals": len(outs),
            "accepted": outs.count("accepted"), "illegal": outs.count("illegal"),
            "repeated": outs.count("repeated"), "best_speedup": best and best["speedup"],
            "start_speedup": d["evaluations"][0]["speedup"], "stop": d["stop"],
            "llm_calls": d["config"].get("llm_calls"),
            "tokens": json.dumps(d["cost"]["tokens"]), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--methods", nargs="+", default=["route", "single", "eat"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in a.backends:
        for m in a.methods:
            try:
                r = run_one(b, m, a.budget, a.seed)
            except Exception as ex:      # one backend failing must not lose the others' records
                r = {"backend": b, "method": m, "stop": f"ERROR {type(ex).__name__}: {ex}"[:300]}
            rows.append(r)
            print(json.dumps(r), flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "smoke.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
