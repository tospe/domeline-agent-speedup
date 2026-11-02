"""M5 smoke runs of the LLM agents and LLM baselines, one per backend, small budget.

On the 45 search sessions, LargeSpace + FastEvaluator, verifier on. For each backend:
  team     the three LLM role agents in C's loop, team.team_search (readings + routing + verifier)
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
from harness import team as T  # noqa: E402
from harness.large_space import FastEvaluator, LargeSpace  # noqa: E402
from harness.tools import RunRecord, Toolbox  # noqa: E402

BACKENDS = {"sonnet": lambda: A.ClaudeCLI(A.SONNET), "haiku": lambda: A.ClaudeCLI(A.HAIKU),
            "local": lambda: A.LocalLlama()}
OUT = ROOT / "results/m5_llm"


def setup(budget):
    train, test = split(sessions())
    space = LargeSpace(library(), model(), ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    ev = FastEvaluator(space, train, gpu(), budget=budget)
    return ev, Toolbox(ev, verifier=T.point_verifier(space, train))


def run_one(name, method, budget, seed):
    ev, tb = setup(budget)
    be = BACKENDS[name]()
    t0 = time.time()
    if method == "team":
        rec = RunRecord("M5-smoke", "llm_team", seed, {"model": be.model, "budget": budget,
                                                        "space": "LargeSpace"})
        agents = A.llm_team(be, record=rec, seed=seed)
        res = T.team_search(ev, tb, agents, record=rec, max_rounds=3)
        rec.config.update(llm_calls=sum(len(a.calls) for a in agents.values()), restructures=res.restructures,
                          rejected=res.rejected, agent_calls=[c for a in agents.values() for c in a.calls])
    elif method == "single":
        rec = A.single_agent_search(ev, be, tb=tb, seed=seed)
    else:
        rec = A.eat_team_search(ev, be, tb=tb, seed=seed, stall_k=4)
    rec.config.update(backend=name, model_id=be.model)
    path = OUT / f"{name}_{method}.json"
    rec.write(path, ev, tb)
    d = json.loads(path.read_text())
    outs = [e["outcome"] for e in d["log"] if e.get("kind", "proposal") == "proposal"]
    best = d["final"]
    return {"backend": name, "model_id": be.model, "method": method, "seed": seed, "budget": budget,
            "evaluations": d["cost"]["evaluations"], "proposals": len(outs),
            "accepted": outs.count("accepted"), "illegal": outs.count("illegal"),
            "duplicate": outs.count("duplicate"), "best_speedup": best and best["speedup"],
            "start_speedup": d["evaluations"][0]["speedup"], "stop": d["stop"],
            "llm_calls": d["config"].get("llm_calls"),
            "tokens": json.dumps(d["cost"]["tokens"]), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("--methods", nargs="+", default=["team", "single", "eat"])
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
