"""Search comparison: best workload time found vs evaluations used, per method.

Every method searches the same design space on the same search set of
sessions with the same budget; each distinct design costs one evaluation.
Each method's best design is then re-timed on the held-out sessions.

  python experiments/search_compare.py --budget 100 --seeds 0 1 2 --tag main
Output: results/search/<tag>.json
"""

import argparse
import json
import math
import time
from pathlib import Path

from common import ROOT, gpu, library, model, sessions, split

from harness.agent import agent_search
from harness.evaluate import Evaluator
from harness.search import domeline_walk, hill_climb, random_search, tpe_search
from harness.space import Space


def log_of(ev):
    return [{"n": e.n, "time_s": e.time_s, "feasible": e.feasible, "speedup": e.speedup_vs_gpu,
             "area_mm2": e.area_mm2, "effort_months": e.effort_months, "reason": e.reason}
            for e in ev.log]


def heldout(space, test, g, point, caps):
    ev = Evaluator(space, test, g, **caps)
    e = ev(point)
    return {"time_s": e.time_s, "speedup": e.speedup_vs_gpu, "feasible": e.feasible,
            "gpu_s": ev.gpu_time, "tools_s": ev.tools_time}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+",
                    default=["walk", "hill_climb", "random", "tpe", "agent", "agent_scalar"])
    ap.add_argument("--agent-model", default="sonnet")
    ap.add_argument("--agent-seeds", type=int, default=1)
    ap.add_argument("--area-cap", type=float, default=math.inf)
    ap.add_argument("--effort-cap", type=float, default=math.inf)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    ref_ctx = max(s.max_ctx for s in train + test)
    space = Space(lib, spec, ref_ctx=ref_ctx, host="cpu")
    caps = {"area_cap_mm2": args.area_cap, "effort_cap_months": args.effort_cap}
    out = {"budget": args.budget, "n_train": len(train), "n_test": len(test), "ref_ctx": ref_ctx,
           "caps": caps, "runs": [], "agent_cost_usd": 0.0}
    print(f"search set {len(train)} sessions, held-out {len(test)}; ref_ctx {ref_ctx}")

    dst = ROOT / "results/search" / f"{args.tag}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        seeds = [0] if method == "walk" else (
            list(range(args.agent_seeds)) if method.startswith("agent") else args.seeds)
        for seed in seeds:
            ev = Evaluator(space, train, g, budget=args.budget, **caps)
            t0 = time.time()
            extra = {}
            if method == "walk":
                tr = domeline_walk(ev)
                extra = {"stop": tr.stop, "trail": [{"n": s.eval.n, "move": s.move, "reading": s.reading,
                                                     "time_s": s.eval.time_s, "speedup": s.eval.speedup_vs_gpu,
                                                     "area_mm2": s.eval.area_mm2,
                                                     "effort_months": s.eval.effort_months,
                                                     "terms": s.eval.terms, "by_kind": s.eval.by_kind,
                                                     "feasible": s.eval.feasible}
                                                    for s in tr.steps]}
            elif method == "hill_climb":
                hill_climb(ev, seed=seed)
            elif method == "random":
                random_search(ev, seed=seed)
            elif method == "tpe":
                tpe_search(ev, seed=seed)
            elif method in ("agent", "agent_scalar"):
                run = agent_search(ev, with_readings=(method == "agent"), model=args.agent_model, seed=seed)
                out["agent_cost_usd"] += run.cost_usd
                extra = {"stop": run.stop, "cost_usd": run.cost_usd, "steps": run.steps}
            best = ev.best()
            rec = {"method": method, "seed": seed, "evals": ev.used, "wall_s": time.time() - t0,
                   "gpu_s": ev.gpu_time, "tools_s": ev.tools_time, "log": log_of(ev), **extra}
            if best:
                rec["best"] = {"point": {k: (v if not isinstance(v, float) else float(v))
                                         for k, v in best.point.items()},
                               "time_s": best.time_s, "speedup": best.speedup_vs_gpu, "n": best.n,
                               "area_mm2": best.area_mm2, "effort_months": best.effort_months,
                               "heldout": heldout(space, test, g, best.point, caps)}
            out["runs"].append(rec)
            dst.write_text(json.dumps(out, indent=1))      # incremental: a crash keeps finished runs
            b = rec.get("best", {})
            print(f"{method:12s} seed {seed}: {ev.used:3d} evals, best {b.get('speedup', float('nan')):.2f}x "
                  f"at eval {b.get('n')}, held-out {b.get('heldout', {}).get('speedup', float('nan')):.2f}x "
                  f"({rec['wall_s']:.0f}s)", flush=True)

    dst.write_text(json.dumps(out, indent=1))
    print(dst, f"agent cost ${out['agent_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
