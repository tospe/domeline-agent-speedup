"""Effort staircase: best speedup reachable within an engineering-effort budget.

For each effort cap (months), the Domeline walk and TPE search the space
under that cap; the best feasible design gives one step of the staircase.
"With 3 more months you get X% more" is read off between steps.

  python experiments/effort_sweep.py --budget 150 --tag main
Output: results/search/effort_<tag>.json
"""

import argparse
import json
import math

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import Evaluator
from harness.search import domeline_walk, tpe_search
from harness.space import Space

CAPS = [0, 1, 3, 4, 5, 7, 8, 12, 16, math.inf]


def run(space, train, g, cap, method, budget):
    ev = Evaluator(space, train, g, budget=budget, effort_cap_months=cap)
    if method == "walk":
        domeline_walk(ev)
    else:
        tpe_search(ev, seed=0)
    b = ev.best()
    if b is None:
        return {"cap": cap, "method": method, "evals": ev.used, "speedup": None}
    return {"cap": cap, "method": method, "evals": ev.used, "speedup": b.speedup_vs_gpu,
            "effort_months": b.effort_months, "area_mm2": b.area_mm2, "n": b.n,
            "point": {k: (float(v) if isinstance(v, float) else v) for k, v in b.point.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    out = {"budget": args.budget, "rows": []}
    for cap in CAPS:
        for method in ("walk", "tpe"):
            r = run(space, train, g, cap, method, args.budget)
            out["rows"].append(r)
            print(f"cap {cap:>4} {method:5s}: {r['speedup'] or 0:.2f}x "
                  f"(effort {r.get('effort_months')}, area {r.get('area_mm2', 0):.0f}, {r['evals']} evals)",
                  flush=True)
    dst = ROOT / "results/search" / f"effort_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1, default=lambda x: None if x == math.inf else x))
    print(dst)


if __name__ == "__main__":
    main()
