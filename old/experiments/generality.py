"""E8, generality: rerun the harness with a second technology library and a second host profile.

Same workload (search set / held-out split as search_compare.py), same knobs,
same searchers; only the harness inputs change:

  main       lib/tech, measured host profile (reference, current code)
  dram       lib/tech + lib/tech_dram with every crossbar removed: the MVM role
             (knob xbar_tech) is filled by DRAM-PIM, xbar_isaac -> upmem_dpu
             (measured UPMEM, with its measured transfer law), xbar_transpose ->
             aim_gddr6 (SK hynix GDDR6-AiM, ISSCC 2022, numbers VERIFY)
  host_roof  lib/tech, host = cpu_roofline (datasheet roofline instead of the
             measured per-op profile)

Per config: the walk (deterministic, one run) and, for reference, TPE and random
search at the same budget over 3 seeds.

  python experiments/generality.py --budget 100 --workers 4 --tag main
Output: results/search/generality_<tag>.json
"""

import argparse
import json
import time
from multiprocessing import Pool

from common import ROOT, gpu, model, sessions, split

from harness.evaluate import Evaluator
from harness.search import domeline_walk, random_search, tpe_search
from harness.space import Space
from harness.tech import load_library

CONFIGS = {
    "main": dict(libs=["lib/tech"], host="cpu", drop_kinds=(), mvm_techs=None),
    "dram": dict(libs=["lib/tech", "lib/tech_dram"], host="cpu", drop_kinds=("crossbar",),
                 mvm_techs={"xbar_isaac": "upmem_dpu", "xbar_transpose": "aim_gddr6"}),
    "host_roof": dict(libs=["lib/tech"], host="cpu_roofline", drop_kinds=(), mvm_techs=None),
}

_STATE = {}


def setup(name):
    if name not in _STATE:
        c = CONFIGS[name]
        lib = load_library([ROOT / p for p in c["libs"]])
        lib = {k: t for k, t in lib.items() if t.kind not in c["drop_kinds"]}
        spec, g = model(), gpu()
        train, test = split(sessions())
        ref_ctx = max(s.max_ctx for s in train + test)
        space = Space(lib, spec, ref_ctx=ref_ctx, host=c["host"], mvm_techs=c["mvm_techs"])
        _STATE[name] = (space, train, test, g)
    return _STATE[name]


def _f(v):
    return float(v) if isinstance(v, float) else v


def run(job):
    name, method, seed, budget = job
    space, train, test, g = setup(name)
    ev = Evaluator(space, train, g, budget=budget)
    t0 = time.time()
    extra = {}
    if method == "walk":
        tr = domeline_walk(ev)
        extra = {"stop": tr.stop,
                 "trail": [{"n": s.eval.n, "move": s.move, "reading": s.reading, "speedup": s.eval.speedup_vs_gpu,
                            "area_mm2": s.eval.area_mm2, "effort_months": s.eval.effort_months,
                            "terms": s.eval.terms, "feasible": s.eval.feasible} for s in tr.steps]}
    elif method == "tpe":
        tpe_search(ev, seed=seed)
    elif method == "random":
        random_search(ev, seed=seed)
    best = ev.best()
    rec = {"config": name, "method": method, "seed": seed, "evals": ev.used, "wall_s": time.time() - t0,
           "gpu_s": ev.gpu_time, "tools_s": ev.tools_time,
           "log": [{"n": e.n, "speedup": e.speedup_vs_gpu, "feasible": e.feasible, "area_mm2": e.area_mm2,
                    "reason": e.reason} for e in ev.log], **extra}
    if best:
        h = Evaluator(space, test, g)(best.point)
        mt = space.mvm_tech(best.point) if "xbar" in (best.point["mvm"], best.point["lm_head"],
                                                       best.point["attention"]) else None
        rec["best"] = {"point": {k: _f(v) for k, v in best.point.items()}, "mvm_tech": mt,
                       "speedup": best.speedup_vs_gpu, "n": best.n, "area_mm2": best.area_mm2,
                       "effort_months": best.effort_months, "terms": best.terms,
                       "heldout_speedup": h.speedup_vs_gpu, "heldout_feasible": h.feasible}
    b = rec.get("best", {})
    print(f"{name:9s} {method:6s} seed {seed}: {ev.used:3d} evals, best {b.get('speedup', float('nan')):.2f}x "
          f"({b.get('mvm_tech')}) at eval {b.get('n')}, held-out {b.get('heldout_speedup', float('nan')):.2f}x "
          f"({rec['wall_s']:.0f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--methods", nargs="+", default=["walk", "tpe", "random"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    jobs = [(c, m, s, args.budget) for c in args.configs for m in args.methods
            for s in ([0] if m == "walk" else args.seeds)]
    dst = ROOT / "results/search" / f"generality_{args.tag}.json"
    out = {"budget": args.budget, "configs": {c: {k: (list(v) if isinstance(v, tuple) else v)
                                                  for k, v in CONFIGS[c].items()} for c in args.configs},
           "runs": []}
    with Pool(args.workers) as pool:
        for rec in pool.imap_unordered(run, jobs):
            out["runs"].append(rec)
            dst.write_text(json.dumps(out, indent=1))
    out["runs"].sort(key=lambda r: (r["config"], r["method"], r["seed"]))
    dst.write_text(json.dumps(out, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
