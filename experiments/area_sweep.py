"""Iso-area study: best speedup reachable under a coprocessor area cap.

The space now has three sourced area knobs (ISAAC Table I breakdown): a lean
tile without the eDRAM buffer and bus, fewer ADCs per IMA, and 4-bit cells.
For each area cap the Domeline walk and TPE search the full space; an
exhaustive sweep of the crossbar subspace (every area-relevant knob, the rest
fixed at the headline design) gives the exact frontier to compare against.

  python experiments/area_sweep.py --budget 150 --workers 20 --tag main
Output: results/search/area_<tag>.json
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import Evaluator
from harness.search import domeline_walk, tpe_search
from harness.space import KNOBS, Space, canonical, key

CAPS = [math.inf, 8000, 6000, 5000, 4000, 3000, 2500, 2000, 1500, 1000]
SUB = ["xbar_tech", "xbar_scale", "kv_bits", "link", "xbar_tile", "adc_per_ima", "cell_bits", "star_units"]


def setup():
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    return space, train, test, g


def row(e, ev_test=None):
    if e is None:
        return None
    r = {"speedup": e.speedup_vs_gpu, "area_mm2": e.area_mm2, "effort_months": e.effort_months, "n": e.n,
         "point": {k: (float(v) if isinstance(v, float) else v) for k, v in canonical(e.point).items()}}
    if ev_test is not None:
        r["heldout_speedup"] = ev_test(e.point).speedup_vs_gpu
    return r


def sub_worker(points):
    space, train, _, g = setup()
    ev = Evaluator(space, train, g)
    out = []
    for p in points:
        e = ev(p)
        out.append((p, e.speedup_vs_gpu, e.feasible, e.area_mm2, e.effort_months))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    space, train, test, g = setup()
    head = json.loads((ROOT / "results/search/main.json").read_text())["runs"][0]["best"]["point"]
    head = canonical(head)

    # Exact frontier over the crossbar subspace.
    seen = {}
    for vals in itertools.product(*(KNOBS[k] for k in SUB)):
        p = canonical({**head, **dict(zip(SUB, vals))})
        seen.setdefault(key(p), p)
    pts = list(seen.values())
    chunks = [pts[i:: args.workers] for i in range(args.workers)]
    with mp.Pool(args.workers) as pool:
        sub = [r for part in pool.map(sub_worker, chunks) for r in part]
    print(f"subspace: {len(sub)} designs", flush=True)

    out = {"budget": args.budget, "n_sub": len(sub), "sub": [
        {"point": p, "speedup": s, "feasible": f, "area_mm2": a, "effort_months": m} for p, s, f, a, m in sub],
        "rows": []}
    ev_test = Evaluator(space, test, g)
    for cap in CAPS:
        feas = [r for r in sub if r[2] and r[3] <= cap]
        ref = max(feas, key=lambda r: r[1]) if feas else None
        for method in ("walk", "tpe"):
            ev = Evaluator(space, train, g, budget=args.budget, area_cap_mm2=cap)
            if method == "walk":
                trail = domeline_walk(ev)
                moves = [s.move for s in trail.steps]
            else:
                tpe_search(ev, seed=0)
                moves = None
            b = ev.best()
            r = {"cap": None if cap == math.inf else cap, "method": method, "evals": ev.used,
                 "best": row(b, ev_test), "sub_best": ref[1] if ref else None,
                 "sub_best_area": ref[3] if ref else None, "moves": moves}
            out["rows"].append(r)
            print(f"cap {cap:>6} {method:5s}: {b.speedup_vs_gpu if b else 0:5.2f}x at {b.area_mm2 if b else 0:6.0f} mm2 "
                  f"(subspace best {ref[1] if ref else 0:.2f}x), {ev.used} evals", flush=True)
    dst = ROOT / "results/search" / f"area_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1, default=lambda x: None if x == math.inf else x))
    print(dst)


if __name__ == "__main__":
    main()
