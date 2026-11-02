"""M0: reproduce the paper's small-space numbers from the current code.

  python experiments/m0_repro.py [--sample 300] [--seed 0]
Output: results/search/m0_repro.json

Checks, each against the paper (docs/results.md):
  - distinct designs in the 14-knob space (area knobs at ISAAC defaults): 575,400
  - feasible designs and the optimum, from the recorded sweep (results/search/sweep_main.tsv),
    with a seeded sample of its rows (plus the optimum and the walk's design) re-timed by the
    current code, so the sweep still describes this evaluator
  - the walk: its design's gap to the optimum (0.7%) and the eval it stops at (20)
  - the time one evaluation takes
"""

import argparse
import csv
import gzip
import itertools
import json
import random
import time

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import Evaluator
from harness.search import domeline_walk
from harness.space import KNOBS, START, XBAR_AREA_KNOBS, Space

SMALL_KNOBS = [k for k in KNOBS if k not in XBAR_AREA_KNOBS]   # the paper's 14 knobs
PAPER = {"n_designs": 575_400, "n_feasible": 280_296, "optimum": 9.48, "walk_gap": 0.007, "walk_eval": 20}
SWEEP = ROOT / "results/search/sweep_main.tsv"


def sweep_rows():
    """The recorded sweep (the .tsv, or the committed .tsv.gz when the .tsv is absent)."""
    f = open(SWEEP) if SWEEP.exists() else gzip.open(str(SWEEP) + ".gz", "rt")
    with f:
        return list(csv.DictReader(f, delimiter="\t"))


def parse(row):
    p = {}
    for k in SMALL_KNOBS:
        v = row[k]
        p[k] = v if not v.replace(".", "", 1).replace("e", "", 1).replace("+", "").isdigit() else (
            int(v) if v.isdigit() else float(v))
    return p


def setup():
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    return space, train, g


def count_designs(space):
    fixed = {k: START[k] for k in XBAR_AREA_KNOBS}
    keys = set()
    for vals in itertools.product(*(KNOBS[k] for k in SMALL_KNOBS)):
        keys.add(space.key({**dict(zip(SMALL_KNOBS, vals)), **fixed}))
    return len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-count", action="store_true")
    args = ap.parse_args()
    space, train, g = setup()
    out = {"paper": PAPER, "seed": args.seed}

    t0 = time.time()
    out["n_designs"] = None if args.skip_count else count_designs(space)
    out["count_s"] = time.time() - t0

    rows = sweep_rows()
    feas = [r for r in rows if r["feasible"] == "1"]
    best = max(feas, key=lambda r: float(r["speedup"]))
    out.update(sweep_rows=len(rows), n_feasible=len(feas), optimum=float(best["speedup"]),
               optimum_point=parse(best))

    ev = Evaluator(space, train, g)
    t0 = time.time()
    tr = domeline_walk(ev)
    walk_s = time.time() - t0
    w = tr.steps[-1].eval
    out["walk"] = {"speedup": w.speedup_vs_gpu, "gap": 1 - w.speedup_vs_gpu / out["optimum"],
                   "evals": ev.log[-1].n, "stop": tr.stop, "point": w.point, "area_mm2": w.area_mm2,
                   "effort_months": w.effort_months, "wall_s": walk_s,
                   "first_within_1pct": next((e.n for e in ev.log if e.feasible and
                                              e.speedup_vs_gpu >= 0.99 * out["optimum"]), None),
                   "moves": [{"n": s.eval.n, "move": s.move, "reading": s.reading,
                              "speedup": s.eval.speedup_vs_gpu} for s in tr.steps]}

    rng = random.Random(args.seed)
    sample = rng.sample(rows, args.sample) + [best]
    ev2 = Evaluator(space, train, g)
    worst, mism, t0 = 0.0, [], time.time()
    for r in sample:
        e = ev2(parse(r))
        ev2.cache.clear()
        d = abs(e.time_s / float(r["time_s"]) - 1)
        worst = max(worst, d)
        if d > 1e-6 or int(e.feasible) != int(r["feasible"]):
            mism.append({"point": parse(r), "sweep_time_s": float(r["time_s"]), "now_time_s": e.time_s,
                         "sweep_feasible": r["feasible"], "now_feasible": e.feasible})
    per_eval = (time.time() - t0) / len(sample)
    out["retime"] = {"n": len(sample), "max_rel_diff": worst, "mismatches": mism[:20],
                     "n_mismatch": len(mism)}
    out["eval_s"] = per_eval
    out["n_sessions"] = len(train)

    dst = ROOT / "results/search/m0_repro.json"
    dst.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("retime",)} | {
        "retime_max_rel_diff": worst, "retime_mismatch": len(mism)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
