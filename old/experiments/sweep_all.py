"""Exhaustive sweep: every distinct design in the space, on the search set.

This is the reference the searchers are measured against ("within X% of the
best design, in M evaluations where the sweep needs S"). Designs that differ
only in knobs that do not matter (e.g. vector-unit size when nothing runs on
the vector unit) are evaluated once.

  python experiments/sweep_all.py --workers 20 --tag main
Output: results/search/sweep_<tag>.tsv (one row per distinct design)
"""

import argparse
import itertools
import multiprocessing as mp
import time
from pathlib import Path

from common import ROOT, gpu, library, model, sessions, split

from harness.space import KNOBS, key

COLS = list(KNOBS) + ["time_s", "llm_s", "speedup", "feasible", "area_mm2", "effort_months", "wall_ctx"]


def distinct_points():
    seen = {}
    for vals in itertools.product(*KNOBS.values()):
        p = dict(zip(KNOBS, vals))
        k = key(p)
        if k not in seen:
            seen[k] = p
    return list(seen.values())


def worker(args):
    wid, points, out_path = args
    from harness.evaluate import Evaluator
    from harness.space import Space
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    ev = Evaluator(space, train, g)
    t0 = time.time()
    with open(out_path, "w") as f:
        for i, p in enumerate(points):
            e = ev(p)
            ev.cache.clear()       # keep memory flat; every point is distinct anyway
            ev.log.clear()
            row = [p[k] for k in KNOBS] + [f"{e.time_s:.6f}", f"{e.llm_s:.6f}", f"{e.speedup_vs_gpu:.6f}",
                                            int(e.feasible), f"{e.area_mm2:.3f}", f"{e.effort_months:g}",
                                            e.wall_ctx if e.wall_ctx is not None else ""]
            f.write("\t".join(str(x) for x in row) + "\n")
            if wid == 0 and i % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"worker 0: {i}/{len(points)} ({rate:.1f}/s, eta {len(points) / rate / 3600:.1f} h)",
                      flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=0, help="first N points only (testing)")
    args = ap.parse_args()
    pts = distinct_points()
    if args.limit:
        pts = pts[: args.limit]
    print(f"{len(pts)} distinct designs, {args.workers} workers", flush=True)
    parts = ROOT / "results/search" / f"sweep_{args.tag}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [(w, pts[w:: args.workers], parts / f"part{w:02d}.tsv") for w in range(args.workers)]
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        files = pool.map(worker, chunks)
    dst = ROOT / "results/search" / f"sweep_{args.tag}.tsv"
    with open(dst, "w") as out:
        out.write("\t".join(COLS) + "\n")
        for fp in files:
            out.write(Path(fp).read_text())
    print(f"{dst}: {len(pts)} designs in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
