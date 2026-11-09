"""The verifier against the exhaustive sweep: does every design the Evaluator calls
feasible pass the verifier, and why not when it does not?

Reads results/search/sweep_main.tsv.gz (the 14-knob space on the search sessions),
verifies every row (or a seeded --sample) and writes a confusion table by rule set
to results/search/verify_sweep_<tag>.json.

    python experiments/verify_sweep.py --workers 16
    python experiments/verify_sweep.py --sample 2000 --tag sample
"""

import argparse
import gzip
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from common import library, model, sessions, split  # noqa: E402
from harness.space import KNOBS, Space  # noqa: E402

SWEEP = ROOT / "results/search/sweep_main.tsv.gz"


def parse(row, cols):
    p = {}
    for k in KNOBS:
        if k not in cols:
            continue
        s = row[cols.index(k)]
        p[k] = next(o for o in KNOBS[k] if str(o) == s or (not isinstance(o, str) and float(o) == float(s)))
    return p, int(row[cols.index("feasible")])


_ENV = {}


def _init():
    from harness.verify import Caps
    train, test = split(sessions())
    spec, lib = model(), library()
    _ENV.update(space=Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu"),
                train=train, caps=Caps(), cache={})


def check(item):
    from harness.verify import verify_point
    p, feasible = item
    r = verify_point(_ENV["space"], p, _ENV["train"], _ENV["caps"], ops_cache=_ENV["cache"])
    if len(_ENV["cache"]) > 64:
        _ENV["cache"].clear()
    return feasible, r.legal, tuple(sorted(r.rules())), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help="seeded random sample of N rows (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    with gzip.open(SWEEP, "rt") as f:
        cols = f.readline().rstrip("\n").split("\t")
        rows = [parse(l.rstrip("\n").split("\t"), cols) for l in f]
    if args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)
    t0 = time.time()
    table, examples = Counter(), {}
    with mp.Pool(args.workers, initializer=_init) as pool:
        for feasible, legal, rules, p in pool.imap_unordered(check, rows, chunksize=256):
            key = (feasible, legal, rules)
            table[key] += 1
            examples.setdefault(key, p)
    wall = time.time() - t0
    out = {"sweep": str(SWEEP.relative_to(ROOT)), "rows": len(rows), "sample": args.sample, "seed": args.seed,
           "wall_s": wall,
           "evaluator_feasible": sum(n for (f, _, _), n in table.items() if f),
           "verifier_legal": sum(n for (_, l, _), n in table.items() if l),
           "feasible_and_legal": sum(n for (f, l, _), n in table.items() if f and l),
           "feasible_not_legal": sum(n for (f, l, _), n in table.items() if f and not l),
           "legal_not_feasible": sum(n for (f, l, _), n in table.items() if l and not f),
           "table": [{"feasible": bool(f), "legal": l, "rules": list(r), "n": n, "example": examples[(f, l, r)]}
                     for (f, l, r), n in sorted(table.items(), key=lambda x: -x[1])]}
    dst = ROOT / "results/search" / f"verify_sweep_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "table"}, indent=1))
    for t in out["table"]:
        print(f"  feasible={t['feasible']!s:5} legal={t['legal']!s:5} n={t['n']:7}  {t['rules']}")
    print("wrote", dst)


if __name__ == "__main__":
    main()
