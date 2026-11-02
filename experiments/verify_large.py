"""The verifier on the large space (harness/large_space.py): FastEvaluator-feasible vs
verifier-legal on seeded samples, by rule set.

Two samples: `uniform` (every slot and hardware knob drawn independently) and `local`
(the tied headline design with 1-8 random slots changed, which is where a search
spends its time). Writes results/search/verify_large_<tag>.json.

    python experiments/verify_large.py --n 400
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from common import gpu, library, model, sessions, split  # noqa: E402
from harness.large_space import HW_KNOBS, FastEvaluator, LargeSpace  # noqa: E402
from harness.space import KNOBS  # noqa: E402
from harness.verify import verify_point  # noqa: E402

HEADLINE = {"mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
            "softmax": "star", "vector": "star", "edge": "cpu", "xbar_scale": 1.0, "vfu_units": 1,
            "vfu_bw": 64e9, "vfu_mem": 4e9, "star_units": 64, "link": "cxl", "kv_bits": 8}


def uniform(space, rng):
    p = {k: rng.choice(KNOBS[k]) for k in HW_KNOBS}
    p.update({s.name: rng.choice(s.options) for s in space.slots.values()})
    return p


def local(space, rng, base):
    p = dict(base)
    for s in rng.sample(list(space.slots.values()), rng.randint(1, 8)):
        p[s.name] = rng.choice(s.options)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="points per sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    train, test = split(sessions())
    spec, lib = model(), library()
    space = LargeSpace(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    fe = FastEvaluator(space, train, gpu())
    base = space.tie(HEADLINE)
    out = {"n": args.n, "seed": args.seed, "samples": {}}
    for name, draw in (("uniform", lambda r: uniform(space, r)), ("local", lambda r: local(space, r, base))):
        rng = random.Random(args.seed)
        table, examples, seen = Counter(), {}, set()
        t0, t_ver, mism = time.time(), 0.0, 0
        while len(seen) < args.n:
            p = space.canonical(draw(rng))
            k = space.key(p)
            if k in seen:
                continue
            seen.add(k)
            e = fe(p)
            t1 = time.time()
            r = verify_point(space, p, train)
            t_ver += time.time() - t1
            mism += any(v.op == "design" and v.rule.startswith("R2") for v in r.violations)
            key = (bool(e.feasible), r.legal, tuple(sorted(r.rules())))
            table[key] += 1
            if key not in examples:
                examples[key] = {"reasons": [v.reason for v in r.violations[:3]], "eval_reason": e.reason,
                                 "speedup": e.speedup_vs_gpu}
        out["samples"][name] = {
            "wall_s": time.time() - t0, "verify_s_per_point": t_ver / args.n,
            "integrator_mismatch_points": mism,
            "feasible": sum(n for (f, _, _), n in table.items() if f),
            "legal": sum(n for (_, l, _), n in table.items() if l),
            "feasible_not_legal": sum(n for (f, l, _), n in table.items() if f and not l),
            "legal_not_feasible": sum(n for (f, l, _), n in table.items() if l and not f),
            "table": [{"feasible": f, "legal": l, "rules": list(rs), "n": n, "example": examples[(f, l, rs)]}
                      for (f, l, rs), n in sorted(table.items(), key=lambda x: -x[1])]}
        s = out["samples"][name]
        print(f"{name}: feasible {s['feasible']} legal {s['legal']} feasible&!legal {s['feasible_not_legal']} "
              f"legal&!feasible {s['legal_not_feasible']}  integrator mismatches {mism}  verify {s['verify_s_per_point'] * 1e3:.0f} ms/pt")
        for row in s["table"]:
            print(f"   feasible={row['feasible']!s:5} legal={row['legal']!s:5} n={row['n']:4} {row['rules']}")
    dst = ROOT / "results/search" / f"verify_large_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("wrote", dst)


if __name__ == "__main__":
    main()
