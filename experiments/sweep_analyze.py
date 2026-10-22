"""Read the exhaustive sweep: the true optimum under the model, and how close each searcher got.

Reads results/search/sweep_<tag>.tsv (or, while the sweep is still running,
the parts written so far; the summary is then marked partial). Compares with
the searcher runs in results/search/<runs>.json and the effort staircase in
results/search/effort_<tag>.json.

  python experiments/sweep_analyze.py --tag main --runs main effort5
Output: results/search/sweep_<tag>_summary.json
"""

import argparse
import json
import math

import pandas as pd

from common import ROOT

from harness.space import KNOBS

COLS = list(KNOBS) + ["time_s", "llm_s", "speedup", "feasible", "area_mm2", "effort_months", "wall_ctx"]
TOL = 0.01          # "best" = smallest design within 1% of the fastest (same rule as Evaluator.best)


def load(tag):
    full = ROOT / "results/search" / f"sweep_{tag}.tsv"
    if full.exists():
        return pd.read_csv(full, sep="\t"), False
    parts = sorted((ROOT / "results/search" / f"sweep_{tag}.parts").glob("part*.tsv"))
    df = pd.concat([pd.read_csv(p, sep="\t", header=None, names=COLS) for p in parts], ignore_index=True)
    return df, True


def best(df, tol=TOL):
    """Same rule as Evaluator.best: smallest (area, effort) design within tol of the fastest feasible one."""
    f = df[df.feasible == 1]
    if f.empty:
        return None
    near = f[f.time_s <= f.time_s.min() * (1 + tol)]
    return near.sort_values(["area_mm2", "effort_months", "time_s"]).iloc[0]


def row(r):
    if r is None:
        return None
    return {"point": {k: (r[k].item() if hasattr(r[k], "item") else r[k]) for k in KNOBS},
            "speedup": float(r.speedup), "time_s": float(r.time_s),
            "area_mm2": float(r.area_mm2), "effort_months": float(r.effort_months)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--runs", nargs="*", default=["main", "effort5"])
    args = ap.parse_args()
    df, partial = load(args.tag)
    feas = df[df.feasible == 1]
    opt = best(df)
    top = float(feas.speedup.max())
    out = {
        "partial": partial, "n_designs": len(df), "n_feasible": len(feas),
        "optimum": row(opt), "fastest_speedup": top,
        "frac_feasible_within": {f"{p:g}": float((feas.speedup >= top * (1 - p)).mean()) for p in (0.01, 0.05, 0.10)},
        "n_within": {f"{p:g}": int((feas.speedup >= top * (1 - p)).sum()) for p in (0.01, 0.05, 0.10)},
        "frac_beat_gpu": float((feas.speedup > 1).mean()),
        "by_effort_cap": {}, "searchers": {},
    }
    for cap in [0, 1, 3, 4, 5, 7, 8, 12, 16, math.inf]:
        b = best(df[df.effort_months <= cap])
        out["by_effort_cap"]["inf" if cap == math.inf else f"{cap:g}"] = row(b)
    for tag in args.runs:
        p = ROOT / "results/search" / f"{tag}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cap = d["caps"]["effort_cap_months"]
        cap = math.inf if cap is None else cap
        ref = best(df[df.effort_months <= cap])
        ref_s = float(df[(df.effort_months <= cap) & (df.feasible == 1)].speedup.max())
        res = []
        for r in d["runs"]:
            if "best" not in r:
                continue
            s = r["best"]["speedup"]
            # Rank among all feasible designs under the same cap (1 = fastest).
            pool = df[(df.effort_months <= cap) & (df.feasible == 1)].speedup
            res.append({"method": r["method"], "seed": r.get("seed"), "speedup": s,
                        "gap_pct": 100 * (1 - s / ref_s), "evals": r["evals"],
                        "rank": int((pool > s * (1 + 1e-9)).sum()) + 1,
                        "n_first_within_1pct": next((e["n"] for e in r["log"]
                                                     if e["feasible"] and e["speedup"] >= ref_s * 0.99), None)})
        out["searchers"][tag] = {"effort_cap": None if cap == math.inf else cap,
                                 "reference": row(ref), "fastest_speedup": ref_s, "runs": res}
    eff = ROOT / "results/search" / f"effort_{args.tag}.json"
    if eff.exists():
        out["effort_check"] = []
        for r in json.loads(eff.read_text())["rows"]:
            cap = math.inf if r["cap"] is None else r["cap"]
            b = out["by_effort_cap"].get("inf" if cap == math.inf else f"{cap:g}")
            out["effort_check"].append({"cap": r["cap"], "method": r["method"], "found": r["speedup"],
                                        "sweep": b["speedup"] if b else None})
    dst = ROOT / "results/search" / f"sweep_{args.tag}_summary.json"
    dst.write_text(json.dumps(out, indent=1, default=lambda x: None if x == math.inf else str(x)))
    print(f"{'PARTIAL ' if partial else ''}{len(df)} designs, {len(feas)} feasible; "
          f"optimum {top:.3f}x, best {out['optimum']['speedup']:.3f}x at {out['optimum']['area_mm2']:.0f} mm2, "
          f"{out['optimum']['effort_months']:g} months")
    print("  within 1/5/10%:", out["n_within"], " beat GPU:", f"{out['frac_beat_gpu']:.1%}")
    for k, v in out["by_effort_cap"].items():
        print(f"  cap {k:>3}: {v['speedup'] if v else 0:.2f}x")
    for tag, s in out["searchers"].items():
        print(f"  [{tag}] sweep best {s['fastest_speedup']:.3f}x")
        for r in s["runs"]:
            print(f"    {r['method']:13s} seed {r['seed']}: {r['speedup']:.3f}x gap {r['gap_pct']:5.2f}% "
                  f"rank {r['rank']:6d}  first<=1% at {r['n_first_within_1pct']}")
    print(dst)


if __name__ == "__main__":
    main()
