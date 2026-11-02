"""Deterministic team (rule agents) on the small and the large space.

  python experiments/team_det.py [--budget 3000] [--tag main]
Output: results/team/det_<tag>.json (summary) and results/team/det_<tag>_<run>.json (RunRecords)

Runs: the small space (== the walk), with and without the verifier; the large space with
FastEvaluator, coordinated moves never / on stall / first, verifier off and on.
"""

import argparse
import json
import time

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import Evaluator
from harness.large_space import FastEvaluator, LargeSpace
from harness.space import Space
from harness.team import point_verifier, rule_agents, team_search
from harness.tools import RunRecord, Toolbox

OPT = 9.484743          # sweep optimum (edge=vfu, illegal under the verifier's R1.sampling)
OPT_LEGAL = 9.4684      # fastest legal design (REPORT-M2)


def first_within(ev, target, tol=0.01):
    return next((e.n for e in ev.log if e.feasible and e.speedup_vs_gpu >= (1 - tol) * target), None)


def run(name, ev, space, train, args, verify=False, **kw):
    tb = Toolbox(ev, verifier=point_verifier(space, train) if verify else None)
    rec = RunRecord("M5-det", name, 0, {"budget": args.budget, "verify": verify, **{k: str(v) for k, v in kw.items()}})
    t = time.time()
    r = team_search(ev, tb, rule_agents(space), record=rec, budget=args.budget, **kw)
    wall = time.time() - t
    rec.write(ROOT / f"results/team/det_{args.tag}_{name}.json", ev, tb)
    e = r.final
    out = {"run": name, "speedup": e.speedup_vs_gpu, "time_s": e.time_s, "feasible": e.feasible,
           "area_mm2": e.area_mm2, "evals": ev.used, "stop": r.stop, "accepted": len(r.accepted) - 1,
           "proposals": r.proposals, "rejected": r.rejected, "restructures": r.restructures,
           "coordinated_accepted": sum(a.get("agent") == "supervisor" and a.get("move") != "start" for a in r.accepted),
           "first_within_1pct_opt": first_within(ev, OPT), "first_within_1pct_legal": first_within(ev, OPT_LEGAL),
           "wall_s": wall, "edge": {k: e.point.get(k) for k in ("edge", "embed", "sample") if k in e.point}}
    print(json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    ref = max(s.max_ctx for s in train + test)
    S = Space(lib, spec, ref_ctx=ref, host="cpu")
    L = LargeSpace(lib, spec, ref_ctx=ref, host="cpu")
    res = []
    res.append(run("small", Evaluator(S, train, g), S, train, args))
    res.append(run("small_verify", Evaluator(S, train, g), S, train, args, verify=True))
    for coord, k in (("never", None), ("stall", 20), ("stall", 100), ("first", None)):
        nm = f"large_{coord}" + (f"_k{k}" if k else "")
        res.append(run(nm, FastEvaluator(L, train, g), L, train, args, coordinate=coord, stall_k=k))
    res.append(run("large_first_verify", FastEvaluator(L, train, g), L, train, args, verify=True,
                   coordinate="first"))
    out = ROOT / f"results/team/det_{args.tag}.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
