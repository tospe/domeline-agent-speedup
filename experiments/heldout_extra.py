"""Second held-out set: re-time the chosen designs on sessions collected after the search.

These 11 SWE-bench tasks were blocked by Docker Hub rate limits during trace
collection and were run after every search had finished, so no searcher (and
no decision in the space or the library) saw them. The design is built exactly
as in the search (same Space, same ref_ctx from the frozen workload).

  python experiments/heldout_extra.py --runs main effort5
Output: results/search/heldout_extra.json
"""

import argparse
import json
import math

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import Evaluator
from harness.space import Space


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["main", "effort5"])
    args = ap.parse_args()
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    extra = sessions(which="extra")
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    out = {"n_extra": len(extra), "sessions": [], "designs": {}}
    for s in extra:
        out["sessions"].append({"name": s.name, "turns": len(s.turns), "max_ctx": s.max_ctx,
                                "exit": s.meta["exit"], "gpu_kv": s.meta["gpu_kv"]})
    for tag in args.runs:
        d = json.loads((ROOT / "results/search" / f"{tag}.json").read_text())
        walk = next(r for r in d["runs"] if r["method"] == "walk")
        point = walk["best"]["point"]
        ev = Evaluator(space, extra, g)
        e = ev(point)
        per = []
        for s in extra:
            es = Evaluator(space, [s], g)(point)
            per.append({"name": s.name, "speedup": es.speedup_vs_gpu, "gpu_s": es.time_s * es.speedup_vs_gpu,
                        "time_s": es.time_s})
        amdahl = ev.gpu_time / ev.tools_time if ev.tools_time else math.inf
        out["designs"][tag] = {"point": point, "search_speedup": walk["best"]["speedup"],
                               "heldout_speedup": walk["best"].get("heldout", {}).get("speedup"),
                               "extra_speedup": e.speedup_vs_gpu, "feasible": e.feasible,
                               "gpu_s": ev.gpu_time, "tools_s": ev.tools_time, "amdahl": amdahl,
                               "per_session": per}
        print(f"[{tag}] walk design: search {walk['best']['speedup']:.2f}x, held-out "
              f"{out['designs'][tag]['heldout_speedup']:.2f}x, extra ({len(extra)} sessions) "
              f"{e.speedup_vs_gpu:.2f}x (Amdahl {amdahl:.1f}x)")
        for p in per:
            print(f"    {p['name']:40s} {p['speedup']:6.2f}x")
    dst = ROOT / "results/search/heldout_extra.json"
    dst.write_text(json.dumps(out, indent=1, default=lambda x: None if x == math.inf else x))
    print(dst)


if __name__ == "__main__":
    main()
