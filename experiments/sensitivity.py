"""Sensitivity of the headline to the uncertain technology numbers.

Each uncertain parameter (the VERIFY entries in lib/tech, the link presets,
the host CPU attention kernel) is scaled by 1/2 and 2. For each change we
report (a) the speedup of the fixed best design and (b) the speedup after
re-running the Domeline walk under the changed number, since a designer
would re-optimize.

  python experiments/sensitivity.py --tag main
Output: results/search/sensitivity_<tag>.json
"""

import argparse
import copy
import json

from common import ROOT, gpu, library, model, sessions, split

from harness import space as S
from harness.design import Link
from harness.evaluate import Evaluator
from harness.search import domeline_walk
from harness.space import Space

XB = ("xbar_isaac", "xbar_transpose", "xbar_isaac_wv")

# name -> (description, function(lib, factor) mutating a copy of the library / link presets)
PARAMS = {
    "xbar_read": ("Crossbar read cycle", lambda lib, f: [_scale(lib[t], "t_read_s", f) for t in XB]),
    "xbar_write": ("Crossbar row write", lambda lib, f: [_scale(lib[t], "t_write_row_s", f) for t in XB]),
    "xbar_area": ("Crossbar area / array", lambda lib, f: [_area(lib[t], f) for t in XB]),
    "vfu_ops": ("Vector unit throughput", lambda lib, f: _scale(lib["vfu_puma"], "ops_per_s", f)),
    "star_ops": ("Nonlinear unit throughput", lambda lib, f: _scale(lib["softmax_star"], "ops_per_s", f)),
    "star_bw": ("Nonlinear unit bandwidth", lambda lib, f: _scale(lib["softmax_star"], "mem_bw_Bps", f)),
    "link": ("CPU-PIM link (bw, 1/latency)", None),
    "host_attn": ("Host attention kernel time", None),
}


def _scale(t, key, f):
    t.params = {**t.params, key: t.params[key] * f}


def _area(t, f):
    t.area_mm2_per_unit *= f


def apply(name, factor, lib):
    lib = copy.deepcopy(lib)
    links = dict(S.LINKS)
    fn = PARAMS[name][1]
    if fn:
        fn(lib, factor)
    elif name == "link":
        links = {k: Link(bw_Bps=v.bw_Bps * factor, launch_s=v.launch_s / factor, sync_s=v.sync_s / factor,
                         t0_s=v.t0_s / factor, alpha_s_per_unit=v.alpha_s_per_unit, noc_bw_Bps=v.noc_bw_Bps)
                 for k, v in S.LINKS.items()}
    elif name == "host_attn":
        # A faster/slower host attention kernel: scale the measured profile's attention rows.
        from harness.tech import cpu_profile
        prof = copy.deepcopy(cpu_profile(lib["cpu"]))
        for k in ("attn_score", "attn_value", "softmax"):
            xs, ys = prof.curves[k]
            prof.curves[k] = (xs, ys * factor)
        import harness.tech as T
        key = (lib["cpu"].p("profile_tsv"), lib["cpu"].p("threads", 8))
        T._PROFILES[key] = prof
    return lib, links


def evaluate(spec, lib, links, train, g, point):
    saved = dict(S.LINKS)
    S.LINKS.clear(); S.LINKS.update(links)
    try:
        sp = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train), host="cpu")
        fixed = Evaluator(sp, train, g)(point)
        ev = Evaluator(sp, train, g, budget=150)
        domeline_walk(ev)
        b = ev.best()
        return fixed, b, ev.used
    finally:
        S.LINKS.clear(); S.LINKS.update(saved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    import harness.tech as T
    spec, base_lib, g = model(), library(), gpu()
    train, test = split(sessions())
    workload = train + test
    base_profiles = dict(T._PROFILES)

    sp = Space(base_lib, spec, ref_ctx=max(s.max_ctx for s in workload), host="cpu")
    ev = Evaluator(sp, workload, g, budget=150)
    domeline_walk(ev)
    best = ev.best()
    out = {"base": {"speedup": best.speedup_vs_gpu, "area_mm2": best.area_mm2,
                    "point": {k: (float(v) if isinstance(v, float) else v) for k, v in best.point.items()}},
           "rows": []}
    print(f"base: {best.speedup_vs_gpu:.2f}x, {best.area_mm2:.0f} mm2")
    for name, (desc, _) in PARAMS.items():
        for f in (0.5, 2.0):
            T._PROFILES.clear(); T._PROFILES.update(base_profiles)
            lib, links = apply(name, f, base_lib)
            fixed, b, used = evaluate(spec, lib, links, workload, g, best.point)
            row = {"param": name, "desc": desc, "factor": f,
                   "fixed_speedup": fixed.speedup_vs_gpu if fixed.feasible else None,
                   "fixed_area": fixed.area_mm2,
                   "reopt_speedup": b.speedup_vs_gpu if b else None,
                   "reopt_area": b.area_mm2 if b else None, "reopt_evals": used}
            out["rows"].append(row)
            print(f"{desc:30s} x{f:<4}: fixed {row['fixed_speedup'] or 0:.2f}x, re-optimized "
                  f"{row['reopt_speedup'] or 0:.2f}x ({row['reopt_area'] or 0:.0f} mm2)", flush=True)
    T._PROFILES.clear(); T._PROFILES.update(base_profiles)
    dst = ROOT / "results/search" / f"sensitivity_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
