"""Capped ground truth on the large space (harness/dp_caps.py): best design per area and effort cap.

  python experiments/dp_caps.py --workers 18 --tag main          # hulls, then caps
  python experiments/dp_caps.py --tag main --caps-only           # caps from saved hulls
Output: results/large/dp_caps_hulls_<tag>.json (every hardware setting's hull),
        results/large/dp_caps_<tag>.json (per cap: lower bound, best design found, gap)

Phase 1: for every time-relevant hardware setting and tech pattern, the hull of (arrays, LLM time)
by the Lagrangian mapping DP. Phase 2, per (area cap, effort cap): the lower bound is the best hull
value at the cap over settings whose effort fits (xbar_tile and xbar_scale applied to area); the
best design comes from hull vertices that fit and, in order of their bound (branch and bound), from
mixes of the two vertices that bracket the cap, re-evaluated exactly (FastEvaluator, verifier).
Sampling and the embedding lookup stay on the host by default (--allow-edge-vfu to lift it).
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

from common import ROOT, gpu, library, model, sessions, split

from harness.dp_caps import (DEFAULT_EXCLUDE, TILE_SCALE, area_per_array, bound_at, hardware_grid, hull, mix,
                             pattern_effort, pattern_options)
from harness.large_space import FastEvaluator, LargeSpace, mapping_dp

AREA_CAPS = [math.inf, 8000, 6000, 5000, 4000, 3000]
EFFORT_CAPS = [math.inf, 7, 5]

_W = {}


def _env(exclude):
    if "fe" not in _W:
        spec, lib, g = model(), library(), gpu()
        train, test = split(sessions())
        L = LargeSpace(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
        _W.update(fe=FastEvaluator(L, train, g), L=L, train=train, exclude=exclude)
    return _W


def _hull_job(args):
    i, pat, which, hw, exclude = args
    W = _env(exclude)
    t = time.time()
    H = hull(W["fe"], hw, pattern_options(W["L"], pat, exclude), which)
    W["fe"].cache.clear()
    W["fe"].log.clear()
    return {"i": i, "pattern": sorted(pat), "which": which, "hw": hw, "seconds": time.time() - t,
            "vertices": [{"lam": v.lam, "llm_s": v.llm_s, "arrays": v.arrays, "used": list(v.used),
                          "attn_vfu": v.attn_vfu} for v, _ in H]}


class V:
    def __init__(self, d):
        self.__dict__.update(d)


def fixed_area(L, hw, used):
    lib = L.fixed_hardware(hw, False)[1]
    a = 0.0
    if "vfu" in used:
        a += lib[L.vfu].area_mm2_per_unit * hw["vfu_units"]
    if "star" in used:
        a += lib[L.star].area_mm2_per_unit * hw["star_units"]
    return a


def vertex_area(L, unit, v, tile, scale):
    """Area of a vertex's design (Space.area_mm2's formula: units of used techs x area per unit)."""
    hw = unit["hw"]
    a = math.ceil(scale * v.arrays) * area_per_array(L, hw, tile) if "xbar" in v.used else 0.0
    return a + fixed_area(L, hw, v.used)


def _cap_job(args):
    area_cap, effort_cap, units, exclude, max_refine = args
    W = _env(exclude)
    L, fe, train = W["L"], W["fe"], W["train"]
    from harness.team import point_verifier
    verifier = point_verifier(L, train)
    tools = fe.tools_time
    # Lower bound, and the per-setting bounds for branch and bound.
    cands = []
    for u in units:
        vs = [V(v) for v in u["vertices"]]
        if not vs:
            continue
        attn = u["which"] == "vfu_attn"
        for tile, scale in TILE_SCALE:
            if "xbar" not in u["pattern"] and (tile, scale) != ("isaac", 1.0):
                continue
            if pattern_effort(L, u["pattern"], u["hw"], tile, attn) > effort_cap + 1e-9:
                continue
            a = area_per_array(L, u["hw"], tile) * scale
            max_arr = math.inf if math.isinf(area_cap) else area_cap / a
            b = bound_at(vs, max_arr)
            if b is not None:
                cands.append((b, u, tile, scale))
    cands.sort(key=lambda c: c[0])
    bound = cands[0][0] + tools if cands else None
    best = None            # (time_s, point, info)

    def consider(point, how):
        nonlocal best
        e = fe(point)
        ok = e.feasible and e.wall_ctx is None and e.area_mm2 <= area_cap + 1e-9 and e.effort_months <= effort_cap + 1e-9
        # Faster wins; at equal time (1e-12 rel) the smaller area, then the smaller effort.
        better = best is None or e.time_s < best[0] * (1 - 1e-12) or (
            e.time_s <= best[0] * (1 + 1e-12) and (e.area_mm2, e.effort_months) < (best[2]["area_mm2"], best[2]["effort_months"]))
        if ok and better:
            if verifier(point)["legal"]:
                best = (e.time_s, dict(point), {"how": how, "speedup": e.speedup_vs_gpu, "area_mm2": e.area_mm2,
                                                "effort_months": e.effort_months, "llm_s": e.llm_s})
        return e

    def point_at(u, lam, tile, scale):
        r = mapping_dp(fe, {**L.start, **u["hw"]}, pattern_options(L, u["pattern"], exclude), lam=lam,
                       which=(u["which"],))
        return {**r.point, "xbar_tile": tile, "xbar_scale": scale}

    # Vertices that fit (by the area/effort formula), fastest first.
    fits = []
    for b, u, tile, scale in cands:
        for v in (V(x) for x in u["vertices"]):
            if vertex_area(L, u, v, tile, scale) <= area_cap and \
                    pattern_effort(L, v.used, u["hw"], tile, v.attn_vfu) <= effort_cap + 1e-9:
                fits.append((v.llm_s, v, u, tile, scale))
    fits.sort(key=lambda f: f[0])
    seen = set()
    for t, v, u, tile, scale in fits[:200]:
        k = (id(u), v.lam, tile, scale)
        if k in seen:
            continue
        seen.add(k)
        consider(point_at(u, v.lam, tile, scale), "vertex")
        if best and t + tools > best[0] * (1 + 1e-12):
            break
    # Branch and bound over settings: mix the vertices that bracket the cap.
    refined = 0
    for b, u, tile, scale in cands:
        if best is not None and b + tools >= best[0] * (1 - 1e-9):
            break
        if refined >= max_refine:
            break
        refined += 1
        vs = sorted((V(x) for x in u["vertices"]), key=lambda v: -v.arrays)
        fit = [v for v in vs if vertex_area(L, u, v, tile, scale) <= area_cap]
        over = [v for v in vs if vertex_area(L, u, v, tile, scale) > area_cap]
        if not fit:
            continue
        lo = fit[0]
        if not over:
            consider(point_at(u, lo.lam, tile, scale), "vertex")
            continue
        hi = over[-1]
        p_hi, p_lo = point_at(u, hi.lam, tile, scale), point_at(u, lo.lam, tile, scale)
        n = L.model.layers
        a, z = 0, n                     # largest k whose mix fits, by bisection on area
        while a < z:
            mid = (a + z + 1) // 2
            if fe(mix(L, p_hi, p_lo, mid)).area_mm2 <= area_cap:
                a = mid
            else:
                z = mid - 1
        for k in sorted({a, max(0, a - 1)}):
            consider(mix(L, p_hi, p_lo, k), f"mix k={k}/{n}")
    out = {"area_cap": area_cap, "effort_cap": effort_cap, "bound_s": bound,
           "bound_speedup": fe.gpu_time / bound if bound else None, "settings_refined": refined,
           "settings_with_bound": len(cands)}
    if best:
        t, p, info = best
        out.update(best_time_s=t, gap=(t - bound) / t if bound else None, **info,
                   point_summary=summarize(L, p), point=p)
    print(json.dumps({k: v for k, v in out.items() if k not in ("point", "point_summary")}), flush=True)
    return out


def summarize(L, p):
    from collections import Counter
    roles = {}
    for s in L.slots.values():
        roles.setdefault(s.role, Counter())[str(p[s.name])] += 1
    return {"hw": {k: p[k] for k in ("xbar_tech", "xbar_scale", "xbar_tile", "adc_per_ima", "cell_bits", "vfu_units",
                                     "vfu_bw", "star_units", "link")},
            "slots": {r: dict(c) for r, c in roles.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--caps-only", action="store_true")
    ap.add_argument("--allow-edge-vfu", action="store_true")
    ap.add_argument("--max-refine", type=int, default=400)
    ap.add_argument("--plain-only", action="store_true",
                    help="caps from hulls without attention on the vector unit (output tag <tag>_plain)")
    args = ap.parse_args()
    exclude = {} if args.allow_edge_vfu else DEFAULT_EXCLUDE
    hp = ROOT / f"results/large/dp_caps_hulls_{args.tag}.json"
    t0 = time.time()
    if not args.caps_only:
        jobs = [(i, pat, which, hw, exclude) for i, (pat, which, hw) in enumerate(hardware_grid())]
        if args.limit:
            jobs = jobs[: args.limit]
        # Slow jobs (VFU attention) first, so the pool ends evenly.
        jobs.sort(key=lambda j: (j[2] != "vfu_attn", -len(j[1])))
        print(f"{len(jobs)} hull jobs, {args.workers} workers", flush=True)
        units = []
        with mp.Pool(args.workers) as pool:
            for n, r in enumerate(pool.imap_unordered(_hull_job, jobs, chunksize=4)):
                units.append(r)
                if n % 200 == 0:
                    print(f"{n}/{len(jobs)} hulls, {time.time() - t0:.0f} s", flush=True)
        units.sort(key=lambda u: u["i"])
        hp.parent.mkdir(parents=True, exist_ok=True)
        hp.write_text(json.dumps({"exclude": {k: sorted(v) for k, v in exclude.items()},
                                  "seconds": time.time() - t0, "units": units}))
        print(f"hulls done in {time.time() - t0:.0f} s", flush=True)
    units = json.loads(hp.read_text())["units"]
    tag = args.tag
    if args.plain_only:           # vfu_bw applies to every VFU op iff some attention is on the VFU
        units = [u for u in units if u["which"] == "plain"]
        tag = f"{args.tag}_plain"
    caps = [(a, e, units, exclude, args.max_refine) for a in AREA_CAPS for e in EFFORT_CAPS]
    with mp.Pool(min(args.workers, len(caps))) as pool:
        res = pool.map(_cap_job, caps)
    W = _env(exclude)
    fe = W["fe"]
    floor = min(v["llm_s"] for u in units for v in u["vertices"]) + fe.tools_time
    out = {"tag": tag, "plain_only": args.plain_only, "exclude": {k: sorted(v) for k, v in exclude.items()}, "n_settings": len(units),
           "hull_seconds_total": sum(u["seconds"] for u in units),
           "dp_floor_s": floor, "dp_floor_speedup": fe.gpu_time / floor, "tools_s": fe.tools_time,
           "gpu_s": fe.gpu_time, "caps": res, "wall_s": time.time() - t0}
    (ROOT / f"results/large/dp_caps_{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    print("dp floor", floor, fe.gpu_time / floor)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
