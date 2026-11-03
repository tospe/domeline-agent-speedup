"""Capped ground truth on the large space: the best mapping under area and effort caps.

Area in this space is crossbar arrays x area per array (tile, ADCs, cell bits, tech) x xbar_scale,
plus vector/softmax unit area (under 2 mm2 at the largest sizes). Arrays are a sum over ops
(Integrator._resident at ref_ctx), so with the hardware fixed

  min  llm_time(m) + lam * arrays(m)        over mappings m

is the mapping DP with a per-op penalty (large_space.mapping_dp(lam=...)). Sweeping lam gives
the vertices of the lower convex hull of the (arrays, time) front: exact optimal points, and a
Lagrangian lower bound (the hull at the cap) on the capped optimum. Mappings between two
vertices are built by taking the first k layers from the larger vertex (`mix`); the best
feasible one bounds the capped optimum from above. Their difference is the duality gap.

Effort depends only on which techs a design uses and on the hardware knobs, so it is handled by
running the DP per tech pattern (which techs a mapping may use) and keeping patterns under the cap.
Tile and xbar_scale change area, not time (below the wall), so one hull serves all their values.

  hardware_grid(space)     time-relevant hardware settings per tech pattern
  hull(fe, hw, options)    the (arrays, time) hull vertices for one setting and pattern
  vertex_info(...)         exact area/effort/time of a vertex under each tile and scale
  bound_at(...)            the hull's lower bound at an area cap
  mix(...)                 designs between two vertices, layer by layer
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .large_space import HW_KNOBS, LargeSpace, mapping_dp, xbar_arrays  # noqa: F401
from .space import EFFORT, KNOBS, START

PLACE_TECHS = ("xbar", "vfu", "star")
TILE_SCALE = [(t, s) for t in ("isaac", "lean") for s in (1.0, 1.25)]
DEFAULT_EXCLUDE = {"embed": {"vfu"}, "sample": {"vfu"}}   # verifier: R2.transfer / R1.sampling
LAM_MAX = 1.0       # s per array: far above any time an array can save, so the min-array mapping


def pattern_options(space: LargeSpace, allowed, exclude=None):
    """Slot -> options when a mapping may use only the techs in `allowed` (and the host)."""
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    out = {}
    for s in space.slots.values():
        if s.role == "kv_bits":
            out[s.name] = list(s.options)
            continue
        vals = [v for v in s.options if (v == "cpu" or v in allowed) and v not in exclude.get(s.name, ())
                and v not in exclude.get(s.role, ())]
        out[s.name] = vals or ["cpu"]
    return out


def hardware_grid(patterns=None):
    """(pattern, which-run, hw) for every time-relevant hardware setting.

    pattern: frozenset of PLACE_TECHS a mapping may use. Knobs that do not change any op time
    are fixed (xbar_tile, xbar_scale: area only; vfu_mem: 64 GB, as memory costs neither area
    nor effort), and knobs of techs outside the pattern are left at START.
    """
    patterns = patterns or [frozenset(c) for n in range(4) for c in itertools.combinations(PLACE_TECHS, n)]
    for pat in patterns:
        knobs = {"link": KNOBS["link"]}
        if "xbar" in pat:
            knobs.update(xbar_tech=KNOBS["xbar_tech"], adc_per_ima=KNOBS["adc_per_ima"], cell_bits=KNOBS["cell_bits"])
        if "vfu" in pat:
            knobs["vfu_units"] = KNOBS["vfu_units"]
        if "star" in pat:
            knobs["star_units"] = KNOBS["star_units"]
        runs = [("plain", None)]
        if "vfu" in pat:
            runs.append(("vfu_attn", KNOBS["vfu_bw"]))
        names = list(knobs)
        for vals in itertools.product(*(knobs[k] for k in names)):
            base = {k: START[k] for k in HW_KNOBS}
            base.update(dict(zip(names, vals)), vfu_mem=64e9)
            for which, bws in runs:
                for bw in (bws or [START["vfu_bw"]]):
                    yield pat, which, {**base, "vfu_bw": bw}


@dataclass
class Vertex:
    lam: float
    llm_s: float              # DP time (LLM side)
    arrays: float
    used: tuple               # techs the mapping uses
    attn_vfu: bool
    info: Dict = field(default_factory=dict)   # (tile, scale) -> {"area", "effort", "time_s", "feasible"}


def _used(space, p):
    return tuple(v for v in PLACE_TECHS if space._used(p, v)), space._used(p, "vfu", ("attention",))


def hull(fe, hw, options, which, tol=1e-9, max_vertices=60):
    """Vertices (by arrays, decreasing; time increasing) of the lower hull of (arrays, llm time)."""
    space = fe.space
    pts = {}

    def solve(lam):
        r = mapping_dp(fe, {**space.start, **hw}, options, lam=lam, which=(which,))
        if r is not None:
            pts.setdefault(round(r.arrays, 6), (r, lam))
        return r

    a, b = solve(0.0), solve(LAM_MAX)
    if a is None:
        return []
    stack = [(a, b)] if b is not None and a.arrays - b.arrays > 0.5 else []
    while stack and len(pts) < max_vertices:
        x, y = stack.pop()
        lam = (y.llm_s - x.llm_s) / (x.arrays - y.arrays)
        c = solve(lam)
        line = x.llm_s + lam * x.arrays
        if c is not None and c.objective < line - tol * max(1.0, line) \
                and y.arrays + 0.5 < c.arrays < x.arrays - 0.5:
            stack += [(x, c), (c, y)]
    out = []
    for r, lam in sorted(pts.values(), key=lambda t: -t[0].arrays):
        used, attn = _used(space, r.point)
        out.append((Vertex(lam=lam, llm_s=r.llm_s, arrays=r.arrays, used=used, attn_vfu=attn), r.point))
    # Keep the lower convex hull only (the extremes may be dominated duplicates).
    hull_pts = []
    for v, p in sorted(out, key=lambda t: t[0].arrays):
        while len(hull_pts) >= 2:
            (v1, _), (v2, _) = hull_pts[-2], hull_pts[-1]
            # v2 is above the segment v1 -> v ?
            if (v2.llm_s - v1.llm_s) * (v.arrays - v1.arrays) >= (v.llm_s - v1.llm_s) * (v2.arrays - v1.arrays):
                hull_pts.pop()
            else:
                break
        if hull_pts and v.llm_s >= hull_pts[-1][0].llm_s:
            continue
        hull_pts.append((v, p))
    return hull_pts[::-1]         # most arrays (fastest) first


def vertex_info(fe, point, verifier=None):
    """Exact evaluation of a vertex's mapping under each (tile, scale)."""
    space = fe.space
    out = {}
    for tile, scale in TILE_SCALE:
        q = {**point, "xbar_tile": tile, "xbar_scale": scale}
        e = fe(q)
        info = {"area": e.area_mm2, "effort": e.effort_months, "time_s": e.time_s, "llm_s": e.llm_s,
                "feasible": e.feasible and e.wall_ctx is None}
        if verifier is not None:
            info["legal"] = verifier(q)["legal"]
        out[(tile, scale)] = info
    return out


def area_per_array(space, hw, tile):
    lib = space.fixed_hardware({**hw, "xbar_tile": tile}, False)[1]
    return lib[space.mvm_tech({**space.start, **hw})].area_mm2_per_unit


def pattern_effort(space, pat, hw, tile, attn_vfu):
    """Effort of a design that uses every tech of pat (the most a design of this run can take)."""
    lib = space.fixed_hardware({**hw, "xbar_tile": tile}, attn_vfu)[1]
    tech = space._tech({**space.start, **hw})
    e = sum(lib[tech[v]].effort_months for v in pat) + EFFORT["link"][hw["link"]]
    if attn_vfu:
        e += EFFORT["vfu_bw"][hw["vfu_bw"]]
    if "xbar" in pat:
        e += EFFORT["xbar_tile"][tile] + EFFORT["adc_per_ima"][hw["adc_per_ima"]] + EFFORT["cell_bits"][hw["cell_bits"]]
    return e


def bound_at(vertices, max_arrays):
    """Lower bound on llm time with arrays <= max_arrays from the hull (None if no mapping fits)."""
    vs = sorted(vertices, key=lambda v: v.arrays)
    if not vs or max_arrays < vs[0].arrays - 1e-9:
        return None
    if max_arrays >= vs[-1].arrays:
        return vs[-1].llm_s
    for lo, hi in zip(vs, vs[1:]):
        if lo.arrays <= max_arrays <= hi.arrays:
            f = (max_arrays - lo.arrays) / (hi.arrays - lo.arrays)
            return lo.llm_s + f * (hi.llm_s - lo.llm_s)
    return vs[-1].llm_s


def mix(space, big, small, k):
    """small's mapping with layers 0..k-1 (and, at k = all layers, the global slots) from big."""
    q = dict(small)
    n = space.model.layers
    for s in space.slots.values():
        if (s.layer is not None and s.layer < k) or (s.layer is None and k >= n):
            q[s.name] = big[s.name]
    return q
