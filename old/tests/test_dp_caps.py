"""M6: capped ground truth (harness/dp_caps.py) on a 2-layer instance, against brute force."""

import itertools
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from common import gpu, library, model  # noqa: E402

from harness.dp_caps import (area_per_array, bound_at, hull, mix, pattern_effort, pattern_options)  # noqa: E402
from harness.large_space import HW_KNOBS, FastEvaluator, LargeSpace, mapping_dp  # noqa: E402
from harness.space import START  # noqa: E402
from harness.synthetic import synthetic_sessions  # noqa: E402

WALK = {**START, "mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
        "softmax": "star", "vector": "cpu", "star_units": 64, "kv_bits": 8, "xbar_scale": 1.25}
FREE = ["L0.attention", "L0.kv_bits", "L0.o_proj", "L0.q_proj", "L1.attention", "L1.kv_bits", "L1.gate_proj",
        "L1.down_proj", "lm_head", "L1.softmax"]


@pytest.fixture(scope="module")
def tiny():
    spec = replace(model(), layers=2)
    sess = synthetic_sessions(2, seed=3, turns=(3, 5))
    L = LargeSpace(library(), spec, ref_ctx=max(s.max_ctx for s in sess), host="cpu")
    fe = FastEvaluator(L, sess, gpu())
    hw = {k: WALK[k] for k in HW_KNOBS}
    base = {**L.tie(WALK), **hw}
    opts = {s: [base[s]] for s in L.slots}
    for s in FREE:
        opts[s] = list(L.slots[s].options)
    allp = []
    for vals in itertools.product(*(opts[s] for s in FREE)):
        p = {**base, **dict(zip(FREE, vals))}
        e = fe(p)
        if e.wall_ctx is None:
            allp.append((e.llm_s, L.xbar_need(p), p, e))
    return L, fe, hw, base, opts, allp


@pytest.mark.parametrize("lam", [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
def test_lagrangian_dp_equals_brute_force(tiny, lam):
    L, fe, hw, base, opts, allp = tiny
    r = mapping_dp(fe, base, opts, lam=lam)
    assert r.objective == pytest.approx(min(t + lam * a for t, a, _, _ in allp), rel=1e-12)
    assert r.arrays == L.xbar_need(r.point)          # the DP's arrays are what capacity (and area) count
    assert fe(r.point).llm_s == pytest.approx(r.llm_s, rel=1e-12)


def test_hull_is_a_lower_bound_and_its_vertices_are_optimal(tiny):
    L, fe, hw, base, opts, allp = tiny
    H = hull(fe, hw, opts, "plain")
    vs = [v for v, _ in H]
    assert len(vs) >= 3
    for v, p in H:              # every vertex is on the brute-force front
        assert not any(a <= v.arrays and t < v.llm_s * (1 - 1e-12) for t, a, _, _ in allp)
    for cap in sorted({a for _, a, _, _ in allp}):
        b = bound_at(vs, cap)
        best = min(t for t, a, _, _ in allp if a <= cap)
        assert b is not None and b <= best * (1 + 1e-12)


def test_area_and_effort_formulas(tiny):
    """vertex area = ceil(scale * arrays) * area per array (+ unit areas); effort from the techs used."""
    L, fe, hw, base, opts, allp = tiny
    for t, a, p, e in allp[::97]:
        used = tuple(v for v in ("xbar", "vfu", "star") if L._used(p, v))
        attn = L._used(p, "vfu", ("attention",))
        for tile in ("isaac", "lean"):
            q = {**p, "xbar_tile": tile}
            area = math.ceil(1.25 * a) * area_per_array(L, hw, tile) if "xbar" in used else 0.0
            lib = L.fixed_hardware(hw, False)[1]
            if "star" in used:
                area += lib[L.star].area_mm2_per_unit * hw["star_units"]
            if "vfu" in used:
                area += lib[L.vfu].area_mm2_per_unit * hw["vfu_units"]
            assert fe(q).area_mm2 == pytest.approx(area, rel=1e-12)
            assert fe(q).effort_months == pattern_effort(L, used, hw, tile, attn)


def test_pattern_options_exclude_edge_vfu(tiny):
    L = tiny[0]
    o = pattern_options(L, {"xbar", "vfu", "star"})
    assert o["sample"] == ["cpu"] and o["embed"] == ["cpu"]
    assert "vfu" in pattern_options(L, {"xbar", "vfu"}, exclude={})["sample"]
    assert "star" not in pattern_options(L, {"xbar"})["L0.softmax"]


def test_mix_endpoints(tiny):
    L, fe, hw, base, opts, allp = tiny
    a, b = allp[0][2], allp[-1][2]
    assert mix(L, a, b, 0) == b
    assert mix(L, a, b, L.model.layers) == a
