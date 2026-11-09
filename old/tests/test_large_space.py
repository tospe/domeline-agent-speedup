"""M3: the large space (harness/large_space.py).

- tied large points time exactly as their small-space points (real search sessions);
- FastEvaluator gives the Evaluator's numbers on untied points, walls included;
- a lone offload costs two link crossings;
- the mapping DP equals brute force on a 2-layer instance;
- the certificate is below every design's time.
"""

import itertools
import math
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from common import gpu, library, model, sessions, split  # noqa: E402

from harness.evaluate import Evaluator  # noqa: E402
from harness.integrator import Integrator  # noqa: E402
from harness.large_space import FastEvaluator, LargeSpace, certificate, mapping_dp  # noqa: E402
from harness.space import KNOBS, START, Space  # noqa: E402
from harness.synthetic import synthetic_sessions  # noqa: E402

WALK_POINT = {"mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
              "softmax": "star", "vector": "star", "edge": "cpu", "xbar_scale": 1.0, "vfu_units": 1,
              "vfu_bw": 64e9, "vfu_mem": 4e9, "star_units": 64, "link": "cxl", "kv_bits": 8}
OPTIMUM = {**WALK_POINT, "edge": "vfu", "xbar_scale": 1.25, "vfu_units": 16, "link": "die", "kv_bits": 16}
# Past a wall: half the arrays; KV on a 4 GB vector-unit memory.
WALLS = [{**WALK_POINT, "xbar_scale": 0.5}, {**START, "attention": "vfu", "vfu_mem": 4e9}]

READINGS = ("time_s", "llm_s", "feasible", "wall_ctx", "area_mm2", "effort_months", "binding_term",
            "second_term", "term_headroom", "D_bytes", "D_wall_bytes", "capacity_headroom_bytes",
            "capacity_headroom_tech", "speedup_vs_gpu")


@pytest.fixture(scope="module")
def env():
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    ref = max(s.max_ctx for s in train + test)
    return Space(lib, spec, ref_ctx=ref, host="cpu"), LargeSpace(lib, spec, ref_ctx=ref, host="cpu"), train, g


def _same(a, b, rel=0.0):
    for f in READINGS:
        x, y = getattr(a, f), getattr(b, f)
        if rel and isinstance(x, float) and math.isfinite(x):
            assert y == pytest.approx(x, rel=rel, abs=1e-12), f
        else:
            assert x == y, f
    for d in ("terms", "by_kind", "D_by_tensor"):
        x, y = getattr(a, d), getattr(b, d)
        for k in set(x) | set(y):
            assert y.get(k, 0.0) == pytest.approx(x.get(k, 0.0), rel=rel or 1e-15, abs=1e-12), (d, k)


def test_slots(env):
    _, L, _, _ = env
    assert L.model.layers == 36
    assert len(L.slots) == 36 * 17 + 4          # 15 placements + attention + kv_bits per layer
    assert L.is_tied(L.start) and L.untie(L.start) == START


def test_tied_equals_small(env):
    """Small space = the large space with every layer tied: identical Evals (exact path), fast path to 1e-9."""
    S, L, train, g = env
    rng = random.Random(0)
    pts = [START, WALK_POINT, OPTIMUM] + WALLS + [{k: rng.choice(v) for k, v in KNOBS.items()} for _ in range(3)]
    ev, ex, fe = Evaluator(S, train, g), Evaluator(L, train, g), FastEvaluator(L, train, g)
    for p in pts:
        a = ev(p)
        _same(a, ex(L.tie(p)))
        _same(a, fe(L.tie(p)), rel=1e-9)
    assert not ev(WALLS[0]).feasible and not ev(WALLS[1]).feasible


def _random_large(L, rng, base):
    p = L.tie(base)
    for s in L.slots.values():
        if rng.random() < 0.3:
            p[s.name] = rng.choice(s.options)
    return p


def test_fast_matches_exact_untied(env):
    _, L, train, g = env
    rng = random.Random(1)
    ex, fe = Evaluator(L, train, g), FastEvaluator(L, train, g)
    pts = [_random_large(L, rng, b) for b in (START, WALK_POINT, OPTIMUM, WALLS[0], WALLS[1])]
    for p in pts:
        assert not L.is_tied(p)
        _same(ex(p), fe(p), rel=1e-9)
    assert any(not ex(p).feasible for p in pts)


def test_lone_offload_crosses_twice(env):
    """One op on the vector unit between host ops: a transfer in, one back per op that reads its
    output (act -> down_proj: 2; rope -> kv_append and attn_score: 3), and one launch + one sync."""
    _, L, _, _ = env
    base = L.tie({**START, "mvm": "cpu"})
    d0, lib0 = L.to_design(base)
    a = Integrator(L.model, lib0).step(d0, 1, 4096)
    assert a.crossings == 0
    for op, n in (("L3.act", 2), ("L3.rope", 3)):
        d1, lib1 = L.to_design({**base, op: "vfu"})
        b = Integrator(L.model, lib1).step(d1, 1, 4096)
        assert b.crossings == n
        assert b.term("launch") - a.term("launch") == pytest.approx(d1.link.launch_s + d1.link.sync_s)


# -- DP against brute force on a 2-layer instance ----------------------------------

@pytest.fixture(scope="module")
def tiny():
    spec = replace(model(), layers=2)
    lib, g = library(), gpu()
    sess = synthetic_sessions(2, seed=3, turns=(3, 5))
    L = LargeSpace(lib, spec, ref_ctx=max(s.max_ctx for s in sess), host="cpu")
    return L, FastEvaluator(L, sess, g), Evaluator(L, sess, g)


FREE = ["L0.attention", "L0.kv_bits", "L0.softmax", "L0.rope", "L0.o_proj", "L1.q_proj", "L1.k_proj",
        "L1.act", "L1.mlp_residual", "lm_head", "sample"]


@pytest.mark.parametrize("hw", [{**WALK_POINT, "vector": "cpu", "softmax": "cpu", "attention": "cpu", "xbar_scale": 1.25},
                                {**START, "vfu_units": 16, "star_units": 4, "link": "package",
                                 "vfu_mem": 64e9, "vfu_bw": 256e9}])
def test_dp_equals_brute_force(tiny, hw):
    L, fe, ex = tiny
    base = L.tie(hw)
    opts = {s: [base[s]] for s in L.slots}
    for s in FREE:
        opts[s] = list(L.slots[s].options)
    best = None
    for vals in itertools.product(*(opts[s] for s in FREE)):
        e = fe({**base, **dict(zip(FREE, vals))})
        if e.wall_ctx is None and (best is None or e.llm_s < best.llm_s):
            best = e
    dp = mapping_dp(fe, base, opts)
    assert dp.llm_s == pytest.approx(best.llm_s, rel=1e-12)
    got = fe(dp.point)
    assert got.llm_s == pytest.approx(dp.llm_s, rel=1e-12)
    assert ex(dp.point).llm_s == pytest.approx(dp.llm_s, rel=1e-9)


def test_dp_free_tiny_beats_tied(tiny):
    """With every slot free the DP is at least as good as every tied mapping on that hardware."""
    L, fe, _ = tiny
    hw = {**START, "vfu_units": 16, "star_units": 16, "vfu_mem": 64e9}
    dp = mapping_dp(fe, L.tie(hw))
    tied = []
    for m in itertools.product(*(KNOBS[k] for k in ("mvm", "lm_head", "attention", "softmax", "vector", "edge"))):
        e = fe(L.tie({**hw, **dict(zip(("mvm", "lm_head", "attention", "softmax", "vector", "edge"), m))}))
        if e.wall_ctx is None:
            tied.append(e.llm_s)
    assert dp.llm_s <= min(tied) * (1 + 1e-12)
    assert fe(dp.point).llm_s == pytest.approx(dp.llm_s, rel=1e-12)


def test_certificate_is_a_floor(tiny):
    L, fe, _ = tiny
    floor, parts = certificate(fe)
    rng = random.Random(5)
    for _ in range(40):
        b = {k: rng.choice(v) for k, v in KNOBS.items()}
        p = _random_large(L, rng, b)
        assert fe(p).time_s >= floor * (1 - 1e-12)
    dp = mapping_dp(fe, L.tie({**OPTIMUM}))
    assert dp.llm_s + fe.tools_time >= floor
