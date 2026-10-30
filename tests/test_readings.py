"""D (data held in PIM), term headroom and capacity headroom; the four-input interface."""

import math
import random
from pathlib import Path

import pytest

from harness.evaluate import Evaluator, binding_and_next
from harness.inputs import HostProfile, Inputs, run
from harness.integrator import Integrator, _array_bytes
from harness.model import ModelSpec
from harness.search import domeline_walk
from harness.session import GpuBaseline
from harness.space import KNOBS, START, Space
from harness.synthetic import synthetic_sessions
from harness.tech import load_library

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/localhome/tsousape/models/mini-coder-4b"


@pytest.fixture(scope="module")
def setup():
    spec = ModelSpec.from_hf_config(MODEL)
    lib = load_library(ROOT / "lib/tech")
    sess = synthetic_sessions(3, seed=4)
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in sess), host="cpu")
    gpu = GpuBaseline(ROOT / "results/gpu/g1.jsonl", weights="Q8_0", kv="f16")
    return spec, lib, sess, space, gpu


def test_D_matches_capacity(setup):
    """Over capacity in D (occupied > capacity) exactly when Integrator.capacity says so."""
    spec, lib, sess, space, _ = setup
    rng = random.Random(1)
    for _ in range(80):
        p = {k: rng.choice(v) for k, v in KNOBS.items()}
        d, l = space.to_design(p)
        integ = Integrator(spec, l)
        for ctx in (1, 2048, space.ref_ctx, 3 * space.ref_ctx):
            cap = integ.capacity(d, ctx)
            res = integ.data_in_pim(d, ctx)
            assert res.over == integ.capacity_over(d, ctx)
            for k, c in cap.items():
                name, _, unit = k.partition(":")
                scale = 1 if unit == "bytes" else _array_bytes(l[name])
                assert res.occupied[name] == pytest.approx(c["needed"] * scale)
                assert res.capacity[name] == pytest.approx(c["have"] * scale)
            assert set(res.occupied) == {k.partition(":")[0] for k in cap}


def test_D_tensors(setup):
    """Weights at weight_bits, LM head only if offloaded, KV at kv_bits growing with context."""
    spec, lib, _, space, _ = setup
    d, l = space.to_design(START)                         # weight MVMs on crossbars, rest on host
    integ = Integrator(spec, l)
    w = spec.weight_params(include_lm_head=False) * d.weight_bits / 8
    for ctx in (1, 10000):
        r = integ.data_in_pim(d, ctx)
        assert r.D == pytest.approx(w)
        assert "lm_head" not in r.by_tensor() and "kv.K" not in r.by_tensor()

    for att, kv in (("xbar", 16), ("vfu", 8)):
        d, l = space.to_design({**START, "lm_head": "xbar", "attention": att, "kv_bits": kv})
        integ = Integrator(spec, l)
        r1, r2 = integ.data_in_pim(d, 1000), integ.data_in_pim(d, 5000)
        assert r1.by_tensor()["lm_head"] == spec.vocab * spec.hidden * d.weight_bits / 8
        assert r2.D - r1.D == pytest.approx(spec.kv_bytes_per_token(kv) * 4000)
        t = r2.by_tensor()
        assert t["kv.K"] == t["kv.V"] == pytest.approx(spec.kv_bytes_per_token(kv) * 5000 / 2)


def test_D_wall_at_fit(setup):
    """xbar_scale 1.0 sizes the arrays to fit at ref_ctx: headroom >= 0 there, < 0 well past it."""
    spec, lib, _, space, _ = setup
    d, l = space.to_design({**START, "attention": "xbar", "xbar_scale": 1.0})
    integ = Integrator(spec, l)
    tech, room = integ.data_in_pim(d, space.ref_ctx).binding_headroom()
    assert tech == START["xbar_tech"] and room >= 0
    assert integ.data_in_pim(d, 2 * space.ref_ctx).binding_headroom()[1] < 0


def test_binding_and_next():
    assert binding_and_next({"host": 6.0, "transfer": 2.0, "launch": 3.0}) == ("host", "launch", 2.0)
    assert binding_and_next({"host": 1.0, "write": 0.0}) == ("host", None, math.inf)
    assert binding_and_next({}) == (None, None, math.inf)


@pytest.mark.parametrize("change", [{}, {"attention": "xbar", "xbar_scale": 0.75},
                                    {"attention": "vfu", "vfu_mem": 4e9, "kv_bits": 16},
                                    {"attention": "xbar", "softmax": "star", "vector": "star"}])
def test_eval_readings(setup, change):
    spec, lib, sess, space, gpu = setup
    ev = Evaluator(space, sess, gpu)
    e = ev({**START, **change})
    b1, b2, ratio = binding_and_next(e.terms)
    assert (e.binding_term, e.second_term) == (b1, b2)
    assert e.term_headroom == ratio >= 1.0
    assert e.max_ctx == max(s.max_ctx for s in sess)
    # Capacity headroom is negative exactly when the wall falls inside the workload.
    assert (e.capacity_headroom_bytes < 0) == (e.wall_ctx is not None)
    assert e.D_bytes == pytest.approx(sum(e.D_by_tensor.values()))
    assert e.D_bytes <= e.D_wall_bytes or e.wall_ctx is not None


def test_inputs_match_hand_built(setup):
    """The four-input interface gives the same evaluations as the hand-built evaluator."""
    spec, lib, sess, space, gpu = setup
    inp = Inputs(model=spec, sessions=sess, library=load_library(ROOT / "lib/tech/pim.yaml"),
                 host=HostProfile.load())
    a = run(inp, domeline_walk, gpu, budget=25)
    ev = Evaluator(space, sess, gpu, budget=25)
    domeline_walk(ev)
    assert [(e.point, e.time_s, e.feasible) for e in a.log] == \
        [(e.point, e.time_s, e.feasible) for e in ev.log]
    assert a.best.point == ev.best().point and a.design is not None
    with pytest.raises(ValueError):
        Inputs(model=spec, sessions=sess, library={}, host=HostProfile.load(), objective="energy")


def test_host_tool_scale(setup):
    spec, _, sess, _, gpu = setup
    h = HostProfile.load(tool_scale=0.5)
    inp = Inputs(model=spec, sessions=sess, library=load_library(ROOT / "lib/tech/pim.yaml"), host=h)
    ev = inp.evaluator(gpu)
    assert ev.tools_time == pytest.approx(0.5 * sum(s.tool_time for s in sess))
