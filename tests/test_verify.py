"""The offload verifier (harness/verify.py, brief M2).

Illegal designs must fail with the right rule; the walk's headline design must pass;
and on a live random sample of the 14-knob space, every design the Evaluator calls
feasible must be verifier-legal unless it hits the one known disagreement
(edge: vfu puts `sample` on the vector unit; docs/harness-dev/REPORT-M2.md).
"""

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
from harness.session import Session, Turn  # noqa: E402
from harness.space import KNOBS, START, Space  # noqa: E402
from harness.verify import Caps, verify, verify_point  # noqa: E402

# The walk's result (results/search/main.json, walk seed 0): 9.42x, 9,729 mm2, 7 months.
HEADLINE = {"mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
            "softmax": "star", "vector": "star", "edge": "cpu", "xbar_scale": 1.0, "vfu_units": 1,
            "vfu_bw": 64e9, "vfu_mem": 4e9, "star_units": 64, "link": "cxl", "kv_bits": 8}
PINNED = {"xbar_tile": "isaac", "adc_per_ima": 8, "cell_bits": 2}   # the paper's 14-knob space
KNOWN = {("R1.sampling", "R2.transfer")}     # edge: vfu (sample and embed on the vector unit)


@pytest.fixture(scope="module")
def env():
    train, test = split(sessions())
    spec, lib = model(), library()
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    return space, train


def headline(space):
    return space.to_design(space.canonical(HEADLINE))


def test_headline_is_legal(env):
    space, train = env
    r = verify_point(space, HEADLINE, train, Caps(min_ctx=65536))
    assert r.legal, r.violations
    assert r.checked["area_mm2"] == pytest.approx(9728.696, abs=1e-3)
    assert r.checked["effort_months"] == 7


def test_headline_kv_write_is_in_write_term(env):
    """Attention on the crossbar implies a KV write every token; it must be the integrator's write term."""
    space, train = env
    d, lib = headline(space)
    r = verify(d, lib, space.model, train)
    kv = [c for c in r.implied_costs if c.kind == "kv_write"]
    assert len(kv) == space.model.n_kv_layers and all(c.charged and c.term == "write" for c in kv)
    ctx = r.checked["max_state"]["ctx"]
    step = Integrator(space.model, lib).step(d, 1, ctx)
    assert step.term("write") == pytest.approx(sum(c.seconds for c in kv), rel=1e-12)


def test_softmax_on_crossbar(env):
    space, train = env
    d, lib = headline(space)
    r = verify(d.remap(softmax="xbar_transpose"), lib, space.model, train)
    assert not r.legal and "R1.support" in r.rules()
    assert all(v.op.endswith("softmax") for v in r.by_rule("R1.support"))


def test_sampling_on_pim(env):
    space, train = env
    d, lib = headline(space)
    r = verify(d.remap(sample="vfu_puma"), lib, space.model, train)
    assert not r.legal and [v.op for v in r.by_rule("R1.sampling")] == ["sample"]
    assert any(v.op == "sample" for v in r.by_rule("R2.transfer"))   # the token never comes back


@pytest.mark.parametrize("how", ["flag", "no_write_param"])
def test_attention_on_crossbar_without_write_mechanism(env, how):
    space, train = env
    d, lib = headline(space)
    x = lib["xbar_transpose"]
    params = dict(x.params)
    if how == "flag":
        params["write_mechanism"] = False
    else:
        params.pop("t_write_row_s")
    lib = {**lib, x.name: replace(x, params=params)}
    r = verify(d, lib, space.model, train)
    assert not r.legal and r.rules() == {"R1.write_mechanism"}
    kinds = {v.op.split(".")[-1] for v in r.violations}
    assert kinds == {"attn_score", "attn_value", "kv_append"}


def test_kv_over_capacity_at_64k(env):
    """Attention on the vector unit with 4 GB next to it: fp16 KV at 64K tokens is 9.7 GB."""
    space, _ = env
    long = [Session("synthetic-64k", [Turn(ctx_start=60000, prefill_tokens=60000, gen_tokens=5536,
                                           t_tool_s=1.0)])]
    p = {**HEADLINE, "attention": "vfu", "vfu_mem": 4e9, "kv_bits": 16}
    r = verify_point(space, p, long)
    assert not r.legal and r.rules() == {"R3.capacity"}
    (v,) = r.violations
    assert v.op == "vfu_puma" and "synthetic-64k" in v.reason
    assert r.checked["max_state"]["ctx"] == 65536
    # The same design with a short trace fits; the 64K requirement can also come from the caps.
    short = [Session("short", [Turn(ctx_start=4000, prefill_tokens=4000, gen_tokens=96, t_tool_s=1.0)])]
    assert verify_point(space, p, short).legal
    assert verify_point(space, p, short, Caps(min_ctx=65536)).rules() == {"R3.capacity"}


def test_crossbar_kv_over_capacity_at_64k(env):
    """The headline design with its crossbar at 0.75x of what 64K needs (Fig. 1's E2 point)."""
    space, train = env
    r = verify_point(space, {**HEADLINE, "xbar_scale": 0.75}, train)
    assert r.rules() == {"R3.capacity"}
    assert any(c.kind == "reprogram" and c.charged for c in r.implied_costs)


def test_capacity_need_is_monotone_in_ctx(env):
    """R3 checks the largest state only; that is sound because need never falls as ctx grows."""
    space, _ = env
    for p in (HEADLINE, {**HEADLINE, "attention": "vfu"}, START):
        d, lib = space.to_design(space.canonical(p))
        integ = Integrator(space.model, lib)
        prev = None
        for ctx in (1, 100, 1000, 4097, 20000, 40000, 65536):
            need = {k: c["needed"] for k, c in integ.capacity(d, ctx).items()}
            if prev:
                assert all(need[k] >= prev[k] for k in prev)
            prev = need


def test_caps(env):
    space, train = env
    r = verify_point(space, HEADLINE, train, Caps(area_mm2=5000, effort_months=4))
    assert r.rules() == {"R4.area", "R4.effort"}


def test_kv_split_across_units(env):
    space, train = env
    d, lib = headline(space)
    r = verify(d.remap(attn_value="vfu_puma"), lib, space.model, train)
    assert "R2.dataflow" in r.rules()


def test_feasible_sample_is_legal(env):
    """A live seeded sample of the 14-knob space: Evaluator-feasible => verifier-legal, except KNOWN."""
    space, train = env
    ev = Evaluator(space, train, gpu())
    rng = random.Random(0)
    seen, disagree, n_feasible = set(), [], 0
    while len(seen) < 40:
        p = space.canonical({**{k: rng.choice(v) for k, v in KNOBS.items()}, **PINNED})
        k = space.key(p)
        if k in seen:
            continue
        seen.add(k)
        e = ev(p)
        r = verify_point(space, p, train)
        if r.legal:
            assert e.feasible, (p, e.reason)          # never legal but infeasible
        if e.feasible:
            n_feasible += 1
            if not r.legal:
                disagree.append((p, tuple(sorted(r.rules()))))
        if not e.feasible and e.wall_ctx is not None:
            assert "R3.capacity" in r.rules()
    assert n_feasible > 5
    unknown = [d for d in disagree if d[1] not in KNOWN]
    assert not unknown, unknown
    assert all(p["edge"] == "vfu" for p, _ in disagree)
