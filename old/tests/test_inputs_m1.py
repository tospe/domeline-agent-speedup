"""M1: the four inputs as files reproduce the hand-built case study exactly.

  workload   lib/workload/mini_coder_4b.yaml regenerates ops.forward_ops (every field)
             and its trace file equals experiments/common.sessions()
  library    every number of every tech has a source (or a VERIFY marker)
  host       lib/host/i9_12900k.yaml gives the same host op times as the TSV path
  all four   Inputs.load_case(...) builds an Evaluator whose Evals equal the hand-built
             one's, bit for bit: walk start, the whole walk, its result, random points
"""

import dataclasses
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import common  # noqa: E402
from harness import ops as O  # noqa: E402
from harness.design import Design  # noqa: E402
from harness.evaluate import Evaluator  # noqa: E402
from harness.inputs import HostProfile, Inputs, Objective  # noqa: E402
from harness.integrator import Integrator  # noqa: E402
from harness.model import BITNET_2B, ModelSpec  # noqa: E402
from harness.search import domeline_walk  # noqa: E402
from harness.space import KNOBS, START, Space  # noqa: E402
from harness.tech import load_library  # noqa: E402
from harness.workload import load_trace_tsv, load_workload  # noqa: E402

WORKLOAD = ROOT / "lib/workload/mini_coder_4b.yaml"
CASE = ROOT / "lib/case/mini_coder_swebench.yaml"
STATES = [(1, 1, None), (1, 777, None), (1, 40000, None), (256, 256, 1), (256, 9000, 1), (5, 17, 3)]


def _fields(op):
    return {f.name: getattr(op, f.name) for f in dataclasses.fields(op) if f.name != "precisions"}


@pytest.fixture(scope="module")
def wl():
    return load_workload(WORKLOAD)


# -- workload ----------------------------------------------------------------

def test_model_block_is_the_hf_config(wl):
    hf = ModelSpec.from_hf_config(common.MODEL_DIR)
    assert dataclasses.replace(hf, name=wl.spec.name) == wl.spec
    assert wl.kv_dim == hf.kv_dim and wl.layers == 36          # attribute delegation


@pytest.mark.parametrize("n,L,T", STATES)
def test_graph_equals_forward_ops(wl, n, L, T):
    ref = O.forward_ops(wl.spec, n, L, T)
    got = wl.forward_ops(n, L, T)
    assert [o.name for o in got] == [o.name for o in ref]
    for a, b in zip(ref, got):
        assert _fields(a) == _fields(b), a.name
        assert type(a.flops) is type(b.flops)
    assert O.ops_for(wl, n, L, T) == got and O.ops_for(wl.spec, n, L, T) == ref


@pytest.mark.parametrize("spec", [BITNET_2B, dataclasses.replace(BITNET_2B, kv_layers=6, name="hybrid")])
def test_graph_equals_forward_ops_other_models(wl, spec):
    """The `when:` branches (sub_norms, relu2, non-attention mixer layers) match too."""
    w = dataclasses.replace(wl, spec=spec)
    for n, L, T in STATES[:4]:
        assert [_fields(o) for o in w.forward_ops(n, L, T)] == \
            [_fields(o) for o in O.forward_ops(spec, n, L, T)]


def test_op_labels(wl):
    ops = {o.name: o for o in wl.forward_ops(1, 100)}
    assert ops["L0.softmax"].subcls == ops["L0.attn_norm"].subcls == ops["L0.act"].subcls \
        == O.REDUCTION_NONLINEAR
    assert ops["sample"].subcls == O.SAMPLING and ops["L0.rope"].subcls is None
    assert ops["L0.q_proj"].operands == [("w.q_proj", "static"), ("L0.h", "dynamic")]
    assert ops["L0.attn_score"].operands[0] == ("kv.K", "dynamic")
    assert ops["L0.q_proj"].precisions == (8, 16) and ops["L0.softmax"].precisions == (16,)
    # Operands written in the YAML equal the rule Op derives when none are given.
    for o in ops.values():
        assert o.operands == O.default_operands(o), o.name


def test_trace_file_equals_raw_traces(wl):
    assert wl.trace.sessions() == common.sessions()
    s, h = wl.trace.split()
    s0, h0 = common.split(common.sessions())
    assert [x.name for x in s] == [x.name for x in s0] and [x.name for x in h] == [x.name for x in h0]
    assert len(s) == 45 and len(s) + len(h) == 89


# -- library -----------------------------------------------------------------

def test_every_number_has_a_source():
    lib = load_library([ROOT / "lib/tech", ROOT / "lib/tech_dram"])
    for t in lib.values():
        nums = {"area_mm2_per_unit", "effort_months"} | {
            k for k, v in t.params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        assert nums <= set(t.sources), (t.name, sorted(nums - set(t.sources)))
        assert all(t.sources[k] for k in nums)
        assert isinstance(t.params.get("write_mechanism"), bool), t.name
    assert not lib["softmax_star"].write_mechanism and lib["xbar_isaac"].write_mechanism
    assert lib["xbar_isaac"].supports_precision(8) and not lib["softmax_star"].supports_precision(8)
    # Measured / checked numbers carry no VERIFY.
    assert lib["upmem_dpu"].unsourced() == ["bank_bw_Bps"]
    assert "t_read_s" not in lib["xbar_isaac"].unsourced()


# -- host --------------------------------------------------------------------

def test_host_file_gives_identical_host_times(wl):
    old = load_library(ROOT / "lib/tech/host.yaml")["cpu"]
    new = HostProfile.from_file(ROOT / "lib/host/i9_12900k.yaml").tech
    d = Design(mapping={"*": "cpu"})
    a, b = Integrator(wl.spec, {"cpu": old}), Integrator(wl, {"cpu": new})
    for n, L, T in STATES + [(1, 70000, None)]:
        ra, rb = a.step(d, n, L, T), b.step(d, n, L, T)
        assert ra.per_op == rb.per_op and ra.terms == rb.terms


def test_objective_file():
    o = Objective.from_file(ROOT / "lib/objective/latency.yaml")
    assert o.metric == "latency" and o.area_cap_mm2 == o.effort_cap_months == float("inf")
    assert o.gpu().wall == common.gpu().wall


# -- all four: bit for bit ----------------------------------------------------

@pytest.fixture(scope="module")
def pair():
    """(hand-built Evaluator, Evaluator from the case-study files), on the search set."""
    sess = common.sessions()
    train, test = common.split(sess)
    lib = common.library()
    space = Space(lib, common.model(), ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    hand = Evaluator(space, train, common.gpu())
    inp = Inputs.load_case(CASE)
    files = inp.evaluator()
    assert inp.space().ref_ctx == space.ref_ctx
    return hand, files


def test_from_files_bit_for_bit(pair):
    hand, files = pair
    assert files.gpu_time == hand.gpu_time and files.tools_time == hand.tools_time
    assert files(START) == hand(START)                                  # walk start
    ta, tb = domeline_walk(hand), domeline_walk(files)
    assert [e for e in files.log] == [e for e in hand.log]              # every walk evaluation
    assert [(s.reading, s.move) for s in tb.steps] == [(s.reading, s.move) for s in ta.steps]
    assert files.best() == hand.best() and tb.stop == ta.stop          # walk result
    rng = random.Random(7)
    for _ in range(4):                                                  # random points
        p = {k: rng.choice(v) for k, v in KNOBS.items()}
        assert files(p) == hand(p)
