"""WorkloadReplayer (shared curves, chunked prefill) agrees with exact per-turn replay."""

import pytest

from harness.design import Design
from harness.integrator import Integrator
from harness.model import ModelSpec
from harness.session import Replayer, WorkloadReplayer
from harness.space import START, Space
from harness.synthetic import synthetic_sessions
from harness.tech import load_library

MODEL = "/localhome/tsousape/models/mini-coder-4b"


@pytest.mark.parametrize("change", [{}, {"attention": "xbar"}, {"attention": "vfu", "softmax": "vfu", "vector": "vfu"},
                                    {"attention": "xbar", "softmax": "star", "vector": "star", "lm_head": "xbar",
                                     "xbar_tech": "xbar_transpose", "link": "die"}])
def test_workload_replay_matches_exact(change):
    spec = ModelSpec.from_hf_config(MODEL)
    lib = load_library("lib/tech")
    sess = synthetic_sessions(3, seed=4)
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in sess), host="cpu")
    design, lib2 = space.to_design({**START, **change})
    integ = Integrator(spec, lib2)
    fast = WorkloadReplayer(integ).run(design, sess)
    for s, f in zip(sess, fast):
        exact = Replayer(integ).run(design, s)
        assert f.llm == pytest.approx(exact.llm, rel=0.02)
