"""M4 tool API on synthetic sessions: same numbers as the Evaluator, moves are the walk's rule table."""

import json
from pathlib import Path

import pytest

from harness.evaluate import Evaluator
from harness.model import ModelSpec
from harness.search import domeline_walk, proposals
from harness.session import GpuBaseline
from harness.space import START, Space
from harness.synthetic import synthetic_sessions
from harness.tech import load_library
from harness.tools import RunRecord, Toolbox

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/localhome/tsousape/models/mini-coder-4b"


@pytest.fixture(scope="module")
def ev():
    spec = ModelSpec.from_hf_config(MODEL)
    sess = synthetic_sessions(3, seed=4)
    space = Space(load_library(ROOT / "lib/tech"), spec, ref_ctx=max(s.max_ctx for s in sess), host="cpu")
    return Evaluator(space, sess, GpuBaseline(ROOT / "results/gpu/g1.jsonl", weights="Q8_0", kv="f16"))


def test_evaluate_and_read(ev):
    tb = Toolbox(ev)
    e = ev(START)
    out = tb.evaluate(START)
    assert out["time_s"] == e.time_s and out["terms"] == e.terms
    r = tb.read(START)
    assert r["binding_term"] == e.binding_term == r["ranking"][0]["term"]
    assert r["amdahl_gap"] == pytest.approx(e.time_s / ev.tools_time - 1)
    assert r["ranking"][0]["headroom"] == pytest.approx(e.term_headroom)


def test_moves_are_the_rule_table(ev):
    tb = Toolbox(ev)
    assert [m["point"] for m in tb.moves(START)] == [q for _, q in proposals(ev(START))]
    host = tb.moves(START, term="host")
    assert host and all("host binds" in m["why"] for m in host)


def test_record_round_trips(ev, tmp_path):
    rec = RunRecord("test", "walk", 0, {"budget": 50})
    tr = domeline_walk(ev)
    for s in tr.steps:
        rec.accept(s.move, s.reading, s.eval)
    rec.add_tokens("none", 0, 0)
    rec.write(tmp_path / "r.json", ev, Toolbox(ev))
    d = json.loads((tmp_path / "r.json").read_text())
    assert d["cost"]["evaluations"] == ev.used and len(d["accepted"]) == len(tr.steps)
