"""H9 loop pieces (harness/team.py): supervisor hook on stall, its escalation record, the ensemble runner.
All with fake supervisors (no LLM)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tests"))

from common import gpu, library, model, sessions, split  # noqa: E402
from test_large_space import WALK_POINT  # noqa: E402

from harness.evaluate import Evaluator  # noqa: E402
from harness.large_space import FastEvaluator, LargeSpace  # noqa: E402
from harness.space import START, Space  # noqa: E402
from harness.team import (Move, SupervisorContext, ensemble, merge_records, owner_of, point_verifier,  # noqa: E402
                          rule_agents, team_search)
from harness.tools import RunRecord, Toolbox  # noqa: E402


@pytest.fixture(scope="module")
def env():
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    ref = max(s.max_ctx for s in train + test)
    return Space(lib, spec, ref_ctx=ref, host="cpu"), LargeSpace(lib, spec, ref_ctx=ref, host="cpu"), train, g


class FakeSupervisor:
    """Bills fixed tokens per call and returns replies[i] (a function of the context or a list of Moves)."""

    def __init__(self, replies, model="fake-sonnet", n_in=1000, n_out=100):
        self.name, self.record, self.model = "fake-supervisor", None, model
        self.replies, self.n_in, self.n_out = replies, n_in, n_out
        self.ctxs = []

    def restructure(self, ctx: SupervisorContext):
        self.ctxs.append((ctx.trigger, ctx.escalation, ctx.feedback))
        self.record.add_tokens(self.model, self.n_in, self.n_out, cache_read=7)
        r = self.replies(ctx) if callable(self.replies) else (
            self.replies[len(self.ctxs) - 1] if len(self.ctxs) <= len(self.replies) else [])
        return [Move(dict(m.changes), why=m.why) for m in r]


def test_supervisor_on_stall_is_called_logged_and_billed(env):
    """Large space, rule agents, stall_k=20: the supervisor (returning the loop's coordinated moves) is
    called on stall; each escalation is in the result and the record, with its own token delta."""
    _, L, train, g = env
    fe = FastEvaluator(L, train, g)
    rec = RunRecord("test", "team+fake-supervisor", 0, {})
    sup = FakeSupervisor(lambda ctx: ctx.coordinated)
    r = team_search(fe, Toolbox(fe), rule_agents(L), budget=800, stall_k=20, record=rec, supervisor=sup,
                    max_escalations=50)
    assert r.escalations and rec.escalations == r.escalations
    assert {x["trigger"] for x in r.escalations} <= {"stall", "exhausted"}
    assert any(x["trigger"] == "stall" for x in r.escalations)
    calls = sum(x["calls"] for x in r.escalations)
    assert calls == len(sup.ctxs)
    for x in r.escalations:
        assert x["tokens"] == {"fake-sonnet": {"in": 1000 * x["calls"], "out": 100 * x["calls"],
                                               "cache_read": 7 * x["calls"]}}
    assert rec.tokens["fake-sonnet"]["in"] == 1000 * calls
    assert [t["eval_n"] for t in rec.token_log] == sorted(t["eval_n"] for t in rec.token_log)
    assert any(x["accepted"] for x in r.escalations)
    assert any(a["agent"] == "supervisor" and a.get("term") == "supervisor" for a in r.accepted)
    assert r.final.speedup_vs_gpu > 9.0          # it restructures its way to the tied walk design
    j = rec.to_json(fe)
    assert j["escalations"] == r.escalations and j["cost"]["token_log"] == rec.token_log


def test_supervisor_cross_role_move_and_retry_with_feedback(env):
    """An illegal supervisor move costs no evaluation and comes back as feedback; the retry, a move
    spanning all three roles, is accepted without an ownership check."""
    S, _, train, g = env
    ev = Evaluator(S, train, g, budget=12)
    tb = Toolbox(ev, verifier=point_verifier(S, train))
    big = {k: v for k, v in WALK_POINT.items() if START.get(k) != v}
    assert len({owner_of(k) for k in big}) == 3
    sup = FakeSupervisor([[Move({"edge": "vfu"}, why="illegal on purpose")], [Move(big, why="the walk design")]])
    rec = RunRecord("test", "team+sup", 0, {})
    r = team_search(ev, tb, rule_agents(S), stall_k=1, stall_gap=0.0, record=rec, supervisor=sup,
                    max_escalations=1, shrink=False)
    x = r.escalations[0]
    assert x["calls"] == 2 and x["outcomes"] == ["illegal", "accepted"] and x["accepted"]
    assert sup.ctxs[0][2] is None and any(v[0] == "R1.sampling" for v in sup.ctxs[1][2])
    assert all(e.point["edge"] == "cpu" for e in ev.log)
    acc = [a for a in r.accepted if a.get("term") == "supervisor"]
    assert acc and acc[0]["move"] == big
    assert "ownership" not in r.rejected
    assert len(r.escalations) == 1                  # the cap


def test_escalation_cap_and_empty_supervisor(env):
    """A supervisor with no ideas: called at most max_escalations times; the run still ends normally."""
    S, _, train, g = env
    ev = Evaluator(S, train, g, budget=10)
    sup = FakeSupervisor([])
    r = team_search(ev, Toolbox(ev), rule_agents(S), stall_k=1, stall_gap=0.0, supervisor=sup,
                    max_escalations=2, record=RunRecord("t", "m", 0, {}), shrink=False)
    assert len(r.escalations) == 2 and all(x["calls"] == 1 and not x["accepted"] for x in r.escalations)
    assert r.stop == "budget exhausted"


def test_supervisor_none_keeps_the_m5_loop(env):
    """Same evaluations with supervisor=None as without the argument (the M5 path)."""
    _, L, train, g = env
    logs = []
    for kw in ({}, {"supervisor": None}):
        fe = FastEvaluator(L, train, g)
        team_search(fe, Toolbox(fe), rule_agents(L), budget=300, stall_k=20, **kw)
        logs.append([L.key(e.point) for e in fe.log])
    assert logs[0] == logs[1]


def _member(seed):
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    L = LargeSpace(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    fe = FastEvaluator(L, train, g)
    rec = RunRecord("test", "rule-team", seed, {"budget": 20 + 30 * seed})
    sup = FakeSupervisor(lambda ctx: ctx.coordinated, model="fake-haiku", n_in=10 + seed, n_out=1)
    team_search(fe, Toolbox(fe), rule_agents(L), budget=20 + 30 * seed, stall_k=10, record=rec,
                supervisor=sup)
    out = rec.to_json(fe)
    out["cost_usd"] = {"total": 0.5 * (seed + 1)}
    return out


def test_ensemble_best_of_and_merged_record():
    seeds = [0, 1, 2]
    m = ensemble(_member, seeds, workers=3)
    members = m["members"]
    assert [x["seed"] for x in members] == seeds
    t = [x["final"]["time_s"] for x in members]
    assert m["best_member"] == t.index(min(t)) and m["final"] == members[m["best_member"]]["final"]
    n = [len(x["evaluations"]) for x in members]
    assert m["cost"]["evaluations"] == sum(x["cost"]["evaluations"] for x in members)
    assert [e["n"] for e in m["evaluations"]] == list(range(1, sum(n) + 1))
    assert [e["member"] for e in m["evaluations"][:3]] == [0, 1, 2]           # round-robin
    tok = sum(x["cost"]["tokens"].get("fake-haiku", {}).get("in", 0) for x in members)
    assert m["cost"]["tokens"].get("fake-haiku", {}).get("in", 0) == tok
    assert m["cost_usd"] == {"total": 3.0}
    assert m["cost"]["wall_s"] == max(x["cost"]["wall_s"] for x in members)
    # token_log eval_n is in merged evaluations; each maps back to the member's own count.
    for tl in m["cost"]["token_log"]:
        if tl["eval_n"]:
            e = m["evaluations"][tl["eval_n"] - 1]
            assert e["member"] == tl["member"] and e["member_n"] <= tl["member_eval_n"]
    assert m["method"] == "ensemble[rule-team]"


def test_merge_records_skips_infeasible_finals():
    a = {"final": {"time_s": 1.0, "feasible": False}, "evaluations": [{"n": 1}], "cost": {}}
    b = {"final": {"time_s": 5.0, "feasible": True}, "evaluations": [{"n": 1}, {"n": 2}], "cost": {}}
    m = merge_records([a, b])
    assert m["best_member"] == 1 and [e["member"] for e in m["evaluations"]] == [0, 1, 1]
