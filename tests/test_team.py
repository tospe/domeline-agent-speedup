"""M5 loop (harness/team.py): ownership table, routing, deterministic mode == the walk."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from common import gpu, library, model, sessions, split  # noqa: E402

from harness.evaluate import Evaluator  # noqa: E402
from harness.large_space import FastEvaluator, LargeSpace  # noqa: E402
from harness.search import domeline_walk  # noqa: E402
from harness.space import KNOBS, START, Space  # noqa: E402
from harness.team import (OWNER_KNOBS, TERM_OWNERS, Move, RuleAgent, owner_of, rule_agents, rule_table,  # noqa: E402
                          team_search, term_class, why_term)
from harness.tools import RunRecord, Toolbox  # noqa: E402


@pytest.fixture(scope="module")
def env():
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    ref = max(s.max_ctx for s in train + test)
    return Space(lib, spec, ref_ctx=ref, host="cpu"), LargeSpace(lib, spec, ref_ctx=ref, host="cpu"), train, g


def test_every_knob_has_one_owner():
    owned = [k for ks in OWNER_KNOBS.values() for k in ks]
    assert sorted(owned) == sorted(KNOBS) and len(owned) == len(set(owned))


def test_term_owner_table_covers_the_rule_table(env):
    """Every move the rule table makes for a term touches only knobs of that term's owners."""
    S, L, train, g = env
    ev = Evaluator(S, train, g)
    domeline_walk(ev)                        # a varied set of designs (start, walk, shrink, repairs)
    fe = FastEvaluator(L, train, g)
    pts = [(ev, e) for e in ev.log] + [(fe, fe(L.tie(e.point))) for e in ev.log[:12]]
    seen = set()
    for E, e in pts:
        for why, ch in rule_table(e, E.space):
            term = why_term(why)
            for k in ch:
                assert owner_of(k, E.space) in TERM_OWNERS[term_class(term)], (why, k)
                seen.add((term_class(term), owner_of(k, E.space)))
    assert ("host", "mapping") in seen and ("write", "hardware") in seen


def test_deterministic_mode_reproduces_the_walk(env):
    """Rule agents split by owner, routed by the binding term: the walk's evaluations, moves and design."""
    S, _, train, g = env
    ew = Evaluator(S, train, g)
    trail = domeline_walk(ew)
    et = Evaluator(S, train, g)
    rec = RunRecord("test", "team-deterministic", 0, {})
    r = team_search(et, Toolbox(et), rule_agents(S), record=rec)
    assert [S.key(e.point) for e in et.log] == [S.key(e.point) for e in ew.log]
    assert et.used == ew.used == 31
    assert [s.eval.n for s in trail.steps] == [a["eval_n"] for a in r.accepted]
    assert r.final.point == trail.steps[-1].eval.point
    assert r.stop == "budget exhausted" or trail.stop.startswith("no move")
    assert len(rec.accepted) == len(trail.steps) - 1


class Rogue(RuleAgent):
    """A mapping agent that also tries to change the link (a hardware knob)."""

    def propose(self, ctx):
        return [Move({"link": "die"}, why="not mine", priority=-1)] + super().propose(ctx)


def test_ownership_is_enforced(env):
    S, _, train, g = env
    ev = Evaluator(S, train, g, budget=6)
    agents = {**rule_agents(S), "mapping": Rogue("mapping")}
    r = team_search(ev, Toolbox(ev), agents, shrink=False)
    assert r.rejected.get("ownership", 0) >= 1
    assert all(e.point["link"] == "cxl" for e in ev.log)


def test_verifier_rejects_without_evaluating(env):
    """Illegal moves (sample on the vector unit) are not evaluated; the violated rule is logged."""
    S, _, train, g = env
    from harness.team import point_verifier
    ev = Evaluator(S, train, g)
    tb = Toolbox(ev, verifier=point_verifier(S, train))
    r = team_search(ev, tb, rule_agents(S), shrink=False)
    assert all(e.point["edge"] == "cpu" for e in ev.log)
    illegal = [x for x in tb.entries if x.get("outcome") == "illegal"]
    assert illegal and any(v[0] == "R1.sampling" for x in illegal for v in x["violations"])


def test_exclude_option(env):
    S, _, train, g = env
    ev = Evaluator(S, train, g, budget=40)
    team_search(ev, Toolbox(ev), rule_agents(S), exclude={"edge": {"vfu"}})
    assert all(e.point["edge"] == "cpu" for e in ev.log)


def test_large_deterministic_runs(env):
    """On the large space the team makes per-slot moves and coordinated ones; stays within budget."""
    _, L, train, g = env
    fe = FastEvaluator(L, train, g)
    r = team_search(fe, Toolbox(fe), rule_agents(L), budget=400, coordinate="first")
    assert fe.used <= 400
    assert r.final.speedup_vs_gpu > 1.0
    assert any(a.get("agent") == "supervisor" for a in r.accepted)
