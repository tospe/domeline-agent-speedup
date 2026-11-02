"""M6: baselines (harness/baselines.py) and the H runner (experiments/hrun.py).

- the four baselines are seeded (same seed -> same evaluation sequence), keep to the budget,
  and never evaluate a design the verifier rejects (verifier on), on the small and large space;
- the verifier's sampling rule is an option (REPORT-M2 §3.2);
- the shadow proxy verifies without blocking and forwards the Evaluator's state;
- the runner reproduces the walk (31 evals, 1% band at 20) and the deterministic team with the
  verifier (24, 15; REPORT-M5-loop), writes one record per (method, seed) and summary.csv, and
  resolves methods by registry name or module:function.
"""

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import hrun  # noqa: E402
from common import gpu, library, model, sessions, split  # noqa: E402
from harness import baselines as B  # noqa: E402
from harness.large_space import FastEvaluator, LargeSpace  # noqa: E402
from harness.space import KNOBS, Space  # noqa: E402
from harness.verify import verify_point  # noqa: E402

HEADLINE = {"mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
            "softmax": "star", "vector": "star", "edge": "cpu", "xbar_scale": 1.0, "vfu_units": 1,
            "vfu_bw": 64e9, "vfu_mem": 4e9, "star_units": 64, "link": "cxl", "kv_bits": 8}


@pytest.fixture(scope="module")
def env():
    train, test = split(sessions())
    spec, lib, g = model(), library(), gpu()
    ref = max(s.max_ctx for s in train + test)
    return Space(lib, spec, ref_ctx=ref, host="cpu"), LargeSpace(lib, spec, ref_ctx=ref, host="cpu"), train, g


def test_knob_table(env):
    S, L, _, _ = env
    t = B.knob_table(S)
    assert t == {k: list(v) for k, v in KNOBS.items()}
    t14 = B.knob_table(S, pin=B.PINNED_14, exclude={"edge": ["vfu"]})
    assert t14["xbar_tile"] == ["isaac"] and t14["edge"] == ["cpu"]
    tl = B.knob_table(L, exclude={"sample": ["vfu"]})
    assert len(tl) == 626 and tl["sample"] == ["cpu"] and tl["L3.q_proj"] == list(KNOBS["mvm"])
    with pytest.raises(ValueError):
        B.knob_table(S, exclude={"edge": ["cpu", "vfu"]})


def test_sampling_rule_is_an_option(env):
    S, _, train, _ = env
    p = {**HEADLINE, "edge": "vfu"}
    assert verify_point(S, p, train).rules() == {"R1.sampling", "R2.transfer"}
    r = verify_point(S, p, train, sampling="any")
    assert r.legal and {c.kind for c in r.implied_costs if not c.charged} == {"token_io"}
    with pytest.raises(ValueError):
        verify_point(S, p, train, sampling="maybe")


def _run(space, train, g, fn, seed, budget, verify=True, **kw):
    ev = FastEvaluator(space, train, g, budget=budget)
    ver = B.make_verifier(space, train) if verify else None
    gate = B.Gate(ev, ver)
    table = B.knob_table(space)
    if fn in (B.anneal, B.hill_climb):
        fn(gate, table, B.start_point(space, table), seed=seed, **kw)
    else:
        fn(gate, table, seed=seed, **kw)
    return ev, gate, ver


@pytest.mark.parametrize("fn", [B.random_search, B.tpe_search, B.anneal, B.hill_climb])
@pytest.mark.parametrize("large", [False, True])
def test_baseline_seeded_budget_and_gate(env, fn, large):
    S, L, train, g = env
    space = L if large else S
    budget = 25
    ev1, gate1, ver = _run(space, train, g, fn, 0, budget)
    ev2, _, _ = _run(space, train, g, fn, 0, budget)
    assert [space.key(e.point) for e in ev1.log] == [space.key(e.point) for e in ev2.log]   # seeded
    assert ev1.used <= budget
    assert all(ver(e.point)["legal"] for e in ev1.log)        # no illegal design was evaluated
    assert gate1.proposals >= ev1.used + gate1.illegal          # the rest are cached repeats
    if fn is B.random_search:
        assert gate1.illegal > 0 and ev1.used == budget        # uniform points hit edge/sample = vfu


def test_verifier_off_evaluates_illegal(env):
    S, _, train, g = env
    ev, gate, _ = _run(S, train, g, B.random_search, 0, 30, verify=False)
    ver = B.make_verifier(S, train)
    assert gate.illegal == 0 and any(not ver(e.point)["legal"] for e in ev.log)


def test_proposal_cap_stops_a_gate_that_only_sees_illegal(env):
    S, _, train, g = env
    ev = FastEvaluator(S, train, g, budget=10)
    gate = B.Gate(ev, lambda p: {"legal": False, "violations": [("R0.test", "x", "always")]})
    B.random_search(gate, B.knob_table(S), seed=0)
    assert ev.used == 0 and gate.proposals == 200 and gate.rules["R0.test"] == 200


def test_shadow_proxy(env):
    S, _, train, g = env
    ver = B.make_verifier(S, train)
    sh = B.VerifiedEvaluator(FastEvaluator(S, train, g), ver)
    sh.budget = 3                                   # forwarded to the wrapped evaluator
    assert sh._ev.budget == 3
    e1 = sh(HEADLINE)
    e2 = sh({**HEADLINE, "edge": "vfu"})
    sh(HEADLINE)                                    # cached: no new verdict
    assert sh.used == 2 and set(sh.verdicts) == {e1.n, e2.n}
    s = sh.summary([e1.n, e2.n], final_n=e2.n)
    assert s["illegal_evaluated"] == 1 and s["accepted_illegal"] == [e2.n] and s["final_legal"] is False
    assert s["rules"] == {"R1.sampling": 1, "R2.transfer": 1}


def plugin_method(run, **kw):
    """A method resolved by module:function (how a plugin without a registry entry plugs in)."""
    run.record.add_tokens("test-model", 10, 2)
    B.random_search(run.gate, run.table, seed=run.seed)


def test_runner_reproduces_walk_and_team(tmp_path):
    rows = hrun.main([str(ROOT / "experiments/h/h4.yaml"), "--seeds", "0", "--budget", "40",
                      "--methods", "walk,team,tests.test_baselines:plugin_method", "--out", str(tmp_path)])
    by = {r["method"]: r for r in rows}
    assert (by["walk"]["evals"], by["walk"]["first_1pct_eval"]) == (31, 20)
    assert by["walk"]["pick_speedup"] == pytest.approx(9.4163, abs=1e-4)
    assert (by["team"]["evals"], by["team"]["first_1pct_eval"], by["team"]["illegal"]) == (24, 15, 7)
    p = by["tests.test_baselines:plugin_method"]
    assert p["tokens_in"] == 10 and p["tokens_out"] == 2 and p["evals"] == 40
    out = tmp_path / "h4"
    rec = json.loads((out / "walk_s0.json").read_text())
    assert rec["method"] == "walk" and len(rec["evaluations"]) == 31 and rec["metrics"]["best@20"] > 9.3
    assert rec["accepted"][0]["move"]["changes"] == "start"
    with open(out / "summary.csv") as f:
        rows_csv = list(csv.DictReader(f))
    assert len(rows_csv) == 3 and "gap_certificate" in rows_csv[0] and "best@30" in rows_csv[0]
    assert 0.14 < float(rows_csv[0]["gap_certificate"]) < 0.16      # 9.457x best vs the 10.83x floor


def test_runner_shadow_mode(tmp_path):
    rows = hrun.main([str(ROOT / "experiments/h/h4.yaml"), "--seeds", "0", "--budget", "40", "--methods", "team",
                      "--set", "verifier=shadow", "--set", "experiment=h6_small", "--out", str(tmp_path)])
    (r,) = rows
    # Verifier off: the team evaluates the walk's 31 designs; 9 of them are illegal (8 past the capacity
    # wall, which the Evaluator also calls infeasible, and 1 edge=vfu), and none of them is accepted.
    assert r["evals"] == 31 and r["shadow"]["illegal_evaluated"] == 9 and r["shadow"]["final_legal"] is True
    assert r["shadow"]["rules"] == {"R3.capacity": 8, "R1.sampling": 1, "R2.transfer": 1}
    assert r["shadow"]["accepted_illegal"] == []
    rec = json.loads((tmp_path / "h6_small" / "team_s0.json").read_text())
    infeasible = {e["n"] for e in rec["evaluations"] if not e["feasible"]}
    assert len(infeasible) >= 8


def test_unknown_method_fails_early():
    with pytest.raises(KeyError):
        hrun.main([str(ROOT / "experiments/h/h4.yaml"), "--methods", "no_such_method"])
