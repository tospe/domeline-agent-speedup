"""M5 LLM agents and LLM baselines, with a fake backend (no network)."""

import json
import subprocess
from pathlib import Path

import pytest

from harness import llm_agents as A
from harness import team as T
from harness.evaluate import Evaluator
from harness.large_space import FastEvaluator, LargeSpace
from harness.model import ModelSpec
from harness.session import GpuBaseline
from harness.space import KNOBS, START, Space
from harness.synthetic import synthetic_sessions
from harness.tech import load_library
from harness.tools import RunRecord, Toolbox

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/localhome/tsousape/models/mini-coder-4b"


@pytest.fixture(scope="module")
def env():
    spec = ModelSpec.from_hf_config(MODEL)
    sess = synthetic_sessions(3, seed=4)
    lib = load_library(ROOT / "lib/tech")
    ref = max(s.max_ctx for s in sess)
    gpu = GpuBaseline(ROOT / "results/gpu/g1.jsonl", weights="Q8_0", kv="f16")
    return spec, sess, lib, ref, gpu


def small(env, budget=40):
    spec, sess, lib, ref, gpu = env
    space = Space(lib, spec, ref_ctx=ref, host="cpu")
    ev = Evaluator(space, sess, gpu, budget=budget)
    return ev, Toolbox(ev, verifier=T.point_verifier(space, sess))


def large(env, budget=40):
    spec, sess, lib, ref, gpu = env
    space = LargeSpace(lib, spec, ref_ctx=ref, host="cpu")
    ev = FastEvaluator(space, sess, gpu, budget=budget)
    return ev, Toolbox(ev, verifier=T.point_verifier(space, sess))


# -- knobs as text and back ------------------------------------------------------

def test_expand_patterns(env):
    ev, _ = large(env)
    knobs = A.all_knobs(ev.space)
    assert len(knobs) == 626
    ch = A.expand_changes(knobs, {"L*.rope": "star", "L3-5.attention": "xbar", "L7.kv_bits": "8",
                                  "vfu_bw": "1e12", "link": "die"})
    assert sum(k.endswith(".rope") for k in ch) == 36 and ch["L0.rope"] == "star"
    assert [k for k in ch if k.endswith(".attention")] == ["L3.attention", "L4.attention", "L5.attention"]
    assert ch["L7.kv_bits"] == 8 and ch["vfu_bw"] == 1e12 and ch["link"] == "die"
    assert A.expand_changes(knobs, {"L<l>.act": "vfu"})["L35.act"] == "vfu"
    # Spellings mini-coder-4b produced in the smoke run.
    assert list(A.expand_changes(knobs, {"L<0>.attn_norm": "star"})) == ["L0.attn_norm"]
    assert len(A.expand_changes(knobs, {"L*l*.attention": "xbar"})) == 36
    assert len(A.expand_changes(knobs, {"L<*>.rope": "vfu"})) == 36
    for bad in ({"L*.nope": "cpu"}, {"warp_drive": "on"}, {"L*.rope": "gpu"}, {"xbar_scale": "3"}):
        with pytest.raises(ValueError):
            A.expand_changes(knobs, bad)


def test_describe_runs(env):
    ev, _ = large(env)
    knobs = A.knobs_of(ev.space, "mapping")
    p = {**ev.space.start, **{f"L{l}.rope": "star" for l in range(10, 36)}}
    txt = A.describe_knobs(p, knobs)
    assert "L<l>.rope (36 layers) = cpu L0-9, star L10-35" in txt
    assert "xbar_scale" not in txt and "kv_bits" not in txt          # not the mapping agent's
    assert A.describe_diff(p, ev.space.start, knobs) == {"L<l>.rope": "star L10-35"}


def test_ownership_partitions_every_knob(env):
    for ev in (small(env)[0], large(env)[0]):
        k = A.all_knobs(ev.space)
        parts = [A.knobs_of(ev.space, r) for r in A.ROLES]
        assert sum(len(x) for x in parts) == len(k) and set().union(*parts) == set(k)
    L = large(env)[0].space
    assert A.owner_of("L4.kv_bits", L) == "capacity" and A.owner_of("L4.rope", L) == "mapping"
    assert A.owner_of("link") == "hardware" and A.owner_of("xbar_scale") == "capacity"


# -- role agent ---------------------------------------------------------------------

def ctx_for(ev, tb, role, term="host", rule_moves=(), reading=True, feedback=None, rnd=0):
    p = ev(A.space_start(ev.space)).point
    return T.AgentContext(role=role, term=term, point=p, reading=tb.read(p) if reading else None,
                          knobs=A.knobs_of(ev.space, role), rule_moves=list(rule_moves), history=[],
                          feedback=feedback, round=rnd, budget_left=ev.budget - ev.used)


def mv(changes, why="w"):
    return {"why": why, "changes": changes}


def test_agent_retries_out_of_scope_then_proposes(env):
    ev, tb = large(env)
    rec = RunRecord("t", "m", 0, {})
    fake = A.FakeBackend([{"pass": False, "moves": [mv({"xbar_scale": "2.0"})]},          # not its knob
                          {"pass": False, "moves": [mv({"L*.attn_norm": "star"}, "norms off the host"),
                                                    mv({"L*.attn_norm": "star"}),          # duplicate: dropped
                                                    mv({"L0-3.rope": "vfu"})]}],
                         model="fake-1")
    ag = A.LLMAgent("mapping", fake, record=rec, seed=3)
    moves = ag.propose(ctx_for(ev, tb, "mapping"))
    assert [len(m.changes) for m in moves] == [36, 4] and all(isinstance(m, T.Move) for m in moves)
    assert set(moves[0].changes.values()) == {"star"} and moves[0].agent == "mapping"
    assert "not one of your knobs" in fake.calls[1]["prompt"]
    assert [c["seed"] for c in fake.calls] == [3_000_000, 3_000_001]
    assert rec.tokens["fake-1"]["in"] > 0 and rec.tokens["fake-1"]["out"] > 0
    head = fake.calls[0]["prompt"].split("Readings")[0]
    assert "binding term: host" in fake.calls[0]["prompt"] and "xbar_scale" not in head
    assert "routed this reading item to you: host" in fake.calls[0]["prompt"]


def test_agent_pass_and_give_up(env):
    ev, tb = small(env)
    rec = RunRecord("t", "m", 0, {})
    bad = A.FakeBackend([{"pass": False, "moves": [mv({})]}] * 3)
    assert A.LLMAgent("hardware", bad, record=rec).propose(ctx_for(ev, tb, "hardware", term="launch")) == []
    assert len(bad.calls) == 2                                  # one corrective retry, then pass
    ok = A.FakeBackend([{"pass": True, "moves": []}])
    assert A.LLMAgent("hardware", ok, record=rec).propose(ctx_for(ev, tb, "hardware")) == []
    assert len(ok.calls) == 1


def test_agent_sees_rule_moves_feedback_round(env):
    ev, tb = small(env)
    fake = A.FakeBackend([{"pass": False, "moves": [mv({"vector": "star"})]}])
    rm = [T.Move({"vector": "vfu"}, why="host binds; rmsnorm runs on the host", priority=0)]
    moves = A.LLMAgent("mapping", fake).propose(
        ctx_for(ev, tb, "mapping", rule_moves=rm, feedback=[("R1.support", "L0.softmax", "x")], rnd=2))
    assert moves[0].changes == {"vector": "star"}
    pr = fake.calls[0]["prompt"]
    assert "Rule-table candidates" in pr and "rmsnorm runs on the host" in pr
    assert "ILLEGAL" in pr and "R1.support" in pr and "round 2" in pr


def test_backend_error_is_retried(env):
    ev, tb = small(env)
    fake = A.FakeBackend([A.BackendError("boom"), {"pass": False, "moves": [mv({"link": "die"})]}])
    moves = A.LLMAgent("hardware", fake).propose(ctx_for(ev, tb, "hardware"))
    assert moves[0].changes == {"link": "die"}


# -- inside C's loop -------------------------------------------------------------------

def test_llm_team_in_team_search(env):
    ev, tb = small(env, budget=8)

    def reply(system, prompt, schema):
        if "MAPPING" in prompt and "vector = cpu" in prompt:
            return {"pass": False, "moves": [mv({"vector": "star"}, "norms off the host")]}
        return {"pass": True, "moves": []}

    fake = A.FakeBackend(reply, model="fake-r")
    rec = RunRecord("t", "llm_team", 0, {})
    res = T.team_search(ev, tb, A.llm_team(fake), record=rec, max_rounds=2)
    assert res.accepted[1]["agent"] == "mapping" and res.accepted[1]["move"] == {"vector": "star"}
    assert rec.tokens["fake-r"]["in"] > 0                       # the loop handed the agents its record
    assert res.stop


# -- baselines ------------------------------------------------------------------------

def test_single_agent_small(env):
    ev, tb = small(env, budget=6)
    replies = [
        {"why": "a", "stop": False, "base": 1, "changes": {"vector": "star"}},
        {"why": "same again", "stop": False, "base": 1, "changes": {"vector": "star"}},     # duplicate
        {"why": "sample on PIM", "stop": False, "base": 1, "changes": {"edge": "vfu"}},    # illegal
        {"why": "b", "stop": False, "base": 2, "changes": {"softmax": "star"}},
        {"why": "c", "stop": False, "base": 3, "changes": {"attention": "xbar"}},
        {"why": "d", "stop": False, "base": 3, "changes": {"link": "die"}},
        {"why": "e", "stop": False, "base": 3, "changes": {"lm_head": "xbar"}},
        {"why": "f", "stop": False, "base": 3, "changes": {"xbar_tech": "xbar_transpose"}},
    ]
    fake = A.FakeBackend(replies, model="fake-s")
    rec = A.single_agent_search(ev, fake, tb=tb)
    outs = [e["outcome"] for e in tb.entries]
    assert outs[:3] == ["accepted", "duplicate", "illegal"]
    assert ev.used == 6 and rec.stop == "budget exhausted"
    assert "readings" not in fake.calls[0]["prompt"] and "binding term" not in fake.calls[-1]["prompt"]
    assert "already evaluated as #2" in fake.calls[2]["prompt"]
    assert rec.accepted and rec.tokens["fake-s"]["in"] > 0
    d = rec.to_json(ev, tb)
    json.dumps(d, default=str)
    assert d["cost"]["evaluations"] == 6


def test_single_agent_large_and_stop(env):
    ev, tb = large(env, budget=20)
    fake = A.FakeBackend([{"why": "vector ops to star", "stop": False, "base": 1, "changes": {"L*.rope": "star"}},
                          {"why": "done", "stop": True, "base": 1, "changes": {}}])
    rec = A.single_agent_search(ev, fake, tb=tb)
    assert ev.used == 2 and rec.stop.startswith("agent stopped")
    assert ev.log[1].point["L35.rope"] == "star"


def test_eat_team_restructures(env):
    ev, tb = large(env, budget=12)
    teams = []

    def reply(system, prompt, schema):
        if "team" in schema["properties"]:
            teams.append(prompt)
            fam = ["rope"] if len(teams) == 1 else ["link", "act"]
            return {"why": "try another split", "stop": False, "team": [
                {"name": f"w{len(teams)}", "brief": "move what you own", "families": fam + ["no_such_family"]}]}
        if "L<l>.rope" in prompt:                     # team 1: a no-op move -> invalid -> pass
            return {"pass": False, "moves": [mv({"L0.rope": "cpu"})]}
        if "link = cxl" in prompt:
            return {"pass": False, "moves": [mv({"link": "die"}, "faster link")]}
        return {"pass": True, "moves": []}

    fake = A.FakeBackend(reply, model="fake-t")
    rec = A.eat_team_search(ev, fake, tb=tb, stall_k=3, max_restructures=3)
    cfg = rec.config["teams"]
    assert cfg[0]["team"][0]["families"] == ["rope"] and cfg[1]["reason"] == "every worker passed"
    assert cfg[1]["team"][0]["families"] == ["act", "link"]
    assert all("binding term" not in c["prompt"] for c in fake.calls)            # no readings anywhere
    outs = [h["outcome"] for h in tb.entries]
    assert outs[0] in ("accepted", "slower") and set(outs[1:]) <= {"duplicate"}    # the same move again
    assert rec.stop == "3 restructures" and rec.tokens["fake-t"]["in"] > 0


# -- real backends' parsing (no network) ----------------------------------------------

def test_claude_cli_parsing(monkeypatch):
    out = {"structured_output": {"why": "w", "stop": False, "changes": {}},
           "modelUsage": {"claude-haiku-5-5": {"inputTokens": 3, "outputTokens": 40, "cacheReadInputTokens": 100,
                                              "cacheCreationInputTokens": 900, "costUSD": 0.001}}}
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    reply, usages = A.ClaudeCLI(A.HAIKU).call("sys", "prompt", A.AGENT_SCHEMA)
    assert reply["why"] == "w" and seen["cmd"][seen["cmd"].index("--model") + 1] == "claude-haiku-5-5"
    assert usages[0].model == "claude-haiku-5-5" and usages[0].n_in == 1003 and usages[0].n_out == 40


def test_local_parsing(monkeypatch):
    import io
    import urllib.request
    body = {"choices": [{"message": {"content": 'x {"why": "w", "stop": true, "changes": {}} y'}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 20}}
    sent = {}

    def urlopen(req, timeout=0):
        sent.update(json.loads(req.data))
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    reply, usages = A.LocalLlama().call("sys", "p", A.AGENT_SCHEMA, seed=42)
    assert reply["stop"] and usages[0].n_in == 500 and usages[0].model == A.LOCAL
    assert sent["temperature"] == 0.0 and sent["seed"] == 42


def test_agent_answers_once_per_round(env):
    """team_search asks an owner once per agenda term; the agent calls the LLM once per (design, round)."""
    ev, tb = small(env)
    fake = A.FakeBackend([{"pass": False, "moves": [mv({"link": "die"})]}] * 4)
    ag = A.LLMAgent("hardware", fake)
    assert ag.propose(ctx_for(ev, tb, "hardware", term="launch"))
    assert ag.propose(ctx_for(ev, tb, "hardware", term="transfer")) == [] and len(fake.calls) == 1
    assert ag.propose(ctx_for(ev, tb, "hardware", term="launch", rnd=1)) and len(fake.calls) == 2
    assert ag.propose(ctx_for(ev, tb, "hardware", feedback=[("R4.area", "-", "x")])) and len(fake.calls) == 3



def test_agent_drops_only_invalid_moves(env):
    ev, tb = small(env)
    fake = A.FakeBackend([{"pass": False, "moves": [mv({"vector": "star"}), mv({"link": "die"})]}])
    moves = A.LLMAgent("mapping", fake).propose(ctx_for(ev, tb, "mapping"))
    assert [m.changes for m in moves] == [{"vector": "star"}] and len(fake.calls) == 1
