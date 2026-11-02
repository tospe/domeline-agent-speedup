"""M5 LLM agents and LLM baselines, with a fake backend (no network)."""

import json
import subprocess
from pathlib import Path

import pytest

from harness import llm_agents as A
from harness.evaluate import Evaluator
from harness.large_space import FastEvaluator, LargeSpace
from harness.model import ModelSpec
from harness.session import GpuBaseline
from harness.space import KNOBS, START, Space
from harness.synthetic import synthetic_sessions
from harness.tech import load_library
from harness.tools import RunRecord, Toolbox
from harness.verify import verify_point

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
    return ev, Toolbox(ev, verifier=lambda p: verify_point(space, p, sess))


def large(env, budget=40):
    spec, sess, lib, ref, gpu = env
    space = LargeSpace(lib, spec, ref_ctx=ref, host="cpu")
    ev = FastEvaluator(space, sess, gpu, budget=budget)
    return ev, Toolbox(ev, verifier=lambda p: verify_point(space, p, sess))


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
    assert A.owner_of("L4.kv_bits") == "capacity" and A.owner_of("L4.rope") == "mapping"
    assert A.owner_of("link") == "hardware" and A.owner_of("xbar_scale") == "capacity"


# -- role agent ---------------------------------------------------------------------

def ctx_for(ev, tb, role, rec, call_id=1, reading=True):
    p = ev(A.space_start(ev.space)).point
    return A.AgentContext(role=role, point=p, reading=tb.read(p) if reading else None,
                          knobs=A.knobs_of(ev.space, role), history=[], feedback=None, toolbox=tb, record=rec,
                          budget_left=ev.budget - ev.used, call_id=call_id)


def test_agent_retries_out_of_scope_then_proposes(env):
    ev, tb = large(env)
    rec = RunRecord("t", "m", 0, {})
    fake = A.FakeBackend([{"why": "x", "stop": False, "changes": {"xbar_scale": "2.0"}},     # not its knob
                          {"why": "norms off the host", "stop": False, "changes": {"L*.attn_norm": "star"}}],
                         model="fake-1")
    prop = A.LLMAgent("mapping", fake).propose(ctx_for(ev, tb, "mapping", rec, call_id=7))
    assert len(prop.changes) == 36 and set(prop.changes.values()) == {"star"} and not prop.stop
    assert prop.meta["attempts"] == 2 and "not one of your knobs" in fake.calls[1]["prompt"]
    assert [c["seed"] for c in fake.calls] == [70, 71]
    assert rec.tokens["fake-1"]["in"] > 0 and rec.tokens["fake-1"]["out"] > 0
    # The prompt carries the reading and only the agent's knobs.
    assert "binding term: host" in fake.calls[0]["prompt"] and "xbar_scale" not in fake.calls[0]["prompt"].split(
        "Readings")[0]


def test_agent_gives_up_cleanly(env):
    ev, tb = small(env)
    rec = RunRecord("t", "m", 0, {})
    fake = A.FakeBackend([{"why": "", "stop": False, "changes": {}}] * 3)
    prop = A.LLMAgent("hardware", fake, retries=1).propose(ctx_for(ev, tb, "hardware", rec))
    assert prop.changes == {} and not prop.stop and "error" in prop.meta
    stop = A.LLMAgent("hardware", A.FakeBackend([])).propose(ctx_for(ev, tb, "hardware", rec))
    assert stop.stop and stop.changes == {}


def test_agent_sees_rule_table_on_small_space(env):
    ev, tb = small(env)
    fake = A.FakeBackend([{"why": "x", "stop": False, "changes": {"vector": "star"}}])
    prop = A.LLMAgent("mapping", fake).propose(ctx_for(ev, tb, "mapping", RunRecord("t", "m", 0, {})))
    assert prop.changes == {"vector": "star"}
    assert "Rule-table candidates" in fake.calls[0]["prompt"]


def test_backend_error_is_retried(env):
    ev, tb = small(env)
    fake = A.FakeBackend([A.BackendError("boom"), {"why": "x", "stop": False, "changes": {"link": "die"}}])
    prop = A.LLMAgent("hardware", fake).propose(ctx_for(ev, tb, "hardware", RunRecord("t", "m", 0, {})))
    assert prop.changes == {"link": "die"}


# -- baselines ------------------------------------------------------------------------

def test_single_agent_small(env):
    ev, tb = small(env, budget=6)
    replies = [
        {"why": "a", "stop": False, "base": 1, "changes": {"vector": "star"}},
        {"why": "same again", "stop": False, "base": 1, "changes": {"vector": "star"}},     # repeated
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
    assert outs[:3] == ["accepted", "repeated", "illegal"]
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
            fam = ["rope", "act"] if len(teams) == 1 else ["link", "softmax"]
            return {"why": "split", "stop": False, "team": [
                {"name": "w1", "brief": "move vector ops", "families": fam[:1]},
                {"name": "w2", "brief": "the rest", "families": fam[1:] + ["no_such_family"]}]}
        if "rope" in prompt.split("Readings")[0] and len(teams) == 1:
            return {"why": "", "stop": False, "changes": {"L0.rope": "cpu"}}     # no-op -> invalid
        return {"why": "try", "stop": False, "changes": {"link": "die"} if "link =" in prompt
                else {"L*.softmax": "star"} if "softmax" in prompt.split("Readings")[0] else {"L*.act": "vfu"}}

    fake = A.FakeBackend(reply, model="fake-t")
    rec = A.eat_team_search(ev, fake, tb=tb, stall_k=3)
    teams_cfg = rec.config["teams"]
    assert len(teams_cfg) >= 2 and teams_cfg[0]["team"][0]["families"] == ["rope"]
    assert all("binding term" not in c["prompt"] for c in fake.calls)            # no readings anywhere
    assert "no accepted move" in teams_cfg[1]["reason"]
    assert rec.stop in ("budget exhausted",) or "restructures" in rec.stop or rec.stop == "supervisor stopped"


def test_route_smoke(env):
    ev, tb = small(env, budget=8)

    def reply(system, prompt, schema):
        if "MAPPING" in prompt:
            return {"why": "norms", "stop": False, "changes": {"vector": "star"}}
        return {"why": "", "stop": True, "changes": {}}

    fake = A.FakeBackend(reply, model="fake-r")
    agents = {r: A.LLMAgent(r, fake) for r in A.ROLES}
    rec = A.route_search(ev, agents, tb=tb)
    assert tb.entries[0]["role"] == "mapping" and tb.entries[0]["outcome"] == "accepted"
    assert rec.accepted[0]["reading"]["binding_term"] == "host"


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
