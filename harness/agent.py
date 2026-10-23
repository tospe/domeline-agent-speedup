"""H8: an LLM agent as the searcher (the autonomous harness).

Each step the agent sees the goal, the knobs, the budget left and the
history of evaluations, and returns the next design point as JSON. With
`readings=True` the history carries Domeline's readings (which terms bind,
which ops sit on the host, where capacity runs out); with `readings=False`
it carries only the scalar results, the ablation that isolates what the
readings are worth.

Backend: the `claude` CLI in headless mode (no tools, structured output).
"""

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from typing import List

from .evaluate import BudgetExhausted
from .search import _fmt, readings
from .space import KNOBS, START

KNOB_HELP = {
    "mvm": "where the 7 weight MVMs of every layer run",
    "lm_head": "where the vocabulary projection runs",
    "attention": "where q.K^T, p.V and the KV appends run (xbar: KV written into crossbar arrays; vfu: KV in memory next to a vector unit)",
    "xbar_tech": "crossbar flavor (xbar_transpose: K append is one row write; +15% area, 3 months)",
    "softmax": "where softmax runs (star: fixed-function nonlinear unit, 4 months)",
    "vector": "where RMSNorm, QK-norm, RoPE, SiLU*up and residual adds run",
    "edge": "where the embedding lookup and sampling run",
    "xbar_scale": "crossbar arrays as a multiple of what the design needs at the longest context",
    "vfu_units": "vector-unit lanes",
    "vfu_bw": "bandwidth of the KV memory next to the vector unit (B/s)",
    "vfu_mem": "capacity of that KV memory (bytes)",
    "star_units": "fixed-function nonlinear unit instances",
    "link": "CPU-coprocessor coupling (cxl < package < die: faster, more effort)",
    "kv_bits": "KV cache precision",
    "xbar_tile": "crossbar tile (lean: no eDRAM buffer/bus, -47% tile area, 2 months)",
    "adc_per_ima": "ADCs per 8 crossbars (fewer: -31% tile area at 2, but each read takes 8/n times longer)",
    "cell_bits": "bits per ReRAM cell (4: half the arrays, 1.27x area per array, 1 us write-verify row writes, 3 months)",
}

SYSTEM = """You design a processing-in-memory (PIM) coprocessor that sits next to a CPU and runs \
an LLM coding agent's decode faster than an RTX 3080 GPU. Tools always run on the CPU. Every \
design you propose is evaluated by an analytical model over recorded agent sessions; evaluations \
are scarce, so each proposal should be the most informative next step. Answer only with the JSON \
object requested."""


@dataclass
class AgentRun:
    steps: List[dict] = field(default_factory=list)
    cost_usd: float = 0.0
    stop: str = ""


def _str(v):
    return str(v)


def _schema():
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "stop": {"type": "boolean"},
            "point": {"type": "object",
                      "properties": {k: {"type": "string", "enum": [_str(o) for o in v]} for k, v in KNOBS.items()},
                      "required": list(KNOBS)},
        },
        "required": ["reasoning", "stop", "point"],
    }


def _describe(e, with_readings):
    diff = {k: _fmt(e.point[k]) for k in KNOBS if e.point[k] != START[k]}
    line = (f"#{e.n}: changes vs start {json.dumps(diff)} -> workload {e.time_s:.1f} s "
            f"({e.speedup_vs_gpu:.2f}x GPU), area {e.area_mm2:.0f} mm2, effort {e.effort_months:.0f} mo, "
            f"{'feasible' if e.feasible else 'INFEASIBLE: ' + e.reason}")
    if with_readings:
        terms = ", ".join(f"{t} {share:.0%}" for t, _, share in readings(e)[:4])
        kinds = sorted(((k, v) for k, v in e.by_kind.items()), key=lambda x: -x[1])[:4]
        tot = sum(e.by_kind.values()) or 1
        line += (f"\n    readings: LLM time {e.llm_s:.1f} s, tools {e.tools_s:.1f} s; binding terms: {terms}; "
                 f"top ops: {', '.join(f'{k} {v / tot:.0%}' for k, v in kinds)}")
        if e.wall_ctx:
            line += f"; capacity runs out at {e.wall_ctx} tokens"
    return line


def _prompt(ev, with_readings, history):
    knobs = "\n".join(f"- {k}: {[_str(o) for o in v]} — {KNOB_HELP[k]}" for k, v in KNOBS.items())
    hist = "\n".join(_describe(e, with_readings) for e in history) or "(none yet)"
    left = ev.budget - ev.used
    caps = []
    if ev.area_cap < math.inf:
        caps.append(f"area <= {ev.area_cap:.0f} mm2")
    if ev.effort_cap < math.inf:
        caps.append(f"effort <= {ev.effort_cap:.0f} months")
    goal = ("Minimize total workload time (LLM + tools) for a feasible design"
            + (f" subject to {' and '.join(caps)}" if caps else "") + ". "
            f"The GPU takes {ev.gpu_time:.1f} s on this workload; tools alone take {ev.tools_time:.1f} s, "
            "so no design can go below that.")
    readings_note = ("Each result lists Domeline's readings: the terms that bind (host = ops left on the CPU, "
                     "launch/transfer = CPU<->PIM round trips, write = KV appends or reprogramming, and each "
                     "coprocessor unit's own compute) and the ops that take the most time."
                     if with_readings else "Each result gives only the workload time, area, effort and feasibility.")
    return (f"{goal}\n\nKnobs (choose one option each):\n{knobs}\n\nStart design: "
            f"{json.dumps({k: _str(v) for k, v in START.items()})}\n\n{readings_note}\n\n"
            f"Evaluations so far:\n{hist}\n\nBudget left: {left} evaluations. Propose the next design "
            "(a point not evaluated yet), with 1-3 sentences of reasoning. Set stop=true only if you are "
            "confident no remaining design is meaningfully better.")


def _parse_point(d):
    out = {}
    for k, opts in KNOBS.items():
        s = d[k]
        out[k] = opts[[_str(o) for o in opts].index(s)]
    return out


def call_claude(prompt, model="sonnet", schema=None, timeout=300):
    cmd = ["claude", "-p", "--model", model, "--output-format", "json",
           "--system-prompt", SYSTEM, "--tools", "", "--no-session-persistence"]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    for attempt in range(4):           # transient CLI/API failures: back off and retry
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        if r.returncode == 0:
            break
        time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"claude failed: {r.stderr[:300]} {r.stdout[-300:]}")
    d = json.loads(r.stdout)
    out = d.get("structured_output")
    if out is None:
        txt = d.get("result", "")
        txt = txt[txt.find("{"): txt.rfind("}") + 1]
        out = json.loads(txt)
    return out, d.get("total_cost_usd", 0.0)


LOCAL_URL = "http://127.0.0.1:8080/v1/chat/completions"


def call_local(prompt, schema=None, seed=0, timeout=600):
    """Same contract as call_claude, against a local llama-server (grammar-constrained JSON)."""
    import urllib.request
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0.7, "seed": seed, "max_tokens": 1024}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema}}
    req = urllib.request.Request(LOCAL_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    return json.loads(txt), 0.0


def agent_search(ev, with_readings=True, model="sonnet", max_repeats=3, seed=0):
    run = AgentRun()
    history = []
    try:
        history.append(ev(START))
        repeats = 0
        while True:
            prompt = _prompt(ev, with_readings, history)
            if model == "local":
                out, cost = call_local(prompt, schema=_schema(), seed=seed * 1000 + ev.used)
            else:
                out, cost = call_claude(prompt, model=model, schema=_schema())
            run.cost_usd += cost
            if out.get("stop"):
                run.stop = "agent stopped: " + out.get("reasoning", "")[:200]
                break
            p = _parse_point(out["point"])
            before = ev.used
            e = ev(p)
            run.steps.append({"reasoning": out.get("reasoning", ""), "n": e.n, "time_s": e.time_s,
                              "feasible": e.feasible, "new": ev.used > before})
            if ev.used == before:          # proposed a design already evaluated
                repeats += 1
                if repeats >= max_repeats:
                    run.stop = "agent repeated evaluated designs"
                    break
                continue
            repeats = 0
            history.append(e)
    except BudgetExhausted:
        run.stop = "budget exhausted"
    return run
