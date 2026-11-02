"""M5: LLM agents for the team loop, and the LLM-based E5 baselines.

Interface with the loop (harness/team.py, Agent C): docs/harness-dev/AGENT_IFACE.md. An agent
has a `role` and `propose(ctx: AgentContext) -> Proposal`; it sees only its own knobs, never
calls `evaluate` (the loop owns the budget) and records every LLM call's tokens with
`ctx.record.add_tokens`.

  Backends     ClaudeCLI("claude-sonnet-5-5" | "claude-haiku-5-5")  the `claude` CLI, headless
               LocalLlama()     mini-coder-4b on llama-server (traces/serve.sh), temperature 0
               FakeBackend(..)  canned responses, for tests (no network)
  LLMAgent     one role agent (mapping / hardware / capacity), structured JSON out
  Baselines    single_agent_search   one LLM, every knob, no readings (agent.py's ablation,
                                     generalized to LargeSpace)
               eat_team_search       an Evolving-Agent-Teams-style team: an LLM supervisor sets
                                     and restructures a team of workers; no readings, no routing
  route_search a minimal binding-term router for smoke tests of the role agents until
               team.team_search exists. Not the paper's loop.

Large spaces: LargeSpace has 626 knobs, so prompts show per-layer slots as runs per op family
("rope: cpu L0-35") and agents write group moves as patterns ("L*.rope", "L10-20.attention"),
which `expand_changes` turns into concrete knobs.

Claude through the CLI has no temperature or seed: those runs are not reproducible call for
call; the record keeps every prompt's outcome. The local backend runs at temperature 0 with a
seed derived from the loop's call counter.
"""

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .agent import KNOB_HELP, LOCAL_URL
from .evaluate import BudgetExhausted, Eval
from .search import better
from .space import KNOBS, START
from .tools import RunRecord, Toolbox

SONNET, HAIKU, LOCAL = "claude-sonnet-5-5", "claude-haiku-5-5", "mini-coder-4b-Q8_0"

SYSTEM = """You are part of a team designing a processing-in-memory (PIM) coprocessor that sits next \
to a CPU and runs an LLM coding agent's decode faster than an RTX 3080 GPU. Tools always run on the \
CPU. Every design is evaluated by an analytical model over recorded agent sessions; evaluations are \
scarce, so each proposal should be the most informative next step. Answer only with the JSON object \
requested."""


# ---------------------------------------------------------------------------
# Loop interface (AGENT_IFACE.md). team.py's classes win when it exists.
# ---------------------------------------------------------------------------

ROLES = ("mapping", "hardware", "capacity")
CAPACITY_HW = ("xbar_scale", "vfu_mem")
MAPPING_SMALL = ("mvm", "lm_head", "attention", "softmax", "vector", "edge")


def _owner_fallback(knob):
    if knob in CAPACITY_HW or knob == "kv_bits" or knob.endswith(".kv_bits"):
        return "capacity"
    if knob in KNOBS and knob not in MAPPING_SMALL:
        return "hardware"
    return "mapping"


@dataclass
class _AgentContext:
    role: str
    point: Dict
    reading: Optional[Dict]
    knobs: Dict[str, list]
    history: List[Dict] = field(default_factory=list)
    feedback: Optional[Dict] = None
    toolbox: Optional[Toolbox] = None
    record: Optional[RunRecord] = None
    budget_left: float = math.inf
    call_id: int = 0


@dataclass
class _Proposal:
    changes: Dict
    why: str
    stop: bool = False
    meta: Dict = field(default_factory=dict)


try:                                     # Agent C's loop module, once merged
    from .team import AgentContext, Proposal, owner_of  # type: ignore
except ImportError:                      # pragma: no cover - until team.py lands
    AgentContext, Proposal, owner_of = _AgentContext, _Proposal, _owner_fallback


def all_knobs(space) -> Dict[str, list]:
    """Every knob of a Space or LargeSpace -> its options, in the space's order."""
    if hasattr(space, "slots"):
        return {k: (KNOBS[k] if k in KNOBS else space.slots[k].options) for k in space.knobs}
    return dict(KNOBS)


def knobs_of(space, role) -> Dict[str, list]:
    return {k: v for k, v in all_knobs(space).items() if owner_of(k) == role}


def space_start(space):
    return dict(getattr(space, "start", START))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    model: str
    n_in: int                 # all input tokens: fresh + cache read + cache write
    n_out: int
    cost_usd: float = 0.0
    cache_read: int = 0
    cache_write: int = 0
    wall_s: float = 0.0


class BackendError(RuntimeError):
    pass


def _json_in(txt):
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < i:
        raise BackendError(f"no JSON object in reply: {txt[:200]!r}")
    return json.loads(txt[i:j + 1])


@dataclass
class ClaudeCLI:
    """The `claude` CLI headless, no tools, structured output; model pinned by ID."""
    model: str = SONNET
    timeout: float = 300
    retries: int = 3
    backoff_s: float = 30

    def call(self, system, prompt, schema=None, seed=0):
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json", "--system-prompt", system,
               "--tools", "", "--no-session-persistence"]
        if schema:
            cmd += ["--json-schema", json.dumps(schema)]
        t0 = time.time()
        err = ""
        for attempt in range(self.retries + 1):
            try:
                r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout,
                                   cwd="/tmp")
                if r.returncode == 0:
                    break
                err = f"exit {r.returncode}: {r.stderr[:300]} {r.stdout[-300:]}"
            except subprocess.TimeoutExpired:
                err = "timeout"
            if attempt < self.retries:
                time.sleep(self.backoff_s * (attempt + 1))
        else:
            raise BackendError(f"claude CLI failed: {err}")
        d = json.loads(r.stdout)
        out = d.get("structured_output")
        if out is None:
            out = _json_in(d.get("result", ""))
        usages = []
        for model, u in (d.get("modelUsage") or {}).items():
            cr, cw = u.get("cacheReadInputTokens", 0), u.get("cacheCreationInputTokens", 0)
            usages.append(Usage(model, u.get("inputTokens", 0) + cr + cw, u.get("outputTokens", 0),
                                u.get("costUSD", 0.0), cr, cw, time.time() - t0))
        if not usages:
            u = d.get("usage") or {}
            cr, cw = u.get("cache_read_input_tokens", 0), u.get("cache_creation_input_tokens", 0)
            usages.append(Usage(self.model, u.get("input_tokens", 0) + cr + cw, u.get("output_tokens", 0),
                                d.get("total_cost_usd", 0.0), cr, cw, time.time() - t0))
        return out, usages


@dataclass
class LocalLlama:
    """mini-coder-4b on llama-server (OpenAI-style endpoint), grammar-constrained JSON."""
    url: str = LOCAL_URL
    model: str = LOCAL
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: float = 600

    def call(self, system, prompt, schema=None, seed=0):
        import urllib.request
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "temperature": self.temperature, "seed": int(seed), "max_tokens": self.max_tokens}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"schema": schema}}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.loads(r.read())
        except OSError as ex:
            raise BackendError(f"llama-server at {self.url}: {ex}") from ex
        u = d.get("usage") or {}
        out = _json_in(d["choices"][0]["message"]["content"])
        return out, [Usage(self.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                           wall_s=time.time() - t0)]


@dataclass
class FakeBackend:
    """Canned replies: a list (used in order; then `default`) or a function (system, prompt, schema) -> dict."""
    replies: object = ()
    model: str = "fake"
    default: Dict = field(default_factory=lambda: {"why": "nothing left", "stop": True, "changes": {}})
    calls: List[Dict] = field(default_factory=list)

    def call(self, system, prompt, schema=None, seed=0):
        i = len(self.calls)
        self.calls.append({"prompt": prompt, "schema": schema, "seed": seed})
        if callable(self.replies):
            out = self.replies(system, prompt, schema)
        else:
            out = self.replies[i] if i < len(self.replies) else self.default
        if isinstance(out, Exception):
            raise out
        return out, [Usage(self.model, len(prompt) // 4, len(json.dumps(out)) // 4)]


def ask(backend, prompt, schema, record: Optional[RunRecord], seed=0, system=SYSTEM):
    """One backend call, its tokens recorded per model. Returns (reply, usages)."""
    out, usages = backend.call(system, prompt, schema, seed)
    for u in usages:
        if record is not None:
            record.add_tokens(u.model, u.n_in, u.n_out)
    return out, usages


# ---------------------------------------------------------------------------
# Knobs as text, and patterns back to knobs
# ---------------------------------------------------------------------------

_SLOT = re.compile(r"^L(\d+)\.(.+)$")
_PATTERN = re.compile(r"^L(\*|\d+(?:-\d+)?)\.(.+)$")


def opt_str(v):
    return str(v)


def parse_value(opts, s):
    """The option of `opts` that `s` names (exact string, else numerically equal)."""
    for o in opts:
        if opt_str(o) == str(s):
            return o
    try:
        f = float(str(s).replace("G", "e9"))
    except ValueError:
        f = None
    if f is not None:
        for o in opts:
            if isinstance(o, (int, float)) and not isinstance(o, bool) and float(o) == f:
                return o
    raise ValueError(f"{s!r} is not one of {[opt_str(o) for o in opts]}")


def families(knobs: Sequence[str]) -> Dict[str, List[str]]:
    """Op family -> its per-layer knobs ("rope" -> [L0.rope, ...]); other knobs are their own family."""
    out: Dict[str, List[str]] = {}
    for k in knobs:
        m = _SLOT.match(k)
        out.setdefault(m.group(2) if m else k, []).append(k)
    return out


def _runs(point, ks):
    """'cpu L0-9, star L10-35' for one family's per-layer knobs."""
    items = sorted(((int(_SLOT.match(k).group(1)), point.get(k)) for k in ks))
    runs, out = [], []
    for l, v in items:
        if runs and runs[-1][2] == v and runs[-1][1] == l - 1:
            runs[-1][1] = l
        else:
            runs.append([l, l, v])
    for a, b, v in runs:
        out.append(f"{opt_str(v)} L{a}" + (f"-{b}" if b > a else ""))
    return ", ".join(out)


def _help(fam):
    if fam in KNOB_HELP:
        return KNOB_HELP[fam]
    from .large_space import role_of
    kind = {"attn_norm": "rmsnorm", "mlp_norm": "rmsnorm", "final_norm": "rmsnorm",
            "attn_residual": "residual", "mlp_residual": "residual"}.get(fam, fam)
    r = role_of(kind)
    return f"one op ({fam}); role: {KNOB_HELP.get(r, r)}" if r else fam


def describe_knobs(point, knobs: Dict[str, list]) -> str:
    """One line per knob or per-layer family: options, current values, what it does."""
    lines = []
    for fam, ks in families(knobs).items():
        opts = [opt_str(o) for o in knobs[ks[0]]]
        if len(ks) == 1 and not _SLOT.match(ks[0]):
            cur = opt_str(point.get(ks[0]))
            lines.append(f"- {ks[0]} = {cur}   options {opts} — {_help(fam)}")
        else:
            lines.append(f"- L<l>.{fam} ({len(ks)} layers) = {_runs(point, ks)}   options {opts} — {_help(fam)}")
    return "\n".join(lines)


def describe_diff(point, base, knobs) -> Dict[str, str]:
    """Knobs where point differs from base, per-layer families compressed to runs."""
    out = {}
    for fam, ks in families(knobs).items():
        changed = [k for k in ks if point.get(k) != base.get(k)]
        if not changed:
            continue
        if len(ks) == 1 and not _SLOT.match(ks[0]):
            out[ks[0]] = opt_str(point.get(ks[0]))
        else:
            out[f"L<l>.{fam}"] = _runs(point, changed)
    return out


def expand_changes(knobs: Dict[str, list], changes: Dict) -> Dict:
    """Concrete knob -> option from the agent's changes; patterns L*.fam, La-b.fam, Ln.fam."""
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object of knob -> value")
    out = {}
    for pat, val in changes.items():
        if pat in knobs:
            targets = [pat]
        else:
            m = _PATTERN.match(str(pat).replace("L<l>.", "L*."))
            if not m:
                raise ValueError(f"{pat!r} is not one of your knobs")
            sel, fam = m.groups()
            fam_ks = [k for k in knobs if _SLOT.match(k) and _SLOT.match(k).group(2) == fam]
            if sel == "*":
                targets = fam_ks
            else:
                a, _, b = sel.partition("-")
                lo, hi = int(a), int(b or a)
                targets = [k for k in fam_ks if lo <= int(_SLOT.match(k).group(1)) <= hi]
            if not targets:
                raise ValueError(f"{pat!r} matches none of your knobs")
        for k in targets:
            out[k] = parse_value(knobs[k], val)
    return out


def describe_reading(r: Optional[Dict]) -> str:
    if not r:
        return "(no readings in this run)"
    rank = "; ".join(f"{x['term']} {x['share']:.0%}" + (f" (x{x['headroom']:.2f} over the next)"
                                                         if x.get("headroom") else "")
                     for x in r["ranking"][:5])
    out = (f"binding term: {r['binding_term']}; LLM-side terms: {rank}. LLM time is {r['llm_share']:.0%} "
           f"of the workload; the design is {r['amdahl_gap']:.2f}x above the Amdahl floor (tools only, "
           f"{r['amdahl_floor_s']:.0f} s).")
    if r.get("wall_ctx"):
        out += f" Capacity runs out at {r['wall_ctx']} tokens (longest context {r['max_ctx']})."
    if r.get("capacity_headroom_tech"):
        h = r.get("capacity_headroom_bytes")
        out += f" Least capacity headroom: {r['capacity_headroom_tech']} ({(h or 0) / 1e9:.2f} GB)."
    if not r.get("feasible", True):
        out += f" INFEASIBLE: {r.get('reason')}."
    return out


def _verdict(tb: Optional[Toolbox], point):
    """Toolbox.verify as a JSON-able dict (None without a verifier)."""
    if tb is None or tb.verifier is None:
        return None
    v = tb.verify(point)
    if v is None or isinstance(v, dict):
        return v
    return {"legal": v.legal, "violations": [[x[0], x[1], x[2]] for x in v.violations],
            "summary": v.summary() if hasattr(v, "summary") else ""}


def _line(e: Eval, base, knobs):
    return (f"#{e.n}: changes vs start {json.dumps(describe_diff(e.point, base, knobs))} -> "
            f"{e.time_s:.1f} s ({e.speedup_vs_gpu:.2f}x GPU), area {e.area_mm2:.0f} mm2, "
            f"effort {e.effort_months:.0f} mo, " + ("feasible" if e.feasible else "INFEASIBLE: " + e.reason))


# ---------------------------------------------------------------------------
# Role agent
# ---------------------------------------------------------------------------

ROLE_BRIEF = {
    "mapping": "You are the MAPPING agent: you decide which unit runs each operator (cpu = the host; xbar = "
               "crossbar arrays; vfu = vector unit; star = fixed-function nonlinear unit), per layer or in "
               "groups of layers. You move the design point.",
    "hardware": "You are the HARDWARE agent: you pick and size the units (crossbar flavor and tile, ADC "
                "sharing, bits per cell, vector and nonlinear unit counts, KV bandwidth, the CPU link). "
                "You raise the roof.",
    "capacity": "You are the CAPACITY agent: you manage what PIM must hold (crossbar arrays relative to "
                "need, KV memory next to the vector unit, KV precision per layer). You move the wall.",
}

AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "why": {"type": "string"},
        "stop": {"type": "boolean"},
        "changes": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["why", "stop", "changes"],
}


def _candidates(ctx, n=6):
    """Rule-table moves (Toolbox.moves) restricted to the agent's knobs; [] where unavailable."""
    if ctx.reading is None or ctx.toolbox is None:
        return []
    try:
        ms = ctx.toolbox.moves(ctx.point)
    except (KeyError, TypeError, ValueError):     # e.g. LargeSpace points (moves reads small knobs)
        return []
    out = []
    for m in ms:
        diff = {k: v for k, v in m["point"].items() if ctx.point.get(k) != v}
        if diff and set(diff) <= set(ctx.knobs):
            out.append(f"- {m['why']}: {json.dumps({k: opt_str(v) for k, v in diff.items()})}")
    return out[:n]


def _history_lines(hist, n):
    out = []
    for h in hist[-n:]:
        ch = h.get("changes") or {}
        shown = dict(list(ch.items())[:6]) if len(ch) <= 6 else {**dict(list(ch.items())[:4]),
                                                                 "...": f"{len(ch) - 4} more"}
        line = f"- {h.get('role', '?')}: {json.dumps({k: opt_str(v) for k, v in shown.items()})} -> {h.get('outcome')}"
        if h.get("speedup") is not None:
            line += f" ({h['speedup']:.2f}x)"
        if h.get("violations"):
            line += f"; violations: {h['violations'][:3]}"
        out.append(line)
    return "\n".join(out) or "(none yet)"


@dataclass
class LLMAgent:
    """One role agent: reading + own knobs + history in, one structured move out."""
    role: str
    backend: object
    brief: str = ""                 # replaces ROLE_BRIEF[role] (the EAT team's workers)
    name: str = ""
    history_n: int = 8
    retries: int = 1                # corrective retries for malformed or out-of-scope answers

    def prompt(self, ctx) -> str:
        brief = self.brief or ROLE_BRIEF.get(self.role, f"You are the {self.role} agent.")
        cands = _candidates(ctx)
        parts = [brief,
                 "Goal: minimize total workload time (LLM + tools) with a feasible, legal design. "
                 "A move is accepted only if it is feasible, passes the offload verifier and is more than "
                 "0.5% faster than the current design.",
                 f"Current design, your knobs only (change nothing else):\n{describe_knobs(ctx.point, ctx.knobs)}",
                 f"Readings of the current design: {describe_reading(ctx.reading)}"]
        if cands:
            parts.append("Rule-table candidates for your knobs (you may pick one or do better):\n" + "\n".join(cands))
        parts.append(f"Recent proposals by the team:\n{_history_lines(ctx.history, self.history_n)}")
        if ctx.feedback:
            parts.append(f"Your last proposal was rejected: {json.dumps(ctx.feedback, default=str)[:800]}")
        parts.append(
            f"Evaluations left: {ctx.budget_left}. Propose ONE move: `changes` maps knobs to new values "
            "(strings from the options). For per-layer knobs you may write a group: \"L*.<op>\" (every layer), "
            "\"L<a>-<b>.<op>\" (layers a..b) or \"L<n>.<op>\". Give 1-2 sentences in `why`. Set stop=true "
            "(with empty changes) only if no change of your knobs can help now.")
        return "\n\n".join(parts)

    def propose(self, ctx) -> "Proposal":
        base = self.prompt(ctx)
        meta = {"model": getattr(self.backend, "model", "?"), "tokens_in": 0, "tokens_out": 0, "attempts": 0}
        err = None
        for attempt in range(self.retries + 1):
            prompt = base if err is None else base + f"\n\nYour previous answer was invalid: {err}. Answer again."
            meta["attempts"] += 1
            try:
                out, usages = ask(self.backend, prompt, AGENT_SCHEMA, ctx.record, seed=ctx.call_id * 10 + attempt)
            except (BackendError, json.JSONDecodeError) as ex:
                err = f"backend error: {ex}"
                continue
            for u in usages:
                meta["tokens_in"] += u.n_in
                meta["tokens_out"] += u.n_out
            meta["raw"] = out
            try:
                changes = expand_changes(ctx.knobs, out.get("changes") or {})
                changes = {k: v for k, v in changes.items() if ctx.point.get(k) != v}
                stop = bool(out.get("stop"))
                if not changes and not stop:
                    raise ValueError("no knob differs from the current design")
                return Proposal(changes=changes, why=str(out.get("why", ""))[:500], stop=stop and not changes,
                                meta=meta)
            except (ValueError, TypeError, AttributeError) as ex:
                err = str(ex)
        meta["error"] = err
        return Proposal(changes={}, why=f"no valid proposal: {err}", stop=False, meta=meta)


# ---------------------------------------------------------------------------
# Shared accept step for the baselines and the smoke router
# ---------------------------------------------------------------------------

@dataclass
class _Run:
    ev: object
    tb: Toolbox
    record: RunRecord
    eps: float
    use_verifier: bool = True
    hist: List[Dict] = field(default_factory=list)
    calls: int = 0

    def check(self, base: Eval, who, prop):
        """Verify and evaluate base + prop.changes. Returns (Eval or None, outcome, entry); outcome is
        illegal | repeated | None (evaluated, not judged yet)."""
        q = {**base.point, **prop.changes}
        entry = {"role": who, "changes": {k: opt_str(v) for k, v in prop.changes.items()}, "why": prop.why,
                 "outcome": None, "time_s": None, "speedup": None, "violations": None, "meta": _slim(prop.meta)}
        verdict = _verdict(self.tb, q) if self.use_verifier else None
        entry["verify"] = verdict
        if verdict is not None and not verdict.get("legal", True):
            entry.update(outcome="illegal", violations=verdict.get("violations"))
            return None, "illegal", entry
        before = self.ev.used
        e = self.ev(q)
        entry.update(time_s=e.time_s, speedup=e.speedup_vs_gpu, eval_n=e.n)
        if self.ev.used == before:
            entry["outcome"] = "repeated"
            return e, "repeated", entry
        return e, None, entry

    def try_move(self, cur: Eval, who, prop, reading=None):
        """The loop's step: accept cur + changes if more than eps faster. Returns (current, outcome, entry)."""
        e, outcome, entry = self.check(cur, who, prop)
        if outcome is None:
            if better(e, cur, self.eps):
                outcome = "accepted"
                self.record.accept(entry["changes"], reading, e, entry["verify"])
                cur = e
            else:
                outcome = "not_faster" if e.feasible else "infeasible: " + e.reason
        entry["outcome"] = outcome
        self.log(entry)
        return cur, outcome, entry

    def log(self, entry):
        self.hist.append(entry)
        self.tb.log(entry)


def _slim(meta):
    return {k: v for k, v in (meta or {}).items() if k != "raw"} | (
        {"raw": meta["raw"]} if meta and len(json.dumps(meta.get("raw", ""), default=str)) < 2000 else {})


def _record(ev, experiment, method, seed, config):
    return RunRecord(experiment, method, seed, config)


def _best_feasible(ev):
    feas = [e for e in ev.log if e.feasible] or list(ev.log)
    return min(feas, key=lambda e: e.time_s)


# ---------------------------------------------------------------------------
# Baseline 1: a single LLM agent, every knob, no readings
# ---------------------------------------------------------------------------

SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "why": {"type": "string"},
        "stop": {"type": "boolean"},
        "base": {"type": "integer"},
        "changes": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["why", "stop", "base", "changes"],
}


def _single_prompt(ev, knobs, start, hist_evals, best, with_readings, tb, feedback, history_n):
    caps = []
    if ev.area_cap < math.inf:
        caps.append(f"area <= {ev.area_cap:.0f} mm2")
    if ev.effort_cap < math.inf:
        caps.append(f"effort <= {ev.effort_cap:.0f} months")
    goal = ("Minimize total workload time (LLM + tools) for a feasible design"
            + (f" subject to {' and '.join(caps)}" if caps else "") + ". "
            f"The GPU takes {ev.gpu_time:.1f} s on this workload; tools alone take {ev.tools_time:.1f} s, so no "
            "design can go below that.")
    shown = hist_evals[-history_n:]
    if best not in shown:
        shown = [best] + shown
    lines = []
    for e in shown:
        line = _line(e, start, knobs)
        if with_readings:
            line += "\n    readings: " + describe_reading(tb.read(e.point))
        lines.append(line)
    note = ("Each result lists Domeline's readings." if with_readings
            else "Each result gives only the workload time, area, effort and feasibility.")
    return "\n\n".join([
        goal,
        f"Knobs at the start design (choose one option each):\n{describe_knobs(start, knobs)}",
        note,
        f"Evaluations so far ({len(hist_evals)}; best is #{best.n}; older ones omitted):\n" + "\n".join(lines),
        (f"Your last answer was rejected: {feedback}" if feedback else ""),
        f"Budget left: {ev.budget - ev.used} evaluations. Propose the next design as `base` (the # of an "
        "evaluated design to start from) plus `changes` (knob -> value; for per-layer knobs a group "
        "\"L*.<op>\", \"L<a>-<b>.<op>\" or \"L<n>.<op>\"). It must differ from every evaluated design. "
        "1-3 sentences in `why`. Set stop=true only if you are confident no remaining design is meaningfully "
        "better."]).replace("\n\n\n\n", "\n\n")


def single_agent_search(ev, backend, *, with_readings=False, tb=None, record=None, start=None, eps=0.005,
                        max_repeats=5, max_illegal=10, history_n=30, seed=0, use_verifier=True,
                        experiment="E5"):
    """One LLM proposes whole designs (as base + changes) from the scalar history. Returns a RunRecord.

    Stops on budget, on the agent's stop, or after max_repeats repeated/invalid answers or
    max_illegal illegal proposals in a row (neither costs an evaluation)."""
    space = ev.space
    knobs = all_knobs(space)
    start = start or space_start(space)
    tb = tb or Toolbox(ev)
    record = record or _record(ev, experiment, f"single_llm{'_readings' if with_readings else ''}", seed,
                               {"model": backend.model, "with_readings": with_readings, "budget": ev.budget,
                                "eps": eps, "space": type(space).__name__, "use_verifier": use_verifier})
    run = _Run(ev, tb, record, eps, use_verifier)
    try:
        best = ev(start)
        evals = [best]
        repeats, illegal, feedback = 0, 0, None
        while True:
            prompt = _single_prompt(ev, knobs, start, evals, best, with_readings, tb, feedback, history_n)
            feedback = None
            run.calls += 1
            try:
                out, usages = ask(backend, prompt, SINGLE_SCHEMA, record, seed=seed * 100000 + run.calls)
            except (BackendError, json.JSONDecodeError) as ex:
                feedback = f"backend error: {ex}"
                repeats += 1
                if repeats >= max_repeats:
                    record.stop = f"backend failed {max_repeats} times: {ex}"
                    break
                continue
            if out.get("stop"):
                record.stop = "agent stopped: " + str(out.get("why", ""))[:200]
                break
            by_n = {e.n: e for e in ev.log}
            base = by_n.get(out.get("base"), best)
            try:
                changes = expand_changes(knobs, out.get("changes") or {})
            except ValueError as ex:
                feedback = str(ex)
                repeats += 1
                if repeats >= max_repeats:
                    record.stop = "agent gave invalid designs"
                    break
                continue
            prop = Proposal(changes=changes, why=str(out.get("why", ""))[:500],
                            meta={"model": backend.model, "base": base.n, "raw": out})
            e, outcome, entry = run.check(base, "single", prop)
            if outcome is None:              # a new design: accepted iff it beats the best so far
                outcome = "accepted" if better(e, best, eps) else (
                    "not_faster" if e.feasible else "infeasible: " + e.reason)
                if outcome == "accepted":
                    record.accept(entry["changes"], None, e, entry["verify"])
                if e.feasible and (not best.feasible or e.time_s < best.time_s):
                    best = e
                evals.append(e)
                repeats = 0
            entry["outcome"] = outcome
            run.log(entry)
            if outcome == "repeated":
                repeats += 1
                feedback = f"that design was already evaluated as #{entry.get('eval_n')}"
                if repeats >= max_repeats:
                    record.stop = "agent repeated evaluated designs"
                    break
            elif outcome == "illegal":
                illegal += 1
                feedback = f"illegal offload: {(entry['violations'] or [])[:3]}"
                if illegal >= max_illegal:
                    record.stop = f"{max_illegal} illegal proposals in a row"
                    break
            else:
                illegal = 0
    except BudgetExhausted:
        record.stop = "budget exhausted"
    record.config["llm_calls"] = run.calls
    return record


# ---------------------------------------------------------------------------
# Baseline 2: Evolving-Agent-Teams-style supervisor team, no readings, no routing
# ---------------------------------------------------------------------------

TEAM_SCHEMA = {
    "type": "object",
    "properties": {
        "why": {"type": "string"},
        "stop": {"type": "boolean"},
        "team": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "brief": {"type": "string"},
                           "families": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "brief", "families"]}},
    },
    "required": ["why", "stop", "team"],
}


def _team_prompt(ev, knobs, start, evals, best, team, hist, reason):
    fams = families(knobs)
    fam_lines = "\n".join(f"- {f}" + (f" ({len(ks)} per-layer knobs)" if len(ks) > 1 else "") + f": {_help(f)}"
                          for f, ks in fams.items())
    cur_team = "\n".join(f"- {w.name}: {sorted(families(wk))} — {w.brief}" for w, wk in team) or "(no team yet)"
    lines = "\n".join(_line(e, start, knobs) for e in ([best] + [e for e in evals[-10:] if e is not best]))
    return "\n\n".join([
        "You are the SUPERVISOR of a team of design agents. You do not propose designs yourself: you decide "
        "the team. Each worker owns a set of knob families and proposes moves on them only; workers take "
        "turns. A move is kept if it is feasible and more than 0.5% faster.",
        f"Goal: minimize total workload time (LLM + tools). GPU: {ev.gpu_time:.1f} s; tools alone: "
        f"{ev.tools_time:.1f} s (no design can go below that).",
        f"Knob families:\n{fam_lines}",
        f"Current team:\n{cur_team}",
        f"Why you are called: {reason}",
        f"Best and recent evaluations:\n{lines}",
        f"Recent worker proposals:\n{_history_lines(hist, 12)}",
        f"Evaluations left: {ev.budget - ev.used}. Return the team to use from now on: 1-4 workers, each with "
        "a name, a one-sentence brief (strategy) and the families it owns (names from the list; each family "
        "at most once). You may keep, split, merge or reassign workers. Set stop=true only if no team could "
        "improve the best design."])


def eat_team_search(ev, backend, *, worker_backend=None, tb=None, record=None, eps=0.005, stall_k=6,
                    max_restructures=20, seed=0, use_verifier=True, experiment="E5"):
    """Supervisor-managed team without readings or binding-term routing. Returns a RunRecord."""
    space = ev.space
    knobs = all_knobs(space)
    fams = families(knobs)
    start = space_start(space)
    worker_backend = worker_backend or backend
    tb = tb or Toolbox(ev)
    record = record or _record(ev, experiment, "eat_team", seed,
                               {"model": backend.model, "worker_model": worker_backend.model,
                                "budget": ev.budget, "eps": eps, "stall_k": stall_k,
                                "space": type(space).__name__, "use_verifier": use_verifier})
    run = _Run(ev, tb, record, eps, use_verifier)
    record.config["teams"] = []
    team: List = []

    def restructure(cur, reason):
        nonlocal team
        run.calls += 1
        try:
            out, _ = ask(backend, _team_prompt(ev, knobs, start, list(ev.log), _best_feasible(ev), team, run.hist, reason),
                         TEAM_SCHEMA, record, seed=seed * 100000 + run.calls)
        except (BackendError, json.JSONDecodeError) as ex:
            out = {"why": f"backend error: {ex}", "stop": False, "team": []}
        new, owned = [], set()
        for w in out.get("team") or []:
            ks = {}
            for f in w.get("families") or []:
                if f in fams and f not in owned:
                    owned.add(f)
                    ks.update({k: knobs[k] for k in fams[f]})
            if ks:
                new.append((LLMAgent(role=str(w.get("name", "worker")), backend=worker_backend,
                                     brief=str(w.get("brief", "")), name=str(w.get("name", ""))), ks))
        if not new:                       # unusable answer: keep the old team, or one worker owning everything
            new = team or [(LLMAgent(role="worker", backend=worker_backend, brief="Improve the design."), dict(knobs))]
        team = new
        record.config["teams"].append({"eval_n": ev.used, "reason": reason, "why": str(out.get("why", ""))[:300],
                                       "team": [{"name": w.role, "brief": w.brief, "families": sorted(families(k))}
                                                for w, k in team]})
        return bool(out.get("stop"))

    try:
        cur = ev(start)
        if restructure(cur, "start of the run"):
            record.stop = "supervisor stopped at the start"
            return record
        turn, since, restructures = 0, 0, 0
        stopped = set()
        feedback: Dict[int, Dict] = {}
        while True:
            i = turn % len(team)
            w, wk = team[i]
            turn += 1
            if i in stopped:
                if len(stopped) == len(team):
                    since = stall_k
                else:
                    continue
            if since < stall_k:
                run.calls += 1
                ctx = AgentContext(role=w.role, point=cur.point, reading=None, knobs=wk, history=run.hist[-8:],
                                   feedback=feedback.pop(i, None), toolbox=tb, record=record,
                                   budget_left=ev.budget - ev.used, call_id=seed * 100000 + run.calls)
                prop = w.propose(ctx)
                if prop.stop:
                    stopped.add(i)
                    continue
                if not prop.changes:
                    since += 1
                    continue
                cur, outcome, entry = run.try_move(cur, w.role, prop)
                if outcome == "accepted":
                    since, stopped = 0, set()
                else:
                    since += 1
                    feedback[i] = {"outcome": outcome, "violations": entry.get("violations")}
            if since >= stall_k:
                restructures += 1
                if restructures > max_restructures:
                    record.stop = f"{max_restructures} restructures without progress"
                    break
                if restructure(cur, f"no accepted move in the last {since} proposals"):
                    record.stop = "supervisor stopped"
                    break
                since, turn, stopped, feedback = 0, 0, set(), {}
    except BudgetExhausted:
        record.stop = "budget exhausted"
    record.config["llm_calls"] = run.calls
    return record


# ---------------------------------------------------------------------------
# Minimal router for smoke tests of the role agents (C's team_search replaces it)
# ---------------------------------------------------------------------------

def route_of(term, reading):
    """My proposal for C's term -> owner table (AGENT_IFACE.md section 1)."""
    if reading.get("wall_ctx"):
        return "capacity"
    if term in ("host", "launch", "transfer"):
        return "mapping"
    return "hardware"


def route_search(ev, agents: Dict[str, LLMAgent], *, tb=None, record=None, eps=0.005, stop_frac=0.02,
                 max_fails=2, seed=0, experiment="M5-smoke"):
    """Route to the owner of the binding term; after max_fails misses, the next term's owner."""
    space = ev.space
    tb = tb or Toolbox(ev)
    model = next(iter(agents.values())).backend.model
    record = record or _record(ev, experiment, "route_smoke", seed,
                               {"model": model, "budget": ev.budget, "eps": eps, "space": type(space).__name__})
    run = _Run(ev, tb, record, eps, True)
    try:
        cur = ev(space_start(space))
        fails, stopped, feedback = 0, set(), {}
        while True:
            r = tb.read(cur.point)
            if cur.feasible and cur.llm_s <= stop_frac * cur.tools_s:
                record.stop = f"within {stop_frac:.0%} of the Amdahl floor"
                break
            order = []
            for x in r["ranking"]:
                o = route_of(x["term"], r)
                if o not in order and o in agents:
                    order.append(o)
            order += [o for o in agents if o not in order]
            live = [o for o in order if o not in stopped]
            if not live:
                record.stop = "every agent stopped"
                break
            role = live[min(fails // max_fails, len(live) - 1)]
            run.calls += 1
            ctx = AgentContext(role=role, point=cur.point, reading=r, knobs=knobs_of(space, role),
                               history=run.hist[-8:], feedback=feedback.pop(role, None), toolbox=tb,
                               record=record, budget_left=ev.budget - ev.used, call_id=seed * 100000 + run.calls)
            prop = agents[role].propose(ctx)
            if prop.stop:
                stopped.add(role)
                continue
            if not prop.changes:
                fails += 1
                continue
            cur, outcome, entry = run.try_move(cur, role, prop, reading={"binding_term": r["binding_term"],
                                                                        "amdahl_gap": r["amdahl_gap"]})
            if outcome == "accepted":
                fails, stopped = 0, set()
            else:
                fails += 1
                feedback[role] = {"outcome": outcome, "violations": entry.get("violations")}
            if fails >= max_fails * len(live) * 2:
                record.stop = "no agent finds an accepted move"
                break
    except BudgetExhausted:
        record.stop = "budget exhausted"
    record.config["llm_calls"] = run.calls
    return record
