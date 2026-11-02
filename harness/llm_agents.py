"""M5: LLM agents for the team loop, and the LLM-based E5 baselines.

Interface with the loop (harness/team.py, Agent C): docs/harness-dev/AGENT_IFACE.md. An agent
has a `role` and `propose(ctx: team.AgentContext) -> [team.Move]`; it sees only its own knobs,
never evaluates (the loop owns the budget) and adds every LLM call's tokens to `self.record`
(team_search sets it), per model ID.

  Backends     ClaudeCLI("claude-sonnet-5-5" | "claude-haiku-5-5")  the `claude` CLI, headless
               LocalLlama()     mini-coder-4b on llama-server (traces/serve.sh), temperature 0
               FakeBackend(..)  canned responses, for tests (no network)
  LLMAgent     one role agent (mapping / hardware / capacity), structured JSON out
  Baselines    single_agent_search   one LLM, every knob, no readings (agent.py's ablation,
                                     generalized to LargeSpace)
               eat_team_search       an Evolving-Agent-Teams-style team: an LLM supervisor sets
                                     and restructures a team of workers; no readings, no routing
The role agents run inside team.team_search (C's loop): team_search(ev, tb, llm_team(backend)).

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

from . import team as T
from .agent import KNOB_HELP, LOCAL_URL
from .evaluate import BudgetExhausted, Eval
from .search import better
from .space import KNOBS, START
from .team import AgentContext, Move
from .tools import RunRecord, Toolbox

SONNET, HAIKU, LOCAL = "claude-sonnet-5-5", "claude-haiku-5-5", "mini-coder-4b-Q8_0"

SYSTEM = """You are part of a team designing a processing-in-memory (PIM) coprocessor that sits next \
to a CPU and runs an LLM coding agent's decode faster than an RTX 3080 GPU. Tools always run on the \
CPU. Every design is evaluated by an analytical model over recorded agent sessions; evaluations are \
scarce, so each proposal should be the most informative next step. Answer only with the JSON object \
requested."""


# ---------------------------------------------------------------------------
# Loop interface: harness/team.py (docs/harness-dev/AGENT_IFACE.md, written by C)
# ---------------------------------------------------------------------------

ROLES = T.ROLES
owner_of = T.owner_of


def all_knobs(space) -> Dict[str, list]:
    """Every knob of a Space or LargeSpace -> its options (the union of team.own_knobs)."""
    out = {}
    for r in ROLES:
        out.update(T.own_knobs(space, r))
    return out


def knobs_of(space, role) -> Dict[str, list]:
    return T.own_knobs(space, role)


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


def _normalize_pattern(pat):
    """Spellings small models produce from the prompt's notation: L<l>.op, L<*>.op, L<3>.op, L*l*.op."""
    s = str(pat).strip()
    s = re.sub(r"^L<\s*l?\s*>\.", "L*.", s)
    s = re.sub(r"^L<([^>]*)>\.", r"L\1.", s)
    s = re.sub(r"^L\*l\*\.|^Ll\.", "L*.", s)
    return s


def expand_changes(knobs: Dict[str, list], changes: Dict) -> Dict:
    """Concrete knob -> option from the agent's changes; patterns L*.fam, La-b.fam, Ln.fam."""
    if not isinstance(changes, dict):
        raise ValueError("changes must be an object of knob -> value")
    out = {}
    for pat, val in changes.items():
        if pat in knobs:
            targets = [pat]
        else:
            m = _PATTERN.match(_normalize_pattern(pat))
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
# Role agent (team.Agent: role, propose(ctx) -> [Move])
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

_MOVE = {"type": "object",
         "properties": {"why": {"type": "string"},
                        "changes": {"type": "object", "additionalProperties": {"type": "string"}}},
         "required": ["why", "changes"]}
AGENT_SCHEMA = {
    "type": "object",
    "properties": {"pass": {"type": "boolean"}, "moves": {"type": "array", "items": _MOVE}},
    "required": ["pass", "moves"],
}


def _strs(ch):
    return {k: opt_str(v) for k, v in (ch or {}).items()}


def _compact(ch, n=6):
    ch = _strs(ch)
    if len(ch) <= n:
        return ch
    return {**dict(list(ch.items())[:4]), "...": f"{len(ch) - 4} more"}


def _history_lines(hist, n):
    """team.py history entries: move, why, agent, outcome, time_s, violations."""
    out = []
    for h in hist[-n:]:
        mv = h.get("move")
        line = f"- {h.get('agent', '?')}: {json.dumps(_compact(mv) if isinstance(mv, dict) else mv)} -> {h.get('outcome')}"
        if h.get("time_s") is not None:
            line += f" ({h['time_s']:.1f} s)"
        if h.get("violations"):
            line += f"; violations: {[list(v)[:2] for v in h['violations'][:3]]}"
        out.append(line)
    return "\n".join(out) or "(none yet)"


@dataclass
class LLMAgent:
    """One role agent: reading + own knobs + rule-table candidates + history in, up to max_moves moves out.

    record   the RunRecord its tokens go to (team_search sets it when None)
    brief    replaces ROLE_BRIEF[role] (the EAT team's workers)
    """
    role: str
    backend: object
    record: Optional[RunRecord] = None
    brief: str = ""
    max_moves: int = 3
    history_n: int = 8
    retries: int = 1                # corrective retries for malformed or out-of-scope answers
    seed: int = 0
    once_per_round: bool = True     # team_search asks each owner once per agenda term; the prompt already
                                    # carries the whole reading, so answer once per (design, round)
    calls: List[Dict] = field(default_factory=list)   # per call: model, tokens, outcome (for the report)
    _asked: set = field(default_factory=set, repr=False)

    def prompt(self, ctx) -> str:
        brief = self.brief or ROLE_BRIEF.get(self.role, f"You are the {self.role} agent.")
        parts = [brief,
                 "Goal: minimize total workload time (LLM + tools) with a feasible, legal design. A move is "
                 "accepted only if it passes the offload verifier, is feasible and is more than 0.5% faster "
                 "than the current design.",
                 f"Current design, your knobs only (change nothing else):\n{describe_knobs(ctx.point, ctx.knobs)}",
                 f"Readings of the current design: {describe_reading(ctx.reading)}"]
        if ctx.term:
            parts.append(f"The supervisor routed this reading item to you: {ctx.term}.")
        if ctx.rule_moves:
            parts.append("Rule-table candidates for it (you may pick, combine or do better):\n" + "\n".join(
                f"- {m.why}: {json.dumps(_compact(m.changes))}" for m in ctx.rule_moves[:6]))
        parts.append(f"Recent proposals by the team:\n{_history_lines(ctx.history, self.history_n)}")
        if ctx.feedback:
            parts.append("Your last move was ILLEGAL; the verifier says: "
                         + json.dumps([list(v) for v in ctx.feedback[:6]], default=str)[:800])
        if ctx.round:
            parts.append(f"This is round {ctx.round} on this design: earlier moves were not accepted; try "
                         "something different.")
        parts.append(
            f"Evaluations left: {ctx.budget_left}. Propose up to {self.max_moves} moves, best first; each move's "
            "`changes` maps your knobs to new values (strings from the options). For per-layer knobs you may "
            "write a group: \"L*.<op>\" (every layer), \"L<a>-<b>.<op>\" (layers a..b) or \"L<n>.<op>\". One "
            "sentence of `why` per move. Set pass=true with no moves only if no change of your knobs can help.")
        return "\n\n".join(parts)

    def propose(self, ctx) -> List[Move]:
        if self.once_per_round and not ctx.feedback:
            k = (json.dumps(ctx.point, sort_keys=True, default=str), ctx.round)
            if k in self._asked:
                return []
            self._asked.add(k)
        base = self.prompt(ctx)
        err = None
        for attempt in range(self.retries + 1):
            prompt = base if err is None else base + f"\n\nYour previous answer was invalid: {err}. Answer again."
            seed = self.seed * 1_000_000 + len(self.calls)
            call = {"model": getattr(self.backend, "model", "?"), "role": self.role, "term": ctx.term,
                    "round": ctx.round, "tokens_in": 0, "tokens_out": 0, "seed": seed}
            self.calls.append(call)
            try:
                out, usages = ask(self.backend, prompt, AGENT_SCHEMA, self.record, seed=seed)
            except (BackendError, json.JSONDecodeError) as ex:
                err = call["error"] = f"backend error: {ex}"
                continue
            call["tokens_in"] = sum(u.n_in for u in usages)
            call["tokens_out"] = sum(u.n_out for u in usages)
            try:
                if out.get("pass") and not out.get("moves"):
                    call["outcome"] = "pass"
                    return []
                moves, seen = [], set()
                for mv in (out.get("moves") or [])[:self.max_moves]:
                    ch = expand_changes(ctx.knobs, mv.get("changes") or {})
                    ch = {k: v for k, v in ch.items() if ctx.point.get(k) != v}
                    key = tuple(sorted((k, str(v)) for k, v in ch.items()))
                    if ch and key not in seen:
                        seen.add(key)
                        moves.append(Move(ch, why=str(mv.get("why", ""))[:300], agent=self.role))
                if not moves:
                    raise ValueError("no move changes any of your knobs from the current design")
                call["outcome"] = f"{len(moves)} moves"
                return moves
            except (ValueError, TypeError, AttributeError) as ex:
                err = call["error"] = str(ex)
                call["raw"] = json.dumps(out, default=str)[:600]
        call["outcome"] = "invalid"
        return []


def llm_team(backend, record=None, seed=0, **kw) -> Dict[str, LLMAgent]:
    """role -> LLMAgent for team.team_search."""
    return {r: LLMAgent(r, backend, record=record, seed=seed, **kw) for r in ROLES}


# ---------------------------------------------------------------------------
# Shared accept step for the baselines (outcomes named as in team.py)
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

    def check(self, base: Eval, who, changes, why):
        """Verify and evaluate base + changes. Returns (Eval or None, outcome, entry); outcome is
        illegal | duplicate | None (evaluated, not judged yet)."""
        q = {**base.point, **changes}
        entry = {"move": _strs(changes), "why": why, "agent": who, "outcome": None, "time_s": None,
                 "violations": None}
        verdict = _verdict(self.tb, q) if self.use_verifier else None
        entry["verify"] = verdict
        if verdict is not None and not verdict.get("legal", True):
            entry.update(outcome="illegal", violations=[list(v) for v in verdict.get("violations", [])])
            return None, "illegal", entry
        before = self.ev.used
        e = self.ev(q)
        entry.update(time_s=e.time_s, speedup=e.speedup_vs_gpu, eval_n=e.n)
        if self.ev.used == before:
            return e, "duplicate", entry
        return e, None, entry

    def try_move(self, cur: Eval, who, changes, why):
        """Accept cur + changes if more than eps faster. Returns (current, outcome, entry)."""
        e, outcome, entry = self.check(cur, who, changes, why)
        if outcome is None:
            if better(e, cur, self.eps):
                outcome = "accepted"
                self.record.accept({"changes": entry["move"], "why": why, "agent": who}, None, e, entry["verify"])
                cur = e
            else:
                outcome = "slower" if e.feasible else "infeasible"
        entry["outcome"] = outcome
        self.log(entry)
        return cur, outcome, entry

    def log(self, entry):
        self.hist.append(entry)
        self.tb.log({"kind": "proposal", **entry})


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
                tb.log({"kind": "invalid", "agent": "single", "error": str(ex)[:300],
                        "raw": json.dumps(out, default=str)[:600]})
                repeats += 1
                if repeats >= max_repeats:
                    record.stop = "agent gave invalid designs"
                    break
                continue
            e, outcome, entry = run.check(base, "single", changes, str(out.get("why", ""))[:500])
            entry["base"] = base.n
            if outcome is None:              # a new design: accepted iff it beats the best so far
                outcome = "accepted" if better(e, best, eps) else ("slower" if e.feasible else "infeasible")
                if outcome == "accepted":
                    record.accept({"changes": entry["move"], "why": entry["why"], "agent": "single",
                                   "base": base.n}, None, e, entry["verify"])
                if e.feasible and (not best.feasible or e.time_s < best.time_s):
                    best = e
                evals.append(e)
                repeats = 0
            entry["outcome"] = outcome
            run.log(entry)
            if outcome == "duplicate":
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
    cur_team = "\n".join(f"- {w.role}: {sorted(families(wk))} — {w.brief}" for w, wk in team) or "(no team yet)"
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
                    max_restructures=10, seed=0, use_verifier=True, experiment="E5"):
    """Supervisor-managed team without readings or binding-term routing. Returns a RunRecord.

    Workers take turns (no routing); each sees only its knobs and the scalar outcomes of the
    team's proposals. The supervisor (an LLM call) sets the team at the start and whenever
    stall_k proposals in a row are not accepted, or every worker passes."""
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

    def restructure(reason):
        nonlocal team
        run.calls += 1
        try:
            out, _ = ask(backend, _team_prompt(ev, knobs, start, list(ev.log), _best_feasible(ev), team, run.hist,
                                               reason), TEAM_SCHEMA, record, seed=seed * 1_000_000 + run.calls)
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
                new.append((LLMAgent(role=str(w.get("name") or "worker"), backend=worker_backend, record=record,
                                     brief=str(w.get("brief", "")), seed=seed), ks))
        if not new:      # unusable answer: keep the old team, or one worker owning everything
            new = team or [(LLMAgent(role="worker", backend=worker_backend, record=record,
                                     brief="Improve the design.", seed=seed), dict(knobs))]
        team = new
        record.config["teams"].append({"eval_n": ev.used, "reason": reason, "why": str(out.get("why", ""))[:300],
                                       "team": [{"name": w.role, "brief": w.brief, "families": sorted(families(k))}
                                                for w, k in team]})
        return bool(out.get("stop"))

    try:
        cur = ev(start)
        reason = "start of the run"
        for _ in range(max_restructures + 1):
            if restructure(reason):
                record.stop = "supervisor stopped"
                break
            since, rnd = 0, 0
            while since < stall_k:
                passed = 0
                for w, wk in team:
                    run.calls += 1
                    feedback = None
                    last = next((h for h in reversed(run.hist) if h["agent"] == w.role), None)
                    if last and last["outcome"] == "illegal":
                        feedback = last["violations"]
                    ctx = AgentContext(role=w.role, term="", point=cur.point, reading=None, knobs=wk,
                                       rule_moves=[], history=run.hist[-8:], feedback=feedback, round=rnd,
                                       budget_left=ev.budget - ev.used)
                    moves = w.propose(ctx)
                    if not moves:
                        passed += 1
                        since += 1
                    for m in moves:
                        cur, outcome, _ = run.try_move(cur, w.role, m.changes, m.why)
                        if outcome == "accepted":
                            since, rnd = 0, 0
                            break
                        since += 1
                    if since >= stall_k:
                        break
                if passed == len(team):
                    break
                rnd += 1
            reason = ("every worker passed" if since < stall_k
                      else f"no accepted move in the last {since} proposals")
        else:
            record.stop = f"{max_restructures} restructures"
    except BudgetExhausted:
        record.stop = "budget exhausted"
    record.config["llm_calls"] = run.calls
    return record
