"""The agent-team loop (brief §M5): a supervisor routes the binding term to the agent that owns it.

  team_search(ev, toolbox, agents, ...)   the loop: read -> route -> propose -> verify ->
                                          evaluate -> accept if > eps faster -> log; stall ->
                                          restructure; stop near the Amdahl floor or at budget
  rule_agents(space)                      deterministic mode: the walk's rule table split by owner
                                          (on Space it reproduces search.domeline_walk exactly)

Agents implement docs/harness-dev/AGENT_IFACE.md: `propose(AgentContext) -> [Move]`. Ownership
is enforced by the loop: a move touching another agent's knob is rejected and logged.

On a LargeSpace the rule table is lifted to slots (`large_proposals`): a small-space move such as
"softmax -> star" becomes one move per layer's softmax slot. The supervisor's restructure turns
those into coordinated moves: the same move in every layer (`tie`) and an op moved with its chain
neighbors (`window`), tried when the agents stall, or first (`coordinate="first"`).

Canonicalization: only the point being evaluated is canonicalized (by the Evaluator). Moves are
applied to the current canonical point; nothing here canonicalizes a point under another mapping.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .evaluate import BudgetExhausted, Eval
from .search import LOWER_IS_SMALLER, _fmt, better, proposals, readings
from .space import ACTIVE_IF, ATTN_OPS, EDGE_OPS, KNOBS, START, VECTOR_OPS, WEIGHT_MVMS

# ---------------------------------------------------------------------------
# Ownership and routing
# ---------------------------------------------------------------------------

OWNER_KNOBS = {
    "mapping": ("mvm", "lm_head", "attention", "softmax", "vector", "edge"),
    "hardware": ("xbar_tech", "vfu_units", "vfu_bw", "star_units", "link", "xbar_tile", "adc_per_ima", "cell_bits"),
    "capacity": ("xbar_scale", "vfu_mem", "kv_bits"),
}
ROLES = tuple(OWNER_KNOBS)
_KNOB_OWNER = {k: r for r, ks in OWNER_KNOBS.items() for k in ks}

TERM_OWNERS = {
    "capacity": ("capacity", "mapping"),          # the wall: arrays, KV bits, KV to the vector unit, its memory
    "area": ("hardware", "capacity", "mapping"),  # tile, ADCs; array count, KV bits; attention off the crossbar
    "host": ("mapping",),                         # take ops off the host
    "launch": ("mapping", "hardware"),            # remove round trips; a faster link
    "transfer": ("mapping", "hardware"),
    "write": ("hardware", "mapping", "capacity"), # transposed writes; attention elsewhere; KV bits; arrays
    "vfu": ("hardware", "mapping"),               # units, bandwidth; ops to the fixed-function unit
    "star": ("hardware",),                        # units
    "xbar": ("hardware", "mapping"),              # ADC sharing; attention off the crossbar
    "unit": ("capacity", "hardware"),             # any other tech: more units, the other MVM tech
}


def owner_of(knob, space=None):
    """The agent that owns a knob (a small-space knob or a large-space slot)."""
    if knob in _KNOB_OWNER:
        return _KNOB_OWNER[knob]
    slots = getattr(space, "slots", None)
    if slots and knob in slots:
        return "capacity" if slots[knob].role == "kv_bits" else "mapping"
    raise KeyError(knob)


def term_class(term):
    """Reading item -> TERM_OWNERS key."""
    if term in TERM_OWNERS:
        return term
    if term.startswith("vfu"):
        return "vfu"
    if term.startswith("softmax"):
        return "star"
    if term.startswith("xbar"):
        return "xbar"
    return "unit"


def why_term(why):
    """The reading a rule-table reason names: 'capacity binds: ...' -> 'capacity', 'host binds (...)' -> 'host'."""
    return why.split(" binds", 1)[0]


def agenda(e: Eval):
    """Reading items in the order the supervisor routes them: wall, area, then terms by size."""
    items = []
    if e.wall_ctx is not None:
        items.append("capacity")
    if "area" in e.reason:
        items.append("area")
    items += [t for t, _, _ in readings(e)]
    return items


# ---------------------------------------------------------------------------
# Interface (docs/harness-dev/AGENT_IFACE.md)
# ---------------------------------------------------------------------------

@dataclass
class Move:
    changes: Dict[str, Any]
    why: str = ""
    priority: float = math.inf
    agent: str = ""


@dataclass
class AgentContext:
    role: str
    term: str
    point: Dict
    reading: Dict
    knobs: Dict[str, list]
    rule_moves: List[Move]
    history: List[Dict]
    feedback: Optional[List] = None
    round: int = 0
    budget_left: float = math.inf


@dataclass
class SupervisorContext:
    """What a supervisor sees when the loop escalates (AGENT_IFACE.md, H9)."""
    trigger: str                 # "stall" | "exhausted"
    point: Dict
    reading: Dict
    knobs: Dict[str, list]       # every knob/slot -> options (a supervisor move may span roles)
    owners: Dict[str, str]       # knob/slot -> owning role
    coordinated: List[Move]      # the deterministic restructure candidates (tie / window)
    singles: List[Move]          # this round's role-agent candidates
    history: List[Dict]
    accepted: List[Dict]
    escalation: int = 0
    feedback: Optional[List] = None
    budget_left: float = math.inf


try:
    from typing import Protocol
except ImportError:          # pragma: no cover
    Protocol = object


class Supervisor(Protocol):
    name: str
    record: Any

    def restructure(self, ctx: SupervisorContext) -> List[Move]: ...


def all_knobs(space):
    """Every knob/slot -> options (the union of the roles' knobs)."""
    out = {}
    for r in ROLES:
        out.update(own_knobs(space, r))
    return out


def _token_delta(before, after):
    out = {}
    for m, t in after.items():
        b = before.get(m, {})
        d = {k: v - b.get(k, 0) for k, v in t.items()}
        if any(d.values()):
            out[m] = d
    return out


# ---------------------------------------------------------------------------
# Space helpers (small Space or LargeSpace)
# ---------------------------------------------------------------------------

def is_large(space):
    return hasattr(space, "slots")


def own_knobs(space, role):
    """Knob -> options an agent may change."""
    out = {k: KNOBS[k] for k in OWNER_KNOBS[role]}
    if is_large(space):
        for k in ("mvm", "lm_head", "attention", "softmax", "vector", "edge", "kv_bits"):
            out.pop(k, None)
        for s in space.slots.values():
            if owner_of(s.name, space) == role:
                out[s.name] = list(s.options)
    return out


def _active(space, p, knob):
    """ACTIVE_IF for the knob on p (large points through their small proxy)."""
    if knob not in ACTIVE_IF:
        return True
    return ACTIVE_IF[knob](space.proxy(p) if is_large(space) else p)


def _step(p, knob, direction):
    opts = KNOBS[knob]
    i = opts.index(p[knob]) + direction
    return opts[i] if 0 <= i < len(opts) else None


# ---------------------------------------------------------------------------
# Rule tables
# ---------------------------------------------------------------------------

def small_proposals(e, space):
    """search.proposals as (why, changes) on a small-space point."""
    p = e.point
    return [(why, {k: v for k, v in q.items() if p.get(k) != v}) for why, q in proposals(e)]


_TARGETS = {"mvm": ("xbar",), "lm_head": ("xbar",), "attention": ("vfu", "xbar"), "softmax": ("vfu", "star"),
            "vector": ("vfu", "star"), "edge": ("vfu",)}


def _kind_role(kind):
    if kind in WEIGHT_MVMS:
        return "mvm"
    if kind == "lm_head":
        return "lm_head"
    if kind in ATTN_OPS:
        return "attention"
    if kind == "softmax":
        return "softmax"
    if kind in VECTOR_OPS:
        return "vector"
    if kind in EDGE_OPS:
        return "edge"
    return None


def _slot_kinds(space):
    if not hasattr(space, "_slot_kinds"):
        kind = {o.name: o.kind for o in space._ref_ops}
        space._slot_kinds = {s.name: {kind[o] for o in s.ops} for s in space.slots.values()}
        order = {o.name: i for i, o in enumerate(space._ref_ops)}
        space._slot_order = sorted((s for s in space.slots.values() if s.role != "kv_bits"),
                                   key=lambda s: order[s.ops[0]])
    return space._slot_kinds


def large_proposals(e, space):
    """The walk's rule table on a LargeSpace point, one move per slot (in op order).

    Mirrors search.proposals rule for rule; a rule on a role knob ("attention -> vfu") becomes
    a move per slot of that role that satisfies the rule's condition on that slot.
    """
    p = e.point
    kinds = _slot_kinds(space)
    order = space._slot_order
    out = []

    def slots(role=None, kind=None, value=None):
        return [s.name for s in order if (role is None or s.role == role)
                and (kind is None or kind in kinds[s.name]) and (value is None or p[s.name] == value)]

    def per_slot(why, names, to):
        for n in names:
            if to in space.slots[n].options and p[n] != to:
                out.append((why, {n: to}))

    def hw(why, knob, direction=None, value=None):
        v = value if value is not None else _step(p, knob, direction)
        if v is not None and p[knob] != v:
            out.append((why, {knob: v}))

    def kv_down(why):
        for s in space.kv_slots:
            v = _step({"kv_bits": p[s.name]}, "kv_bits", +1)
            if v is not None:
                out.append((why, {s.name: v}))

    xbar_used = space._used(p, "xbar")
    attn_xbar = slots("attention", value="xbar")
    attn_vfu = slots("attention", value="vfu")

    def host_kind(why, kind):
        role = _kind_role(kind)
        if role is None:
            return
        for to in _TARGETS[role]:
            per_slot(why, slots(role, kind=kind, value="cpu"), to)

    if e.wall_ctx is not None:
        why = f"capacity binds: out of room at {e.wall_ctx} tokens"
        if attn_xbar or xbar_used:
            hw(why, "xbar_scale", +1)
        kv_down(why)
        per_slot(why, attn_xbar, "vfu")
        if attn_vfu:
            hw(why, "vfu_mem", +1)
    if "area" in e.reason:
        why = f"area binds: {e.area_mm2:.0f} mm2 over the cap"
        if xbar_used:
            hw(why + "; eDRAM buffer and bus are 47% of a tile", "xbar_tile", value="lean")
            hw(why + "; ADCs are 31% of a tile", "adc_per_ima", +1)
        hw(why, "xbar_scale", -1)
        kv_down(why)
        per_slot(why, attn_xbar, "vfu")

    for term, sec, share in readings(e):
        why = f"{term} binds ({share:.0%} of LLM time)"
        if term == "host":
            for kind, _ in sorted(((k, v) for k, v in e.by_kind.items() if k != "link"), key=lambda x: -x[1]):
                host_kind(f"{why}; {kind} runs on the host", kind)
        elif term in ("launch", "transfer"):
            for kind in ("softmax", "rmsnorm", "rope", "act", "residual", "attn_score", "sample"):
                host_kind(f"{why}; round trips around {kind}", kind)
            hw(why, "link", +1)
        elif term == "write":
            if attn_xbar:
                hw(why + "; K appends touch head_dim rows", "xbar_tech", value="xbar_transpose")
                per_slot(why, attn_xbar, "vfu")
            kv_down(why)
            hw(why + "; weights reprogrammed", "xbar_scale", +1)
        elif term.startswith("vfu"):
            if attn_vfu:
                hw(why + "; KV bandwidth", "vfu_bw", +1)
            hw(why, "vfu_units", +1)
            per_slot(why + "; softmax to the fixed-function unit", slots("softmax", value="vfu"), "star")
            per_slot(why + "; norms and activations to the fixed-function unit", slots("vector", value="vfu"), "star")
        elif term.startswith("softmax"):
            hw(why, "star_units", +1)
        elif term.startswith("xbar"):
            hw(why + "; read cycle set by ADC sharing", "adc_per_ima", -1)
            per_slot(why + "; attention reads grow with context", attn_xbar, "vfu")
        elif term not in ("host", "link"):
            if xbar_used:
                hw(why + "; more units", "xbar_scale", +1)
            for alt in KNOBS["xbar_tech"]:
                hw(why + "; other MVM tech", "xbar_tech", value=alt)
    return out


def rule_table(e, space):
    """(why, changes) candidates for e, deduplicated by the design they lead to (first kept)."""
    raw = large_proposals(e, space) if is_large(space) else small_proposals(e, space)
    seen, out = {space.key(e.point)}, []
    for why, ch in raw:
        k = space.key({**e.point, **ch})
        if k not in seen:
            seen.add(k)
            out.append((why, ch))
    return out


# ---------------------------------------------------------------------------
# Deterministic agents
# ---------------------------------------------------------------------------

class RuleAgent:
    """Proposes the rule table's moves for the routed term and its own knobs, once per design."""

    def __init__(self, role):
        self.role = role

    def propose(self, ctx: AgentContext) -> List[Move]:
        if ctx.round > 0 or ctx.feedback is not None:
            return []
        return list(ctx.rule_moves)


def rule_agents(space=None):
    return {r: RuleAgent(r) for r in ROLES}


# ---------------------------------------------------------------------------
# Supervisor restructure (coordinated moves)
# ---------------------------------------------------------------------------

def coordinated_moves(space, p, singles: List[Move], window=1):
    """Coordinated versions of single-slot moves (large space; none on the small space).

    tie:    every single-slot move of one rule (same reason, same target value, same role)
            applied in every slot it names, as one move
    window: a slot moved together with its `window` neighbors on each side in the op chain
            (same layer, placement slots that allow the target)
    """
    if not is_large(space):
        return []
    _slot_kinds(space)
    groups: Dict[tuple, Dict] = {}
    for m in singles:
        if len(m.changes) != 1:
            continue
        (k, v), = m.changes.items()
        if k not in space.slots:
            continue
        g = groups.setdefault((m.why, v, space.slots[k].role), {})
        g[k] = v
    out = []
    for (why, v, role), ch in groups.items():
        if len(ch) > 1:
            out.append(Move(dict(ch), why=f"restructure tie: {why} -> {role}={v} in {len(ch)} slots",
                            agent="supervisor"))
    order = [s.name for s in space._slot_order]
    pos = {n: i for i, n in enumerate(order)}
    for m in singles:
        if len(m.changes) != 1:
            continue
        (k, v), = m.changes.items()
        if k not in pos:
            continue
        i, layer = pos[k], space.slots[k].layer
        ch = {k: v}
        for j in list(range(i - window, i)) + list(range(i + 1, i + window + 1)):
            if 0 <= j < len(order):
                n = order[j]
                s = space.slots[n]
                if s.layer == layer and v in s.options and p[n] != v:
                    ch[n] = v
        if len(ch) > 1:
            out.append(Move(ch, why=f"restructure window: {m.why} -> {', '.join(ch)}={v}", agent="supervisor"))
    return out


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

AREA_MOVES = (("xbar_tile", "lean"), ("cell_bits", 4), ("adc_per_ima", +1), ("xbar_scale", -1), ("kv_bits", +1))


def _repair_area(ev, e, max_steps=6):
    """search._repair_area on either space: apply the area move that costs least time until it fits."""
    space = ev.space
    paid = []
    for _ in range(max_steps):
        if e.feasible or "area" not in e.reason:
            break
        cands = []
        for knob, v in AREA_MOVES:
            if not _active(space, e.point, knob):
                continue
            if knob == "kv_bits" and is_large(space):
                ch = {s.name: _step({"kv_bits": e.point[s.name]}, "kv_bits", +1) for s in space.kv_slots}
                ch = {k: x for k, x in ch.items() if x is not None}
                if not ch:
                    continue
            elif isinstance(v, int) and knob != "cell_bits":
                x = _step(e.point, knob, v)
                if x is None:
                    continue
                ch = {knob: x}
            else:
                ch = {knob: v}
            q = {**e.point, **ch}
            if space.key(q) == space.key(e.point):
                continue
            r = ev(q)
            if r.area_mm2 < e.area_mm2 - 1e-9 and r.wall_ctx is None:
                cands.append((not r.feasible, r.time_s, knob, r))
        if not cands:
            break
        _, _, knob, e = min(cands, key=lambda c: (c[0], c[1]))
        paid.append(f"{knob}={_fmt(e.point[knob]) if knob in e.point else 'kv8'}")
    return e, ", ".join(paid)


def _shrink(ev, cur, tol, on_accept):
    """search._shrink on either space: walk area/effort knobs down while the time holds."""
    space = ev.space
    t_best = cur.time_s
    changed = True
    while changed:
        changed = False
        cands = []
        for knob in LOWER_IS_SMALLER + ("link", "kv_bits", "adc_per_ima"):
            if knob == "kv_bits" and is_large(space):
                for s in space.kv_slots:
                    x = _step({"kv_bits": cur.point[s.name]}, "kv_bits", +1)
                    if x is not None:
                        cands.append((s.name, {s.name: x}))
                continue
            if not _active(space, cur.point, knob):
                continue
            direction = +1 if knob in ("kv_bits", "adc_per_ima") else -1
            x = _step(cur.point, knob, direction)
            if x is not None:
                cands.append((knob, {knob: x}))
        for knob, ch in cands:
            e = ev({**cur.point, **ch})
            smaller = (e.area_mm2 < cur.area_mm2 - 1e-9) or (e.effort_months < cur.effort_months)
            if e.feasible and smaller and e.time_s <= t_best * (1 + tol):
                on_accept(Move(ch, why=f"shrink: {knob} costs area/effort and is not binding",
                               agent=owner_of(knob, space)), e, None)
                cur, changed = e, True
                break
    return cur


def point_verifier(space, sessions, caps=None):
    """verify(point) -> {"legal", "violations": [(rule, op, reason)], "summary"} for a Toolbox (M2 verifier)."""
    from .verify import verify_point
    cache = {}

    def f(point):
        k = space.key(point)
        if k not in cache:
            r = verify_point(space, point, sessions, caps)
            cache[k] = {"legal": r.legal, "violations": [tuple(v) for v in r.violations], "summary": r.summary()}
        return cache[k]
    return f


@dataclass
class TeamResult:
    final: Eval
    stop: str
    accepted: List[Dict] = field(default_factory=list)     # move, agent, term, reading, eval n, time
    proposals: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)  # outcome -> count
    restructures: int = 0
    escalations: List[Dict] = field(default_factory=list)   # supervisor calls (H9)


def team_search(ev, toolbox, agents: Dict[str, Any], *, start=None, eps=0.005, stall_k=None,
                stall_gap=0.02, stop_frac=0.02, budget=None, record=None, retry_budget=2,
                coordinate="stall", window=1, exclude: Optional[Dict[str, set]] = None,
                shrink=True, shrink_tol=0.01, history_len=20, max_rounds=50,
                supervisor=None, max_escalations=5) -> TeamResult:
    """The team loop (brief §M5 steps 1-5). Returns the final design and what happened.

    ev, toolbox  the Evaluator (its budget counts evaluations) and the M4 Toolbox (read, verify, log)
    agents       role -> agent (AGENT_IFACE.md); rule_agents() for the deterministic mode
    start        first design (default START, tied on a LargeSpace)
    eps          accept a move only if it is more than eps faster (the walk's tol)
    stall_k      stall = this many proposals without an accept while the Amdahl gap > stall_gap;
                 the supervisor then tries coordinated moves (default: never stall mid-round)
    stop_frac    stop when the LLM time is within stop_frac of the tools time (the Amdahl floor)
    budget       evaluations (sets ev.budget)
    coordinate   "stall" (coordinated moves on stall and at the end of a fruitless round),
                 "first" (before the agents' single moves every round), "never"
    exclude      knob or slot -> values no move may set (e.g. {"edge": {"vfu"}, "sample": {"vfu"}})
    supervisor   H9: a Supervisor (`.restructure(SupervisorContext) -> [Move]`) or a callable; called on
                 stall (instead of the deterministic restructure) and before stopping for lack of moves,
                 at most max_escalations times. None = the M5 loop, unchanged.
    """
    space = ev.space
    if budget is not None:
        ev.budget = budget
    if start is None:
        start = space.start if is_large(space) else START
    exclude = exclude or {}
    for a in agents.values():
        if getattr(a, "record", "absent") is None:
            a.record = record
    if record is not None and record.clock is None:
        record.clock = lambda: ev.used
    sup_fn = None
    if supervisor is not None:
        if getattr(supervisor, "record", "absent") is None:
            supervisor.record = record
        sup_fn = getattr(supervisor, "restructure", supervisor)
        sup_name = getattr(supervisor, "name", getattr(sup_fn, "__name__", "supervisor"))
        sup_rec = getattr(supervisor, "record", None) or record
        knobs_all = all_knobs(space)
        owners_all = {k: owner_of(k, space) for k in knobs_all}
    history: List[Dict] = []
    res = TeamResult(final=None, stop="")

    def note(outcome, m, e=None, violations=None):
        res.proposals += outcome != "duplicate"
        if outcome != "accepted":
            res.rejected[outcome] = res.rejected.get(outcome, 0) + 1
        h = {"move": m.changes, "why": m.why, "agent": m.agent, "outcome": outcome,
             "time_s": e.time_s if e else None, "violations": violations}
        history.append(h)
        del history[:-history_len]
        toolbox.log({"kind": "proposal", **h})

    def on_accept(m, e, verdict, term=None, reading=None):
        entry = {"move": m.changes, "why": m.why, "agent": m.agent, "term": term, "eval_n": e.n,
                 "time_s": e.time_s, "speedup": e.speedup_vs_gpu}
        res.accepted.append(entry)
        if record is not None:
            record.accept({"changes": m.changes, "why": m.why, "agent": m.agent, "term": term},
                          reading, e, verdict)
        toolbox.log({"kind": "accept", **entry})

    def excluded(m):
        return any(v in exclude.get(k, ()) for k, v in m.changes.items())

    def sup_active():
        return sup_fn is not None and len(res.escalations) < max_escalations

    def escalate(trigger, cur, reading, singles, coord, tried):
        """One supervisor escalation: ask, verify, evaluate in order; retry on illegal moves.
        Returns (Move, Eval, verdict, "supervisor") on an accept, else None."""
        import copy
        import time as _time
        t0 = _time.time()
        before = copy.deepcopy(sup_rec.tokens) if sup_rec is not None else {}
        entry = {"n": len(res.escalations), "trigger": trigger, "eval_n": ev.used, "supervisor": sup_name,
                 "binding_term": reading.get("binding_term"), "time_s": cur.time_s, "moves": [], "outcomes": [],
                 "accepted": False, "calls": 0}
        res.escalations.append(entry)
        res.restructures += 1
        ctx = SupervisorContext(trigger=trigger, point=cur.point, reading=reading, knobs=knobs_all,
                                owners=owners_all, coordinated=list(coord), singles=list(singles),
                                history=list(history), accepted=[dict(a) for a in res.accepted],
                                escalation=entry["n"], budget_left=ev.budget - ev.used)
        toolbox.log({"kind": "escalation", "trigger": trigger, "n": entry["n"]})
        out = None
        try:
            for attempt in range(retry_budget + 1):
                entry["calls"] += 1
                moves = list(sup_fn(ctx) or [])
                if not moves:
                    break
                illegal = None
                for m in moves:
                    m.agent = "supervisor"
                    entry["moves"].append(dict(m.changes))
                    if excluded(m):
                        note("excluded", m)
                        entry["outcomes"].append("excluded")
                        continue
                    q = {**cur.point, **m.changes}
                    k = space.key(q)
                    if k in tried or k == space.key(cur.point):
                        note("duplicate", m)
                        entry["outcomes"].append("duplicate")
                        continue
                    tried.add(k)
                    verdict = None
                    if toolbox.verifier is not None:
                        v = toolbox.verify(q)
                        verdict = {"legal": v["legal"], "violations": v.get("violations", [])}
                        if not v["legal"]:
                            note("illegal", m, violations=verdict["violations"])
                            entry["outcomes"].append("illegal")
                            illegal = verdict["violations"]
                            continue
                    e = ev(q)
                    why = m.why
                    if not e.feasible and e.wall_ctx is None and e.time_s < cur.time_s * (1 - eps):
                        e, paid = _repair_area(ev, e)
                        if paid:
                            why += "; area paid by " + paid
                    if better(e, cur, eps):
                        entry["outcomes"].append("accepted")
                        entry["accepted"] = True
                        out = (Move(dict(m.changes), why=why, agent="supervisor"), e, verdict, "supervisor")
                        return out
                    note("slower" if e.feasible else "infeasible", m, e)
                    entry["outcomes"].append("slower" if e.feasible else "infeasible")
                if illegal is None:
                    break
                ctx.feedback = illegal
                ctx.history = list(history)
                ctx.budget_left = ev.budget - ev.used
            return out
        finally:
            entry["wall_s"] = _time.time() - t0
            entry["tokens"] = _token_delta(before, sup_rec.tokens) if sup_rec is not None else {}
            if record is not None:
                record.escalations.append(entry)

    cur = None
    try:
        cur = ev(start)
        res.accepted.append({"move": "start", "agent": "supervisor", "eval_n": cur.n, "time_s": cur.time_s,
                             "speedup": cur.speedup_vs_gpu})
        rnd = 0
        since_accept = 0
        while True:
            if cur.feasible and cur.llm_s <= stop_frac * cur.tools_s:
                res.stop = f"within {stop_frac:.0%} of the Amdahl bound (tools only)"
                break
            reading = toolbox.read(cur.point)
            gap = reading["amdahl_gap"]
            table = rule_table(cur, space) if rnd == 0 else []
            # Ask the owners of each agenda term, binding term first.
            cands: List[tuple] = []          # (priority, seq, Move, term)
            seq = 0
            asked = []
            for term in agenda(cur):
                for role in TERM_OWNERS[term_class(term)]:
                    if role not in agents or (term, role) in asked:
                        continue
                    asked.append((term, role))
                    rule_moves = [Move(ch, why=why, priority=i, agent=role) for i, (why, ch) in enumerate(table)
                                  if why_term(why) == term and all(owner_of(k, space) == role for k in ch)]
                    ctx = AgentContext(role=role, term=term, point=cur.point, reading=reading,
                                       knobs=own_knobs(space, role), rule_moves=rule_moves, history=list(history),
                                       round=rnd, budget_left=ev.budget - ev.used)
                    for m in agents[role].propose(ctx) or []:
                        m.agent = role
                        cands.append((m.priority, seq, m, term, ctx))
                        seq += 1
            cands.sort(key=lambda c: (c[0], c[1]))
            singles = [c[2] for c in cands]
            coord = [] if coordinate == "never" else coordinated_moves(space, cur.point, singles, window)
            queue = [(m, "restructure", None) for m in coord] if coordinate == "first" else []
            queue += [(m, term, ctx) for _, _, m, term, ctx in cands]
            pending_coord = list(coord) if coordinate == "stall" else []
            tried = set()
            accepted = None
            i = 0
            while i < len(queue) and accepted is None:
                m, term, ctx = queue[i]
                i += 1
                if m.agent != "supervisor" and not all(owner_of(k, space) == m.agent for k in m.changes):
                    note("ownership", m)
                    continue
                if excluded(m):
                    note("excluded", m)
                    continue
                q = {**cur.point, **m.changes}
                k = space.key(q)
                if k in tried or k == space.key(cur.point):
                    note("duplicate", m)
                    continue
                tried.add(k)
                verdict = None
                if toolbox.verifier is not None:
                    v = toolbox.verify(q)
                    verdict = {"legal": v["legal"], "violations": v.get("violations", [])}
                    if not v["legal"]:
                        note("illegal", m, violations=verdict["violations"])
                        # Return the violated rule to the agent; it may retry.
                        if ctx is not None and m.agent != "supervisor":
                            for _ in range(retry_budget):
                                ctx.feedback = verdict["violations"]
                                ctx.history = list(history)
                                retry = agents[m.agent].propose(ctx) or []
                                if not retry:
                                    break
                                for r in retry:
                                    r.agent = m.agent
                                queue[i:i] = [(r, term, ctx) for r in retry]
                                break
                        continue
                e = ev(q)
                why = m.why
                if not e.feasible and e.wall_ctx is None and e.time_s < cur.time_s * (1 - eps):
                    e, paid = _repair_area(ev, e)
                    if paid:
                        why += "; area paid by " + paid
                if better(e, cur, eps):
                    accepted = (Move(dict(m.changes), why=why, agent=m.agent), e, verdict, term)
                    break
                note("slower" if e.feasible else "infeasible", m, e)
                since_accept += 1
                if stall_k and since_accept >= stall_k and gap > stall_gap and sup_active():
                    since_accept = 0
                    accepted = escalate("stall", cur, reading, singles, coord, tried)
                    if accepted is not None:
                        break
                    continue
                if stall_k and since_accept >= stall_k and gap > stall_gap and pending_coord:
                    res.restructures += 1
                    queue[i:i] = [(c, "restructure", None) for c in pending_coord]
                    pending_coord = []
                    since_accept = 0
            if sup_active():
                pending_coord = []        # the supervisor replaces the deterministic restructure (until its cap)
            if accepted is None and pending_coord:
                # The agents' moves are exhausted: the supervisor restructures before giving up.
                res.restructures += 1
                for c in pending_coord:
                    q = {**cur.point, **c.changes}
                    if excluded(c) or space.key(q) in tried:
                        continue
                    tried.add(space.key(q))
                    verdict = None
                    if toolbox.verifier is not None:
                        v = toolbox.verify(q)
                        verdict = {"legal": v["legal"], "violations": v.get("violations", [])}
                        if not v["legal"]:
                            note("illegal", c, violations=verdict["violations"])
                            continue
                    e = ev(q)
                    if better(e, cur, eps):
                        accepted = (c, e, verdict, "restructure")
                        break
                    note("slower" if e.feasible else "infeasible", c, e)
            if accepted is not None:
                m, e, verdict, term = accepted
                note("accepted", m, e)
                on_accept(m, e, verdict, term, reading)
                cur = e
                rnd = 0
                since_accept = 0
                continue
            if cands and rnd < max_rounds and any(not isinstance(a, RuleAgent) for a in agents.values()):
                rnd += 1          # non-rule agents may propose something new next round
                continue
            if sup_active():
                accepted = escalate("exhausted", cur, reading, singles, coord, tried)
                if accepted is not None:
                    m, e, verdict, term = accepted
                    note("accepted", m, e)
                    on_accept(m, e, verdict, term, reading)
                    cur = e
                    rnd = 0
                    since_accept = 0
                    continue
            res.stop = "no move on any binding term improves the design"
            break
        if shrink and cur.feasible:
            cur = _shrink(ev, cur, shrink_tol, on_accept)
    except BudgetExhausted:
        res.stop = "budget exhausted"
    res.final = cur
    if record is not None:
        record.stop = res.stop
    return res


# ---------------------------------------------------------------------------
# Ensemble (H9 R5): independent runs, best of
# ---------------------------------------------------------------------------

def _final_key(rec):
    f = rec.get("final") or {}
    return f.get("time_s", math.inf) if f.get("feasible") else math.inf


def merge_records(records: List[Dict], seeds=None) -> Dict:
    """Merge independent members' RunRecord JSONs into one record (AGENT_IFACE.md, H9 ensemble).

    Evaluations are interleaved round-robin (members advance in lockstep), renumbered 1..N and
    tagged with `member`; token_log eval_n values are mapped to that merged count."""
    if not records:
        raise ValueError("no members")
    best = min(range(len(records)), key=lambda i: (_final_key(records[i]), i))
    # Round-robin interleave; pos[(member, local n)] = merged n.
    evals, pos = [], {}
    queues = [list(r.get("evaluations", [])) for r in records]
    idx = [0] * len(records)
    while any(idx[i] < len(q) for i, q in enumerate(queues)):
        for i, q in enumerate(queues):
            if idx[i] < len(q):
                e = dict(q[idx[i]])
                idx[i] += 1
                local = e.get("n", idx[i])
                e["member_n"] = local
                e["member"] = i
                e["n"] = len(evals) + 1
                pos[(i, local)] = e["n"]
                evals.append(e)

    def merged_n(i, local):
        if local is None:
            return None
        if (i, local) in pos:
            return pos[(i, local)]
        # A call before the member's first evaluation (or between): the last merged n at or before it.
        prior = [pos[(i, n)] for (j, n) in pos if j == i and n <= local]
        return max(prior) if prior else 0

    tokens, token_log = {}, []
    for i, r in enumerate(records):
        c = r.get("cost", {})
        for m, t in (c.get("tokens") or {}).items():
            acc = tokens.setdefault(m, {})
            for k, v in t.items():
                acc[k] = acc.get(k, 0) + v
        for t in c.get("token_log") or []:
            token_log.append({**t, "member": i, "member_eval_n": t.get("eval_n"),
                              "eval_n": merged_n(i, t.get("eval_n"))})
    token_log.sort(key=lambda t: (math.inf if t["eval_n"] is None else t["eval_n"], t["member"]))
    walls = [r.get("cost", {}).get("wall_s", 0.0) for r in records]
    out = {"experiment": records[0].get("experiment"),
           "method": f"ensemble[{records[0].get('method')}]",
           "seed": list(seeds) if seeds is not None else [r.get("seed") for r in records],
           "config": {"members": len(records), "member_config": records[0].get("config")},
           "stop": f"best of {len(records)}: member {best} ({records[best].get('stop')})",
           "cost": {"evaluations": sum(r.get("cost", {}).get("evaluations", 0) for r in records),
                    "wall_s": max(walls), "wall_s_sum": sum(walls),
                    "eval_wall_s": sum(r.get("cost", {}).get("eval_wall_s", 0.0) for r in records),
                    "tokens": tokens, "token_log": token_log},
           "evaluations": evals,
           "accepted": records[best].get("accepted", []),
           "escalations": [dict(x, member=i) for i, r in enumerate(records) for x in r.get("escalations", [])],
           "log": [],
           "final": records[best].get("final"),
           "best_member": best,
           "members": records}
    if all("cost_usd" in r for r in records):
        cu = [r["cost_usd"] for r in records]
        if all(isinstance(x, dict) for x in cu):
            tot = {}
            for x in cu:
                for k, v in x.items():
                    if isinstance(v, (int, float)):
                        tot[k] = tot.get(k, 0.0) + v
            out["cost_usd"] = tot
        else:
            out["cost_usd"] = sum(cu)
    return out


def ensemble(run_one: Callable[[int], Dict], seeds: List[int], *, workers=None, executor="thread") -> Dict:
    """Run independent members in parallel (run_one(seed) -> RunRecord JSON) and merge them, best of."""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    seeds = list(seeds)
    Pool = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    with Pool(max_workers=workers or len(seeds)) as pool:
        records = list(pool.map(run_one, seeds))
    return merge_records(records, seeds)
