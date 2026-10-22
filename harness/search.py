"""Searchers over the design space, all sharing one Evaluator budget.

domeline_walk   reads which term binds (the Domeline reading) and proposes only
                the moves that act on that term; then shrinks area and effort.
hill_climb      same single-knob moves, but sees only the scalar time
                (the "no readings" ablation).
random_search   uniform random points.
tpe_search      Bayesian optimization (optuna TPE) on the scalar time.
sweep           exhaustive over a subspace.
"""

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .evaluate import BudgetExhausted, Eval
from .space import ACTIVE_IF, ATTN_OPS, EDGE_OPS, KNOBS, START, VECTOR_OPS, WEIGHT_MVMS, key

LOWER_IS_SMALLER = ("xbar_scale", "vfu_units", "star_units", "vfu_bw", "vfu_mem")


def rank(e: Eval):
    """Feasible beats infeasible; then lower time."""
    return (not e.feasible, e.time_s)


def better(a: Eval, b: Eval, tol=0.005):
    if a.feasible != b.feasible:
        return a.feasible
    return a.time_s < b.time_s * (1 - tol)


def step_knob(p, knob, direction):
    opts = KNOBS[knob]
    i = opts.index(p[knob]) + direction
    if 0 <= i < len(opts):
        return {**p, knob: opts[i]}
    return None


@dataclass
class WalkStep:
    reading: str
    move: str
    eval: Eval


@dataclass
class Trail:
    steps: List[WalkStep] = field(default_factory=list)
    stop: str = ""


# ---------------------------------------------------------------------------
# Domeline walk
# ---------------------------------------------------------------------------

def _host_kind_moves(p, kind):
    """Moves that take op `kind` off the host."""
    out = []
    if kind in WEIGHT_MVMS and p["mvm"] == "cpu":
        out.append(("mvm", "xbar"))
    elif kind == "lm_head" and p["lm_head"] == "cpu":
        out.append(("lm_head", "xbar"))
    elif kind in ATTN_OPS and p["attention"] == "cpu":
        out += [("attention", "vfu"), ("attention", "xbar")]
    elif kind == "softmax" and p["softmax"] == "cpu":
        out += [("softmax", "vfu"), ("softmax", "star")]
    elif kind in VECTOR_OPS and p["vector"] == "cpu":
        out += [("vector", "vfu"), ("vector", "star")]
    elif kind in EDGE_OPS and p["edge"] == "cpu":
        out.append(("edge", "vfu"))
    return out


def readings(e: Eval):
    """LLM-side terms, largest first, with their share of LLM time."""
    tot = sum(e.terms.values()) or 1.0
    return sorted(((k, v, v / tot) for k, v in e.terms.items() if v > 0), key=lambda x: -x[1])


def proposals(e: Eval):
    """Ordered (why, point) moves for the binding surface(s) of e."""
    p = e.point
    out = []

    def add(why, knob, value):
        if value is not None and p.get(knob) != value:
            out.append((why, knob, value))

    def up(why, knob):
        q = step_knob(p, knob, +1)
        if q:
            add(why, knob, q[knob])

    def down(why, knob):
        q = step_knob(p, knob, -1)
        if q:
            add(why, knob, q[knob])

    if e.wall_ctx is not None:
        why = f"capacity binds: out of room at {e.wall_ctx} tokens"
        if p["attention"] == "xbar" or "xbar" in (p["mvm"], p["lm_head"]):
            up(why, "xbar_scale")
        down(why, "kv_bits")
        if p["attention"] == "xbar":
            add(why, "attention", "vfu")
        if p["attention"] == "vfu":
            up(why, "vfu_mem")
    if "area" in e.reason:
        why = f"area binds: {e.area_mm2:.0f} mm2 over the cap"
        down(why, "xbar_scale")
        down(why, "kv_bits")
        if p["attention"] == "xbar":
            add(why, "attention", "vfu")

    for term, sec, share in readings(e):
        why = f"{term} binds ({share:.0%} of LLM time)"
        if term == "host":
            host_kinds = sorted(((k, v) for k, v in e.by_kind.items() if k != "link"), key=lambda x: -x[1])
            for kind, _ in host_kinds:
                for knob, val in _host_kind_moves(p, kind):
                    add(f"{why}; {kind} runs on the host", knob, val)
        elif term in ("launch", "transfer"):
            # Every op left on the host costs a round trip per layer.
            for kind in ("softmax", "rmsnorm", "rope", "act", "residual", "attn_score", "sample"):
                for knob, val in _host_kind_moves(p, kind):
                    add(f"{why}; round trips around {kind}", knob, val)
            up(why, "link")
        elif term == "write":
            if p["attention"] == "xbar":
                add(why + "; K appends touch head_dim rows", "xbar_tech", "xbar_transpose")
                add(why, "attention", "vfu")
            down(why, "kv_bits")
            up(why + "; weights reprogrammed", "xbar_scale")
        elif term.startswith("vfu"):
            if p["attention"] == "vfu":
                up(why + "; KV bandwidth", "vfu_bw")
            up(why, "vfu_units")
            if p["softmax"] == "vfu" or p["vector"] == "vfu":
                add(why + "; offload nonlinear ops to the fixed-function unit", "softmax", "star")
        elif term.startswith("softmax"):
            up(why, "star_units")
        elif term.startswith("xbar"):
            if p["attention"] == "xbar":
                add(why + "; attention reads grow with context", "attention", "vfu")
        elif term == "host" or term not in ("link",):
            pass
    seen, uniq = set(), []
    for why, knob, val in out:
        q = {**p, knob: val}
        if key(q) not in seen and key(q) != key(p):
            seen.add(key(q))
            uniq.append((f"{why} -> {knob}={_fmt(val)}", q))
    return uniq


def _fmt(v):
    if isinstance(v, float) and v >= 1e9:
        return f"{v / 1e9:g}G"
    return str(v)


def domeline_walk(ev, start=START, tol=0.005, stop_frac=0.02, shrink=True, shrink_tol=0.01):
    trail = Trail()
    try:
        cur = ev(start)
        trail.steps.append(WalkStep("start: crossbars run the weight MVMs, the host runs the rest", "start", cur))
        while True:
            if cur.feasible and cur.llm_s <= stop_frac * cur.tools_s:
                trail.stop = f"within {stop_frac:.0%} of the Amdahl bound (tools only)"
                break
            moved = False
            for why, q in proposals(cur):
                e = ev(q)
                if better(e, cur, tol):
                    trail.steps.append(WalkStep(why, _diff(cur.point, e.point), e))
                    cur, moved = e, True
                    break
            if not moved:
                trail.stop = "no move on any binding term improves the design"
                break
        if shrink and cur.feasible:
            _shrink(ev, cur, trail, shrink_tol)
    except BudgetExhausted:
        trail.stop = "budget exhausted"
    return trail


def _shrink(ev, cur, trail, tol):
    """Smallest/cheapest design within tol of the best time: walk knobs down while it holds."""
    t_best = cur.time_s
    changed = True
    while changed:
        changed = False
        cands = []
        for knob in LOWER_IS_SMALLER + ("link", "kv_bits"):
            if knob in ACTIVE_IF and not ACTIVE_IF[knob](cur.point):
                continue
            direction = -1 if knob != "kv_bits" else +1
            q = step_knob(cur.point, knob, direction)
            if q:
                cands.append((knob, q))
        for knob, q in cands:
            e = ev(q)
            smaller = (e.area_mm2 < cur.area_mm2 - 1e-9) or (e.effort_months < cur.effort_months)
            if e.feasible and smaller and e.time_s <= t_best * (1 + tol):
                trail.steps.append(WalkStep(f"shrink: {knob} costs area/effort and is not binding",
                                            _diff(cur.point, e.point), e))
                cur, changed = e, True
                break


def _diff(a, b):
    return ", ".join(f"{k}={_fmt(b[k])}" for k in KNOBS if a.get(k) != b.get(k)) or "none"


# ---------------------------------------------------------------------------
# Baselines (scalar objective only)
# ---------------------------------------------------------------------------

def neighbors(p):
    for k, opts in KNOBS.items():
        for v in opts:
            if v != p[k]:
                yield {**p, k: v}


def hill_climb(ev, start=START, seed=0, tol=0.005):
    rng = random.Random(seed)
    try:
        cur = ev(start)
        while True:
            nb = list(neighbors(cur.point))
            rng.shuffle(nb)
            moved = False
            for q in nb:
                e = ev(q)
                if better(e, cur, tol):
                    cur, moved = e, True
                    break
            if not moved:
                break
    except BudgetExhausted:
        pass


def random_point(rng):
    return {k: rng.choice(v) for k, v in KNOBS.items()}


def random_search(ev, seed=0):
    rng = random.Random(seed)
    try:
        while True:
            ev(random_point(rng))
    except BudgetExhausted:
        pass


def tpe_search(ev, seed=0, penalty=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    penalty = penalty or 10 * ev.gpu_time

    def objective(trial):
        p = {k: trial.suggest_categorical(k, [str(o) for o in v]) for k, v in KNOBS.items()}
        p = {k: KNOBS[k][[str(o) for o in KNOBS[k]].index(p[k])] for k in KNOBS}
        e = ev(p)
        return e.time_s if e.feasible else penalty + e.time_s

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    try:
        study.optimize(objective, n_trials=10**9, catch=())
    except BudgetExhausted:
        pass


def sweep(ev, fixed: Dict, knobs: List[str]):
    """Evaluate every combination of `knobs` with the rest fixed. Returns the evals."""
    out = []
    for vals in itertools.product(*(KNOBS[k] for k in knobs)):
        out.append(ev({**fixed, **dict(zip(knobs, vals))}))
    return out
