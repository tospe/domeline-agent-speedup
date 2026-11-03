"""Non-LLM baselines on any space (Space or LargeSpace), seeded, sharing one Evaluator budget.

  random_search   uniform points over the knob table
  tpe_search      optuna TPE on the scalar time (as search.tpe_search, any knob table)
  anneal          simulated annealing over single-knob moves (the hill climb's neighbourhood)
  hill_climb      first-improvement single-knob hill climb (as search.hill_climb, any knob table)

All four see only the scalar objective. They propose through a `Gate`, which applies the
verifier the way the team loop does (harness/team.py): an illegal proposal is logged with
the rules it broke and is **not evaluated** (costs no budget). With no verifier every
proposal is evaluated. Caps are the Evaluator's (over the cap = infeasible), also as in the
team. Budget semantics are the Evaluator's: each distinct design counts once.

Also here: `make_verifier` (the team's point_verifier with the sampling rule as an option)
and `VerifiedEvaluator`, a shadow proxy that verifies every evaluated point without
blocking it (H6: what a search does when the verifier is off).
"""

import math
import random
from collections import Counter
from typing import Callable, Dict, List, Optional

from .evaluate import BudgetExhausted, Eval
from .search import better, rank
from .space import KNOBS

# The paper's 14-knob space: the three area knobs pinned at ISAAC's tile.
PINNED_14 = {"xbar_tile": "isaac", "adc_per_ima": 8, "cell_bits": 2}


def is_large(space):
    return hasattr(space, "slots")


def knob_table(space, pin: Optional[Dict] = None, exclude: Optional[Dict] = None) -> Dict[str, list]:
    """knob or slot -> options a baseline may set. pin fixes knobs; exclude removes values."""
    if is_large(space):
        from .large_space import HW_KNOBS
        table = {k: list(KNOBS[k]) for k in HW_KNOBS}
        table.update({s.name: list(s.options) for s in space.slots.values()})
    else:
        table = {k: list(v) for k, v in KNOBS.items()}
    for k, v in (pin or {}).items():
        table[k] = [v]
    for k, vals in (exclude or {}).items():
        if k in table:
            table[k] = [o for o in table[k] if o not in set(vals)]
            if not table[k]:
                raise ValueError(f"exclude leaves no option for {k}")
    return table


def tied_groups(space, table):
    """role -> slot names, for 'tied' moves that set every slot of one role at once (large space only).

    The roles are the small space's placement and precision knobs (mvm, lm_head, attention,
    softmax, vector, edge, kv_bits), so a tied move is exactly a small-space move lifted to
    the large space. Slots whose options the table pins or excludes keep only what is allowed."""
    if not is_large(space):
        return {}
    groups = {}
    for sl in space.slots.values():
        if sl.name in table:
            groups.setdefault(sl.role, []).append(sl.name)
    return groups


def tied_options(table, slots):
    """Values a tied move may set: allowed in every slot of the group."""
    opts = None
    for s in slots:
        opts = list(table[s]) if opts is None else [o for o in opts if o in table[s]]
    return opts or []


def tied_neighbors(p, table, groups):
    """Every tied move from p: (role, value) -> point with all slots of role set to value."""
    for role, slots in groups.items():
        for v in tied_options(table, slots):
            if any(p.get(s) != v for s in slots):
                yield {**p, **{s: v for s in slots}}


def start_point(space, table=None, start=None):
    """The search's first point: START (tied on a large space), or `start`, a small-space
    point (tied on a large space) or a full point of the space."""
    from .space import START
    if start is not None:
        q = {**START, **start}
        # A small point (only small-space knob names; note lm_head is both a knob and a slot) is tied.
        small = all(k in KNOBS for k in start)
        p = space.tie(q) if is_large(space) and small else (q if not is_large(space) else {**space.start, **start})
        p = space.canonical(p)
        for k, opts in (table or {}).items():
            if p.get(k) not in opts:
                raise ValueError(f"start sets {k}={p.get(k)!r}, which the knob table does not allow")
        return p
    return _start_default(space, table)


def _start_default(space, table=None):
    from .space import START
    p = dict(space.start) if is_large(space) else dict(START)
    for k, opts in (table or {}).items():
        if p.get(k) not in opts:
            p[k] = opts[0]
    return p


def make_verifier(space, sessions, caps=None, sampling="host", prefilter=False):
    """verify(point) -> {"legal", "violations": [(rule, op, reason)], "summary"}, cached by key.

    The same dict as team.point_verifier, so a Toolbox can carry either.
    prefilter: run `static_violations` first and, if it finds any, return them without the full
    verify. The legal/illegal verdict is the full verifier's (tests/test_baselines.py checks it
    on random samples); the violation list is then the static subset, marked "prefilter": True.
    """
    from .verify import verify_point
    cache = {}
    f_stats = {"prefiltered": 0, "full": 0}

    def f(point):
        k = space.key(point)
        if k not in cache:
            if prefilter:
                pre = static_violations(space, point, sampling)
                if pre:
                    f_stats["prefiltered"] += 1
                    cache[k] = {"legal": False, "violations": pre, "prefilter": True,
                                "summary": "; ".join(sorted({r for r, _, _ in pre}))}
                    return cache[k]
            f_stats["full"] += 1
            r = verify_point(space, point, sessions, caps, sampling=sampling)
            cache[k] = {"legal": r.legal, "violations": [tuple(v) for v in r.violations], "summary": r.summary()}
        return cache[k]
    f.sampling = sampling
    f.stats = f_stats
    return f


def static_violations(space, point, sampling="host"):
    """The verifier's checks that depend only on the placement: R1.support for every op,
    R1.sampling for `sample`, R2.transfer for the token in/out of PIM (sampling="host").

    Same tests as verify._r1 / _r2_static on the same ops, without building the design's
    library (the area knobs and VFU memory knobs change no tech's `supports` or kind).
    No design, no integrator: a dict lookup per op."""
    from . import verify as Vf
    p = space.canonical(point)
    mapping = space._mapping(p)
    lib = space.lib
    ops = _static_ops(space)
    out = []

    def tech(op):
        for key in (op.name, op.kind, op.cls, "*"):
            if key in mapping:
                return mapping[key]
        return space.host

    for op in ops:
        tname = tech(op)
        t = lib.get(tname)
        if t is None:
            out.append(("R1.support", op.name, f"{tname!r} is not in the library"))
            continue
        if not t.can_run(op.cls):
            out.append(("R1.support", op.name, f"{tname} ({t.kind}) supports {list(t.supports)}, not {op.cls}"))
            continue
        if sampling == "host" and Vf.op_subcls(op) in Vf.EXPLICIT_SUBCLS and not t.on_host \
                and Vf.op_subcls(op) not in t.supports:
            out.append(("R1.sampling", op.name, f"{op.kind} on {tname} (prefilter)"))
    if sampling == "host":
        for op in (ops[0], ops[-1]):
            t = lib.get(tech(op))
            if t is not None and not t.on_host:
                out.append(("R2.transfer", op.name, f"{op.kind} on {t.name}: token crosses the link uncharged (prefilter)"))
    return out


_OPS = {}


def _static_ops(space):
    k = id(space.model)
    if k not in _OPS:
        from .ops import forward_ops
        _OPS[k] = forward_ops(space.model, 1, 2)
    return _OPS[k]


class Gate:
    """Ask the evaluator through the verifier; track proposals, illegal ones and the best so far.

    ask(point) -> Eval, or None if the verifier rejects it (then nothing is evaluated).
    Raises BudgetExhausted when the evaluator's budget is spent, or after max_proposals
    proposals (so a searcher whose proposals are mostly illegal still stops).
    """

    def __init__(self, ev, verifier: Optional[Callable] = None, toolbox=None, record=None,
                 max_proposals=None, label=""):
        self.ev, self.verifier, self.toolbox, self.record = ev, verifier, toolbox, record
        self.max_proposals = max_proposals if max_proposals is not None else (
            20 * ev.budget if math.isfinite(ev.budget) else math.inf)
        self.label = label
        self.proposals = 0
        self.illegal = 0
        self.rules = Counter()
        self.best: Optional[Eval] = None

    def ask(self, point) -> Optional[Eval]:
        if self.proposals >= self.max_proposals:
            raise BudgetExhausted()
        self.proposals += 1
        verdict = None
        if self.verifier is not None:
            v = self.verifier(point)
            verdict = {"legal": v["legal"], "violations": v["violations"]}
            if not v["legal"]:
                self.illegal += 1
                rules = sorted({r for r, _, _ in v["violations"]})
                self.rules.update(rules)
                if self.toolbox is not None:
                    self.toolbox.log({"kind": "proposal", "outcome": "illegal", "agent": self.label,
                                      "rules": rules})
                return None
        e = self.ev(point)
        if self.best is None or rank(e) < rank(self.best):
            prev, self.best = self.best, e
            if self.record is not None:
                changes = {k: v for k, v in e.point.items() if prev is None or prev.point.get(k) != v}
                self.record.accept({"changes": changes if prev else "start", "why": f"{self.label}: new best",
                                    "agent": self.label}, None, e, verdict)
        return e


def objective(e: Eval, penalty_s):
    """Scalar the baselines minimize: time, plus a penalty when infeasible (as search.tpe_search)."""
    return e.time_s if e.feasible else penalty_s + e.time_s


# ---------------------------------------------------------------------------
# The four baselines. Each runs until the budget (or the proposal cap) is spent.
# ---------------------------------------------------------------------------

def random_search(gate: Gate, table, seed=0):
    rng = random.Random(seed)
    try:
        while True:
            gate.ask({k: rng.choice(v) for k, v in table.items()})
    except BudgetExhausted:
        pass


def tpe_search(gate: Gate, table, seed=0, penalty_s=None, illegal_factor=2.0):
    """optuna TPE over the knob table. An illegal proposal is told penalty * illegal_factor
    without an evaluation (TPE needs a value; it learns to avoid the region)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    penalty_s = penalty_s or 10 * gate.ev.gpu_time
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    names = list(table)
    try:
        while True:
            tr = study.ask()
            p = {k: table[k][tr.suggest_categorical(k, list(range(len(table[k]))))] for k in names}
            e = gate.ask(p)
            study.tell(tr, illegal_factor * penalty_s if e is None else objective(e, penalty_s))
    except BudgetExhausted:
        pass


def tied_tpe(gate: Gate, space, table, seed=0, tied_frac=0.5, tol=0.005, **kw):
    """TPE over the tied subspace (hardware knobs + one value per role), then a per-slot phase.

    TPE has no notion of a move, so "tied subspace + per-slot moves" is two phases: TPE over
    the ~17 tied knobs for tied_frac of the budget, then hill_climb with single-slot and
    tied moves from TPE's best feasible point for the rest."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    groups = tied_groups(space, table)
    hw = {k: v for k, v in table.items() if k not in space.slots}
    roles = {r: tied_options(table, sl) for r, sl in groups.items()}
    names = list(hw) + list(roles)
    opts = {**hw, **roles}
    penalty_s = kw.get("penalty_s") or 10 * gate.ev.gpu_time
    budget = gate.ev.budget
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    try:
        while gate.ev.used < tied_frac * budget:
            tr = study.ask()
            c = {k: opts[k][tr.suggest_categorical(k, list(range(len(opts[k]))))] for k in names}
            p = {k: c[k] for k in hw}
            for r, sl in groups.items():
                p.update({s: c[r] for s in sl})
            e = gate.ask(p)
            study.tell(tr, 2 * penalty_s if e is None else objective(e, penalty_s))
    except BudgetExhausted:
        return
    best = gate.best
    if best is None or not best.feasible:
        return
    hill_climb(gate, table, best.point, seed=seed, tol=tol, groups=groups)


def _neighbor(rng, table, p, knobs):
    """p with one knob changed to another of its options (None if no knob has a choice)."""
    for _ in range(100):
        k = rng.choice(knobs)
        opts = [o for o in table[k] if o != p.get(k)]
        if opts:
            return {**p, k: rng.choice(opts)}
    return None


def anneal(gate: Gate, table, start, seed=0, t0=0.05, t1=5e-4, penalty_s=None, groups=None, p_tied=0.5):
    """Simulated annealing on log(objective) over single-knob moves (plus tied moves if groups).

    A move that is d (relative, in log time) worse is accepted with probability exp(-d / T);
    T falls geometrically from t0 (a 5% worse move accepted w.p. 1/e) to t1 as the
    evaluation budget is spent. Illegal moves are rejected without an evaluation.
    groups (tied_groups): with probability p_tied the move is a tied one (one role, every slot);
    0.5 because the tied moves are few (~15) against ~1,300 single-slot moves.
    """
    rng = random.Random(seed)
    penalty_s = penalty_s or 10 * gate.ev.gpu_time
    knobs = [k for k, v in table.items() if len(v) > 1]
    budget = gate.ev.budget if math.isfinite(gate.ev.budget) else 10_000
    try:
        cur = gate.ask(start)
        if cur is None:
            raise ValueError("annealing start point is illegal")
        f_cur = math.log(objective(cur, penalty_s))
        while True:
            q = None
            if groups and rng.random() < p_tied:
                tied = list(tied_neighbors(cur.point, table, groups))
                q = rng.choice(tied) if tied else None
            if q is None:
                q = _neighbor(rng, table, cur.point, knobs)
            if q is None:
                break
            e = gate.ask(q)
            if e is None:
                continue
            f = math.log(objective(e, penalty_s))
            T = t0 * (t1 / t0) ** min(1.0, gate.ev.used / budget)
            if f <= f_cur or rng.random() < math.exp(-(f - f_cur) / T):
                cur, f_cur = e, f
    except BudgetExhausted:
        pass


def hill_climb(gate: Gate, table, start, seed=0, tol=0.005, groups=None):
    """First improvement over every single-knob neighbour, in a seeded random order (search.hill_climb).

    groups (tied_groups) adds the tied moves to the neighbourhood, shuffled in with the rest.
    tol=0 accepts any strict improvement."""
    rng = random.Random(seed)
    try:
        cur = gate.ask(start)
        if cur is None:
            raise ValueError("hill-climb start point is illegal")
        while True:
            nb = [{**cur.point, k: o} for k, opts in table.items() for o in opts if o != cur.point.get(k)]
            if groups:
                nb += list(tied_neighbors(cur.point, table, groups))
            rng.shuffle(nb)
            moved = False
            for q in nb:
                e = gate.ask(q)
                if e is not None and better(e, cur, tol):
                    cur, moved = e, True
                    break
            if not moved:
                break
    except BudgetExhausted:
        pass


# ---------------------------------------------------------------------------
# Shadow verification (verifier off)
# ---------------------------------------------------------------------------

class VerifiedEvaluator:
    """An Evaluator proxy that verifies every point it evaluates, without blocking any.

    Everything else (budget, cache, log, best, space, ...) is the wrapped evaluator's, so it
    drops in wherever an Evaluator goes (team_search, the walk, the baselines). `verdicts`
    maps eval number n -> the verifier's dict for that point.
    """

    def __init__(self, ev, verifier):
        object.__setattr__(self, "_ev", ev)
        object.__setattr__(self, "_verifier", verifier)
        object.__setattr__(self, "verdicts", {})

    def __getattr__(self, k):
        return getattr(self._ev, k)

    def __setattr__(self, k, v):
        setattr(self._ev, k, v)

    def __call__(self, point):
        n0 = self._ev.used
        e = self._ev(point)
        if self._ev.used > n0:
            self.verdicts[e.n] = self._verifier(point)
        return e

    def summary(self, accepted_ns=(), final_n=None):
        """Illegal evaluations, rules hit, and which accepted / final designs were illegal."""
        illegal = {n: v for n, v in self.verdicts.items() if not v["legal"]}
        rules = Counter(r for v in illegal.values() for r in sorted({x[0] for x in v["violations"]}))
        acc_illegal = [n for n in accepted_ns if n in illegal]
        return {"evaluated": len(self.verdicts), "illegal_evaluated": len(illegal), "rules": dict(rules),
                "accepted": len(list(accepted_ns)), "accepted_illegal": acc_illegal,
                "final_legal": None if final_n is None else final_n not in illegal}
