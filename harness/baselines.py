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


def start_point(space, table=None):
    from .space import START
    p = dict(space.start) if is_large(space) else dict(START)
    for k, opts in (table or {}).items():
        if p.get(k) not in opts:
            p[k] = opts[0]
    return p


def make_verifier(space, sessions, caps=None, sampling="host"):
    """verify(point) -> {"legal", "violations": [(rule, op, reason)], "summary"}, cached by key.

    The same dict as team.point_verifier, so a Toolbox can carry either."""
    from .verify import verify_point
    cache = {}

    def f(point):
        k = space.key(point)
        if k not in cache:
            r = verify_point(space, point, sessions, caps, sampling=sampling)
            cache[k] = {"legal": r.legal, "violations": [tuple(v) for v in r.violations], "summary": r.summary()}
        return cache[k]
    f.sampling = sampling
    return f


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


def _neighbor(rng, table, p, knobs):
    """p with one knob changed to another of its options (None if no knob has a choice)."""
    for _ in range(100):
        k = rng.choice(knobs)
        opts = [o for o in table[k] if o != p.get(k)]
        if opts:
            return {**p, k: rng.choice(opts)}
    return None


def anneal(gate: Gate, table, start, seed=0, t0=0.05, t1=5e-4, penalty_s=None):
    """Simulated annealing on log(objective) over single-knob moves.

    A move that is d (relative, in log time) worse is accepted with probability exp(-d / T);
    T falls geometrically from t0 (a 5% worse move accepted w.p. 1/e) to t1 as the
    evaluation budget is spent. Illegal moves are rejected without an evaluation.
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


def hill_climb(gate: Gate, table, start, seed=0, tol=0.005):
    """First improvement over every single-knob neighbour, in a seeded random order (search.hill_climb)."""
    rng = random.Random(seed)
    try:
        cur = gate.ask(start)
        if cur is None:
            raise ValueError("hill-climb start point is illegal")
        while True:
            nb = [{**cur.point, k: o} for k, opts in table.items() for o in opts if o != cur.point.get(k)]
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
