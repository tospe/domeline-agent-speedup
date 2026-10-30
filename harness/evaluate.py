"""Evaluate design points over a workload (a set of sessions), with a budget and a log.

Each distinct design counts once against the budget, whoever asks for it. The
result carries everything a searcher may read: the scalar objective, and, for
searchers allowed to, the Domeline readings (the term breakdown and per-kind
times) that say which surface binds.

Readings on an Eval:
  terms, by_kind              the breakdown (which term binds)
  binding_term, second_term,  term headroom: binding / second, the factor by which
  term_headroom               the binding term must fall before another one binds
  D_bytes, D_wall_bytes,      the third axis: data held in PIM at the workload's longest
  D_by_tensor                 context, and the bytes the design can hold
  capacity_headroom_bytes,    capacity headroom: min over techs of (capacity - occupied)
  capacity_headroom_tech      at the longest context (negative = past the wall)
  wall_ctx                    the context where the wall is hit (None = never)
The objective is latency only (time_s, inf if infeasible).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .integrator import Integrator
from .session import GpuBaseline, WorkloadReplayer
from .space import Space


class BudgetExhausted(Exception):
    pass


@dataclass
class Eval:
    point: Dict
    time_s: float                 # workload time: sum of session times (LLM + tools)
    llm_s: float
    tools_s: float
    feasible: bool                # fits area cap and has room for every session
    reason: str
    area_mm2: float
    effort_months: float
    terms: Dict[str, float]       # Domeline readings: LLM-side terms over the workload
    by_kind: Dict[str, float]
    wall_ctx: Optional[int]       # first context where the design runs out of room (None = never)
    speedup_vs_gpu: float
    n: int                        # evaluation index (1-based) in this run
    # Readings added Oct 29 (defaults keep old constructors working).
    binding_term: Optional[str] = None    # largest LLM-side term
    second_term: Optional[str] = None     # next largest (None if only one term is nonzero)
    term_headroom: float = math.inf       # binding / second: factor the binding term must fall
    max_ctx: int = 0                      # longest context in the workload
    D_bytes: float = 0.0                  # data held in PIM at max_ctx (Integrator.data_in_pim)
    D_wall_bytes: float = 0.0             # bytes the design can hold
    D_by_tensor: Dict[str, float] = field(default_factory=dict)   # at max_ctx
    capacity_headroom_bytes: float = math.inf   # min over techs of capacity - occupied, at max_ctx
    capacity_headroom_tech: Optional[str] = None

    @property
    def objective(self):
        return self.time_s if self.feasible else math.inf


def binding_and_next(terms):
    """(binding term, second term, binding / second) over the nonzero terms.

    The ratio is how many times the binding term must shrink before the second one
    binds (inf when only one term is nonzero; (None, None, inf) when none is).
    """
    nz = sorted(((v, k) for k, v in terms.items() if v > 0), reverse=True)
    if not nz:
        return None, None, math.inf
    if len(nz) == 1:
        return nz[0][1], None, math.inf
    return nz[0][1], nz[1][1], nz[0][0] / nz[1][0]


class Evaluator:
    def __init__(self, space: Space, sessions, gpu: GpuBaseline, budget=math.inf,
                 area_cap_mm2=math.inf, effort_cap_months=math.inf):
        self.space = space
        self.sessions = sessions
        self.gpu = gpu
        self.budget = budget
        self.area_cap = area_cap_mm2
        self.effort_cap = effort_cap_months
        self._base = Integrator(space.model, space.lib)
        self.cache: Dict[tuple, Eval] = {}
        self.log: List[Eval] = []
        self.gpu_time = sum(gpu.session_time(s)["total"] for s in sessions)
        self.tools_time = sum(s.tool_time for s in sessions)
        self.max_ctx = max(s.max_ctx for s in sessions)
        self.wall_s = 0.0

    @property
    def used(self):
        return len(self.log)

    @property
    def amdahl_bound_s(self):
        """No design runs the workload faster than its tools alone."""
        return self.tools_time

    def best(self, tol=0.01):
        """The smallest design within `tol` of the fastest feasible one (same rule for every searcher)."""
        feas = [e for e in self.log if e.feasible]
        if not feas:
            return None
        t_min = min(e.time_s for e in feas)
        near = [e for e in feas if e.time_s <= t_min * (1 + tol)]
        return min(near, key=lambda e: (e.area_mm2, e.effort_months, e.time_s))

    def __call__(self, point) -> Eval:
        k = self.space.key(point)
        if k in self.cache:
            return self.cache[k]
        if self.used >= self.budget:
            raise BudgetExhausted()
        t0 = time.time()
        p = self.space.canonical(point)
        design, lib = self.space.to_design(p)
        integ = Integrator(self.space.model, lib)
        integ._ops_cache = self._base._ops_cache
        terms, by_kind = {}, {}
        llm = tools = 0.0
        wall = None
        for r in WorkloadReplayer(integ).run(design, self.sessions):
            llm += r.llm
            tools += r.tools
            for k2, v in r.terms.items():
                terms[k2] = terms.get(k2, 0.0) + v
            for k2, v in r.by_kind.items():
                by_kind[k2] = by_kind.get(k2, 0.0) + v
            if r.over_capacity_at is not None:
                wall = min(wall or r.over_capacity_at, r.over_capacity_at)
        area = self.space.area_mm2(p)
        effort = self.space.effort_months(p)
        reasons = []
        if wall is not None:
            reasons.append(f"out of room at {wall} tokens")
        if area > self.area_cap:
            reasons.append(f"area {area:.0f} > {self.area_cap:.0f} mm2")
        if effort > self.effort_cap:
            reasons.append(f"effort {effort:.0f} > {self.effort_cap:.0f} months")
        total = llm + tools
        res = integ.data_in_pim(design, self.max_ctx)
        cap_tech, cap_room = res.binding_headroom()
        b1, b2, ratio = binding_and_next(terms)
        e = Eval(point=p, time_s=total, llm_s=llm, tools_s=tools, feasible=not reasons,
                 reason="; ".join(reasons), area_mm2=area, effort_months=effort, terms=terms,
                 by_kind=by_kind, wall_ctx=wall, speedup_vs_gpu=self.gpu_time / total,
                 n=self.used + 1, binding_term=b1, second_term=b2, term_headroom=ratio,
                 max_ctx=self.max_ctx, D_bytes=res.D, D_wall_bytes=res.D_wall,
                 D_by_tensor=res.by_tensor(), capacity_headroom_bytes=cap_room,
                 capacity_headroom_tech=cap_tech)
        self.cache[k] = e
        self.log.append(e)
        self.wall_s += time.time() - t0
        return e
