"""Evaluate design points over a workload (a set of sessions), with a budget and a log.

Each distinct design counts once against the budget, whoever asks for it. The
result carries everything a searcher may read: the scalar objective, and, for
searchers allowed to, the Domeline readings (the term breakdown and per-kind
times) that say which surface binds.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .integrator import Integrator
from .session import GpuBaseline, WorkloadReplayer
from .space import Space, canonical, key


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

    @property
    def objective(self):
        return self.time_s if self.feasible else math.inf


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
        self.wall_s = 0.0

    @property
    def used(self):
        return len(self.log)

    @property
    def amdahl_bound_s(self):
        """No design runs the workload faster than its tools alone."""
        return self.tools_time

    def best(self):
        feas = [e for e in self.log if e.feasible]
        return min(feas, key=lambda e: e.time_s) if feas else None

    def __call__(self, point) -> Eval:
        k = key(point)
        if k in self.cache:
            return self.cache[k]
        if self.used >= self.budget:
            raise BudgetExhausted()
        t0 = time.time()
        p = canonical(point)
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
        e = Eval(point=p, time_s=total, llm_s=llm, tools_s=tools, feasible=not reasons,
                 reason="; ".join(reasons), area_mm2=area, effort_months=effort, terms=terms,
                 by_kind=by_kind, wall_ctx=wall, speedup_vs_gpu=self.gpu_time / total,
                 n=self.used + 1)
        self.cache[k] = e
        self.log.append(e)
        self.wall_s += time.time() - t0
        return e
