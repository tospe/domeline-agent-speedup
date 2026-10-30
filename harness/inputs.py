"""The harness interface: four explicit inputs, a pluggable search policy, one output.

  workload     traces (Session list: per-turn context, prefill, generated tokens,
               tool time tau) and the op graph source (ModelSpec from config.json)
  library      coprocessor technologies (lib/tech/pim.yaml, or any YAML in that format)
  host         HostProfile: the cost model of every op left on the CPU (lib/tech/host.yaml,
               whose `cpu` entry points at the measured per-op profile
               results/cpu/cpu_ops.tsv), and the host the tool times tau were measured on
  objective    latency only: workload time (LLM + tools), inf if infeasible

A policy is any callable policy(evaluator) that asks the evaluator for designs until it
is done or the budget runs out (search.domeline_walk, hill_climb, random_search,
tpe_search, agent.agent_search). The output is the chosen design and the log of every
evaluation with its readings (and, for the walk, the reading behind every move).

This module only composes existing pieces: building an Evaluator through Inputs gives
the same results as building it by hand from load_library("lib/tech").
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .evaluate import Eval, Evaluator
from .session import Session
from .space import Space
from .tech import Tech, load_library

ROOT = Path(__file__).resolve().parent.parent
OBJECTIVES = ("latency",)


@dataclass(frozen=True)
class HostProfile:
    """The host CPU as one input.

    tech        the CPU's cost model (kind cpu): per-op times from its measured profile
                (params.profile_tsv, params.threads), a roofline fallback for ops the
                profile does not cover (ops_per_s, mem_bw_Bps) and a per-op overhead.
    tool_scale  tool times tau are measured on this host and stored in the traces; a
                different host can rescale them (1.0 = the traced host, unchanged).
    """
    tech: Tech
    tool_scale: float = 1.0

    @property
    def name(self):
        return self.tech.name

    @classmethod
    def load(cls, path=ROOT / "lib/tech/host.yaml", name="cpu", profile_tsv=None, threads=None,
             tool_scale=1.0):
        t = load_library(path)[name]
        if t.kind != "cpu":
            raise ValueError(f"{name!r} in {path} is a {t.kind}, not a host cpu")
        params = dict(t.params)
        if profile_tsv is not None:
            params["profile_tsv"] = str(profile_tsv)
        if threads is not None:
            params["threads"] = threads
        if params != t.params:
            t = replace(t, params=params)
        return cls(tech=t, tool_scale=tool_scale)

    def sessions(self, sessions: List[Session]) -> List[Session]:
        """The traced sessions with tool times as they would run on this host."""
        if self.tool_scale == 1.0:
            return sessions
        return [replace(s, turns=[replace(t, t_tool_s=t.t_tool_s * self.tool_scale) for t in s.turns])
                for s in sessions]


@dataclass
class Inputs:
    model: object                    # workload: ModelSpec, the op graph source
    sessions: List[Session]          # workload: traces
    library: Dict[str, Tech]         # coprocessor technologies
    host: HostProfile
    objective: str = "latency"

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective {self.objective!r}: only {OBJECTIVES} is implemented")
        other = self.library.get(self.host.name)
        if other is not None and other != self.host.tech:
            raise ValueError(f"library has a different tech named {self.host.name!r}")

    def lib(self) -> Dict[str, Tech]:
        """Coprocessor library plus the host entry: what the integrator times against."""
        return {**self.library, self.host.name: self.host.tech}

    def workload(self) -> List[Session]:
        return self.host.sessions(self.sessions)

    def space(self, ref_ctx=None) -> Space:
        ref = ref_ctx or max(s.max_ctx for s in self.workload())
        return Space(self.lib(), self.model, ref_ctx=ref, host=self.host.name)

    def evaluator(self, gpu, space=None, sessions=None, budget=math.inf, area_cap_mm2=math.inf,
                  effort_cap_months=math.inf) -> Evaluator:
        sess = self.workload() if sessions is None else self.host.sessions(sessions)
        return Evaluator(space or self.space(), sess, gpu, budget=budget,
                         area_cap_mm2=area_cap_mm2, effort_cap_months=effort_cap_months)


@dataclass
class Result:
    best: Optional[Eval]             # the chosen design point and its readings
    design: object                   # the Design (mapping, units, link, precisions) of `best`
    log: List[Eval]                  # every evaluation, in order, with its readings
    trail: object = None             # what the policy returned (the walk: reading per move)


def run(inputs: Inputs, policy: Callable, gpu, budget=math.inf, tol=0.01, **caps) -> Result:
    """Run a search policy on the inputs; the output is a design plus the log."""
    ev = inputs.evaluator(gpu, budget=budget, **caps)
    trail = policy(ev)
    best = ev.best(tol)
    design = ev.space.to_design(best.point)[0] if best else None
    return Result(best=best, design=design, log=list(ev.log), trail=trail)
