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

Each input is also a file (M1), loaded by `Inputs.from_files` or, for a case study that
names the four files, `Inputs.load_case` (lib/case/*.yaml):
  workload   lib/workload/<name>.yaml (harness.workload: op graph + trace file)
  library    lib/tech/pim.yaml (any tech YAML files or directories)
  host       lib/host/<name>.yaml (HostProfile.from_file: measured per-kind curves inline)
  objective  lib/objective/<name>.yaml (Objective: latency, area/effort caps, GPU baseline)
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from .evaluate import Eval, Evaluator
from .session import Session
from .space import Space
from .tech import CpuProfile, Tech, load_library

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

    @classmethod
    def from_file(cls, path, tool_scale=None):
        """A host profile file (lib/host/*.yaml): measured curves inline, roofline fallback."""
        d = yaml.safe_load(_path(path).read_text())
        prof = CpuProfile(by_kind={k: [(int(c), float(sec)) for c, sec in v]
                                   for k, v in d["measured"].items()})
        r = d["roofline"]
        params = {"profile": prof, "threads": d["threads"], "ops_per_s": float(r["ops_per_s"]),
                  "mem_bw_Bps": float(r["mem_bw_Bps"]), "op_overhead_s": float(r["op_overhead_s"]),
                  "write_mechanism": True}
        t = Tech(name=d["name"], kind="cpu", supports=tuple(d["supports"]), params=params,
                 source=d.get("source", ""), maturity=d.get("maturity", "projected"),
                 notes=d.get("host", ""), sources=dict(d.get("sources") or {}))
        return cls(tech=t, tool_scale=float(d.get("tool_scale", 1.0) if tool_scale is None else tool_scale))

    def sessions(self, sessions: List[Session]) -> List[Session]:
        """The traced sessions with tool times as they would run on this host."""
        if self.tool_scale == 1.0:
            return sessions
        return [replace(s, turns=[replace(t, t_tool_s=t.t_tool_s * self.tool_scale) for t in s.turns])
                for s in sessions]


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Objective:
    """What is minimized and under which caps (lib/objective/*.yaml).

    metric        latency only: workload time (LLM + tools), inf if infeasible
    area_cap_mm2, effort_cap_months   caps (inf = none)
    baseline      the GPU the speedup is reported against (GpuBaseline arguments)
    tol           best(): the smallest design within tol of the fastest
    """
    metric: str = "latency"
    area_cap_mm2: float = math.inf
    effort_cap_months: float = math.inf
    baseline: Optional[Dict] = None
    tol: float = 0.01

    @classmethod
    def from_file(cls, path):
        d = yaml.safe_load(_path(path).read_text())
        caps = d.get("caps") or {}
        cap = lambda k: math.inf if caps.get(k) is None else float(caps[k])   # noqa: E731
        return cls(metric=d.get("metric", "latency"), area_cap_mm2=cap("area_mm2"),
                   effort_cap_months=cap("effort_months"), baseline=d.get("baseline"),
                   tol=float(d.get("tol", 0.01)))

    def gpu(self):
        from .session import GpuBaseline
        if not self.baseline:
            return None
        b = self.baseline
        return GpuBaseline(_path(b["g1"]), weights=b.get("weights", "Q8_0"), kv=b.get("kv", "f16"),
                           prefill_tok_s=b.get("prefill_tok_s"))


@dataclass
class Inputs:
    model: object                    # workload: ModelSpec or harness.workload.Workload (op graph)
    sessions: List[Session]          # workload: traces
    library: Dict[str, Tech]         # coprocessor technologies
    host: HostProfile
    objective: str = "latency"
    caps: Objective = field(default_factory=Objective)   # caps and baseline (objective file)
    ref_ctx: Optional[int] = None    # crossbar sizing context (default: longest session here)

    @classmethod
    def from_files(cls, workload, library, host, objective, sessions="all"):
        """The four inputs from files. sessions: "all", "search" or "heldout" (the
        workload's split), or a list of Session. ref_ctx is the longest session of the
        whole trace, as in the paper's experiments."""
        from .workload import load_workload
        w = load_workload(workload)
        obj = Objective.from_file(objective)
        allsess = w.trace.sessions()
        if isinstance(sessions, str):
            search, held = w.trace.split(allsess)
            sess = {"all": allsess, "search": search, "heldout": held}[sessions]
        else:
            sess = list(sessions)
        libs = [library] if isinstance(library, (str, Path)) else list(library)
        return cls(model=w, sessions=sess, library=load_library([_path(p) for p in libs]),
                   host=HostProfile.from_file(host), objective=obj.metric, caps=obj,
                   ref_ctx=max(s.max_ctx for s in allsess))

    @classmethod
    def load_case(cls, path, sessions=None):
        """A case-study file (lib/case/*.yaml) naming the four input files."""
        d = yaml.safe_load(_path(path).read_text())
        return cls.from_files(d["workload"], d["library"], d["host"], d["objective"],
                              sessions=sessions or d.get("sessions", "all"))

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
        ref = ref_ctx or self.ref_ctx or max(s.max_ctx for s in self.workload())
        return Space(self.lib(), self.model, ref_ctx=ref, host=self.host.name)

    def evaluator(self, gpu=None, space=None, sessions=None, budget=math.inf, area_cap_mm2=None,
                  effort_cap_months=None) -> Evaluator:
        """Caps default to the objective's; gpu to its baseline."""
        sess = self.workload() if sessions is None else self.host.sessions(sessions)
        gpu = gpu or self.caps.gpu()
        if gpu is None:
            raise ValueError("no GPU baseline: pass gpu or give the objective a baseline")
        return Evaluator(space or self.space(), sess, gpu, budget=budget,
                         area_cap_mm2=self.caps.area_cap_mm2 if area_cap_mm2 is None else area_cap_mm2,
                         effort_cap_months=(self.caps.effort_cap_months if effort_cap_months is None
                                            else effort_cap_months))


@dataclass
class Result:
    best: Optional[Eval]             # the chosen design point and its readings
    design: object                   # the Design (mapping, units, link, precisions) of `best`
    log: List[Eval]                  # every evaluation, in order, with its readings
    trail: object = None             # what the policy returned (the walk: reading per move)


def run(inputs: Inputs, policy: Callable, gpu=None, budget=math.inf, tol=None, **caps) -> Result:
    """Run a search policy on the inputs; the output is a design plus the log."""
    tol = inputs.caps.tol if tol is None else tol
    ev = inputs.evaluator(gpu, budget=budget, **caps)
    trail = policy(ev)
    best = ev.best(tol)
    design = ev.space.to_design(best.point)[0] if best else None
    return Result(best=best, design=design, log=list(ev.log), trail=trail)
