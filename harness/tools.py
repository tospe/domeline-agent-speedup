"""M4 tool API: the functions a search policy (the walk or an LLM agent team) calls.

  evaluate(point)  -> time, speedup, feasibility, term breakdown
  read(point)      -> binding term and ranking, headroom per term, Amdahl gap, capacity wall
  moves(point, term=None) -> candidate moves (the walk's rule table, Alg. 1), optionally for one term
  verify(point)    -> the offload verifier (M2), or None when no verifier is attached
  log(entry)       -> append to the run's learning log

`Toolbox` wraps one Evaluator, so every call goes through its cache, budget and log.
`RunRecord` is the JSON record every run writes (brief §2.3–2.4): config, seed, every
evaluation, every accepted move with the reading behind it, the final design, and cost
(evaluations, wall-clock, LLM tokens per model). `TOOL_SCHEMAS` exposes the same
functions to an LLM.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .evaluate import Eval, Evaluator
from .search import proposals, readings


def _finite(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


def eval_summary(e: Eval) -> Dict:
    return {"n": e.n, "time_s": e.time_s, "llm_s": e.llm_s, "tools_s": e.tools_s,
            "speedup": e.speedup_vs_gpu, "feasible": e.feasible, "reason": e.reason,
            "area_mm2": e.area_mm2, "effort_months": e.effort_months, "terms": e.terms}


@dataclass
class Toolbox:
    ev: Evaluator
    verifier: Optional[Callable] = None     # verify(point) -> dict with "legal", "violations"
    entries: List[Dict] = field(default_factory=list)

    def evaluate(self, point) -> Dict:
        return eval_summary(self.ev(point))

    def read(self, point) -> Dict:
        e = self.ev(point)
        ranking = [{"term": t, "seconds": s, "share": sh} for t, s, sh in readings(e)]
        # Headroom per term: how many times it must shrink before the next term ties it.
        for a, b in zip(ranking, ranking[1:] + [None]):
            a["headroom"] = _finite(a["seconds"] / b["seconds"]) if b else None
        floor = self.ev.amdahl_bound_s
        return {"binding_term": e.binding_term, "ranking": ranking,
                "amdahl_floor_s": floor, "amdahl_gap": e.time_s / floor - 1,
                "llm_share": e.llm_s / e.time_s,
                "wall_ctx": e.wall_ctx, "max_ctx": e.max_ctx, "D_bytes": e.D_bytes,
                "D_wall_bytes": _finite(e.D_wall_bytes),
                "capacity_headroom_bytes": _finite(e.capacity_headroom_bytes),
                "capacity_headroom_tech": e.capacity_headroom_tech,
                "feasible": e.feasible, "reason": e.reason}

    def moves(self, point, term=None) -> List[Dict]:
        """The walk's rule table for this design; with `term`, only the moves that reading names."""
        e = self.ev(point)
        out = []
        for why, q in proposals(e):
            if term is None or why.startswith(term) or f"{term} binds" in why:
                out.append({"why": why, "point": q})
        return out

    def verify(self, point) -> Optional[Dict]:
        return None if self.verifier is None else self.verifier(point)

    def log(self, entry: Dict):
        self.entries.append({"eval_n": self.ev.used, **entry})


@dataclass
class RunRecord:
    """One run's JSON record; the paper's figures are made from these."""
    experiment: str
    method: str
    seed: int
    config: Dict
    t0: float = field(default_factory=time.time)
    accepted: List[Dict] = field(default_factory=list)   # move, reading, verifier result, eval summary
    tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)   # model -> {"in", "out"}
    final: Optional[Dict] = None
    stop: str = ""

    def accept(self, move, reading, e: Eval, verdict=None):
        self.accepted.append({"move": move, "reading": reading, "verify": verdict, **eval_summary(e)})

    def add_tokens(self, model, n_in, n_out):
        t = self.tokens.setdefault(model, {"in": 0, "out": 0})
        t["in"] += n_in
        t["out"] += n_out

    def to_json(self, ev: Evaluator, tb: Optional[Toolbox] = None) -> Dict:
        best = ev.best()
        return {"experiment": self.experiment, "method": self.method, "seed": self.seed,
                "config": self.config, "stop": self.stop,
                "cost": {"evaluations": ev.used, "wall_s": time.time() - self.t0,
                         "eval_wall_s": ev.wall_s, "tokens": self.tokens},
                "evaluations": [{"point": e.point, **eval_summary(e)} for e in ev.log],
                "accepted": self.accepted, "log": tb.entries if tb else [],
                "final": self.final or ({"point": best.point, **eval_summary(best)} if best else None)}

    def write(self, path, ev: Evaluator, tb: Optional[Toolbox] = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(ev, tb), indent=1, default=str))


_POINT = {"type": "object", "description": "a design point: knob -> value (see the space description)"}
TOOL_SCHEMAS = [
    {"name": "evaluate", "description": "Time the design on the workload: total time, speedup vs the GPU, "
     "feasibility, and the time split into named terms. Costs one evaluation unless already seen.",
     "input_schema": {"type": "object", "properties": {"point": _POINT}, "required": ["point"]}},
    {"name": "read", "description": "Domeline readings: binding term and term ranking with headroom, "
     "gap to the Amdahl floor (tools-only time), capacity wall and data held in PIM.",
     "input_schema": {"type": "object", "properties": {"point": _POINT}, "required": ["point"]}},
    {"name": "moves", "description": "Candidate moves from the rule table for this design, optionally only "
     "those for one term.",
     "input_schema": {"type": "object", "properties": {"point": _POINT, "term": {"type": "string"}},
                      "required": ["point"]}},
    {"name": "verify", "description": "Check the design is a legal offload: supported classes and precisions, "
     "transfers at host-PIM crossings, capacity at every state, area and effort caps.",
     "input_schema": {"type": "object", "properties": {"point": _POINT}, "required": ["point"]}},
    {"name": "log", "description": "Append a note (what you tried, what you learned) to the run's learning log.",
     "input_schema": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}},
]
