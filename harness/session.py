"""Session replay: a trace of turns -> session time on a design, with a breakdown.

A turn is: prefill the new tokens (the previous output plus the tool's
output), decode `gen_tokens` tokens while the context grows by one each,
then run the tool on the host. A session's time is the sum over turns, since
the agent waits for each part before the next.

Trace records (one per turn) use the fields logged by traces/run_swebench.py:
ctx_start, prefill_tokens, cached_tokens, gen_tokens, t_tool_s, and the GPU's
measured t_prefill_s / t_decode_s.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Turn:
    ctx_start: int          # context length when decode starts (prefill included)
    prefill_tokens: int
    gen_tokens: int
    t_tool_s: float
    t_prefill_gpu_s: Optional[float] = None
    t_decode_gpu_s: Optional[float] = None


@dataclass
class Session:
    name: str
    turns: List[Turn]
    meta: Dict = field(default_factory=dict)

    @property
    def max_ctx(self):
        return max(t.ctx_start + t.gen_tokens for t in self.turns)

    @property
    def tool_time(self):
        return sum(t.t_tool_s for t in self.turns)

    @property
    def gen_tokens(self):
        return sum(t.gen_tokens for t in self.turns)


def load_trace(path, name=None, exec_overhead_s=0.0):
    """Read a trace.jsonl. exec_overhead_s (podman exec cost) is subtracted from tool times."""
    turns = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r.get("prefill_tokens") is None:
            continue
        t_tool = max(0.0, r["t_tool_s"] - (exec_overhead_s if r.get("command") else 0.0))
        turns.append(Turn(ctx_start=r["ctx_start"], prefill_tokens=r["prefill_tokens"],
                          gen_tokens=r["gen_tokens"] or 0, t_tool_s=t_tool,
                          t_prefill_gpu_s=r.get("t_prefill_s"), t_decode_gpu_s=r.get("t_decode_s")))
    return Session(name or Path(path).parent.name, turns)


# ---------------------------------------------------------------------------
# Decode curves: step time vs context, integrated over a session
# ---------------------------------------------------------------------------

class Curve:
    """A per-token cost t(L) on a grid, with its running integral for fast sums."""

    def __init__(self, L, values):
        self.L = np.asarray(L, float)
        self.v = np.asarray(values, float)
        seg = (self.v[1:] + self.v[:-1]) / 2 * np.diff(self.L)
        self.F = np.concatenate([[0.0], np.cumsum(seg)])

    def at(self, L):
        return np.interp(L, self.L, self.v)

    def _F(self, x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(self.L, x, side="right") - 1, 0, len(self.L) - 2)
        dx = x - self.L[i]
        slope = (self.v[i + 1] - self.v[i]) / (self.L[i + 1] - self.L[i])
        return self.F[i] + self.v[i] * dx + 0.5 * slope * dx * dx

    def sum_tokens(self, L0, n):
        """Sum of t(L) for L = L0+1 .. L0+n (one decode step per token)."""
        if n <= 0:
            return 0.0
        return float(self._F(L0 + n + 0.5) - self._F(L0 + 0.5))


def ctx_grid(max_ctx, n=48):
    hi = max(1024, int(max_ctx * 1.05) + 1)
    g = np.unique(np.concatenate([np.linspace(1, hi, n), [1, 512, 1024, 2048, 4096]]).astype(int))
    return g[g <= hi]


@dataclass
class SessionResult:
    total: float
    llm: float
    tools: float
    terms: Dict[str, float]          # LLM-side terms, summed over the session
    by_kind: Dict[str, float]        # LLM-side time per op kind (+ "link")
    over_capacity_at: Optional[int]  # first context length where the design runs out of room
    max_ctx: int
    decode_curve: Curve


class Replayer:
    """Times sessions on designs with one Integrator. Caches curves per design."""

    def __init__(self, integrator, grid_points=48):
        self.integ = integrator
        self.grid_points = grid_points

    def decode_terms(self, design, max_ctx):
        grid = ctx_grid(max_ctx, self.grid_points)
        steps = [self.integ.step(design, 1, int(L)) for L in grid]
        keys = sorted({k for s in steps for k in s.terms})
        curves = {k: Curve(grid, [s.terms.get(k, 0.0) for s in steps]) for k in keys}
        kinds = sorted({k for s in steps for k in s.by_kind})
        kind_curves = {k: Curve(grid, [s.by_kind.get(k, 0.0) for s in steps]) for k in kinds}
        total = Curve(grid, [s.total for s in steps])
        first_over = None
        if any(s.over_capacity for s in steps):
            first_over = self._first_over(design, grid, steps)
        return total, curves, kind_curves, first_over

    def _first_over(self, design, grid, steps):
        bad = [i for i, s in enumerate(steps) if s.over_capacity]
        i = bad[0]
        if i == 0:
            return int(grid[0])
        lo, hi = int(grid[i - 1]), int(grid[i])
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.integ.capacity_over(design, mid):
                hi = mid
            else:
                lo = mid
        return hi

    def run(self, design, session: Session, prefill=True):
        total_c, curves, kind_curves, first_over = self.decode_terms(design, session.max_ctx)
        terms = {k: 0.0 for k in curves}
        by_kind = {k: 0.0 for k in kind_curves}
        for t in session.turns:
            L0 = t.ctx_start
            for k, c in curves.items():
                terms[k] += c.sum_tokens(L0, t.gen_tokens)
            for k, c in kind_curves.items():
                by_kind[k] += c.sum_tokens(L0, t.gen_tokens)
            if prefill and t.prefill_tokens:
                r = self.integ.step(design, t.prefill_tokens, t.ctx_start, logits_tokens=1)
                for k, v in r.terms.items():
                    terms[k] = terms.get(k, 0.0) + v
                for k, v in r.by_kind.items():
                    by_kind[k] = by_kind.get(k, 0.0) + v
        llm = sum(terms.values())
        tools = session.tool_time
        return SessionResult(total=llm + tools, llm=llm, tools=tools, terms=terms, by_kind=by_kind,
                             over_capacity_at=first_over if first_over and first_over <= session.max_ctx else None,
                             max_ctx=session.max_ctx, decode_curve=total_c)


# ---------------------------------------------------------------------------
# GPU baseline from measurements
# ---------------------------------------------------------------------------

class GpuBaseline:
    """RTX 3080 decode time vs context (llama-bench G1), and prefill rate."""

    def __init__(self, g1_jsonl, weights="Q8_0", kv="f16", prefill_tok_s=None):
        pts = []
        for line in Path(g1_jsonl).read_text().splitlines():
            r = json.loads(line)
            if r.get("type_k") == kv and r.get("model_filename", "").endswith(f"-{weights}.gguf") \
                    and r.get("n_gen", 0) > 0:
                pts.append((r["n_depth"], 1.0 / r["avg_ts"]))
        if not pts:
            raise ValueError(f"no G1 points for {weights}/{kv}")
        pts.sort()
        self.depths = np.array([p[0] for p in pts], float)
        self.wall = int(self.depths[-1])     # deepest context that fit in VRAM
        self.curve = Curve(np.maximum(self.depths, 1), [p[1] for p in pts])
        self.prefill_tok_s = prefill_tok_s

    def session_time(self, session: Session, measured_prefill=True, measured_decode=True):
        """GPU time for a session: measured in the trace where available, else from the G1 curve."""
        if measured_decode and all(t.t_decode_gpu_s is not None for t in session.turns):
            dec = sum(t.t_decode_gpu_s for t in session.turns)
        else:
            dec = sum(self.curve.sum_tokens(t.ctx_start, t.gen_tokens) for t in session.turns)
        pre = 0.0
        for t in session.turns:
            if measured_prefill and t.t_prefill_gpu_s is not None:
                pre += t.t_prefill_gpu_s
            elif self.prefill_tok_s:
                pre += t.prefill_tokens / self.prefill_tok_s
        return {"decode": dec, "prefill": pre, "tools": session.tool_time,
                "total": dec + pre + session.tool_time, "hits_wall": session.max_ctx > self.wall}


class WorkloadReplayer:
    """Times a set of sessions on a design from shared per-token curves.

    Decode: per-token step time vs context (exact up to grid interpolation).
    Prefill: a chunk of n tokens at context L costs n * p(L), with p(L) the
    per-token cost of a `chunk`-token prefill step at L. Prefill steps are
    linear in tokens on every unit here except for per-step fixed costs
    (launches, reprogramming), which the chunk amortizes; see tests/test_replay.py
    for the error against Replayer.run.
    """

    def __init__(self, integrator, grid_points=48, chunk=256):
        self.integ = integrator
        self.grid_points = grid_points
        self.chunk = chunk

    def run(self, design, sessions):
        max_ctx = max(s.max_ctx for s in sessions)
        grid = ctx_grid(max_ctx, self.grid_points)
        dec = [self.integ.step(design, 1, int(L)) for L in grid]
        pre = [self.integ.step(design, self.chunk, max(int(L), self.chunk), logits_tokens=1) for L in grid]
        tkeys = sorted({k for s in dec + pre for k in s.terms})
        kkeys = sorted({k for s in dec + pre for k in s.by_kind})
        dT = {k: Curve(grid, [s.terms.get(k, 0.0) for s in dec]) for k in tkeys}
        dK = {k: Curve(grid, [s.by_kind.get(k, 0.0) for s in dec]) for k in kkeys}
        pT = {k: Curve(grid, [s.terms.get(k, 0.0) / self.chunk for s in pre]) for k in tkeys}
        pK = {k: Curve(grid, [s.by_kind.get(k, 0.0) / self.chunk for s in pre]) for k in kkeys}
        first_over = None
        if any(s.over_capacity for s in dec):
            first_over = Replayer(self.integ)._first_over(design, grid, dec)
        out = []
        for sess in sessions:
            terms = {k: 0.0 for k in tkeys}
            by_kind = {k: 0.0 for k in kkeys}
            for t in sess.turns:
                for k in tkeys:
                    terms[k] += dT[k].sum_tokens(t.ctx_start, t.gen_tokens)
                    if t.prefill_tokens:
                        terms[k] += t.prefill_tokens * float(pT[k].at(t.ctx_start))
                for k in kkeys:
                    by_kind[k] += dK[k].sum_tokens(t.ctx_start, t.gen_tokens)
                    if t.prefill_tokens:
                        by_kind[k] += t.prefill_tokens * float(pK[k].at(t.ctx_start))
            llm = sum(terms.values())
            over = first_over if first_over is not None and first_over <= sess.max_ctx else None
            out.append(SessionResult(total=llm + sess.tool_time, llm=llm, tools=sess.tool_time,
                                     terms=terms, by_kind=by_kind, over_capacity_at=over,
                                     max_ctx=sess.max_ctx, decode_curve=None))
        return out
