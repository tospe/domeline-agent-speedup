"""Roofline baseline for E1 (docs/plan.md item 3): the prediction a Roofline user would make.

Per op, max(flops / peak, bytes / BW) on the device the op is mapped to
(`tech.roofline_time` with no fixed cost), summed per step. Nothing else:
no host time (main variant), no link or NoC transfer, no launch/sync, no
writes (KV appends and reprogramming are left out), no capacity wall (a KV
cache that does not fit is still read at the device's bandwidth).

Device roofs, from the same technology library the Domeline uses:

  crossbar  compute only: the weights and KV sit in the arrays, so no bytes move.
            "agg"  : the device peak, every array of the design at once:
                     2 * units * rows * (cols * bits_per_cell / weight_bits) / (input_bits/dac_bits * t_read).
                     This is the usual device roof and the main definition.
            "local": the peak of only the arrays the op occupies (the arrays
                     holding that matrix or that head's K/V), i.e. a
                     per-op roof. A friendlier Roofline, reported as a check.
  vfu/star  ops_per_s and mem_bw_Bps from `tech.unit_rates` (units and the
            design's KV bandwidth knob applied, as in the integrator).
            Bytes: weights for static MVMs, KV elements for attention,
            activations read + written for vector/gather ops.
  host      (second variant only) compute = the CPU tech's ops_per_s, BW =
            the measured STREAM-triad bandwidth of the host profile
            (`CpuProfile.stream_bw_Bps`); same byte counts.

Each op's time is filed under "<device>:<c|b>" (compute- or bandwidth-bound),
so a Roofline bottleneck is the device and bound of the largest summed
contribution.
"""

import math
from collections import defaultdict

import numpy as np

from harness import ops as O
from harness.session import Curve, ctx_grid
from harness.tech import cpu_profile, roofline_time, unit_rates, xbar_arrays_for

# Component columns written by the E1 scripts (seconds).
COMPONENTS = ["xbar_agg", "xbar_local", "vfu:c", "vfu:b", "star:c", "star:b", "host:c", "host:b"]

# Roofline variants: which components they sum.
VARIANTS = {
    "main": ["xbar_agg", "vfu:c", "vfu:b", "star:c", "star:b"],
    "host": ["xbar_agg", "vfu:c", "vfu:b", "star:c", "star:b", "host:c", "host:b"],
    "local": ["xbar_local", "vfu:c", "vfu:b", "star:c", "star:b"],
    "local_host": ["xbar_local", "vfu:c", "vfu:b", "star:c", "star:b", "host:c", "host:b"],
}


def device_class(tech):
    """Common-vocabulary device of a tech: xbar | vfu | star | host."""
    if tech.kind == "crossbar":
        return "xbar"
    if tech.kind == "cpu":
        return "host"
    if "star" in tech.name:
        return "star"
    if tech.kind == "digital":
        return "vfu"
    return tech.kind


def _group(ops):
    """Identical ops (same kind and shape, different layers) collapsed with a count."""
    g = {}
    for op in ops:
        act = sum(b for _, b in op.reads) + sum(b for _, b in op.writes)
        k = (op.kind, op.cls, op.flops, act, op.kv_read_elems, op.shape, op.tokens)
        if k in g:
            g[k][1] += 1
        else:
            g[k] = [op, 1, act]
    return list(g.values())


class Roofline:
    def __init__(self, model):
        self.model = model
        self._groups = {}

    def groups(self, n_new, ctx, logits_tokens=None):
        k = (n_new, ctx, logits_tokens)
        if k not in self._groups:
            self._groups[k] = _group(O.forward_ops(self.model, n_new, ctx, logits_tokens))
        return self._groups[k]

    def step(self, design, lib, n_new=1, ctx=1, logits_tokens=None):
        """Roofline seconds for one step, by component (see COMPONENTS)."""
        spec, wb, kb = self.model, design.weight_bits, design.kv_bits
        out = defaultdict(float)
        for op, count, act in self.groups(n_new, ctx, logits_tokens):
            if op.cls == O.KV_APPEND:
                continue                              # no writes in the Roofline view
            t = lib[design.tech_for(op)]
            dev = device_class(t)
            if t.kind == "crossbar":
                pass_s = math.ceil(t.p("input_bits", 8) / t.p("dac_bits", 1)) * t.p("t_read_s")
                per_array = lambda bits: 2 * t.p("rows") * t.p("cols") * t.p("bits_per_cell") / bits / pass_s
                units = design.units.get(t.name, 0)
                out["xbar_agg"] += count * op.flops / (units * per_array(wb))
                if op.cls == O.MVM_STATIC:
                    M, N = op.shape
                    arrays, bits = xbar_arrays_for(t, M, N, wb), wb
                elif op.kind == "attn_score":
                    arrays, bits = spec.kv_heads * xbar_arrays_for(t, ctx, spec.head_dim, kb), kb
                else:
                    arrays, bits = spec.kv_heads * xbar_arrays_for(t, spec.head_dim, ctx, kb), kb
                out["xbar_local"] += count * op.flops / (arrays * per_array(bits))
                continue
            if t.kind == "cpu":
                prof = cpu_profile(t)
                ops_per_s = t.p("ops_per_s_by_class", {}).get(op.cls, t.p("ops_per_s"))
                bw = prof.stream_bw_Bps if prof is not None else t.p("mem_bw_Bps")
            else:
                ops_per_s, bw = unit_rates(t, design.units.get(t.name, 1), op.cls)
            nbytes = 0.0
            if op.cls == O.MVM_STATIC:
                M, N = op.shape
                nbytes = M * N * wb / 8
            elif op.cls == O.MVM_DYNAMIC:
                nbytes = op.kv_read_elems * kb / 8
            elif op.cls in (O.VECTOR, O.GATHER):
                nbytes = act
            if t.p("compute_bound_only", False):
                nbytes = 0.0
            tc = op.flops / ops_per_s if ops_per_s else 0.0
            tm = nbytes / bw if bw else 0.0
            out[f"{dev}:{'c' if tc >= tm else 'b'}"] += count * roofline_time(op.flops, nbytes, ops_per_s, bw)
        return dict(out)


class SessionWeights:
    """Linear weights that replay per-step values over sessions exactly like WorkloadReplayer.

    WorkloadReplayer integrates decode curves (piecewise linear on the grid) and
    interpolates the per-token prefill cost; both are linear in the grid values,
    so a session total is w_dec . dec(grid) + w_pre . pre_per_token(grid).
    """

    def __init__(self, sessions, grid_points=48, chunk=256):
        self.chunk = chunk
        self.grid = ctx_grid(max(s.max_ctx for s in sessions), grid_points)
        n = len(self.grid)
        self.w_dec = np.zeros(n)
        self.w_pre = np.zeros(n)
        turns = [t for s in sessions for t in s.turns]
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            c = Curve(self.grid, e)
            self.w_dec[i] = sum(c.sum_tokens(t.ctx_start, t.gen_tokens) for t in turns)
            self.w_pre[i] = sum(t.prefill_tokens * float(c.at(t.ctx_start)) for t in turns if t.prefill_tokens)

    def replay(self, step_fn):
        """step_fn(n_new, ctx, logits_tokens) -> {component: seconds}; returns workload totals."""
        tot = defaultdict(float)
        for i, L in enumerate(self.grid):
            if self.w_dec[i]:
                for k, v in step_fn(1, int(L), None).items():
                    tot[k] += self.w_dec[i] * v
            if self.w_pre[i]:
                for k, v in step_fn(self.chunk, max(int(L), self.chunk), 1).items():
                    tot[k] += self.w_pre[i] * v / self.chunk
        return dict(tot)


# ---------------------------------------------------------------------------
# Common vocabulary for bottlenecks
# ---------------------------------------------------------------------------

def domeline_class(term):
    """Domeline term -> common class: xbar | vfu | star | host | transfer | launch | write."""
    if term.startswith("xbar"):
        return "xbar"
    if term.startswith("vfu"):
        return "vfu"
    if "star" in term:
        return "star"
    return term              # host, transfer, launch, write


COARSE = {"xbar": "coproc compute", "vfu": "coproc compute", "star": "coproc compute",
          "host": "host", "transfer": "data movement", "launch": "data movement",
          "write": "write", "capacity": "capacity", "none": "none"}
