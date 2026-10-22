"""Integrator: VIPER's closed form generalized to many ops, many techs, one step.

For one forward step it walks the op graph in dependency order and adds up
(the dependency chain leaves nothing to overlap, except ops in one `group`
running on separate crossbar arrays):

  host     ops left on the CPU
  <tech>   compute on each coprocessor tech
  transfer activations crossing the CPU <-> coprocessor link (and the NoC)
  launch   offload commands and completion syncs, one pair per round trip
  write    KV appends, and reprogramming when weights do not fit

It also reports capacity: arrays (or bytes) each tech needs at this context
length against what the design provides. Needing more than provided is the
capacity wall; the step is still timed, with the overflow reprogrammed.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from . import ops as O
from .tech import (Tech, cpu_profile, roofline_time, tiled_gemv_time, unit_rates, xbar_arrays_for,
                   xbar_mvm_time, xbar_write_rows_time)

HOST, COPROC = "host", "coproc"


@dataclass
class StepResult:
    total: float
    terms: Dict[str, float]
    per_op: List[Tuple[str, str, str, float]]     # (op name, op kind, tech, seconds)
    crossings: int
    link_bytes: float
    capacity: Dict[str, Dict[str, float]]
    over_capacity: bool

    def term(self, k):
        return self.terms.get(k, 0.0)


class Integrator:
    def __init__(self, model, lib: Dict[str, Tech]):
        self.model = model
        self.lib = lib
        self._ops_cache = {}

    def ops(self, n_new, ctx, logits_tokens=None):
        key = (n_new, ctx, logits_tokens)
        if key not in self._ops_cache:
            if len(self._ops_cache) > 256:
                self._ops_cache.clear()
            self._ops_cache[key] = O.forward_ops(self.model, n_new, ctx, logits_tokens)
        return self._ops_cache[key]

    # -- capacity ------------------------------------------------------------

    def capacity(self, design, ctx, ops=None):
        """Units each tech needs (crossbar arrays; bytes for digital local memory)."""
        ops = ops or self.ops(1, ctx)
        need = defaultdict(float)
        spec, wb, kb = self.model, design.weight_bits, design.kv_bits
        group = spec.heads // spec.kv_heads
        for op in ops:
            t = self.lib[design.tech_for(op)]
            if t.kind == "crossbar":
                if op.cls == O.MVM_STATIC:
                    M, N = op.shape
                    need[t.name] += xbar_arrays_for(t, M, N, wb)
                elif op.kind == "attn_score":
                    # K stored with the head dim on rows: N = head_dim inputs, ctx outputs.
                    need[t.name] += spec.kv_heads * xbar_arrays_for(t, ctx, spec.head_dim, kb)
                elif op.kind == "attn_value":
                    need[t.name] += spec.kv_heads * xbar_arrays_for(t, spec.head_dim, ctx, kb)
            elif t.kind in ("digital", "dram_pim"):
                if op.cls == O.MVM_DYNAMIC:
                    need[t.name + ":bytes"] += op.kv_read_elems * kb / 8
                elif op.cls == O.MVM_STATIC and t.p("weights_local", True):
                    M, N = op.shape
                    need[t.name + ":bytes"] += M * N * wb / 8
        cap = {}
        for k, v in need.items():
            name, _, unit = k.partition(":")
            t = self.lib[name]
            if unit == "bytes":
                have = t.p("local_mem_bytes", float("inf")) * (
                    design.units.get(name, 1) if t.p("mem_scales_with_units", False) else 1)
            else:
                have = design.units.get(name, 0)
            cap[k] = {"needed": v, "have": have}
        return cap

    def capacity_over(self, design, ctx):
        return any(c["needed"] > c["have"] for c in self.capacity(design, ctx).values())

    # -- one step --------------------------------------------------------------

    def step(self, design, n_new=1, ctx=1, logits_tokens=None) -> StepResult:
        ops = self.ops(n_new, ctx, logits_tokens)
        spec, wb, kb = self.model, design.weight_bits, design.kv_bits
        link = design.link
        cap = self.capacity(design, ctx, ops)
        over = {k for k, c in cap.items() if c["needed"] > c["have"]}

        terms = defaultdict(float)
        per_op = []
        where: Dict[str, str] = {}       # tensor -> tech holding the latest copy
        copies: Dict[str, set] = defaultdict(set)
        side = HOST
        crossings = 0
        link_bytes = 0.0
        group_key, group_max = None, 0.0

        def flush_group():
            nonlocal group_key, group_max
            if group_key is not None:
                terms[group_key[1]] += group_max
            group_key, group_max = None, 0.0

        for op in ops:
            tname = design.tech_for(op)
            t = self.lib[tname]
            if not t.can_run(op.cls):
                raise ValueError(f"{tname} cannot run {op.name} ({op.cls})")
            op_side = HOST if t.on_host else COPROC

            # Launch / sync on every change of side.
            if op_side != side:
                flush_group()
                if op_side == COPROC:
                    terms["launch"] += link.launch_s
                else:
                    terms["launch"] += link.sync_s
                side = op_side

            # Move inputs that live elsewhere.
            xfer_bytes, noc_bytes = 0.0, 0.0
            for tensor, nbytes in op.reads:
                src = where.get(tensor)
                if src is None or src == tname or tname in copies[tensor]:
                    continue
                src_side = HOST if self.lib[src].on_host else COPROC
                if src_side != op_side:
                    xfer_bytes += nbytes
                elif op_side == COPROC:
                    noc_bytes += nbytes        # host-side units share memory: free
                copies[tensor].add(tname)
            if xfer_bytes:
                flush_group()
                n_units = design.units.get(tname, 0) if t.p("host_scatter", False) else 0
                terms["transfer"] += (link.t0_s + link.alpha_s_per_unit * n_units
                                      + xfer_bytes / link.bw_Bps)
                crossings += 1
                link_bytes += xfer_bytes
            if noc_bytes:
                terms["transfer"] += noc_bytes / link.noc_bw_Bps

            # Compute.
            t_op, term = self._op_time(op, t, design, ctx, over, cap)
            if op.group and t.kind == "crossbar" and term == tname:
                if group_key != (op.group, term):
                    flush_group()
                    group_key = (op.group, term)
                group_max = max(group_max, t_op)
            else:
                flush_group()
                terms[term] += t_op
            per_op.append((op.name, op.kind, tname, t_op))

            for tensor, _ in op.writes:
                where[tensor] = tname
                copies[tensor] = {tname}
        flush_group()

        # Weights that do not fit are reprogrammed every step.
        for k in over:
            if ":" in k:
                continue
            t = self.lib[k]
            overflow = cap[k]["needed"] - cap[k]["have"]
            ports = max(1, t.p("write_parallel_arrays", 1))
            terms["write"] += math.ceil(overflow / ports) * xbar_write_rows_time(t, t.p("rows"))

        total = sum(terms.values())
        return StepResult(total=total, terms=dict(terms), per_op=per_op, crossings=crossings,
                          link_bytes=link_bytes, capacity=cap, over_capacity=bool(over))

    # -- per-op cost -----------------------------------------------------------

    def _op_time(self, op, t: Tech, design, ctx, over, cap):
        spec, wb, kb = self.model, design.weight_bits, design.kv_bits
        units = design.units.get(t.name, 1)
        group = spec.heads // spec.kv_heads

        if t.kind == "crossbar":
            if op.cls == O.MVM_STATIC:
                return xbar_mvm_time(t, op.shape[1], tokens=op.tokens), t.name
            if op.kind == "attn_score":
                return xbar_mvm_time(t, spec.head_dim, tokens=op.tokens, repeats=group), t.name
            if op.kind == "attn_value":
                return xbar_mvm_time(t, ctx, tokens=op.tokens, repeats=group), t.name
            if op.cls == O.KV_APPEND:
                # V gains one row per token. K gains a column (head_dim rows touched)
                # unless the array supports transposed writes.
                k_rows = 1 if t.p("transpose_write", False) else spec.head_dim
                return xbar_write_rows_time(t, op.tokens * (k_rows + 1)), "write"
            raise ValueError(f"crossbar cannot run {op.kind}")

        if t.kind == "tiled_gemv":
            if op.cls != O.MVM_STATIC:
                raise ValueError(f"tiled_gemv cannot run {op.kind}")
            M, N = op.shape
            return tiled_gemv_time(t, M, N, tokens=op.tokens), t.name

        if t.kind == "cpu":
            prof = cpu_profile(t)
            if prof is not None and prof.has(op.kind):
                return op.tokens * prof.time(op.kind, ctx), "host"

        if t.kind in ("digital", "dram_pim", "cpu"):
            ops_per_s, bw = unit_rates(t, units, op.cls) if t.kind != "cpu" else (
                t.p("ops_per_s_by_class", {}).get(op.cls, t.p("ops_per_s")), t.p("mem_bw_Bps"))
            nbytes = 0.0
            if op.cls == O.MVM_STATIC:
                M, N = op.shape
                nbytes += M * N * wb / 8
            if op.cls == O.MVM_DYNAMIC:
                nbytes += op.kv_read_elems * kb / 8
                if t.name + ":bytes" in over:      # KV spilled to host memory over the link
                    bw = min(bw, design.link.bw_Bps)
            if op.cls == O.KV_APPEND:
                nbytes += op.kv_write_elems * kb / 8
            if op.cls == O.VECTOR or op.cls == O.GATHER:
                nbytes += sum(b for _, b in op.reads) + sum(b for _, b in op.writes)
            if t.p("compute_bound_only", False):
                nbytes = 0.0
            fixed = t.p("op_overhead_s", 0.0)
            tt = roofline_time(op.flops, nbytes, ops_per_s, bw, fixed)
            term = "host" if t.kind == "cpu" else t.name
            if op.cls == O.KV_APPEND and t.kind != "cpu":
                term = "write"
            return tt, term

        raise ValueError(f"no cost model for kind {t.kind}")
