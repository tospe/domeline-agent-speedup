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
    by_kind: Dict[str, float] = None              # seconds per op kind (link costs under "link")

    def term(self, k):
        return self.terms.get(k, 0.0)


def _weight_tensor(op):
    return "lm_head" if op.kind == "lm_head" else "w." + op.kind


def _kv_tensor(op):
    return {"attn_score": "kv.K", "attn_value": "kv.V"}.get(op.kind, "kv." + op.kind)


def _array_bytes(t):
    """Bytes one crossbar array stores."""
    return t.p("rows") * t.p("cols") * t.p("bits_per_cell") / 8


@dataclass
class Residency:
    """D, the data held in PIM at one context length (bytes).

    tensors   tech -> tensor -> logical bytes (w.<kind>, lm_head, kv.K, kv.V)
    occupied  tech -> bytes of storage those tensors take (whole crossbar arrays)
    capacity  tech -> bytes the design provides (arrays x array bytes, or local memory)
    """
    ctx: int
    tensors: Dict[str, Dict[str, float]]
    occupied: Dict[str, float]
    capacity: Dict[str, float]

    @property
    def D(self):
        """Total bytes resident on the coprocessor."""
        return sum(sum(v.values()) for v in self.tensors.values())

    @property
    def D_wall(self):
        """Total bytes the design can hold (inf if some tech has unbounded memory)."""
        return sum(self.capacity.values())

    def by_tensor(self):
        out = defaultdict(float)
        for v in self.tensors.values():
            for k, b in v.items():
                out[k] += b
        return dict(out)

    def headroom(self):
        """Per tech: capacity - occupied, in bytes (negative = past the wall)."""
        return {n: self.capacity[n] - self.occupied[n] for n in self.occupied}

    def binding_headroom(self):
        """(tech, bytes) with the least headroom; (None, inf) if nothing is held."""
        h = self.headroom()
        if not h:
            return None, math.inf
        n = min(h, key=lambda k: h[k])
        return n, h[n]

    @property
    def over(self):
        return any(v < 0 for v in self.headroom().values())


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
            self._ops_cache[key] = O.ops_for(self.model, n_new, ctx, logits_tokens)
        return self._ops_cache[key]

    # -- capacity ------------------------------------------------------------

    def _resident(self, design, ctx, ops):
        """What each coprocessor tech holds at context ctx, one entry per op that stores data.

        Yields (capacity key, tensor, logical bytes, units needed). The capacity key is
        the tech name (units = crossbar arrays) or "<tech>:bytes" (units = bytes of local
        memory). Logical bytes are the tensor at its precision, before array padding.
        """
        spec, wb = self.model, design.weight_bits
        for op in ops:
            kb = design.kv_bits_for(op)
            t = self.lib[design.tech_for(op)]
            if t.kind == "crossbar":
                if op.cls == O.MVM_STATIC:
                    M, N = op.shape
                    yield t.name, _weight_tensor(op), M * N * wb / 8, xbar_arrays_for(t, M, N, wb)
                elif op.kind == "attn_score":
                    # K stored with the head dim on rows: N = head_dim inputs, ctx outputs.
                    yield (t.name, "kv.K", spec.kv_heads * ctx * spec.head_dim * kb / 8,
                           spec.kv_heads * xbar_arrays_for(t, ctx, spec.head_dim, kb))
                elif op.kind == "attn_value":
                    yield (t.name, "kv.V", spec.kv_heads * spec.head_dim * ctx * kb / 8,
                           spec.kv_heads * xbar_arrays_for(t, spec.head_dim, ctx, kb))
            elif t.kind in ("digital", "dram_pim"):
                if op.cls == O.MVM_DYNAMIC:
                    nbytes = op.kv_read_elems * kb / 8
                    yield t.name + ":bytes", _kv_tensor(op), nbytes, nbytes
                elif op.cls == O.MVM_STATIC and t.p("weights_local", True):
                    M, N = op.shape
                    nbytes = M * N * wb / 8
                    yield t.name + ":bytes", _weight_tensor(op), nbytes, nbytes

    def _have(self, design, k):
        name, _, unit = k.partition(":")
        t = self.lib[name]
        if unit == "bytes":
            return t.p("local_mem_bytes", float("inf")) * (
                design.units.get(name, 1) if t.p("mem_scales_with_units", False) else 1)
        return design.units.get(name, 0)

    def capacity(self, design, ctx, ops=None):
        """Units each tech needs (crossbar arrays; bytes for digital local memory)."""
        ops = ops or self.ops(1, ctx)
        need = defaultdict(float)
        for k, _, _, units in self._resident(design, ctx, ops):
            need[k] += units
        return {k: {"needed": v, "have": self._have(design, k)} for k, v in need.items()}

    def data_in_pim(self, design, ctx, ops=None) -> "Residency":
        """D: the bytes resident on the coprocessor at context ctx, per tech and tensor.

        Counts exactly what `capacity` counts: static weights (and the LM head) on a
        crossbar or on a digital/DRAM-PIM tech that keeps weights locally, at
        weight_bits; K and V of every cached layer at kv_bits when attention runs on
        the coprocessor. Activations, the embedding table and host-resident data are
        not counted. `occupied` adds crossbar padding (whole arrays), so a tech is over
        capacity exactly when occupied > capacity, the same test as `capacity`.
        """
        ops = ops or self.ops(1, ctx)
        tensors = defaultdict(lambda: defaultdict(float))
        occupied = defaultdict(float)
        for k, tensor, nbytes, units in self._resident(design, ctx, ops):
            name, _, unit = k.partition(":")
            tensors[name][tensor] += nbytes
            occupied[name] += units * (1 if unit == "bytes" else _array_bytes(self.lib[name]))
        cap = {}
        for name in occupied:
            k = name if self.lib[name].kind == "crossbar" else name + ":bytes"
            have = self._have(design, k)
            cap[name] = have * (1 if k.endswith(":bytes") else _array_bytes(self.lib[name]))
        return Residency(ctx=ctx, tensors={n: dict(v) for n, v in tensors.items()},
                         occupied=dict(occupied), capacity=cap)

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
                # A tech may carry its own measured transfer law (UPMEM: t0 + alpha * N);
                # otherwise the link's applies.
                t0 = t.p("xfer_t0_s", link.t0_s)
                alpha = t.p("xfer_s_per_unit", link.alpha_s_per_unit)
                terms["transfer"] += t0 + alpha * n_units + xfer_bytes / link.bw_Bps
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
        by_kind = defaultdict(float)
        for _, kind, _, sec in per_op:
            by_kind[kind] += sec
        by_kind["link"] = terms.get("launch", 0.0) + terms.get("transfer", 0.0)
        if terms.get("write") and not any(k == "kv_append" for k in by_kind):
            by_kind["reprogram"] = terms["write"]
        return StepResult(total=total, terms=dict(terms), per_op=per_op, crossings=crossings,
                          link_bytes=link_bytes, capacity=cap, over_capacity=bool(over),
                          by_kind=dict(by_kind))

    # -- per-op cost -----------------------------------------------------------

    def _op_time(self, op, t: Tech, design, ctx, over, cap):
        spec, wb, kb = self.model, design.weight_bits, design.kv_bits_for(op)
        units = design.units.get(t.name, 1)
        group = spec.heads // spec.kv_heads

        if t.kind == "crossbar":
            # Each layer's arrays are separate, so a multi-token step pipelines tokens
            # across layers (ISAAC's inter-layer pipeline): per op, the first token pays
            # the full latency and each further token 1/layers of it.
            tok = op.tokens
            if tok > 1 and t.p("layer_pipelined", False) and op.layer is not None:
                tok = 1 + (tok - 1) / spec.layers
            if op.cls == O.MVM_STATIC:
                return xbar_mvm_time(t, op.shape[1], tokens=tok), t.name
            if op.kind == "attn_score":
                return xbar_mvm_time(t, spec.head_dim, tokens=tok, repeats=group), t.name
            if op.kind == "attn_value":
                return xbar_mvm_time(t, ctx, tokens=tok, repeats=group), t.name
            if op.cls == O.KV_APPEND:
                # V gains one row per token. K gains a column (head_dim rows touched)
                # unless the array supports transposed writes.
                k_rows = 1 if t.p("transpose_write", False) else spec.head_dim
                return xbar_write_rows_time(t, tok * (k_rows + 1)), "write"
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
