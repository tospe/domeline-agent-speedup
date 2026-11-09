"""The large design space: per-layer, per-operator placement and KV precision.

A large point is a dict: the small space's hardware knobs (HW_KNOBS), plus one
slot per decision in the op graph:

  L<l>.<op>        placement of one op of layer l (q_proj, ..., attn_norm, rope, ...),
                   options from the small-space knob of its role (mvm, softmax, vector)
  L<l>.attention   q.K^T, p.V and the KV append of layer l, together (the KV lives
                   where attention reads it)
  L<l>.kv_bits     KV precision of layer l
  lm_head, final_norm, embed, sample   the ops outside the layers

`tie(p)` lifts a small-space point to the large point with every layer on the
small choice; it times identically (tests/test_large_space.py). Slot values are
role names (xbar, cpu, vfu, star) mapped to techs as in Space.

Three tools on top, none changing the integrator's equations:

  FastEvaluator   an Evaluator for the large space that uses the replay's linearity:
                  the workload's LLM time is sum_j w_j * step_j over the sampled steps j
                  (Curve.sum_tokens and the prefill interpolation are linear in the
                  sampled values), and a step is per-op costs plus boundary costs. Per-op
                  costs come from Integrator._op_time once per tech setting; the
                  integrator's transfer, launch, group and capacity logic is replayed on
                  weighted scalars. Same numbers as Evaluator (to float rounding).
  mapping_dp      the exact mapping optimum with the hardware fixed, by dynamic
                  programming over the op chain.
  certificate     a floor on any design's time: tools + per-slot cheapest cost over every
                  tech and hardware setting.
"""

import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ops as O
from .design import Design
from .evaluate import BudgetExhausted, Eval, Evaluator, binding_and_next
from .integrator import COPROC, HOST, Integrator
from .session import Curve, Replayer, ctx_grid
from .space import ATTN_OPS, EDGE_OPS, EFFORT, KNOBS, START, VECTOR_OPS, WEIGHT_MVMS, Space
from .space import canonical as small_canonical
from .tech import xbar_write_rows_time

HW_KNOBS = ("xbar_tech", "xbar_scale", "vfu_units", "vfu_bw", "vfu_mem", "star_units", "link",
            "xbar_tile", "adc_per_ima", "cell_bits")


def role_of(kind):
    """The small-space knob that places op kind (None: always on the host)."""
    if kind in WEIGHT_MVMS:
        return "mvm"
    if kind == "lm_head":
        return "lm_head"
    if kind in ATTN_OPS:
        return "attention"
    if kind == "softmax":
        return "softmax"
    if kind in VECTOR_OPS:
        return "vector"
    if kind in EDGE_OPS:
        return "edge"
    return None


@dataclass
class Slot:
    name: str
    role: str                 # small-space knob whose options it takes ("kv_bits" for precision)
    ops: List[str]            # op names it places
    layer: Optional[int]

    @property
    def options(self):
        return KNOBS[self.role]


def build_slots(spec):
    """Slots of the large space, in op order (from a decode step's op graph)."""
    slots: Dict[str, Slot] = {}
    for op in O.forward_ops(spec, 1, 2):
        role = role_of(op.kind)
        if role is None:
            continue
        if role == "attention":
            name = f"L{op.layer}.attention"
            kv = f"L{op.layer}.kv_bits"
            slots.setdefault(kv, Slot(kv, "kv_bits", [], op.layer)).ops.append(op.name)
        else:
            name = op.name
        slots.setdefault(name, Slot(name, role, [], op.layer)).ops.append(op.name)
    return slots


class LargeSpace(Space):
    """Space with per-layer slots. Same interface as Space (canonical, key, to_design, area, effort)."""

    def __init__(self, lib, model, ref_ctx, **kw):
        super().__init__(lib, model, ref_ctx, **kw)
        self.slots = build_slots(model)
        self.knobs = list(HW_KNOBS) + list(self.slots)
        self.place_slots = [s for s in self.slots.values() if s.role != "kv_bits"]
        self.kv_slots = [s for s in self.slots.values() if s.role == "kv_bits"]
        self.op_slot = {o: s.name for s in self.place_slots for o in s.ops}
        self.start = self.tie(START)
        self._ref_ops = O.forward_ops(model, 1, ref_ctx)
        self._designs = {}

    # -- points ------------------------------------------------------------------

    def tie(self, p):
        """The large point with every layer on small point p's choices."""
        p = {**START, **p}
        q = {k: p[k] for k in HW_KNOBS}
        for s in self.slots.values():
            q[s.name] = p[s.role]
        return q

    def _used(self, p, value, roles=None):
        return any(p[s.name] == value for s in self.place_slots if roles is None or s.role in roles)

    def proxy(self, p):
        """A small point with p's hardware knobs that is active exactly where p is."""
        q = {**START, **{k: p[k] for k in HW_KNOBS}}
        q.update(mvm="xbar" if self._used(p, "xbar") else "cpu", lm_head="cpu", edge="cpu",
                 attention="vfu" if self._used(p, "vfu", ("attention",)) else "cpu",
                 softmax="star" if self._used(p, "star") else "cpu",
                 vector="vfu" if self._used(p, "vfu") else "cpu")
        return q

    def canonical(self, p):
        q = {**self.start, **p}
        c = super().canonical(self.proxy(q))
        q.update({k: c[k] for k in HW_KNOBS})
        return q

    def key(self, p):
        q = self.canonical(p)
        return tuple(q[k] for k in self.knobs)

    def is_tied(self, p):
        """True when p is tie(small point) for some small point."""
        by_role = defaultdict(set)
        for s in self.slots.values():
            by_role[s.role].add(p[s.name])
        return all(len(v) == 1 for v in by_role.values())

    def untie(self, p):
        """The small point of a tied large point."""
        q = {k: p[k] for k in HW_KNOBS}
        for s in self.slots.values():
            q[s.role] = p[s.name]
        return small_canonical(q)

    # -- designs -----------------------------------------------------------------

    def _tech(self, p):
        return {"xbar": self.mvm_tech(p), "cpu": self.host, "vfu": self.vfu, "star": self.star}

    def _mapping(self, p):
        tech = self._tech(p)
        return {o: tech[p[s]] for o, s in self.op_slot.items()}

    def _kv(self, p):
        """(Design.kv_bits, Design.kv_bits_by_layer): empty per-layer map when all layers agree."""
        by = {s.layer: p[s.name] for s in self.kv_slots}
        vals = set(by.values())
        if len(vals) <= 1:
            return (vals.pop() if vals else START["kv_bits"]), {}
        return by[min(by)], by

    def _bare(self, p, units=None, link=None, label=""):
        kv, by = self._kv(p)
        kw = {"link": link} if link is not None else {}
        return Design(mapping=self._mapping(p), units=units or {}, kv_bits=kv, kv_bits_by_layer=by,
                      host=self.host, label=label, **kw)

    def xbar_lib(self, p):
        return Space.xbar_lib(self, self.proxy(self.canonical(p)))

    def xbar_need(self, p, ctx=None):
        ctx = ctx or self.ref_ctx
        ops = self._ref_ops if ctx == self.ref_ctx else O.forward_ops(self.model, 1, ctx)
        cap = Integrator(self.model, self.xbar_lib(p)).capacity(self._bare(p), ctx, ops)
        tech = self.mvm_tech(p)
        t = self.lib[tech]
        if t.kind == "crossbar":
            return cap.get(tech, {}).get("needed", 0)
        need = cap.get(tech + ":bytes", {}).get("needed", 0)
        per_unit = t.p("local_mem_bytes", float("inf")) if t.p("mem_scales_with_units", False) else float("inf")
        return max(math.ceil(need / per_unit), t.p("units_ref", 1))

    def to_design(self, p):
        p = self.canonical(p)
        k = tuple(p[x] for x in self.knobs)
        if k not in self._designs:
            if len(self._designs) > 4096:
                self._designs.clear()
            self._designs[k] = self._to_design(p)
        return self._designs[k]

    def _to_design(self, p):
        units = {}
        if self._used(p, "xbar"):
            units[self.mvm_tech(p)] = int(math.ceil(p["xbar_scale"] * self.xbar_need(p)))
        units[self.vfu] = p["vfu_units"]
        units[self.star] = p["star_units"]
        lib = self.xbar_lib(p)
        if self._used(p, "vfu", ("attention",)):
            lib = dict(lib)
            v = lib[self.vfu]
            lib[self.vfu] = type(v)(**{**v.__dict__, "params": {**v.params, "mem_bw_Bps": p["vfu_bw"],
                                                                 "local_mem_bytes": p["vfu_mem"]}})
        from .space import LINKS
        d = self._bare(p, units=units, link=(self.links or LINKS)[p["link"]], label=str(self.key(p)))
        return d, lib

    def fixed_hardware(self, p, attn_vfu):
        """(Design, lib) with p's hardware knobs as given, every tech active, mapping empty.

        For tools that vary the mapping with the hardware fixed (mapping_dp): to_design
        resets knobs that p's own mapping leaves inactive, which would change the hardware.
        Crossbar units are left out (op times do not depend on them below the wall); the
        vector unit's KV knobs apply iff attn_vfu, as in to_design.
        """
        from .space import LINKS
        q = {**self.start, **p}
        proxy = {**START, **{k: q[k] for k in HW_KNOBS}, "mvm": "xbar", "softmax": "star", "vector": "vfu",
                 "attention": "vfu" if attn_vfu else "cpu"}
        lib = Space.xbar_lib(self, proxy)          # Space.canonical: proxy is a small point
        if attn_vfu:
            lib = dict(lib)
            v = lib[self.vfu]
            lib[self.vfu] = type(v)(**{**v.__dict__, "params": {**v.params, "mem_bw_Bps": q["vfu_bw"],
                                                                 "local_mem_bytes": q["vfu_mem"]}})
        units = {self.vfu: q["vfu_units"], self.star: q["star_units"]}
        d = Design(mapping={}, units=units, link=(self.links or LINKS)[q["link"]], host=self.host)
        return d, lib

    def effort_months(self, p):
        p = self.canonical(p)
        d, lib = self.to_design(p)
        e = sum(lib[t].effort_months for t in set(d.mapping.values()) - {self.host})
        e += EFFORT["link"][p["link"]]
        if self._used(p, "vfu", ("attention",)):
            e += EFFORT["vfu_bw"][p["vfu_bw"]]
        if self._used(p, "xbar") and self._crossbar(p):
            e += EFFORT["xbar_tile"][p["xbar_tile"]] + EFFORT["adc_per_ima"][p["adc_per_ima"]] \
                + EFFORT["cell_bits"][p["cell_bits"]]
        return e


# ---------------------------------------------------------------------------
# Linear replay: sampled steps and their weights
# ---------------------------------------------------------------------------

def replay_weights(sessions, grid_points=48, chunk=256):
    """The sampled steps of WorkloadReplayer and their weights over a set of sessions.

    Returns (steps, w): steps[j] = (n_new, ctx, logits_tokens); the workload's LLM time
    on any design is sum_j w[j] * step_j.total, the same sum WorkloadReplayer.run makes
    (decode: Curve.sum_tokens is linear in the curve's values; prefill: n * p(L) with p
    linearly interpolated).
    """
    max_ctx = max(s.max_ctx for s in sessions)
    grid = ctx_grid(max_ctx, grid_points)
    G = len(grid)
    L0 = np.array([t.ctx_start for s in sessions for t in s.turns], float)
    n = np.array([t.gen_tokens for s in sessions for t in s.turns], float)
    pre = np.array([t.prefill_tokens or 0 for s in sessions for t in s.turns], float)
    dec_w, pre_w = np.zeros(G), np.zeros(G)
    for g in range(G):
        e = np.zeros(G)
        e[g] = 1.0
        c = Curve(grid, e)
        m = n > 0
        dec_w[g] = float(np.sum(c._F(L0[m] + n[m] + 0.5) - c._F(L0[m] + 0.5)))
        pre_w[g] = float(np.sum(pre * np.interp(L0, grid, e))) / chunk
    steps = [(1, int(L), None) for L in grid] + [(chunk, max(int(L), chunk), 1) for L in grid]
    return grid, steps, np.concatenate([dec_w, pre_w])


@dataclass
class OpInfo:
    """One op of the step graph, across every sampled step."""
    name: str
    kind: str
    cls: str
    layer: Optional[int]
    group: Optional[str]
    reads: List[Tuple[str, float]]       # (tensor, weighted bytes)
    writes: List[str]
    sig: tuple                           # ops with one sig cost the same on every tech
    per_step: list                       # the Op at each sampled step


class Tables:
    """Per-op costs for one integrator/library, on the sampled steps of one workload.

    cost(i, tech, units, kb) -> (weighted seconds, term, per-step vector), cached per op
    signature (ops of one kind in different layers cost the same). resident(i, tech, kb)
    -> (capacity key, per-step units needed) for the capacity test.
    """

    def __init__(self, model, sessions, grid_points=48, chunk=256):
        self.model = model
        self.grid, self.steps, self.w = replay_weights(sessions, grid_points, chunk)
        self.W = float(self.w.sum())
        self.n_dec = len(self.grid)
        self.max_ctx = max(s.max_ctx for s in sessions)
        per = [O.forward_ops(model, n, c, lt) for n, c, lt in self.steps]
        self.ops: List[OpInfo] = []
        for i, op0 in enumerate(per[0]):
            col = [p[i] for p in per]
            assert all(o.name == op0.name for o in col)
            reads = []
            for r, (tensor, _) in enumerate(op0.reads):
                reads.append((tensor, float(sum(wj * o.reads[r][1] for wj, o in zip(self.w, col)))))
            sig = (op0.kind, op0.cls, op0.layer is None, op0.group is not None,
                   tuple((o.flops, o.shape, o.tokens, o.kv_read_elems, o.kv_write_elems,
                          tuple(b for _, b in o.reads), tuple(b for _, b in o.writes)) for o in col))
            self.ops.append(OpInfo(op0.name, op0.kind, op0.cls, op0.layer, op0.group, reads,
                                   [t for t, _ in op0.writes], sig, col))
        ids = {}
        for o in self.ops:
            o.sig = ids.setdefault(o.sig, len(ids))
        # KV reads grow linearly with context (ops.forward_ops); the wall bisection relies on it.
        for o in self.ops:
            if o.cls == O.MVM_DYNAMIC:
                r = {op.kv_read_elems / c for op, (n, c, lt) in zip(o.per_step, self.steps)}
                assert max(r) - min(r) < 1e-9 * max(r), o.name
        self.index = {o.name: i for i, o in enumerate(self.ops)}
        self._cost = {}
        self._res = {}
        self._spill = {}

    @staticmethod
    def tech_key(t, units):
        return (t.name, repr(sorted(t.params.items())), units)

    def cost(self, i, t, units, kb, link_bw=None, tkey=None):
        """(weighted s, term, vector over steps) of op i on tech t. link_bw: KV spilled over the link."""
        op = self.ops[i]
        k = (op.sig, tkey or self.tech_key(t, units), kb, link_bw)
        if k not in self._cost:
            d = Design(mapping={}, units={t.name: units}, kv_bits=kb)
            over = set()
            if link_bw is not None:
                over = {t.name + ":bytes"}
                d = d.with_(link=type(d.link)(bw_Bps=link_bw))
            integ = Integrator(self.model, {t.name: t})
            vec, term = np.zeros(len(self.steps)), None
            for j, (n, ctx, lt) in enumerate(self.steps):
                vec[j], term = integ._op_time(op.per_step[j], t, d, ctx, over, {})
            self._cost[k] = (float(self.w @ vec), term, vec)
        return self._cost[k]

    def resident(self, i, t, kb):
        """(capacity key, units needed per sampled step, logical bytes per step) of op i on t; None if it stores nothing."""
        op = self.ops[i]
        k = (op.sig, self.tech_key(t, 0), kb)
        if k not in self._res:
            d = Design(mapping={op.name: t.name}, kv_bits=kb)
            integ = Integrator(self.model, {t.name: t})
            key, units = None, np.zeros(len(self.steps))
            for j, (n, ctx, lt) in enumerate(self.steps):
                for ck, _, _, u in integ._resident(d, ctx, [op.per_step[j]]):
                    key, units[j] = ck, units[j] + u
            self._res[k] = (key, units) if key else None
        return self._res[k]


# ---------------------------------------------------------------------------
# The integrator's step logic, on weighted scalars
# ---------------------------------------------------------------------------

class Chain:
    """One hardware setting: per-op cost options, and the integrator's boundary rules.

    `advance` is Integrator.step's loop body (launch on a change of side, transfers of
    inputs produced on the other side, NoC bytes, crossbar groups adding their max);
    both the evaluator and the DP call it, so they cannot disagree.
    """

    def __init__(self, tables: Tables, design: Design, lib):
        self.T, self.d, self.lib = tables, design, lib
        self.link = design.link
        self._memo = {}

    def op_cost(self, i, tname, kb, spill_mask=None):
        """(weighted s, term, vector) of op i on tname; spill_mask: steps where its KV spills."""
        t = self.lib[tname]
        units = self.d.units.get(tname, 1)
        mk = (self.T.ops[i].sig, tname, kb)
        c = self._memo.get(mk)
        if c is None:
            c = self._memo[mk] = self.T.cost(i, t, units, kb)
        if spill_mask is not None and spill_mask.any() and self.T.ops[i].cls == O.MVM_DYNAMIC \
                and t.kind in ("digital", "dram_pim"):
            s = self.T.cost(i, t, units, kb, link_bw=self.link.bw_Bps)
            vec = np.where(spill_mask, s[2], c[2])
            return float(self.T.w @ vec), c[1], vec
        return c

    def advance(self, st, i, tname, kb, add, spill_mask=None):
        """Apply op i on tname to state st = [side, group, where, copies]; add(term, kind, s)."""
        T, link, lib = self.T, self.link, self.lib
        op = T.ops[i]
        t = lib[tname]
        if not t.can_run(op.cls):
            raise ValueError(f"{tname} cannot run {op.name} ({op.cls})")
        side, grp, where, copies = st
        op_side = HOST if t.on_host else COPROC

        def flush():
            nonlocal grp
            if grp is not None:
                (gname, term), members = grp
                vec = np.max([m for m in members], axis=0)
                add(term, None, float(T.w @ vec))
            grp = None

        if op_side != side:
            flush()
            add("launch", None, (link.launch_s if op_side == COPROC else link.sync_s) * T.W)
            side = op_side
        xfer, noc = 0.0, 0.0
        for tensor, wbytes in op.reads:
            src = where.get(tensor)
            if src is None or src == tname or tname in copies.get(tensor, ()):
                continue
            src_side = HOST if lib[src].on_host else COPROC
            if src_side != op_side:
                xfer += wbytes
            elif op_side == COPROC:
                noc += wbytes
            copies[tensor] = copies.get(tensor, frozenset()) | {tname}
        if xfer:
            flush()
            n_units = self.d.units.get(tname, 0) if t.p("host_scatter", False) else 0
            t0 = t.p("xfer_t0_s", link.t0_s)
            alpha = t.p("xfer_s_per_unit", link.alpha_s_per_unit)
            add("transfer", None, (t0 + alpha * n_units) * T.W + xfer / link.bw_Bps)
            add("_crossing", None, 1.0)
        if noc:
            add("transfer", None, noc / link.noc_bw_Bps)
        sec, term, vec = self.op_cost(i, tname, kb, spill_mask)
        add(None, op.kind, sec)
        if op.group and t.kind == "crossbar" and term == tname:
            if grp is None or grp[0] != (op.group, term):
                flush()
                grp = ((op.group, term), ())
            grp = (grp[0], grp[1] + (vec,))
        else:
            flush()
            add(term, None, sec)
        for tensor in op.writes:
            where[tensor] = tname
            copies[tensor] = frozenset((tname,))
        return [side, grp, where, copies]

    def finish(self, st, add):
        side, grp, where, copies = st
        if grp is not None:
            (gname, term), members = grp
            add(term, None, float(self.T.w @ np.max(list(members), axis=0)))


# ---------------------------------------------------------------------------
# Fast evaluator
# ---------------------------------------------------------------------------

class FastEvaluator(Evaluator):
    """Evaluator for a LargeSpace (or Space) with the linear replay; same Eval fields and numbers.

    exact=True times every point with the base Evaluator instead (for checks).
    """

    def __init__(self, space, sessions, gpu, budget=math.inf, area_cap_mm2=math.inf,
                 effort_cap_months=math.inf, exact=False):
        super().__init__(space, sessions, gpu, budget, area_cap_mm2, effort_cap_months)
        self.tables = Tables(space.model, sessions)
        self.exact = exact
        self.t_fast = self.t_exact = 0.0

    def chain(self, p):
        """(Chain, design, lib, canonical point) for point p."""
        p = self.space.canonical(p)
        design, lib = self.space.to_design(p)
        return Chain(self.tables, design, lib), design, lib, p

    def capacity_masks(self, ch: Chain, design):
        """Per capacity key: (units needed per step, have). Mirrors Integrator.capacity."""
        need = {}
        memo = {}
        for i, op in enumerate(self.tables.ops):
            tname = design.tech_for(op)
            kb = design.kv_bits_for(op)
            mk = (op.sig, tname, kb)
            if mk not in memo:
                memo[mk] = self.tables.resident(i, ch.lib[tname], kb)
            r = memo[mk]
            if r is None:
                continue
            k, units = r
            need[k] = need.get(k, 0.0) + units
        integ = Integrator(self.space.model, ch.lib)
        return {k: (v, integ._have(design, k)) for k, v in need.items()}

    def over_at(self, design, lib, ctx):
        """Integrator.capacity_over(design, ctx), from one representative op per signature.

        Weights do not depend on ctx; KV ops at ctx are the decode step's op with its KV
        read scaled to ctx (linear in ctx, asserted in Tables)."""
        T = self.tables
        integ = Integrator(self.space.model, lib)
        ref_ctx = T.steps[0][1]
        need, memo = defaultdict(float), {}
        for i, op in enumerate(T.ops):
            tname = design.tech_for(op)
            kb = design.kv_bits_for(op)
            mk = (op.sig, tname, kb)
            if mk not in memo:
                o = op.per_step[0]
                if o.cls == O.MVM_DYNAMIC:
                    o = replace(o, kv_read_elems=o.kv_read_elems * ctx / ref_ctx)
                d = Design(mapping={o.name: tname}, kv_bits=kb)
                memo[mk] = [(k, u) for k, _, _, u in integ._resident(d, ctx, [o])]
            for k, u in memo[mk]:
                need[k] += u
        return any(v > integ._have(design, k) for k, v in need.items())

    def first_over(self, design, lib, dec_over):
        """Replayer._first_over with over_at in place of the integrator's capacity test."""
        grid = self.tables.grid
        i = int(np.argmax(dec_over))
        if i == 0:
            return int(grid[0])
        lo, hi = int(grid[i - 1]), int(grid[i])
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.over_at(design, lib, mid):
                hi = mid
            else:
                lo = mid
        return hi

    def llm_terms(self, p):
        """(terms, by_kind, crossings per step, wall ctx) of the workload on point p, linearly."""
        ch, design, lib, p = self.chain(p)
        T = self.tables
        caps = self.capacity_masks(ch, design)
        spill = {k.partition(":")[0]: v > have for k, (v, have) in caps.items() if k.endswith(":bytes")}
        terms, by_kind = defaultdict(float), defaultdict(float)

        def add(term, kind, s):
            if kind is not None:
                by_kind[kind] += s
            elif term is not None:
                terms[term] += s

        st = [HOST, None, {}, {}]
        for i, op in enumerate(T.ops):
            tname = design.tech_for(op)
            st = ch.advance(st, i, tname, design.kv_bits_for(op), add, spill.get(tname))
        ch.finish(st, add)
        # Weights that do not fit are reprogrammed every step (Integrator.step).
        over_any = np.zeros(len(T.steps), bool)
        for k, (v, have) in caps.items():
            over = v > have
            over_any |= over
            if ":" in k or not over.any():
                continue
            t = lib[k]
            ports = max(1, t.p("write_parallel_arrays", 1))
            vec = np.where(over, np.ceil((v - have) / ports) * xbar_write_rows_time(t, t.p("rows")), 0.0)
            terms["write"] += float(T.w @ vec)
        crossings = terms.pop("_crossing", 0.0)
        by_kind["link"] = terms.get("launch", 0.0) + terms.get("transfer", 0.0)
        wall = None
        dec_over = over_any[:T.n_dec]
        if dec_over.any():
            wall = self.first_over(design, lib, dec_over)
        return dict(terms), dict(by_kind), crossings, wall, design, lib, p

    def __call__(self, point) -> Eval:
        if self.exact:
            t0 = time.time()
            e = super().__call__(point)
            self.t_exact += time.time() - t0
            return e
        k = self.space.key(point)
        if k in self.cache:
            return self.cache[k]
        if self.used >= self.budget:
            raise BudgetExhausted()
        t0 = time.time()
        terms, by_kind, _, wall, design, lib, p = self.llm_terms(point)
        e = self._make_eval(p, terms, by_kind, wall, design, lib)
        self.cache[k] = e
        self.log.append(e)
        dt = time.time() - t0
        self.wall_s += dt
        self.t_fast += dt
        return e

    def _make_eval(self, p, terms, by_kind, wall, design, lib):
        """Eval from the workload's terms; the rest exactly as Evaluator.__call__."""
        llm = sum(terms.values())
        tools = self.tools_time
        if wall is not None and wall > self.max_ctx:
            wall = None
        if wall is not None:
            # Evaluator takes the min over the sessions that reach the wall.
            wall = min((wall for s in self.sessions if wall <= s.max_ctx), default=None)
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
        integ = Integrator(self.space.model, lib)
        integ._ops_cache = self._base._ops_cache
        res = integ.data_in_pim(design, self.max_ctx)
        cap_tech, cap_room = res.binding_headroom()
        b1, b2, ratio = binding_and_next(terms)
        return Eval(point=p, time_s=total, llm_s=llm, tools_s=tools, feasible=not reasons,
                    reason="; ".join(reasons), area_mm2=area, effort_months=effort, terms=terms,
                    by_kind=by_kind, wall_ctx=wall, speedup_vs_gpu=self.gpu_time / total,
                    n=self.used + 1, binding_term=b1, second_term=b2, term_headroom=ratio,
                    max_ctx=self.max_ctx, D_bytes=res.D, D_wall_bytes=res.D_wall,
                    D_by_tensor=res.by_tensor(), capacity_headroom_bytes=cap_room,
                    capacity_headroom_tech=cap_tech)


# ---------------------------------------------------------------------------
# Exact mapping optimum with the hardware fixed (DP over the op chain)
# ---------------------------------------------------------------------------

@dataclass
class DPResult:
    point: Dict
    llm_s: float              # the DP's LLM time (weighted, below the wall) of the returned mapping
    states_max: int           # largest DP frontier
    transitions: int
    seconds: float
    arrays: float = 0.0       # crossbar arrays the mapping holds at ref_ctx (sum over ops, as capacity counts)
    objective: float = 0.0    # llm_s + lam * arrays (the minimized quantity)
    lam: float = 0.0


def mapping_dp(fe: FastEvaluator, hw_point, options: Optional[Dict[str, list]] = None, check_ctx=None,
               lam=0.0, which=("plain", "vfu_attn")):
    """Minimum LLM time over every mapping (placements and KV bits) with hw_point's hardware.

    options: slot -> allowed values (default: all; a fixed slot gets one value). The hardware
    is hw_point's HW_KNOBS; crossbar arrays still scale with each mapping's need (xbar_scale),
    so below the wall the op times do not depend on the mapping. Assumptions (checked by
    re-evaluating the result): no crossbar past its wall at any sampled step (xbar_scale >= 1
    gives this up to ref_ctx); VFU KV capacity is a constraint at check_ctx (default the
    workload's longest context), so designs past the VFU wall are excluded, not timed.

    The VFU's bandwidth and memory knobs apply only when some attention runs on the VFU
    (Space.to_design), which changes the VFU's op costs; so the DP runs twice, once with no
    attention on the VFU and once requiring some, each with its own library, and keeps the
    better. State: side, open crossbar group, live tensors (producer, copies), open slots
    (attention and kv_bits of the current layer), KV bytes on the VFU (only when the VFU's
    memory can run out; otherwise just whether any attention is there).

    lam > 0 minimizes llm_s + lam * arrays instead: arrays is the crossbar arrays every op
    mapped to the crossbar holds at ref_ctx (Integrator._resident, the count capacity and so
    xbar_scale and area use). which: which of the two runs to make.
    """
    t_start = time.time()
    space: LargeSpace = fe.space
    # Not canonical: resetting knobs inactive under hw_point's own mapping would change the hardware.
    base = {**space.start, **hw_point}
    opts = {s: list((options or {}).get(s, space.slots[s].options)) for s in space.slots}
    attn = [s.name for s in space.place_slots if s.role == "attention"]
    no_vfu = {**opts, **{a: [v for v in opts[a] if v != "vfu"] for a in attn}}
    runs = []
    if "plain" in which and all(no_vfu[a] for a in attn):
        runs.append(_dp_run(fe, base, no_vfu, False, check_ctx, lam))
    if "vfu_attn" in which and any("vfu" in opts[a] for a in attn):
        runs.append(_dp_run(fe, base, opts, True, check_ctx, lam))
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    best = min(runs, key=lambda r: r.objective)
    best.seconds = time.time() - t_start
    best.states_max = max(r.states_max for r in runs)
    best.transitions = sum(r.transitions for r in runs)
    return best


def xbar_arrays(fe, i, t, kb):
    """Crossbar arrays op i holds on tech t at ref_ctx (0 off a crossbar): Integrator._resident's count."""
    space = fe.space
    if t.kind != "crossbar":
        return 0.0
    memo = fe.__dict__.setdefault("_arrays_memo", {})
    op = fe.tables.ops[i]
    k = (op.sig, Tables.tech_key(t, 0), kb)
    if k not in memo:
        o = space._ref_ops[i]
        d = Design(mapping={o.name: t.name}, kv_bits=kb)
        memo[k] = float(sum(u for _, _, _, u in Integrator(space.model, {t.name: t})._resident(
            d, space.ref_ctx, [o])))
    return memo[k]


def _dp_run(fe, base, options, need_vfu_attn, check_ctx, lam=0.0):
    space: LargeSpace = fe.space
    T = fe.tables
    design, lib = space.fixed_hardware(base, need_vfu_attn)
    ch = Chain(T, design, lib)
    tech = space._tech(base)
    vfu = space.vfu
    cap_vfu = Integrator(space.model, lib)._have(design, vfu + ":bytes")
    ops_check = O.forward_ops(space.model, 1, check_ctx or fe.max_ctx)
    max_kv = sum(o.kv_read_elems * 16 / 8 for o in ops_check if o.cls == O.MVM_DYNAMIC)
    kv_track = cap_vfu < max_kv         # else the VFU's memory cannot run out: track only "any"

    last, slot_of, kv_of, first_i, last_i = {}, {}, {}, {}, {}
    for i, op in enumerate(T.ops):
        for tensor, _ in op.reads:
            last[tensor] = i
        slot_of[i] = space.op_slot.get(op.name)
        if op.layer is not None and op.kind in ATTN_OPS and f"L{op.layer}.kv_bits" in space.slots:
            kv_of[i] = f"L{op.layer}.kv_bits"
        for name in (slot_of[i], kv_of.get(i)):
            if name:
                first_i.setdefault(name, i)
                last_i[name] = i

    vec_ids = {}

    def skey(st, open_slots, kvb):
        side, grp, where, copies = st
        g = None if grp is None else (grp[0], tuple(vec_ids.setdefault(v.tobytes(), len(vec_ids))
                                                    for v in grp[1]))
        live = tuple(sorted((k, where[k], tuple(sorted(copies.get(k, ())))) for k in where))
        return side, g, live, tuple(sorted(open_slots.items())), kvb

    st0 = [HOST, None, {}, {}]
    frontier = {skey(st0, {}, 0.0): (0.0, 0.0, 0.0, st0, {}, 0.0, None)}
    states_max, transitions = 1, 0
    for i, op in enumerate(T.ops):
        s, kvs = slot_of[i], kv_of.get(i)
        nxt = {}
        for cost, tm, arr, st, open_slots, kvb, back in frontier.values():
            if s is None:
                choices = [(None, space.host)]
            elif s in open_slots:
                choices = [(None, tech[open_slots[s]])]
            else:
                choices = [(v, tech[v]) for v in options[s]]
            if kvs is None:
                kbs = [design.kv_bits]
            elif kvs in open_slots:
                kbs = [open_slots[kvs]]
            else:
                kbs = options[kvs]
            for v, tname in choices:
                if not lib[tname].can_run(op.cls):
                    continue
                for kb in kbs:
                    transitions += 1
                    acc = [0.0]

                    def add(term, kind, x, acc=acc):
                        if term is not None and term != "_crossing":
                            acc[0] += x

                    st2 = ch.advance([st[0], st[1], dict(st[2]), dict(st[3])], i, tname, kb, add)
                    kvb2 = kvb
                    if lib[tname].kind in ("digital", "dram_pim") and op.cls == O.MVM_DYNAMIC:
                        if kv_track:
                            kvb2 = kvb + ops_check[i].kv_read_elems * kb / 8
                            if kvb2 > cap_vfu:
                                continue
                        else:
                            kvb2 = 1.0
                    a = xbar_arrays(fe, i, lib[tname], kb)
                    os2 = dict(open_slots)
                    dec = {}
                    if s and s not in open_slots:
                        dec[s] = v
                        os2[s] = v
                    if kvs and kvs not in open_slots:
                        dec[kvs] = kb
                        os2[kvs] = kb
                    for name in (s, kvs):
                        if name and last_i[name] == i:
                            os2.pop(name, None)
                    w2, c2 = st2[2], st2[3]
                    for tensor in [k for k in w2 if last.get(k, -1) <= i]:
                        del w2[tensor]
                    for tensor in [k for k in c2 if k not in w2]:
                        del c2[tensor]
                    k2 = skey(st2, os2, kvb2)
                    c_new = cost + acc[0] + lam * a
                    if k2 not in nxt or c_new < nxt[k2][0]:
                        nxt[k2] = (c_new, tm + acc[0], arr + a, st2, os2, kvb2, (back, dec))
        frontier = nxt
        states_max = max(states_max, len(frontier))
    best = None
    for cost, tm, arr, st, _, kvb, back in frontier.values():
        if need_vfu_attn and kvb <= 0:
            continue
        acc = [0.0]
        ch.finish(st, lambda term, kind, x: acc.__setitem__(0, acc[0] + x))
        if best is None or cost + acc[0] < best[0]:
            best = (cost + acc[0], tm + acc[0], arr, back)
    if best is None:
        return None
    obj, tot, arr, back = best
    point = dict(base)
    while back is not None:
        back, dec = back
        point.update(dec)
    return DPResult(point=point, llm_s=tot, states_max=states_max, transitions=transitions, seconds=0.0,
                    arrays=arr, objective=obj, lam=lam)


# ---------------------------------------------------------------------------
# Certificate: a floor on any design's time
# ---------------------------------------------------------------------------

def hardware_settings(space: LargeSpace):
    """Every distinct setting of the knobs that change an op's cost (not sizes or links)."""
    keys = ("xbar_tech", "vfu_units", "vfu_bw", "star_units", "adc_per_ima", "cell_bits")
    for vals in itertools.product(*(KNOBS[k] for k in keys)):
        yield dict(zip(keys, vals))


def certificate(fe: FastEvaluator):
    """Floor on the workload time of every design in the large space (any hardware, any mapping).

    floor = tools (the Amdahl bound) + sum over slots of the slot's cheapest weighted op cost,
    minimized over its options, the KV bits and every hardware setting. Valid because every
    term is >= 0 (launch, transfer, NoC and reprogramming are dropped) and op costs do not
    depend on other ops' placement; crossbar groups (q/k/v, gate/up) count their max when
    their members are on the crossbar (bounded below by the largest member's cheapest cost). Returns (floor_s, parts) where
    parts[role] is the floor's LLM time from slots of that role.
    """
    space: LargeSpace = fe.space
    T = fe.tables
    # Every distinct (tech, units) variant a role value can take over the hardware settings.
    variants = defaultdict(dict)          # value -> tech_key -> (Tech, units)
    for hw in hardware_settings(space):
        for attn_vfu in (False, True):
            d, lib = space.fixed_hardware(hw, attn_vfu)
            for v, tname in space._tech({**space.start, **hw}).items():
                t, u = lib[tname], d.units.get(tname, 1)
                variants[v].setdefault(Tables.tech_key(t, u), (t, u))
    best, best_vec = {}, {}               # (op, value, kb) -> min weighted s / argmin vector
    for i, op in enumerate(T.ops):
        s = space.op_slot.get(op.name)
        vals = space.slots[s].options if s else ["cpu"]
        kbs = KNOBS["kv_bits"]
        for v in vals:
            for tk, (t, u) in variants[v].items():
                if not t.can_run(op.cls):
                    continue
                for kb in kbs:
                    sec, term, vec = T.cost(i, t, u, kb, tkey=tk)
                    k = (i, v, kb)
                    if k not in best or sec < best[k]:
                        best[k], best_vec[k] = sec, vec
    parts = defaultdict(float)
    done = set()
    for i, op in enumerate(T.ops):
        if i in done:
            continue
        s = space.op_slot.get(op.name)
        role = space.slots[s].role if s else "host_only"
        kb_min = lambda j, v: min((best[(j, v, kb)] for kb in KNOBS["kv_bits"] if (j, v, kb) in best),
                                  default=math.inf)
        if op.group:
            members = [j for j, o in enumerate(T.ops) if o.group == op.group]
            done.update(members)
            lb = math.inf
            for assign in itertools.product(space.slots[s].options, repeat=len(members)):
                # Members on the crossbar add their max (per step); max over members of each
                # one's cheapest weighted cost is below it for every hardware setting.
                xb = [kb_min(j, v) for j, v in zip(members, assign) if v == "xbar"]
                rest = sum(kb_min(j, v) for j, v in zip(members, assign) if v != "xbar")
                lb = min(lb, rest + (max(xb) if xb else 0.0))
            parts[role] += lb
        elif s and space.slots[s].role == "attention":
            sl = space.slots[s]
            members = [T.index[o] for o in sl.ops]
            done.update(members)
            parts[role] += min(sum(best.get((j, v, kb), math.inf) for j in members)
                               for v in sl.options for kb in KNOBS["kv_bits"])
        else:
            done.add(i)
            vals = space.slots[s].options if s else ["cpu"]
            parts[role] += min(kb_min(i, v) for v in vals)
    floor = fe.tools_time + sum(parts.values())
    return floor, dict(parts)
