"""Offload verifier: is a design a legal offload of the workload, and what does it imply?

`verify(design, lib, model, sessions, caps)` checks four rules and lists the extra
work the placement implies. It never times the workload; where a rule is about what
the evaluator charges, it runs `Integrator.step` and compares against that.

  R1  every op runs on a tech whose library entry supports its class, its finer
      label (subcls) and its precision; an op whose matrix operand is dynamic (KV)
      on a crossbar needs a write mechanism.
        R1.support          tech does not list op.cls
        R1.sampling         `sample` off the host (a tech must list "sampling")
        R1.precision        op runs at a bit width the op or the tech does not allow
        R1.write_mechanism  KV on a crossbar that cannot rewrite its arrays
  R2  dependencies hold and every host <-> coprocessor crossing is charged.
        R2.dependency       an op reads a tensor nothing earlier in the step wrote
        R2.dataflow         KV appended on one tech and read on another (the
                            integrator moves no KV between units)
        R2.transfer         a crossing the integrator does not charge, or charges
                            differently from the op graph (token in/out included:
                            the host feeds the token in and reads every sampled token)
        R2.implied          an implied cost (below) that is not in its term
  R3  resident data fits capacity at every state of the trace.
        R3.capacity         Integrator.capacity over at some turn's context
  R4  area and effort caps.
        R4.area, R4.effort

Implied costs (ImpliedCost, seconds per decode step at the longest state):
  kv_write    KV on a crossbar: K and V rows written every token  -> `write`
  reprogram   weights past a crossbar's capacity, rewritten every step -> `write`
  kv_spill    KV past a digital unit's memory, read over the link -> the unit's term
  scatter     host-mediated scatter (UPMEM alpha * N) per transfer  -> `transfer`
  token_io    the token crossing in (embed) or out (sample) of PIM   -> not modeled

Fields M1 adds (Op.subcls, Op.precisions, Op.operands, Tech.precisions,
Tech.params["write_mechanism"]) are read with the CONTRACT.md defaults, so this works
with or without them.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional

from . import ops as O
from .integrator import Integrator, _array_bytes
from .tech import xbar_write_rows_time

HOST, COPROC = "host", "coproc"

# Finer op labels (CONTRACT.md). `reduction_nonlinear` runs on any tech that runs
# `vector`; `sampling` needs a tech that lists it (the host always may).
REDUCTION_NONLINEAR = "reduction_nonlinear"
SAMPLING = "sampling"
SUBCLS_BY_KIND = {"softmax": REDUCTION_NONLINEAR, "rmsnorm": REDUCTION_NONLINEAR,
                  "qk_norm": REDUCTION_NONLINEAR, "act": REDUCTION_NONLINEAR, "sample": SAMPLING}
EXPLICIT_SUBCLS = (SAMPLING,)        # subclasses a non-host tech must list by name

TOKEN_BYTES = 4                      # one token id (the `token` tensor ops.py writes)
STEP_IN, STEP_OUT = "token_in", "token"
REL = 1e-9                           # tolerance for comparing against the integrator


class Violation(NamedTuple):
    rule: str
    op: str          # op name, or a tech / "design" for whole-design rules
    reason: str


@dataclass
class ImpliedCost:
    kind: str        # kv_write | reprogram | kv_spill | scatter | token_io
    op: str
    tech: str
    seconds: float   # per decode step at the longest state (0 if not estimated)
    term: Optional[str]   # the evaluator term that must hold it (None: not modeled)
    charged: bool
    note: str = ""


@dataclass
class Caps:
    area_mm2: float = math.inf
    effort_months: float = math.inf
    min_ctx: Optional[int] = None    # also hold this context (e.g. 65536), trace or not


@dataclass
class VerifyResult:
    legal: bool
    violations: List[Violation]
    implied_costs: List[ImpliedCost]
    checked: Dict = field(default_factory=dict)   # what was compared, for the report

    def rules(self):
        return {v.rule for v in self.violations}

    def by_rule(self, rule):
        return [v for v in self.violations if v.rule == rule]

    def summary(self):
        if self.legal:
            return "legal"
        out = {}
        for v in self.violations:
            out.setdefault(v.rule, []).append(v.op)
        return "; ".join(f"{r}: {ops[0]}" + (f" (+{len(ops) - 1})" if len(ops) > 1 else "")
                         for r, ops in out.items())


# ---------------------------------------------------------------------------
# Field defaults (CONTRACT.md)
# ---------------------------------------------------------------------------

def op_subcls(op):
    s = getattr(op, "subcls", None)
    return s if s is not None else SUBCLS_BY_KIND.get(op.kind)


def op_precisions(op):
    return tuple(getattr(op, "precisions", ()) or ())


def op_operands(op):
    """[(tensor, "static"|"dynamic")]: weights static, KV and activations dynamic."""
    given = getattr(op, "operands", None)
    if given:
        return list(given)
    out = [(t, "dynamic") for t, _ in op.reads]
    if op.cls == O.MVM_STATIC:
        out.append(("lm_head" if op.kind == "lm_head" else "w." + op.kind, "static"))
    elif op.cls in (O.MVM_DYNAMIC, O.KV_APPEND):
        out.append(("kv", "dynamic"))
    return out


def tech_precisions(t):
    return tuple(getattr(t, "precisions", ()) or ())


def has_write_mechanism(t):
    w = t.p("write_mechanism")
    if w is not None:
        return bool(w)
    return t.p("t_write_row_s") is not None


def op_bits(op, design):
    """Bit width the op's operand runs at: weights, KV, or fp16 activations."""
    if op.cls == O.MVM_STATIC:
        return design.weight_bits
    if op.cls in (O.MVM_DYNAMIC, O.KV_APPEND):
        return design.kv_bits
    return O.ACT_BYTES * 8


# ---------------------------------------------------------------------------
# Trace states
# ---------------------------------------------------------------------------

def trace_states(sessions):
    """[(ctx, session, turn index)]: each turn's start (prefill done) and end context."""
    out = []
    for s in sessions:
        for i, t in enumerate(s.turns):
            out.append((t.ctx_start, s.name, i))
            out.append((t.ctx_start + t.gen_tokens, s.name, i))
    return out


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------

def _r1(design, lib, ops, V):
    for op in ops:
        tname = design.tech_for(op)
        t = lib.get(tname)
        if t is None:
            V.append(Violation("R1.support", op.name, f"{tname!r} is not in the library"))
            continue
        if not t.can_run(op.cls):
            V.append(Violation("R1.support", op.name,
                               f"{tname} ({t.kind}) supports {list(t.supports)}, not {op.cls} ({op.kind})"))
            continue
        sub = op_subcls(op)
        if sub in EXPLICIT_SUBCLS and not t.on_host and sub not in t.supports:
            V.append(Violation("R1.sampling", op.name,
                               f"{op.kind} is {sub}: it needs the full logits, an RNG and the token on the "
                               f"host; {tname} does not list {sub!r}"))
        bits = op_bits(op, design)
        for who, allowed in (("op", op_precisions(op)), (tname, tech_precisions(t))):
            if allowed and bits not in allowed:
                V.append(Violation("R1.precision", op.name, f"runs at {bits} b; {who} allows {list(allowed)}"))
        if t.kind == "crossbar" and any(k == "dynamic" for tn, k in op_operands(op) if not _activation(tn)) \
                and not has_write_mechanism(t):
            V.append(Violation("R1.write_mechanism", op.name,
                               f"{op.kind} keeps a dynamic operand (KV) in {tname}, which has no write "
                               f"mechanism (no t_write_row_s / write_mechanism false)"))


def _activation(tensor):
    return not (tensor == "kv" or tensor.startswith(("kv.", "w.")) or tensor == "lm_head")


def _r2_static(design, lib, ops, V, implied):
    """Dependencies, KV placement and the token's way in and out."""
    written = {STEP_IN}
    for op in ops:
        for tensor, _ in op.reads:
            if tensor not in written:
                V.append(Violation("R2.dependency", op.name, f"reads {tensor!r} before anything writes it"))
        written.update(tn for tn, _ in op.writes)

    by_layer = {}
    for op in ops:
        if op.cls in (O.KV_APPEND, O.MVM_DYNAMIC):
            by_layer.setdefault(op.layer, []).append(op)
    for layer, kv_ops in sorted(by_layer.items(), key=lambda x: (x[0] is None, x[0])):
        techs = {op.kind: design.tech_for(op) for op in kv_ops}
        if len(set(techs.values())) > 1:
            app = techs.get("kv_append")
            V.append(Violation("R2.dataflow", f"L{layer}.kv_append",
                               f"KV appended on {app} but read on "
                               + ", ".join(f"{k}->{v}" for k, v in techs.items() if k != "kv_append" and v != app)
                               + "; the integrator moves no KV between units, so this traffic is not charged"))

    first, last = ops[0], ops[-1]
    for op, tensor, way in ((first, STEP_IN, "in from"), (last, STEP_OUT, "back to")):
        tname = design.tech_for(op)
        t = lib.get(tname)
        if t is None or t.on_host:
            continue
        nbytes = op.tokens * TOKEN_BYTES
        sec = design.link.t0_s + nbytes / design.link.bw_Bps + (design.link.sync_s if way == "back to" else 0.0)
        implied.append(ImpliedCost("token_io", op.name, tname, sec, None, False,
                                   f"token {way} the host every step ({nbytes:.0f} B); not modeled"))
        V.append(Violation("R2.transfer", op.name,
                           f"{op.kind} on {tname}: the token must cross {way} the host every step "
                           f"(~{sec * 1e6:.2f} us at {design.link.bw_Bps / 1e9:.0f} GB/s), and the integrator "
                           f"charges no transfer for it"))


def expected_link(design, lib, ops):
    """Crossings the op graph implies: (n crossings, link bytes, launch s, transfer s).

    Same semantics the integrator states in its docstring: an activation read by a
    tech on the other side of the link crosses once per (version, consuming tech);
    each change of side costs a launch (to PIM) or a sync (back); a crossing event
    costs t0 (+ alpha * N for host-scatter techs) + bytes / bw; PIM-to-PIM moves go
    over the NoC.
    """
    link = design.link
    where, copies = {}, {}
    side = HOST
    n = 0
    nbytes = launch = transfer = 0.0
    for op in ops:
        tname = design.tech_for(op)
        t = lib[tname]
        s = HOST if t.on_host else COPROC
        if s != side:
            launch += link.launch_s if s == COPROC else link.sync_s
            side = s
        xb = noc = 0.0
        for tensor, b in op.reads:
            src = where.get(tensor)
            if src is None or src == tname or tname in copies.get(tensor, ()):
                continue
            if (HOST if lib[src].on_host else COPROC) != s:
                xb += b
            elif s == COPROC:
                noc += b
            copies.setdefault(tensor, set()).add(tname)
        if xb:
            n += 1
            nbytes += xb
            units = design.units.get(tname, 0) if t.p("host_scatter", False) else 0
            transfer += t.p("xfer_t0_s", link.t0_s) + t.p("xfer_s_per_unit", link.alpha_s_per_unit) * units \
                + xb / link.bw_Bps
        transfer += noc / link.noc_bw_Bps
        for tensor, _ in op.writes:
            where[tensor] = tname
            copies[tensor] = {tname}
    return n, nbytes, launch, transfer


def _close(a, b):
    return abs(a - b) <= REL * max(abs(a), abs(b), 1e-30)


def _r2_charged(design, lib, integ, V, checked, steps):
    """Compare the op graph's crossings with what Integrator.step charges."""
    for label, (n_new, ctx, logits) in steps.items():
        ops = integ.ops(n_new, ctx, logits)
        r = integ.step(design, n_new, ctx, logits)
        n, nb, la, tr = expected_link(design, lib, ops)
        got = (r.crossings, r.link_bytes, r.term("launch"), r.term("transfer"))
        checked[f"link_{label}"] = {"expected": (n, nb, la, tr), "integrator": got}
        for name, e, g in zip(("crossings", "link bytes", "launch s", "transfer s"), (n, nb, la, tr), got):
            if not _close(e, g):
                V.append(Violation("R2.transfer", "design",
                                   f"{label} step: op graph implies {name} = {e:.6g}, integrator charges {g:.6g}"))


def _implied(design, lib, integ, r, ctx, V, implied):
    """Implied costs at one decode step r (context ctx), checked against r's terms."""
    spec = integ.model
    kv_app = 0.0
    for name, kind, tname, sec in r.per_op:
        t = lib[tname]
        if kind != "kv_append" or t.on_host:
            continue
        kv_app += sec
        if t.kind == "crossbar":
            k_rows = 1 if t.p("transpose_write", False) else spec.head_dim
            want = xbar_write_rows_time(t, k_rows + 1)
            ok = _close(sec, want)
            implied.append(ImpliedCost("kv_write", name, tname, sec, "write", ok,
                                       f"{k_rows} K row(s) + 1 V row per token"))
            if not ok:
                V.append(Violation("R2.implied", name, f"KV write on {tname} should cost {want:.4g} s per "
                                                       f"token, integrator charges {sec:.4g} s"))
    # Static weights past the wall are rewritten every step (integrator: `over` crossbar keys).
    reprog = 0.0
    for k, c in r.capacity.items():
        if c["needed"] <= c["have"]:
            continue
        name, _, unit = k.partition(":")
        t = lib[name]
        if unit == "bytes":
            implied.append(ImpliedCost("kv_spill", name, name, 0.0, name, True,
                                       f"{c['needed'] - c['have']:.3g} B past local memory, KV read at link "
                                       f"bandwidth"))
            continue
        ports = max(1, t.p("write_parallel_arrays", 1))
        sec = math.ceil((c["needed"] - c["have"]) / ports) * xbar_write_rows_time(t, t.p("rows"))
        reprog += sec
        implied.append(ImpliedCost("reprogram", name, name, sec, "write", True,
                                   f"{c['needed'] - c['have']:.0f} arrays past capacity, rewritten every step"))
    want = kv_app + reprog
    got = r.term("write")
    if not _close(want, got):
        V.append(Violation("R2.implied", "design",
                           f"write term {got:.6g} s at ctx {ctx} != KV appends {kv_app:.6g} s + reprogram "
                           f"{reprog:.6g} s"))
        for c in implied:
            if c.term == "write":
                c.charged = False
    for tname in sorted(set(design.mapping.values()) | {design.host}):
        t = lib.get(tname)
        if t is not None and t.p("host_scatter", False):
            n = design.units.get(tname, 0)
            implied.append(ImpliedCost("scatter", tname, tname,
                                       t.p("xfer_s_per_unit", design.link.alpha_s_per_unit) * n, "transfer",
                                       True, f"host-mediated scatter to {n} units, per transfer into {tname}"))


def _r3(design, integ, states, V, checked):
    """Capacity at every state. Need is nondecreasing in ctx (weights fixed, KV arrays and
    bytes grow with ctx), so it fits everywhere iff it fits at the largest state."""
    if not states:
        return
    top = max(states)
    cap = integ.capacity(design, top[0])
    res = integ.data_in_pim(design, top[0])
    checked["max_state"] = {"ctx": top[0], "session": top[1], "turn": top[2], "D_bytes": res.D,
                            "D_wall_bytes": res.D_wall, "capacity": cap}
    over = {k: c for k, c in cap.items() if c["needed"] > c["have"]}
    if not over:
        return
    ctxs = sorted({s[0] for s in states})
    lo, hi = 0, len(ctxs) - 1                   # first ctx that is over
    while lo < hi:
        mid = (lo + hi) // 2
        if integ.capacity_over(design, ctxs[mid]):
            hi = mid
        else:
            lo = mid + 1
    first = ctxs[lo]
    who = min((s for s in states if s[0] == first), key=lambda s: (s[1], s[2]))
    for k, c in over.items():
        name, _, unit = k.partition(":")
        what = "bytes" if unit == "bytes" else "arrays"
        tensors = res.tensors.get(name, {})
        V.append(Violation("R3.capacity", name,
                           f"over capacity from ctx {first} ({who[1]} turn {who[2]}); at ctx {top[0]} needs "
                           f"{c['needed']:.4g} {what}, has {c['have']:.4g} (holds "
                           + ", ".join(f"{tn} {b / 1e9:.2f} GB" for tn, b in sorted(tensors.items()))
                           + ")"))


def design_area_mm2(design, lib):
    """Space.area_mm2's formula on a built design: units x area of every tech the mapping uses."""
    used = set(design.mapping.values())
    return sum(lib[t].area_mm2_per_unit * n for t, n in design.units.items() if t in used)


def _r4(design, lib, caps, area, effort, V, checked):
    if area is None:
        area = design_area_mm2(design, lib)
    if effort is None:   # techs only; link and knob efforts need the Space point
        effort = sum(lib[t].effort_months for t in set(design.mapping.values()) - {design.host} if t in lib)
        checked["effort_lower_bound_only"] = True
    checked["area_mm2"], checked["effort_months"] = area, effort
    if area > caps.area_mm2:
        V.append(Violation("R4.area", "design", f"area {area:.0f} > cap {caps.area_mm2:.0f} mm2"))
    if effort > caps.effort_months:
        V.append(Violation("R4.effort", "design", f"effort {effort:.1f} > cap {caps.effort_months:.1f} months"))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def verify(design, lib, model, sessions, caps: Caps = None, *, area_mm2=None, effort_months=None,
           ops_cache=None, prefill_chunk=256) -> VerifyResult:
    """Check design (on library lib, for model) against the four rules over the sessions' states."""
    caps = caps or Caps()
    integ = Integrator(model, lib)
    if ops_cache is not None:
        integ._ops_cache = ops_cache
    states = trace_states(sessions)
    if caps.min_ctx:
        states.append((caps.min_ctx, "min_ctx", 0))
    top = max(s[0] for s in states) if states else 1
    V: List[Violation] = []
    implied: List[ImpliedCost] = []
    checked: Dict = {}

    ops = integ.ops(1, top)
    _r1(design, lib, ops, V)
    _r2_static(design, lib, ops, V, implied)
    timeable = not any(v.rule in ("R1.support", "R1.write_mechanism") for v in V)
    if timeable:
        steps = {"decode": (1, top, None),
                 "prefill": (prefill_chunk, max(top, prefill_chunk), 1)}
        _r2_charged(design, lib, integ, V, checked, steps)
        _implied(design, lib, integ, integ.step(design, 1, top), top, V, implied)
        _r3(design, integ, states, V, checked)
    else:
        checked["not_timed"] = "R1 violation: the integrator cannot time this design"
    _r4(design, lib, caps, area_mm2, effort_months, V, checked)
    return VerifyResult(legal=not V, violations=V, implied_costs=implied, checked=checked)


def verify_point(space, point, sessions, caps: Caps = None, **kw) -> VerifyResult:
    """verify() for a Space point: the design and library from Space.to_design, area and
    effort from Space.area_mm2 / effort_months (the evaluator's numbers)."""
    p = space.canonical(point)
    design, lib = space.to_design(p)
    return verify(design, lib, space.model, sessions, caps, area_mm2=space.area_mm2(p),
                  effort_months=space.effort_months(p), **kw)
