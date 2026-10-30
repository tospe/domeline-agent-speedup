"""The integrator's special cases (docs/theory.md): VIPER's closed form and per-op Roofline.

Each test builds the degenerate design and compares Integrator.step to the closed form
written out independently here. Where the integrator does not reduce (past the wall,
the gather side of the alpha term), the test pins down what it computes instead.
"""

import math
from pathlib import Path

import pytest

from harness import ops as O
from harness.design import Design, Link
from harness.integrator import Integrator
from harness.model import ModelSpec
from harness.tech import Tech, load_library, xbar_arrays_for, xbar_mvm_time

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/localhome/tsousape/models/mini-coder-4b"
CTX = 4096


@pytest.fixture(scope="module")
def env():
    spec = ModelSpec.from_hf_config(MODEL)
    lib = load_library(ROOT / "lib/tech")
    return spec, lib


def host_time(r, host="cpu"):
    return sum(s for _, _, tech, s in r.per_op if tech == host)


# ---------------------------------------------------------------------------
# (i) One offloaded kernel, one step: VIPER, T = t_host + m (t_in + t_prog + t_comp + t_out)
# ---------------------------------------------------------------------------

def viper(t_host, t_in, t_prog, t_comp, t_out, D, C):
    m = math.ceil(D / C)
    return t_host + m * (t_in + t_prog + t_comp + t_out)


LINK = Link(bw_Bps=64e9, launch_s=1e-6, sync_s=1e-6, t0_s=0.5e-6)


def test_viper_crossbar_below_wall(env):
    """lm_head alone on crossbars, resident (D <= C): exact, with launch in t_in, sync in t_out."""
    spec, lib = env
    t = lib["xbar_isaac"]
    need = xbar_arrays_for(t, spec.vocab, spec.hidden, 8)
    d = Design(mapping={"lm_head": "xbar_isaac"}, units={"xbar_isaac": need}, link=LINK)
    r = Integrator(spec, lib).step(d, 1, CTX)
    a = O.ACT_BYTES
    t_in = LINK.launch_s + LINK.t0_s + spec.hidden * a / LINK.bw_Bps
    t_out = LINK.sync_s + LINK.t0_s + spec.vocab * a / LINK.bw_Bps
    t_comp = xbar_mvm_time(t, spec.hidden)
    T = viper(host_time(r), t_in, 0.0, t_comp, t_out, D=need, C=need)
    assert r.total == pytest.approx(T, rel=1e-12)
    assert r.term("xbar_isaac") == pytest.approx(t_comp, rel=1e-12)
    assert r.term("launch") + r.term("transfer") == pytest.approx(t_in + t_out, rel=1e-12)
    assert r.crossings == 2


def test_viper_alpha_only_on_scatter(env):
    """UPMEM-style host scatter: alpha*N is charged on the way in, not on the way out."""
    spec, lib = env
    N = 2560
    link = Link(bw_Bps=64e9, launch_s=1e-6, sync_s=1e-6, t0_s=1.82e-3, alpha_s_per_unit=1.68e-6)
    d = Design(mapping={"lm_head": "upmem_dpu"}, units={"upmem_dpu": N}, link=link)
    r = Integrator(spec, lib).step(d, 1, CTX)
    a = O.ACT_BYTES
    t_in = link.launch_s + link.t0_s + link.alpha_s_per_unit * N + spec.hidden * a / link.bw_Bps
    t_out = link.sync_s + link.t0_s + spec.vocab * a / link.bw_Bps          # no alpha * N
    assert r.term("launch") + r.term("transfer") == pytest.approx(t_in + t_out, rel=1e-12)
    assert r.total == pytest.approx(host_time(r) + t_in + r.term("upmem_dpu") + t_out, rel=1e-12)


def test_viper_past_wall_is_additive(env):
    """Past the wall the integrator reprograms only the overflow, once per step.

    It charges T_1 + t_rp with t_rp = ceil((D - C) / P_w) * R * t_w, not VIPER's
    m passes. This is a lower bound on VIPER whenever VIPER's per-pass programming
    rewrites C arrays: t_rp <= (m - 1) t_prog,pass.
    """
    spec, lib = env
    t = lib["xbar_isaac"]
    need = xbar_arrays_for(t, spec.vocab, spec.hidden, 8)
    integ = Integrator(spec, lib)
    full = integ.step(Design(mapping={"lm_head": "xbar_isaac"}, units={"xbar_isaac": need}, link=LINK), 1, CTX)
    for C in (need // 2, need // 3):
        r = integ.step(Design(mapping={"lm_head": "xbar_isaac"}, units={"xbar_isaac": C}, link=LINK), 1, CTX)
        ports = t.p("write_parallel_arrays")
        t_rp = math.ceil((need - C) / ports) * t.p("rows") * t.p("t_write_row_s")
        assert r.total == pytest.approx(full.total + t_rp, rel=1e-12)
        T1 = full.total - host_time(full)
        t_prog_pass = math.ceil(C / ports) * t.p("rows") * t.p("t_write_row_s")
        T_viper = host_time(full) + math.ceil(need / C) * (T1 + t_prog_pass)
        assert r.total < T_viper


def test_viper_per_layer_kernel(env):
    """o_proj offloaded in every layer: VIPER's form once per layer (L invocations, m = 1)."""
    spec, lib = env
    t = lib["xbar_isaac"]
    need = spec.layers * xbar_arrays_for(t, spec.hidden, spec.q_dim, 8)
    d = Design(mapping={"o_proj": "xbar_isaac"}, units={"xbar_isaac": need}, link=LINK)
    r = Integrator(spec, lib).step(d, 1, CTX)
    a = O.ACT_BYTES
    t_in = LINK.launch_s + LINK.t0_s + spec.q_dim * a / LINK.bw_Bps
    t_out = LINK.sync_s + LINK.t0_s + spec.hidden * a / LINK.bw_Bps
    t_comp = xbar_mvm_time(t, spec.q_dim)
    assert r.total == pytest.approx(host_time(r) + spec.layers * (t_in + t_comp + t_out), rel=1e-12)


# ---------------------------------------------------------------------------
# (ii) Below the wall, no host work, no link: per-op Roofline, sum_o max(F_o/P, B_o/BW)
# ---------------------------------------------------------------------------

ALL = ("vector", "gather", "mvm_static", "mvm_dynamic", "kv_append")


def op_bytes(op, wb, kb):
    """B_o: the bytes the integrator charges a roofline unit for op o."""
    if op.cls == O.MVM_STATIC:
        return op.shape[0] * op.shape[1] * wb / 8
    if op.cls == O.MVM_DYNAMIC:
        return op.kv_read_elems * kb / 8
    if op.cls == O.KV_APPEND:
        return op.kv_write_elems * kb / 8
    return sum(b for _, b in op.reads) + sum(b for _, b in op.writes)


def roofline(ops, P, BW, wb, kb, fixed=0.0):
    return sum(max(op.flops / P, op_bytes(op, wb, kb) / BW) + fixed for op in ops)


@pytest.mark.parametrize("n_new,ctx", [(1, 1), (1, CTX), (1, 40000), (256, 2048)])
def test_roofline_one_device(env, n_new, ctx):
    spec, _ = env
    P, BW = 2e12, 1e12
    dev = Tech("dev", "digital", ALL, {"ops_per_s": P, "mem_bw_Bps": BW, "op_overhead_s": 0.0})
    lib = {"dev": dev, "cpu": Tech("cpu", "cpu", ALL, {"ops_per_s": 1e11, "mem_bw_Bps": 5e10})}
    integ = Integrator(spec, lib)
    for launch in (0.0, 1e-6):
        d = Design(mapping={"*": "dev"}, link=Link(launch_s=launch, sync_s=launch))
        r = integ.step(d, n_new, ctx)
        T = roofline(integ.ops(n_new, ctx), P, BW, d.weight_bits, d.kv_bits)
        # The only link cost left is the one launch that starts the step (no sync: it never returns).
        assert r.total == pytest.approx(T + launch, rel=1e-12)
        assert r.term("transfer") == 0 and r.term("host") == 0 and not r.over_capacity


def test_roofline_on_host_roof(env):
    """Everything on the host's roofline model (no measured profile): Roofline + a fixed cost per op."""
    spec, lib = env
    h = lib["cpu_roofline"]
    integ = Integrator(spec, lib)
    d = Design(mapping={}, host="cpu_roofline")
    r = integ.step(d, 1, CTX)
    ops = integ.ops(1, CTX)
    T = roofline(ops, h.p("ops_per_s"), h.p("mem_bw_Bps"), d.weight_bits, d.kv_bits,
                 fixed=h.p("op_overhead_s"))
    assert r.total == pytest.approx(T, rel=1e-12)
    assert set(r.terms) == {"host"}
