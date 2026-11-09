"""H2: under soc_inference_estimator.py's assumptions, the integrator gives its numbers.

Assumptions matched: CPU time = FLOPs / GFLOP/s (no memory term), every
BitLinear on the tiled GEMV accelerator, everything else on the CPU, no
transfer (the estimator's default, --memory-model off). The estimator has no
sampling op, so ours is mapped to a zero-cost unit.
"""

import importlib.util
import math
import os
from pathlib import Path

import pytest

from harness.design import Design, Link
from harness.integrator import Integrator
from harness.model import BITNET_2B
from harness.tech import Tech

EST = Path(os.environ.get("TERINT_ESTIMATOR",
                          "/localhome/tsousape/tools/TERINT-GEMV/analysis/soc_inference_estimator.py"))
pytestmark = pytest.mark.skipif(not EST.exists(), reason="TERINT-GEMV estimator not found")


def _load_estimator():
    spec = importlib.util.spec_from_file_location("soc_est", EST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ours(cpu_gflops, accel):
    lib = {
        "cpu": Tech("cpu", "cpu", ("vector", "gather", "mvm_static", "mvm_dynamic", "kv_append"),
                    {"ops_per_s": cpu_gflops * 1e9, "mem_bw_Bps": 1.0, "compute_bound_only": True}),
        "gemv": Tech("gemv", "tiled_gemv", ("mvm_static",),
                     {"num_cols": accel.num_cols,
                      "inputs_per_chunk": accel.num_lanes * accel.elements_per_lane,
                      "cycles_per_chunk": accel.cycles_per_chunk,
                      "output_cycles": accel.output_cycle, "clock_hz": accel.clock_freq_hz}),
        "free": Tech("free", "cpu", ("vector",),
                     {"ops_per_s": float("inf"), "mem_bw_Bps": 1.0, "compute_bound_only": True}),
    }
    mapping = {k: "gemv" for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj")}
    mapping["sample"] = "free"
    link = Link(bw_Bps=float("inf"), launch_s=0.0, sync_s=0.0)
    return Integrator(BITNET_2B, lib), Design(mapping=mapping, link=link)


@pytest.mark.parametrize("seq_len", [1, 128, 512])
@pytest.mark.parametrize("cpu,accel", [("cortex-a55", "32x32"), ("cortex-a78", "128x128")])
def test_matches_estimator(seq_len, cpu, accel):
    est = _load_estimator()
    soc = est.SoCConfig(accelerator=est.ACCEL_PROFILES[accel], cpu=est.CPU_PROFILES[cpu])
    ref = est.estimate_full_inference(seq_len, soc, est.BitNetModelConfig(), memory_model=False)

    integ, design = _ours(soc.cpu.gflops, soc.accelerator)
    r = integ.step(design, n_new=seq_len, ctx=seq_len)

    assert math.isclose(r.term("gemv"), ref.accel_total_s, rel_tol=1e-9)
    assert math.isclose(r.term("host"), ref.cpu_total_s, rel_tol=1e-9)
    assert math.isclose(r.total, ref.total_time_s, rel_tol=1e-9)
