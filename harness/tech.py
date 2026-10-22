"""Technology cost models and the YAML library loader.

Every technology is one YAML entry (see lib/tech/*.yaml). Its `kind` picks
the cost model below; `units` in a Design sets how many arrays or lanes a
design instantiates. All times are seconds, sizes bytes, rates per second.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

from . import ops as O


@dataclass
class Tech:
    name: str
    kind: str                     # crossbar | digital | dram_pim | cpu | cam
    supports: tuple               # op classes it can run
    params: Dict
    source: str = ""
    maturity: str = "projected"   # published-silicon | simulated | projected
    effort_months: float = 0.0
    area_mm2_per_unit: float = 0.0
    notes: str = ""

    @property
    def on_host(self):
        return self.kind == "cpu"

    def can_run(self, cls):
        return cls in self.supports

    def p(self, key, default=None):
        return self.params.get(key, default)


# ---------------------------------------------------------------------------
# Crossbar (analog MVM, ISAAC/PUMA style)
# ---------------------------------------------------------------------------

def xbar_arrays_for(tech, M, N, bits):
    """Arrays to hold an M x N matrix (N = inputs on rows) at `bits` per weight."""
    rows, cols, bpc = tech.p("rows"), tech.p("cols"), tech.p("bits_per_cell")
    return math.ceil(N / rows) * math.ceil(M * bits / (cols * bpc))


def xbar_mvm_time(tech, N, tokens=1, repeats=1):
    """One MVM pass with the matrix resident: all arrays in parallel.

    Inputs stream bit-serially through the DACs; partial sums from the
    ceil(N/rows) row tiles are reduced in a digital adder tree.
    """
    cycles = math.ceil(tech.p("input_bits", 8) / tech.p("dac_bits", 1))
    tiles = math.ceil(N / tech.p("rows"))
    reduce_t = math.ceil(math.log2(tiles)) * tech.p("t_add_s", 1e-9) if tiles > 1 else 0.0
    return tokens * repeats * (cycles * tech.p("t_read_s") + reduce_t)


def xbar_write_rows_time(tech, n_rows_serial):
    """Writing rows that land in the same array is serial; arrays write in parallel."""
    return n_rows_serial * tech.p("t_write_row_s")


# ---------------------------------------------------------------------------
# Tiled digital GEMV engine (TERINT-GEMV bit-serial accelerator)
# ---------------------------------------------------------------------------

def tiled_gemv_time(tech, M, N, tokens=1):
    """Cycle formula from soc_inference_estimator.compute_accelerator_latency."""
    m_passes = math.ceil(M / tech.p("num_cols"))
    n_passes = math.ceil(N / tech.p("inputs_per_chunk"))
    cycles = m_passes * n_passes * tech.p("cycles_per_chunk") + m_passes * tech.p("output_cycles", 0)
    return tokens * cycles / tech.p("clock_hz")


# ---------------------------------------------------------------------------
# Generic roofline-style unit (digital vector unit, ASIC, CPU, DRAM-PIM)
# ---------------------------------------------------------------------------

def roofline_time(flops, bytes_moved, ops_per_s, bytes_per_s, fixed_s=0.0):
    t_c = flops / ops_per_s if ops_per_s else 0.0
    t_m = bytes_moved / bytes_per_s if bytes_per_s else 0.0
    return max(t_c, t_m) + fixed_s


def unit_rates(tech, units, cls):
    """(ops/s, bytes/s) for `units` lanes of a digital or DRAM-PIM tech on op class cls."""
    per_cls = tech.p("ops_per_s_by_class", {}) or {}
    ops_per_s = per_cls.get(cls, tech.p("ops_per_s", 0.0)) * units
    if tech.kind == "dram_pim":
        bw = tech.p("bank_bw_Bps") * units
    else:
        bw = tech.p("mem_bw_Bps", 0.0) * (units if tech.p("bw_scales_with_units", False) else 1)
    return ops_per_s, bw


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def load_library(paths) -> Dict[str, Tech]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.yaml")) if p.is_dir() else [p]
    lib: Dict[str, Tech] = {}
    for f in files:
        for e in yaml.safe_load(f.read_text()) or []:
            name = e.pop("name")
            if name in lib:
                raise ValueError(f"duplicate tech {name!r} in {f}")
            lib[name] = Tech(
                name=name, kind=e.pop("kind"), supports=tuple(e.pop("supports", ())),
                source=e.pop("source", ""), maturity=e.pop("maturity", "projected"),
                effort_months=float(e.pop("effort_months", 0.0)),
                area_mm2_per_unit=float(e.pop("area_mm2_per_unit", 0.0)),
                notes=e.pop("notes", ""), params=_floats(e.pop("params", {})),
            )
            if e:
                raise ValueError(f"unknown fields for tech {name!r}: {sorted(e)}")
    return lib


def _floats(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _floats(v)
        elif isinstance(v, str):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out
