"""A design point: which tech runs each op, how much of each tech, and the link."""

from dataclasses import dataclass, field, replace
from typing import Dict


@dataclass(frozen=True)
class Link:
    """CPU <-> coprocessor coupling (VIPER task-offloading mode)."""
    bw_Bps: float = 64e9          # link bandwidth
    launch_s: float = 1e-6        # CPU issues an offload command
    sync_s: float = 1e-6          # CPU learns the offload finished
    t0_s: float = 0.0             # fixed cost per transfer event
    alpha_s_per_unit: float = 0.0 # host-mediated scatter/gather cost per unit (UPMEM: 1.68 us/DPU)
    noc_bw_Bps: float = 1e12      # between techs inside the coprocessor


@dataclass(frozen=True)
class Design:
    mapping: Dict[str, str]                     # op name | kind | class | "*" -> tech name
    units: Dict[str, int] = field(default_factory=dict)
    link: Link = Link()
    weight_bits: int = 8
    kv_bits: int = 16
    host: str = "cpu"
    label: str = ""

    def tech_for(self, op):
        m = self.mapping
        for key in (op.name, op.kind, op.cls, "*"):
            if key in m:
                return m[key]
        return self.host

    def with_(self, **kw):
        return replace(self, **kw)

    def remap(self, **assign):
        return replace(self, mapping={**self.mapping, **assign})

    def resize(self, **units):
        return replace(self, units={**self.units, **units})

    def techs_used(self):
        return sorted(set(self.mapping.values()) | {self.host})
