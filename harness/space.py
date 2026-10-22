"""The design space: discrete knobs -> Design, plus area and effort.

A point is a dict knob -> option. `to_design` turns it into a Design the
integrator can time. Crossbar arrays are sized relative to what the design
needs at the longest context in the workload (`ref_ctx`), so `xbar_scale`
1.0 means "exactly fits" and <1 means the weights or KV do not fit.

Link presets and memory/link effort costs are assumptions (see LINKS, EFFORT);
every number is a knob value the user can change.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List

from .design import Design, Link
from .ops import forward_ops

WEIGHT_MVMS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VECTOR_OPS = ("rmsnorm", "qk_norm", "rope", "act", "residual")
ATTN_OPS = ("attn_score", "attn_value", "kv_append")
EDGE_OPS = ("embed", "sample")

# CPU <-> coprocessor coupling classes. VERIFY: placeholder numbers to be replaced by
# measured/published values for CXL.mem, on-package (UCIe-class) and on-die links.
LINKS = {
    "cxl":     Link(bw_Bps=64e9,  launch_s=1.0e-6, sync_s=1.0e-6, t0_s=0.5e-6),
    "package": Link(bw_Bps=256e9, launch_s=0.3e-6, sync_s=0.3e-6, t0_s=0.1e-6),
    "die":     Link(bw_Bps=1e12,  launch_s=50e-9,  sync_s=50e-9,  t0_s=0.0),
}

# Effort in months to adopt a link class or a KV memory bandwidth class (assumptions).
EFFORT = {
    "link": {"cxl": 0, "package": 3, "die": 9},
    "vfu_bw": {64e9: 0, 256e9: 1, 1e12: 3},
}

KNOBS: Dict[str, List] = {
    "mvm": ["xbar", "cpu"],                      # weight MVMs of every layer
    "lm_head": ["xbar", "cpu"],
    "attention": ["cpu", "xbar", "vfu"],         # q.K^T, p.V and the KV appends
    "xbar_tech": ["xbar_isaac", "xbar_transpose"],
    "softmax": ["cpu", "vfu", "star"],
    "vector": ["cpu", "vfu", "star"],            # norms, RoPE, SiLU*up, residuals
    "edge": ["cpu", "vfu"],                      # embedding lookup, sampling
    "xbar_scale": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "vfu_units": [1, 4, 16, 64, 256],
    "vfu_bw": [64e9, 256e9, 1e12],
    "vfu_mem": [4e9, 16e9, 64e9],
    "star_units": [1, 4, 16, 64],
    "link": ["cxl", "package", "die"],
    "kv_bits": [16, 8],      # int4 KV left out: its accuracy cost is not modeled
}

# Knobs that matter only when another knob takes a value (for counting distinct designs).
ACTIVE_IF = {
    "xbar_tech": lambda p: "xbar" in (p["mvm"], p["lm_head"], p["attention"]),
    "xbar_scale": lambda p: "xbar" in (p["mvm"], p["lm_head"], p["attention"]),
    "vfu_units": lambda p: "vfu" in (p["attention"], p["softmax"], p["vector"], p["edge"]),
    "vfu_bw": lambda p: p["attention"] == "vfu",
    "vfu_mem": lambda p: p["attention"] == "vfu",
    "star_units": lambda p: "star" in (p["softmax"], p["vector"]),
}

START = {  # step 1 of the walk: crossbars run the weight MVMs, the host runs the rest
    "mvm": "xbar", "lm_head": "cpu", "attention": "cpu", "xbar_tech": "xbar_isaac",
    "softmax": "cpu", "vector": "cpu", "edge": "cpu", "xbar_scale": 1.25,
    "vfu_units": 1, "vfu_bw": 64e9, "vfu_mem": 4e9, "star_units": 1, "link": "cxl", "kv_bits": 16,
}


def canonical(p):
    """Point with inactive knobs reset, so equivalent designs share one key."""
    q = dict(p)
    for k, active in ACTIVE_IF.items():
        if not active(q):
            q[k] = START[k]
    return q


def key(p):
    q = canonical(p)
    return tuple(q[k] for k in KNOBS)


def n_designs():
    total = 1
    for v in KNOBS.values():
        total *= len(v)
    distinct = len({key(dict(zip(KNOBS, vals))) for vals in itertools.product(*KNOBS.values())}) \
        if total < 3_000_000 else None
    return total, distinct


@dataclass
class Space:
    lib: Dict
    model: object
    ref_ctx: int
    host: str = "cpu"
    vfu: str = "vfu_puma"
    star: str = "softmax_star"

    def _mapping(self, p):
        xb = p["xbar_tech"]
        tech = {"xbar": xb, "cpu": self.host, "vfu": self.vfu, "star": self.star}
        m = {}
        for k in WEIGHT_MVMS:
            m[k] = tech[p["mvm"]]
        m["lm_head"] = tech[p["lm_head"]]
        for k in ATTN_OPS:
            m[k] = tech[p["attention"]]
        m["softmax"] = tech[p["softmax"]]
        for k in VECTOR_OPS:
            m[k] = tech[p["vector"]]
        for k in EDGE_OPS:
            m[k] = tech[p["edge"]]
        return m

    def xbar_need(self, p, ctx=None):
        """Arrays the crossbar part of design p needs at context ctx (default ref_ctx)."""
        from .integrator import Integrator
        d = Design(mapping=self._mapping(p), units={}, kv_bits=p["kv_bits"], host=self.host)
        cap = Integrator(self.model, self.lib).capacity(d, ctx or self.ref_ctx,
                                                        forward_ops(self.model, 1, ctx or self.ref_ctx))
        return cap.get(p["xbar_tech"], {}).get("needed", 0)

    def to_design(self, p) -> Design:
        p = canonical(p)
        units = {}
        if ACTIVE_IF["xbar_scale"](p):
            units[p["xbar_tech"]] = int(math.ceil(p["xbar_scale"] * self.xbar_need(p)))
        units[self.vfu] = p["vfu_units"]
        units[self.star] = p["star_units"]
        lib = self.lib
        if p["attention"] == "vfu":
            # The KV memory next to the vector unit is a knob of the design, not of the tech.
            lib = dict(lib)
            v = lib[self.vfu]
            lib[self.vfu] = type(v)(**{**v.__dict__, "params": {**v.params, "mem_bw_Bps": p["vfu_bw"],
                                                                 "local_mem_bytes": p["vfu_mem"]}})
        d = Design(mapping=self._mapping(p), units=units, link=LINKS[p["link"]],
                   kv_bits=p["kv_bits"], host=self.host, label=str(key(p)))
        return d, lib

    def area_mm2(self, p):
        p = canonical(p)
        d, lib = self.to_design(p)
        used = set(d.mapping.values())
        return sum(lib[t].area_mm2_per_unit * n for t, n in d.units.items() if t in used)

    def effort_months(self, p):
        p = canonical(p)
        d, lib = self.to_design(p)
        techs = set(d.mapping.values()) - {self.host}
        e = sum(lib[t].effort_months for t in techs)
        e += EFFORT["link"][p["link"]]
        if p["attention"] == "vfu":
            e += EFFORT["vfu_bw"][p["vfu_bw"]]
        return e
