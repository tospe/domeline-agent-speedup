"""Fig. 1 (draft, E2): one design read with the Roofline and with the Domeline.

The design is the headline design (crossbars for every MVM and attention,
64-instance nonlinear unit for softmax/vector ops, CXL, int8 KV) with its
crossbar under-provisioned to 0.75x of what a 64K context needs: 25% less
crossbar area. Per decoded token vs context, one column, two panels on the
same log axis:
  (a) Roofline view (host ops on the host roof): per-op max(flops/peak,
      bytes/BW) by device. The softmax unit's compute is the bottleneck; the
      design looks as fast as the full-size one, and the KV cache filling the
      arrays is invisible.
  (b) Domeline view (the integrator): the crossbar read floor, transfers that
      grow with context, and, past the capacity wall at ~35K tokens, the
      reprogramming writes that bind.
Data: computed from the harness (same model/library as the sweep); session
numbers from experiments/roofline.py over the 45 search sessions.
  python figs/fig_rvd_teaser.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import gpu, library, model, sessions, split  # noqa: E402
from harness import figstyle as fs  # noqa: E402
from harness.evaluate import Evaluator  # noqa: E402
from harness.integrator import Integrator  # noqa: E402
from harness.space import START, Space, canonical  # noqa: E402
from roofline import VARIANTS, Roofline, SessionWeights  # noqa: E402

DESIGN = {"mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
          "softmax": "star", "vector": "star", "edge": "cpu", "star_units": 64, "link": "cxl", "kv_bits": 8,
          "xbar_scale": 0.75}
FULL = {**DESIGN, "xbar_scale": 1.0}
CTX = np.unique(np.concatenate([np.geomspace(1024, 65536, 40).astype(int), [35000, 35457, 36000]]))

# Shared series slots so a cost keeps its color across the two panels.
SLOT = {"xbar": 0, "star": 1, "transfer": 2, "write": 3, "host": 4}
NAME = {"xbar": "Crossbar", "star": "Softmax unit", "transfer": "Transfers", "write": "Writes",
        "host": "Host + launch"}


def roofline_parts(r):
    return {"xbar": r.get("xbar_agg", 0.0), "star": r.get("star:c", 0.0) + r.get("star:b", 0.0),
            "host": r.get("host:c", 0.0) + r.get("host:b", 0.0)}


def domeline_parts(terms):
    out = {"xbar": 0.0, "star": 0.0, "transfer": 0.0, "write": 0.0, "host": 0.0}
    for k, v in terms.items():
        if k.startswith("xbar"):
            out["xbar"] += v
        elif "star" in k:
            out["star"] += v
        elif k in ("host", "launch"):
            out["host"] += v
        else:
            out[k] += v
    return out


def session_numbers(space, train, g, p):
    ev = Evaluator(space, train, g)
    e = ev(p)
    d, lib = space.to_design(canonical(p))
    rf = Roofline(space.model)
    tot = SessionWeights(train).replay(lambda n, c, lt: rf.step(d, lib, n, c, lt))
    rf_llm = sum(tot.get(k, 0.0) for k in VARIANTS["host"])
    return {"dl_speedup": e.speedup_vs_gpu, "dl_llm": e.llm_s, "wall": e.wall_ctx,
            "rf_llm": rf_llm, "rf_speedup": ev.gpu_time / (rf_llm + ev.tools_time),
            "area": space.area_mm2(canonical(p))}


def main():
    spec, lib, g = model(), library(), gpu()
    tr, te = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in tr + te))
    p = canonical({**START, **DESIGN})
    d, dlib = space.to_design(p)
    integ, rf = Integrator(spec, dlib), Roofline(spec)
    R = {k: [] for k in ("xbar", "star", "host", "total")}
    D = {k: [] for k in ("xbar", "star", "transfer", "write", "host", "total")}
    for c in CTX:
        r = roofline_parts(rf.step(d, dlib, 1, int(c)))
        for k, v in r.items():
            R[k].append(v * 1e3)
        R["total"].append(sum(r.values()) * 1e3)
        s = integ.step(d, 1, int(c))
        for k, v in domeline_parts(s.terms).items():
            D[k].append(v * 1e3)
        D["total"].append(s.total * 1e3)
    wall = next(int(c) for c in CTX if integ.capacity_over(d, int(c)))
    sn = session_numbers(space, tr, g, p)
    sf = session_numbers(space, tr, g, canonical({**START, **FULL}))
    print(f"design 0.75x: Domeline {sn['dl_speedup']:.2f}x (wall {sn['wall']}), Roofline {sn['rf_speedup']:.2f}x, "
          f"{sn['area']:.0f} mm2, LLM time Roofline {sn['rf_llm']:.1f} s vs Domeline {sn['dl_llm']:.1f} s; full-size: Domeline {sf['dl_speedup']:.2f}x, Roofline {sf['rf_speedup']:.2f}x, "
          f"{sf['area']:.0f} mm2")
    i64 = len(CTX) - 1
    print(f"at 64K: Roofline {R['total'][i64]:.3f} ms, Domeline {D['total'][i64]:.3f} ms "
          f"(write {D['write'][i64]:.3f}); at 16K: Roofline {R['total'][np.searchsorted(CTX, 16384)]:.3f}, "
          f"Domeline {D['total'][np.searchsorted(CTX, 16384)]:.3f}")

    fs.apply()
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.COL_W, 2.05), sharey=True)
    x = CTX / 1024
    for ax, data, keys, title in (
            (a, R, ("xbar", "star", "host"), "(a) Roofline"),
            (b, D, ("xbar", "transfer", "star", "write", "host"), "(b) Domeline")):
        for k in keys:
            y = np.array(data[k])
            if not (y > 0).any():
                continue
            st = fs.series(SLOT[k])
            ax.plot(x, np.where(y > 0, y, np.nan), color=st["color"], ls=st["linestyle"], lw=0.9)
        ax.plot(x, data["total"], color=fs.INK, lw=1.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16, 64], ["1K", "4K", "16K", "64K"])
        ax.set_xlabel("Context (tokens)")
        ax.set_title(title, loc="left", color=fs.INK_2)
    a.set_yscale("log")
    a.set_ylim(5e-4, 20)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_ylabel("Time per decoded token (ms)")

    # Direct labels (ink), placed on each line where it is clear of the others.
    def at(data, k, xx):
        return float(np.interp(xx, x, data[k]))

    def lab(ax, xx, yy, text, dy=0, ha="center"):
        ax.annotate(text, (xx, yy), xytext=(0, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=6, color=fs.INK)
    lab(a, 16, at(R, "total", 16), "total", dy=7, ha="right")
    lab(a, 16, at(R, "star", 16), "softmax unit", dy=-7, ha="left")
    lab(a, 2, at(R, "host", 2), "host", dy=-6)
    lab(a, 4, at(R, "xbar", 4), "crossbar", dy=6)
    a.annotate("bound: softmax unit\n(compute)", (1.05, 2.5), fontsize=6, color=fs.INK_2, ha="left", va="center",
               linespacing=1.0)
    lab(b, 60, D["total"][-1], "total", dy=0, ha="right")
    lab(b, 4, at(D, "xbar", 4), "crossbar", dy=-6)
    lab(b, 4, at(D, "transfer", 4), "transfers", dy=6)
    lab(b, 30, at(D, "star", 30), "softmax unit", dy=-8, ha="right")
    lab(b, 8, at(D, "host", 8), "host + launch", dy=-6)
    lab(b, 2, at(D, "write", 2), "writes", dy=-6)
    b.axvline(wall / 1024, color=fs.INK_3, lw=0.7, ls=":")
    b.annotate(f"capacity\nwall {wall / 1024:.0f}K", (wall / 1024, 2.2), xytext=(-2, 0), textcoords="offset points",
               ha="right", va="center", fontsize=6, color=fs.INK_2, linespacing=1.0)
    a.annotate(f"session: {sn['rf_speedup']:.0f}x vs 3080", (1.05, 9), fontsize=6, color=fs.INK, ha="left")
    b.annotate(f"session: {sn['dl_speedup']:.1f}x vs 3080", (1.05, 9), fontsize=6, color=fs.INK, ha="left")
    fig.tight_layout(w_pad=0.6)
    print(fs.save(fig, "rvd_teaser"))


if __name__ == "__main__":
    main()
