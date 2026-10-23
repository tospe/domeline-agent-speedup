"""Fig: speedup vs coprocessor area with the sourced area knobs.

The step line is the exact Pareto front of the crossbar subspace (every
area-relevant knob, the rest at the headline design). Markers are the designs
the Domeline walk and TPE return under each area cap, searching the full space.
Labels name what each front step changes relative to the headline design.
Data: results/search/area_<tag>.json (experiments/area_sweep.py).
  python figs/fig_area.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

GA102_MM2 = 628.4
SLOT = {"walk": 0, "tpe": 1}
NAME = {"walk": "Domeline walk (per area cap)", "tpe": "TPE (per area cap)"}
SHORT = {"xbar_tile": {"lean": "lean tile"}, "cell_bits": {4: "4-b cells"},
         "adc_per_ima": {4: "4 ADCs", 2: "2 ADCs"}, "kv_bits": {16: "fp16 KV"},
         "xbar_tech": {"xbar_isaac": "no transpose"}}


def front(rows):
    out, best = [], -1.0
    for r in sorted(rows, key=lambda r: (round(r["area_mm2"], -1), -r["speedup"])):
        if r["speedup"] > best * 1.005:
            out.append(r)
            best = r["speedup"]
    return out


def describe(p):
    parts = [SHORT[k][p[k]] for k in SHORT if p.get(k) in SHORT[k]]
    return ", ".join(parts) or "headline tile"


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"area_{tag}.json").read_text())
    sub = [r for r in d["sub"] if r["feasible"] and r["speedup"] >= 1.0]
    f = front(sub)
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.4))
    fa = np.array([r["area_mm2"] for r in f])
    fsx = np.array([r["speedup"] for r in f])
    ax.step(np.append(fa, fa[-1] * 1.6), np.append(fsx, fsx[-1]), where="post", color=fs.INK_3, lw=1.0,
            label="Exact front (crossbar subspace)", zorder=2)
    for m in ("walk", "tpe"):
        rs = [r for r in d["rows"] if r["method"] == m and r["best"] and r["best"]["speedup"] >= 1.0]
        st = fs.series(SLOT[m])
        ax.plot([r["best"]["area_mm2"] for r in rs], [r["best"]["speedup"] for r in rs], ls="none",
                marker=st["marker"], color=st["color"], markersize=4.5 if m == "walk" else 3.5,
                markeredgecolor=fs.SURFACE, markeredgewidth=0.4, label=NAME[m], zorder=4 if m == "walk" else 3)
    # Name the knob settings at the corners of the front: one label per distinct setting,
    # left of its point (the region left of the front is empty).
    labels = {}
    for r in f:
        labels[describe(r["point"])] = r          # keep the fastest point of each setting
    low = {k: r for k, r in labels.items() if "no transpose" in k}
    for lab, r in labels.items():
        if lab in low:
            continue
        up = r["speedup"] == fsx.max()
        ax.annotate(f"{lab}: {r['speedup']:.1f}×", (r["area_mm2"], r["speedup"]),
                    xytext=(-6, 5 if up else 0), textcoords="offset points", fontsize=5.8, color=fs.INK_2,
                    ha="right", va="bottom" if up else "center")
    if low:
        r = max(low.values(), key=lambda r: r["speedup"])
        r0 = min(low.values(), key=lambda r: r["area_mm2"])
        ax.annotate(f"< 2.5K mm²: no transposed\nK writes, ≤ {r['speedup']:.1f}×", (r0["area_mm2"], r0["speedup"]),
                    xytext=(-14, 9), textcoords="offset points", fontsize=5.8, color=fs.INK_2, ha="center", va="bottom")
    ax.axvline(GA102_MM2, color=fs.INK_2, lw=0.7, ls="--")
    ax.annotate("RTX 3080 die\n(628 mm²)", (GA102_MM2, fsx.max() * 1.15), xytext=(3, 0), textcoords="offset points",
                fontsize=6.3, color=fs.INK_2, va="top", ha="left")
    ax.axhline(1.0, color=fs.INK_2, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(500, max(fa.max(), 1e4) * 1.8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}K"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylim(0, fsx.max() * 1.18)
    ax.set_xlabel("Coprocessor area (mm², ISAAC 32 nm)")
    ax.set_ylabel("Speedup over RTX 3080")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=6, columnspacing=1.0)
    print(fs.save(fig, f"area_{tag}"))
    for r in f:
        print(f"  front {r['area_mm2']:7.0f} mm2 {r['speedup']:.2f}x  {describe(r['point'])}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
