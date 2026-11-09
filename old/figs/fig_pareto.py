"""Fig: speedup vs coprocessor area for every feasible design any searcher evaluated.

The Pareto front (most speedup per area) is drawn as a step line. The RTX
3080's die (GA102, 628 mm^2 in Samsung 8 nm) is the reference; crossbar
areas are ISAAC's 32 nm tile-amortized figures, so the comparison is
node-unadjusted and favors the GPU.
Data: results/search/<tag>.json.  python figs/fig_pareto.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

GA102_MM2 = 628.4


def front(points):
    pts = sorted(points)                      # by area
    out, best = [], -1
    for a, s in pts:
        if s > best:
            out.append((a, s))
            best = s
    return out


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"{tag}.json").read_text())
    seen = {}
    for r in d["runs"]:
        for e in r["log"]:
            if e["feasible"] and e["area_mm2"] > 0:   # all-CPU designs have no coprocessor
                seen[(round(e["area_mm2"], 1), round(e["speedup"], 4))] = e
    pts = [(e["area_mm2"], e["speedup"]) for e in seen.values()]
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.2))
    a = np.array(pts)
    ax.scatter(a[:, 0], a[:, 1], s=6, color=fs.INK_3, alpha=0.5, lw=0, label="Evaluated designs", zorder=2)
    f = np.array(front(pts))
    ax.step(f[:, 0], f[:, 1], where="post", color=fs.SERIES[0], lw=1.4, label="Pareto front", zorder=3)
    ax.plot(f[:, 0], f[:, 1], "o", color=fs.SERIES[0], markersize=3, zorder=4)
    ax.axvline(GA102_MM2, color=fs.SERIES[1], lw=0.9, ls="--")
    ax.annotate("RTX 3080 die\n(628 mm²)", (GA102_MM2, a[:, 1].max()), xytext=(-3, 0),
                textcoords="offset points", fontsize=6.5, color=fs.INK_2, va="top", ha="right")
    ax.axhline(1.0, color=fs.INK_2, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(100, a[:, 0].max() * 1.6)   # vector-unit-only designs (< 100 mm2) never beat the GPU
    ax.set_xlabel("Coprocessor area (mm², ISAAC 32 nm)")
    ax.set_ylabel("Speedup over RTX 3080")
    ax.set_ylim(0, a[:, 1].max() * 1.15)
    ax.legend(loc="lower right", fontsize=6.5)
    print(fs.save(fig, f"pareto_{tag}"))
    for x, y in f:
        print(f"  front: {x:8.0f} mm2  {y:.2f}x")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
