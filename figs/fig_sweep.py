"""Fig: the whole design space, from the exhaustive sweep.

(a) How rare good designs are: fraction of feasible designs at or above each
    speedup (survival curve), with the RTX 3080 and the optimum marked.
(b) Speedup vs coprocessor area: the exhaustive Pareto front, with the
    design each searcher returned on top.
Data: results/search/sweep_<tag>.tsv (or its parts, marked partial),
results/search/<tag>.json.  python figs/fig_sweep.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402
from sweep_analyze import load  # noqa: E402

GA102_MM2 = 628.4
SLOT = {"walk": 0, "tpe": 1, "random": 2, "hill_climb": 3, "agent": 4, "agent_scalar": 5}
NAME = {"walk": "Domeline walk", "tpe": "TPE", "random": "Random", "hill_climb": "Hill climbing",
        "agent": "LLM agent + readings", "agent_scalar": "LLM agent, time only"}


def front(a, s):
    o = np.argsort(a, kind="stable")
    keep, best = [], -1.0
    for i in o:
        if s[i] > best:
            keep.append(i)
            best = s[i]
    return np.array(keep)


def main(tag="main"):
    df, partial = load(tag)
    f = df[df.feasible == 1]
    s, a = f.speedup.to_numpy(), f.area_mm2.to_numpy()
    top = s.max()
    runs = json.loads((ROOT / "results/search" / f"{tag}.json").read_text())["runs"]

    fs.apply()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fs.PAGE_W, 2.2))

    # (a) survival curve
    x = np.sort(s)[::-1]
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=fs.SERIES[0], lw=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color=fs.INK_2, lw=0.7, ls="--")
    ax.annotate("RTX 3080", (1.0, 1.0), xytext=(3, -2), textcoords="offset points", va="top",
                fontsize=6.5, color=fs.INK_2)
    beat = (s > 1).mean()
    within = (s >= top * 0.99).sum()
    ax.annotate(f"{beat:.1%} beat the GPU", (1.0, beat), xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color=fs.INK_2)
    ax.plot([1.0], [beat], "o", color=fs.INK_2, markersize=3)
    ax.plot([top * 0.99], [within / len(x)], "o", color=fs.INK_2, markersize=3)
    ax.annotate(f"{within} designs within 1% of\nthe optimum ({top:.2f}×)", (top * 0.99, within / len(x)),
                xytext=(-75, -22), textcoords="offset points", ha="right", va="top", fontsize=6.5,
                color=fs.INK_2, arrowprops=dict(arrowstyle="-", color=fs.INK_3, lw=0.6))
    ax.set_xlabel("Speedup over RTX 3080")
    ax.set_ylabel("Fraction of designs ≥ speedup")
    ax.set_title(f"(a) {len(f):,} feasible designs{' (partial sweep)' if partial else ''}",
                 loc="left", color=fs.INK_2)

    # (b) Pareto front over the whole space, searchers' returned designs on top
    m = a > 0                                  # all-host designs have no coprocessor
    s, a = s[m], a[m]
    inset = ax2.inset_axes([0.14, 0.40, 0.40, 0.42])
    ax2.scatter(a, s, s=1.5, color=fs.INK_3, alpha=0.15, lw=0, rasterized=True, zorder=1)
    k = front(a, s)
    ax2.step(a[k], s[k], where="post", color=fs.SERIES[0], lw=1.2, label="Exhaustive Pareto front", zorder=3)
    seen = set()
    for r in runs:
        if "best" not in r or r["method"] in seen:
            continue
        seen.add(r["method"])
        b = r["best"]
        st = fs.series(SLOT[r["method"]])
        for axx in (ax2, inset):
            axx.plot([b["area_mm2"]], [b["speedup"]], marker=st["marker"], color=st["color"], markersize=4.5,
                     ls="none", markeredgecolor=fs.SURFACE, markeredgewidth=0.5, zorder=4,
                     label=NAME[r["method"]] if axx is ax2 else None)
    # Inset: the top of the space, where the searchers' designs sit.
    zoom = (s >= top * 0.97)
    inset.scatter(a[zoom], s[zoom], s=4, color=fs.INK_3, alpha=0.5, lw=0, zorder=1)
    inset.step(a[k], s[k], where="post", color=fs.SERIES[0], lw=1.0, zorder=3)
    inset.set_xscale("log")
    inset.set_xlim(a[zoom].min() * 0.92, a[zoom].max() * 1.08)
    inset.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}K"))
    inset.xaxis.set_minor_formatter(mticker.NullFormatter())
    inset.set_ylim(top * 0.97, top * 1.003)
    inset.tick_params(labelsize=5.5)
    inset.set_title(f"within 3% of the optimum", fontsize=6, color=fs.INK_2, pad=2)
    ax2.indicate_inset_zoom(inset, edgecolor=fs.INK_3, lw=0.5)
    ax2.axhline(1.0, color=fs.INK_2, lw=0.7, ls="--")
    ax2.set_xscale("log")
    ax2.set_xlim(2000, a.max() * 1.3)
    ax2.set_ylim(0, top * 1.12)
    for axx in (ax2,):
        axx.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}K"))
        axx.xaxis.set_minor_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v / 1e3:g}K" if round(v / 1e3) in (2, 5, 20) else ""))
    ax2.set_xlabel(f"Coprocessor area (mm², ISAAC 32 nm; RTX 3080 die = {GA102_MM2:.0f})")
    ax2.set_ylabel("Speedup over RTX 3080")
    ax2.set_title("(b) Speedup vs area, every design", loc="left", color=fs.INK_2)
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6, ncol=1, handletextpad=0.3,
               title="Design returned (seed 0)", title_fontsize=6)
    print(fs.save(fig, f"sweep_{tag}"))
    for i in k:
        if s[i] >= 1:
            print(f"  front: {a[i]:8.0f} mm2  {s[i]:.2f}x")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
