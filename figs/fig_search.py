"""Fig: sample efficiency. Best feasible speedup found vs evaluations used, per search method.

Lines are the median over seeds; bands span min..max. All classical searchers see time only. A dashed line marks the
best design any method found, or the exhaustive sweep's optimum once
experiments/sweep_analyze.py has run on the complete sweep.
Data: results/search/<tag>.json (experiments/search_compare.py).
  python figs/fig_search.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

# Fixed order = fixed color per method across every figure.
METHODS = [
    ("walk", "Domeline walk (readings)"),
    ("agent", "LLM agent + readings"),
    ("agent_scalar", "LLM agent, time only"),
    ("tpe", "Bayesian opt. (TPE)"),
    ("random", "Random search"),
    ("hill_climb", "Hill climbing"),
]


def best_so_far(log, budget):
    cur, out = 0.0, []
    by_n = {e["n"]: e for e in log}
    for n in range(1, budget + 1):
        e = by_n.get(n)
        if e and e["feasible"]:
            cur = max(cur, e["speedup"])
        out.append(cur)
    return np.array(out)


# One color per method across panels and figures (palette slots; each panel's set validates).
SLOT = {"walk": 0, "tpe": 1, "random": 2, "hill_climb": 3, "agent": 4, "agent_scalar": 5}

PANELS = [
    ("Classical searchers (scalar time only)", ["walk", "tpe", "random", "hill_climb"]),
    ("LLM agents (Sonnet)", ["walk", "agent", "agent_scalar"]),
]


def draw(ax, d, methods, top):
    B = d["budget"]
    x = np.arange(1, B + 1)
    for slot, m in enumerate(methods):
        runs = [r for r in d["runs"] if r["method"] == m]
        if not runs:
            continue
        label = dict(METHODS)[m]
        curves = np.stack([best_so_far(r["log"], B) for r in runs])
        st = fs.series(SLOT[m])
        med = np.median(curves, axis=0)
        ax.plot(x, med, color=st["color"], linestyle=st["linestyle"], lw=1.4,
                label=label + (f" ({len(runs)} seeds)" if len(runs) > 1 else ""), zorder=3)
        if len(runs) > 1:
            ax.fill_between(x, curves.min(0), curves.max(0), color=st["color"], alpha=0.12, lw=0)
        # Marker where the (median) curve first gets within 1% of the best any method found.
        if (med >= top * 0.99).any():
            n = int(np.argmax(med >= top * 0.99)) + 1
            ax.plot([n], [med[n - 1]], marker=st["marker"], color=st["color"], markersize=4, zorder=4)
    ax.axhline(top, color=fs.INK_3, lw=0.7, ls=":")
    ax.axhline(1.0, color=fs.INK_2, lw=0.7, ls="--")
    ax.annotate("RTX 3080", (B, 1.0), xytext=(0, -2), textcoords="offset points", ha="right",
                va="top", fontsize=6.5, color=fs.INK_2)
    ax.set_xlabel("Design evaluations")
    ax.set_xlim(1, B)
    ax.set_ylim(0, top * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.2, columnspacing=1.0)


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"{tag}.json").read_text())
    top = max((r["best"]["speedup"] for r in d["runs"] if "best" in r), default=1)
    # Reference: the exhaustive sweep's optimum under the same cap, when the sweep has run.
    ref_label = f"best found {top:.2f}×"
    sw = ROOT / "results/search/sweep_main_summary.json"
    if sw.exists():
        s = json.loads(sw.read_text())
        if tag in s["searchers"] and not s["partial"]:
            top = s["searchers"][tag]["fastest_speedup"]
            ref_label = f"exhaustive optimum {top:.2f}× ({s['n_designs']:,} designs)"
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.PAGE_W, 2.3), sharey=True)
    for ax, (title, methods) in zip(axes, PANELS):
        draw(ax, d, methods, top)
        ax.set_title(title, loc="left", color=fs.INK_2)
    axes[0].set_ylabel("Best speedup over RTX 3080")
    axes[0].annotate(ref_label, (d["budget"], top), xytext=(0, 2), textcoords="offset points",
                     ha="right", va="bottom", fontsize=6.5, color=fs.INK_2)
    print(fs.save(fig, f"search_{tag}"))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
