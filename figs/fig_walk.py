"""Fig: the co-design walk. Each step is one Domeline reading and one move.

Top: workload speedup over the RTX 3080 after each accepted move, against
the Amdahl bound (tools alone). Bottom: where the workload's time goes at
each step (shares of total, tools included), so the binding term is visible
as the largest LLM-side block, and each move shrinks it.

Data: the walk run in results/search/<tag>.json (experiments/search_compare.py).
  python figs/fig_walk.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

# Display groups for the integrator's terms, in stacking order (bottom first).
GROUPS = [
    ("host", "Host CPU (LLM ops)", lambda k: k == "host"),
    ("link", "CPU-PIM link", lambda k: k in ("launch", "transfer")),
    ("xbar", "Crossbar MVM", lambda k: k.startswith("xbar")),
    ("write", "KV writes", lambda k: k == "write"),
    ("star", "Nonlinear unit", lambda k: k.startswith("softmax")),
    ("vfu", "Vector unit", lambda k: k.startswith("vfu")),
]
SHORT = {
    "attention=xbar": "attention\n→ xbar", "attention=vfu": "attention\n→ vector unit",
    "softmax=star": "softmax→\nnonlin. unit", "softmax=vfu": "softmax\n→ vector unit",
    "lm_head=xbar": "LM head\n→ xbar", "vector=vfu": "norms→\nvector unit",
    "vector=star": "norms→\nnonlin. unit", "xbar_tech=xbar_transpose": "transposed\nK writes",
    "edge=vfu": "sampling\n→ unit", "link=package": "on-package\nlink", "link=die": "on-die\nlink",
}


def short(move):
    # A move can reset knobs that stop mattering (e.g. vfu_units once nothing uses the
    # vector unit); label it by the knob that was chosen.
    move = move.split(", ")[0]
    if move in SHORT:
        return SHORT[move]
    k, _, v = move.partition("=")
    return {"vfu_units": f"vector\nunit ×{v}", "star_units": f"nonlin.\nunit ×{v}",
            "xbar_scale": f"arrays\n×{v}", "kv_bits": f"KV int{v}" if v != "16" else "KV fp16",
            "vfu_bw": f"KV bw\n{v}B/s"}.get(k, move.replace(", ", "\n"))


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"{tag}.json").read_text())
    walk = next(r for r in d["runs"] if r["method"] == "walk")
    steps = [s for s in walk["trail"] if not s["reading"].startswith("shrink")]
    shrink = [s for s in walk["trail"] if s["reading"].startswith("shrink")]
    gpu, tools = walk["gpu_s"], walk["tools_s"]

    fs.apply()
    n = len(steps)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(fs.PAGE_W, 3.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.08})
    x = np.arange(n)
    sp = [s["speedup"] for s in steps]
    ax.plot(x, sp, color=fs.SERIES[0], marker="o", lw=1.4, zorder=3)
    ax.set_yscale("log")
    ax.axhline(1.0, color=fs.INK_2, lw=0.8, ls="--")
    ax.annotate("RTX 3080", (n - 0.5, 1.0), xytext=(0, -2), textcoords="offset points",
                ha="right", va="top", fontsize=6.5, color=fs.INK_2)
    bound = gpu / tools
    ax.axhline(bound, color=fs.INK_3, lw=0.8, ls=":")
    ax.annotate(f"Amdahl bound (tools only): {bound:.1f}×", (-0.3, bound), xytext=(0, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=6.5, color=fs.INK_2)
    for i, s in enumerate(steps):
        below = s["speedup"] > bound / 2.2
        ax.annotate(f"{s['speedup']:.2f}×", (i, s["speedup"]), xytext=(0, -9 if below else 5),
                    textcoords="offset points", ha="center", va="center", fontsize=6.3, color=fs.INK)
    ax.set_ylabel("Speedup over\nRTX 3080")
    ax.set_ylim(min(sp) * 0.6, bound * 1.8)
    ax.grid(axis="x", visible=False)

    # Bottom: shares of the workload time per step.
    bottom = np.zeros(n)
    width = 0.66
    for gi, (gk, label, pred) in enumerate(GROUPS):
        vals = np.array([sum(v for k, v in s["terms"].items() if pred(k)) for s in steps])
        tot = np.array([sum(s["terms"].values()) + tools for s in steps])
        share = vals / tot
        if share.max() < 0.005:
            continue
        bx.bar(x, share, bottom=bottom, width=width, color=fs.SERIES[gi], edgecolor=fs.SURFACE,
               linewidth=0.8, label=label, zorder=2)
        bottom += share
    tshare = tools / np.array([sum(s["terms"].values()) + tools for s in steps])
    bx.bar(x, tshare, bottom=bottom, width=width, color="#d9d8d3", edgecolor=fs.SURFACE, linewidth=0.8,
           label="Tools (host)", hatch="....", zorder=2)
    bx.set_ylim(0, 1)
    bx.set_ylabel("Share of\nworkload time")
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    bx.set_xticks(x, ["start"] + [short(s["move"]) for s in steps[1:]], fontsize=5.8)
    bx.grid(axis="x", visible=False)
    bx.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=6.5, handlelength=1.2)
    if shrink:
        last = walk["trail"][-1]
        bx.annotate(f"then shrink: area {steps[-1]['area_mm2']:.0f} → {last['area_mm2']:.0f} mm² "
                    f"at {last['speedup']:.2f}×", (0.0, -0.42), xycoords="axes fraction",
                    fontsize=6.5, color=fs.INK_2)
    print(fs.save(fig, f"walk_{tag}"))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
