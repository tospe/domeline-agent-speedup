"""Fig: the co-design walk. Each step is one Domeline reading and one move.

(a) Workload speedup over the RTX 3080 after each accepted move, against the
    Amdahl bound (tools alone). Moves that gain under 3% are drawn hollow and
    greyed as minor.
(b) Coprocessor area after each move (its own panel: no dual y-axis). The two
    shrink moves at the right drop area at no speed cost.
(c) Where the workload's time goes at each step (shares of total, tools
    included). The black outline marks the binding term the walk read on that
    design; the next move (label of the next bar, arrow) targets it. On the
    final speed design the dashed outline is the binding term no move improves.

Colors: figstyle's validated palette in stack order (adjacent pairs pass the
CVD check, validate_palette.js); the vector unit, whose only non-adjacent
near-clash is the host CPU (never visible together), also carries a hatch.
Legend order matches stack order (top of legend = top of stack).

Data: the walk run in results/search/<tag>.json (experiments/search_compare.py).
  python figs/fig_walk.py [tag]
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from harness import figstyle as fs  # noqa: E402

# Display groups for the integrator's terms, in stacking order (bottom first).
# Colors: figstyle slots 0-4 and 6 (slot 5, dark green, clashes with the
# crossbar teal under deuteranopia/protanopia, so it is skipped here).
GROUPS = [
    ("host", "Host CPU (LLM ops)", lambda k: k == "host", fs.SERIES[0], None),
    ("link", "CPU-PIM link", lambda k: k in ("launch", "transfer"), fs.SERIES[1], None),
    ("xbar", "Crossbar MVM", lambda k: k.startswith("xbar"), fs.SERIES[2], None),
    ("write", "KV writes", lambda k: k == "write", fs.SERIES[3], None),
    ("star", "Nonlinear unit", lambda k: k.startswith("softmax"), fs.SERIES[4], None),
    ("vfu", "Vector unit", lambda k: k.startswith("vfu"), fs.SERIES[6], "////"),
]
TOOLS_COLOR = "#d9d8d3"
MINOR_GAIN = 0.03          # a move that gains less than 3% is drawn as minor

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
    if k == "xbar_scale":
        return "no spare\narrays" if float(v) == 1.0 else f"arrays\n×{float(v):g} need"
    return {"vfu_units": f"vector\nunit ×{v}", "star_units": f"nonlin.\nunit ×{v}",
            "kv_bits": f"KV int{v}" if v != "16" else "KV fp16",
            "vfu_bw": f"KV bw\n{v}B/s"}.get(k, move.replace(", ", "\n"))


def group_of(term):
    for gk, _, pred, _, _ in GROUPS:
        if pred(term):
            return gk
    raise KeyError(term)


def binding_group(reading):
    """The display group of the binding term a reading names ('<term> binds ...')."""
    m = re.match(r"(\S+) binds", reading)
    return group_of(m.group(1)) if m else None


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"{tag}.json").read_text())
    walk = next(r for r in d["runs"] if r["method"] == "walk")
    trail = walk["trail"]
    n_speed = sum(1 for s in trail if not s["reading"].startswith("shrink"))
    gpu, tools = walk["gpu_s"], walk["tools_s"]
    n = len(trail)
    x = np.arange(n)
    sp = np.array([s["speedup"] for s in trail])
    area = np.array([s["area_mm2"] for s in trail]) / 1e3
    gain = np.r_[np.inf, sp[1:] / sp[:-1] - 1]
    minor = np.array([1 <= i < n_speed and gain[i] < MINOR_GAIN for i in range(n)])

    fs.apply()
    fig, (ax, cx, bx) = plt.subplots(
        3, 1, figsize=(fs.PAGE_W, 4.55), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.42, 1.25], "hspace": 0.12})

    # Shrink phase: light band behind every panel.
    if n_speed < n:
        for a in (ax, cx, bx):
            a.axvspan(n_speed - 0.5, n - 0.5, color="#f3f2ee", lw=0, zorder=0)
        ax.text((n_speed - 0.5 + n - 0.5) / 2, 0.3, "shrink:\nno speed lost",
                transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=6.3, color=fs.INK_2)

    # (a) speedup.
    c0 = fs.SERIES[0]
    ax.plot(x[:n_speed], sp[:n_speed], color=c0, lw=1.4, zorder=3)
    ax.plot(x[n_speed - 1:], sp[n_speed - 1:], color=c0, lw=1.0, ls="--", zorder=3)
    ax.scatter(x[~minor], sp[~minor], s=16, color=c0, zorder=4, linewidths=0)
    ax.scatter(x[minor], sp[minor], s=16, facecolor=fs.SURFACE, edgecolor=c0, linewidths=1.0, zorder=4)
    ax.set_yscale("log")
    ax.axhline(1.0, color=fs.INK_2, lw=0.8, ls="--")
    ax.annotate("RTX 3080", (n_speed - 1.5, 1.0), xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=6.3, color=fs.INK_2)
    bound = gpu / tools
    ax.axhline(bound, color=fs.INK_3, lw=0.8, ls=":")
    ax.annotate(f"Amdahl bound (tools only): {bound:.1f}×", (-0.4, bound), xytext=(0, 2),
                textcoords="offset points", ha="left", va="bottom", fontsize=6.3, color=fs.INK_2)
    for i in range(n_speed):
        s = sp[i]
        if i == 0:      # far below the rest: label to the right of the point
            ax.annotate(f"{s:.2f}×", (i, s), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=6.3, color=fs.INK)
            continue
        txt = f"{s:.2f}×"
        if minor[i]:
            txt += f"\n+{gain[i]:.0%}: minor"
        # Above the line, except where the curve climbs steeply (label would sit on it).
        below = i >= n_speed - 2 or minor[i]
        steep = i < 4 and gain[i + 1] > 0.5       # outgoing line climbs fast: label up-left
        ax.annotate(txt, (i, s), xytext=(-3, 3) if steep else (0, -5 if below else 5),
                    textcoords="offset points", ha="right" if steep else "center",
                    va="top" if below else "bottom", fontsize=6.3,
                    color=fs.INK_3 if minor[i] else fs.INK, linespacing=0.95)
    ax.set_ylabel("Speedup over\nRTX 3080")
    ax.set_ylim(sp.min() * 0.55, bound * 2.0)
    ax.grid(axis="x", visible=False)
    ax.text(-0.135, 1.0, "(a)", transform=ax.transAxes, fontsize=8, va="top", ha="left")

    # (b) area.
    cx.step(x, area, where="mid", color=fs.INK_2, lw=1.0, zorder=3)
    cx.scatter(x, area, s=9, color=fs.INK_2, zorder=4, linewidths=0)
    cx.set_ylim(0, area.max() * 1.45)
    cx.set_ylabel("Area\n(K mm²)")
    cx.yaxis.set_major_locator(plt.MultipleLocator(10))
    cx.grid(axis="x", visible=False)
    peak = int(np.argmax(area[:n_speed]))
    # Label each area change once: start, each jump, final.
    for i in range(n):
        if i == 0 or i == n - 1 or abs(area[i] - area[i - 1]) > 0.2:
            cx.annotate(f"{area[i]:.1f}K", (i, area[i]), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom",
                        fontsize=6.0, color=fs.INK)
    cx.text(-0.135, 1.0, "(b)", transform=cx.transAxes, fontsize=8, va="top", ha="left")
    del peak

    # (c) shares of the workload time.
    width = 0.66
    tot = np.array([sum(s["terms"].values()) + tools for s in trail])
    bottom = np.zeros(n)
    seg = {}            # (group, step) -> (y0, h)
    handles = []
    for gk, label, pred, color, hatch in GROUPS:
        vals = np.array([sum(v for k, v in s["terms"].items() if pred(k)) for s in trail])
        share = vals / tot
        if share.max() < 0.005:
            continue
        h = bx.bar(x, share, bottom=bottom, width=width, color=color, edgecolor=fs.SURFACE,
                   linewidth=0.8, hatch=hatch, label=label, zorder=2)
        handles.append(h)
        for i in range(n):
            seg[gk, i] = (bottom[i], share[i])
        bottom += share
    tshare = tools / tot
    handles.append(bx.bar(x, tshare, bottom=bottom, width=width, color=TOOLS_COLOR,
                          edgecolor=fs.SURFACE, linewidth=0.8, label="Tools (host)",
                          hatch="....", zorder=2))

    # Binding term read on each design -> the next move targets it.
    for i in range(n_speed):
        if i + 1 < n_speed:
            g, ls = binding_group(trail[i + 1]["reading"]), "-"
        else:       # final speed design: the walk stopped; largest LLM term binds
            lt = {}
            for k, v in trail[i]["terms"].items():
                lt[group_of(k)] = lt.get(group_of(k), 0) + v
            g, ls = max(lt, key=lt.get), (0, (2, 1))
        y0, h = seg[g, i]
        bx.add_patch(Rectangle((i - width / 2, y0), width, h, fill=False, edgecolor=fs.INK,
                               lw=1.1, ls=ls, zorder=5, joinstyle="miter"))
        if i + 1 < n_speed:
            bx.annotate("", xy=(i + 1 - width / 2 - 0.02, y0 + h / 2),
                        xytext=(i + width / 2 + 0.02, y0 + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=fs.INK, lw=0.6,
                                        mutation_scale=5, shrinkA=0, shrinkB=0), zorder=6)

    bx.set_ylim(0, 1)
    bx.set_xlim(-0.6, n - 0.4)
    bx.set_ylabel("Share of\nworkload time")
    bx.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    labels = ["start"] + [short(s["move"]) for s in trail[1:]]
    bx.set_xticks(x, labels, fontsize=5.8)
    for i, t in enumerate(bx.get_xticklabels()):
        if minor[i]:
            t.set_color(fs.INK_3)
            t.set_fontstyle("italic")
    bx.tick_params(axis="x", length=0, pad=3)
    bx.grid(axis="x", visible=False)
    bx.text(-0.135, 1.0, "(c)", transform=bx.transAxes, fontsize=8, va="top", ha="left")

    # Legend: top of legend = top of stack, then the outline keys.
    handles = handles[::-1] + [
        Patch(facecolor="none", edgecolor=fs.INK, lw=1.1, label="binding term;\nnext move targets it"),
        Patch(facecolor="none", edgecolor=fs.INK, lw=1.1, ls=(0, (2, 1)),
              label="binding term;\nno move improves it"),
    ]
    bx.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.02), fontsize=6.3,
              handlelength=1.3, handleheight=1.1, labelspacing=0.45)
    print(fs.save(fig, f"walk_{tag}"))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
