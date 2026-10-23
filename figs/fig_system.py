"""Fig: the system and the method.

(a) The headline design: host CPU, a CXL-class link, and the PIM coprocessor.
    The crossbar bar is drawn to scale by arrays (weights, LM head, KV cache at
    the 64K reference context).
(b) The Domeline harness loop: integrator -> readings -> policy -> next design.
Data: results/search/main.json (array counts recomputed with the search's Space).
  python figs/fig_system.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from common import library, model, sessions, split  # noqa: E402
from harness import figstyle as fs  # noqa: E402
from harness.space import Space, canonical  # noqa: E402


def box(ax, x0, y0, x1, y1, fc=fs.SURFACE, ec=fs.INK_3, lw=0.7, r=0.8, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def text(ax, x, y, s, size=7, color=fs.INK, **kw):
    ax.text(x, y, s, fontsize=size, color=color, zorder=5, **kw)


def arrow(ax, a, b, both=False, rad=0.0, color=fs.INK_2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>" if both else "-|>", mutation_scale=7, lw=0.8,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=4,
                                 shrinkA=0, shrinkB=0))


def arrays():
    spec, lib = model(), library()
    tr, te = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in tr + te))
    p = canonical(json.loads((ROOT / "results/search/main.json").read_text())["runs"][0]["best"]["point"])
    tot = space.xbar_need(p)
    no_kv = space.xbar_need({**p, "attention": "vfu"})
    w = space.xbar_need({**p, "attention": "vfu", "lm_head": "cpu"})
    return {"weights": w, "lm": no_kv - w, "kv": tot - no_kv}, space.area_mm2(p), space.ref_ctx, p


def main():
    n, area, ref_ctx, p = arrays()
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.PAGE_W, 2.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 33)
    ax.axis("off")

    # (a) the design.
    text(ax, 0, 32.5, "(a) The PIM design for a coding agent", 8, fs.INK_2, va="top")
    box(ax, 0.5, 6, 12, 27)
    text(ax, 6.25, 24.5, "Host CPU", 7.5, ha="center", va="top", weight="bold")
    text(ax, 6.25, 20.5, "embedding\nsampling\nagent loop\ntool calls", 6.3, fs.INK_2, ha="center", va="top",
         linespacing=1.35)
    arrow(ax, (12, 16.5), (17.5, 16.5), both=True)
    text(ax, 14.75, 15.5, "CXL-\nclass\nlink", 5.6, fs.INK_2, ha="center", va="top", linespacing=1.1)
    box(ax, 17.5, 0, 48, 29, fc="#f6f6f3")
    text(ax, 18.7, 27.7, "PIM coprocessor", 7.5, va="top", weight="bold")
    text(ax, 18.7, 25.4, f"{area / 1e3:.1f}K mm² at ISAAC 32 nm density", 5.8, fs.INK_2, va="top")
    # Crossbar bar to scale by arrays.
    x0, x1, y0, y1 = 19.5, 46, 13.2, 18.4
    tot = sum(n.values())
    parts = [("weights", "Weights", fs.SEQ_BLUE[5]), ("lm", "LM head", fs.SEQ_BLUE[3]),
             ("kv", f"KV cache ({ref_ctx // 1024}K ctx)", fs.SEQ_BLUE[1])]
    text(ax, x0, 22.6, f"ReRAM crossbars: {tot / 1e6:.2f}M arrays, to scale", 6.3, va="top")
    text(ax, x0, 20.6, "8-bit KV with transposed K writes", 5.8, fs.INK_2, va="top")
    x = x0
    for k, lab, c in parts:
        w = (x1 - x0) * n[k] / tot
        ax.add_patch(plt.Rectangle((x, y0), w, y1 - y0, fc=c, ec=fs.SURFACE, lw=1.0, zorder=3))
        cx = x + w / 2
        if k == "lm":
            text(ax, cx, y0 - 0.6, f"{lab}\n{n[k] / 1e3:.0f}K", 5.8, fs.INK_2, ha="center", va="top", linespacing=1.1)
        else:
            text(ax, cx, (y0 + y1) / 2, f"{lab}  {n[k] / 1e3:.0f}K" if k == "weights" else f"{lab}\n{n[k] / 1e6:.2f}M",
                 5.8, fs.INK if k == "kv" else fs.SURFACE, ha="center", va="center", linespacing=1.1)
        x += w
    box(ax, x0, 1.5, x1, 7.5, fc=fs.SURFACE)
    text(ax, (x0 + x1) / 2, 4.5, f"Nonlinear unit ({p['star_units']} lanes)\nsoftmax, norms", 6.1,
         ha="center", va="center", linespacing=1.2)
    arrow(ax, (40, 7.5), (40, 13.2), both=True)

    # (b) the harness loop.
    text(ax, 53, 32.5, "(b) The Domeline harness", 8, fs.INK_2, va="top")
    box(ax, 53, 21.5, 66, 29.5)
    text(ax, 59.5, 25.5, "Workload\n89 agent sessions\n(per-turn traces)", 6.3, ha="center", va="center", linespacing=1.2)
    box(ax, 72, 19.5, 99.5, 29.5, ec=fs.SERIES[0], lw=1.0)
    text(ax, 85.75, 28.3, "Integrator", 7, ha="center", va="top", weight="bold")
    text(ax, 85.75, 25.3, "op graph × technology library\n× session replay\n→ time, area, effort", 6.0, fs.INK_2,
         ha="center", va="top", linespacing=1.2)
    arrow(ax, (66, 25.5), (72, 25.5))
    box(ax, 72, 3, 99.5, 14.5, ec=fs.SERIES[1], lw=1.0)
    text(ax, 85.75, 13.2, "Readings", 7, ha="center", va="top", weight="bold")
    text(ax, 85.75, 10.3, "binding term · headroom\nAmdahl bound (tools alone)", 6.0, fs.INK_2,
         ha="center", va="top", linespacing=1.2)
    arrow(ax, (85.75, 19.5), (85.75, 14.5))
    box(ax, 53, 3, 66, 14.5, ec=fs.SERIES[2], lw=1.0)
    text(ax, 59.5, 13.2, "Policy", 7, ha="center", va="top", weight="bold")
    text(ax, 59.5, 10.3, "one knob move\n+ the reading\nthat justifies it", 6.0, fs.INK_2, ha="center", va="top",
         linespacing=1.2)
    arrow(ax, (72, 8.75), (66, 8.75))
    text(ax, 69, 9.4, "why", 5.8, fs.INK_2, ha="center", va="bottom", style="italic")
    arrow(ax, (64, 14.5), (72, 20.5), rad=-0.3)
    text(ax, 63.9, 17.6, "next\ndesign", 5.8, fs.INK_2, ha="right", va="center", style="italic", linespacing=1.1)
    text(ax, 53, 1.8, "policies: walk · hill climbing · TPE · random · LLM agent\nconstraints: area cap, effort cap", 5.8,
         fs.INK_3, va="top", linespacing=1.2)
    print(fs.save(fig, "system"))
    print({k: int(v) for k, v in n.items()}, round(area), ref_ctx)


if __name__ == "__main__":
    main()
