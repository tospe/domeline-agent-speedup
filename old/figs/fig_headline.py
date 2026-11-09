"""Fig: the chosen designs against the RTX 3080, per decoded token.

(a) Decode time per token vs context: the 3080 measured (G1, fp16 and int8 KV,
    up to its VRAM wall) and the Domeline designs from the integrator: the
    headline design (ISAAC tile) and the 3,700 mm2 design (lean tile, 4-bit cells).
(b) Where the headline design's time per token goes, at 4K, 16K and 64K context.
Data: results/gpu/g1.jsonl, results/search/main.json, results/search/area_main.json.
  python figs/fig_headline.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import library, model, sessions, split  # noqa: E402
from harness import figstyle as fs  # noqa: E402
from harness.integrator import Integrator  # noqa: E402
from harness.space import Space, canonical  # noqa: E402

CTX = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536]
TERM_NAME = {"xbar_transpose": "Crossbar reads", "xbar_isaac": "Crossbar reads", "transfer": "Link transfers",
             "softmax_star": "Nonlinear unit", "launch": "Launch / sync", "host": "Host CPU", "write": "KV writes"}


def gpu_curves():
    out = {}
    for line in (ROOT / "results/gpu/g1.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["model_filename"].endswith("-Q8_0.gguf") and r["n_gen"] > 0:
            out.setdefault(r["type_k"], []).append((r["n_depth"], 1e3 / r["avg_ts"]))
    return {k: sorted(v) for k, v in out.items()}


def design_curve(space, p):
    d, lib = space.to_design(p)
    integ = Integrator(space.model, lib)
    return [(c, integ.step(d, 1, c)) for c in CTX]


def main():
    spec, lib = model(), library()
    tr, te = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in tr + te))
    head = canonical(json.loads((ROOT / "results/search/main.json").read_text())["runs"][0]["best"]["point"])
    area = json.loads((ROOT / "results/search/area_main.json").read_text())
    small = next(r["best"] for r in area["rows"] if r["method"] == "walk" and r["cap"] == 4000)
    designs = [("Headline design, 9.7K mm²", head, 0),
               (f"Lean tile + 4-b cells, {small['area_mm2'] / 1e3:.1f}K mm²", canonical(small["point"]), 2)]

    fs.apply()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fs.PAGE_W, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    g = gpu_curves()
    for kv, slot, lab in (("f16", 1, "RTX 3080, fp16 KV"), ("q8_0", 3, "RTX 3080, int8 KV")):
        st = fs.series(slot)
        x = [d / 1024 for d, _ in g[kv] if d > 0]
        y = [ms for d, ms in g[kv] if d > 0]
        ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["linestyle"], markersize=3, label=lab)
    curves = {}
    for lab, p, slot in designs:
        st = fs.series(slot)
        c = design_curve(space, p)
        curves[lab] = c
        ax.plot([x / 1024 for x, _ in c], [s.total * 1e3 for _, s in c], color=st["color"], marker=st["marker"],
                ls=st["linestyle"], markersize=3, label=lab)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64], ["1K", "4K", "16K", "64K"])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Decode time per token (ms)")
    ax.set_title("(a) Per-token decode latency", loc="left", color=fs.INK_2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6, columnspacing=1.0)
    # Ratio callouts at 4K and 32K for the headline design.
    hc = dict((c, s.total * 1e3) for c, s in curves[designs[0][0]])
    gf = dict(g["f16"])
    for c in (4096, 32768):
        if c in gf:
            ax.annotate(f"{gf[c] / hc[c]:.0f}×", (c / 1024, np.sqrt(gf[c] * hc[c])), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5, color=fs.INK_2, va="center")
            ax.plot([c / 1024] * 2, [hc[c], gf[c]], color=fs.INK_3, lw=0.6, zorder=1)

    # (b) breakdown of the headline design.
    sel = [4096, 16384, 65536]
    steps = dict(curves[designs[0][0]])
    terms = {}
    for c in sel:
        for k, v in steps[c].terms.items():
            name = TERM_NAME.get(k, "Other")
            terms.setdefault(name, {})[c] = terms.get(name, {}).get(c, 0) + v * 1e3
    order = sorted(terms, key=lambda k: -terms[k].get(65536, 0))
    x = np.arange(len(sel))
    bottom = np.zeros(len(sel))
    for i, name in enumerate(order):
        v = np.array([terms[name].get(c, 0.0) for c in sel])
        ax2.bar(x, v, bottom=bottom, width=0.6, color=fs.SERIES[i], label=name, edgecolor=fs.SURFACE, lw=0.6)
        bottom += v
    for xi, tot in zip(x, bottom):
        ax2.annotate(f"{tot:.2f} ms", (xi, tot), xytext=(0, 2), textcoords="offset points", ha="center",
                     va="bottom", fontsize=6.3, color=fs.INK)
    ax2.set_xticks(x, [f"{c // 1024}K" for c in sel])
    ax2.set_xlabel("Context length (tokens)")
    ax2.set_ylabel("ms per token")
    ax2.set_ylim(0, bottom.max() * 1.7)
    ax2.grid(axis="x", visible=False)
    ax2.set_title("(b) Headline design, by term", loc="left", color=fs.INK_2)
    ax2.legend(loc="upper left", fontsize=6, ncol=2, columnspacing=0.8)
    print(fs.save(fig, "headline"))
    for lab, c in curves.items():
        print(lab, {x: round(s.total * 1e3, 3) for x, s in c})
    print("gpu f16", {d: round(ms, 2) for d, ms in g["f16"]})


if __name__ == "__main__":
    main()
