"""Fig: effort staircase. Best speedup over the RTX 3080 reachable within an effort budget.

Each step is labeled with what the extra months buy (the components the
best design adds at that cap). Data: results/search/effort_<tag>.json.
  python figs/fig_effort.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

ADDS = {  # knob=value -> what it buys, in the paper's words
    ("vector", "vfu"): "vector unit", ("softmax", "vfu"): "vector unit",
    ("attention", "vfu"): "vector unit + KV memory",
    ("softmax", "star"): "nonlinear unit", ("vector", "star"): "nonlinear unit",
    ("xbar_tech", "xbar_transpose"): "transposed K writes",
    ("link", "package"): "on-package link", ("link", "die"): "on-die link",
}


def adds(prev, cur):
    if prev is None:
        return "crossbars only"
    new = []
    for (k, v), name in ADDS.items():
        if cur["point"].get(k) == v and prev["point"].get(k) != v and name not in new:
            new.append(name)
    return " + ".join(new) if new else "resizing"


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"effort_{tag}.json").read_text())
    rows = d["rows"]
    for r in rows:   # an unbounded cap is stored as Infinity or null
        if r["cap"] is not None and r["cap"] == float("inf"):
            r["cap"] = None
    caps = sorted({r["cap"] for r in rows if r["cap"] is not None})
    xmax = caps[-1] + 3
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.2))
    for i, (m, label) in enumerate([("walk", "Domeline walk"), ("tpe", "Bayesian opt. (TPE)")]):
        rs = {r["cap"]: r for r in rows if r["method"] == m}
        xs, ys = [], []
        for c in caps + [None]:
            r = rs.get(c)
            xs.append(xmax if c is None else c)
            ys.append((r or {}).get("speedup") or 0.0)
        # The best reachable within a cap can only grow with the cap.
        ys = np.maximum.accumulate(ys)
        st = fs.series([0, 3][i])
        ax.step(xs, ys, where="post", color=st["color"], linestyle=st["linestyle"], lw=1.4, label=label,
                zorder=3 - i)
        if m == "walk":
            prev = None
            for c, y in zip(xs, ys):
                r = rs.get(None if c == xmax else c)
                if r and r.get("speedup") and (prev is None or r["speedup"] > prev["speedup"] * 1.03):
                    ax.annotate(adds(prev, r), (c, y), xytext=(2, 3), textcoords="offset points",
                                fontsize=6, color=fs.INK, ha="left", va="bottom")
                    ax.plot([c], [y], marker="o", color=st["color"], markersize=3.5, zorder=4)
                    prev = r
    ax.axhline(1.0, color=fs.INK_2, lw=0.7, ls="--")
    ax.annotate("RTX 3080", (xmax, 1.0), xytext=(0, -2), textcoords="offset points", ha="right",
                va="top", fontsize=6.5, color=fs.INK_2)
    ax.set_xlabel("Engineering effort budget (months)")
    ax.set_ylabel("Best speedup over RTX 3080")
    ax.set_xlim(-0.3, xmax)
    ax.set_xticks(caps + [xmax], [f"{c:g}" for c in caps] + ["∞"])
    ax.set_ylim(0, max(r.get("speedup") or 0 for r in rows) * 1.25)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), fontsize=6.5)
    print(fs.save(fig, f"effort_{tag}"))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
