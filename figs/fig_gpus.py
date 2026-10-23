"""Fig: the headline design's session-time speedup over three GPUs.

The 3080 is measured; the 4090 and H100 SXM are scaled from it (decode by
memory bandwidth, prefill by fp16 tensor peak, tools unchanged), which is
generous to them.
Data: results/search/gpu_scale.json (experiments/gpu_scale.py).
  python figs/fig_gpus.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

SETS = [("search", "Search (45)"), ("heldout", "Held-out (44)"), ("late", "Late held-out (11)")]


def main():
    d = json.loads((ROOT / "results/search/gpu_scale.json").read_text())
    gpus = list(d["gpus"])
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 1.9))
    x = np.arange(len(SETS))
    w = 0.26
    for i, g in enumerate(gpus):
        v = [d["sets"][k]["speedup"][g] for k, _ in SETS]
        xs = x + (i - 1) * w
        ax.bar(xs, v, width=w - 0.03, color=fs.SERIES[i], label=g + ("" if i == 0 else " (scaled)"), zorder=3)
        for xi, vi in zip(xs, v):
            ax.annotate(f"{vi:.1f}×", (xi, vi), xytext=(0, 1.5), textcoords="offset points", ha="center",
                        va="bottom", fontsize=5.8, color=fs.INK)
    ax.axhline(1.0, color=fs.INK_2, lw=0.7, ls="--", zorder=2)
    ax.set_xticks(x, [lab for _, lab in SETS])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Session speedup of the design")
    ax.set_ylim(0, max(max(r["speedup"].values()) for r in d["sets"].values()) * 1.18)
    ax.legend(loc="upper left", fontsize=6, ncol=1)
    print(fs.save(fig, "gpus"))


if __name__ == "__main__":
    main()
