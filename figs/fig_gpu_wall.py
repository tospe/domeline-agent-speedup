"""Fig: RTX 3080 decode time per token vs context, per KV precision, with the VRAM wall.

Data: results/gpu/g1.jsonl and g1_status.tsv (gpu/g1_sweep.sh); past-the-wall
points from results/gpu/g2.jsonl when present (gpu/g2_sweep.sh).
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from harness import figstyle as fs  # noqa: E402

KV_LABEL = {"f16": "KV fp16", "q8_0": "KV int8 (q8_0)", "q4_0": "KV int4 (q4_0)"}


def load(weights):
    pts = {}
    for line in (ROOT / "results/gpu/g1.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["model_filename"].endswith(f"-{weights}.gguf") and r["n_gen"] > 0:
            pts.setdefault(r["type_k"], []).append((r["n_depth"], 1e3 / r["avg_ts"], 1e3 * r["stddev_ts"] / r["avg_ts"] ** 2))
    fails = {}
    with open(ROOT / "results/gpu/g1_status.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["weights"] == weights and row["status"] == "fail":
                fails[row["kv"]] = int(row["depth"])
    g2 = {}
    p2 = ROOT / "results/gpu/g2.jsonl"
    if p2.exists():
        for line in p2.read_text().splitlines():
            r = json.loads(line)
            if r["model_filename"].endswith(f"-{weights}.gguf") and r["n_gen"] > 0:
                g2.setdefault((r["type_k"], r.get("_mode", "nkvo")), []).append((r["n_depth"], 1e3 / r["avg_ts"]))
    return {k: sorted(v) for k, v in pts.items()}, fails, {k: sorted(v) for k, v in g2.items()}


def draw(ax, weights, show_ylabel=True):
    pts, fails, g2 = load(weights)
    order = ["f16", "q8_0", "q4_0"]
    ymax = 0
    for i, kv in enumerate(order):
        if kv not in pts:
            continue
        st = fs.series(i)
        xs = [d / 1024 for d, _, _ in pts[kv]]
        ys = [ms for _, ms, _ in pts[kv]]
        ymax = max(ymax, max(ys))
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                markevery=1, label=KV_LABEL[kv], zorder=3)
        if kv in fails:
            wall = (pts[kv][-1][0] + fails[kv]) / 2 / 1024
            ax.plot([wall], [ys[-1]], marker="x", color=fs.INK, markersize=5, zorder=4)
            fs.label_end(ax, wall, ys[-1], f"wall ~{wall:.0f}K", dx=3, dy=-7)
        for (kv2, mode), p in g2.items():
            if kv2 == kv:
                ax.plot([d / 1024 for d, _ in p], [ms for _, ms in p], color=st["color"],
                        linestyle=":", marker=st["marker"], markerfacecolor="white", zorder=3)
    ax.set_xlabel("Context length (K tokens)")
    if show_ylabel:
        ax.set_ylabel("Decode time per token (ms)")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlim(left=0)
    ax.set_title(f"{weights} weights", loc="left", color=fs.INK_2)


def main():
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.PAGE_W * 0.75, 2.0), sharey=False)
    draw(axes[0], "Q8_0")
    draw(axes[1], "Q4_K_M", show_ylabel=False)
    axes[0].legend(loc="upper left")
    print(fs.save(fig, "gpu_wall"))


if __name__ == "__main__":
    main()
