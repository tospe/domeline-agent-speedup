"""Fig: sensitivity of the headline speedup to the uncertain technology numbers.

Tornado chart: each parameter scaled by 1/2 and 2; bars show the speedup of
the design after re-optimizing (Domeline walk) under the changed number.
Parameters sorted by the size of their effect.
Data: results/search/sensitivity_<tag>.json.  python figs/fig_sensitivity.py [tag]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402


def main(tag="main"):
    d = json.loads((ROOT / "results/search" / f"sensitivity_{tag}.json").read_text())
    base = d["base"]["speedup"]
    by, area = {}, {}
    for r in d["rows"]:
        by.setdefault(r["desc"], {})[r["factor"]] = r["reopt_speedup"] or 0.0
        area.setdefault(r["desc"], {})[r["factor"]] = r["reopt_area"] or 0.0
    base_area = d["base"]["area_mm2"]
    # A parameter that moves only area gets its area range in the label.
    relabel = {}
    for desc, a in area.items():
        if all(abs(v - base) / base < 0.01 for v in by[desc].values()) and \
                any(abs(x - base_area) / base_area > 0.1 for x in a.values()):
            relabel[desc] = f"{desc}\n(area {min(a.values()):.0f}-{max(a.values()):.0f} mm², speed unchanged)"
    rows = sorted(by.items(), key=lambda kv: max(abs(v - base) for v in kv[1].values()))
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 0.24 * len(rows) + 0.6))
    y = np.arange(len(rows))
    for i, (desc, v) in enumerate(rows):
        for f, slot, label in ((0.5, 0, "parameter ×½"), (2.0, 1, "parameter ×2")):
            s = v.get(f, base)
            ax.barh(i, s - base, left=base, height=0.62, color=fs.SERIES[slot],
                    label=label if i == len(rows) - 1 else None, zorder=2)
            if abs(s - base) / base > 0.02:
                ax.annotate(f"{s:.1f}×", (s, i), xytext=(3 if s > base else -3, 0), textcoords="offset points",
                            ha="left" if s > base else "right", va="center", fontsize=6.3, color=fs.INK)
    ax.axvline(base, color=fs.INK_2, lw=0.8)
    ax.set_yticks(y, [relabel.get(desc, desc) for desc, _ in rows], fontsize=6.5)
    ax.set_xlabel(f"Speedup over RTX 3080 (base {base:.2f}×)")
    ax.grid(axis="y", visible=False)
    lo = min(min(v.values()) for _, v in rows)
    hi = max(max(v.values()) for _, v in rows)
    ax.set_xlim(lo - (hi - lo) * 0.15, hi + (hi - lo) * 0.15)
    ax.legend(loc="lower right", fontsize=6.5)
    print(fs.save(fig, f"sensitivity_{tag}"))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["main"]))
