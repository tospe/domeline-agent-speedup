"""One style for every paper figure (IEEE two-column: 3.5 in column, 7.16 in page).

Categorical hues come from the validated reference palette, assigned in fixed
order (validate_palette.js, light surface: all checks pass; slots 3-5 sit
below 3:1 contrast, so every figure also direct-labels series and gives each
a distinct marker / line style). Text is ink, never series color.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL_W = 3.5      # inches, one IEEE column
PAGE_W = 7.16    # inches, full width

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 1)), (0, (6, 2))]
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e6e5e0"
SURFACE = "#ffffff"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def apply():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_3,
        "axes.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def series(i):
    """Color, marker and dash for series i (fixed order, never cycled past 8)."""
    if i >= len(SERIES):
        raise ValueError("more than 8 series: fold into Other or facet")
    return dict(color=SERIES[i], marker=MARKERS[i], linestyle=DASHES[i])


def label_end(ax, x, y, text, dx=4, dy=0, **kw):
    """Direct label at a line's end, in ink."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha="left", color=INK, fontsize=7, **kw)


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    return FIG_DIR / f"{name}.pdf"
