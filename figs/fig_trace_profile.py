"""Fig + table: where an agent session's time goes on the RTX 3080, and how far its context grows.

Data: results/traces/swebench/<instance>/{trace.jsonl,result.json} (traces/run_swebench.py).
Tool time has the per-session podman exec overhead removed (it is the harness's
container cost, not the tool's).
Outputs: figures/trace_profile.pdf, results/traces/swebench_summary.tsv
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from harness import figstyle as fs  # noqa: E402

TRACE_DIR = ROOT / "results/traces/swebench"


def summarize():
    rows = []
    for d in sorted(TRACE_DIR.iterdir()):
        res, tr = d / "result.json", d / "trace.jsonl"
        if not res.exists() or not tr.exists():
            continue
        r = json.loads(res.read_text())
        turns = [json.loads(l) for l in tr.read_text().splitlines()]
        ov = r.get("exec_overhead_s", 0.0)
        dec = sum(t["t_decode_s"] for t in turns)
        pre = sum(t["t_prefill_s"] for t in turns)
        tool = sum(max(0.0, t["t_tool_s"] - (ov if t["command"] else 0.0)) for t in turns)
        tool_raw = sum(t["t_tool_s"] for t in turns)
        rows.append({
            "instance": d.name, "repo": r["repo"].split("/")[-1], "exit": r["exit_status"],
            "turns": len(turns), "max_ctx": max(t["ctx_start"] + (t["gen_tokens"] or 0) for t in turns),
            "gen_tokens": sum(t["gen_tokens"] or 0 for t in turns),
            "decode_s": dec, "prefill_s": pre, "tool_s": tool, "tool_raw_s": tool_raw,
            "exec_overhead_s": ov,
        })
    return rows


def write_tsv(rows):
    out = ROOT / "results/traces/swebench_summary.tsv"
    keys = list(rows[0])
    with open(out, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    return out


def main():
    rows = summarize()
    if not rows:
        print("no sessions yet")
        return
    write_tsv(rows)
    fs.apply()
    rows.sort(key=lambda r: r["decode_s"] + r["prefill_s"] + r["tool_s"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(fs.PAGE_W, 0.16 * len(rows) + 0.9),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    y = np.arange(len(rows))
    parts = [("decode_s", "LLM decode", 0), ("prefill_s", "LLM prefill", 1), ("tool_s", "Tool execution", 2)]
    left = np.zeros(len(rows))
    for key, label, i in parts:
        v = np.array([r[key] for r in rows])
        ax.barh(y, v, left=left, height=0.72, color=fs.SERIES[i], label=label,
                edgecolor=fs.SURFACE, linewidth=0.8, hatch=[None, "////", None][i])
        left += v
    ax.set_yticks(y, [f"{r['repo']}-{r['instance'].split('-')[-1]}" for r in rows], fontsize=6)
    ax.set_xlabel("Session time on the RTX 3080 (s)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncol=1)
    tot = sum(r["decode_s"] + r["prefill_s"] + r["tool_s"] for r in rows)
    shares = {k: sum(r[k] for r in rows) / tot for k, _, _ in parts}
    ax.set_title(f"{len(rows)} SWE-bench Verified sessions: decode {shares['decode_s']:.0%}, "
                 f"prefill {shares['prefill_s']:.0%}, tools {shares['tool_s']:.0%} of total time",
                 loc="left", color=fs.INK_2)

    ctx = np.sort([r["max_ctx"] for r in rows]) / 1024
    ax2.step(ctx, np.arange(1, len(ctx) + 1) / len(ctx), where="post", color=fs.SERIES[0], lw=1.4)
    ax2.axvspan(32, 36, color=fs.SERIES[1], alpha=0.15, lw=0)
    ax2.annotate("3080 wall\n(Q8_0, fp16 KV)", (36, 0.08), xytext=(3, 0), textcoords="offset points",
                 fontsize=6.5, color=fs.INK_2, va="bottom")
    ax2.set_xlabel("Final context (K tokens)")
    ax2.set_ylabel("Fraction of sessions")
    ax2.set_xlim(0, max(40, ctx.max() * 1.1))
    ax2.set_ylim(0, 1.02)
    print(fs.save(fig, "trace_profile"))
    print({k: round(v, 3) for k, v in shares.items()}, "max ctx", ctx.max(), "median", np.median(ctx))


if __name__ == "__main__":
    main()
