"""E1 analysis: does the Roofline see what the Domeline sees?

Reads the output of roofline_vs_domeline.py (the finished tsv, or the chunks
written so far, marked partial) and reports, per Roofline variant
(main / host / local / local_host, see roofline.py):

  (a) bottleneck agreement: Roofline's bottleneck class vs the Domeline
      binding class (fine classes and coarse ones), overall, broken down by
      every mapping knob, and the confusion table;
  (b) Spearman rank correlation between the Roofline and Domeline times
      (the same as between the speedups: speedup = GPU time / time);
  (c) where Roofline's best design lands in the Domeline ranking (ties
      reported as a range), its Domeline speedup vs the Domeline optimum, and
      top-k overlap (k = 1, 10, 100, 1000).

  python experiments/rvd_analyze.py --tag smoke              # session level
  python experiments/rvd_analyze.py --ctx --tag main         # context grid
Output: results/search/rvd_<tag>_summary.json, results/search/rvd_ctx_<tag>.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ROOT
from roofline import COARSE, COMPONENTS, VARIANTS
from roofline_vs_domeline import CTX_COLS, CTX_POINTS, DL_TERMS, KNOBS14, SESSION_COLS

COMP_CLASS = {"xbar_agg": "xbar", "xbar_local": "xbar", "vfu:c": "vfu", "vfu:b": "vfu",
              "star:c": "star", "star:b": "star", "host:c": "host", "host:b": "host"}
BREAKDOWN = ["attention", "mvm", "lm_head", "softmax", "vector", "edge", "link", "kv_bits", "xbar_tech"]
TOPK = (1, 10, 100, 1000)


def load(name, cols):
    full = ROOT / "results/search" / f"{name}.tsv"
    if full.exists():
        return pd.read_csv(full, sep="\t"), False
    parts = sorted((ROOT / "results/search" / f"{name}.parts").glob("chunk*.tsv"))
    return pd.concat([pd.read_csv(p, sep="\t", header=None, names=cols) for p in parts], ignore_index=True), True


def rf_view(df, variant):
    comps = VARIANTS[variant]
    m = df[[f"rf_{c}" for c in comps]].to_numpy()
    tot = m.sum(axis=1)
    arg = m.argmax(axis=1)
    cls = np.array([COMP_CLASS[comps[i]] for i in arg], dtype=object)
    bound = np.array(["compute" if (comps[i].startswith("xbar") or comps[i].endswith(":c")) else "bandwidth"
                      for i in arg], dtype=object)
    cls[tot <= 0] = "none"
    bound[tot <= 0] = "none"
    return tot, cls, bound


def ranks(x):
    return pd.Series(x).rank(method="min").to_numpy().astype(int)


def analyze(df, dl_t, dl_cls, gpu=None):
    """dl_t: Domeline time (ranking), dl_cls: Domeline binding class; gpu: dict for session speedups."""
    n = len(df)
    out = {"n": n, "domeline_binding": pd.Series(dl_cls).value_counts().to_dict()}
    dl_rank = ranks(dl_t)
    dl_best = float(np.min(dl_t))
    sp = (lambda t: gpu["gpu_time_s"] / t) if gpu else None
    if gpu:
        out["domeline_best_speedup"] = sp(dl_best)
    for v in VARIANTS:
        tot, cls, bound = rf_view(df, v)
        r = {}
        agree = cls == dl_cls
        r["roofline_bottleneck"] = pd.Series(cls).value_counts().to_dict()
        r["roofline_bound"] = pd.Series(bound).value_counts().to_dict()
        r["agree"] = float(agree.mean())
        coarse_rf = np.array([COARSE[c] for c in cls], dtype=object)
        coarse_dl = np.array([COARSE[c] for c in dl_cls], dtype=object)
        r["agree_coarse"] = float((coarse_rf == coarse_dl).mean())
        # Where the Domeline's binding term is one Roofline can name at all.
        nameable = np.isin(dl_cls, ["xbar", "vfu", "star"] + (["host"] if "host" in v else []))
        r["frac_dl_binding_nameable"] = float(nameable.mean())
        r["agree_when_nameable"] = float(agree[nameable].mean()) if nameable.any() else None
        r["by_knob"] = {k: {str(val): {"n": int(m.sum()), "agree": float(agree[m].mean())}
                            for val in sorted(df[k].unique(), key=str) for m in [(df[k] == val).to_numpy()]}
                        for k in BREAKDOWN if k in df}
        r["confusion"] = pd.crosstab(pd.Series(dl_cls, name="domeline"), pd.Series(cls, name="roofline")) \
            .to_dict(orient="index")
        # The designs that matter: the Domeline's top 1% / 10% (and, at session level, those beating the GPU).
        subsets = {"dl_top1pct": dl_rank <= max(1, n // 100), "dl_top10pct": dl_rank <= max(1, n // 10)}
        if gpu:
            subsets["beats_gpu"] = gpu["gpu_time_s"] / dl_t > 1
        r["subsets"] = {}
        for name, m in subsets.items():
            s = {"n": int(m.sum()), "agree": float(agree[m].mean()) if m.any() else None,
                 "domeline_binding": pd.Series(dl_cls[m]).value_counts().to_dict(),
                 "roofline_bottleneck": pd.Series(cls[m]).value_counts().to_dict()}
            if m.sum() > 2 and (tot[m] > 0).all():
                s["spearman"] = float(spearmanr(tot[m], dl_t[m]).statistic)
                rr = tot[m] / (dl_t[m] - (gpu["tools_time_s"] if gpu else 0.0))
                s["roofline_over_domeline_llm_median"] = float(np.median(rr))
            r["subsets"][name] = s
        pos = tot > 0
        rho = spearmanr(tot[pos], dl_t[pos]).statistic if pos.sum() > 2 else None
        r["spearman"] = float(rho) if rho is not None else None
        r["n_ranked"] = int(pos.sum())
        ratio = tot[pos] / (dl_t[pos] - (gpu["tools_time_s"] if gpu else 0.0))
        r["roofline_over_domeline_llm"] = {q: float(np.quantile(ratio, float(q))) for q in ("0.1", "0.5", "0.9")}
        # (c) Roofline's best (ties as a range); 0-time designs count, they are Roofline's answer.
        best_rf = tot.min()
        tied = np.flatnonzero(tot <= best_rf * (1 + 1e-9) + 1e-15)
        tr = dl_rank[tied]
        c = {"n_tied": int(len(tied)), "domeline_rank": {"best": int(tr.min()), "median": float(np.median(tr)),
                                                         "worst": int(tr.max())},
             "domeline_rank_pct": {"best": float(tr.min() / n), "worst": float(tr.max() / n)},
             "domeline_time_over_opt": {"best": float(dl_t[tied].min() / dl_best),
                                        "median": float(np.median(dl_t[tied]) / dl_best),
                                        "worst": float(dl_t[tied].max() / dl_best)}}
        if len(tied) <= 5:
            c["points"] = [{k: (df[k].iloc[i].item() if hasattr(df[k].iloc[i], "item") else df[k].iloc[i])
                            for k in KNOBS14} for i in tied]
        if gpu:
            c["domeline_speedup"] = {"best": sp(dl_t[tied].min()), "worst": sp(dl_t[tied].max())}
            c["roofline_speedup"] = gpu["gpu_time_s"] / (best_rf + gpu["tools_time_s"])
            c["roofline_speedup_llm_only"] = (gpu["gpu_llm_s"] / best_rf) if best_rf > 0 else float("inf")
        # Roofline among offloading designs only (rf > 0): its best there.
        if (~pos).any() and pos.any():
            i = np.flatnonzero(pos)[np.argmin(tot[pos])]
            c["best_with_offload"] = {"domeline_rank": int(dl_rank[i]), "domeline_time_over_opt": float(dl_t[i] / dl_best),
                                      "point": {k: (df[k].iloc[i].item() if hasattr(df[k].iloc[i], "item") else df[k].iloc[i])
                                                for k in KNOBS14}}
        # Where the Domeline optimum lands in the Roofline ranking.
        j = int(np.argmin(dl_t))
        c["domeline_opt_roofline_rank"] = int(ranks(tot)[j])
        r["best"] = c
        order_rf = np.argsort(tot, kind="stable")
        order_dl = np.argsort(dl_t, kind="stable")
        r["topk_overlap"] = {str(k): float(len(set(order_rf[:k]) & set(order_dl[:k])) / k) for k in TOPK if k < n}
        r["topk_rf_domeline_time_over_opt_median"] = {str(k): float(np.median(dl_t[order_rf[:k]]) / dl_best)
                                                      for k in TOPK if k < n}
        out[v] = r
    return out


def session(tag):
    df, partial = load(f"rvd_{tag}", SESSION_COLS)
    meta = json.loads((ROOT / "results/search" / f"rvd_{tag}.meta.json").read_text())
    dl_cls = df.dl_bind.astype(str).to_numpy(dtype=object)
    res = {"partial": partial, "space": meta["space"], "gpu_time_s": meta["gpu_time_s"],
           "tools_time_s": meta["tools_time_s"]}
    # Check: the Domeline numbers reproduce the sweep.
    rel = (df.dl_time / df.sw_time - 1).abs()
    res["check_vs_sweep"] = {"max_rel_time_err": float(rel.max()),
                             "max_abs_speedup_err": float((df.dl_speedup - df.sw_speedup).abs().max())}
    res.update(analyze(df, df.dl_time.to_numpy(), dl_cls, gpu=meta))
    dst = ROOT / "results/search" / f"rvd_{tag}_summary.json"
    dst.write_text(json.dumps(res, indent=1, default=str))
    return dst, res


def ctx(tag):
    df, partial = load(f"rvd_ctx_{tag}", CTX_COLS)
    res = {"partial": partial, "n": len(df), "points": {}}
    for p, n_new, c in CTX_POINTS:
        sub = df[KNOBS14].copy()
        for k in COMPONENTS:
            sub[f"rf_{k}"] = df[f"{p}_rf_{k}"]
        dl_cls = df[f"{p}_dl_bind"].astype(str).to_numpy(dtype=object).copy()
        over = df[f"{p}_over"].to_numpy() == 1
        dl_cls[over] = "capacity"
        r = analyze(sub, df[f"{p}_dl_total"].to_numpy(), dl_cls)
        r.update(n_new=n_new, ctx=c, frac_over_capacity=float(over.mean()))
        res["points"][p] = r
    dst = ROOT / "results/search" / f"rvd_ctx_{tag}.json"
    dst.write_text(json.dumps(res, indent=1, default=str))
    return dst, res


def brief(res, label):
    lines = [label]
    for v in VARIANTS:
        r = res[v]
        b = r["best"]
        lines.append(f"  {v:10s} agree {r['agree']:.1%} (coarse {r['agree_coarse']:.1%}; when nameable "
                     f"{(r['agree_when_nameable'] or 0):.1%})  spearman {r['spearman']:.3f}  "
                     f"best: {b['n_tied']} tied, dl rank {b['domeline_rank']['best']}-{b['domeline_rank']['worst']} "
                     f"of {res['n']}, dl time x{b['domeline_time_over_opt']['best']:.2f}-"
                     f"{b['domeline_time_over_opt']['worst']:.2f} of opt; top10/100 overlap "
                     f"{r['topk_overlap'].get('10', float('nan')):.2f}/{r['topk_overlap'].get('100', float('nan')):.2f}")
        for name, s in r["subsets"].items():
            lines.append(f"      {name:12s} n={s['n']:6d} agree {s['agree'] if s['agree'] is not None else float('nan'):.1%} "
                         f"spearman {s.get('spearman', float('nan')):.3f} rf/dl {s.get('roofline_over_domeline_llm_median', float('nan')):.3f} "
                         f"dl {s['domeline_binding']} rf {s['roofline_bottleneck']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ctx", action="store_true", help="analyze the context grid instead")
    args = ap.parse_args()
    if args.ctx:
        dst, res = ctx(args.tag)
        for p, r in res["points"].items():
            print(brief(r, f"{p} (over capacity {r['frac_over_capacity']:.1%}; Domeline binding {r['domeline_binding']})"))
    else:
        dst, res = session(args.tag)
        print("check vs sweep:", res["check_vs_sweep"])
        print(brief(res, f"session level, Domeline binding {res['domeline_binding']}"))
    print(dst)


if __name__ == "__main__":
    main()
