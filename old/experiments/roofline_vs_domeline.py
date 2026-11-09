"""E1: Roofline vs Domeline over the feasible designs of the exhaustive sweep.

Design list: the feasible rows of results/search/sweep_main.tsv (280,296 of
575,400), in file order (the sweep file is its 20 worker parts concatenated,
each a stride through the product, so a prefix is a fair sample). The space is
the 14-knob space of that sweep: the 3 area knobs (xbar_tile, adc_per_ima,
cell_bits) stay at their defaults (isaac, 8, 2).

For each design:
  Domeline  `Evaluator` on the 45 search sessions (the sweep's numbers):
            workload time, LLM time, speedup vs the 3080, feasible, wall_ctx,
            the LLM-side terms (host, crossbar, vfu, star, transfer, launch,
            write), per-kind time groups, and the binding term = the largest
            LLM-side term.
  Roofline  experiments/roofline.py: per op max(flops/peak, bytes/BW) on the
            op's device, summed per step and replayed over the same sessions
            (same grid, same token counts and contexts as the Domeline replay,
            exactly). Components xbar_agg, xbar_local, vfu:c/b, star:c/b,
            host:c/b; the variants (main = no host ops; host = host ops on the
            host roof; local = per-op crossbar roof) are sums of them.

Common vocabulary (so bottlenecks can be compared):
  Domeline term          -> class          Roofline component -> class
  xbar_isaac/_transpose  -> xbar           xbar_agg/xbar_local -> xbar (compute)
  vfu_puma               -> vfu            vfu:c / vfu:b       -> vfu (compute / bandwidth)
  softmax_star           -> star           star:c / star:b     -> star
  host                   -> host           host:c / host:b     -> host (host variants only)
  transfer               -> transfer       (never: no link in Roofline)
  launch                 -> launch         (never)
  write                  -> write          (never: no writes)
  wall crossed           -> capacity       (never: no wall)
A Roofline variant whose sum is 0 (everything left on the host, main
variant) has bottleneck "none". Coarse classes: xbar/vfu/star = coproc
compute; transfer/launch = data movement; host; write; capacity.

Modes
  --mode session  (E1)  -> results/search/rvd_<tag>.tsv (+ .parts/ chunks, resumable) and rvd_<tag>.meta.json
  --mode ctx      (E1 along the session path) single steps: decode at 4K/16K/32K/64K
                  (inside the workload) and 128K (past it: where the wall is), plus one
                  256-token prefill chunk at 16K -> results/search/rvd_ctx_<tag>.tsv
  Analysis: experiments/rvd_analyze.py.

  python experiments/roofline_vs_domeline.py --mode session --workers 20 --tag main
  python experiments/roofline_vs_domeline.py --mode ctx --workers 20 --tag main
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from common import ROOT, gpu, library, model, sessions, split

from harness.space import KNOBS
from roofline import COMPONENTS, Roofline, SessionWeights, domeline_class

SWEEP = Path("/localhome/tsousape/domeline-agent-speedup/results/search/sweep_main.tsv")
KNOBS14 = [k for k in KNOBS if k not in ("xbar_tile", "adc_per_ima", "cell_bits")]
DL_TERMS = ["host", "xbar", "vfu", "star", "transfer", "launch", "write"]
KIND_GROUPS = {
    "attn": ("attn_score", "attn_value"), "softmax": ("softmax",),
    "mvm": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    "lm_head": ("lm_head",), "vector": ("rmsnorm", "qk_norm", "rope", "act", "residual"),
    "edge": ("embed", "sample"), "kv_append": ("kv_append",), "link": ("link",),
    "reprogram": ("reprogram",),
}
CTX_POINTS = [("d4k", 1, 4096), ("d16k", 1, 16384), ("d32k", 1, 32768), ("d64k", 1, 65536),
              ("d128k", 1, 131072), ("p16k", 256, 16384)]
CHUNK = 500

SESSION_COLS = (KNOBS14 + ["dl_time", "dl_llm", "dl_speedup", "feasible", "wall_ctx", "dl_bind"]
                + [f"dl_{k}" for k in DL_TERMS] + [f"dlk_{k}" for k in KIND_GROUPS]
                + [f"rf_{c}" for c in COMPONENTS] + ["sw_time", "sw_speedup"])
CTX_COLS = KNOBS14 + [f"{p}_{c}" for p, _, _ in CTX_POINTS
                      for c in ["dl_total", "dl_bind", "over"] + [f"dl_{k}" for k in DL_TERMS]
                      + [f"rf_{k}" for k in COMPONENTS]]


def parse_value(k, s):
    if k in ("xbar_scale", "vfu_bw", "vfu_mem"):
        return float(s)
    if k in ("vfu_units", "star_units", "kv_bits"):
        return int(s)
    return s


def feasible_points(limit=0):
    """(point, sweep time_s, sweep speedup) for each feasible row of sweep_main.tsv, in file order."""
    out = []
    with open(SWEEP) as f:
        head = f.readline().rstrip("\n").split("\t")
        ix = {c: i for i, c in enumerate(head)}
        for line in f:
            r = line.rstrip("\n").split("\t")
            if r[ix["feasible"]] != "1":
                continue
            p = {k: parse_value(k, r[ix[k]]) for k in KNOBS14}
            out.append((p, r[ix["time_s"]], r[ix["speedup"]]))
            if limit and len(out) >= limit:
                break
    return out


def fold_terms(terms):
    """Domeline terms -> common classes (summed), plus the binding class."""
    cls = {k: 0.0 for k in DL_TERMS}
    for k, v in terms.items():
        cls[domeline_class(k)] += v
    bind = max(terms, key=terms.get) if terms else "none"
    return cls, domeline_class(bind)


# -- workers -----------------------------------------------------------------

_W = {}


def _init(mode):
    from harness.evaluate import Evaluator
    from harness.integrator import Integrator
    from harness.space import Space
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    _W.update(space=space, rf=Roofline(spec), mode=mode, Integrator=Integrator)
    if mode == "session":
        _W["ev"] = Evaluator(space, train, g)
        _W["sw"] = SessionWeights(train)


def _fmt(x):
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def _session_row(p, sw_time, sw_speedup):
    ev, space, rf, sw = _W["ev"], _W["space"], _W["rf"], _W["sw"]
    e = ev(p)
    ev.cache.clear()
    ev.log.clear()
    d, lib = space.to_design(p)
    rft = sw.replay(lambda n, c, lt: rf.step(d, lib, n, c, lt))
    cls, bind = fold_terms(e.terms)
    if e.wall_ctx is not None:
        bind = "capacity"
    kinds = {g: sum(e.by_kind.get(k, 0.0) for k in ks) for g, ks in KIND_GROUPS.items()}
    row = ([p[k] for k in KNOBS14] + [e.time_s, e.llm_s, e.speedup_vs_gpu, int(e.feasible),
                                       e.wall_ctx if e.wall_ctx is not None else "", bind]
           + [cls[k] for k in DL_TERMS] + [kinds[k] for k in KIND_GROUPS]
           + [rft.get(c, 0.0) for c in COMPONENTS] + [sw_time, sw_speedup])
    return "\t".join(_fmt(x) for x in row)


def _ctx_row(p):
    space, rf = _W["space"], _W["rf"]
    d, lib = space.to_design(p)
    integ = _W["Integrator"](space.model, lib)
    row = [p[k] for k in KNOBS14]
    for _, n, c in CTX_POINTS:
        lt = 1 if n > 1 else None
        s = integ.step(d, n, c, logits_tokens=lt)
        cls, bind = fold_terms(s.terms)      # rvd_analyze maps over=1 to "capacity"
        r = rf.step(d, lib, n, c, lt)
        row += [s.total, bind, int(s.over_capacity)] + [cls[k] for k in DL_TERMS] + [r.get(k, 0.0) for k in COMPONENTS]
    return "\t".join(_fmt(x) for x in row)


def _chunk(args):
    i, items, out = args
    out = Path(out)
    if out.exists():
        return i, 0.0
    t0 = time.time()
    lines = []
    for it in items:
        lines.append(_session_row(*it) if _W["mode"] == "session" else _ctx_row(it[0]))
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(out)
    return i, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["session", "ctx"], default="session")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=0, help="first N feasible designs only")
    args = ap.parse_args()
    pts = feasible_points(args.limit)
    name = f"rvd_{args.tag}" if args.mode == "session" else f"rvd_ctx_{args.tag}"
    cols = SESSION_COLS if args.mode == "session" else CTX_COLS
    parts = ROOT / "results/search" / f"{name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(i, pts[i * CHUNK:(i + 1) * CHUNK], str(parts / f"chunk{i:05d}.tsv"))
            for i in range((len(pts) + CHUNK - 1) // CHUNK)]
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(f"{args.mode}: {len(pts)} feasible designs (14-knob space, area knobs at defaults), "
          f"{len(jobs)} chunks, {len(todo)} to do, {args.workers} workers", flush=True)
    t0, done = time.time(), 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.mode,)) as pool:
        for i, dt in pool.imap_unordered(_chunk, todo):
            done += 1
            el = time.time() - t0
            print(f"chunk {i} ({dt:.0f} s); {done}/{len(todo)}, eta {el / done * (len(todo) - done) / 60:.1f} min",
                  flush=True)
    dst = ROOT / "results/search" / f"{name}.tsv"
    with open(dst, "w") as out:
        out.write("\t".join(cols) + "\n")
        for _, _, fp in jobs:
            out.write(Path(fp).read_text())
    meta = {"mode": args.mode, "n_designs": len(pts), "limit": args.limit, "source": str(SWEEP),
            "space": "14-knob space of the 575,400-design sweep; area knobs at defaults (isaac, 8 ADCs, 2-b cells)",
            "seconds": time.time() - t0, "workers": args.workers}
    if args.mode == "session":
        _init("session")
        ev = _W["ev"]
        meta.update(gpu_time_s=ev.gpu_time, tools_time_s=ev.tools_time, gpu_llm_s=ev.gpu_time - ev.tools_time,
                    n_sessions=len(ev.sessions))
    else:
        meta["points"] = [{"name": a, "n_new": n, "ctx": c} for a, n, c in CTX_POINTS]
    (ROOT / "results/search" / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{dst}: {len(pts)} designs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
