"""Every quantitative claim of the paper, computed from the primary results.

Writes paper/numbers.tex (one \\newcommand per number, formatted as the paper
prints it) and paper/numbers.json (value, printed text, source). The claim
checker (tests/test_claims.py, registry paper/claims.yaml) compares the
numbers hard-coded in main.tex against these.

Sources (all read-only):
  results/traces/swebench{,_kv8}/*      the 100 traced sessions (trace profile, VRAM wall)
  results/gpu/g1.jsonl, g2.jsonl        RTX 3080 decode vs context, in VRAM and past the wall
  results/search/main.json, effort5*.json, area3000*.json, local_*.json   searcher runs
  results/search/sweep_main_summary.json, sweep_main.tsv[.gz]             exhaustive sweep
  results/search/area_main.json, effort_main.json, sensitivity_main.json,
  heldout_extra.json, gpu_scale.json
  harness/ (imported, not modified)     per-token step times, capacity, the Domeline of Fig. 9

  python experiments/make_numbers.py            # everything (about a minute)
  python experiments/make_numbers.py --fast     # skip the numbers that need the integrator
"""

import argparse
import ast
import gzip
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

RES = ROOT / "results"
SEARCH = RES / "search"
# The 78 MB sweep table is not in git; the main checkout keeps it.
SWEEP_TSV = [SEARCH / "sweep_main.tsv", SEARCH / "sweep_main.tsv.gz",
             Path("/localhome/tsousape/domeline-agent-speedup/results/search/sweep_main.tsv")]

# External constants the paper uses, with their source (not computed here).
GA102_DIE_MM2 = 628.4          # NVIDIA Ampere GA102 whitepaper: 628.4 mm2 (RTX 3080 die)
GEM5_H_PER_POINT = (1, 3)      # assumption stated in the paper: a gem5-class evaluator at 1-3 h per design
HOURS_PER_YEAR = 24 * 365.25

NUM = {}


def put(name, value, fmt, source, note=""):
    """fmt: int d -> d decimals; ',' -> integer with thousands separator; ',1' etc."""
    if name in NUM:
        raise KeyError(f"duplicate number {name}")
    if value is None:
        text = "??"
    elif fmt == ",":
        text = f"{round(value):,}"
    elif isinstance(fmt, int):
        text = f"{value:.{fmt}f}"
    else:
        raise ValueError(fmt)
    NUM[name] = {"value": None if value is None else float(value), "text": text, "source": source,
                 "note": note}


def J(name):
    return json.loads((SEARCH / name).read_text())


# --------------------------------------------------------------------------- traces
def trace_rows():
    """Same rule as figs/fig_trace_profile.py: the int8-KV re-run replaces a wall session."""
    tdir, kv8 = RES / "traces/swebench", RES / "traces/swebench_kv8"
    frozen = set((ROOT / "traces/workload_main.tsv").read_text().split()[1:])
    rows = []
    for d in sorted(tdir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if (kv8 / name / "result.json").exists():
            d = kv8 / name
        res, tr = d / "result.json", d / "trace.jsonl"
        if not (res.exists() and tr.exists()):
            continue
        r = json.loads(res.read_text())
        turns = [json.loads(line) for line in tr.read_text().splitlines()]
        ov = r.get("exec_overhead_s", 0.0)
        rows.append({
            "name": name, "frozen": name in frozen, "exit": r["exit_status"],
            "wall": d.parent == kv8, "turns": len(turns),
            "max_ctx": max(t["ctx_start"] + (t["gen_tokens"] or 0) for t in turns),
            "decode": sum(t["t_decode_s"] for t in turns), "prefill": sum(t["t_prefill_s"] for t in turns),
            "tools": sum(max(0.0, t["t_tool_s"] - (ov if t["command"] else 0.0)) for t in turns),
        })
    return rows


def traces():
    rows = trace_rows()
    src = "results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py)"
    allr = rows
    fro = [r for r in rows if r["frozen"]]
    late = [r for r in rows if not r["frozen"]]
    put("nSessionsTraced", len(allr), ",", src)
    put("nSessionsFrozen", len(fro), ",", src + "; traces/workload_main.tsv")
    put("nSessionsLate", len(late), ",", src)
    put("nSessionsSubmitted", sum(r["exit"] == "Submitted" for r in allr), ",", src)
    for tag, rs in (("", allr), ("Frozen", fro)):
        tot = sum(r["decode"] + r["prefill"] + r["tools"] for r in rs)
        for k, nm in (("decode", "Decode"), ("prefill", "Prefill"), ("tools", "Tool")):
            put(f"share{nm}{tag}Pct", 100 * sum(r[k] for r in rs) / tot, 0,
                src, f"share of 3080 session time, {'all 100' if not tag else 'frozen 89'} sessions")
    ctx = sorted(r["max_ctx"] for r in allr)
    put("maxCtxK", ctx[-1] / 1024, 0, src, "final context, K tokens")
    put("medianCtxK", statistics.median(ctx) / 1024, 1, src, "median final context, K tokens")
    for tag, rs in (("", allr), ("Frozen", fro), ("Late", late)):
        n = sum(r["wall"] for r in rs)
        put(f"vramWall{tag}Count", n, ",", src, "sessions re-run with int8 KV (fp16 run hit the 32K wall)")
        put(f"vramWall{tag}Pct", 100 * n / len(rs), 0, src, f"{n}/{len(rs)}")
    return rows


# --------------------------------------------------------------------------- GPU
def gpu_points(f, weights="Q8_0"):
    out = {}
    for line in (RES / "gpu" / f).read_text().splitlines():
        r = json.loads(line)
        if r["model_filename"].endswith(f"-{weights}.gguf") and r["n_gen"] > 0:
            out.setdefault(r["type_k"], {})[r["n_depth"]] = 1e3 / r["avg_ts"]
    return out


def gpu_walls(weights="Q8_0"):
    """Last depth that fit and first that failed, per KV type (G1 status table)."""
    ok, fail = {}, {}
    lines = (RES / "gpu/g1_status.tsv").read_text().splitlines()[1:]
    for line in lines:
        w, kv, depth, st = line.split("\t")
        if w != weights:
            continue
        depth = int(depth)
        if st == "ok":
            ok[kv] = max(ok.get(kv, 0), depth)
        else:
            fail[kv] = min(fail.get(kv, 1 << 30), depth)
    return ok, fail


def gpu():
    src1, src2 = "results/gpu/g1.jsonl", "results/gpu/g2.jsonl"
    g1, g2 = gpu_points("g1.jsonl"), gpu_points("g2.jsonl")
    ok, fail = gpu_walls()
    st = "results/gpu/g1_status.tsv (Q8_0)"
    put("wallFpSixteenLoK", ok["f16"] / 1024, 0, st, "last fp16-KV depth that fit")
    put("wallFpSixteenHiK", fail["f16"] / 1024, 0, st, "first fp16-KV depth that failed")
    put("wallIntEightLoK", ok["q8_0"] / 1024, 0, st)
    put("wallIntEightHiK", fail["q8_0"] / 1024, 0, st)
    put("gpuDecodeEmptyMs", g1["f16"][0], 1, src1, "Q8_0, fp16 KV, empty context")
    put("gpuDecodeFourKMs", g1["f16"][4096], 1, src1)
    put("gpuDecodeThirtyTwoKMs", g1["f16"][32768], 0, src1, "Q8_0, fp16 KV, 32K")
    put("gpuDecodeSixtyFourKMs", g1["q8_0"][65536], 0, src1, "Q8_0, int8 KV, 64K (fp16 does not fit)")
    put("gpuOffloadThirtyTwoKMs", g2["f16"][32768], 0, src2, "fp16 KV in host memory (-nkvo 1), 32K")
    put("gpuCliffX", g2["f16"][32768] / g1["f16"][32768], 0, f"{src2} / {src1}", "past-the-wall slowdown at 32K")
    put("gpuVramGB", 10, 0, "external: RTX 3080 spec (10 GB GDDR6X)")


# --------------------------------------------------------------------------- sweep
def sweep():
    s = J("sweep_main_summary.json")
    src = "results/search/sweep_main_summary.json"
    put("nDesigns", s["n_designs"], ",", src, "14-knob space, area knobs at ISAAC defaults")
    put("nFeasible", s["n_feasible"], ",", src)
    put("beatGpuPct", 100 * s["frac_beat_gpu"], 1, src, "share of feasible designs faster than the 3080")
    put("nWithinOnePct", s["n_within"]["0.01"], ",", src)
    put("withinOnePctOfFeasiblePct", 100 * s["frac_feasible_within"]["0.01"], 2, src)
    put("nWithinTenPct", s["n_within"]["0.1"], ",", src)
    put("sweepOptimum", s["fastest_speedup"], 2, src, "fastest design in the 14-knob space")
    put("sweepOptimumEffortFive", s["searchers"]["effort5"]["fastest_speedup"], 2, src)
    for cap, nm in (("0", "Zero"), ("1", "One"), ("4", "Four"), ("7", "Seven")):
        put(f"stairSweep{nm}", s["by_effort_cap"][cap]["speedup"], 2, src, f"best design at effort <= {cap} months")
    # Walk rank and first evaluation within 1 % (unconstrained and 5-month cap).
    for tag, nm in (("main", ""), ("effort5", "EffortFive")):
        runs = s["searchers"][tag]["runs"]
        w = next(r for r in runs if r["method"] == "walk")
        put(f"walkGap{nm}Pct", w["gap_pct"], 1, src, "gap of the walk's design to the fastest design")
        put(f"walkFirstWithin{nm}", w["n_first_within_1pct"], ",", src, "first evaluation within 1%")
        put(f"walkEvals{nm}", w["evals"], ",", src, "evaluations the walk used in all")
        for m, mn in (("tpe", "Tpe"), ("hill_climb", "Hill"), ("random", "Random")):
            rs = [r for r in runs if r["method"] == m]
            hit = [r["n_first_within_1pct"] for r in rs if r["n_first_within_1pct"] is not None]
            put(f"{mn.lower()}Within{nm}Count", len(hit), ",", src, f"of {len(rs)} seeds within 1%")
            put(f"{mn.lower()}Seeds{nm}", len(rs), ",", src)
            if hit and m == "tpe":
                put(f"tpeFirstWithin{nm}Lo", min(hit), ",", src)
                put(f"tpeFirstWithin{nm}Hi", max(hit), ",", src)
        ag = [r["n_first_within_1pct"] for r in runs if r["method"].startswith("agent")
              and r["n_first_within_1pct"] is not None]
        put(f"sonnetFirstWithin{nm}Lo", min(ag), ",", src, "Sonnet agent, with and without readings")
        put(f"sonnetFirstWithin{nm}Hi", max(ag), ",", src)
    w = NUM["walkFirstWithin"]["value"]
    put("tpeEvalRatioLo", NUM["tpeFirstWithinLo"]["value"] / w, 1, src, "TPE first-within-1% eval / walk's")
    put("tpeEvalRatioHi", NUM["tpeFirstWithinHi"]["value"] / w, 1, src)
    hz = [r for r in s["searchers"]["effort5_haiku"]["runs"]]
    son = [r for r in s["searchers"]["effort5"]["runs"] if r["method"].startswith("agent")]
    put("agentsWithinEffortFiveCount", sum(r["n_first_within_1pct"] is not None for r in hz + son), ",", src,
        f"Sonnet + Haiku runs within 1% under the 5-month cap, of {len(hz + son)}")
    # Sweep cost and the gem5 extrapolation.
    log = (SEARCH / "sweep_main.log").read_text().splitlines()
    hours = float(log[-1].split(" in ")[-1].split()[0])
    put("sweepHours", hours, 1, "results/search/sweep_main.log (last line)")
    put("sweepWorkers", int(log[0].split(",")[1].split()[0]), ",", "results/search/sweep_main.log (first line)")
    n = s["n_designs"]
    put("gemFiveCpuYearsLo", n * GEM5_H_PER_POINT[0] / HOURS_PER_YEAR, 0, src + " x 1 h per design")
    put("gemFiveCpuYearsHi", n * GEM5_H_PER_POINT[1] / HOURS_PER_YEAR, 0, src + " x 3 h per design")
    return s


def sweep_tsv():
    """Distinct designs once the three area knobs are added (needs the sweep table)."""
    path = next((p for p in SWEEP_TSV if p.exists()), None)
    if path is None:
        put("nDesignsWithArea", None, ",", "sweep_main.tsv not found")
        return
    op = gzip.open if path.suffix == ".gz" else open
    n_x = n_o = 0
    with op(path, "rt") as f:
        head = f.readline().rstrip("\n").split("\t")
        i = [head.index(k) for k in ("mvm", "lm_head", "attention")]
        for line in f:
            c = line.split("\t", 4)
            if "xbar" in (c[i[0]], c[i[1]], c[i[2]]):
                n_x += 1
            else:
                n_o += 1
    from harness.space import KNOBS
    mult = len(KNOBS["xbar_tile"]) * len(KNOBS["adc_per_ima"]) * len(KNOBS["cell_bits"])
    put("nDesignsWithArea", n_x * mult + n_o, ",", f"{path.name}: designs using crossbars x {mult} area variants",
        "distinct designs in the 17-knob space (area knobs are active only when crossbars are used)")
    put("nDesignsXbar", n_x, ",", path.name)


# --------------------------------------------------------------------------- searchers
def first_within(run, target):
    return next((e["n"] for e in run["log"] if e["feasible"] and e["speedup"] >= target), None)


def searchers():
    m = J("main.json")
    src = "results/search/main.json"
    w = next(r for r in m["runs"] if r["method"] == "walk")
    put("walkSpeedup", w["best"]["speedup"], 2, src + " (walk best)")
    put("walkSpeedupShort", w["best"]["speedup"], 1, src + " (walk best)")
    put("walkHeldout", w["best"]["heldout"]["speedup"], 2, src + " (walk best, held-out 44)")
    put("walkBestEval", w["best"]["n"], ",", src)
    put("budget", m["budget"], ",", src)
    put("nSearch", m["n_train"], ",", src)
    put("nHeldout", m["n_test"], ",", src)
    put("amdahlBound", w["gpu_s"] / w["tools_s"], 1, src, "GPU time / tool time, search sessions")
    put("toolShareAtWalkPct", 100 * w["tools_s"] / w["best"]["time_s"], 0, src, "tools / design's workload time")
    tr = w["trail"]
    put("walkStart", tr[0]["speedup"], 2, src + " trail[0]")
    put("walkStartSlowerX", 1 / tr[0]["speedup"], 0, src + " trail[0]")
    by = {t["move"]: t for t in tr}
    put("walkAttention", by["attention=xbar"]["speedup"], 2, src + " trail")
    put("walkSoftmax", by["softmax=star"]["speedup"], 2, src + " trail")
    put("walkLmHead", by["lm_head=xbar"]["speedup"], 2, src + " trail")
    put("walkNorms", by["vector=vfu"]["speedup"], 2, src + " trail")
    sizing = [t["speedup"] for t in tr if t["move"].startswith(("star_units", "vfu_units"))
              and t["speedup"] < by["xbar_tech=xbar_transpose"]["speedup"]]
    put("walkSizingLo", min(sizing), 1, src + " trail (unit sizing before transposed writes)")
    put("walkSizingHi", max(sizing), 1, src + " trail")
    put("walkTranspose", by["xbar_tech=xbar_transpose"]["speedup"], 1, src + " trail")
    put("walkNormsStar", by["vector=star, vfu_units=1"]["speedup"], 2, src + " trail")
    put("walkStarUnits", w["best"]["point"]["star_units"], ",", src + " best point")
    put("walkAreaBeforeShrink", max(t["area_mm2"] for t in tr), ",", src + " trail (largest area)")
    put("walkArea", w["best"]["area_mm2"], ",", src)
    put("walkEffort", w["best"]["effort_months"], 0, src)
    put("walkAreaOverDie", w["best"]["area_mm2"] / GA102_DIE_MM2, 0, src + " / GA102 628.4 mm2")
    remaps = [t for t in tr if t["move"] in ("attention=xbar", "softmax=star", "lm_head=xbar", "vector=vfu")]
    put("walkRemapsFirst", len(remaps), ",", src + " trail")
    # Hill climbing / random / TPE, unconstrained and under the effort cap (gap rows of Table II).
    e5 = J("effort5.json")
    a3 = J("area3000.json")
    ah = J("area3000_haiku.json")
    src3 = "results/search/area3000.json"
    best_known = max(r["best"]["speedup"] for d in (a3, ah) for r in d["runs"] if r.get("best"))
    put("areaThreeKBestKnown", best_known, 2, src3 + " + area3000_haiku.json (best design found by any run)")
    wa = next(r for r in a3["runs"] if r["method"] == "walk")
    put("walkGapAreaThreeKPct", 100 * (1 - wa["best"]["speedup"] / best_known), 1, src3)
    put("walkFirstWithinAreaThreeK", first_within(wa, 0.99 * best_known), ",", src3)
    put("walkEvalsAreaThreeK", wa["evals"], ",", src3)
    put("walkAreaThreeK", wa["best"]["speedup"], 2, src3)
    for m_, nm in (("tpe", "tpe"), ("hill_climb", "hill"), ("random", "random")):
        rs = [r for r in a3["runs"] if r["method"] == m_]
        put(f"{nm}WithinAreaThreeKCount", sum(first_within(r, 0.99 * best_known) is not None for r in rs), ",", src3)
        put(f"{nm}BestAreaThreeK", max((r["best"]["speedup"] for r in rs if r.get("best")), default=0), 2, src3,
            "best of the seeds")
    ag = [first_within(r, 0.99 * best_known) for r in a3["runs"] if r["method"].startswith("agent")]
    put("sonnetFirstWithinAreaThreeKLo", min(ag), ",", src3)
    put("sonnetFirstWithinAreaThreeKHi", max(ag), ",", src3)
    put("sonnetRunsPerVariantLo", 1, ",", src, "main.json: 1 run per variant")
    put("sonnetRunsPerVariantHi", max(sum(r["method"] == "agent" for r in d["runs"]) for d in (m, e5, a3)), ",",
        "effort5.json / area3000.json")
    put("walkFirstWithinRangeHi", max(NUM["walkFirstWithin"]["value"], NUM["walkFirstWithinEffortFive"]["value"],
                                      NUM["walkFirstWithinAreaThreeK"]["value"]), ",", "Table II settings")
    put("walkEvalsRangeLo", min(w["evals"], next(r for r in e5["runs"] if r["method"] == "walk")["evals"], wa["evals"]),
        ",", "main / effort5 / area3000")
    put("walkEvalsRangeHi", max(w["evals"], next(r for r in e5["runs"] if r["method"] == "walk")["evals"], wa["evals"]),
        ",", "main / effort5 / area3000")
    # Haiku: evaluations before stopping, with vs without readings (area cap).
    wr = sum(r["evals"] for r in ah["runs"] if r["method"] == "agent")
    wo = sum(r["evals"] for r in ah["runs"] if r["method"] == "agent_scalar")
    put("haikuStopsSoonerPct", 100 * (1 - wr / wo), 0, "results/search/area3000_haiku.json",
        f"total evals {wr} with readings vs {wo} without")
    # The 4B model: runs that beat the GPU.
    lm = J("local_main.json")
    put("localBeatGpuReadings", sum(r["best"]["speedup"] > 1 for r in lm["runs"] if r["method"] == "agent"), ",",
        "results/search/local_main.json")
    put("localBeatGpuScalar", sum(r["best"]["speedup"] > 1 for r in lm["runs"] if r["method"] == "agent_scalar"),
        ",", "results/search/local_main.json")
    put("localRuns", sum(r["method"] == "agent" for r in lm["runs"]), ",", "results/search/local_main.json")


# --------------------------------------------------------------------------- effort, area, held-out, GPUs, sensitivity
def effort():
    d = J("effort_main.json")
    src = "results/search/effort_main.json"
    rows = {r["cap"]: r for r in d["rows"] if r["method"] == "walk"}
    for cap, nm in ((0, "Zero"), (1, "One"), (4, "Four"), (7, "Seven")):
        put(f"stairWalk{nm}", rows[cap]["speedup"], 2, src, f"walk under effort <= {cap} months")
    put("stairGainLastPct", 100 * (rows[7]["speedup"] / rows[4]["speedup"] - 1), 0, src, "7 vs 4 months")
    tpe = [r for r in d["rows"] if r["method"] == "tpe"]
    put("tpeEffortBudget", d["budget"], ",", src)
    lo = min(r["cap"] for r in tpe if r.get("speedup"))
    put("tpeFirstFeasibleCap", lo, 0, src, "lowest effort cap where TPE found a feasible design")


def area():
    d = J("area_main.json")
    src = "results/search/area_main.json"
    sub = [s for s in d["sub"] if s["feasible"]]
    put("nSub", d["n_sub"], ",", src, "crossbar subspace (8 knobs, rest at the headline design)")
    put("areaBudget", d["budget"], ",", src)
    caps = sorted({r["cap"] for r in d["rows"]}, key=lambda c: math.inf if c is None else c)
    put("nAreaCaps", len(caps) - (None in caps), ",", src, "finite caps")
    feas_caps = [c for c in caps if any(s["area_mm2"] <= (c or math.inf) for s in sub)]
    put("nAreaCapsFeasible", len(feas_caps), ",", src, "caps (the uncapped run included) where a design fits")

    def front(cap):
        f = [s for s in sub if s["area_mm2"] <= (cap or math.inf)]
        return max(f, key=lambda s: s["speedup"]) if f else None

    gaps, evals, misses = [], [], {}
    rows = {(r["cap"], r["method"]): r for r in d["rows"]}
    for c in feas_caps:
        ref = front(c)["speedup"]
        w = rows[(c, "walk")]
        gaps.append(100 * (1 - w["best"]["speedup"] / ref))
        evals.append(w["evals"])
        t = rows[(c, "tpe")]["best"]
        misses[c] = (ref / t["speedup"], t["speedup"], ref)
    put("walkAreaGapMaxPct", max(gaps), 1, src, "walk vs exact subspace front, max over the feasible caps")
    put("walkAreaEvalsLo", min(evals), ",", src, "evaluations used, feasible caps")
    put("walkAreaEvalsHi", max(evals), ",", src)
    put("tpeAreaMissMaxX", max(m[0] for m in misses.values()), 0, src, "front / TPE, worst feasible cap")
    low = {c: m for c, m in misses.items() if c is not None and c <= 5000}
    put("tpeAreaMissLowLoX", min(m[0] for m in low.values()), 1, src, "front / TPE at caps <= 5,000 mm2")
    put("tpeAreaMissLowHiX", max(m[0] for m in low.values()), 1, src)
    put("tpeAreaLowLo", min(m[1] for m in low.values()), 2, src, "TPE's result, caps <= 5,000 mm2")
    put("tpeAreaLowHi", max(m[1] for m in low.values()), 2, src)
    put("frontAreaLowLo", min(m[2] for m in low.values()), 2, src, "exact front, caps <= 5,000 mm2")
    put("frontAreaLowHi", max(m[2] for m in low.values()), 2, src)
    put("tpeAreaEightK", misses[8000][1], 2, src, "TPE at the 8,000 mm2 cap")
    put("frontAreaEightK", misses[8000][2], 2, src, "exact front at 8,000 mm2")
    tpe_ok = [c for c, m in misses.items() if m[0] <= 1.0 + 1e-9]
    put("tpeMatchesFrontCount", len(tpe_ok), ",", src, f"caps where TPE >= front: {tpe_ok}")
    # The front itself (Table in results.md; the Area paragraph of the paper).
    pts = sorted(sub, key=lambda s: (s["area_mm2"], -s["speedup"]))
    fr, best = [], 0
    for s in pts:
        if s["speedup"] > best + 1e-12:
            best = s["speedup"]
            fr.append(s)

    def at(pred):
        """Smallest front point with the given area-knob settings and 64 units, transposed writes."""
        c = [s for s in fr if pred(s["point"])]
        return min(c, key=lambda s: s["area_mm2"])

    lean = at(lambda p: p["xbar_tile"] == "lean" and p["cell_bits"] == 2 and p["adc_per_ima"] == 8)
    c4 = at(lambda p: p["xbar_tile"] == "lean" and p["cell_bits"] == 4 and p["adc_per_ima"] == 8
            and p["star_units"] == 64)
    a4 = at(lambda p: p["cell_bits"] == 4 and p["adc_per_ima"] == 4 and p["star_units"] == 64
            and p["xbar_tech"] == "xbar_transpose")
    a2 = at(lambda p: p["cell_bits"] == 4 and p["adc_per_ima"] == 2 and p["star_units"] == 64
            and p["xbar_tech"] == "xbar_transpose")
    for nm, s in (("Lean", lean), ("CellFour", c4), ("AdcFour", a4), ("AdcTwo", a2)):
        put(f"front{nm}Speedup", s["speedup"], 2, src, f"exact front, {s['point']['xbar_tile']} tile, "
            f"{s['point']['cell_bits']}-b cells, {s['point']['adc_per_ima']} ADCs, link {s['point']['link']}")
        put(f"front{nm}Area", s["area_mm2"], ",", src)
    put("frontMinArea", min(s["area_mm2"] for s in sub), ",", src, "smallest feasible subspace design")
    put("frontTop", fr[-1]["speedup"], 2, src, "fastest subspace design")


def heldout():
    d = J("heldout_extra.json")
    src = "results/search/heldout_extra.json"
    put("lateSpeedup", d["designs"]["main"]["extra_speedup"], 1, src)
    put("lateAmdahl", d["designs"]["main"]["amdahl"], 0, src)
    put("lateEffortFive", d["designs"]["effort5"]["extra_speedup"], 1, src)


def gpus():
    d = J("gpu_scale.json")
    src = "results/search/gpu_scale.json"
    s = d["sets"]["search"]["speedup"]
    put("vsFourNinety", s["RTX 4090"], 1, src, "search sessions, scaled")
    put("vsHHundred", s["H100 SXM"], 1, src, "search sessions, scaled")
    put("gpuBandwidthThreeEighty", d["gpus"]["RTX 3080"]["bw_gb_s"], 0, src + " (vendor peak)")


def sensitivity():
    d = J("sensitivity_main.json")
    src = "results/search/sensitivity_main.json"
    base = d["base"]["speedup"]
    put("sensBase", base, 2, src, "all 89 sessions")
    r = {(x["param"], x["factor"]): x for x in d["rows"]}
    put("sensReadHalf", r[("xbar_read", 0.5)]["reopt_speedup"], 1, src)
    put("sensReadDouble", r[("xbar_read", 2.0)]["reopt_speedup"], 1, src)

    def dev(params):
        return max(abs(r[(p, f)][k] / base - 1) * 100 for p in params for f in (0.5, 2.0)
                   for k in ("fixed_speedup", "reopt_speedup"))
    put("sensStarMaxPct", dev(("star_ops", "star_bw")), 1, src, "nonlinear unit throughput and bandwidth")
    put("sensSmallMaxPct", dev(("link", "xbar_write", "host_attn", "vfu_ops")), 1, src,
        "link, write latency, host kernels, vector unit")


# --------------------------------------------------------------------------- integrator
def integrator(rows):
    from common import gpu as gpu_baseline, library, model, sessions, split
    from harness.integrator import Integrator
    from harness.session import GpuBaseline, Replayer, WorkloadReplayer
    from harness.space import Space, canonical

    spec, lib = model(), library()
    tr, te = split(sessions())
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in tr + te), host="cpu")
    head = canonical(J("main.json")["runs"][0]["best"]["point"])
    d, l2 = space.to_design(head)
    integ = Integrator(spec, l2)
    src = "harness integrator, headline design of results/search/main.json"
    st = {c: integ.step(d, 1, c) for c in (4096, 32768, 65536)}
    put("decodeFourKMs", st[4096].total * 1e3, 2, src, "per-token decode at 4K")
    put("decodeSixtyFourKMs", st[65536].total * 1e3, 2, src, "per-token decode at 64K")
    g1 = gpu_points("g1.jsonl")
    put("perTokenFourKX", g1["f16"][4096] / (st[4096].total * 1e3), 0, src + " vs g1.jsonl fp16")
    put("perTokenThirtyTwoKX", g1["f16"][32768] / (st[32768].total * 1e3), 0, src + " vs g1.jsonl fp16")
    xb = head["xbar_tech"]
    put("xbarFloorMs", st[4096].terms[xb] * 1e3, 2, src, "crossbar read term per token at 4K")
    put("xbarFloorUs", st[4096].terms[xb] * 1e6, 0, src)
    put("nArraysM", d.units[xb] / 1e6, 2, src, "crossbar arrays")
    w_only = space.xbar_need({**head, "attention": "cpu"})
    put("kvArraysPct", 100 * (1 - w_only / space.xbar_need(head)), 0, src, "arrays holding KV at the 64K ref ctx")

    def wall(lo=1024, hi=1 << 20):
        while hi - lo > 64:
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if integ.capacity_over(d, mid) else (mid, hi)
        return hi
    put("designWallK", wall() / 1024, 1, src, "first context the headline design runs out of arrays")
    # Fig. domeline_agent (rule of Domeline/figures/src/fig_domeline.py, panel_real).
    tools = sum(t.t_tool_s for s in tr for t in s.turns)
    gen = sum(t.gen_tokens for s in tr for t in s.turns)
    t_fix = tools / gen
    put("tFixMs", t_fix * 1e3, 2, src + "; search sessions' tool time per generated token")
    put("amdahlTokS", 1 / t_fix, ",", src)
    import numpy as np
    L = np.linspace(1024, 65536, 128).astype(int)
    roof = np.array([1 / (t_fix + integ.step(d, 1, int(c)).total) for c in L])
    frac = roof * t_fix
    put("roofAmdahlLoPct", 100 * frac.min(), 0, src, "roof / Amdahl sheet over 1K..64K")
    put("roofAmdahlHiPct", 100 * frac.max(), 0, src)
    gw = 34816
    put("gpuWallFigK", gw / 1024, 0, "fig_domeline.py: between the last fit (32K) and first fail (36K)")
    fl = sorted(g1["f16"].items())
    m = L <= gw
    tg = np.interp(L[m], [x for x, _ in fl], [y / 1e3 for _, y in fl])
    ratio = roof[m] * (t_fix + tg)
    put("roofOverGpuLoX", ratio.min(), 0, src + " vs g1.jsonl fp16", "design roof / 3080, before the 3080's wall")
    put("roofOverGpuHiX", ratio.max(), 0, src + " vs g1.jsonl fp16")
    # Replay accuracy: shared curves vs exact per-turn replay, headline design, search sessions.
    fast = WorkloadReplayer(integ).run(d, tr)
    exact = [Replayer(integ).run(d, s).llm for s in tr]
    errs = [abs(f.llm / e - 1) * 100 for f, e in zip(fast, exact)]
    put("replayErrMaxPct", max(errs), 2, src, "max per-session |fast/exact - 1| of LLM time, search sessions")
    put("replayErrWorkloadPct", abs(sum(f.llm for f in fast) / sum(exact) - 1) * 100, 2, src,
        "|fast/exact - 1| of the workload's LLM time")
    # G3: the 3080's measured decode curve predicts the measured session decode time.
    g16 = gpu_baseline()
    g8 = GpuBaseline(ROOT / "results/gpu/g1.jsonl", weights="Q8_0", kv="q8_0")
    pred = meas = 0.0
    per = []
    for s in sessions(which="all"):
        g = g8 if s.meta["gpu_kv"] == "q8_0" else g16
        p = sum(g.curve.sum_tokens(t.ctx_start, t.gen_tokens) for t in s.turns)
        mm = sum(t.t_decode_gpu_s for t in s.turns)
        pred += p
        meas += mm
        per.append(100 * (p / mm - 1))
    put("gpuReplayErrPct", 100 * (pred / meas - 1), 1, "harness GpuBaseline curve vs measured decode, 100 sessions",
        f"per-session range {min(per):.1f}..{max(per):.1f}%")
    per.sort()
    put("gpuReplayErrP10Pct", per[len(per) // 10], 1, "per-session decile")
    put("gpuReplayErrP90Pct", per[9 * len(per) // 10], 1, "per-session decile")


# --------------------------------------------------------------------------- method parameters (from the code)
def params():
    import inspect
    from harness import search
    from harness.space import KNOBS, LINKS
    sig = inspect.signature(search.domeline_walk).parameters
    src = "harness/search.py domeline_walk defaults"
    put("walkEpsPct", 100 * sig["tol"].default, 1, src)
    put("walkStopPct", 100 * sig["stop_frac"].default, 0, src)
    put("shrinkTolPct", 100 * sig["shrink_tol"].default, 0, src)
    put("nKnobsAll", len(KNOBS), ",", "harness/space.py KNOBS")
    put("nKnobsArea", 3, ",", "harness/space.py KNOBS (xbar_tile, adc_per_ima, cell_bits)")
    put("nKnobsBase", len(KNOBS) - 3, ",", "harness/space.py KNOBS minus the 3 area knobs")
    put("nLinks", len(LINKS), ",", "harness/space.py LINKS")
    import yaml
    lib = yaml.safe_load((ROOT / "lib/tech/pim.yaml").read_text())
    x = next(t for t in lib if t["name"] == "xbar_isaac")
    put("xbarRows", x["params"]["rows"], ",", "lib/tech/pim.yaml xbar_isaac")
    put("xbarCellBits", x["params"]["bits_per_cell"], ",", "lib/tech/pim.yaml xbar_isaac")
    put("xbarReadNs", float(x["params"]["t_read_s"]) * 1e9, 0, "lib/tech/pim.yaml xbar_isaac")
    t = ast.parse((ROOT / "tests/test_terint_repro.py").read_text())
    n = 1
    for node in ast.walk(t):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "parametrize":
            v = node.args[1]
            if isinstance(v, (ast.List, ast.Tuple)):
                n *= len(v.elts)
    put("terintCases", n, ",", "tests/test_terint_repro.py (product of the parametrizations)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip the integrator-based numbers")
    ap.add_argument("--out", default=str(ROOT / "paper"))
    args = ap.parse_args()
    rows = traces()
    gpu()
    sweep()
    sweep_tsv()
    searchers()
    effort()
    area()
    heldout()
    gpus()
    sensitivity()
    params()
    if not args.fast:
        integrator(rows)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    lines = ["% Generated by experiments/make_numbers.py from the results files. Do not edit.",
             "% Each macro prints a number as the paper does; see numbers.json for its source."]
    for k, v in NUM.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v['text']}}}  % {v['source']}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    (out / "numbers.json").write_text(json.dumps(NUM, indent=1) + "\n")
    print(f"{len(NUM)} numbers -> {out / 'numbers.tex'}, {out / 'numbers.json'}")


if __name__ == "__main__":
    main()
