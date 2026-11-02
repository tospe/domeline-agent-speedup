"""M3 checks on the real workload: regression, speed, coupling, DP optimum, certificate.

  python experiments/large_space_check.py [--quick]
Output: results/large/m3_check.json (and a summary on stdout)

On the 45 search sessions (the frozen split), mini-coder-4b, lib/tech:
  regression  tied large points vs the small space (exact Evaluator), fast evaluator agreement
  speed       ms per evaluation: exact Evaluator on the large space, FastEvaluator with the
              hardware fixed (mapping moves) and with random hardware
  coupling    from START, every window of consecutive ops in one layer moved to one tech:
              the windows that help while each of their ops alone hurts
  dp          the exact mapping optimum (placements and KV bits) for fixed hardware settings,
              against the best tied mapping on the same hardware
  certificate the floor on any design and each design's gap to it
"""

import argparse
import math
import itertools
import json
import random
import sys
import time
from collections import Counter

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import BudgetExhausted, Evaluator
from harness.large_space import FastEvaluator, LargeSpace, certificate, mapping_dp
from harness.space import KNOBS, START, Space

sys.path.insert(0, str(ROOT / "tests"))
from test_large_space import OPTIMUM, WALK_POINT  # noqa: E402

ROLES = ("mvm", "lm_head", "attention", "softmax", "vector", "edge")
HW = {
    "start": START,
    "walk": WALK_POINT,
    "optimum": OPTIMUM,
    # The optimum's hardware with attention allowed on a 4 GB vector-unit KV memory: the whole
    # KV (9.7 GB at 64K, fp16) does not fit, a per-layer split can.
    "vfu4G": {**OPTIMUM, "vfu_mem": 4e9, "vfu_bw": 1e12},
}


def summary(e):
    return {"time_s": e.time_s, "llm_s": e.llm_s, "speedup": e.speedup_vs_gpu, "feasible": e.feasible,
            "reason": e.reason, "binding": e.binding_term, "area_mm2": e.area_mm2, "effort": e.effort_months}


def regression(S, L, train, g, n_random, rng):
    pts = [START, WALK_POINT, OPTIMUM] + [{k: rng.choice(v) for k, v in KNOBS.items()} for _ in range(n_random)]
    ev, fe = Evaluator(S, train, g), FastEvaluator(L, train, g)
    worst = 0.0
    rows = []
    for p in pts:
        a, b = ev(p), fe(L.tie(p))
        rel = abs(b.time_s / a.time_s - 1)
        worst = max(worst, rel)
        rows.append({"small_time_s": a.time_s, "fast_time_s": b.time_s, "rel": rel,
                     "feasible": [a.feasible, b.feasible], "wall": [a.wall_ctx, b.wall_ctx]})
        assert a.feasible == b.feasible and a.wall_ctx == b.wall_ctx
    return {"n": len(pts), "max_rel_fast_vs_small": worst, "points": rows[:3]}


def random_large(L, rng, base, frac=0.3):
    p = L.tie(base)
    for s in L.slots.values():
        if rng.random() < frac:
            p[s.name] = rng.choice(s.options)
    return p


def speed(L, train, g, rng, n_exact, n_fast):
    ex, fe = Evaluator(L, train, g), FastEvaluator(L, train, g)
    pts = [random_large(L, rng, {k: rng.choice(v) for k, v in KNOBS.items()}) for _ in range(n_exact)]
    t = time.time()
    exact = [ex(p) for p in pts]
    t_exact = (time.time() - t) / n_exact
    t = time.time()
    fast = [fe(p) for p in pts]
    t_fast_cold = (time.time() - t) / n_exact
    rel = max(abs(b.time_s / a.time_s - 1) for a, b in zip(exact, fast))
    # Mapping moves, hardware fixed (what a mapping agent does).
    fe2 = FastEvaluator(L, train, g)
    fe2(L.tie(OPTIMUM))
    mp = [random_large(L, rng, OPTIMUM, frac=0.1) for _ in range(n_fast)]
    t = time.time()
    for p in mp:
        fe2(p)
    t_map = (time.time() - t) / n_fast
    # Random hardware every time.
    hp = [random_large(L, rng, {k: rng.choice(v) for k, v in KNOBS.items()}) for _ in range(n_fast)]
    t = time.time()
    for p in hp:
        fe2(p)
    t_hw = (time.time() - t) / n_fast
    return {"exact_s_per_eval": t_exact, "fast_s_per_eval_cold_tables": t_fast_cold,
            "fast_s_per_eval_mapping_moves": t_map, "fast_s_per_eval_random_hw": t_hw,
            "max_rel_fast_vs_exact": rel, "n_exact": n_exact, "n_fast": n_fast,
            "exact_infeasible": sum(not e.feasible for e in exact)}


def coupling(L, train, g, bases=None, layer=17, max_win=5):
    """Ops whose move hurts alone and helps together with their chain neighbors.

    For each base and each window W of consecutive slots of one layer moved to one tech v:
    an op X in W is coupled when moving X alone hurts (delta(X) > 0) but adding X to the rest
    of the window helps (delta(W) < delta(W - X)) and the window as a whole helps (delta(W) < 0).
    """
    from harness.integrator import Integrator
    fe = FastEvaluator(L, train, g)
    bases = bases or {"start": START, "all_cpu": {**START, "mvm": "cpu"},
                      "attn_xbar": {**START, "attention": "xbar", "vfu_units": 16, "star_units": 16}}
    found = []
    for bname, b in bases.items():
        base = L.tie(b)
        t0 = fe(base).time_s
        slots = []
        for o in fe.tables.ops:
            if o.layer == layer and o.name in L.op_slot and L.op_slot[o.name] not in slots:
                slots.append(L.op_slot[o.name])
        delta = lambda m: fe({**base, **m}).time_s - t0 if m else 0.0
        for k in range(2, max_win + 1):
            for i in range(len(slots) - k + 1):
                win = slots[i:i + k]
                for v in ("xbar", "vfu", "star"):
                    if not all(v in L.slots[s].options and base[s] != v for s in win):
                        continue
                    dw = delta({s: v for s in win})
                    if dw >= 0:
                        continue
                    for x in win:
                        dx = delta({x: v})
                        rest = delta({s: v for s in win if s != x})
                        if dx > 0 and dw < rest:
                            found.append({"base": bname, "op": x, "tech": v, "window": win, "delta_alone_s": dx,
                                          "delta_window_s": dw, "delta_window_without_op_s": rest,
                                          "base_time_s": t0})
    # Prefer the smallest window, then the largest flip of the op's marginal.
    found.sort(key=lambda f: (len(f["window"]), -(f["delta_alone_s"] + f["delta_window_without_op_s"]
                                                  - f["delta_window_s"])))
    for f in found[:8]:
        base, v, win, x = L.tie(bases[f["base"]]), f["tech"], f["window"], f["op"]
        detail = {}
        for tag, m in (("base", {}), ("op_alone", {x: v}), ("window_without_op", {s: v for s in win if s != x}),
                       ("window", {s: v for s in win})):
            p = {**base, **m}
            d, lib = L.to_design(p)
            st = Integrator(L.model, lib).step(d, 1, 4096)
            e = fe(p)
            detail[tag] = {"crossings_per_step_4k": st.crossings,
                           "link_s_per_step_4k": st.term("launch") + st.term("transfer"),
                           "step_4k_s": st.total, "workload_time_s": e.time_s,
                           "workload_link_s": e.terms.get("launch", 0) + e.terms.get("transfer", 0)}
        allm = lambda names: {n.replace(f"L{layer}.", f"L{l}."): v for n in names for l in range(L.model.layers)}
        detail["all_layers"] = {"op_alone_time_s": fe({**base, **allm([x])}).time_s,
                                "window_time_s": fe({**base, **allm(win)}).time_s,
                                "window_without_op_time_s": fe({**base, **allm([s for s in win if s != x])}).time_s,
                                "base_time_s": fe(base).time_s}
        f["detail"] = detail
    best = found[0] if found else None
    detail = best["detail"] if best else None
    return {"layer": layer, "n_found": len(found), "smallest": best, "detail": detail, "found": found[:30]}


def single_slot_climb(L, train, g, eps=0.005, start=START, max_evals=100000):
    """Best-improvement climb over single-knob moves (one slot or one hardware knob), accept if > eps faster.

    The large-space analogue of hill climbing with the walk's acceptance rule: does it stall?
    Also counts, at the stall, the tied moves (one role in every layer) that would still clear eps.
    """
    fe = FastEvaluator(L, train, g, budget=max_evals)
    cur = fe(L.tie(start))
    trail = [(cur.n, cur.speedup_vs_gpu, "start")]
    knobs = [(k, KNOBS[k]) for k in L.knobs if k in KNOBS] + [(s.name, s.options) for s in L.slots.values()]
    stop = "no single-knob move clears eps"
    try:
        while True:
            best = None
            for k, opts in knobs:
                for v in opts:
                    if v == cur.point.get(k):
                        continue
                    e = fe({**cur.point, k: v})
                    if e.feasible and (best is None or e.time_s < best[0].time_s):
                        best = (e, f"{k}={v}")
            if best is None or best[0].time_s >= cur.time_s * (1 - eps):
                break
            cur = best[0]
            trail.append((cur.n, cur.speedup_vs_gpu, best[1]))
    except BudgetExhausted:
        stop = "budget exhausted"
    fe.budget = math.inf
    tied = []
    for role in ("mvm", "lm_head", "attention", "softmax", "vector", "edge", "kv_bits"):
        for v in KNOBS[role]:
            q = dict(cur.point)
            for s in L.slots.values():
                if s.role == role:
                    q[s.name] = v
            e = fe(q)
            if e.feasible and e.time_s < cur.time_s * (1 - eps):
                tied.append({"move": f"{role}={v} in every layer", "gain": 1 - e.time_s / cur.time_s})
    return {"eps": eps, "stop": stop, "evals": fe.used, "speedup": cur.speedup_vs_gpu, "trail": trail[-12:],
            "n_accepted": len(trail) - 1, "tied_moves_that_clear_eps": tied}


def best_tied(L, fe, hw):
    best = None
    for vals in itertools.product(*(KNOBS[r] for r in ROLES), KNOBS["kv_bits"]):
        p = L.tie({**hw, **dict(zip(ROLES, vals[:-1])), "kv_bits": vals[-1]})
        e = fe(p)
        if e.wall_ctx is None and (best is None or e.llm_s < best.llm_s):
            best = e
    return best


def dp_runs(L, train, g):
    fe = FastEvaluator(L, train, g)
    ex = Evaluator(L, train, g)
    out = {}
    for name, hw in HW.items():
        dp = mapping_dp(fe, L.tie(hw))
        e = fe(dp.point)
        x = ex(dp.point)
        tied = best_tied(L, fe, hw)
        counts = {}
        for s in L.slots.values():
            counts.setdefault(s.role, Counter())[str(dp.point[s.name])] += 1
        out[name] = {"dp_llm_s": dp.llm_s, "eval_llm_s": e.llm_s, "exact_llm_s": x.llm_s,
                     "dp_vs_eval_rel": e.llm_s / dp.llm_s - 1, "dp": summary(e),
                     "tied_best": summary(tied) if tied else None,
                     "tied_best_point": {r: tied.point.get(f"L0.{r}" if r not in ("lm_head",) else r)
                                         for r in ("attention",)} if tied else None,
                     "untie_gain_llm": (tied.llm_s / e.llm_s - 1) if tied else None,
                     "dp_seconds": dp.seconds, "states_max": dp.states_max, "transitions": dp.transitions,
                     "slot_values": {r: dict(c) for r, c in counts.items()}, "point": dp.point}
        print(f"dp {name}: llm {e.llm_s:.3f} s (dp {dp.llm_s:.3f}), {e.speedup_vs_gpu:.3f}x, tied best "
              f"{tied.llm_s if tied else float('nan'):.3f} s, {dp.seconds:.1f} s, {dict(counts['attention'])}",
              flush=True)
    return out


def cert(L, train, g, dp):
    fe = FastEvaluator(L, train, g)
    t = time.time()
    floor, parts = certificate(fe)
    designs = {"start": L.tie(START), "walk": L.tie(WALK_POINT), "optimum": L.tie(OPTIMUM)}
    designs.update({f"dp_{k}": v["point"] for k, v in dp.items()})
    gaps = {}
    for k, p in designs.items():
        e = fe(p)
        gaps[k] = {"time_s": e.time_s, "feasible": e.feasible, "gap": e.time_s / floor - 1,
                   "llm_gap": e.llm_s / (floor - fe.tools_time) - 1}
    return {"floor_s": floor, "amdahl_s": fe.tools_time, "llm_floor_s": floor - fe.tools_time,
            "floor_parts_s": parts, "floor_speedup": fe.gpu_time / floor, "amdahl_speedup": fe.gpu_time / fe.tools_time,
            "seconds": time.time() - t, "gaps": gaps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    ref = max(s.max_ctx for s in train + test)
    S = Space(lib, spec, ref_ctx=ref, host="cpu")
    L = LargeSpace(lib, spec, ref_ctx=ref, host="cpu")
    res = {"seed": args.seed, "sessions": len(train), "layers": spec.layers, "n_slots": len(L.slots),
           "n_knobs": len(L.knobs),
           "log10_points": sum(__import__("math").log10(len(L.slots[s].options)) for s in L.slots)
           + sum(__import__("math").log10(len(KNOBS[k])) for k in L.knobs if k in KNOBS)}
    t0 = time.time()
    res["regression"] = regression(S, L, train, g, 3 if args.quick else 12, rng)
    print("regression", res["regression"]["max_rel_fast_vs_small"], flush=True)
    res["speed"] = speed(L, train, g, rng, 3 if args.quick else 8, 50 if args.quick else 300)
    print("speed", res["speed"], flush=True)
    res["coupling"] = coupling(L, train, g)
    print("coupling", json.dumps(res["coupling"]["smallest"]), flush=True)
    res["single_slot_climb"] = single_slot_climb(L, train, g)
    print("single-slot climb", {k: v for k, v in res["single_slot_climb"].items() if k != "trail"}, flush=True)
    res["dp"] = dp_runs(L, train, g)
    res["certificate"] = cert(L, train, g, res["dp"])
    print("certificate", {k: v for k, v in res["certificate"].items() if k != "gaps"}, flush=True)
    for k, v in res["certificate"]["gaps"].items():
        print(f"  gap {k}: {v['gap']:.4f} (llm {v['llm_gap']:.3f}) feasible={v['feasible']}")
    res["wall_s"] = time.time() - t0
    out = ROOT / "results/large/m3_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
