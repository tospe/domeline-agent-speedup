"""Common runner for the H experiments: config -> one RunRecord JSON per (method, seed) + summary.csv.

    python experiments/hrun.py experiments/h/h4.yaml [--seeds 0,1] [--budget 50] [--methods walk,team]
                               [--workers 8] [--set key=value ...] [--out results/h]

Config (YAML; every key can be overridden with --set, values parsed as YAML):

  experiment: h4               results go to results/h/<experiment>/
  space: small | small14 | large   (small14 pins the three area knobs, the paper's 14-knob space)
  evaluator: fast | exact      FastEvaluator (same numbers, ~100x faster) or Evaluator
  sessions: search | heldout | all
  budget: 100                  evaluations per run (distinct designs)
  seeds: [0, 1, 2]
  caps: {area_mm2: .inf, effort_months: .inf}
  verifier: on | off | shadow  on: illegal proposals are not evaluated; shadow: everything is
                               evaluated and verified on the side (H6); off: no verifier
  sampling: host | any         the verifier's sampling rule (REPORT-M2 §3.2; pending, default host)
  exclude: {edge: [vfu]}       values no method may set (team: team_search exclude; baselines: removed)
  reference: {speedup: 9.4684, dp_opt_s: null, dp_file: null}
                               first-1%-band reference and the DP optimum (seconds) if known
  compress: auto               write records as .json.gz (auto: on the large space, ~17 KB per eval raw)
  plugins: [harness.llm_agents]   modules imported first, so their @method registrations exist
  methods:                     names from the registry or "module:function", plain or with kwargs:
    - walk
    - {name: team, coordinate: first}
    - {name: tpe, budget: 1000, label: tpe_1k}

A method is fn(run: Run, **kwargs); it searches with run.ev (budget already set), and may use
run.toolbox, run.record, run.gate, run.seed, run.space, run.sessions. Register with
@method("name") (here or in a plugin module). The runner writes the record and the metrics.
"""

import argparse
import csv
import importlib
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from harness import baselines as B  # noqa: E402
from harness.tools import RunRecord, Toolbox  # noqa: E402

DEFAULTS = {"experiment": "scratch", "space": "small", "evaluator": "fast", "sessions": "search", "budget": 100,
            "seeds": [0], "caps": {}, "verifier": "on", "sampling": "host", "exclude": {},
            "reference": {"speedup": 9.4684, "dp_opt_s": None, "dp_file": None}, "plugins": [], "methods": [],
            "compress": "auto"}
CHECKPOINTS = (10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 30000, 100000)
METHODS: Dict[str, Callable] = {}


def method(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def resolve(name) -> Callable:
    if name in METHODS:
        return METHODS[name]
    if ":" in name:
        mod, fn = name.split(":", 1)
        return getattr(importlib.import_module(mod), fn)
    raise KeyError(f"unknown method {name!r}; registered: {sorted(METHODS)} (or give module:function)")


# ---------------------------------------------------------------------------
# Environment and one run
# ---------------------------------------------------------------------------

@dataclass
class Env:
    space: object
    sessions: list
    gpu: object
    cfg: Dict
    certificate_s: float


_ENV: Dict[str, Env] = {}


def build_env(cfg) -> Env:
    k = json.dumps({x: cfg[x] for x in ("space", "sessions")}, sort_keys=True)
    if k in _ENV:
        return _ENV[k]
    from common import gpu, library, model, sessions, split
    from harness.large_space import FastEvaluator, LargeSpace, certificate
    from harness.space import Space
    train, test = split(sessions())
    sess = {"search": train, "heldout": test, "all": train + test}[cfg["sessions"]]
    spec, lib, g = model(), library(), gpu()
    ref = max(s.max_ctx for s in train + test)
    if cfg["space"] in ("small", "small14"):
        space = Space(lib, spec, ref_ctx=ref, host="cpu")
    elif cfg["space"] == "large":
        space = LargeSpace(lib, spec, ref_ctx=ref, host="cpu")
    else:
        raise ValueError(f"space {cfg['space']!r}")
    # The large space's certificate is a floor for every design (the small space is its tied subset).
    cert, _ = certificate(FastEvaluator(LargeSpace(lib, spec, ref_ctx=ref, host="cpu"), sess, g))
    _ENV[k] = Env(space, sess, g, cfg, cert)
    return _ENV[k]


@dataclass
class Run:
    space: object
    sessions: list
    ev: object
    toolbox: Toolbox
    record: RunRecord
    gate: B.Gate
    seed: int
    cfg: Dict
    table: Dict[str, list]
    extra: Dict = field(default_factory=dict)     # anything a method wants in its record


def _caps(cfg):
    c = cfg.get("caps") or {}
    return float(c.get("area_mm2", math.inf)), float(c.get("effort_months", math.inf))


def make_evaluator(env: Env, cfg, budget):
    from harness.evaluate import Evaluator
    from harness.large_space import FastEvaluator
    area, effort = _caps(cfg)
    E = FastEvaluator if cfg["evaluator"] == "fast" else Evaluator
    return E(env.space, env.sessions, env.gpu, budget=budget, area_cap_mm2=area, effort_cap_months=effort)


def run_one(cfg, mspec, seed) -> Dict:
    for mod in cfg.get("plugins") or []:
        importlib.import_module(mod)
    mspec = {"name": mspec} if isinstance(mspec, str) else dict(mspec)
    name = mspec.pop("name")
    label = mspec.pop("label", name)
    budget = mspec.pop("budget", cfg["budget"])
    fn = resolve(name)
    env = build_env(cfg)
    ev = make_evaluator(env, cfg, budget)
    verifier = None if cfg["verifier"] == "off" else B.make_verifier(env.space, env.sessions,
                                                                   sampling=cfg["sampling"])
    shadow = None
    if cfg["verifier"] == "shadow":
        shadow = ev = B.VerifiedEvaluator(ev, verifier)
        verifier = None
    tb = Toolbox(ev, verifier=verifier)
    rec = RunRecord(cfg["experiment"], label, seed,
                    {**{k: v for k, v in cfg.items() if k != "methods"}, "method": name, "kwargs": mspec,
                     "budget": budget})
    pin = B.PINNED_14 if cfg["space"] == "small14" else None
    table = B.knob_table(env.space, pin=pin, exclude=cfg.get("exclude"))
    gate = B.Gate(ev, verifier, tb, rec, label=label)
    run = Run(env.space, env.sessions, ev, tb, rec, gate, seed, cfg, table)
    t0 = time.time()
    fn(run, **mspec)
    wall = time.time() - t0
    rec_json = rec.to_json(ev, tb)
    m = metrics(ev, rec, gate, env, cfg, budget, wall)
    m.update(method=label, seed=seed)
    if shadow is not None:
        best = ev.best()
        m["shadow"] = shadow.summary([a["n"] for a in rec.accepted if "n" in a], best.n if best else None)
    rec_json.update(metrics=m, extra=run.extra)
    out = Path(cfg["out"]) / cfg["experiment"] / f"{label}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rec_json, indent=1, default=str)
    gz = cfg.get("compress", "auto")
    if gz is True or (gz == "auto" and cfg["space"] == "large"):
        import gzip
        with gzip.open(str(out) + ".gz", "wt") as f:
            f.write(text)
    else:
        out.write_text(text)
    return m


def dp_optimum_s(cfg):
    ref = cfg.get("reference") or {}
    if ref.get("dp_opt_s"):
        return float(ref["dp_opt_s"])
    f = ref.get("dp_file")
    if f and Path(ROOT / f).exists():
        # Agent C's results/large/dp_caps_<tag>.json; read leniently until its format is fixed.
        d = json.loads(Path(ROOT / f).read_text())
        key = cfg.get("dp_key", "uncapped")
        row = d.get(key) if isinstance(d, dict) else None
        if isinstance(row, dict):
            return row.get("time_s")
    return None


def metrics(ev, rec, gate, env, cfg, budget, wall) -> Dict:
    feas = [e for e in ev.log if e.feasible]
    best = min(feas, key=lambda e: e.time_s) if feas else None
    pick = ev.best()
    ref = (cfg.get("reference") or {}).get("speedup")
    m = {"evals": ev.used, "budget": budget, "proposals": gate.proposals or None,
         "illegal": gate.illegal if gate.verifier is not None and gate.proposals else None,
         "illegal_rules": dict(gate.rules) or None,
         "best_speedup": best.speedup_vs_gpu if best else None, "best_eval": best.n if best else None,
         "pick_speedup": pick.speedup_vs_gpu if pick else None,
         "pick_area_mm2": pick.area_mm2 if pick else None, "pick_effort_months": pick.effort_months if pick else None,
         "first_1pct_eval": None, "gap_certificate": None, "gap_dp": None,
         "certificate_s": env.certificate_s, "wall_s": wall, "eval_wall_s": ev.wall_s,
         "tokens_in": sum(t["in"] for t in rec.tokens.values()),
         "tokens_out": sum(t["out"] for t in rec.tokens.values()), "tokens": rec.tokens, "stop": rec.stop}
    if ref:
        m["first_1pct_eval"] = next((e.n for e in ev.log if e.feasible and e.speedup_vs_gpu >= 0.99 * ref), None)
    if best:
        m["gap_certificate"] = best.time_s / env.certificate_s - 1
        dp = dp_optimum_s(cfg)
        if dp:
            m["gap_dp"] = best.time_s / dp - 1
    run_best = None
    for c in CHECKPOINTS:
        if c > max(budget if math.isfinite(budget) else 0, ev.used):
            break
        sub = [e for e in ev.log[:c] if e.feasible]
        run_best = max((e.speedup_vs_gpu for e in sub), default=None)
        m[f"best@{c}"] = run_best
    m["curve"] = _curve(ev)
    return m


def _curve(ev):
    """[(eval n, best feasible speedup so far)] at every improvement."""
    out, best = [], 0.0
    for e in ev.log:
        if e.feasible and e.speedup_vs_gpu > best:
            best = e.speedup_vs_gpu
            out.append((e.n, best))
    return out


# ---------------------------------------------------------------------------
# Built-in methods
# ---------------------------------------------------------------------------

@method("walk")
def m_walk(run: Run, **kw):
    """The paper's Domeline walk (small space only; it does not use the verifier)."""
    from harness.search import domeline_walk
    if B.is_large(run.space):
        raise ValueError("walk runs on the small space; use team for the large space")
    trail = domeline_walk(run.ev, **kw)
    for s in trail.steps:
        run.record.accept({"changes": s.move, "why": s.reading, "agent": "walk"}, None, s.eval)
    run.record.stop = trail.stop


@method("team")
def m_team(run: Run, agents=None, **kw):
    """C's team loop (harness/team.py) with rule agents (deterministic mode) unless agents= names a factory."""
    from harness.team import rule_agents, team_search
    if agents is None:
        ag = rule_agents(run.space)
    else:
        ag = resolve(agents)(run)
    excl = {k: set(v) for k, v in (run.cfg.get("exclude") or {}).items()}
    r = team_search(run.ev, run.toolbox, ag, record=run.record, budget=run.ev.budget, exclude=excl or None, **kw)
    run.extra.update(proposals=r.proposals, rejected=r.rejected, restructures=r.restructures)
    run.gate.illegal = r.rejected.get("illegal", 0)
    run.gate.proposals = r.proposals


@method("random")
def m_random(run: Run):
    B.random_search(run.gate, run.table, seed=run.seed)


@method("tpe")
def m_tpe(run: Run, **kw):
    B.tpe_search(run.gate, run.table, seed=run.seed, **kw)


@method("sa")
def m_sa(run: Run, **kw):
    B.anneal(run.gate, run.table, B.start_point(run.space, run.table), seed=run.seed, **kw)


@method("hill")
def m_hill(run: Run, **kw):
    B.hill_climb(run.gate, run.table, B.start_point(run.space, run.table), seed=run.seed, **kw)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path=None, overrides=()):
    cfg = {**DEFAULTS, **(yaml.safe_load(Path(path).read_text()) if path else {})}
    for kv in overrides:
        k, _, v = kv.partition("=")
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(v)
    cfg.setdefault("out", str(ROOT / "results/h"))
    return cfg


SUMMARY_COLS = ["method", "seed", "evals", "budget", "best_speedup", "best_eval", "first_1pct_eval", "pick_speedup",
                "pick_area_mm2", "pick_effort_months", "gap_certificate", "gap_dp", "proposals", "illegal",
                "wall_s", "eval_wall_s", "tokens_in", "tokens_out", "stop"]


def _job(a):
    cfg, mspec, seed = a
    return run_one(cfg, mspec, seed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--seeds")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--methods", help="comma list of method names (replaces the config's)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    if args.budget:
        cfg["budget"] = args.budget
    if args.methods:
        cfg["methods"] = args.methods.split(",")
    if args.out:
        cfg["out"] = args.out
    for mod in cfg.get("plugins") or []:
        importlib.import_module(mod)
    for m in cfg["methods"]:
        resolve(m if isinstance(m, str) else m["name"])        # fail early on an unknown name
    jobs = [(cfg, m, s) for m in cfg["methods"] for s in cfg["seeds"]]
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            rows = pool.map(_job, jobs, chunksize=1)
    else:
        rows = [_job(j) for j in jobs]
    out = Path(cfg["out"]) / cfg["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    ckpts = [f"best@{c}" for c in CHECKPOINTS if any(f"best@{c}" in r for r in rows)]
    cols = SUMMARY_COLS + ckpts + (["shadow_illegal_evaluated", "shadow_accepted_illegal", "shadow_final_legal",
                                    "shadow_rules"] if any("shadow" in r for r in rows) else [])
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            sh = r.get("shadow") or {}
            r = {**r, "shadow_illegal_evaluated": sh.get("illegal_evaluated"),
                 "shadow_accepted_illegal": len(sh.get("accepted_illegal", [])) if sh else None,
                 "shadow_final_legal": sh.get("final_legal"), "shadow_rules": json.dumps(sh.get("rules")) if sh else None}
            w.writerow([_fmt(r.get(c)) for c in cols])
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    for r in rows:
        print(f"{r['method']:>14} s{r['seed']}  best {r['best_speedup'] or 0:.3f}x  evals {r['evals']}  "
              f"1% band at {r['first_1pct_eval']}  illegal {r['illegal']}  "
              f"gap cert {100 * (r['gap_certificate'] or 0):.1f}%  {r['wall_s']:.1f}s"
              + (f"  shadow {r['shadow']['illegal_evaluated']} illegal evals, accepted illegal "
                 f"{len(r['shadow']['accepted_illegal'])}, final legal {r['shadow']['final_legal']}"
                 if "shadow" in r else ""))
    print("wrote", out / "summary.csv")
    return rows


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return "" if v is None else v


if __name__ == "__main__":
    main()
