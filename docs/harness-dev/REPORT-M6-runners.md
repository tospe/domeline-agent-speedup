# M6 report: baselines + runner scaffold (Agent B)

Branch `h/m6-runners` (worktree `.worktrees/m6-runners`, base autopilot 4ed392d).

## 0. Plan (after orienting, before code)

What exists: `search.py` has random, TPE, hill climb on the small space's `KNOBS` (hard-wired); `team.py` (C) has
`team_search` + `rule_agents` + `point_verifier`; `tools.py` has `Toolbox` and `RunRecord`; `large_space.py` has
`LargeSpace`, `FastEvaluator` (4–6 ms/eval), `certificate` (0.1 s). Measured: verify is 14–30 ms/point on the large space
(slower than an eval); optuna TPE over the 626 large-space knobs costs ~60 ms/trial at 150 trials and grows with history.

1. **`harness/baselines.py`**: random, TPE, simulated annealing, single-knob hill climb, written once over a *knob
   table* (`knob_table(space)`: the small space's KNOBS, optionally the 14-knob pinned version, or LargeSpace's
   hardware knobs + slots), so they run on Space and LargeSpace alike. Seeded. Budget = the Evaluator's (distinct
   designs; cached repeats free). A `Gate` sits in front of the evaluator: with the verifier on, an illegal proposal
   is logged (rules hit) and **not evaluated**, as in the team; with it off, everything is evaluated. Caps are the
   Evaluator's (infeasible), as in the team. A proposal cap (20× budget) guarantees termination when most proposals
   are illegal. Pick **SA over GA**: its neighbourhood is the hill climb's single-knob move, so SA vs hill climb
   isolates "escaping local minima"; a GA's crossover on 626 per-layer slots needs encoding choices we'd have to defend.
2. **Sampling rule stays configurable**: `verify(..., sampling="host"|"any")`, default "host" (rule (a)). "any" drops
   R1.sampling and turns the token in/out crossing into an implied cost instead of a violation. The runner also takes
   the team's `exclude` dict.
3. **`experiments/hrun.py`**: config YAML (+ CLI overrides) -> one RunRecord JSON per (method, seed) under
   `results/h/<exp>/`, and `summary.csv` (best speedup at eval checkpoints, eval of the first point in the 1% band of a
   reference, gap to the certificate, gap to the DP optimum when given, evals, illegal proposals, wall, tokens).
   A methods registry (`@method("name")`, or `module:function` in the config) so A's LLM methods and C's team plug in
   by name.
4. **Shadow verifier** (`VerifiedEvaluator` proxy): with the verifier *off*, every evaluated point is still verified on
   the side, so H6 can report illegal proposals per run, rules hit and which accepted moves / final designs were illegal.
5. Configs `experiments/h/h4.yaml` (small space: walk, team-det, baselines; LLM methods by name later) and
   `experiments/h/h6.yaml` (verifier off, shadowed; team-det now). Tests in `tests/test_baselines.py` (it also covers the runner).

## 1. Status: done. Baselines, runner, H4/H5/H6 configs and first runs are committed.

- **`harness/baselines.py`** has `random_search`, `tpe_search`, `anneal` (SA) and `hill_climb`, written once over
  `knob_table(space, pin, exclude)`, so they run on Space, the 14-knob space and LargeSpace (626 knobs).
  - Every proposal goes through a `Gate`: with the verifier on, an illegal proposal is logged with its rules and not
    evaluated, as in the team. The Gate stops after 20× budget proposals.
  - Caps are the Evaluator's. All four are seeded and use the Evaluator's budget, where each distinct design counts once.
  - New bests go into `RunRecord.accepted`. Illegal proposals go into the Toolbox log.
  - Also here: `make_verifier` (the team's dict format, with the sampling option) and `VerifiedEvaluator`, a shadow
    proxy that verifies every evaluated point without blocking it.
- **`harness/verify.py`** has a new keyword, `sampling="host"|"any"`. "host", the default, is rule (a). "any" (rule b)
  drops R1.sampling, and the token crossing becomes an uncharged implied cost instead of a violation. The change is
  additive and the default output is unchanged. In configs it is `sampling:`, alongside `exclude:`.
- **`experiments/hrun.py`** takes a YAML config plus `--set k=v` overrides and runs the (method, seed) jobs, in
  parallel with `--workers`.
  - **Records:** each job writes a RunRecord JSON to `results/h/<exp>/<label>_s<seed>.json[.gz]`. Large-space records
    are gzipped because they are about 17 KB per evaluation raw (~50 MB per 3,000-eval run). Each record also carries
    `metrics` and, in shadow mode, `metrics.shadow`.
  - **Summary:** `summary.csv` has one row per run: best speedup and the eval it came at, best@N checkpoints, eval of
    the first point in the 1% band of `reference.speedup`, the `ev.best()` pick (speedup, area, effort), gap to the
    certificate (`large_space.certificate`, 349.3 s, a floor for both spaces), gap to the DP optimum (`reference.dp_opt_s`, or
    C's `dp_file`, read leniently until its format is fixed), evals, proposals, illegal, wall, eval wall and tokens.
  - **Registry:** built-in methods are `walk`, `team`, `random`, `tpe`, `sa` and `hill`, registered with
    `@method("name")`. A's and C's methods plug in by importing a module listed under `plugins:` that registers names,
    by `module:function`, or with `{name: team, agents: <factory>}` to run the team loop with other agents. A method
    adds LLM tokens through `run.record.add_tokens`. That path is tested with a dummy plugin.
  - **Evaluator:** `evaluator: fast` (the default) uses FastEvaluator. On the small space it gives the identical walk
    (31 evals, 9.4163×) in 0.35 s instead of 21 s. `evaluator: exact` keeps the original.
- **Configs:** `experiments/h/h4.yaml` (small space: walk, team-det, 4 baselines; budget 100, seeds 0–2), `h5.yaml`
  (large space: team-det and 4 baselines; budget 3,000, TPE 1,000), `h6.yaml` (large space, verifier in shadow mode,
  team-det). For LLM methods, add the plugin module and the method name.

## 2. Numbers (45 search sessions, verifier on and sampling=host unless noted)

**H4, small space** (`results/h/h4/`, budget 100, 1% band of the legal optimum 9.4684×):

| method | best (3 seeds) | 1% band first at | pick area (mm²) |
|---|---|---|---|
| walk | 9.457 / 9.457 / 9.457 | 20 / 20 / 20 (31 evals, then stops) | 9,729 |
| team (det) | 9.457 ×3 | 15 ×3 (24 evals, 7 illegal not evaluated) | 9,729 |
| TPE | 9.468 ×3 | 99 / 50 / 47 | 18,030 / 11,270 / 11,270 |
| hill climb | 9.468 ×3 | 71 / 21 / 36 | 11,270 |
| SA | 9.468 / 8.51 / 7.49 | 42 / – / – | |
| random | 3.27 / 1.79 / 7.43 | – | |

Two things about this table:
- "Best" means the fastest feasible design evaluated. The walk's and team's shrunk pick is 9.416× at 9,729 mm².
- The scalar baselines reach the exact legal optimum but at 1.2–1.9× the area, because they never shrink. As in
  docs/results.md, TPE on the small space is strong.

**H5, large space, uncapped** (`results/h/h5/`, budget 3,000, TPE 1,000; certificate floor 10.83×):

| method | best (3 seeds) | 1% band | best@1000 | wall per run |
|---|---|---|---|---|
| team (det, coordinate first) | 9.457 ×3 | 274 | 9.457 | 25–31 s |
| hill climb | 0.77 / 0.77 / 0.82 | – | 0.10–0.12 | ~170 s |
| SA | 0.58 / 0.51 / 0.78 | – | 0.06–0.07 | ~185 s |
| TPE (1,000) | 0.17 / 0.18 / 0.16 | – | 0.16–0.18 | 570–700 s |
| random | 0.10 / 0.11 / 0.12 | – | 0.09 | ~1,400 s |

**H6, verifier off, shadow-verified** (`results/h/h6/`, large space, team-det):

| run | evals | illegal designs evaluated | rules | accepted illegal | final legal |
|---|---|---|---|---|---|
| team, coordinate first | 506 (vs 497 with the verifier on) | 46 | R3.capacity 44, R2.transfer 2, R1.sampling 1 | 0 | yes |
| team, restructure on stall (k=20) | 558 | 46 | same | 0 | yes |

For the deterministic team, the verifier saves budget but changes no decision: the 46 illegal designs were all
rejected as slower or infeasible anyway. The test on the small space finds the same (9 of 31 illegal, 0 accepted).

## 3. Findings and caveats (for the lead)

1. **Correction to REPORT-M5-loop.** It says the small-space team with the verifier rejects "7 illegal `edge=vfu` moves".
   The run log shows **6 R3.capacity and 1 R1.sampling+R2.transfer**. The counts (24 evals, 15 for the band) are
   right; only the attribution is wrong.
2. **The large-space baseline gap is large, but not yet a fair fight; please decide before the paper uses it.**
   SA and hill climb start from tied START (0.04×). There, a single-slot move changes one of 36 layers, so 3,000 evals
   give fewer than ~3 "full-model" moves. M3 found the same: the climb reached 3.38× after 100k evals. Rows that would
   make H5 defensible, none of them run yet. (c) is a config change. (a) needs a `start:` kwarg on the `sa`/`hill`
   methods, about 5 lines. (b) needs a tied knob table, a small addition to `knob_table`:
   - (a) SA and hill started from the walk's tied design (does local search improve on it?);
   - (b) SA, hill and TPE over the *tied* subspace plus per-slot moves (a coordinated-neighbourhood ablation);
   - (c) hill with `tol: 0`. Per-layer gains can be below the 0.5% acceptance threshold.

   Random and TPE are hopeless in 626 dimensions by construction. That is expected and worth one sentence, not a
   headline.
3. **Cost.**
   - The verifier is slower than an evaluation on the large space: 14–30 ms against 4–6 ms. Uniform random proposals are
     ~83% illegal, so random spends about 95% of its 1,400 s verifying. If that matters, a cheap pre-filter for the edge
     and sample slots, or `exclude:`, removes most of it.
   - TPE costs ~0.6 s per trial at 1,000 trials over 626 knobs, so its budget is capped at 1,000 in h5.yaml.
4. **SA rather than GA.** SA's neighbourhood is exactly the hill climb's single-knob move. SA against hill therefore
   measures only the value of accepting worse moves. A GA's crossover over per-layer slots would need encoding choices
   (layer-aligned or not, ties) that we would have to defend. Schedule: T falls from 0.05 to 5e-4 over the budget, on
   log time; the infeasible penalty is the same 10× GPU time as `search.tpe_search`.
5. **The walk never consults the verifier.** It is the paper's algorithm, and its `illegal` column is blank. The team
   and the baselines use the verifier as configured.

## 4. Files touched
- New: `harness/baselines.py`, `experiments/hrun.py`, `experiments/h/{h4,h5,h6}.yaml`, `tests/test_baselines.py`,
  `results/h/{h4,h5,h6}/` (records, summary.csv, config.json; 22 MB in total, h5 gzipped), this report.
- Changed (additive): `harness/verify.py`, which gains the `sampling` keyword. The default behaviour and output are
  unchanged.
- Not touched: team.py, large_space.py, tools.py, search.py and the model equations.

## 5. Tests
- `tests/test_baselines.py`: 16 passed (2 min 20 s). It covers:
  - the knob tables;
  - the sampling option;
  - each baseline × {small, large}: seeded, within budget, and never evaluating an illegal design;
  - verifier off evaluates illegal designs;
  - the proposal cap;
  - the shadow proxy;
  - the runner reproducing the walk (31 evals, band at 20, 9.4163×) and team-det (24 evals, band at 15, 7 illegal);
  - `module:function` methods with token accounting;
  - summary.csv columns;
  - shadow mode on the small team (9 illegal evaluated: 8 R3, 1 sampling; 0 accepted);
  - early failure on an unknown method.
- Rest of the suite (`pytest tests -q`): 82 passed, 3 failed, all in `tests/test_claims.py` (the same pre-existing
  main.tex/registry failures). `test_m0_repro.py` passes.

## 6. Questions
1. H5 fairness (§3.2): which of rows (a)/(b)/(c) should the paper's large-space comparison include?
2. C's `dp_caps_<tag>.json` format: I read `d[<cap key>]["time_s"]`. Tell me or C the real keys and I'll match them.
