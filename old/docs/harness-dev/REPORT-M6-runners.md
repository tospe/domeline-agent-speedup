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

## 7. Round 2: H5 fairness rows, pre-filter, DP file format (after merging autopilot c443d44)

### 7.1 The three rows (`experiments/h/h5_fair.yaml`, `results/h/h5_fair/`)

Same space, sessions, budgets (3,000; TPE 1,000) and verifier (on, sampling=host) as h5.yaml, seeds 0–2.

| row | method | best (3 seeds) | 1% band of 9.4684× first at | best@100 / @300 / @1000 | pick area (mm²) |
|---|---|---|---|---|---|
| ref | team (det, h5) | 9.457 ×3 | 274 | 5.43 / – / 9.457 | 9,729 |
| (a) | SA from the walk's design | 9.416 ×3 (= start) | 1 (the start) | never improves | 9,729 |
| (a) | hill from the walk's design | **9.468 ×3** | 1 (the start) | 9.416 / 9.416 / 9.467–9.468 | **5,833** |
| (b) | SA, tied + per-slot moves | 9.468 / 9.468 / 9.457 | 1,274 / 1,693 / 1,892 | 0.7–7.3 / 2.7–7.3 / 2.8–9.0 | 8,749–11,665 |
| (b) | hill, tied + per-slot moves | 9.468 / 9.457 / 9.468 | 1,434 / 2,926 / 1,593 | 0.04 / 0.05–0.06 / 0.12–8.47 | 11,270–18,612 |
| (b) | **TPE over the tied subspace, then per-slot hill** | **9.468 ×3** | **83 / 56 / 88** | 9.468 / 9.468 / 9.468 | **5,833** |
| (c) | hill, tol 0, from START | 6.78 / 5.30 / 5.33 | – | 0.04–0.05 / 0.05 / 0.22–0.33 | |

**The comparison changes, and this matters for the paper.** Once a baseline gets the tied (small-space) moves, the large
space stops favouring the team.
- **TPE on the tied subspace** reaches 9.468× in 56–88 evaluations. That is 3–5× fewer evaluations than the
  deterministic team needs for 9.457× (274).
- **The hill climb from the walk's design** reaches 9.468× too (by eval ~1,000; it stops at 1,391–1,671).
- **The 9.468× / 5,833 mm² design that both pick is not a per-layer design.** I checked: it is tied, and it is a
  small-space design, the walk's design plus `link: die` and `xbar_tile: lean`. It costs **18 months against the walk's
  7**. H4's SA seed 0 found the same point in the small space. docs/results.md already lists it as the area front's
  "5,833 mm², 9.47×, lean tile" row, and its point 4 says the walk's shrink drops the die link deliberately. So this
  is the known area/effort trade, not a miss by the walk. These baselines pick it because `ev.best()` breaks ties
  within 1% by area before effort. Under an effort cap of 7 months it would be infeasible.
- SA from the walk's design never improves on it. At T0 = 5% it drifts away and does not come back within the budget.
- With tied moves, SA and hill get there too, slowly (1.3k–2.9k evaluations).
- `tol: 0` explains most of the plain hill climb's failure in h5: 0.8× becomes 5.3–6.8×. The 0.5% acceptance
  threshold blocks the per-layer gains (each worth ~1/36 of a full move). It still never reaches the band.

What I take from this, for you to judge:
- The uncapped large space does **not** show "simple search stalls" once the baseline has coordinated (tied) moves.
  This agrees with M3: untying gains 0–0.28% uncapped. The team's advantage in h5 came from coordinated moves, not
  from the readings.
- The comparison that can still separate the methods is the **capped** one (H5 capped, C's DP). The h5_fair rows run
  under caps unchanged: `--set caps.area_mm2=5000 --set experiment=h5_fair_a5000 --set reference.dp_file=...`.
- "Best" for every row is 9.4683×, the same as the 14-knob space's legal optimum (9.4684×). No row found a per-layer
  design faster than the tied optimum.

Note on the 1% band: the walk's design is already in the band (9.416 ≥ 0.99 × 9.4684), so row (a) is in it from
eval 1 by construction.

Implementation:
- `sa`/`hill` take `start:` (`walk`, or a small or large point; a small point is tied) and `tied: true` (adds
  `tied_neighbors`: one move per (role, value) that sets every slot of a role, i.e. a small-space move lifted).
  SA picks a tied move with p = 0.5, because there are ~15 tied moves against ~1,300 single-slot moves.
- `tied_tpe`: TPE has no notion of a move, so it runs in two phases. Phase 1 is TPE over the tied subspace (hardware
  + 7 role knobs) for half the budget; phase 2 is a hill climb with single-slot and tied moves from TPE's best.
- `tol` was already a hill-climb keyword.
- Tests (`test_start_walk_and_tied_moves`): the tied start equals `tie(walk)` and times at 9.4163×; tied moves keep a
  tied point tied; one tied group per role, 36 slots each for attention and KV bits.

### 7.2 Pre-filter (`baselines.static_violations`, `make_verifier(..., prefilter=True)`)

The checks that depend only on the placement:
- R1.support for every op;
- R1.sampling for `sample`;
- R2.transfer for the token in and out (sampling=host).

They use the same tests as `verify._r1` / `_r2_static`, with a dict lookup per op and no design or integrator. If any
fires, the point is illegal and the full verify is skipped. The legal/illegal verdict is the full verifier's; the
violation list is the static subset, flagged `"prefilter": True`. Shadow mode always runs the full verify, so H6 rule
counts are complete.
- **Identical verdicts, tested:** `test_prefilter_gives_identical_verdicts`, sampling host and any; 480 points per run,
  uniform and walk-local, small and large space. Every pre-filter hit is full-verify illegal, its (rule, op) pairs are
  a subset of the full ones, and the gated verdict equals the full one on every point.
- **End to end:** random seed 0 on h5 with the pre-filter gives an **identical 3,000-evaluation sequence**
  (points and times) and the same 14,984 illegal proposals. 13,478 of 17,985 verifies were pre-filtered, and wall time
  fell from **1,442 s to 325 s** (`results/h/h5_prefilter_check/`).
- It is on by default (`prefilter: true`, verifier: on only).

**Bug found and fixed in passing: YAML reads `verifier: on` as `True` and `off` as `False`.** As a result, the
pre-filter never switched on: the h5_fair runs above used the full verify, which gives the same verdicts, so their
results stand. `verifier: off` would also have left the verifier on, since `False != "off"`. No committed config used
`off` (H6 uses `shadow`), so no committed result is affected. `hrun.normalize` now maps the booleans back to modes,
and a test checks that `off` really evaluates without verifying and that `on` enables the pre-filter.

### 7.3 dp_caps format (from C's committed `experiments/dp_caps.py` in .worktrees/m6-dp, 98bd8a6)

`{"caps": [{"area_cap", "effort_cap", "bound_s", "best_time_s", "gap", "speedup", "point", ...}], "dp_floor_s", ...}`.
`hrun.dp_row(cfg)` picks the row whose caps equal the run's `caps:`, where an absent cap means inf. `gap_dp` is measured
against `best_time_s` (the best design C found) and the new `gap_dp_bound` against `bound_s` (C's lower bound).
`test_dp_row_reads_c_format` covers it. Only smoke files exist so far; once `dp_caps_main.json` lands, set
`reference.dp_file: results/large/dp_caps_main.json`.

### 7.4 Files and tests (round 2)
- Changed: `harness/baselines.py` (tied moves, `start`, `tied_tpe`, pre-filter), `experiments/hrun.py` (method options,
  `tied_tpe`, the pre-filter wiring, the dp_caps reader, `normalize`), `tests/test_baselines.py` (+5 tests).
- New: `experiments/h/h5_fair.yaml`, `results/h/h5_fair/`, `results/h/h5_prefilter_check/`.
- `tests/test_baselines.py`: 21 passed. Full suite: 117 passed, 3 failed (only the pre-existing `test_claims` ones).

### 7.5 Questions
1. Given 7.1, should H5's headline be the capped comparison, with the uncapped rows kept as "the large space adds
   nothing without caps"? That is consistent with your E5/H5 note.
2. The H4/H5 tables compare picks under `ev.best()` (area first, then effort). The baselines' 5,833 mm² picks cost 18
   months and the walk's cost 7. Should the runner's summary also report the pick under an effort-first tie-break, or
   should every H run carry an effort cap? Otherwise a reader may take "smaller pick" for "better design".
