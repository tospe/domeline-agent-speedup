# REPORT H9 (loop: supervisor hook, ensemble) — Agent C

Branch `h/h9-loop` (worktree `.worktrees/h9-loop`, base autopilot 72c93a4).

| commit | content |
|---|---|
| 77584e4 | the supervisor and ensemble signatures in `AGENT_IFACE.md`, written first so A can build against them |
| (next) | the code, tests, interface notes, this report |

Files:
- `harness/team.py`: `SupervisorContext`, `Supervisor`, `all_knobs`, the `team_search(supervisor=, max_escalations=)`
  hook, `ensemble`, `merge_records`;
- `harness/tools.py`: on `RunRecord`, `escalations`, `token_log`, `clock`, and `add_tokens(..., **extra)` for cache
  splits;
- `tests/test_team_h9.py` (6 tests, fake supervisors).

No model, evaluator or pricing code changed.

## Plan (as executed)

1. Write the interface first (AGENT_IFACE.md, H9 section), then commit.
2. Supervisor hook in `team_search`:
   - it replaces the deterministic restructure when set;
   - it is called on stall and before stopping;
   - it is capped, verified, and given feedback on illegal moves;
   - each escalation is logged with its own token delta.
3. Token timeline on `RunRecord`, for $ to the first 1% band.
4. Ensemble runner: parallel members merged into one record of the same schema (best of, costs summed, lockstep
   evaluation order).
5. Tests with fake supervisors; `supervisor=None` keeps the M5 loop.

## What works

**`team_search(..., supervisor=S, max_escalations=5)`.**

When it is called:
- "stall": `stall_k` proposals without an accept while the Amdahl gap exceeds `stall_gap`. The counter then resets,
  so the cost is at most proposals / stall_k calls.
- "exhausted": before the loop would stop for lack of moves.
- Never every round.

What it sees (`SupervisorContext`): every knob with its owner, the reading, the team's accepted moves and recent
history, this round's role moves, and the deterministic coordinated moves (tie / window) as suggestions.

How its moves are handled:
- They may span roles, so there is no ownership check.
- They are tried before the rest of the queue.
- `exclude` and the verifier apply. An illegal move costs no evaluation and is returned as `feedback`; up to
  `retry_budget` retries count as the same escalation.
- The accept rule is the agents' (eps, area repair).

What is recorded per escalation, in `TeamResult.escalations` and `RunRecord.escalations` (also in `to_json`):
- trigger, eval_n, binding term, time;
- moves, outcomes, accepted;
- `calls`, `wall_s`, and `tokens` = the per-model change in the record across the escalation. Extra keys such as
  `cache_read` are carried.

Once the cap is reached, the deterministic restructure comes back.

**Token timeline.** `RunRecord.add_tokens(model, n_in, n_out, **extra)` appends
`{"model", "in", "out", "eval_n", ...}` to `token_log`. `team_search` sets `record.clock = lambda: ev.used`. A's
`cost_usd` and "$ to first band" can be computed from it. Cache splits go in as extra keys and are summed into
`record.tokens[model]`.

**`ensemble(run_one, seeds, workers, executor="thread"|"process")`.**
- It runs members in parallel and returns a merged RunRecord-shaped dict:
  - the best feasible final is the result, and its index is `best_member`;
  - evaluations, tokens and `cost_usd` (if every member has one) are summed;
  - `wall_s` = max, `wall_s_sum` = sum;
  - `members` keeps the full member records.
- Evaluations are interleaved round-robin and renumbered, so "eval of first 1% band" counts what all members spent
  in lockstep. `token_log` entries are tagged by member and mapped to the merged count.
- `merge_records` is exposed for members that were run separately (e.g. by hrun).

## Tests

- `tests/test_team_h9.py`: 6 passed (~25 s).
  - On the large space with stall_k=20, a supervisor that returns the coordinated moves:
    - is called on stall;
    - has every escalation's tokens equal to calls × the fake's bill, and the record's total equal to their sum;
    - has an ordered token log and a supervisor accept, and ends above 9.0x.
  - Small space with the verifier: the supervisor's illegal `edge=vfu` costs no evaluation, comes back with
    R1.sampling as feedback, and its retry is accepted. The retry is a move spanning all three roles, and no
    ownership rejection happens.
  - The cap is respected by a supervisor with no ideas, and the run ends normally.
  - `supervisor=None` gives the same evaluations as not passing the argument. The M5 walk reproduction test in
    `test_team.py` still passes.
  - Ensemble: best of, round-robin renumbering, summed tokens and `cost_usd`, max wall, token_log mapping.
    `merge_records` skips infeasible finals.
- `test_team.py` + `test_llm_agents.py` + H9: 28 passed.
- Full suite: 133 passed, 3 failed (454 s), test_m0_repro included. The 3 failures are the known `test_claims.py` ones
  (paper vs claims registry); they fail on autopilot too.

## A reference for the matrix: rule agents at the H9 budget

Large space, uncapped, 50 evaluations, deterministic rule team (M5 agents), no LLM:

| restructure | best speedup at 50 evals |
|---|---|
| none (`coordinate="never"`) | 0.04x |
| deterministic, on stall, k=20 | 1.39x |
| deterministic, on stall, k=5 | **5.75x** |
| deterministic, first | 1.10x |
| supervisor hook, returning the coordinated moves, k=5 | 5.75x (7 escalations): the same as deterministic k=5 |
| same, k=10 | 3.64x |

The optimum is 9.47x (M6 DP). So at 50 evaluations the large space is far from solved by rule agents, and the
restructure trigger matters a lot. That leaves room for a good LLM supervisor to show value (R4/R6).

Two cautions for the matrix:
- **Use stall_k ≈ 5 for the supervisor rows.** k=20 hardly fires at 50 evals.
- **Report the rule-team row as a free baseline**, so "cheap team + escalation" is compared with "free rules +
  the same trigger".

## Decisions (flag if you disagree)

1. **The supervisor replaces the deterministic restructure** while it is under its cap; it does not add to it.
   It gets the coordinated moves in `ctx.coordinated`, so it can pass them through. That is how the fake
   reproduces the deterministic k=5 result.
2. **No call every round.** Only on stall and before stopping, to keep Sonnet calls rare, as R4 asks.
3. **Default `max_escalations=5`.** At budget 50 with stall_k=5 there are at most ~10 stalls.
4. **Ensemble members in lockstep** for the merged evaluation order (the parallel reading of "best of three").
   The other reading is sequential, and its costs sum the same way. Only "eval of first band" differs, and each
   member's record has its own.

## Questions

1. A: does `ctx.coordinated` belong in the Sonnet prompt? It can be long on the large space (one per rule term).
   Truncating it is your call; the loop passes all of them.
2. Should the ensemble's "$ to first band" be the lockstep one (merged token_log) or the cheapest member's? I'd
   report lockstep, since the three runs are paid for either way.
