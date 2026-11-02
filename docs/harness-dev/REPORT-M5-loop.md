# REPORT M5 (loop, deterministic mode) — Agent C

Branch `h/m5-loop` (worktree `.worktrees/m5-loop`, base autopilot dc07f9d).
Files: `harness/team.py` (new), `tests/test_team.py` (new, 7 tests), `experiments/team_det.py` (new),
`docs/harness-dev/AGENT_IFACE.md` (new, the loop <-> agent contract for Agent A),
`results/team/det_main.json` + one RunRecord per run (`results/team/det_main_<run>.json`).
No existing file was changed.

## Plan (as executed)

1. Write the agent interface first (Agent A builds against it), then the loop, then the rule agents.
2. Make deterministic mode reproduce the walk by construction: the rule agents return the walk's own
   rule table (`search.proposals`), filtered by term and owner, each move tagged with its index in that
   table. The supervisor merges the owners' candidates by that index, so the evaluation order is the walk's.
3. Lift the rule table to the large space (one move per slot), add the supervisor's coordinated moves,
   and run on `LargeSpace` with `FastEvaluator`.

## What works

`team_search(ev, toolbox, agents, *, start, eps=0.005, stall_k, stall_gap=0.02, stop_frac=0.02, budget,
record, retry_budget=2, coordinate="stall"|"first"|"never", window=1, exclude, shrink=True, ...)`:

1. **Read and route.** The supervisor calls `toolbox.read`. It builds the agenda: the wall if hit, the
   area if over the cap, then the terms by size. For each agenda item it asks that term's owners, in
   `TERM_OWNERS` order; the first owner is the one the binding term routes to.
2. **Propose.** Each agent gets an `AgentContext` with its own knobs only, the rule table's moves for
   this term and its knobs, the reading, and the recent history. It returns `[Move]`.
3. **Ownership.** A move touching another agent's knob is rejected without evaluation and logged as
   "ownership" (tested with a rogue mapping agent that touches `link`).
4. **Verify.** When a verifier is attached (`point_verifier(space, sessions, caps)` wraps M2's
   `verify_point`), illegal moves cost no evaluation. The violated rules go back to the agent as
   `ctx.feedback`, and it may retry up to `retry_budget` times.
5. **Evaluate and accept.** A move is accepted if it is more than eps faster. A faster move that is over
   the area cap is repaired first (the walk's `_repair_area`). Every proposal and its outcome
   (accepted / slower / infeasible / illegal / ownership / excluded / duplicate) goes to
   `toolbox.log`. Accepts go to `RunRecord.accept` with the move, the reading behind it and the
   verifier result.
6. **Stall and restructure.** A stall is `stall_k` proposals without an accept while the Amdahl gap
   exceeds `stall_gap`. The supervisor then inserts coordinated moves: **tie** applies one rule's
   per-slot moves in every slot it names, and **window** moves a slot with its chain neighbors. It also
   tries them once at the end of a fruitless round before stopping. `coordinate="first"` tries them
   before the single moves. On the small space there are none, so the walk is unchanged.
7. **Stop** when the LLM time is within `stop_frac` of the tools time (the walk's Amdahl stop), when the
   budget runs out, or when a round yields nothing. A shrink phase follows (the walk's `_shrink`).
   Non-rule agents get further rounds (up to `max_rounds`) while they still propose.

Tables (explicit and tested; also in AGENT_IFACE.md):

- `OWNER_KNOBS`: mapping = placements (mvm, lm_head, attention, softmax, vector, edge, and every
  placement slot); hardware = xbar_tech, vfu_units, vfu_bw, star_units, link, xbar_tile, adc_per_ima,
  cell_bits; capacity = xbar_scale, vfu_mem, kv_bits (and every `L<l>.kv_bits`).
- `TERM_OWNERS`: capacity -> (capacity, mapping); area -> (hardware, capacity, mapping); host ->
  (mapping); launch, transfer -> (mapping, hardware); write -> (hardware, mapping, capacity); vfu ->
  (hardware, mapping); star -> (hardware); xbar -> (hardware, mapping); other unit -> (capacity, hardware).
- `test_term_owner_table_covers_the_rule_table`: every move the rule table emits for a term, on every
  design the walk evaluates (small and tied-large), touches only knobs of that term's owners. Every
  knob has exactly one owner.

`exclude` (knob or slot -> forbidden values, e.g. `{"edge": {"vfu"}}` or `{"sample": {"vfu"}}`) keeps
the sampling decision open, as you asked. It is tested.

## Numbers (45 search sessions; `results/team/det_main.json`)

| Run | Speedup | Evals | 1% band of 9.48x first at | Notes |
|---|---|---|---|---|
| walk (search.domeline_walk) | 9.416x | 31 | 20 | reference |
| team, small space | **9.416x** | **31** | **20** | identical evaluation sequence, moves and design (test) |
| team, small, verifier on | 9.416x | 24 | 15 | 7 illegal `edge=vfu` moves rejected unevaluated (R1.sampling); same design |
| team, large, no coordination | 4.71x | 3000 (budget) | — | 147 per-slot accepts, never stalls long enough to stop |
| team, large, restructure on stall, k=20 | 9.416x | 558 | 332 | 11 restructures, 8 coordinated accepts; ends on the walk's design |
| team, large, restructure on stall, k=100 | 7.88x | 3000 (budget) | — | 2 restructures |
| team, large, coordinated first | 9.416x | 506 | 280 | 8 coordinated accepts |
| same + verifier | 9.416x | 497 | 274 | 9 illegal moves rejected |

The large-space runs take 2–22 s of wall clock at 3000 evals (FastEvaluator); the verifier adds ~10 ms per proposal.

Readings:
- **Deterministic mode reproduces the walk** on the small space: the same 31 evaluations in the same
  order, the same 14 accepted moves at the same eval numbers, the same final design
  (`test_deterministic_mode_reproduces_the_walk`).
- **On the large space the team ends on the tied (walk) design**, which M3 showed is the uncapped
  optimum at that hardware. It needs **~16–18x more evaluations** than the small walk (506–558 vs 31).
  Without coordinated moves it climbs one layer at a time and is at 4.71x when the budget runs out. So
  the restructure step is what makes the large space searchable at all, and the trigger matters:
  k=20 works, k=100 is too late at 3000 evals. This is the M3 §4a finding again: per-layer moves
  clear eps one at a time, so the agents rarely "stall" in the brief's sense. Coordination has to come
  from the supervisor, not from a stall detector tuned to stop.
- None of this beats the small walk. Under no cap, the large space only offers the same design
  at a higher search cost. As M3 said, E5 needs caps for the large space to matter.

## Decisions I made (flag if you disagree)

1. **Routing asks all owners of every agenda term, binding term first,** not only the binding term's
   owner. The walk tries every term's moves before giving up, so routing only the binding term would
   not reproduce it. "The binding term decides which agent acts next" holds: its owner acts first.
2. **The restructure on the large space is deterministic** (tie / window, built from the agents'
   per-slot moves). "Reassign the agents" is not implemented. With fixed rule agents it means
   nothing; Agent A's LLM supervisor can plug in there.
3. **Stop** is the walk's (LLM time <= 2% of tools time). `stall_gap` uses `read()`'s `amdahl_gap`
   (time / tools - 1), as specified.
4. `point_verifier` caches by design key.

## Tests

- `tests/test_team.py`: 7 passed (~110 s; the walk reproduction itself takes ~45 s).
- Full suite: 79 passed, 5 failed. **None of the failures come from this branch**; all 5 fail
  identically on a clean autopilot dc07f9d checkout:
  - `test_claims.py` x3: `main.tex` vs `paper/claims.yaml`, known since M3.
  - `test_verify.py::test_attention_on_crossbar_without_write_mechanism[flag]`: kinds
    `{attn_score, attn_value}` vs the expected `{..., kv_append}`.
  - `[no_write_param]`: `TypeError` at `harness/tech.py:88` (`int * None`).

  Both `test_verify` failures look like the M1/M2 merge (M1 changed write params or the kv_append
  path; M2's tests assume the old one). They are for whoever owns that merge.
- `test_m0_repro.py` passes.

## Questions

1. Is it OK to ask every owner of every agenda term (decision 1)? The alternative is to route only the
   binding term and drop exact walk reproduction.
2. For E5/E7, should "no restructure" be `coordinate="never"`? That is the 4.71x run, so the ablation
   would show a large effect, by construction of the large space.
3. Agent A: `AGENT_IFACE.md` is in this worktree (`docs/harness-dev/AGENT_IFACE.md`, commit d03871c),
   so it is not on autopilot yet. If A wrote a different one, I'll adapt.
