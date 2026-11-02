# REPORT M3: large design space (Agent C)

Branch `h/m3-large-space` (worktree `.worktrees/m3-large-space`), merged with autopilot 237a240 (M0).
Numbers: `results/large/m3_check.json`, from `experiments/large_space_check.py` (45 search sessions,
mini-coder-4b, lib/tech). Tests: `tests/test_large_space.py`.

## 0. Plan (written after orienting, before code)

What I found:
- mini-coder-4b has 36 layers, all with a KV cache, qk_norm on (Qwen3). One decode step is 18 ops
  per layer + 4 global ops = 652 ops. One small-space Eval on the 45 search sessions takes ~0.66 s
  (104 integrator steps x 652 ops); the walk reaches 9.42x at eval 20, the sweep optimum is 9.48x.
- `Design.mapping` already resolves per-op names first, so per-layer placement needs no integrator change.
  Precision does: `kv_bits` is read once per design in `Integrator`. A per-layer KV precision needs a
  per-op lookup (plumbing, not an equation change; flagged below).
- The workload's LLM time is *linear* in the step totals the replay samples: `WorkloadReplayer` sums
  `Curve.sum_tokens` (linear in the curve's values) and `prefill_tokens * pT.at(ctx)` (linear). So
  LLM time = sum_j w_j * step_j(design) over the 104 sampled steps j, with weights w_j that depend
  only on the sessions. A step is per-op costs plus boundary costs (launch, transfer), and per-op
  costs do not depend on the layer index, nor on the mapping of other ops.

Plan: (1) `LargeSpace` with per-layer slots; (2) a tied == small regression test; (3) a fast
evaluator that exploits the linearity, tested equal to `Evaluator`; (4) coupling check and example;
(5) DP over the op chain (tested against brute force) and a certificate; (6) one experiment script.
Skipped (as allowed): the session-varying policy.

## 1. Summary

| Item | Result |
|---|---|
| Encoding | 616 slots (36 layers x 17 + 4 global) + 10 hardware knobs = 626 knobs, ~10^248 points |
| Regression | tied large point == small point, **identical Eval** (exact path, all readings, `==`), on START, the walk's result, the sweep optimum, two past-the-wall designs and random points; fast path within 8e-15 |
| M0 | `tests/test_m0_repro.py` passes (575,400 / 280,296 / 9.48x / walk 0.7% at eval 20) |
| Speed | exact `Evaluator` on the large space: **0.71 s/eval**. `FastEvaluator`: **4.3 ms/eval** for mapping moves, **6.0 ms** with random hardware every eval, 10 ms with cold tables. Max rel. diff vs exact: 2.2e-15 (8 random untied points, 2 past a wall). Same numbers, not an approximation beyond the replay's own sampled curves. |
| Coupling | modeled: a lone offload adds one transfer in plus one per consumer of its output, and a launch + sync. 4 sign-flip examples found (below); effects are small |
| DP | exact mapping optimum with hardware fixed: 11 s per hardware setting on the 36-layer model (4,860 states max, 1.5M transitions); equals brute force on a 2-layer instance (2 hardware settings, 15,552 mappings each) |
| Certificate | floor 349.3 s (10.83x vs the 3080); Amdahl bound 227.0 s (16.66x). Sweep optimum 9.48x is **14.2%** above the floor |

## 2. What was built (files touched)

- `harness/large_space.py` (new):
  - `LargeSpace(Space)`: slots `L<l>.<op>` (placement of each op, options from its small-space
    role: mvm, softmax, vector), `L<l>.attention` (q.Kᵀ, p.V and the KV append of a layer together;
    the KV must live where attention reads it), `L<l>.kv_bits`, and global `lm_head`, `final_norm`,
    `embed`, `sample`. Points become `Design.mapping` per-op-name keys. `tie(small)`, `untie`,
    `is_tied`. Same interface as `Space` (canonical, key, to_design, area_mm2, effort_months), so
    the existing `Evaluator` and readings work unchanged. `fixed_hardware(p, attn_vfu)` gives the
    hardware with knobs as given (for the DP; see the bug note in §6).
  - `replay_weights`, `Tables`, `Chain`, `FastEvaluator(Evaluator)`: the linear replay. Per-op costs
    come from `Integrator._op_time` once per (op signature, tech setting, KV bits), over the 104
    sampled steps. `Chain.advance` is a line-for-line copy of `Integrator.step`'s loop body
    (launch on change of side, transfers, NoC, crossbar groups adding their max) run on weighted
    scalars. Capacity, reprogramming past the wall, KV spill on digital techs and the wall
    bisection are reproduced too, so the fast path never falls back to the exact one.
    **This duplicates the integrator's boundary logic**: if someone changes `Integrator.step`,
    `test_tied_equals_small` and `test_fast_matches_exact_untied` will fail until `Chain` follows.
  - `mapping_dp`, `certificate` (§5).
- `harness/design.py` (additive): `Design.kv_bits_by_layer` (default empty) and `kv_bits_for(op)`.
- `harness/integrator.py` (**plumbing, flagged**): `_resident` and `_op_time` read
  `design.kv_bits_for(op)` instead of `design.kv_bits`. No equation changes; with an empty
  per-layer map it returns `kv_bits`, so every existing result is bit-identical (M0 test passes).
- `harness/space.py` (refactor, no behavior change): `xbar_lib` split into `xbar_lib` +
  `_xbar_lib_canonical`, so `LargeSpace` can pass a small proxy point without dispatching to its
  own `canonical`.
- `tests/test_large_space.py` (new, 8 tests, ~30 s), `experiments/large_space_check.py` (new),
  `results/large/m3_check.json`.

Weight precision is still global (`weight_bits` = 8, as in the paper's space). Precision per layer
is KV only, {16, 8}, as the small space has it. Per-layer weight bits would be the same hook
(`weight_bits_for`), but no tech in the library models accuracy or an int4 path, so I did not add one.
Question 1 below.

## 3. Coupling

The integrator tracks where each activation tensor lives (`where`, `copies`) and charges a
transfer (t0 + bytes/BW) for each op whose inputs were produced on the other side, plus a launch
or sync for each change of side. So a lone op offloaded between host ops costs **one transfer in
plus one transfer back per op that reads its output**, not always two: `L3.act` on the VFU costs 2
(its output goes only to down_proj), `L3.rope` costs 3 (q goes to attn_score and k to kv_append).
Plus one launch + one sync. Tested in `test_lone_offload_crosses_twice`.

Example (layer 17, one layer moved; deltas in workload seconds; START = weight MVMs on crossbars,
the rest on the host, 324 crossings per decode step):

| Base | Op alone | Its neighbors without it | Op + neighbors |
|---|---|---|---|
| START: `attn_residual` -> STAR, neighbor `mlp_norm` | **+0.347 s** (hurts) | -0.736 s | **-1.324 s** (op's marginal -0.588 s) |
| all on host: `qk_norm` -> STAR, neighbor `rope` | **+0.192 s** (2 crossings) | -15.012 s | **-15.944 s** (marginal -0.932 s) |

Mechanism (first row): `attn_residual` on STAR reads x from the host and o_proj's output from the
crossbar, then mlp_norm on the host has to pull x back: one more crossing (325 vs 324 per step).
With mlp_norm on STAR too, x stays on the coprocessor and h2 goes to the crossbar over the NoC
(323 per step). Applied in all 36 layers, the first row is +12.5 s alone and -47.7 s together
(93,846 s base).

**This weakens a paper claim, so I'm flagging it:** the coupling exists, but its size is about
1e-5 of the workload time per layer. A crossing costs microseconds (CXL: 1 µs launch, 1 µs sync,
0.5 µs t0; a decode activation is 5 KB, 0.08 µs at 64 GB/s), against tens of µs per op. That is
far below the walk's ε = 0.5%. So link coupling is not what will stall single-knob search in this
model. What does stall it is granularity: one layer's move is 1/36 of a tied move, and almost
every per-layer move gains less than 0.5% (§4).

## 4. Does the large space hold anything the small one does not?

Exact DP optimum per hardware setting, against the best tied mapping on the same hardware
(the tied best is an enumeration of all 432 tied mappings x KV bits; LLM-side seconds):

| Hardware | DP mapping | Best tied | Untying gains | DP design |
|---|---|---|---|---|
| START's | 1184.55 s (2.68x) | 1185.05 s | 0.04% | everything on xbar/STAR, embed on VFU, sample on host |
| walk result's | 174.20 s (9.43x) | 174.68 s | 0.28% | same; gain = embed on VFU while sample stays on host |
| sweep optimum's | 171.78 s (9.485x) | 171.78 s | 0 | the tied optimum |
| optimum + 4 GB VFU KV, 1 TB/s | 171.78 s | 171.78 s | 0 | attention all on xbar; VFU KV not used |

**Finding: with no area or effort cap, the large space's optimum is (almost) the tied one.** All 36
layers cost the same, and the coupling between neighbors is microseconds, so the best per-layer
choice is the same in every layer. The only gain comes from splitting two ops the small space
ties (embed and sample in one `edge` knob). This matters for E5. An uncapped latency search on the
large space will not show "simple search stalls far from the optimum": the tied optimum is 9.48x and
the walk already finds it. Where per-layer choices do trade off is under a cap. Under an area cap,
attention on the crossbar for some layers only (each layer's KV arrays cost area) or KV8 for some
layers. Under a capacity limit, part of the KV on the VFU. The DP does not handle caps yet (Question 2).

Single-knob climb on the large space (best improvement over all 626 knobs, accept if > 0.5%
faster, from START): see §4a, filled in from the run below.

## 5. Ground truth without a sweep

**Exact mapping DP** (`mapping_dp(fe, hw_point, options)`): minimizes the workload's LLM time
over placements and KV bits with the hardware knobs fixed. It walks the 652-op chain with
state = (side, open crossbar group, live tensors -> (producer tech, copies), open slots
(attention, kv_bits) of the current layer, KV bytes on the VFU). Transition costs come from the
same `Chain.advance` the evaluator uses. VFU KV capacity is a constraint at the workload's
longest context. The VFU's bandwidth and memory knobs change its op costs only when some attention
runs there (`Space.to_design`), so the DP runs twice: once without VFU attention, once requiring
some. Test: equals brute force (rel 1e-12) on a 2-layer model, 11 free slots (15,552 mappings),
two hardware settings. Its result re-evaluates to the same time, both fast and exact.

DP caveat: the crossbar reprogramming past the wall. Arrays are sized to the need at `ref_ctx`
(65,536 = max over train and test). But the replay grid's top point is 1.05 x max ctx + 1 = 68,813, so
at `xbar_scale` = 1.0 with attention on the crossbar the last sampled step is past the wall and
pays a reprogramming cost. That cost depends on the whole mapping (an overflow count under a
ceil), so the DP leaves it out. Effect on the walk's hardware: DP 174.168 s vs evaluated 174.200 s
(0.018%). At `xbar_scale` >= 1.25 it is zero. The existing evaluator already charges this cost to
every small-space design at scale 1.0 with crossbar attention, including the walk's result. It is
part of the M0 numbers, so I left it alone; noting it in case it matters.

**Certificate** (`certificate(fe)`): floor = tools (the Amdahl bound) + sum over slots of the
slot's cheapest weighted cost, over its options, the KV bits and every setting of the cost
knobs (xbar tech x ADCs x cell bits, VFU units x bandwidth, STAR units). Launch, transfer, NoC
and reprogramming are bounded below by 0. Crossbar groups are bounded by their largest member.
It is valid for any design in the large space, hardware included; it takes 0.13 s.

| | time | gap to floor (349.31 s) | LLM-side gap (floor 122.29 s) |
|---|---|---|---|
| START | 93,846 s | 268x | — |
| walk result (9.42x) | 401.70 s | 15.0% | 42.8% |
| sweep optimum (9.48x) | 398.80 s | 14.2% | 40.5% |
| DP on optimum hardware | 398.80 s | 14.2% | 40.5% |

Floor parts (LLM s): attention 71.6, weight MVMs 35.3, softmax 13.8, vector 1.26, lm_head 0.23,
edge 0.10. The floor is loose because it takes each op's best hardware separately (e.g. the
fastest crossbar read settings and the largest STAR count together, ignoring area) and drops all
link costs. A tighter floor is the DP's minimum over hardware settings (720 settings x 11 s, about
2.2 h on one core, parallel by setting). I did not run it.

## 6. Bugs found on the way (mine, fixed)

- The DP first canonicalized the hardware point, which resets knobs that the *base* mapping leaves
  inactive (e.g. `star_units` -> 1 when the base uses no STAR). Brute force caught it.
- The DP then built its hardware from a "use every tech" mapping and hit the same reset
  (VFU overwrote every STAR slot). Fixed with `LargeSpace.fixed_hardware`, which takes knobs as given.
  Lesson for M4/M5: **`canonical` is only safe on the point being evaluated**. A tool that varies the
  mapping with "hardware fixed" must not canonicalize the hardware under a different mapping.

## 7. Tests

`/localhome/tsousape/tools/venv/bin/python -m pytest tests -q`: 44 passed, 3 failed. The 3 failures
are `test_claims.py` (registry vs `main.tex`: 6 mismatched numbers, unregistered phrases). They fail
identically on a clean autopilot checkout, since the paper text moved ahead of `paper/claims.yaml`.
They are not caused by this branch. `test_m0_repro.py` (4) and `test_large_space.py` (8) pass.

## 8. Questions for the lead

1. Per-layer **weight** precision: add it (e.g. {8, 4}) through the same hook? No accuracy model
   exists, which is why int4 KV was left out of the small space. I would keep it out.
2. **Caps in the DP.** Since the uncapped optimum is tied (§4), the large space's interest is under an
   area/effort cap. Area is nearly separable (crossbar arrays per op, x `xbar_scale`, under one ceil).
   A Lagrangian DP (minimize time + λ·area, sweep λ) gives exact points on the convex hull of the
   time–area front cheaply; an exact capped optimum needs an area dimension in the DP state. Want either?
3. The coupling claim in the outline ("two link transfers ... is what should stall single-knob
   search") does not hold at this model's link costs. The stall mechanism here is per-layer
   granularity vs ε. Should E5 run with a per-layer ε (e.g. ε/36)? Or should the claim be reworded?
4. `Chain` duplicates `Integrator.step`'s loop. The alternative is to refactor the integrator so both
   call one function. That edits integrator.py beyond plumbing, so I did not. The tests keep the two
   in sync.
