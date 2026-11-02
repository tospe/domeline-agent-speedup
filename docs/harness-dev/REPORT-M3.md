# REPORT M3: large design space (Agent C)

## 0. Plan (written after orienting, before code)

What I found:
- mini-coder-4b has 36 layers, all with a KV cache, qk_norm on (Qwen3). One decode step is 15 ops
  per layer + 5 global ops = 545 ops. Today one small-space Eval on the 45-session search set takes
  ~0.66 s (104 integrator steps x 545 ops); the walk reaches 9.42x at eval 20, sweep optimum 9.48x.
- `Design.mapping` already resolves per-op names first, so per-layer placement needs no integrator change.
  Precision does: `kv_bits`/`weight_bits` are read once per design in `Integrator`. A per-layer KV
  precision needs a per-op lookup (plumbing, not an equation change; flagged below).
- The workload's LLM time is *linear* in the step totals the replay samples: `WorkloadReplayer` sums
  `Curve.sum_tokens` (linear in the curve values) and `t.prefill_tokens * pT.at(ctx)` (linear). So
  LLM time = sum_j w_j * step_j(design) over the ~104 sampled steps j, with weights w_j that depend
  only on the sessions. And a step is a sum of per-op costs plus boundary costs (launch, transfer).
  Per-op costs do not depend on the layer index, nor on the mapping of other ops (below the wall).

Plan:
1. `harness/large_space.py`: `LargeSpace(Space)` with slots per layer: 7 weight MVMs, `attention`
   (score/value/append tied, the KV must live where it is read), `softmax`, 7 vector ops
   (attn_norm, qk_norm, rope, attn_residual, mlp_norm, act, mlp_residual), `kv_bits`; global slots
   lm_head, final_norm, embed, sample; plus the 10 hardware knobs of the small space. Points map to
   `Design.mapping` per-op-name keys. `tie(small_point)` builds the tied large point.
2. Regression test: tied large point == small point, identical Eval (time, terms, by_kind, readings)
   on a sample incl. START and the walk's result; existing tests still pass.
3. Fast evaluator using the linearity above: per-(op signature, tech) weighted cost tables built once
   per hardware config from `Integrator._op_time`, then a replay of the integrator's own transfer /
   launch / group logic once per mapping, on weighted scalars. Falls back to the exact Evaluator when
   any sampled step is past the wall. Test: equals `Evaluator` (rel 1e-9) on random large points.
4. Coupling: check `StepResult.crossings` +2 for a lone offload; search single-op vs neighbor-group moves
   from START and report one concrete case.
5. DP: exact mapping optimum with hardware fixed, over the op chain with state = (side, pending
   crossbar group, live tensors -> (producer tech, copies), open tied slots, KV bytes on the VFU).
   Test vs brute force on a 2-layer toy model. Certificate: tools + sum over slots of the cheapest
   slot cost over every tech and every hardware setting; report gaps.
6. Experiment script `experiments/large_space_check.py` -> `results/large/m3_check.json`.

Skipped (as allowed): session-varying policy.

