# REPORT M1 — input schemas (Agent A, branch h/m1-schemas)

## Plan (written after orienting, before code)

Orientation facts: one evaluation on the 45-session search set takes ~0.8 s; the walk from
START takes 31 evaluations / ~21 s and ends at 9.42x on the search set. The case study is
`experiments/common.py`: mini-coder-4b config, `lib/tech` (pim.yaml + host.yaml), 89 frozen
sessions split by `split(seed=0)`, `ref_ctx` = max over all 89, GPU = g1.jsonl Q8_0/f16.

1. **Workload** `lib/workload/mini_coder_4b.yaml` + `harness/workload.py`.
   - `model:` block with every ModelSpec field (checked equal to `from_hf_config`).
   - `graph:` op templates (`edge_in`, `layer` (per-layer instances), `edge_out`), each with
     kind, cls, subcls, flops / bytes as arithmetic expressions in S (= n new tokens), L (ctx),
     T (logits tokens) and model dims, tensor names templated on the layer prefix, `when:`
     conditions (qk_norm, sub_norms, kv layer), operands static/dynamic, precisions.
   - `Workload` wraps the ModelSpec (attribute delegation) and has `forward_ops(n, ctx, T)`.
     `ops.ops_for(model, ...)` dispatches to it, so Integrator/Space use the YAML graph when
     given a Workload and `forward_ops` otherwise (no equation changes).
   - `trace:` points at an explicit per-turn state file `lib/workload/mini_coder_4b_trace.tsv`
     (session, ctx_start, prefill, gen, tau, GPU times; floats as repr) exported from the raw
     results/traces by a script, plus the split (seed 0) and ref_ctx rule.
   - Op gets `subcls`, `precisions`, `operands` (additive, defaults per CONTRACT).
     forward_ops sets subcls; operands derived in `__post_init__`.
   - Test: YAML ops == forward_ops ops, all fields, for several (n, L, T).
2. **Tech library**: `Tech.precisions`, `Tech.sources` (param -> source or "VERIFY: ..."),
   `params.write_mechanism` on every entry; audit every number; test that every number has a
   `sources` entry. Unsourced numbers listed below.
3. **Host profile** `lib/host/i9_12900k.yaml`: the per-kind measured curves (8 threads)
   inline, the roofline fallback for unprofiled kinds, overhead, tool_scale and where tau
   comes from. `HostProfile.from_file`. Test: identical host op times for every op at many
   (n, L) vs the TSV path.
4. **Objective** `lib/objective/latency.yaml`: latency, area cap, effort cap, GPU baseline,
   best-of tolerance.
5. **Case** `lib/case/mini_coder_swebench.yaml` naming the four files; `Inputs.from_files`.
   Test: evaluator from files == hand-built, bit for bit (every Eval field) on walk start,
   the whole walk log and result, and random points.

## Status
(filled in below when done)
