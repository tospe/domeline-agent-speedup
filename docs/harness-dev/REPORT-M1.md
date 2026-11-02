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

## What works

| Input | File | Loader | Check (tests/test_inputs_m1.py) |
|---|---|---|---|
| Workload: op graph | `lib/workload/mini_coder_4b.yaml` | `harness.workload.load_workload` -> `Workload` | ops == `forward_ops`, every Op field, 6 states (n, L, T): decode at L = 1 / 777 / 40000, prefill chunks, odd sizes; also BitNet (sub_norms, relu2) and a hybrid spec with 6 of 30 layers keeping KV, which covers every `when:` branch |
| Workload: trace | `lib/workload/mini_coder_4b_trace.tsv` (3,940 turns, 89 sessions) | `TraceRef.sessions()`, `.split()` | == `common.sessions()` (Session equality, meta included); split == `common.split` (45/44) |
| Technology library | `lib/tech/pim.yaml`, `host.yaml`, `lib/tech_dram/aim.yaml` | `load_library` (new fields `precisions`, `sources`) | every number has a `sources` entry; `write_mechanism` is explicit in every entry |
| Host profile | `lib/host/i9_12900k.yaml` | `HostProfile.from_file` | per-op times and terms of an all-host step are identical to the TSV path, at 7 states including extrapolation at 70K |
| Objective + caps | `lib/objective/latency.yaml` | `Objective.from_file` | metric, caps (null = inf), tol, GPU baseline |
| Case study | `lib/case/mini_coder_swebench.yaml` | `Inputs.load_case` / `Inputs.from_files` | **bit for bit**: `Eval ==` on every field (time, terms, by_kind, readings, D, headroom) for the walk start, all 31 walk evaluations, the walk's steps and stop reason, `best()`, and 4 random points |

How the evaluator uses the YAML graph: `Workload` wraps the ModelSpec (it passes unknown attributes
through to it) and has `forward_ops(n, ctx, T)`. The new `ops.ops_for(model, ...)` calls it when it is
present and `ops.forward_ops` otherwise. Integrator.ops and Space.xbar_need call `ops_for`. No equation
was changed.

Numbers: loaded from the files, the walk reproduces the hand-built run exactly (31 evaluations, the
same stop, the same best design). Evaluation time is unchanged apart from building op lists from YAML:
about 3.6 ms per list, with about 105 lists cached per Evaluator, so roughly +0.2 s, paid once on the
first evaluation. Measured over three evaluations while the test suite was loading the machine:
2.0/1.4/1.4 s from the files vs 1.8/1.35/1.4 s hand-built.

## Schema decisions (CONTRACT field names kept)

- `Op` gains three fields:
  - `subcls`: `reduction_nonlinear` for rmsnorm, qk_norm, softmax and act; `sampling` for sample;
    None for rope, residual, mixer, the MVMs, embed and kv_append.
  - `precisions`: () means any.
  - `operands`: None at construction, then derived in `__post_init__`. Weights (`w.<kind>`,
    `lm_head`) and the embedding table (`w.embed`) are static; `kv.K`, `kv.V` and every activation
    read are dynamic.

  Tensor names match the ones `Residency` reports, so B can join the two. `forward_ops` now sets
  `subcls`. It leaves `precisions` empty, because the YAML sets them per class; the equivalence test
  excludes that one field and checks it separately. The YAML writes every op's operands out
  explicitly, and the test checks that they equal the derived rule.
- Workload precisions are an assumption, and the YAML says so: weights {8, 16}, KV {8, 16},
  activations/vector/gather {16}. int4 is left out because its accuracy cost is not modeled (the same
  choice as space.py).
- Tech changes:
  - `Tech.precisions`: crossbars, vfu and upmem {8, 16}; star {16}; cpu any; **aim_gddr6 {16}**.
  - `Tech.write_mechanism` property: the params value if present; otherwise a crossbar has one only
    if it has a `t_write_row_s`. Every entry now sets `params.write_mechanism` explicitly, and only
    softmax_star is False.
  - New helpers `Tech.supports_precision(bits)` and `Tech.unsourced()` (numbers with no source or a
    VERIFY one), and a `Tech.sources` map from each number to its source.

  Storage geometry and capacity are unchanged. They already live in params (rows/cols/bits_per_cell;
  local_mem_bytes/mem_scales_with_units). I added no `capacity_bytes` field, so Integrator.capacity /
  Residency stays the only source of truth. softmax_star has no storage.
- Host profile file:
  - It inlines the measured curves (the 8-thread rows of cpu_ops.tsv, floats written as repr) and the
    roofline fallback.
  - `CpuProfile` gained a `by_kind=` constructor, and `cpu_profile(tech)` accepts an inline
    `params["profile"]`.
  - Per class at the workload's sizes, every op kind is measured except embed (gather) and sample
    (vector), which use the roofline (200 GOP/s, 60 GB/s, 1 us).
  - tau stays in the trace (column `t_tool_s`, measured on this host). `tool_scale` is in the file.
- `experiments/export_inputs.py` generates the trace and host files deterministically. The tests show
  they equal the raw sources, so they can be regenerated whenever the traces change.
- `Inputs` gains `caps: Objective` and `ref_ctx`. `evaluator(gpu=None, ...)` takes the GPU and the caps
  from the objective by default, and `run(..., gpu=None, tol=None)` does the same. Old call sites are
  unchanged (test_readings passes).

## Numbers with no real source (marked `VERIFY` in `sources`)

| tech | number | value | why |
|---|---|---|---|
| cpu, cpu_roofline | ops_per_s / mem_bw_Bps | 200e9 / 60e9 | i9-12900K datasheet-level estimate; on `cpu`, used only for embed and sample |
| cpu, cpu_roofline | op_overhead_s | 1e-6 | assumed |
| xbar_isaac, _wv, _transpose | t_add_s | 1e-9 | assumed adder-tree level |
| xbar_isaac, _transpose | t_write_row_s | 50.88e-9 | PipeLayer value, not checked (headline moves <1%) |
| xbar_isaac_wv | t_write_row_s | 1e-6 | assumed write-verify corner |
| xbar_isaac, _wv, _transpose | write_parallel_arrays | 1024 | assumed |
| xbar_transpose | area_mm2_per_unit, effort_months | 0.0045, 3 | assumed (+15% area) |
| vfu_puma | ops_per_s, op_overhead_s, area, effort | 4e9, 5e-9, 0.004, 1 | PUMA Table 3 not checked |
| softmax_star | ops_per_s, mem_bw_Bps, op_overhead_s, area, effort | 64e9, 64e9, 2e-9, 0.02, 4 | placeholders to size |
| upmem_dpu | bank_bw_Bps | 0.63e9 | Gomez-Luna 2022, not checked |
| aim_gddr6 | ops_per_s, bank_bw_Bps, op_overhead_s | 1e12, 1e12, 1e-7 | recalled, not checked |

Three more groups of numbers have no source but sit outside the library, in `harness/space.py`. I
didn't move them, because Agent C edits that file:
- the link presets `LINKS` (cxl/package/die bandwidth, launch, sync, t0; already marked VERIFY there);
- the `EFFORT` table (months for link, vfu_bw, xbar_tile and cell_bits);
- `CELL4_T_WRITE_ROW_S` = 1 us, the 4-bit-cell write time.

The ISAAC-derived area constants in that file are sourced in comments.

Some numbers are conventions rather than VERIFY: effort 0 for reference and commercial parts, area 0
for DRAM-PIM dies, the `units_ref` sizing references, and `input_bits` 8 (a workload choice; ISAAC
itself uses 16-b inputs).

## Differences from the paper / decisions for the lead

1. **aim_gddr6 precisions = {16} (BF16).** The evaluator ignores precisions, so no number moves today.
   But once M2's verifier checks rule 1 (precision), every E8 AiM design at weight_bits 8 becomes
   illegal. The entry's own notes already say the harness flatters AiM by charging int8 bytes. The
   choice: keep {16}, which is honest but means E8 AiM runs need weight_bits 16, or widen it to {8, 16}.
2. **Crossbar write path.** The ISAAC entries now state that ISAAC never writes at run time and that
   the write path (`write_mechanism: true`, 50.88 ns) is borrowed from PipeLayer. The existing notes
   already said this. I flag it because M2's rule "attention on a crossbar with no write mechanism"
   depends on it. Every shipped crossbar has a write mechanism, so that M2 test needs a test-only
   tech or a `params.write_mechanism: false` override.
3. **Prefill on the host.** The host profile charges a prefill of n tokens as n times the measured
   per-token decode time (`op.tokens * prof.time`). That is the existing model; the host file header
   now makes it visible, and I didn't change it.
4. **No reproduced number differs.** Loading from the files gives the same results as the hand-built
   path, bit for bit.

## Files touched

- Shared code, with minimal additive edits:
  - `harness/ops.py`: Op fields, `SUBCLS`, `default_operands`, `weight_tensor`, `kv_layer_ids`
    (factored out of forward_ops), `ops_for`.
  - `harness/tech.py`: Tech fields and methods, the CpuProfile `by_kind` constructor, the inline profile.
  - `harness/inputs.py`: Objective, from_file / from_files / load_case, the caps and ref_ctx defaults.
  - `harness/integrator.py` and `harness/space.py`: one line each, `forward_ops` -> `ops_for`.
- Library YAML: `lib/tech/pim.yaml`, `lib/tech/host.yaml`, `lib/tech_dram/aim.yaml`. I added
  `precisions`, `sources` and `params.write_mechanism`; no value changed.
- New files: `harness/workload.py`, `lib/workload/mini_coder_4b.yaml`,
  `lib/workload/mini_coder_4b_trace.tsv`, `lib/host/i9_12900k.yaml`, `lib/objective/latency.yaml`,
  `lib/case/mini_coder_swebench.yaml`, `experiments/export_inputs.py`, `tests/test_inputs_m1.py`.

## Questions for the lead

- Decision 1 above (AiM precision).
- Should `experiments/common.sessions()` read the trace file once this is merged, so there is one
  source? I left common.py alone; the test keeps the two equal.
- Should LINKS/EFFORT move out of space.py into a library file with sources? I didn't do it, since
  it's C's file.

## Test output

`/localhome/tsousape/tools/venv/bin/python -m pytest tests -q` (autopilot 237a240 merged in, so
tests/test_m0_repro.py is included):

```
FAILED tests/test_claims.py::test_registry_matches_paper - AssertionError: ph...
FAILED tests/test_claims.py::test_no_mismatched_numbers - AssertionError: 6 n...
FAILED tests/test_claims.py::test_no_unregistered_result_numbers - AssertionE...
3 failed, 51 passed in 336.32s (0:05:36)
```

All of tests/test_inputs_m1.py (15) and tests/test_m0_repro.py pass. The three test_claims failures
are not from this branch: they fail the same way (3 failed, 3 passed) on a clean `autopilot`
checkout. They compare main.tex against paper/claims.yaml, so the cause is the paper text versus the
registry.
