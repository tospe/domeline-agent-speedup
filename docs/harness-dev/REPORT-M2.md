# M2 report: offload verifier (Agent B)

## 0. Plan (written after orienting, before code)

`harness/verify.py`, `verify(design, lib, model, sessions, caps, *, area_mm2=None, effort_months=None) -> VerifyResult`,
plus `verify_point(space, point, sessions, caps)` which builds the design with `Space.to_design` and takes
area/effort from `Space.area_mm2` / `Space.effort_months`. Plain checker, no change to integrator/session/evaluate.

Rules (named so tests and the agents can key on them):
- **R1 support / precision / write_mechanism.** The op's tech lists the op's `cls`. Finer labels (`Op.subcls`,
  read with `getattr` default; derived from kind when M1's field is absent): `reduction_nonlinear` (softmax,
  norms, act) needs a tech that runs `vector`; `sampling` (sample) is host-only unless a tech lists `sampling`
  explicitly. Precision: the op runs at weight_bits (static MVM), kv_bits (KV ops) or 16 (activations); it must
  be in `Op.precisions` and `Tech.precisions` when those are non-empty (default () = any). An op with a dynamic
  operand on a crossbar needs `write_mechanism` (default: True iff `t_write_row_s` is set).
- **R2 dataflow / transfer.** Every read has an earlier producer; KV is produced (kv_append) on the tech that
  reads it (attn_score / attn_value), since the integrator moves no KV between units. Boundary crossings
  are derived from the op graph and compared with what `Integrator.step` charges (crossings, link bytes, launch
  term) for a decode and a prefill step. The step's input token comes from the host and the sampled token must
  return to it (the agent reads every token); the integrator does not model either, so a design that puts
  `embed`/`sample` on PIM has an uncharged crossing -> violation.
- **R3 capacity.** At every state of the trace (each turn's start and end context, every session), using
  `Integrator.capacity` (the same test the evaluator uses). Need is nondecreasing in ctx, so the check is at the
  max state and, if over, the first over state is reported (session, turn, ctx). Tested at 64K.
- **R4 caps.** Area and effort against caps, via Space.area_mm2 / effort_months.

Implied costs: per-token KV writes for KV on a crossbar (cross-checked against the `write` term and the
per-op kv_append times), weight reprogramming past the wall, KV spill over the link, host scatter (alpha*N).
Each carries `charged: bool`; an uncharged implied cost is an R2 violation.

Tests (`tests/test_verify.py`): softmax on a crossbar (R1 support), sample on the vector unit (R1 sampling),
attention on a crossbar with no write mechanism (R1 write_mechanism), KV over capacity at 64K (R3), the walk's
headline design legal, and a live random sample of the 14-knob space where every Evaluator-feasible design must
be verifier-legal *or* hit a disagreement listed in this report. A script compares against the full sweep TSV.

Expected disagreement (decided before running): the 14-knob space has `edge: vfu`, which puts `sample` on the
vector unit. The Evaluator calls those feasible; the verifier (by the brief's own test) calls them illegal.
I will count it and report it, not hide it.
