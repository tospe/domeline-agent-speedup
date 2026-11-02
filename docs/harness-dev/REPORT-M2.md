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

## 1. Status: done. All M2 tests pass; M0 still reproduces.

`harness/verify.py`: `verify(design, lib, model, sessions, caps) -> VerifyResult(legal, violations, implied_costs, checked)`,
where `violations` holds `(rule, op, reason)` named tuples. `verify_point(space, point, sessions, caps)` builds the
design with `Space.to_design` and takes area and effort from `Space.area_mm2` / `effort_months`. One call takes about 10 ms
(an evaluation takes 0.66 s). It changes no equations: integrator.py, session.py and evaluate.py are untouched.

Rules as implemented (the module docstring lists them):
- **R1**: `R1.support` (the tech's `supports` lacks op.cls), `R1.sampling` (`sample` off the host unless the tech lists
  `"sampling"`), `R1.precision` (Op/Tech precisions; default () accepts any), `R1.write_mechanism` (a dynamic
  operand on a crossbar with `write_mechanism` false, or no `t_write_row_s`).
- **R2**: `R2.dependency` (a read with no earlier writer), `R2.dataflow` (kv_append, attn_score and attn_value of a
  layer on different techs: the integrator moves no KV between units, so that traffic would go uncharged),
  `R2.transfer` (the crossings, link bytes, launch term and transfer term the op graph implies are compared with
  `Integrator.step` for a decode step and a 256-token prefill step at the longest state; also the token entering
  through embed and leaving through sample, which the integrator never charges), and `R2.implied` (an implied cost
  missing from its term).
- **R3** `R3.capacity`: `Integrator.capacity` at every turn's start and end context in every session, plus an optional
  `Caps.min_ctx` (e.g. 65536). Need is nondecreasing in ctx, so the check runs at the largest state and binary-searches
  the first state over capacity (session and turn appear in the reason). A test checks the monotonicity.
- **R4** `R4.area` / `R4.effort` use the Space numbers. Without a Space point, area uses the same formula on the design
  and effort is a lower bound (tech efforts only), flagged in `checked`.

Implied costs: `kv_write`, `reprogram`, `kv_spill`, `scatter` and `token_io`, each with seconds per decode step, the
term it must appear in, and `charged`. **Cross-check:** for the headline design, the 36 per-token KV writes
(xbar_transpose: 1 K row + 1 V row = 101.76 ns per layer) add up exactly (rel 1e-12) to the integrator's `write` term at
64K. The op graph's crossings, link bytes, launch and transfer matched `Integrator.step` (to 1e-9) for every edge=cpu
design in the sweep and for a seeded sample of 2,000 edge=vfu designs. No R2.transfer or R2.implied mismatch occurred inside
the 14-knob space; R2.transfer only fires on the edge=vfu token
I/O.

## 2. Numbers: the verifier against the exhaustive sweep (all 575,400 designs)

`experiments/verify_sweep.py` produced `results/search/verify_sweep_main.json` (16 workers, 645 s):

| Evaluator | Verifier | n | rules |
|---|---|---|---|
| feasible | legal | **131,256** | none |
| feasible | illegal | **149,040** | R1.sampling + R2.transfer (every one has `edge: vfu`) |
| infeasible | illegal | 143,184 | R3.capacity |
| infeasible | illegal | 151,920 | R1.sampling + R2.transfer + R3.capacity |
| infeasible | legal | **0** | |

- Evaluator-feasible: 280,296, which matches M0. Verifier-legal: 131,256. **There is no legal-but-infeasible design,**
  and every capacity infeasibility the Evaluator finds, R3 also finds.
- **The one disagreement is `edge: vfu`**, which puts `embed` and `sample` on the vector unit. This was expected
  before the run. The brief's required test ("sampling on PIM is illegal") forces it.

## 3. Decisions that change a paper claim (lead to decide; nothing was tuned)

1. **The exhaustive 9.48× optimum is a design the verifier rejects.** The fastest feasible design overall is
   edge=vfu: 398.80 s, **9.4847×** (die link, 16 VFUs, xbar_scale 1.25, 18,796 mm², 17 months). The fastest
   *legal* design is edge=cpu: 399.49 s, **9.4684×** (rounds to 9.47×). If the paper adopts the verifier,
   "280,296 feasible / optimum 9.48×" becomes "131,256 legal / optimum 9.47×", and the walk's 9.42× is within
   **0.55%** of the legal optimum instead of 0.7%. The walk's own design is edge=cpu and legal. The M0 regression
   test is unchanged; it still checks the Evaluator's numbers.
2. **Why edge=vfu is faster, and whether the missing charge explains it.** It is not the missing charge. The
   uncharged token I/O (`token_io`) is one token id each way plus one sync per step, about 50 ns per token on the die
   link. Over 288,662 generated tokens that is ~0.014 s, against a 0.69 s gap. The gain is real under the model:
   sampling next to the LM head avoids moving the 300 KB logits to the host. So the illegality rests only on the
   *rule* that sampling is host-only. The rule follows the brief: sampling needs the full logits, an RNG and
   top-k/top-p, and the agent reads every token on the host. Options:
   (a) keep the rule: report 9.47× and 131,256, and drop `vfu` from the `edge` knob in M3;
   (b) allow a tech to list `sampling` (supported today: add it to `vfu_puma.supports`), and add a new named
   `io` term for the per-token token-in/out crossing, so R2.transfer stops firing. That term is an equation change
   and needs your sign-off. Under (b), 9.48× stands up to that ~0.014 s.
   I have not chosen. The code implements (a), and (b) is a one-line library change plus the new term.
3. `embed` on PIM is also R2.transfer (the uncharged token id coming in). In M3's per-op space this makes embed on PIM
   illegal until an `io` term exists. It is the same decision as 2.

## 4. Files touched
- New: `harness/verify.py`, `tests/test_verify.py`, `experiments/verify_sweep.py`,
  `results/search/verify_sweep_main.json`, this report.
- Not touched: ops.py, tech.py, inputs.py and the model equations. M1's fields (`Op.subcls`, `Op.precisions`,
  `Op.operands`, `Tech.precisions`, `params.write_mechanism`) are read with `getattr` and the CONTRACT defaults
  (subcls derived from kind: softmax/rmsnorm/qk_norm/act -> reduction_nonlinear, sample -> sampling). Once M1 lands,
  its values take precedence automatically. There is nothing to merge by hand.
- Merged `autopilot` at 237a240 (M0 test). autopilot has since moved on (e770828, M4 draft with `tests/test_tools.py`);
  I did not merge that.

## 5. Tests
- `tests/test_verify.py`: 12 passed (28 s). Coverage: softmax on a crossbar -> R1.support; sample on vfu -> R1.sampling
  (+ R2.transfer); attention on a crossbar with no write mechanism, both via the explicit flag and via a missing
  `t_write_row_s` -> R1.write_mechanism on attn_score/attn_value/kv_append only; KV at 64K: attention on the VFU with
  4 GB and fp16 KV -> R3.capacity naming the 64K session, legal on a short trace, illegal again with
  `Caps(min_ctx=65536)`; the headline at xbar_scale 0.75 -> R3.capacity with a charged `reprogram`; the headline legal
  (also at min_ctx 65536) with area 9,728.696 mm² and 7 months; the KV write equal to the write term; R4 caps;
  R2.dataflow for a split KV; monotone capacity; a live seeded sample of 40 designs (Evaluator vs verifier;
  disagreements may only be the known edge=vfu set, and legal must imply feasible).
- `tests/test_m0_repro.py`: 4 passed (86 s).
- Full `pytest tests -q`: 48 passed, 3 failed. All three failures are in `tests/test_claims.py` (main.tex vs
  paper/claims.yaml: stale phrases, 6 mismatched and some unregistered numbers). They fail identically on the main
  checkout without my changes, so they come from paper edits, not M2.

## 6. Questions for the lead
1. Rule (a) or (b) in §3.2? This decides whether the paper says 9.48× or 9.47×.
2. Should the 14-knob claims stay "Evaluator-feasible" (280,296) with the verifier reported separately, or switch to
   "legal"?
