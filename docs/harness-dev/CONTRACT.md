# Harness-dev contract (lead: pane w1:p2). Read this first; it overrides your task file on conflicts.

Source brief: /localhome/tsousape/Domeline/notes/harness-dev-brief.md (read §1–§2 and your milestone).
Paper: /localhome/tsousape/Domeline/main.tex (§3–§5). Plan: /localhome/tsousape/Domeline/notes/paper-outline.md.
The evaluator IS this repo (`harness/`). Do not rebuild it. Repo docs: docs/harness.md, docs/results.md, docs/theory.md.

## Ground rules (from the brief, non-negotiable)
- Do NOT change the model's equations (step time, replay, readings) in harness/integrator.py, session.py,
  evaluate.py. A new cost = a new, named term, and you flag it in your report.
- The small-space numbers must keep reproducing: 575,400 designs / 280,296 feasible / optimum 9.48x /
  walk within 0.7% at eval 20. The lead is adding `tests/test_m0_repro.py`; run it (and the existing
  `python -m pytest tests -q`) before you report done. Until it lands, run the existing tests.
- Deterministic, seeded. Plain Python in the repo's style (dataclasses, short docstrings, no frameworks).
- Report honestly; a number that differs from the paper gets reported, not tuned away.

## Shared interface (all three work against the EXISTING Python objects; no one blocks on another)
- Op graph: `harness/ops.py` `Op` (name like "L3.q_proj", kind, cls, layer, reads/writes tensors) from
  `forward_ops(spec, n_new, ctx)`. Classes: mvm_static, mvm_dynamic, kv_append, vector, gather.
- Tech: `harness/tech.py` `Tech` (kind, supports, params, source, area, effort) loaded from lib/tech/*.yaml.
- Design: `harness/design.py` `Design.mapping` already accepts per-op-name keys ("L3.q_proj") over kind/class/"*".
- Space: `harness/space.py` (17 knobs; the paper's 14-knob space pins xbar_tile=isaac, adc_per_ima=8, cell_bits=2).
- Inputs: `harness/inputs.py` (four-input interface already partly exists; extend it, don't fork it).

New fields — names fixed now so M1 (adds them) and M2 (uses them) agree. All optional with defaults that
reproduce today's behavior:
- `Tech.precisions: tuple[int,...]` weight/operand bit widths supported (default: () = any).
- `Tech.params["write_mechanism"]: bool` can it rewrite its stored matrix per token (crossbar default: read
  existing write params; True iff t_write_row_s is set).
- `Tech.params["capacity_bytes"]` or crossbar geometry (rows, cols, bits_per_cell) x units — already used by
  Integrator.capacity / Residency; reuse that, don't recompute differently.
- `Op.precisions: tuple[int,...]` allowed precisions (default ()), `Op.operands: list[(tensor, "static"|"dynamic")]`
  (default derived: weights static, KV dynamic, activations dynamic).
- Op classes gain two finer labels used by the verifier: `reduction_nonlinear` (softmax, norms, act) and
  `sampling` (sample). Keep `cls` as is; add `Op.subcls: str | None` instead, so nothing existing breaks.

## Git
- You work ONLY in your worktree (below) on your branch. Commit small, clear messages, end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push, never touch `autopilot`; the lead merges.
- If you must edit a file another agent likely edits (tech.py, ops.py, inputs.py), keep the diff minimal and
  additive and say so in your report.

## Reporting
When done (or blocked), write `docs/harness-dev/REPORT-<milestone>.md` in your worktree: what works, numbers,
what differs from the paper and why, decisions that change a claim, files touched, test output. Commit it.
Then stop and wait for the lead. If blocked on a decision, write it in the report and stop; don't guess on
anything that changes a paper claim.

## Environment (added by lead after M0)
- Python: `/localhome/tsousape/tools/venv/bin/python` (system python3 is 3.6 and lacks deps). Run tests as
  `/localhome/tsousape/tools/venv/bin/python -m pytest tests -q`.
- M0 regression: `tests/test_m0_repro.py` (~40 s; on autopilot — `git merge autopilot` into your branch).
  Uses `experiments/m0_repro.py` helpers. All four checks reproduce exactly today.
- Measured: one uncached evaluation on the 45 search sessions = 0.66 s (full replay). The walk = 31 evals, 20 s.
- LLM backends (decided Nov 2): pin explicit IDs `claude-sonnet-5-5` and `claude-haiku-5-5` (the paper's earlier
  `sonnet`/`haiku` alias runs are taken to be 5.5), plus the local mini-coder-4b via llama.cpp. Record the ID in every run record.
