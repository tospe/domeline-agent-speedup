# Agent B — M2: offload verifier.  Worktree: /localhome/tsousape/domeline-agent-speedup/.worktrees/m2-verifier (branch h/m2-verifier)
Read CONTRACT.md first. Brief §M2 is the spec.

Write `harness/verify.py`: `verify(design, lib, model, sessions, caps) -> VerifyResult{legal, violations:[(rule, op, reason)], implied_costs:[...]}`.
Work against the existing objects (Design with per-op mapping, Tech, forward_ops, Integrator.capacity/Residency).
Fields from M1 (precisions, write_mechanism, subcls) may not exist yet in your branch: read them with the defaults
in CONTRACT.md, and add the minimal additive bits yourself if needed (say so in the report).
Rules: (1) unit supports class + precision; (2) dependencies respected and every host<->PIM boundary crossing is
charged a transfer — check this against what Integrator actually charges, don't re-derive it; (3) resident data fits
capacity at EVERY state of the trace (use the trace's max ctx per session, and check the 64K case); (4) area and effort caps
(reuse Space.area_mm2 / effort_months).
Implied costs: e.g. dynamic-operand MVM on a crossbar implies a per-token write; confirm it appears in the
evaluator's write term (cross-check with Integrator output) and report it.
Tests `tests/test_verify.py` (must pass): illegal with the right rule — softmax on a crossbar; sampling on PIM;
attention on a crossbar with no write mechanism; KV footprint over capacity at 64K tokens. Legal — the paper's headline
design (the walk's result, see docs/results.md / results/search). Also a test that every feasible design in a random sample
of the 14-knob space that the Evaluator calls feasible is verifier-legal (and report any disagreement — don't hide it).
No Lean.
