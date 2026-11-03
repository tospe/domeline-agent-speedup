# R: residency under tight capacity (user decision, Nov 2). Owner: Agent C. Base: autopilot.
Goal: a library parameter caps how much storage a design may have. Start where everything fits (default = no cap,
M0 bit-identical); then tighten it so weights + KV cannot all be resident and residency becomes a real design choice.

## What exists (read first)
- integrator.py ~L278: weights that do not fit are reprogrammed EVERY step:
  write += ceil(overflow / write_parallel_arrays) * t_write_row * rows. KV past a digital tech's memory spills to the link.
- But Evaluator marks any over-capacity design infeasible (wall_ctx), and Space sizes arrays to fit (xbar_scale x need),
  so the reprogram path never decides anything today. The DP omits it (your M3 caveat).

## Phase A — parameter only, no equation change (do first, ~half day)
1. Library: per-tech `max_units` (crossbar arrays) and/or `max_bytes` (local memory), with a `source`/rationale
   (e.g. a die-size or reticle limit; mark VERIFY if assumed). Default absent = unbounded.
2. Space/LargeSpace sizing: units = min(ceil(scale * need), max_units).
3. Evaluator option `over_capacity: "infeasible" | "reprogram"` (default "infeasible" = today). With "reprogram",
   a weight overflow is feasible and pays the EXISTING write term; KV that does not fit stays infeasible unless the
   tech spills (existing rule). Verifier R3 follows the same option (overflow -> implied cost, not violation).
4. Regression: default config == M0 bit-identical (test_m0_repro, test_large_space). With max_units = need at ref_ctx
   ("everything just fits"), identical results too.
5. Measure honestly: sweep max_units from "fits" down to e.g. 25% of need. For each: optimum (DP with an exact
   array-count dimension — the overflow cost is a hinge on the TOTAL arrays, so a pseudo-polynomial DP should still be
   exact; verify vs brute force on 2 layers), walk, det team, TPE-tied, hill. Report whether this already makes the
   space non-separable in practice, or whether the DP stays exact (expected: exact, since the cost depends only on the sum).

## Phase B — residency policy over the session (NEW NAMED TERM "residency"; flagged to the user, who asked for it)
The per-step-overflow model says nothing about WHICH weights are resident. Real hardware pins a set and reprograms
the rest; as the KV grows over a session (KV on crossbar takes arrays), the pinned set must change.
- Design adds a residency policy: per context phase (a few phases along the trace, e.g. by ctx bins), which layers'
  weights (and KV) are pinned. Pinned ops pay nothing extra; unpinned ops pay their own reprogram each step they run;
  changing the pinned set between phases pays a one-time switch cost (rewrite the newly pinned arrays).
- Keep the old term available: policy "overflow-every-step" must reproduce Phase A exactly.
- The capacity agent owns the policy knobs; add the moves to the rule table (pin/unpin a layer, shift a phase boundary)
  and verifier checks (pinned set fits at every state of its phase).
- Ground truth: exact DP state now includes the pinned set (2^layers) — say whether that is intractable at 36 layers.
  Keep the certificate as a VALID lower bound (e.g. relax the switch costs and the integrality of pinning — LP or
  Lagrangian), report its gap to the best design found. Test the bound against brute force on a tiny instance.
- This is the "unstructured" regime: report where exact DP stops working and how loose the certificate becomes.

Deliver: REPORT-R.md (Phase A numbers first; stop and report after Phase A before starting B if Phase A takes > 1 day).
Worktree .worktrees/r-residency, branch h/r-residency. Model equations: only the new named term in Phase B.
