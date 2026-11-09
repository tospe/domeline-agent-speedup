# Agent C — M3: large design space.  Worktree: /localhome/tsousape/domeline-agent-speedup/.worktrees/m3-large-space (branch h/m3-large-space)
Read CONTRACT.md first. Brief §M3 is the spec.

1. **Encoding** (new module, e.g. `harness/large_space.py`, alongside space.py, not replacing it): per-layer, per-operator
   placement and precision across all layers of the workload model (check its layer count in harness/model.py /
   the config — the brief says 36), plus the existing hardware knobs. Use `Design.mapping` per-op-name keys.
2. **Regression:** the small space is the special case with every layer tied; a test that a tied large point gives the
   identical Eval to the 14-knob point for a sample of designs (incl. walk start and walk result).
3. **Coupling:** verify the Integrator charges two link transfers when one op moves to PIM with neighbors on host
   (read integrator.py; it tracks tensor crossings). Find and document a concrete example where moving one op hurts and
   moving it with its neighbors helps. If the integrator does not model something needed, STOP and report — don't
   change equations.
4. **Speed:** time per evaluation on the full trace; target ms. If slow, use replay on sampled curves (paper: within 0.3%
   of exact) and report the error. Report numbers; stop and report if > ~1 s/eval.
5. **Ground truth without a sweep:** (a) certificate: Amdahl bound + per-term lower bounds -> a floor on any design's
   time (see docs/theory.md, evaluate.py readings); (b) exact mapping optimum with hardware fixed by DP over the op chain;
   test DP against brute force on a tiny instance (e.g. 2 layers, few ops).
Optional, cut first: session-varying policy. Skip it.
