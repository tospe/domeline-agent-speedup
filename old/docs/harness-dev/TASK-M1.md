# Agent A — M1: input schemas.  Worktree: /localhome/tsousape/domeline-agent-speedup/.worktrees/m1-schemas (branch h/m1-schemas)
Read CONTRACT.md first. Brief §M1 is the spec.

Deliver four inputs as YAML files the evaluator loads, extending `harness/inputs.py` and `lib/` (don't fork):
1. **Workload** (`lib/workload/<name>.yaml` + loader): operator graph with per-layer instances; per op: class
   (+ `subcls`), op counts and bytes in/out as functions of state (L = context, n = new tokens) — expressions
   or coefficients, your choice, but they must regenerate exactly what `ops.forward_ops` produces today;
   operands marked static/dynamic; allowed precisions. Plus a pointer to the trace (traces/, states + tau per turn).
   Prove equivalence: a test that compares the YAML-generated op list to `forward_ops` for several (n, L).
2. **Technology library**: add `precisions`, `write_mechanism`, storage geometry/capacity where missing, and a
   `source` for EVERY number (audit lib/tech/*.yaml; list the numbers that have no real source as `VERIFY`
   rather than inventing one).
3. **Host profile**: make the per-class CPU/GPU cost explicit as a file (today implicit in tau + results/cpu/cpu_ops.tsv
   + host.yaml). Loading it must give identical host op times.
4. **Objective and caps**: latency, area cap, effort cap (a small YAML).
**Check:** `Inputs.from_files(...)` (or similar) on the current case study builds an Evaluator whose results equal the
hand-built one bit for bit on a set of designs (walk start, walk result, a few random points). Add as a test.
Coordinate: Agent B (verifier) and C (large space) read these fields; keep the names in CONTRACT.md.
