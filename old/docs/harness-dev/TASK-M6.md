# M6 + leftovers (lead-assigned). Base: autopilot. Experiment IDs are H4–H8 (repo already uses E1–E8).
Pending user decisions (do NOT bake in): sampling rule (keep `exclude`/edge options configurable; default = verifier on,
sampling host-only), and E5/H5 framing (capped is certain; uncapped kept as a row).

## Agent C — capped ground truth (harness/large_space.py)
- Mapping DP under area and effort caps with hardware fixed. Start with the Lagrangian (min time + λ·area, sweep λ):
  exact points on the hull of the time–area front; report the duality gap at each cap (best hull point under cap vs cap).
  If the gap is material, add an exact area dimension (discretized arrays) and test it against brute force on 2 layers.
- Outer loop over hardware settings (parallel by setting, 20 cores ok) -> `results/large/dp_caps_<tag>.json` with the
  best design per cap for caps {uncapped, 8000, 6000, 5000, 4000, 3000 mm²} and effort {5, 7 months}.
- Tighten the certificate if cheap (e.g. per-hardware DP min as the floor). Report each.
- Branch h/m6-dp, worktree .worktrees/m6-dp, report REPORT-M6-dp.md.

## Agent B — baselines + runner scaffold
- Non-LLM baselines on LargeSpace with FastEvaluator, all seeded, same budget semantics as Evaluator:
  random, TPE (optuna, as search.tpe_search), simulated annealing (and/or GA — pick one, justify), single-knob hill climb.
  All respect caps and use the verifier the same way the team does (illegal = no eval) with a flag to turn that off.
- `experiments/hrun.py` common runner: config YAML/CLI -> RunRecord JSON per (method, seed) under results/h/<exp>/,
  plus summary CSV (best speedup vs evals, eval of first 1% band, gap to certificate, gap to DP optimum when available,
  evals, wall, tokens). Methods registry so A's LLM methods and C's team plug in by name.
- H4 runner (small space: walk, deterministic team, + LLM methods by name once A lands) and H6 runner (verifier off:
  illegal proposals per run, rules hit, what would have been accepted) wired to the deterministic team now, LLMs later.
- Branch h/m6-runners, worktree .worktrees/m6-runners, report REPORT-M6-runners.md.
