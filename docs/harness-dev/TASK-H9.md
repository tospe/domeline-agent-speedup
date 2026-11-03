# H9: cost-efficient mixed-model teams (user chose option 4, Nov 2). Base: autopilot.
Question: can a team of CHEAP models (Haiku 5.5, local 4B) match Sonnet 5.5's result at a fraction of the $ cost,
in settings where a single cheap agent is not enough? Report honestly either way: if single Haiku already suffices
everywhere, that IS the finding ("one cheap agent + readings is enough; teams not needed").

Prices: lib/pricing/claude.yaml. Every RunRecord gets `cost_usd` per model and total (split input into fresh /
cache-read / cache-write when the CLI reports them; otherwise all at input rate and say so).

Methods (all with readings + verifier, sampling=host):
- R1 sonnet single (reference)            - R2 haiku single
- R3 haiku team (3 role agents, routing)  - R4 haiku team + SONNET supervisor called ONLY on stall/restructure
- R5 ensemble: 3 independent haiku singles (different seeds), best of three (cost = sum)
- R6 4B role agents + sonnet supervisor on stall (4B is free; tests "free workers, paid escalation")
Settings: small space; large uncapped; large capped {area 5000} and {area 6000 + effort 5} with DP reference
(results/large/dp_caps_main.json). Budget 50 evals. 3 seeds for haiku/4B-based rows, 2 for rows with sonnet.
Metrics per run: best speedup, gap to DP/legal optimum, eval of first 1% band, success (in band at budget), $ total,
$ to first 1% band, tokens per model, wall. Summary table: success rate and median $ per row per setting.

Split:
- Agent C (loop): supervisor hook in team.py: `supervisor=` callable invoked on stall (restructure) and logged;
  escalation counts and its tokens in the record; ensemble runner (parallel independent runs, best-of). Tests with fake backend.
- Agent A (agents): LLM supervisor agent (proposes a coordinated multi-role move from the reading + team history),
  cost_usd accounting from lib/pricing, hrun method entries R1–R6, then run the matrix. Estimate $ before launching;
  if > $40 total, run 1 seed and stop for the lead.
Interface between C and A: extend AGENT_IFACE.md (C writes the supervisor signature first).
