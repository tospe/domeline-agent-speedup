# M5 split (lead-assigned). Read CONTRACT.md. Base: autopilot (M0+M1+M2+M3+M4 draft merged).
M4 tool API: harness/tools.py (Toolbox: evaluate/read/moves/verify/log; RunRecord JSON; TOOL_SCHEMAS).

## Agent C — the loop (harness/team.py), deterministic mode first
- `team_search(ev, toolbox, agents, *, eps=0.005, stall_k=..., stop_frac=0.02, budget)` per brief §M5 steps 1–5:
  supervisor calls read -> routes to the owner of the binding term (mapping / hardware / capacity; write the
  term->owner table explicitly and test it), agent proposes, verify (illegal -> return violated rule, retry within
  a budget), evaluate, accept if > eps faster, log (move, reading, verifier result) into RunRecord.
  Stall: no accept in k proposals while Amdahl gap > threshold -> supervisor restructure (coordinated multi-op move,
  e.g. tie a role across all layers or a window of neighbors — your M3 §4a finding — or reassign).
  Stop within 2% of the Amdahl floor or budget.
- Knob ownership enforced: an agent's proposal touching another agent's knobs is rejected (logged).
- **Deterministic mode**: agents = the fixed rule table (search.proposals partitioned by owner). On the small
  space it MUST reproduce domeline_walk (same accepted moves, same final design, same eval count); make that a test.
  Then run it on LargeSpace with FastEvaluator and report.
- Mind your own lesson: canonicalize only the point being evaluated.
- Do not wait on the lead's open decisions (sampling rule, E5 framing); keep the edge/sample options configurable.

## Agent A — LLM agents (harness/llm_agents.py)
- Backends with pinned IDs: `claude-sonnet-5-5`, `claude-haiku-5-5` (via the `claude` CLI headless, as harness/agent.py
  does; or the API if already set up — check), and local mini-coder-4b via llama.cpp (agent.py has call_local; reuse).
  Record tokens in/out per model into RunRecord.add_tokens on every call; deterministic seeds/temperature 0 where possible.
- Three role agents (mapping / hardware / capacity) with the same interface as C's deterministic agents: given the
  reading, the agent's own knobs and their options, and recent history, return one move (structured JSON). They may
  call moves(term) for candidates. Each agent sees only its own knobs.
- Also the E5 baselines that are LLM-based: single LLM agent without readings (agent.py's agent_scalar is the
  small-space version; generalize to LargeSpace), and an "Evolving Agent Teams"-style LLM supervisor team without
  readings or routing.
- Interface with C: agree via a short `docs/harness-dev/AGENT_IFACE.md` (C owns the loop, you own the agents). Write it
  first and message nothing — C will read it; if C has already written it, follow it.
- Test with a fake backend (canned responses) so tests need no network; one smoke run per real backend, budget 10.
