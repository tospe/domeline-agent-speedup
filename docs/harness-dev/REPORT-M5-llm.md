# REPORT M5 (LLM part) — Agent A, branch h/m5-llm

## Plan (written after orienting, before code)

Facts found:
- There is no `harness/team.py` or `AGENT_IFACE.md` anywhere yet. I wrote `docs/harness-dev/AGENT_IFACE.md`
  first (commit c9ebb96): ownership table, `AgentContext`, `Proposal`, and `propose(ctx)`.
- Claude backend: the `claude` CLI 2.1.316 accepts the pinned IDs (`--model claude-haiku-5-5` checked) and
  returns `modelUsage[<id>]` with input / cache-read / cache-creation / output tokens and cost. There is no API
  key and no `anthropic` SDK, so the CLI it is. The CLI has **no temperature or seed flag**, so Claude runs are
  not reproducible call for call. Local: llama-server is not running, but the 3080 is idle;
  `traces/serve.sh` starts it, and llama.cpp takes temperature 0 plus a seed.
- `Toolbox.moves` calls `search.proposals`, which reads small-space knobs (`p["attention"]`), so it does
  not work on LargeSpace points. The LLM agents call it only on a small Space, or catch the error.
- LargeSpace has 626 knobs (616 slots). The prompts summarize slots as runs per op family
  (`rope: cpu L0-35`), and the agents write group moves as patterns (`L*.rope`, `L10-20.attention`),
  expanded to concrete knobs before returning.

Build (`harness/llm_agents.py`):
1. Backends: `ClaudeCLI(model_id)`, `LocalLlama` (temperature 0, seed from call_id), `FakeBackend`
   (canned responses, for the tests). Each returns usage, and every call goes into `RunRecord.add_tokens`.
2. `LLMAgent(role, backend)`. Its prompt holds the reading, its own knobs only (options + current values),
   the rule-table candidates for its knobs when available, recent history, and the last rejection.
   It returns one `Proposal` (structured JSON). Bad JSON, unknown knobs or values outside its own knobs
   get one corrective retry inside the agent.
3. Baselines (self-contained loops, return a RunRecord):
   - `single_agent_search`: one LLM, all knobs, scalar history only (agent.py's no-readings run,
     generalized to LargeSpace via base + pattern changes).
   - `eat_team_search`: an Evolving-Agent-Teams-style team. Workers act round-robin and see no readings.
     An LLM supervisor sets the team (knob families per worker) at the start and restructures it when
     it stalls (no accept in k proposals).
4. A minimal routing driver (`route_search`), only to smoke-test the role agents until C's `team_search`
   lands. It is not the paper's loop.
5. Tests with `FakeBackend` (no network). Then one smoke run per real backend (sonnet-5-5, haiku-5-5,
   mini-coder-4b), budget 10, on LargeSpace + FastEvaluator. Records go in `results/m5_llm/`.
