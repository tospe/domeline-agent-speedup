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

Build (`harness/llm_agents.py`), as planned before C's interface landed (see §1 for what changed):
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

## 1. What works

After the plan, C's `harness/team.py` and its `AGENT_IFACE.md` were merged into autopilot (4ed392d).
The lead made C's version canonical. I merged autopilot (taking C's AGENT_IFACE.md on the conflict) and
adapted the code:

| Piece | Status | Check (`tests/test_llm_agents.py`, 14 tests, ~8 s, no network) |
|---|---|---|
| `LLMAgent(role, backend)`: `propose(ctx: team.AgentContext) -> [team.Move]` | works in `team.team_search` unchanged; `llm_team(backend)` builds the three | returns up to 3 ranked moves; corrective retry when the answer is out of scope (asks the model again with the error); ownership is enforced before the loop sees a move; prompt holds only the agent's own knobs, the reading, the routed term, `rule_moves`, the verifier feedback and the round; the loop hands the agent its RunRecord |
| Backends: `ClaudeCLI("claude-sonnet-5-5" / "claude-haiku-5-5")`, `LocalLlama()` (mini-coder-4b-Q8_0), `FakeBackend` | all three used in the smoke runs | CLI and llama-server replies parsed from canned JSON; tokens per model in `RunRecord.tokens` on every call |
| Group moves on LargeSpace | `L*.op`, `La-b.op`, `Ln.op` (plus small-model spellings `L<0>.op`, `L*l*.op`) | 36-layer expansion, ranges, unknown knob / bad value rejected |
| `single_agent_search` (E5: single LLM, no readings; `with_readings=True` for the ablation) | Space and LargeSpace | duplicate / illegal / invalid handling, no readings in any prompt, budget and stop |
| `eat_team_search` (E5: Evolving-Agent-Teams-style supervisor team, no readings, no routing) | LargeSpace | the supervisor sets the team, workers take turns, restructure when every worker passes or after `stall_k` misses; no readings in any prompt |

Interface details beyond C's doc (agent-side choices):
- `LLMAgent.once_per_round` (default on). `team_search` asks an owner once per agenda term, but every
  call already carries the whole reading. So the agent calls the LLM once per (design, round) and
  returns [] for later terms in the same round. Retries with verifier feedback always call again. In a
  dry run without this, the role agents made 40 LLM calls for 4 evaluations.
- Seeds: `seed * 1e6 + call index`. The local backend uses them, at temperature 0. The Claude CLI has
  **no temperature or seed flag**, so Claude runs are not reproducible call by call. Every proposal is
  still in the record.
- `n_in` = fresh + cache-read + cache-write input tokens (the CLI reports all three); `n_out` = output tokens.
- My baselines log outcomes with team.py's names (accepted / slower / infeasible / illegal / duplicate),
  and use `team.point_verifier`.
- `route_search`, my stand-in router, is deleted: `team_search` replaces it.

## 2. Smoke runs (budget 10 each, pinned IDs, seed 0, one run each)

`experiments/m5_llm_smoke.py`: 45 search sessions, LargeSpace (626 knobs) + FastEvaluator, M2 verifier on.
The start is the tied START, 0.040x. Records: `results/m5_llm/<backend>_<method>.json`; summary
`results/m5_llm/smoke.csv`.

| backend (model ID) | method | best speedup | best so far after each evaluation | accepted / illegal / duplicate | LLM calls | tokens in / out | wall |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5-5 | team (C's loop, readings) | 7.46x | .04 .04 .04 .04 .04 .05 1.75 6.52 6.54 7.46 | 6 / 2 / 1 | 20 | 131.7k / 10.2k | 148 s |
| claude-sonnet-5-5 | single, no readings | **9.47x** (stopped itself at 7 evals) | .04 8.51 9.47 9.47 9.47 9.47 9.47 | 2 / 9 / 0 | 16 | 129.6k / 7.5k | 115 s |
| claude-sonnet-5-5 | EAT team, no readings | 3.20x | .04 .04 .51 .53 .53 .53 .53 .53 1.63 3.20 | 5 / 1 / 0 | 8 | 54.8k / 4.8k | 61 s |
| claude-haiku-5-5 | team | 6.07x | .04 .04 .04 .04 1.39 1.63 2.52 3.97 5.05 6.07 | 8 / 1 / 0 | 25 | 79.4k / 28.8k | 196 s |
| claude-haiku-5-5 | single, no readings | 8.11x | .04 .04 7.0 7.0 8.06 8.1 8.1 8.1 8.11 8.11 | 4 / 6 / 0 | 18 | 104.7k / 20.5k | 133 s |
| claude-haiku-5-5 | EAT team | 8.47x | .04 .04 1.84 5.43 7.31 7.34 8.47 8.47 8.47 8.47 | 5 / 3 / 0 | 9 | 40.4k / 9.9k | 73 s |
| mini-coder-4b-Q8_0 | team | 0.10x | (6 small accepts) | 6 / 3 / 2 | 48 | 71.1k / 11.6k | 159 s |
| mini-coder-4b-Q8_0 | single | 0.042x | stopped after 2 evals, invalid answers | 1 / 0 / 0 | 6 | 9.8k / 2.0k | 32 s |
| mini-coder-4b-Q8_0 | EAT team | 0.040x | supervisor stopped after 3 evals | 0 / 1 / 2 | 16 | 19.8k / 5.8k | 79 s |

Reference points: the uncapped optimum is 9.48x (small sweep; M3 shows untying layers adds 0 to 0.28%),
the small-space walk reaches 9.42x in 31 evaluations, and the M3 certificate floor is 10.83x.

What these runs say. They are smoke tests: one seed, budget 10. They show that every path runs
end to end and records tokens; they are not evidence for or against the hypothesis.
- **The single Sonnet agent without readings reached 9.47x in 7 evaluations, then stopped on its own.**
  That is within 0.1% of the uncapped optimum, on the large space, with no readings. It repeats the
  paper's existing small-space observation that a strong LLM finds the uncapped optimum unaided, and it
  agrees with M3's finding that the uncapped large space has no headroom beyond the tied optimum. If E5
  stays uncapped, it will likely show "a single strong LLM is enough". Area/effort caps (where M3 found
  per-layer trade-offs) are where the team could differ.
- The team loop (readings + routing) was the slowest of the Claude methods to leave the start: it spent
  4–6 evaluations near 0.04x. It was not making tiny moves: its accepted moves are whole-model moves
  (36, 108, 253 slots). But each move stays within one role's knobs, and the binding term at START is
  host attention (98% of LLM time). Softmax and lm_head moves were accepted at 0.04x before attention
  moved. The single agent changes many families in one proposal, attention included. So the difference
  here is move scope per proposal, not readings. A coordinated cross-role move at round 0 (C's
  `coordinate="first"`) is the obvious knob to test.
- Illegal proposals are common in the no-readings baselines (Sonnet single: 9 of 15). The rules they
  hit: R2.transfer (embed on the vfu: the token crosses in from the host and the integrator charges no
  transfer for it), R1.sampling (sample on PIM) and R3.capacity (fewer arrays than KV at 35K+ tokens).
  This is E6 material: the verifier is doing real work.
- mini-coder-4b cannot follow the format reliably. Its first run produced pattern spellings that
  echoed the prompt's notation (`L<0>.attn_norm`, `L*l*.attention`); the parser now accepts those, and
  the reruns above use it. The remaining failures are values copied from the display format
  (`"xbar L0-35"`), a nonexistent layer 36, and options the knob doesn't have. Pre-fix local results:
  team 0.040x (0 accepted), single 0.040x (0 valid), EAT 0.59x. A further leniency (strip a trailing
  `" L<a>-<b>"` from values) would likely recover more moves. I did not add it: it is tuning the
  format around one model, and the lead should decide whether the 4B backend gets that help.

## 3. Issues for C / the lead

1. **Evaluations the loop doesn't log as proposals (C).** In the first local `team` run, 10
   evaluations were used with only 4 proposals logged (3 duplicates, 1 slower) and 1 accept (`xbar_scale`
   1.0). So 8 or so evaluations came from somewhere other than the agents' proposals, presumably the
   supervisor's coordinated restructure moves, which `team_search` evaluates without a `note(...)`
   log entry when they are slower. Costs are counted correctly (`ev.used`); the move log is just
   incomplete for E6/E7 analysis.
2. **Decision (lead): help the 4B model or not.** See the last bullet of §2.
3. **E5 framing (lead).** Uncapped, the single Sonnet agent already reaches the optimum in 7 evaluations;
   see §2. Decide whether E5 runs under caps.
4. The Claude CLI can't set temperature or seed. If reproducible LLM runs matter, the API (needs a key
   and the `anthropic` package, neither installed) is the way. With the CLI, use ≥3 seeds and report spread.

## 4. Files touched

- New: `harness/llm_agents.py`, `tests/test_llm_agents.py`, `experiments/m5_llm_smoke.py`,
  `results/m5_llm/*.json` (9 RunRecords) and `results/m5_llm/smoke.csv`.
- Docs: `docs/harness-dev/REPORT-M5-llm.md`. My AGENT_IFACE.md draft is superseded by C's (autopilot's
  copy was taken in the merge).
- No edits to team.py, tools.py, the integrator or any other shared file.

## 5. Test output

`/localhome/tsousape/tools/venv/bin/python -m pytest tests -q` on this branch (autopilot 4ed392d merged in),
run before the parser leniency was added:

```
FAILED tests/test_claims.py::test_registry_matches_paper - AssertionError: ph...
FAILED tests/test_claims.py::test_no_mismatched_numbers - AssertionError: 6 n...
FAILED tests/test_claims.py::test_no_unregistered_result_numbers - AssertionE...
3 failed, 96 passed in 353.12s (0:05:53)
```

The three are the main.tex-vs-registry checks that fail identically on clean autopilot (REPORT-M1).
After the parser leniency: `tests/test_llm_agents.py` 14 passed.
