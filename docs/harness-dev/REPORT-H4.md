# REPORT H4 (LLM methods on the small space): Agent A, branch h/m5-llm

## 0. Token estimate (written before launching)

Per-call input tokens measured in the M5 smoke runs (`results/m5_llm/`; the CLI reports fresh +
cache-read + cache-write input, all counted):

| model | team (per call) | single (per call) | EAT (per call) | calls per evaluation (team / single / EAT) |
|---|---|---|---|---|
| claude-sonnet-5-5 | 6.6k | 8.1k | 6.9k | 2.0 / 1.6 / 0.8 |
| claude-haiku-5-5 | 3.2k | 5.8k | 4.5k | 2.5 / 1.8 / 0.9 |
| mini-coder-4b | 1.5k | 1.6k | 1.2k | 4.8 / 1.6 / 1.6 |

The per-call cost is set mostly by the CLI's own overhead: about 1k tokens of system context, and two
API turns for structured output. Prompt size matters less. Small-space prompts are 1.6–2.8k characters
against 1.6–5.0k on the large space (measured), so I keep the per-call numbers. Single-agent prompts
grow with history: +0.65k tokens per call at 30 history lines without readings, +2.9k with readings
(×2 turns), which puts single+readings at about 12k per call for Sonnet and 9k for Haiku.

At budget 50, assuming each run spends its full budget:

| | team | single | single + readings | EAT | per seed | seeds | total |
|---|---|---|---|---|---|---|---|
| Sonnet | 0.66M | 0.70M | 0.96M | 0.28M | 2.6M | 3 | 7.8M |
| Haiku | 0.40M | 0.59M | 0.81M | 0.20M | 2.0M | 3 | 6.0M |
| local 4B | 0.36M | 0.13M | 0.18M | 0.10M | 0.8M | 1 | 0.8M |
| **total** | | | | | | | **≈14.6M input** (output ≈ 1.2M) |

Uncertainty: the team loop keeps asking agents round after round on a plateau (capped here at
`max_rounds: 8`, about 24 extra calls), and history grows. So the upper estimate is about 1.5× the
central one, **≈22M**.

**Decision:** the central estimate sits 3% under 15M and the upper estimate is well over it, so I treat
it as over. I'm running **seed 0 only** for every method (Sonnet, Haiku, local), then stopping for the
lead. Seed 0's measured tokens replace this estimate for seeds 1–2.

Configs: `experiments/h/h4_llm.yaml` (Claude methods) and `experiments/h/h4_llm_local.yaml` (local 4B).
Both use B's `hrun.py` with methods given as `harness.llm_agents:h_team|h_single|h_eat`.
