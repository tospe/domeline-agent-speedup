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

## 1. Seed 0 results (all methods, budget 50, small space, verifier on, sampling rule = host)

Records: `results/h/h4_llm/<method>_s0.json`, `results/h/h4_llm_local/<method>_s0.json`, plus `summary.csv`
in each directory. Gap = target / best − 1. 9.4684x is the fastest legal design of the sweep (h4.yaml's
reference, REPORT-M2); 9.4847x is the second target the lead gave (I take it to be the sweep optimum without
the verifier, but have not checked that).

| method | model | best | gap to 9.4684 | gap to 9.4847 | 1% band at eval | evals used | best at eval | proposals / illegal | LLM calls | tokens in / out | wall | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| team (readings + routing) | sonnet-5-5 | 9.4684x | 0.00% | 0.17% | 9 | 39 | 11 | 96 / 18 | 72 | 330k / 34k | 447 s | no move improves |
| single, no readings | sonnet-5-5 | 9.4684x | 0.00% | 0.17% | 7 | 12 | 7 | 17 / 6 | 18 | 78k / 6k | 93 s | agent stopped |
| single + readings | sonnet-5-5 | 9.4684x | 0.00% | 0.17% | **6** | 15 | 7 | 22 / 8 | 23 | 158k / 9k | 127 s | agent stopped |
| EAT team (no readings) | sonnet-5-5 | 9.4684x | 0.00% | 0.17% | 30 | 50 | 30 | 84 / 14 | 58 | 302k / 24k | 355 s | supervisor stopped |
| team | haiku-5-5 | 9.4684x | 0.00% | 0.17% | 23 | 50 | 38 | 93 / 18 | 87 | 347k / 97k | 677 s | budget |
| single, no readings | haiku-5-5 | 9.4684x | 0.00% | 0.17% | 11 | 19 | 11 | 34 / 16 | 35 | 246k / 47k | 322 s | agent stopped |
| single + readings | haiku-5-5 | 9.4684x | 0.00% | 0.17% | 8 | 15 | 10 | 16 / 2 | 18 | 115k / 23k | 158 s | agent stopped |
| EAT team | haiku-5-5 | 9.4684x | 0.00% | 0.17% | 23 | 47 | 32 | 83 / 10 | 63 | 223k / 72k | 475 s | 10 restructures |
| team | mini-coder-4b | 1.187x | 698% | 699% | — | 44 | 15 | 58 / 5 | 58 | 57k / 10k | 105 s | no move improves |
| single, no readings | mini-coder-4b | 0.040x | — | — | — | 1 | 1 | 1 / 0 | 5 | 5k / 1k | 13 s | invalid answers |
| single + readings | mini-coder-4b | 0.454x | 1986% | 1990% | — | 5 | 5 | 11 / 5 | 20 | 33k / 5k | 50 s | repeated designs |
| EAT team | mini-coder-4b | 0.534x | 1673% | 1677% | — | 8 | 6 | 44 / 2 | 78 | 61k / 12k | 165 s | 10 restructures |

The Claude runs went four in parallel, so their wall times include CLI contention; local runs went one at a
time. Every Claude run's best design is the same legal optimum (9.4684x).

Non-LLM methods on the same space from B's `results/h/h4/summary.csv` (budget 100, 3 seeds; deterministic
methods give the same result for every seed). 1% band at eval: walk 20, deterministic team 15, TPE
99/50/47, hill 71/21/36, SA 42/—/—, random never. Best: walk and deterministic team 9.457x; TPE and hill
9.468x in every seed.

## 2. What seed 0 says (one seed; not yet a result)

- **Every Claude method reaches the legal optimum on the small space** (best = 9.4684x exactly in all
  8 runs), 0.17% below 9.4847x. Since every run stops at the same design, I read the 0.17% as the
  verifier's price, not as a search gap; I have not checked which rule the 9.4847x design breaks. On this
  space the LLM methods differ only in cost.
- **The cheapest and fastest are the single agents.** The single agent with readings reaches the 1% band
  at eval 6 (Sonnet) and 8 (Haiku); without readings, at 7 and 11. Both are ahead of the walk (20), the
  deterministic team (15) and TPE (47–99), at 78–246k input tokens.
- **Readings help the single agent a little:** 1 to 3 evaluations earlier, and far fewer illegal
  proposals for Haiku (2 vs 16). It is one seed per cell, so this is a direction, not a finding.
- **The LLM team (readings + routing) is slower and costlier than the single agent:** band at eval 9
  (Sonnet) and 23 (Haiku), 330–347k input tokens, 72–87 LLM calls. The same mechanism as in the smoke
  runs: each agent moves only its own knobs, so the first big step (attention, MVMs and vector ops
  together) takes several accepted moves. The EAT team (no readings, no routing) is about as slow
  (band at 23–30). So on the small space, the team structure costs evaluations and tokens compared with
  one agent. This is the honest-reporting case the brief anticipates. The brief's hypothesis is about
  the large, coupled space under caps; uncapped, M3 found that space has no headroom beyond the tied
  optimum.
- **Illegal proposals:** every Claude method hits R1.sampling + R2.transfer (sample or embed on PIM, always
  both together: the `edge` knob moves both ops) and some R3.capacity. The verifier rejects them at no
  evaluation cost. For the team runs the counts are in the record's `extra.rejected`; `hrun`'s
  `illegal_rules` column is empty for them because team_search doesn't report rules per move.
- **mini-coder-4b cannot drive the harness** (lead's decision: no further format help, reported as is).
  Best 1.19x (team), and none of its runs reach the band. Its failure modes: invalid values copied from
  the display format, nonexistent layers, repeated designs. The workload's own 4B model is not a usable
  design agent here.

## 3. Tokens: measured vs the estimate, and seeds 1–2

| | estimated, seed 0 | measured, seed 0 |
|---|---|---|
| Sonnet (4 methods) | 2.6M | **0.87M** in / 0.07M out |
| Haiku (4 methods) | 2.0M | **0.93M** in / 0.24M out |
| local 4B (4 methods) | 0.8M | **0.16M** in / 0.03M out |
| total | 5.4M | **1.96M** in / 0.34M out |

The estimate was about 3x high, because the single agents stop early: they converge and say so (12–19
evaluations, not 50). On this measurement, **seeds 1–2 for Sonnet and Haiku would cost ≈3.6M input
tokens** (≈ 2 x 1.80M), plus ≈0.6M output, about 20 min wall at 4 workers. Total for H4 with 3 seeds:
≈5.6M input, well under 15M. I stopped here as instructed (the pre-launch estimate was over the
threshold); the command for the rest is:

    python experiments/hrun.py experiments/h/h4_llm.yaml --seeds 1,2 --workers 4

## 4. Changes in this step

- `harness/llm_agents.py`: hrun entry points `h_team`, `h_single`, `h_eat` (`backend: sonnet | haiku | local`).
  `LLMAgent` now drops an invalid move and keeps the valid ones in the same answer. Before, one
  out-of-scope move discarded the whole answer and forced a retry. A dry run through hrun caught this.
  Test added: 15 tests, all pass.
- New: `experiments/h/h4_llm.yaml`, `experiments/h/h4_llm_local.yaml`, `results/h/h4_llm/`, `results/h/h4_llm_local/`.
- No changes to hrun.py, team.py or any other shared file.

## 5. Notes

1. The record shows the team loop spending evaluations outside agent proposals, at the end of runs
   (shrink steps such as `xbar_scale` and `adc_per_ima`, logged as accepts) and on supervisor
   restructures. That's the same issue as in REPORT-M5-llm §3.1, for C.
2. On the small space every Claude method ties at the optimum, so H4 separates methods only by cost
   (evaluations to the band, tokens). Seeds 1–2 will say whether "single + readings fastest" holds.
