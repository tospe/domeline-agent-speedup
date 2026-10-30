# Claims report (Oct 29, 2026)

Every number in `/localhome/tsousape/Domeline/main.tex` checked against the results files. The pipeline:

- `experiments/make_numbers.py` computes 220+ numbers from the primary results. It writes `paper/numbers.tex` (`\newcommand{\walkSpeedup}{9.42}` and so on, printed as the paper prints them) and `paper/numbers.json` (value, text, source).
- `paper/claims.yaml` registers each claim: a phrase copied from the paper, and for each of its numbers the macro it must equal, a bound (`within 0.6%`), a citation (unsourced) or a setup condition.
- `tests/test_claims.py` fails on any of these:
  - a mismatch;
  - a result-like number (×, %, mm², ms, K tokens, evaluation and design counts) that is not registered;
  - a registered phrase that is no longer in the paper;
  - a stale `numbers.json`;
  - a reading the paper names that `harness/evaluate.py` does not compute. `headroom` and *D* (data held in PIM) are pending and only warn.
- `python experiments/claims.py --report docs/claims_report.md` regenerates the section below the marker.

Once `main.tex` does `\input{numbers}` and uses the macros, a number can no longer drift from its source. The registry stays the check for the phrasing around each one.

## Summary

Of the 264 numbers checked:

| Status | Count |
|---|---|
| Matched | 224 |
| Mismatched | 6 |
| Unsourced | 12 |
| Setup | 22 |
| Unregistered | 0 |

The unsourced numbers are citations (VIPER, yuan2026, ISAAC, the 3080's 10 GB), one log-only number (0.04× without the repair) and the 4-bit-area-factor table, which no script produces. 18 registered claims also carry a phrasing or scope flag. Readings: binding term and Amdahl bound are computed; headroom and *D* are pending.

## Mismatches, in order of importance

1. **l.309, "7.5–28 ms on the 3080"**: the 3080 decodes at 27.4 ms/token at 64K with int8 KV (`g1.jsonl`). Use "7.5–27 ms". Fixed in `results.md` and `paper.md` as well.
2. **l.284, "the walk reproduces the exact staircase of the sweep: 0.59, 3.08, 8.00 and 9.42×"**: at 1 month the walk returns 3.07× (`effort_main.json`); 3.08× is the sweep's best. The walk applies the smallest-design-within-1% rule. Say "matches the sweep's staircase within 0.2%: 0.59, 3.07, 8.00 and 9.42×", or keep 3.08 and attribute the numbers to the sweep.
3. **l.311, "the roof stays at 50–67% of the Amdahl sheet"**: over 1K–64K it is 50.6–67.5%, which prints as "51–68%". Or say "about half to two thirds".
4. **l.37, abstract "spend 90% of their time decoding"**: all 100 sessions give 91.0%, as §1, §2 and Fig. 1 say; 90.4% is the frozen 89. Use 91%.
5. **l.248, "tools take 56% of workload time"**: 227.0 / 401.7 s = 56.5%, which prints as 57%.

(Item 3 accounts for two of the six mismatched numbers.)

## Scope and phrasing problems (numbers right, sentence wrong)

These are the flags in `claims.yaml`. The ones that matter most:

- **Area front, l.322.** Every front point quoted (9.47× at 5,833 mm², 9.00× at 3,692, 7.19× at 2,872, 5.13× at 2,461) also switches to the **on-die link** (+9 months of effort). With the headline's CXL link, the same areas give 9.42, 8.95, 7.16 and 5.12×, and that is what the walk returns under the caps. Either add "and an on-die link" or quote the CXL numbers.
- **"The walk one day", l.288.** The walk runs serially. At 1–3 h per design, its 30–50 evaluations take 1.2–6.2 days; 20 evaluations at 1 h is 0.8 days. Say "1–6 days".
- **Conclusion, l.339, "it stays at the exact optimum where black-box search fails"**: it is within 0.6%, and TPE matches at some caps. Say "within 0.6% of the exact optimum, where black-box search misses by up to 14×".
- **Abstract, "On 100 SWE-bench sessions … 9.4× faster"**: the 9.4× is on the 45 search sessions.
- **"ten area caps … all eight caps where a 64K-context design fits", l.284**: both counts include the uncapped run (no cap plus 9 caps; 7 finite caps fit).
- **"within 0.3% of the exact per-turn replay", l.129**: over the whole workload the gap is 0.01%; the worst single session is 0.52% (headline design, search set).
- **"within 2–3%" (GPU replay), l.187**: the per-session error is −0.4 to −2.9% (aggregate −1.7%, 100 sessions). Say "within 3%" as §1 does.
- **"TPE needs 2–3× more evaluations"**: 41, 42 and 69 against 20, which is 2.0–3.5×.
- **"Every searcher gets 100 evaluations"**: the area study gives 150.

## Reconciled numbers (plan items 13–15, 17)

### 13. VRAM wall: 11% or 8%

Both numbers are correct; they count different sets. A session "hit the wall" if its fp16-KV run ended in ContextExceeded at the trace server's 32K context (set by the 3080's 32–36K fp16 wall, `g1_status.tsv`). It was then re-run with int8 KV (`results/traces/swebench_kv8`).

| Set | Hit the wall | Share |
|---|---|---|
| All 100 traced sessions (Fig. 1, `trace_profile`) | 11 | **11%** |
| Frozen workload, 89 (search + held-out; `paper.md`'s 8%) | 7 | 7.9% |
| Late held-out, 11 | 4 | 36% |

**Recommendation:** use 11%, because Fig. 1 and every workload-characterization number (91/4/5%, 64K max) use all 100. The paper already does this (abstract, §1, Fig. 1 caption). Suggested phrasing for §2 or the Fig. 1 caption:

> 11 of the 100 sessions (7 of the 89 search and held-out sessions, 4 of the 11 late ones) outgrow the 3080's 32K fp16-KV context.

`paper.md` and `results.md` are now aligned to this.

One caveat for whoever words it. The wall sessions' contexts in the trace profile come from their int8 re-runs, which take different trajectories, so three of them end below 32K (17.8K, 19.4K, 29.5K). Their fp16 runs did reach 32K (`results/traces/swebench/<name>/trace.jsonl`). So "11% ran out of VRAM" is measured. Do not infer it from the final-context CDF, which crosses 32K for fewer sessions.

### 14. The 14-knob scope

The 575,400-design sweep (`sweep_main.tsv`, Oct 22) and the unconstrained and effort-cap searches (`main.json`, `effort5*.json`: 14-knob points) predate the three area knobs. Every one of those designs uses ISAAC's tile: eDRAM buffer, 8 ADCs per IMA, 2-bit cells.

With the area knobs the space has **6,749,832** distinct designs. 561,312 of the 575,400 use crossbars, and each takes 2 × 3 × 2 = 12 area variants. That is 12× more, about 3.2 years at the sweep's rate; it was not swept. The current `sweep_all.py` would enumerate the 17-knob space, so it no longer reproduces the 575,400 sweep without fixing the area knobs. Note this in the README.

In that larger space the walk's design is not the smallest within 1% of the optimum. Its lean-tile variant is the same speed at 5,833 mm².

Scope sentence for the Design-space paragraph (§4), replacing "After removing knobs that are inactive under a mapping, there are 575,400 distinct designs, all with ISAAC's tile":

> With ISAAC's tile fixed, the 14 knobs give 575,400 distinct designs once knobs that are inactive under a mapping are removed. We evaluate all of them, and the unconstrained and effort-capped searches run in this space. For the area study we add three knobs from ISAAC's tile breakdown (…); they multiply the space to 6.7M designs, so there we compare against the exact front of the 3,456-design crossbar subspace instead.

And in §5.2 keep "The walk's design is the smallest of these 96" with its parenthesis. The intro's "575,400 distinct designs, before area knobs" is already correct.

### 15. Area-cap claims (`area_main.json`, budget 150, TPE seed 0)

Reference: the exact front of the 3,456-design crossbar subspace (8 knobs: crossbar variant, scale, KV bits, link, tile, ADCs per IMA, cell bits, nonlinear units; the rest fixed at the headline design).

| Cap (mm²) | Front | Walk | Walk gap | TPE | Front / TPE |
|---|---|---|---|---|---|
| none | 9.47× | 9.42× (31 evals) | 0.55% | 9.48× | 1.00× |
| 8,000 | 9.47× | 9.42× (38) | 0.55% | 9.01× | 1.05× |
| 6,000 | 9.47× | 9.42× (39) | 0.55% | 9.48× | 1.00× |
| 5,000 | 9.00× | 8.95× (42) | 0.56% | 7.79× | 1.16× |
| 4,000 | 9.00× | 8.95× (47) | 0.52% | 0.63× | **14.3×** |
| 3,000 | 7.19× | 7.16× (50) | 0.42% | 1.97× | 3.65× |
| 2,500 | 5.13× | 5.12× (55) | 0.30% | 0.63× | 8.16× |
| 2,000 | 1.78× | 1.78× (90) | 0.10% | 0.63× | 2.83× |
| 1,500, 1,000 | nothing fits | none | | infeasible | |

Exact numbers:

- **"0.6%"**: the walk's worst gap is 0.56%, over the 8 settings where a design fits (uncapped plus 7 finite caps).
- **"Up to 14×"**: 14.3×, at the 4,000 mm² cap only.
- **"1.2–14×"**: 1.16–14.3× at caps ≤ 5,000 mm² (one TPE seed).

Uncapped and at 6,000 mm², TPE beats the "exact front" (9.48 vs 9.47×) by leaving the subspace. So the front is exact only within the subspace. The walk's 0.5% gap is its smallest-design rule: it keeps the CXL link where the front switches to the on-die link.

Suggested phrasing for §5.2:

> Under eight area settings where a 64K-context design fits (no cap and caps from 8,000 to 2,000 mm²), the walk lands within 0.6% of the exact area–speed front of the 3,456-design crossbar subspace, in 31–90 evaluations. TPE (150 evaluations) matches it without a cap and at 6,000 mm². At 8,000 mm² it returns 9.01× where 9.47× is available, and at 5,000 mm² and below it misses by 1.2–14×.

And for the abstract:

> …within 0.6% of the exact optimum under effort caps and of the exact area–speed front under area caps, where Bayesian optimization misses by up to 14×.

### 17. GPU vendor peaks

All values in `experiments/gpu_scale.py` are confirmed; none changed. VERIFY is removed and the sources are cited in the file. `gpu_scale.json` and the paper's 7.1× / 2.5× stand.

| GPU | Memory bandwidth | FP16 tensor, fp32 accumulate (dense / sparse) | FP16 tensor, fp16 accumulate | Source |
|---|---|---|---|---|
| RTX 3080 | 760 GB/s | 59.5 / 119 TF | 119 / 238 TF | [GA102 whitepaper, Table 2](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf) |
| RTX 4090 | 1,008 GB/s | 165.2 / 330.4 TF | 330.3 / 660.6 TF | [Ada whitepaper, Table 2](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf) |
| H100 SXM | 3.35 TB/s | 989 TF dense (1,979 "with sparsity") | same rate | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) |

GeForce parts halve the tensor rate with fp32 accumulate; data-center parts do not. If llama.cpp's prefill runs at the fp16-accumulate or int8 rate (MMQ for Q8_0: 238 / 660.6 / 1,979 dense TOPS), the 3080 and 4090 double and the H100 does not. The H100's prefill ratio then falls from 16.6× to 8.3×, and the design's lead over the H100 rises from 2.52× to 2.54× on the search set (2.54 → 2.57 held-out). So the column used is the one generous to the H100, which is how the paper describes it.

The GA102 die (628.4 mm², same whitepaper) confirms "about 15× the 3080's die" (9,729 / 628.4 = 15.5).

## Generated

<!-- BEGIN GENERATED (experiments/claims.py) -->
Generated by `python experiments/claims.py --report docs/claims_report.md` from `/localhome/tsousape/Domeline/main.tex`, `paper/claims.yaml` and `paper/numbers.json`.

| Status | Numbers |
|---|---|
| matched | 224 |
| mismatched | 6 |
| unsourced | 12 |
| setup | 22 |
| unregistered | 0 |

### Mismatched

| Line | Paper | Computed | In context | Note |
|---|---|---|---|---|
| 37 | 90 | \shareDecodePct=91 | `maketitle \begin{abstract} Agentic sessions spend 90\% of their time decoding one token at a` | all 100 sessions: 91.0%; 90.4% is the frozen 89 (shareDecodeFrozenPct) |
| 248 | 56 | \toolShareAtWalkPct=57 | `gs also say when to stop: at 9.42\X{}, tools take 56\% of workload time, and the Amdahl boun` |  |
| 284 | 3.08 | \stairSweepOne=3.08, \stairWalkOne=3.07 | `eproduces the exact staircase of the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 mo` |  |
| 309 | 28 | \gpuDecodeSixtyFourKMs=27 | `s at 4K context and 0.77\,ms at 64K, against 7.5--28\,ms on the 3080 (Fig.~\ref{fig:headline` |  |
| 311 | 50 | \roofAmdahlLoPct=51 | `ssbar reads color the roof, and the roof stays at 50--67\% of the Amdahl sheet; the 3080 sit` |  |
| 311 | 67 | \roofAmdahlHiPct=68 | `r reads color the roof, and the roof stays at 50--67\% of the Amdahl sheet; the 3080 sits 7-` |  |

### Phrasing and scope flags

| Line | Claim | Flag |
|---|---|---|
| 37 | `Agentic sessions spend 90\% of their time decoding` | Use 91% (all 100 traced sessions, as Sec. 1 and Fig. 1 do), or say "about 90%". |
| 37 | `On 100 SWE-bench sessions of a coding agent, the harness finds in about 20 evaluations` | The 9.4x is measured on the 45 search sessions (9.61x on the 44 held-out), not on all 100; say "On SWE-bench sessions of a coding agent" or "on 45 search sessions". |
| 37 | `runs the sessions 9.4\X{} faster than an RTX 3080. The design is within 1\% of the best of` | 575,400 is the 14-knob space (area knobs fixed at ISAAC's tile); see the scope sentence in docs/claims_report.md. |
| 37 | `Under area and effort caps the harness stays within 0.6\% of the exact optimum, where Baye` | "exact optimum" is, under area caps, the exact front of the 3,456-design crossbar subspace (not the full space); the 14x is TPE at the 4,000 mm2 cap only. Suggested: "within 0.6% of the exact optimum under effort caps and of the exact area-speed front under area caps, where TPE misses by up to 14x". |
| 59 | `of a 575,400-design space: the walk lands within 1\% unconstrained and within 0.6\% of the` | Name the front: "of the exact front of the 3,456-design crossbar subspace". |
| 129 | `it stays within 0.3\% of the exact per-turn replay` | True for the workload total (0.01%), not per session (worst 0.52% on the headline design). Say "within 0.3% over the workload (0.5% for any one session)". |
| 187 | `predicts its measured decode time within 2--3\%` | The 3080 curve under-predicts each session by 0.4-2.9% (aggregate -1.7%, 100 sessions); "within 2-3%" reads as a band. Say "within 3%" as in Sec. 1. |
| 195 | `there are 575,400 distinct designs, all with ISAAC's tile` | Scope: with the 3 area knobs the space has 6,749,832 distinct designs (17 knobs). See docs/claims_report.md for the sentence. |
| 228 | `Every searcher gets 100 evaluations` | The area study gives 150 (stated in Sec. 5.2); say "100 evaluations (150 in the area study)". |
| 280 | `The walk's design is the smallest of these 96` | Holds in the 14-knob space only; with the area knobs, the lean-tile variant of the same design (5,833 mm2, 9.42x) is smaller. The parenthesis says so; keep it. |
| 282 | `TPE needs 2--3\X{} more evaluations unconstrained and fails on 2 of 3 seeds under either c` | TPE's first design within 1% comes at evals 41, 42 and 69 against the walk's 20: 2.0-3.5x. "2-3x" rounds 3.45 down; "2-3.5x" is exact. |
| 284 | `We run the walk and TPE with 150 evaluations under ten area caps` | "ten area caps" counts the uncapped run: the study has no cap plus nine caps (8,000 to 1,000 mm2), and "all eight caps where a design fits" also counts the uncapped run (7 finite caps fit). Say "under no cap and nine area caps ... at all eight settings where a 64K-context design fits". |
| 284 | `At all eight caps where a 64K-context design fits, the walk lands within 0.6\% of the exac` | The front is exact over the 3,456-design crossbar subspace only (the rest fixed at the headline design); TPE beats it at 6,000 mm2 and uncapped (9.48x vs 9.47x) by leaving the subspace. 31-90 counts all evaluations used; the design returned is found at evals 28-80. |
| 284 | `Under effort caps the walk reproduces the exact staircase of the sweep: 0.59, 3.08, 8.00 a` | At 1 month the walk returns 3.07x (smallest design within 1%) where the sweep's best is 3.08x; say "matches the sweep's staircase within 0.2%: 0.59, 3.07, 8.00 and 9.42x". |
| 288 | `With a gem5-class evaluator at 1--3\,h per design, the sweep would take 66--200 CPU-years ` | "the walk one day" holds only at 1 h per design for its first 20 evaluations; 30-50 serial evaluations at 1-3 h take 1.2-6.2 days. Say "and the walk 1-6 days". |
| 311 | `the roof stays at 50--67\% of the Amdahl sheet; the 3080 sits 7--11\X{} lower before its V` | Over 1K-64K the roof is at 50.6-67.5% of the Amdahl sheet: "51-68%" at the printed precision (or "about 50-70%"). |
| 322 | `The exact front trades area for speed: 9.47\X{} at 5,833\,mm$^2$ with a lean tile, 9.00\X{` | Every one of these front points also uses the on-die link (+9 months of effort); with the headline's CXL link the same areas give 9.42, 8.95, 7.16 and 5.12x, which is what the walk returns under the caps. Say "with a lean tile and an on-die link" or quote the CXL numbers. The smallest design that fits is 1,612 mm2. |
| 339 | `Under area and effort constraints it stays at the exact optimum where black-box search fai` | Overclaim: it stays within 0.6% (the 1% smallest-design rule), not at the optimum, and TPE does reach it at some caps (6,000 mm2, 5 months on 1 of 3 seeds). Say "within 0.6% of the exact optimum, where black-box search misses by up to 14x". |

### Unsourced (no results file computes them)

| Line | Paper | Source given | In context |
|---|---|---|---|
| 51 | 10 | external: VIPER paper [viper] | ` end-to-end latency of an offloaded kernel within 10\%. A closed form does more than predict` |
| 75 | 10 | external: NVIDIA RTX 3080 spec (10 GB GDDR6X) | `rved by llama.cpp~\cite{llamacpp} on an RTX 3080 (10\,GB, 760\,GB/s). Each turn logs the con` |
| 75 | 91 | external: [yuan2026agentic] | `hed characterizations of larger agents (decode is 91--99\% of LLM time~\cite{yuan2026agentic` |
| 75 | 99 | external: [yuan2026agentic] | `characterizations of larger agents (decode is 91--99\% of LLM time~\cite{yuan2026agentic}). ` |
| 93 | 10 | external: VIPER paper [viper] | ` and $C$ the capacity. It predicts latency within 10\% in under a minute, but for one kernel` |
| 187 | 2.42 | external: earlier VIPER UPMEM study (docs/harness.md); no results file here | `d venue}. It reproduces VIPER's UPMEM validation: 2.42\X{} predicted against 2.46\X{} measured` |
| 187 | 2.46 | external: earlier VIPER UPMEM study (docs/harness.md); no results file here | `ER's UPMEM validation: 2.42\X{} predicted against 2.46\X{} measured for a Llama2 layer. Replay` |
| 226 | 0.04 | log only: docs/autopilot.md 19:45 (first iso-area run, overwritten by area_main.json) | ` int8 KV). Without this repair, the walk returned 0.04\X{} under every cap. Effort caps prune ` |
| 322 | 32 | external: ISAAC [isaac], 32 nm | `.} The headline design keeps ISAAC's tile: at its 32\,nm density it takes 9,729\,mm$^2$, abo` |
| 322 | 1 | external: ISAAC Table I (arrays 0.0024 of the 0.372 mm2 tile); not computed by any script | `about 15\X{} the 3080's die. The arrays are under 1\% of a tile; ADCs and the eDRAM buffer ` |
| 324 | 25 | docs/results.md table (4-b area factor 1.0/1.27/1.5/2.0); no script or results file produces it | `ea factor shifts the area front by one step per $+25\%$. \section{Discussion and Limitations` |
| 330 | 32 | external: ISAAC (ISCA 2016) | `the result. The ISAAC parameters are from 2016 at 32\,nm; newer ReRAM macros are denser, so ` |

### Unregistered result-like numbers

None.

### Domeline readings named by the paper

| Reading | Status | Why |
|---|---|---|
| Amdahl bound | ok | computed by Evaluator.amdahl_bound_s, Eval.tools_s |
| binding term | ok | computed by Eval.terms |
| data held in PIM | pending | not in harness/evaluate.py yet (expected one of ['Eval.data_in_pim', 'Eval.data_in_pim_bytes', 'Eval.pim_bytes', 'Eval.footprint_bytes', 'Eval.resident_bytes', 'Eval.d_bytes', 'Eval.D']); the D axis (plan item 1); another session is adding it |
| headroom | pending | not in harness/evaluate.py yet (expected one of ['Eval.headroom', 'Eval.headroom_s', 'Eval.headroom_ctx', 'Evaluator.headroom']); another session is adding it to harness/evaluate.py (plan item 4) |

### Every number, in paper order

| Line | Paper | Status | Computed | Source | In context |
|---|---|---|---|---|---|
| 37 | 90 | mismatched | \shareDecodePct=91 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `maketitle \begin{abstract} Agentic sessions spend 90\% of their time decoding one token at a` |
| 37 | 11 | matched | \vramWallPct=11 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `und by memory, and a consumer GPU runs out of it: 11\% of the coding-agent sessions we trace` |
| 37 | 100 | matched | \nSessionsTraced=100 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `oom to every other term, and the Amdahl bound. On 100 SWE-bench sessions of a coding agent, t` |
| 37 | 20 | matched | \walkFirstWithin=20 | results/search/sweep_main_summary.json | `ons of a coding agent, the harness finds in about 20 evaluations a ReRAM coprocessor that ru` |
| 37 | 9.4 | matched | \walkSpeedupShort=9.4 | results/search/main.json (walk best) | `ations a ReRAM coprocessor that runs the sessions 9.4\X{} faster than an RTX 3080. The design` |
| 37 | 1 | matched | \walkGapPct=0.7 | results/search/sweep_main_summary.json | `X{} faster than an RTX 3080. The design is within 1\% of the best of 575,400 candidates, an` |
| 37 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `RTX 3080. The design is within 1\% of the best of 575,400 candidates, and every design decision i` |
| 37 | 0.6 | matched | \walkAreaGapMaxPct=0.6, \walkGapEffortFivePct=0.6 | results/search/area_main.json; results/search/sweep_main_summary.json | `der area and effort caps the harness stays within 0.6\% of the exact optimum, where Bayesian ` |
| 37 | 14 | matched | \tpeAreaMissMaxX=14 | results/search/area_main.json | `imum, where Bayesian optimization misses by up to 14\X. \end{abstract} \begin{IEEEkeywords} ` |
| 47 | 91 | matched | \shareDecodePct=91 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | ` coding agent on SWE-bench Verified, decode takes 91\% of session time on an RTX 3080, and t` |
| 47 | 64K | matched | \maxCtxK=64 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `ime on an RTX 3080, and the final context reaches 64K tokens (Fig.~\ref{fig:trace}). Batch-1 ` |
| 47 | 11 | matched | \vramWallPct=11 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `ts footprint grows with the session. On the 3080, 11\% of the sessions run out of VRAM; past` |
| 47 | 26 | matched | \gpuCliffX=26 | results/gpu/g2.jsonl / results/gpu/g1.jsonl | `ssions run out of VRAM; past that wall, decode is 26\X{} slower. Processing-in-memory (PIM) ` |
| 49 | 14 | matched | \nKnobsBase=14 | harness/space.py KNOBS minus the 3 area knobs | `e, precision, and link to the host. Our space has 14 knobs and 575,400 distinct designs, bef` |
| 49 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | ` and link to the host. Our space has 14 knobs and 575,400 distinct designs, before area knobs. De` |
| 51 | 10 | unsourced |  | external: VIPER paper [viper] | ` end-to-end latency of an offloaded kernel within 10\%. A closed form does more than predict` |
| 55 | 9.4 | matched | \walkSpeedupShort=9.4 | results/search/main.json (walk best) | ` finds a ReRAM coprocessor that runs its sessions 9.4\X{} faster than an RTX 3080. It reaches` |
| 55 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | `faster than an RTX 3080. It reaches the optimum's 1\% band in 20 evaluations, where an exha` |
| 55 | 20 | matched | \walkFirstWithin=20 | results/search/sweep_main_summary.json | `an RTX 3080. It reaches the optimum's 1\% band in 20 evaluations, where an exhaustive sweep ` |
| 55 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `n 20 evaluations, where an exhaustive sweep needs 575,400. We make four contributions: \begin{ite` |
| 57 | 3 | matched | \gpuReplayErrMaxAbsPct=2.9 | harness GpuBaseline vs measured, worst session | `s GPU replay predicts measured decode time within 3\%. \item \textbf{A model-guided harness` |
| 59 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `f the search against the exhaustive optimum} of a 575,400-design space: the walk lands within 1\%` |
| 59 | 1 | matched | \walkGapPct=0.7 | results/search/sweep_main_summary.json | ` of a 575,400-design space: the walk lands within 1\% unconstrained and within 0.6\% of the` |
| 59 | 0.6 | matched | \walkAreaGapMaxPct=0.6 | results/search/area_main.json | `he walk lands within 1\% unconstrained and within 0.6\% of the exact front under every feasib` |
| 60 | 9.42 | matched | \walkSpeedup=9.42 | results/search/main.json (walk best) | `em \textbf{A PIM coprocessor for a coding agent}: 9.42\X{} faster than the 3080 on the search ` |
| 60 | 9.61 | matched | \walkHeldout=9.61 | results/search/main.json (walk best, held-out 44) | `\X{} faster than the 3080 on the search sessions, 9.61\X{} on held-out sessions, and 2.5\X{} f` |
| 60 | 2.5 | matched | \vsHHundred=2.5 | results/search/gpu_scale.json | `arch sessions, 9.61\X{} on held-out sessions, and 2.5\X{} faster than an H100 when scaled; de` |
| 60 | 0.40 | matched | \decodeFourKMs=0.40 | harness integrator, headline design of results/search/main.json | `X{} faster than an H100 when scaled; decode takes 0.40--0.77\,ms per token from 4K to 64K cont` |
| 60 | 0.77 | matched | \decodeSixtyFourKMs=0.77 | harness integrator, headline design of results/search/main.json | `ster than an H100 when scaled; decode takes 0.40--0.77\,ms per token from 4K to 64K context (\` |
| 60 | 4K | setup |  |  | `caled; decode takes 0.40--0.77\,ms per token from 4K to 64K context (\S\ref{sec:design}). \e` |
| 60 | 64K | setup |  |  | ` decode takes 0.40--0.77\,ms per token from 4K to 64K context (\S\ref{sec:design}). \end{item` |
| 75 | 36 | matched | \nLayers=36 | mini-coder-4b config.json | `b~\cite{minicoder} (Qwen3-4B layout~\cite{qwen3}: 36 layers, grouped-query attention, Q8\_0 ` |
| 75 | 100 | matched | \nSessionsTraced=100 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `ayers, grouped-query attention, Q8\_0 weights) on 100 repository-stratified SWE-bench Verifie` |
| 75 | 10 | unsourced |  | external: NVIDIA RTX 3080 spec (10 GB GDDR6X) | `rved by llama.cpp~\cite{llamacpp} on an RTX 3080 (10\,GB, 760\,GB/s). Each turn logs the con` |
| 75 | 760 | matched | \gpuBandwidthThreeEighty=760 | results/search/gpu_scale.json (vendor peak) | `llama.cpp~\cite{llamacpp} on an RTX 3080 (10\,GB, 760\,GB/s). Each turn logs the context leng` |
| 75 | 91 | matched | \shareDecodePct=91 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `ig.~\ref{fig:trace} shows where time goes: decode 91\%, prefill 4\%, tools 5\%, consistent w` |
| 75 | 4 | matched | \sharePrefillPct=4 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `race} shows where time goes: decode 91\%, prefill 4\%, tools 5\%, consistent with published` |
| 75 | 5 | matched | \shareToolPct=5 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | ` where time goes: decode 91\%, prefill 4\%, tools 5\%, consistent with published characteri` |
| 75 | 91 | unsourced |  | external: [yuan2026agentic] | `hed characterizations of larger agents (decode is 91--99\% of LLM time~\cite{yuan2026agentic` |
| 75 | 99 | unsourced |  | external: [yuan2026agentic] | `characterizations of larger agents (decode is 91--99\% of LLM time~\cite{yuan2026agentic}). ` |
| 75 | 32 | matched | \wallFpSixteenLoK=32 | results/gpu/g1_status.tsv (Q8_0) | `\cite{yuan2026agentic}). The 3080's wall falls at 32--36K tokens with an fp16 KV cache and 6` |
| 75 | 36K | matched | \wallFpSixteenHiK=36 | results/gpu/g1_status.tsv (Q8_0) | `e{yuan2026agentic}). The 3080's wall falls at 32--36K tokens with an fp16 KV cache and 64--80` |
| 75 | 64 | matched | \wallIntEightLoK=64 | results/gpu/g1_status.tsv (Q8_0) | `falls at 32--36K tokens with an fp16 KV cache and 64--80K with int8; with the KV cache in ho` |
| 75 | 80K | matched | \wallIntEightHiK=80 | results/gpu/g1_status.tsv (Q8_0) | `s at 32--36K tokens with an fp16 KV cache and 64--80K with int8; with the KV cache in host me` |
| 75 | 14 | matched | \gpuDecodeThirtyTwoKMs=14 | results/gpu/g1.jsonl | `th the KV cache in host memory, decode slows from 14\,ms to 373\,ms per token at 32K. The wa` |
| 75 | 373 | matched | \gpuOffloadThirtyTwoKMs=373 | results/gpu/g2.jsonl | `cache in host memory, decode slows from 14\,ms to 373\,ms per token at 32K. The wall is a cli` |
| 75 | 32K | setup |  |  | ` decode slows from 14\,ms to 373\,ms per token at 32K. The wall is a cliff, not a slope. \beg` |
| 80 | 100 | matched | \nSessionsTraced=100 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `\columnwidth]{figures/trace_profile.pdf} \caption{100 coding-agent sessions on an RTX 3080. L` |
| 80 | 11 | matched | \vramWallPct=11 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | ` decode, prefill and tools. Right: final context; 11\% of sessions cross the 3080's fp16-KV ` |
| 93 | 10 | unsourced |  | external: VIPER paper [viper] | ` and $C$ the capacity. It predicts latency within 10\% in under a minute, but for one kernel` |
| 106 | 128 | matched | \xbarRows=128 | lib/tech/pim.yaml xbar_isaac | `y holds ISAAC-style ReRAM crossbars~\cite{isaac} (128$\times$128 arrays, 2-bit cells, 100\,ns` |
| 106 | 128 | matched | \xbarRows=128 | lib/tech/pim.yaml xbar_isaac | `AC-style ReRAM crossbars~\cite{isaac} (128$\times$128 arrays, 2-bit cells, 100\,ns read), a v` |
| 106 | 2 | matched | \xbarCellBits=2 | lib/tech/pim.yaml xbar_isaac | `AM crossbars~\cite{isaac} (128$\times$128 arrays, 2-bit cells, 100\,ns read), a variant wit` |
| 106 | 100 | matched | \xbarReadNs=100 | lib/tech/pim.yaml xbar_isaac | `\cite{isaac} (128$\times$128 arrays, 2-bit cells, 100\,ns read), a variant with transposable ` |
| 129 | 0.3 | matched | \replayErrWorkloadPct=0.01 | harness integrator, headline design of results/search/main.json | `ad of sessions take milliseconds; it stays within 0.3\% of the exact per-turn replay. The spe` |
| 187 | 2.42 | unsourced |  | external: earlier VIPER UPMEM study (docs/harness.md); no results file here | `d venue}. It reproduces VIPER's UPMEM validation: 2.42\X{} predicted against 2.46\X{} measured` |
| 187 | 2.46 | unsourced |  | external: earlier VIPER UPMEM study (docs/harness.md); no results file here | `ER's UPMEM validation: 2.42\X{} predicted against 2.46\X{} measured for a Llama2 layer. Replay` |
| 187 | 2 | setup |  | lower end of a range | `de curve predicts its measured decode time within 2--3\%. \todo{gem5 spot check of the fina` |
| 187 | 3 | matched | \gpuReplayErrMaxAbsPct=2.9 | harness GpuBaseline vs measured, worst session | `curve predicts its measured decode time within 2--3\%. \todo{gem5 spot check of the final d` |
| 195 | 14 | matched | \nKnobsBase=14 | harness/space.py KNOBS minus the 3 area knobs | `Design space.} For the coding agent the space has 14 knobs: where the weight MVMs, LM head, ` |
| 195 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `nobs that are inactive under a mapping, there are 575,400 distinct designs, all with ISAAC's tile` |
| 195 | 8 | matched | \knobAdcA=8 | harness/space.py KNOBS adc_per_ima | ` breakdown: a lean tile without the eDRAM buffer, 8, 4 or 2 ADCs per IMA, and 2- or 4-bit c` |
| 195 | 4 | matched | \knobAdcB=4 | harness/space.py KNOBS adc_per_ima | `eakdown: a lean tile without the eDRAM buffer, 8, 4 or 2 ADCs per IMA, and 2- or 4-bit cell` |
| 195 | 2 | matched | \knobAdcC=2 | harness/space.py KNOBS adc_per_ima | `wn: a lean tile without the eDRAM buffer, 8, 4 or 2 ADCs per IMA, and 2- or 4-bit cells. \b` |
| 195 | 2 | matched | \knobCellBitsA=2 | harness/space.py KNOBS cell_bits | `out the eDRAM buffer, 8, 4 or 2 ADCs per IMA, and 2- or 4-bit cells. \begin{algorithm}[t] \` |
| 195 | 4 | matched | \knobCellBitsB=4 | harness/space.py KNOBS cell_bits | `e eDRAM buffer, 8, 4 or 2 ADCs per IMA, and 2- or 4-bit cells. \begin{algorithm}[t] \captio` |
| 202 | 2 | matched | \walkStopPct=2 | harness/search.py domeline_walk defaults | `s, the host runs the rest \While{LLM time of $d > 2\%$ of tool time} \For{each term $k$ in ` |
| 213 | 1 | matched | \shrinkTolPct=1 | harness/search.py domeline_walk defaults | `s that cost area or effort while $T$ stays within 1\% \end{algorithmic} \end{algorithm} \te` |
| 224 | 0.5 | matched | \walkEpsPct=0.5 | harness/search.py domeline_walk defaults | `st move that improves time by more than $\epsilon=0.5\%$ and logs the reading that proposed i` |
| 224 | 2 | matched | \walkStopPct=2 | harness/search.py domeline_walk defaults | `gn's explanation. It stops when LLM time falls to 2\% of tool time, that is, when the roof ` |
| 224 | 2 | matched | \walkStopPct=2 | harness/search.py domeline_walk defaults | `\% of tool time, that is, when the roof is within 2\% of the Amdahl sheet, or when no move ` |
| 226 | 0.04 | unsourced |  | log only: docs/autopilot.md 19:45 (first iso-area run, overwritten by area_main.json) | ` int8 KV). Without this repair, the walk returned 0.04\X{} under every cap. Effort caps prune ` |
| 228 | 100 | matched | \budget=100 | results/search/main.json | `with or without the readings. Every searcher gets 100 evaluations; the algorithmic ones run t` |
| 228 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | `A searcher's result is the smallest design within 1\% of the fastest it found. We also eval` |
| 228 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `1\% of the fastest it found. We also evaluate all 575,400 designs (11.5\,h on 20 cores) to know t` |
| 228 | 11.5 | matched | \sweepHours=11.5 | results/search/sweep_main.log (last line) | `t it found. We also evaluate all 575,400 designs (11.5\,h on 20 cores) to know the exact optim` |
| 228 | 20 | matched | \sweepWorkers=20 | results/search/sweep_main.log (first line) | ` We also evaluate all 575,400 designs (11.5\,h on 20 cores) to know the exact optimum, and, ` |
| 228 | 3,456 | matched | \nSub=3,456 | results/search/area_main.json | ` area study, the exact area--speed front over the 3,456 designs that vary the crossbar and area` |
| 234 | 89 | matched | \nSessionsFrozen=89 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py); traces/workload_main.tsv | `{Evaluation} \label{sec:eval} \textbf{Setup.} The 89 sessions collected first form a frozen ` |
| 234 | 45 | matched | \nSearch=45 | results/search/main.json | `a frozen workload, split by a seeded shuffle into 45 search sessions and 44 held-out session` |
| 234 | 44 | matched | \nHeldout=44 | results/search/main.json | `t by a seeded shuffle into 45 search sessions and 44 held-out sessions; 11 sessions collecte` |
| 234 | 11 | matched | \nSessionsLate=11 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `into 45 search sessions and 44 held-out sessions; 11 sessions collected after every search f` |
| 234 | 64K | setup |  | trace setup (docs/autopilot.md) | `t the 3080's fp16 wall are re-run with int8 KV at 64K, so the GPU always uses its best config` |
| 242 | 16.7 | matched | \amdahlBound=16.7 | results/search/main.json | `epted move (log scale), under the Amdahl bound of 16.7\X. Bottom: where workload time goes aft` |
| 246 | 25 | matched | \walkStartSlowerX=25 | results/search/main.json trail[0] | ` weight MVMs and the host for everything else, is 25\X{} \emph{slower} than the GPU (0.04\X)` |
| 246 | 0.04 | matched | \walkStart=0.04 | results/search/main.json trail[0] | `thing else, is 25\X{} \emph{slower} than the GPU (0.04\X): attention over the KV cache on the ` |
| 246 | 0.45 | matched | \walkAttention=0.45 | results/search/main.json trail | `host binds. Moving attention onto crossbars gives 0.45\X; now the host's softmax binds, and wi` |
| 246 | 0.80 | matched | \walkSoftmax=0.80 | results/search/main.json trail | `layer. Moving softmax to the fixed-function unit (0.80\X) and the LM head to crossbars (1.39\X` |
| 246 | 1.39 | matched | \walkLmHead=1.39 | results/search/main.json trail | `ction unit (0.80\X) and the LM head to crossbars (1.39\X) passes the GPU; norms leave the host` |
| 246 | 1.75 | matched | \walkNorms=1.75 | results/search/main.json trail | `rs (1.39\X) passes the GPU; norms leave the host (1.75\X). Then the two units bind in turn, an` |
| 246 | 2.5 | matched | \walkSizingLo=2.5 | results/search/main.json trail (unit sizing before transposed writes) | ` two units bind in turn, and the walk sizes them (2.5--6.5\X). At that point KV writes bind: ` |
| 246 | 6.5 | matched | \walkSizingHi=6.5 | results/search/main.json trail | `units bind in turn, and the walk sizes them (2.5--6.5\X). At that point KV writes bind: a K a` |
| 246 | 7.4 | matched | \walkTranspose=7.4 | results/search/main.json trail | `rossbar, and transposable writes make it one row (7.4\X). Moving norms to the fixed-function ` |
| 246 | 8.47 | matched | \walkNormsStar=8.47 | results/search/main.json trail | `(7.4\X). Moving norms to the fixed-function unit (8.47\X) and sizing it to 64 units gives 9.42` |
| 246 | 64 | matched | \walkStarUnits=64 | results/search/main.json best point | `the fixed-function unit (8.47\X) and sizing it to 64 units gives 9.42\X. The shrink phase dr` |
| 246 | 9.42 | matched | \walkSpeedup=9.42 | results/search/main.json (walk best) | `ion unit (8.47\X) and sizing it to 64 units gives 9.42\X. The shrink phase drops resources tha` |
| 246 | 18,796 | matched | \walkAreaBeforeShrink=18,796 | results/search/main.json trail (largest area) | ` resources that no longer bind and cuts area from 18,796 to 9,729\,mm$^2$ at the same speed. On ` |
| 246 | 9,729 | matched | \walkArea=9,729 | results/search/main.json | ` that no longer bind and cuts area from 18,796 to 9,729\,mm$^2$ at the same speed. On the Domel` |
| 248 | 64 | matched | \walkStarUnits=64 | results/search/main.json best point | `emapping (norms onto the fixed-function unit) and 64 units finish the climb. The arrays are ` |
| 248 | 9.42 | matched | \walkSpeedup=9.42 | results/search/main.json (walk best) | `ot enough. The readings also say when to stop: at 9.42\X{}, tools take 56\% of workload time, ` |
| 248 | 56 | mismatched | \toolShareAtWalkPct=57 | results/search/main.json | `gs also say when to stop: at 9.42\X{}, tools take 56\% of workload time, and the Amdahl boun` |
| 248 | 16.7 | matched | \amdahlBound=16.7 | results/search/main.json | `ke 56\% of workload time, and the Amdahl bound is 16.7\X. \subsection{Search efficiency agains` |
| 256 | 3 | matched | \tpeSeeds=3 | results/search/sweep_main_summary.json | `ed. Left: classical searchers (shaded: range over 3 seeds). Right: LLM agents. Dotted: the ` |
| 256 | 9.48 | matched | \sweepOptimum=9.48 | results/search/sweep_main_summary.json | `ight: LLM agents. Dotted: the exhaustive optimum, 9.48\X.} \label{fig:search} \end{figure} \be` |
| 262 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | ` design and evaluation of the first design within 1\% of it; for multi-seed rows, how many ` |
| 262 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | `it; for multi-seed rows, how many runs got within 1\%. Budget 100. Algorithmic searchers: 3` |
| 262 | 100 | matched | \budget=100 | results/search/main.json | `i-seed rows, how many runs got within 1\%. Budget 100. Algorithmic searchers: 3 seeds; Sonnet` |
| 262 | 3 | matched | \tpeSeeds=3 | results/search/sweep_main_summary.json | `ot within 1\%. Budget 100. Algorithmic searchers: 3 seeds; Sonnet: 1--2 runs per variant, w` |
| 262 | 1 | matched | \sonnetRunsPerVariantLo=1 | results/search/main.json | `dget 100. Algorithmic searchers: 3 seeds; Sonnet: 1--2 runs per variant, with and without r` |
| 262 | 2 | matched | \sonnetRunsPerVariantHi=2 | effort5.json / area3000.json | `t 100. Algorithmic searchers: 3 seeds; Sonnet: 1--2 runs per variant, with and without read` |
| 268 | 5 | setup |  |  | `ular}{@{}lccc@{}} \toprule & No cap & Effort $\le$5\,mo & Area $\le$3,000\,mm$^2$ \\ & (opt` |
| 268 | 3,000 | setup |  |  | `\toprule & No cap & Effort $\le$5\,mo & Area $\le$3,000\,mm$^2$ \\ & (opt.\ 9.48\X) & (opt.\ 8.` |
| 269 | 9.48 | matched | \sweepOptimum=9.48 | results/search/sweep_main_summary.json | ` $\le$5\,mo & Area $\le$3,000\,mm$^2$ \\ & (opt.\ 9.48\X) & (opt.\ 8.05\X) & (best 7.20\X) \\ ` |
| 269 | 8.05 | matched | \sweepOptimumEffortFive=8.05 | results/search/sweep_main_summary.json | `a $\le$3,000\,mm$^2$ \\ & (opt.\ 9.48\X) & (opt.\ 8.05\X) & (best 7.20\X) \\ \midrule \sys{} w` |
| 269 | 7.20 | matched | \areaThreeKBestKnown=7.20 | results/search/area3000.json + area3000_haiku.json (best design found by any run) | `$^2$ \\ & (opt.\ 9.48\X) & (opt.\ 8.05\X) & (best 7.20\X) \\ \midrule \sys{} walk & 0.7\%, eva` |
| 271 | 0.7 | matched | \walkGapPct=0.7 | results/search/sweep_main_summary.json | `8.05\X) & (best 7.20\X) \\ \midrule \sys{} walk & 0.7\%, eval 20 & 0.6\%, eval 24 & 0.5\%, ev` |
| 271 | 20 | matched | \walkFirstWithin=20 | results/search/sweep_main_summary.json | `est 7.20\X) \\ \midrule \sys{} walk & 0.7\%, eval 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TP` |
| 271 | 0.6 | matched | \walkGapEffortFivePct=0.6 | results/search/sweep_main_summary.json | `.20\X) \\ \midrule \sys{} walk & 0.7\%, eval 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/` |
| 271 | 24 | matched | \walkFirstWithinEffortFive=24 | results/search/sweep_main_summary.json | `idrule \sys{} walk & 0.7\%, eval 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41` |
| 271 | 0.5 | matched | \walkGapAreaThreeKPct=0.5 | results/search/area3000.json | `e \sys{} walk & 0.7\%, eval 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 &` |
| 271 | 41 | matched | \walkFirstWithinAreaThreeK=41 | results/search/area3000.json | `k & 0.7\%, eval 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 ` |
| 272 | 3 | matched | \tpeWithinCount=3 | results/search/sweep_main_summary.json | `val 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill cli` |
| 272 | 3 | matched | \tpeSeeds=3 | results/search/sweep_main_summary.json | `l 20 & 0.6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climb` |
| 272 | 41 | matched | \tpeFirstWithinLo=41 | results/search/sweep_main_summary.json | `6\%, eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 ` |
| 272 | 69 | matched | \tpeFirstWithinHi=69 | results/search/sweep_main_summary.json | ` eval 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/` |
| 272 | 1 | matched | \tpeWithinEffortFiveCount=1 | results/search/sweep_main_summary.json | ` 24 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & ` |
| 272 | 3 | matched | \tpeSeedsEffortFive=3 | results/search/sweep_main_summary.json | `4 & 0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\` |
| 272 | 1 | matched | \tpeWithinAreaThreeKCount=1 | results/search/area3000.json | `0.5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0` |
| 272 | 3 | matched | \tpeSeeds=3 | results/search/sweep_main_summary.json | `5\%, eval 41 \\ TPE & 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0.0` |
| 273 | 2 | matched | \hillWithinCount=2 | results/search/sweep_main_summary.json | ` 3/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 &` |
| 273 | 3 | matched | \hillSeeds=3 | results/search/sweep_main_summary.json | `/3, evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0` |
| 273 | 1 | matched | \hillWithinEffortFiveCount=1 | results/search/sweep_main_summary.json | `evals 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 &` |
| 273 | 3 | matched | \hillSeedsEffortFive=3 | results/search/sweep_main_summary.json | `als 41--69 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $` |
| 273 | 0.04 | matched | \hillBestAreaThreeK=0.04 | results/search/area3000.json | `9 & 1/3 & 1/3 \\ Hill climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $\le$1.41\X \` |
| 274 | 0 | matched | \randomWithinCount=0 | results/search/sweep_main_summary.json | `ll climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $\le$1.41\X \\ Sonnet agent &` |
| 274 | 3 | matched | \randomSeeds=3 | results/search/sweep_main_summary.json | ` climbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $\le$1.41\X \\ Sonnet agent & e` |
| 274 | 0 | matched | \randomWithinEffortFiveCount=0 | results/search/sweep_main_summary.json | `mbing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $\le$1.41\X \\ Sonnet agent & evals` |
| 274 | 3 | matched | \randomSeedsEffortFive=3 | results/search/sweep_main_summary.json | `ing & 2/3 & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $\le$1.41\X \\ Sonnet agent & evals 4` |
| 274 | 1.41 | matched | \randomBestAreaThreeK=1.41 | results/search/area3000.json | ` & 1/3 & $\le$0.04\X \\ Random & 0/3 & 0/3 & $\le$1.41\X \\ Sonnet agent & evals 4--10 & evals` |
| 275 | 4 | matched | \sonnetFirstWithinLo=4 | results/search/sweep_main_summary.json | `& 0/3 & 0/3 & $\le$1.41\X \\ Sonnet agent & evals 4--10 & evals 9--11 & evals 5--8 \\ \bott` |
| 275 | 10 | matched | \sonnetFirstWithinHi=10 | results/search/sweep_main_summary.json | `/3 & 0/3 & $\le$1.41\X \\ Sonnet agent & evals 4--10 & evals 9--11 & evals 5--8 \\ \bottomru` |
| 275 | 9 | matched | \sonnetFirstWithinEffortFiveLo=9 | results/search/sweep_main_summary.json | `$\le$1.41\X \\ Sonnet agent & evals 4--10 & evals 9--11 & evals 5--8 \\ \bottomrule \end{ta` |
| 275 | 11 | matched | \sonnetFirstWithinEffortFiveHi=11 | results/search/sweep_main_summary.json | `e$1.41\X \\ Sonnet agent & evals 4--10 & evals 9--11 & evals 5--8 \\ \bottomrule \end{tabula` |
| 275 | 5 | matched | \sonnetFirstWithinAreaThreeKLo=5 | results/search/area3000.json | ` Sonnet agent & evals 4--10 & evals 9--11 & evals 5--8 \\ \bottomrule \end{tabular} \end{ta` |
| 275 | 8 | matched | \sonnetFirstWithinAreaThreeKHi=8 | results/search/area3000.json | `nnet agent & evals 4--10 & evals 9--11 & evals 5--8 \\ \bottomrule \end{tabular} \end{table` |
| 280 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `8 \\ \bottomrule \end{tabular} \end{table} Of the 575,400 designs, 280,296 are feasible; only 3.8` |
| 280 | 280,296 | matched | \nFeasible=280,296 | results/search/sweep_main_summary.json | `\end{tabular} \end{table} Of the 575,400 designs, 280,296 are feasible; only 3.8\% of those beat ` |
| 280 | 3.8 | matched | \beatGpuPct=3.8 | results/search/sweep_main_summary.json | `f the 575,400 designs, 280,296 are feasible; only 3.8\% of those beat the 3080, and 96 (0.03\` |
| 280 | 96 | matched | \nWithinOnePct=96 | results/search/sweep_main_summary.json | ` feasible; only 3.8\% of those beat the 3080, and 96 (0.03\%) are within 1\% of the optimum ` |
| 280 | 0.03 | matched | \withinOnePctOfFeasiblePct=0.03 | results/search/sweep_main_summary.json | `sible; only 3.8\% of those beat the 3080, and 96 (0.03\%) are within 1\% of the optimum of 9.4` |
| 280 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | `f those beat the 3080, and 96 (0.03\%) are within 1\% of the optimum of 9.48\X. The walk's ` |
| 280 | 9.48 | matched | \sweepOptimum=9.48 | results/search/sweep_main_summary.json | ` and 96 (0.03\%) are within 1\% of the optimum of 9.48\X. The walk's design is the smallest of` |
| 280 | 96 | matched | \nWithinOnePct=96 | results/search/sweep_main_summary.json | `.48\X. The walk's design is the smallest of these 96 (the area knobs, added later, shrink it` |
| 282 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | `y algorithmic searcher that reaches the optimum's 1\% band in every setting. Hill climbing ` |
| 282 | 1 | matched | \hillMissedCount=1 | results/search/sweep_main_summary.json | `e-knob moves, guided by time alone, gets stuck on 1 of 3 seeds unconstrained and 2 of 3 und` |
| 282 | 3 | matched | \hillSeeds=3 | results/search/sweep_main_summary.json | `b moves, guided by time alone, gets stuck on 1 of 3 seeds unconstrained and 2 of 3 under th` |
| 282 | 2 | matched | \hillMissedEffortFiveCount=2 | results/search/sweep_main_summary.json | `one, gets stuck on 1 of 3 seeds unconstrained and 2 of 3 under the effort cap: without the ` |
| 282 | 3 | matched | \hillSeedsEffortFive=3 | results/search/sweep_main_summary.json | `gets stuck on 1 of 3 seeds unconstrained and 2 of 3 under the effort cap: without the readi` |
| 282 | 2 | matched | \tpeEvalRatioLo=2.0 | results/search/sweep_main_summary.json | `ings it cannot tell which knob matters. TPE needs 2--3\X{} more evaluations unconstrained a` |
| 282 | 3 | matched | \tpeEvalRatioHi=3.5 | results/search/sweep_main_summary.json | `s it cannot tell which knob matters. TPE needs 2--3\X{} more evaluations unconstrained and ` |
| 282 | 2 | matched | \tpeMissedEffortFiveCount=2, \tpeMissedAreaThreeKCount=2 | results/search/sweep_main_summary.json; results/search/area3000.json | `3\X{} more evaluations unconstrained and fails on 2 of 3 seeds under either cap. Random sea` |
| 282 | 3 | matched | \tpeSeeds=3 | results/search/sweep_main_summary.json | ` more evaluations unconstrained and fails on 2 of 3 seeds under either cap. Random search f` |
| 284 | 150 | matched | \areaBudget=150 | results/search/area_main.json | `where readings pay.} We run the walk and TPE with 150 evaluations under ten area caps. At all` |
| 284 | 64K | setup |  |  | `ns under ten area caps. At all eight caps where a 64K-context design fits, the walk lands wit` |
| 284 | 0.6 | matched | \walkAreaGapMaxPct=0.6 | results/search/area_main.json | ` a 64K-context design fits, the walk lands within 0.6\% of the exact area--speed front, in 31` |
| 284 | 31 | matched | \walkAreaEvalsLo=31 | results/search/area_main.json | `s within 0.6\% of the exact area--speed front, in 31--90 evaluations. TPE matches the front ` |
| 284 | 90 | matched | \walkAreaEvalsHi=90 | results/search/area_main.json | `thin 0.6\% of the exact area--speed front, in 31--90 evaluations. TPE matches the front only` |
| 284 | 6,000 | setup |  |  | `. TPE matches the front only without a cap and at 6,000\,mm$^2$: at 8,000\,mm$^2$ it returns 9.` |
| 284 | 8,000 | setup |  |  | `front only without a cap and at 6,000\,mm$^2$: at 8,000\,mm$^2$ it returns 9.01\X{} where 9.47\` |
| 284 | 9.01 | matched | \tpeAreaEightK=9.01 | results/search/area_main.json | `and at 6,000\,mm$^2$: at 8,000\,mm$^2$ it returns 9.01\X{} where 9.47\X{} is available, and at` |
| 284 | 9.47 | matched | \frontAreaEightK=9.47 | results/search/area_main.json | `m$^2$: at 8,000\,mm$^2$ it returns 9.01\X{} where 9.47\X{} is available, and at 5,000\,mm$^2$ ` |
| 284 | 5,000 | setup |  |  | `urns 9.01\X{} where 9.47\X{} is available, and at 5,000\,mm$^2$ and below 0.63--7.79\X{} where ` |
| 284 | 0.63 | matched | \tpeAreaLowLo=0.63 | results/search/area_main.json | `\X{} is available, and at 5,000\,mm$^2$ and below 0.63--7.79\X{} where 1.78--9.00\X{} is. Unde` |
| 284 | 7.79 | matched | \tpeAreaLowHi=7.79 | results/search/area_main.json | `s available, and at 5,000\,mm$^2$ and below 0.63--7.79\X{} where 1.78--9.00\X{} is. Under effo` |
| 284 | 1.78 | matched | \frontAreaLowLo=1.78 | results/search/area_main.json | `d at 5,000\,mm$^2$ and below 0.63--7.79\X{} where 1.78--9.00\X{} is. Under effort caps the wal` |
| 284 | 9.00 | matched | \frontAreaLowHi=9.00 | results/search/area_main.json | `,000\,mm$^2$ and below 0.63--7.79\X{} where 1.78--9.00\X{} is. Under effort caps the walk repr` |
| 284 | 0.59 | matched | \stairSweepZero=0.59, \stairWalkZero=0.59 | results/search/sweep_main_summary.json; results/search/effort_main.json | `walk reproduces the exact staircase of the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 an` |
| 284 | 3.08 | mismatched | \stairSweepOne=3.08, \stairWalkOne=3.07 | results/search/sweep_main_summary.json; results/search/effort_main.json | `eproduces the exact staircase of the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 mo` |
| 284 | 8.00 | matched | \stairSweepFour=8.00, \stairWalkFour=8.00 | results/search/sweep_main_summary.json; results/search/effort_main.json | `ces the exact staircase of the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 months o` |
| 284 | 9.42 | matched | \stairSweepSeven=9.42, \stairWalkSeven=9.42 | results/search/sweep_main_summary.json; results/search/effort_main.json | `xact staircase of the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 months of effort.` |
| 284 | 0 | setup |  |  | `e of the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 months of effort. ``Three m` |
| 284 | 1 | setup |  |  | `f the sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 months of effort. ``Three more` |
| 284 | 4 | setup |  |  | `he sweep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 months of effort. ``Three more mo` |
| 284 | 7 | setup |  |  | `ep: 0.59, 3.08, 8.00 and 9.42\X{} for 0, 1, 4 and 7 months of effort. ``Three more months b` |
| 284 | 18 | matched | \stairGainLastPct=18 | results/search/effort_main.json | `4 and 7 months of effort. ``Three more months buy 18\%'' is read off the walk. \textbf{LLM a` |
| 286 | 40 | matched | \haikuStopsSoonerPct=38 | results/search/area3000_haiku.json | `e in this space. With readings, Haiku stops about 40\% sooner and cites the binding terms wh` |
| 286 | 2 | matched | \localBeatGpuReadings=2 | results/search/local_main.json | `erms when it stops; the 4B model beats the GPU on 2 of 3 runs with readings and 0 of 3 with` |
| 286 | 3 | matched | \localRuns=3 | results/search/local_main.json | `when it stops; the 4B model beats the GPU on 2 of 3 runs with readings and 0 of 3 without, ` |
| 286 | 0 | matched | \localBeatGpuScalar=0 | results/search/local_main.json | `el beats the GPU on 2 of 3 runs with readings and 0 of 3 without, but stays far from the op` |
| 286 | 3 | matched | \localRuns=3 | results/search/local_main.json | `ats the GPU on 2 of 3 runs with readings and 0 of 3 without, but stays far from the optimum` |
| 288 | 1 | matched | \bestTolPct=1 | harness/evaluate.py Evaluator.best (smallest design within tol of the fastest) | `s every move. \textbf{Cost.} The walk reaches the 1\% band in 20--41 evaluations (30--50 in` |
| 288 | 20 | matched | \walkFirstWithin=20 | results/search/sweep_main_summary.json | `. \textbf{Cost.} The walk reaches the 1\% band in 20--41 evaluations (30--50 in all, across ` |
| 288 | 41 | matched | \walkFirstWithinRangeHi=41 | Table II settings | `extbf{Cost.} The walk reaches the 1\% band in 20--41 evaluations (30--50 in all, across the ` |
| 288 | 30 | matched | \walkEvalsRangeLo=30 | main / effort5 / area3000 | ` walk reaches the 1\% band in 20--41 evaluations (30--50 in all, across the three settings);` |
| 288 | 50 | matched | \walkEvalsRangeHi=50 | main / effort5 / area3000 | `k reaches the 1\% band in 20--41 evaluations (30--50 in all, across the three settings); the` |
| 288 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | ` all, across the three settings); the sweep needs 575,400. \sys{} makes the sweep affordable here` |
| 288 | 1 | matched | \gemFiveHoursLo=1 | assumption (paper): gem5-class evaluator, h per design | `s measure the gap. With a gem5-class evaluator at 1--3\,h per design, the sweep would take ` |
| 288 | 3 | matched | \gemFiveHoursHi=3 | assumption (paper): gem5-class evaluator, h per design | `easure the gap. With a gem5-class evaluator at 1--3\,h per design, the sweep would take 66-` |
| 288 | 66 | matched | \gemFiveCpuYearsLo=66 | results/search/sweep_main_summary.json x 1 h per design | `uator at 1--3\,h per design, the sweep would take 66--200 CPU-years and the walk one day. \s` |
| 288 | 200 | matched | \gemFiveCpuYearsHi=197 | results/search/sweep_main_summary.json x 3 h per design | `r at 1--3\,h per design, the sweep would take 66--200 CPU-years and the walk one day. \subsec` |
| 307 | 2.16M | matched | \nArraysM=2.16 | harness integrator, headline design of results/search/main.json | ` the weights, the LM head and an int8 KV cache in 2.16M ReRAM crossbars with transposable K wri` |
| 307 | 64 | matched | \walkStarUnits=64 | results/search/main.json best point | `e K writes, runs softmax, norms, RoPE and SiLU on 64 fixed-function units, keeps embedding a` |
| 307 | 9.42 | matched | \walkSpeedup=9.42 | results/search/main.json (walk best) | `ver a CXL-class link. It runs the search sessions 9.42\X{} faster than the 3080 and the held-o` |
| 307 | 9.61 | matched | \walkHeldout=9.61 | results/search/main.json (walk best, held-out 44) | `{} faster than the 3080 and the held-out sessions 9.61\X{} faster. On the 11 sessions collecte` |
| 307 | 11 | matched | \nSessionsLate=11 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `and the held-out sessions 9.61\X{} faster. On the 11 sessions collected after the search, it` |
| 307 | 15.4 | matched | \lateSpeedup=15.4 | results/search/heldout_extra.json | `the 11 sessions collected after the search, it is 15.4\X{} faster: those sessions spend less t` |
| 307 | 54 | matched | \lateAmdahl=54 | results/search/heldout_extra.json | `s spend less time in tools (their Amdahl bound is 54\X), and the gain tracks the tool share ` |
| 309 | 0.40 | matched | \decodeFourKMs=0.40 | harness integrator, headline design of results/search/main.json | `s the model says it must. Per token, decode takes 0.40\,ms at 4K context and 0.77\,ms at 64K, ` |
| 309 | 4K | setup |  |  | `says it must. Per token, decode takes 0.40\,ms at 4K context and 0.77\,ms at 64K, against 7.` |
| 309 | 0.77 | matched | \decodeSixtyFourKMs=0.77 | harness integrator, headline design of results/search/main.json | `er token, decode takes 0.40\,ms at 4K context and 0.77\,ms at 64K, against 7.5--28\,ms on the ` |
| 309 | 64K | setup |  |  | `code takes 0.40\,ms at 4K context and 0.77\,ms at 64K, against 7.5--28\,ms on the 3080 (Fig.~` |
| 309 | 7.5 | matched | \gpuDecodeEmptyMs=7.5 | results/gpu/g1.jsonl | `40\,ms at 4K context and 0.77\,ms at 64K, against 7.5--28\,ms on the 3080 (Fig.~\ref{fig:head` |
| 309 | 28 | mismatched | \gpuDecodeSixtyFourKMs=27 | results/gpu/g1.jsonl | `s at 4K context and 0.77\,ms at 64K, against 7.5--28\,ms on the 3080 (Fig.~\ref{fig:headline` |
| 309 | 21 | matched | \perTokenFourKX=21 | harness integrator, headline design of results/search/main.json vs g1.jsonl fp16 | `5--28\,ms on the 3080 (Fig.~\ref{fig:headline}a): 21\X{} at 4K and 25\X{} at 32K. The crossb` |
| 309 | 4K | setup |  |  | `on the 3080 (Fig.~\ref{fig:headline}a): 21\X{} at 4K and 25\X{} at 32K. The crossbar read cy` |
| 309 | 25 | matched | \perTokenThirtyTwoKX=25 | harness integrator, headline design of results/search/main.json vs g1.jsonl fp16 | `3080 (Fig.~\ref{fig:headline}a): 21\X{} at 4K and 25\X{} at 32K. The crossbar read cycle set` |
| 309 | 32K | setup |  |  | `~\ref{fig:headline}a): 21\X{} at 4K and 25\X{} at 32K. The crossbar read cycle sets a flat fl` |
| 309 | 0.35 | matched | \xbarFloorMs=0.35 | harness integrator, headline design of results/search/main.json | `32K. The crossbar read cycle sets a flat floor of 0.35\,ms; at long context, score traffic bet` |
| 311 | 50 | mismatched | \roofAmdahlLoPct=51 | harness integrator, headline design of results/search/main.json | `ssbar reads color the roof, and the roof stays at 50--67\% of the Amdahl sheet; the 3080 sit` |
| 311 | 67 | mismatched | \roofAmdahlHiPct=68 | harness integrator, headline design of results/search/main.json | `r reads color the roof, and the roof stays at 50--67\% of the Amdahl sheet; the 3080 sits 7-` |
| 311 | 7 | matched | \roofOverGpuLoX=7 | harness integrator, headline design of results/search/main.json vs g1.jsonl fp16 | `ys at 50--67\% of the Amdahl sheet; the 3080 sits 7--11\X{} lower before its VRAM wall at 3` |
| 311 | 11 | matched | \roofOverGpuHiX=11 | harness integrator, headline design of results/search/main.json vs g1.jsonl fp16 | `at 50--67\% of the Amdahl sheet; the 3080 sits 7--11\X{} lower before its VRAM wall at 34K t` |
| 311 | 34K | matched | \gpuWallFigK=34 | fig_domeline.py: between the last fit (32K) and first fail (36K) | `3080 sits 7--11\X{} lower before its VRAM wall at 34K tokens and two orders of magnitude lowe` |
| 311 | 64K | matched | \designWallK=64.1 | harness integrator, headline design of results/search/main.json | `er after it. The design's own wall lies just past 64K tokens, where KV writes would take over` |
| 311 | 2,872 | matched | \frontAdcFourArea=2,872 | results/search/area_main.json | `ed it for the longest session and no further. The 2,872\,mm$^2$ design has the same shape and a` |
| 316 | 0.79 | matched | \tFixMs=0.79 | harness integrator, headline design of results/search/main.json; search sessions' tool time per generated token | `n charged its share of tool time ($t_\mathrm{fix}=0.79$\,ms, so the Amdahl sheet is at 1,272 t` |
| 316 | 1,272 | matched | \amdahlTokS=1,272 | harness integrator, headline design of results/search/main.json | `\mathrm{fix}=0.79$\,ms, so the Amdahl sheet is at 1,272 tokens/s). The headline roof is colored` |
| 320 | 7.1 | matched | \vsFourNinety=7.1 | results/search/gpu_scale.json | `Us (same efficiency, no VRAM wall), the design is 7.1\X{} faster than an RTX 4090 and 2.5\X{}` |
| 320 | 2.5 | matched | \vsHHundred=2.5 | results/search/gpu_scale.json | `the design is 7.1\X{} faster than an RTX 4090 and 2.5\X{} faster than an H100 SXM on session ` |
| 322 | 32 | unsourced |  | external: ISAAC [isaac], 32 nm | `.} The headline design keeps ISAAC's tile: at its 32\,nm density it takes 9,729\,mm$^2$, abo` |
| 322 | 9,729 | matched | \walkArea=9,729 | results/search/main.json | `eeps ISAAC's tile: at its 32\,nm density it takes 9,729\,mm$^2$, about 15\X{} the 3080's die. T` |
| 322 | 15 | matched | \walkAreaOverDie=15 | results/search/main.json / GA102 628.4 mm2 | ` its 32\,nm density it takes 9,729\,mm$^2$, about 15\X{} the 3080's die. The arrays are unde` |
| 322 | 1 | unsourced |  | external: ISAAC Table I (arrays 0.0024 of the 0.372 mm2 tile); not computed by any script | `about 15\X{} the 3080's die. The arrays are under 1\% of a tile; ADCs and the eDRAM buffer ` |
| 322 | 55 | matched | \kvArraysPct=55 | harness integrator, headline design of results/search/main.json | `le; ADCs and the eDRAM buffer are most of it, and 55\% of the arrays hold KV sized for the s` |
| 322 | 64K | matched | \maxCtxK=64 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py) | `d 55\% of the arrays hold KV sized for the single 64K session. The exact front trades area fo` |
| 322 | 9.47 | matched | \frontLeanSpeedup=9.47 | results/search/area_main.json | `K session. The exact front trades area for speed: 9.47\X{} at 5,833\,mm$^2$ with a lean tile, ` |
| 322 | 5,833 | matched | \frontLeanArea=5,833 | results/search/area_main.json | `he exact front trades area for speed: 9.47\X{} at 5,833\,mm$^2$ with a lean tile, 9.00\X{} at 3` |
| 322 | 9.00 | matched | \frontCellFourSpeedup=9.00 | results/search/area_main.json | `peed: 9.47\X{} at 5,833\,mm$^2$ with a lean tile, 9.00\X{} at 3,692\,mm$^2$ with 4-bit cells, ` |
| 322 | 3,692 | matched | \frontCellFourArea=3,692 | results/search/area_main.json | `{} at 5,833\,mm$^2$ with a lean tile, 9.00\X{} at 3,692\,mm$^2$ with 4-bit cells, 7.19\X{} at 2` |
| 322 | 4 | matched | \knobCellBitsB=4 | harness/space.py KNOBS cell_bits | ` with a lean tile, 9.00\X{} at 3,692\,mm$^2$ with 4-bit cells, 7.19\X{} at 2,872\,mm$^2$ an` |
| 322 | 7.19 | matched | \frontAdcFourSpeedup=7.19 | results/search/area_main.json | `tile, 9.00\X{} at 3,692\,mm$^2$ with 4-bit cells, 7.19\X{} at 2,872\,mm$^2$ and 5.13\X{} at 2,` |
| 322 | 2,872 | matched | \frontAdcFourArea=2,872 | results/search/area_main.json | `{} at 3,692\,mm$^2$ with 4-bit cells, 7.19\X{} at 2,872\,mm$^2$ and 5.13\X{} at 2,461\,mm$^2$ w` |
| 322 | 5.13 | matched | \frontAdcTwoSpeedup=5.13 | results/search/area_main.json | `$ with 4-bit cells, 7.19\X{} at 2,872\,mm$^2$ and 5.13\X{} at 2,461\,mm$^2$ with 4 and 2 ADCs ` |
| 322 | 2,461 | matched | \frontAdcTwoArea=2,461 | results/search/area_main.json | ` cells, 7.19\X{} at 2,872\,mm$^2$ and 5.13\X{} at 2,461\,mm$^2$ with 4 and 2 ADCs per IMA; belo` |
| 322 | 4 | matched | \knobAdcB=4 | harness/space.py KNOBS adc_per_ima | ` 2,872\,mm$^2$ and 5.13\X{} at 2,461\,mm$^2$ with 4 and 2 ADCs per IMA; below 1,600\,mm$^2$` |
| 322 | 2 | matched | \knobAdcC=2 | harness/space.py KNOBS adc_per_ima | `\,mm$^2$ and 5.13\X{} at 2,461\,mm$^2$ with 4 and 2 ADCs per IMA; below 1,600\,mm$^2$ a 64K` |
| 322 | 1,600 | matched | \frontMinArea=1,612 | results/search/area_main.json | `at 2,461\,mm$^2$ with 4 and 2 ADCs per IMA; below 1,600\,mm$^2$ a 64K context no longer fits. A` |
| 322 | 64K | setup |  |  | ` with 4 and 2 ADCs per IMA; below 1,600\,mm$^2$ a 64K context no longer fits. A GPU cannot bu` |
| 322 | 22 | setup |  |  | `he front prices that trade. \todo{node scaling to 22/12\,nm ReRAM with a sourced ADC-area ru` |
| 322 | 12 | setup |  |  | `front prices that trade. \todo{node scaling to 22/12\,nm ReRAM with a sourced ADC-area rule.` |
| 324 | 89 | matched | \nSessionsFrozen=89 | results/traces/swebench{,_kv8} (rule of figs/fig_trace_profile.py); traces/workload_main.tsv | `r by $\frac12$ and $2$ and re-run the walk on all 89 sessions (base 9.52\X). Only the crossb` |
| 324 | 9.52 | matched | \sensBase=9.52 | results/search/sensitivity_main.json | ` $2$ and re-run the walk on all 89 sessions (base 9.52\X). Only the crossbar read cycle (100\,` |
| 324 | 100 | matched | \xbarReadNs=100 | lib/tech/pim.yaml xbar_isaac | `ions (base 9.52\X). Only the crossbar read cycle (100\,ns, from ISAAC) moves the result much:` |
| 324 | 11.0 | matched | \sensReadHalf=11.0 | results/search/sensitivity_main.json | `ycle (100\,ns, from ISAAC) moves the result much: 11.0\X{} and 7.5\X. The fixed-function unit'` |
| 324 | 7.5 | matched | \sensReadDouble=7.5 | results/search/sensitivity_main.json | `, from ISAAC) moves the result much: 11.0\X{} and 7.5\X. The fixed-function unit's throughput` |
| 324 | 4 | matched | \sensStarMaxPct=3.6 | results/search/sensitivity_main.json | `nit's throughput and bandwidth move it by at most 4\%; the link, write latency and host ker` |
| 324 | 1 | matched | \sensSmallMaxPct=0.6 | results/search/sensitivity_main.json | `the link, write latency and host kernels by under 1\%. Per-array area scales area linearly ` |
| 324 | 4 | matched | \knobCellBitsB=4 | harness/space.py KNOBS cell_bits | `les area linearly and leaves speed unchanged; the 4-bit-cell area factor shifts the area fr` |
| 324 | 25 | unsourced |  | docs/results.md table (4-b area factor 1.0/1.27/1.5/2.0); no script or results file produces it | `ea factor shifts the area front by one step per $+25\%$. \section{Discussion and Limitations` |
| 330 | 32 | unsourced |  | external: ISAAC (ISCA 2016) | `the result. The ISAAC parameters are from 2016 at 32\,nm; newer ReRAM macros are denser, so ` |
| 339 | 9.4 | matched | \walkSpeedupShort=9.4 | results/search/main.json (walk best) | ` finds a PIM coprocessor that runs a coding agent 9.4\X{} faster than an RTX 3080 in about 20` |
| 339 | 20 | matched | \walkFirstWithin=20 | results/search/sweep_main_summary.json | `ng agent 9.4\X{} faster than an RTX 3080 in about 20 evaluations, within 1\% of the best of ` |
| 339 | 1 | matched | \walkGapPct=0.7 | results/search/sweep_main_summary.json | ` than an RTX 3080 in about 20 evaluations, within 1\% of the best of 575,400 designs, and e` |
| 339 | 575,400 | matched | \nDesigns=575,400 | results/search/sweep_main_summary.json | `n about 20 evaluations, within 1\% of the best of 575,400 designs, and explains each decision. Un` |
<!-- END GENERATED -->
