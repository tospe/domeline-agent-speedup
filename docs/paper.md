# Paper pitch (draft, Oct 22 2026)

Working notes on how to sell the paper. Numbers come from `docs/results.md`. Items marked **TODO** are not done yet.

## The one-sentence claim

Agentic sessions spend 90% of their time decoding one token at a time over a context that grows to tens of thousands of tokens. A model-guided harness finds, in ~20 evaluations, a processing-in-memory coprocessor that runs these sessions 9.4× faster than an RTX 3080. That design is within 1% of the best of 575,400 candidates, and every design decision is explained by a reading of the model.

## Why now (the hook)

- Agents, not chatbots, are the workload that matters: long multi-turn sessions, tool calls, a context that keeps growing. Single-batch decode is memory-bound and the KV cache keeps growing. Consumer GPUs hit a VRAM wall: 8% of our sessions ran out of memory on the 3080 with fp16 KV, and 36% of the late set did.
- PIM papers evaluate kernels or single-prompt inference. Nobody designs PIM for the session: the context path, tool gaps, prefill bursts, KV growth.
- Design-space exploration for accelerators is either brute force over a simulator or an opaque optimizer. Neither explains why a design wins, and simulators make brute force unaffordable.

## Contributions, each with its evidence

1. **Domeline: an analytical model for CPU + PIM coprocessors**, generalizing VIPER's closed form to an op graph × technologies × session. It is validated on UPMEM (VIPER: 2.46× measured vs 2.42× predicted), it reproduces the TERINT estimator exactly, and its GPU replay predicts measured decode within 3%.
   - Figures: model diagram, validation scatter (**TODO**: UPMEM points + gem5 spot checks).
2. **A harness that designs PIM for a workload, guided by the model's readings.** The readings are the binding term, headroom and Amdahl bound. The walk proposes moves from those readings.
   - It is not brute force. The walk reaches the optimum's 1% band in 20 evaluations; the exhaustive sweep needs 575,400. With a gem5-class evaluator (1–3 h per point) the sweep would cost 66–200 CPU-years, and the walk costs one day.
   - It is reliable. The same single-knob moves without readings (hill climbing) fail on 1 of 3 seeds unconstrained and 2 of 3 under an effort cap. TPE needs 2–3× more evaluations and fails under the cap. Random search fails because only 3.8% of designs beat the GPU and 0.03% are within 1% of the optimum.
   - It is explainable. Every move cites the reading that justified it (walk figure). An LLM agent finds the optimum as fast, but gives no such account. That is a different selling point, and we state it plainly.
   - It handles constraints. Under area caps the walk lands within 0.6% of the exact front at every cap; TPE with 150 evaluations misses by 1.2–14× at caps of 5,000 mm² and below.
   - It handles effort budgets. The effort staircase is 0 → 1 → 4 → 7 months for 0.59 → 3.08 → 8.00 → 9.42×, exact against the sweep. "Three more months buy +18%" is read off the figure.
   - Figures: walk, search efficiency (both panels), sweep (whole space), effort staircase.
3. **A PIM design for a coding agent.** ReRAM crossbars hold the weights, the LM head and the KV cache (with transposable K writes), a fixed-function nonlinear unit handles softmax and norms, the CPU keeps embedding and sampling, and a CXL-class link connects them.
   - **9.42×** on the search sessions, **9.61×** on the held-out sessions, and **15.4×** on 11 tasks collected after the search (more LLM-bound, higher bound).
   - Decode takes 0.4–0.8 ms/token from 4K to 64K context, against 7.5–28 ms on the 3080.
   - Sensitivity: only the crossbar read cycle moves the result (7.5–11.0× over ½–2×). It is taken from ISAAC Table I.
   - Figures: trace profile, GPU wall, headline breakdown (`headline`: 21× per token at 4K, 25× at 32K), sensitivity tornado, area front.

## Where a reviewer will push, and the answer

| Objection | Answer now | Still to do |
|---|---|---|
| "The chip is 15× a GPU die." | The area figure: with sourced knobs (lean tile, 4-bit cells, ADC sharing) 9.0× costs 3,700 mm² and 5.1× costs 2,500 mm². The arrays are <1% of a tile; ADCs and buffers are the area. A GPU cannot trade area for single-session latency, but this design can, and the front is that trade. | Node scaling (22/12 nm ReRAM exists in silicon) with a sourced ADC-area rule; a KV tier for the rare >32K sessions. |
| "Analytical model, not silicon." | Validated pieces (UPMEM, TERINT, GPU replay); sensitivity shows which inputs matter. | gem5 spot checks of the final design (Domeline's own pipeline). |
| "ISAAC numbers are 2016, 32 nm." | They are the conservative case. Newer macros are denser. | Literature table from ISSCC/VLSI ReRAM CIM macros (agent collecting). |
| "Why not just a better GPU / quantization?" | The 3080 runs Q8_0 like the design does; its int8/int4 KV walls are measured. | Add a stronger GPU point (e.g. 4090 from published bandwidth, or a measured one) as a sanity bar. |
| "An LLM agent did as well, so why readings?" | True, including under an area cap (Sonnet: 7.20× in 8 evals without readings). The readings make a deterministic, free, explainable searcher as reliable as the agent. Without them, the same moves (hill climbing) fail at 0.04×, and TPE fails on 2 of 3 seeds. Pitch: Domeline readings let a 50-line policy match an LLM, and every move comes with a reason. | Done: Haiku finds it equally fast either way, but stops ~40% sooner with readings; the 4B model is helped a little (2/3 vs 0/3 runs beat the GPU) but is far from the optimum. |
| "One workload." | 100 SWE-bench Verified sessions, held-out sets, repo-stratified. | Second agentic workload (retrieval/web agent, T3), even if small. |

## Framing choices to settle (authors)

- **Title.** Candidates: "Designing PIM for Agents, One Reading at a Time"; "Domeline: Model-Guided Design of PIM Coprocessors for Agentic Workloads"; "Beyond the VRAM Wall: PIM Coprocessors for Coding Agents".
- **Lead with the harness or with the design?** For ISCA/MICRO the design plus its explanation is the story, and the harness is the method. For a DSE/tools venue (e.g. a methodology track, ISPASS) the harness leads. I'd lead with the design and the explanation: "here is why PIM wins for agents, term by term", with the harness as the way we got there.
- **Speedup metric.** Session time (LLM plus tools) is the honest end-to-end metric, and it caps at the Amdahl bound. Per-token decode speedup (~20–35×) is the larger number. Report both, headline the session time.

## Figure plan (paper order)

1. Trace profile: where session time goes, plus the context CDF (`trace_profile`).
2. GPU wall: decode time vs context and KV precision; the step past the wall (`gpu_wall`, G2 running).
3. System model and Domeline readings (diagram, **TODO**).
4. The walk: speedup per move, each annotated with its reading (`walk_main`).
5. Search efficiency vs the exhaustive optimum (`search_main`, `search_effort5`).
6. The whole space: survival curve plus speedup vs area with the front (`sweep_main`).
7. Effort staircase (`effort_main`).
8. Area: speedup vs area with the exact front and the walk under each cap (`area_main`). This is also the strongest walk-vs-TPE result: under area caps TPE misses by up to 14×.
9. Sensitivity tornado (`sensitivity_main`).
