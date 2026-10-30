# Domeline paper outline (Oct 29, 2026)

Working outline for the DAC 2027 submission (abstract Nov 11, paper Nov 18). Decisions and open items: `docs/plan.md`.

## Thesis

Roofline is not enough for a heterogeneous host + PIM system, because performance there depends on how much data is offloaded to PIM, not only on arithmetic intensity. The Domeline adds offloaded data (*D*, the data held in PIM) as a third axis. It is a 3D Roofline that shows which cost binds: host, transfer, writes, compute, or capacity. The design space is too large to read by eye, so a harness reads the Domeline and climbs it automatically. A coding-agent case study shows that the harness produces a good design and explains why the design is good.

## Contributions

1. **The Domeline.** A 3D Roofline for host + PIM systems, with *D* as the third axis. Its sections (map, roofline, cliff) show the binding term, the headroom and the Amdahl bound. Roofline and VIPER are special cases of it.
2. **A harness that uses the Domeline.** Inputs: workload (traces, operator graph), PIM technology library, host profile, objective (latency for now). Output: a design plus a log of the reading behind every move. The search policy plugs in (walk, random, hill climbing, TPE, LLM agent). It lands near the exhaustive optimum in few evaluations and stays reliable under area and effort caps.
3. **Case study.** A ReRAM coprocessor for a coding agent, 9.4x faster than an RTX 3080; the Domeline explains each design decision.

## Sections (6 pages)

- Abstract: Roofline falls short -> Domeline -> harness -> numbers (9.4x; 1% of optimum in ~20 evals; 0.6% under caps; TPE misses by up to 14x) -> one Roofline-vs-Domeline number from E1.
- 1 Introduction. Para 2 (Roofline names the wrong bottleneck or ranks designs wrongly once data is offloaded) needs E1/E2. Fig. 1 teaser: the same design read with Roofline and with the Domeline, from E2.
- 2 Background: Roofline and extensions (Gables, Ridgeline, DECA Roof-Surface), VIPER, DSE.
- 3 The Domeline: 3.1 definition; 3.2 reading it; 3.3 why Roofline is not enough (new, backed by E1); 3.4 computing it.
- 4 The harness: 4.1 interface with four explicit inputs (new); 4.2 walk; 4.3 constraints; 4.4 driving it with an LLM agent.
- 5 Evaluation: 5.1 does the Domeline see what Roofline misses (validation + E1, E2); 5.2 does the harness find good designs (E4-E6); 5.3 case study (E7).
- 6 Discussion and limitations.

## Experiments

| ID | Experiment | Status |
|---|---|---|
| E1 | Over the 280,296 feasible designs: Roofline's prediction (device bandwidth and compute only) vs the Domeline. (a) bottleneck agreement with the binding term; (b) rank correlation of speedups; (c) where Roofline's best design lands. Repeat along the session path. | new, priority 1 |
| E2 | One concrete example from E1 drawn as Fig. 1 (attention over the KV cache, or a design that crosses the wall mid-session) | new, priority 1 |
| E3 | Model validation (UPMEM, GPU replay, TERINT) | have; citation todo |
| E4-E7 | Search vs optimum, caps, readings ablation, case study | have |
| E8 | Generality: rerun the harness with a second technology library or host profile | optional |

Fallback for E1: if Roofline mostly agrees, lead with what it cannot express (the capacity wall, write cost on PUM, the Amdahl bound).
