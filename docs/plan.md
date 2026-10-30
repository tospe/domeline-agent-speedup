# DAC 2027 plan: decisions and open items

Abstract due Nov 11, paper due Nov 18, 2026. The outline is in the Oct 29 note. This file tracks what is decided and what is still open.

## Decided (Oct 29)

- **Third axis: *D*, the data held in PIM** (bytes resident on the coprocessor). The point is to offload data, so the axis measures how much of it is offloaded. Bytes over the link stay a cost term (transfer), not an axis.
- **LLM agents use the harness; they are not a contribution.** The harness is the contribution and the search policy plugs into it: the walk, random search, hill climbing, TPE or an LLM agent. Do not claim "LLM-based."
- **Name: not final; it does not block anything.** "Domeline" is the working name for now.

## To solve

### Definitions (block E1 and the text)

| # | Item | Proposal | Status |
|---|---|---|---|
| 1 | Define *D* precisely: which tensors count (weights, LM head, KV), units, and how *D* grows with context along a session | Bytes resident in PIM at context *c*, taken from `Integrator.capacity`; a session is a path along *D* as the KV cache grows | done (Oct 29 overnight): `docs/theory.md` §1, `Integrator.data_in_pim`; weights at w_b + LM head if offloaded + K,V at k_b if attention is offloaded; 147 KB/token at fp16 KV |
| 2 | Rename the axis in Definition 1, `figures/`, `fig_domeline*` and the text | Use *D* everywhere | open (after 1) |
| 3 | Define the Roofline baseline for E1 | Per op, max(flops/peak, bytes/BW) on the device the op is mapped to (`tech.roofline_time`), summed. No host, link, writes or wall. Its bottleneck is the slowest device and whether it is compute- or bandwidth-bound. Main variant leaves host ops out; a second variant puts them on the host roof. | decided (Oct 29) |
| 4 | Headroom: compute it or drop it from C1 | Compute it in `evaluate.py`: the gap from the binding term to the next term, and the margin in *D* to the capacity wall | done (Oct 29 overnight): `Eval.term_headroom` (binding / second term) and `Eval.capacity_headroom_bytes` (min over techs of capacity − occupied at the longest context); `docs/theory.md` §2; easy to drop from the text |
| 5 | Show that Roofline and VIPER are special cases | Write the reduction (below the wall, no host, no link gives Roofline; one offloaded kernel gives VIPER) and check that the integrator agrees, including the launch terms | done with caveats (Oct 29 overnight): `docs/theory.md` §3, `tests/test_special_cases.py`. VIPER exact below the wall; past it the integrator adds overflow reprogramming once, not ⌈D/C⌉ passes, so Definition 1 must change or be called schematic. Roofline is per-op and exact on one roofline-kind device, plus one launch and per-op overhead |
| 6 | Harness interface: four explicit inputs | Workload, technology library, host profile (now implicit: τ and `results/cpu/cpu_ops.tsv`), objective (latency only, stated as such). Policy is a plug-in, per the decision above. | done (Oct 29 overnight): `harness/inputs.py` (`Inputs`, `HostProfile`, `run`), `docs/harness.md` §8 |

### Experiments

| # | Item | Status |
|---|---|---|
| 7 | E1 script `experiments/roofline_vs_domeline.py`: Roofline time, Domeline terms, binding term and both speedups for each of the 280,296 feasible designs (about 6 h on 20 cores) | open (after 3, 4) |
| 8 | E1 along the session path: the same comparison at 4K/16K/32K/64K context (minutes) | open (after 3) |
| 9 | E2: pick the strongest disagreement (a design that crosses the wall mid-session, or attention over the KV cache) and draw Fig. 1 | open (after 7, 8) |
| 10 | Fallback: decide by Nov 2 whether to lead with what Roofline cannot express (wall, writes, Amdahl bound) | open |
| 11 | E8, a second technology library or host profile: only if E1 is done by Nov 3 | optional |
| 12 | Walk figure fixes (binding-term outline, area path, CVD clash, label placement, low-gain moves) | open |

### Numbers to reconcile

| # | Item | Status |
|---|---|---|
| 13 | VRAM wall: 11% (11/100 traced, `results.md`) vs 8% (`paper.md`). Pick the session set and use one number. | done (Oct 29): both right. 11/100 = all traced (Fig. 1), 7/89 = frozen workload, 4/11 = late. Use 11% (as the paper does); phrasing in `docs/claims_report.md`; `paper.md` aligned |
| 14 | The 575,400 designs and "smallest of 96" hold only in the 14-knob space (the area knobs came later). Either state the scope or rerun E1 and the sweep with the area knobs fixed at their defaults. | done (Oct 29): state the scope. With the area knobs the space has 6,749,832 distinct designs (561,312 crossbar designs x 12). Scope sentence in `docs/claims_report.md`. `sweep_all.py` now enumerates the 17-knob space, so E1 must fix the area knobs at ISAAC defaults to reuse the 575,400 sweep |
| 15 | "0.6% under caps" and "TPE misses by up to 14×" are measured against the exact front of the 3,456-design crossbar subspace, at the 8 feasible caps. Say so. | done (Oct 29): walk gap ≤ 0.56% at the 8 settings (no cap + 7 finite caps); TPE misses by 14.3× (4,000 mm² only), 1.16–14.3× at ≤ 5,000 mm²; TPE beats the subspace front uncapped and at 6,000. Front points use the on-die link. Phrasing in `docs/claims_report.md` |
| 16 | TERINT-GEMV citation: authors and venue are not in the repo | needs your input |
| 17 | GPU vendor peaks are marked VERIFY (`gpu_scale.py`) | done (Oct 29): all confirmed against NVIDIA GA102/Ada whitepapers and nvidia.com/h100 (dense, fp32 accumulate); no change; sources cited in `gpu_scale.py` |

### Paper and process

| # | Item | Status |
|---|---|---|
| 18 | Automatic claim checker: `experiments/make_numbers.py` generates `numbers.tex` macros from the results; a pytest test fails on hard-coded numbers in `main.tex`, and on any reading the paper names that the code does not compute | done (Oct 29): `make_numbers.py` → `paper/numbers.{tex,json}`, registry `paper/claims.yaml`, `tests/test_claims.py`, report `docs/claims_report.md`. 224 matched, 6 mismatched, 12 unsourced; headroom and D pending |
| 19 | Restructure `main.tex` to the outline (the domeline-1c session owns `main.tex`; coordinate) | open (Nov 3–6) |
| 20 | Add the four inputs to `figures/harness.tex` | open (after 6) |
| 21 | Tell the advisors the LLM agent is a usage mode, not the contribution, correcting what the Oct 29 email led them to expect | open |
| 22 | Page budget check (6 pages) | Nov 6–7 |
