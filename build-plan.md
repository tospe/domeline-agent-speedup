# Brief: build PIMDA from scratch and run its experiments

You are a coding agent building the code and running the experiments for a DAC 2027 paper. You start a new codebase. Follow the order of needs in §4, keep working while you wait for answers, and never go past what a stop point blocks.

Written Nov 9. It builds everything new and uses the old code only as described in §2.

**Where to work:** in this repository, on the Linux virtual machine Tomas provides (§4, "Machines"). All paths below are relative to the repository root. Work on the branch `pimda`.

**Which documents apply:**
- This file decides what to do and in what order.
- `AGENTS.md` holds the working rules.
- §11 of this file gives the model, the walk and the facts you need from the paper. The paper's text is not in this repository and you do not need it.
- Everything under `old/` belongs to the previous version of this work. Its documents describe older goals. Use it only as §2 allows.

## 1. The goal

The paper claims that an optimizer can only choose among options someone wrote down, and that PIMDA adds new options. PIMDA is a loop: it finds the best design, reads what limits it, asks an LLM agent for a new option that relieves that limit, verifies the option, and adds it to the library. Your job is to build this carefully, find out whether it works, and report honestly either way.

The order matters. The library of numbers comes first, because every result rests on it. The evaluator comes second. Agents come last.

## 2. The old work: what you may take from it

This repository is organized in three parts:

| Folder | What it holds | Rule |
|---|---|---|
| `data/` | Inputs: traces, measurements, the workload description, and the oracle | Never edit |
| `old/` | The previous version: its code, tests, documents and results | Read-only. Do not copy code from it, import from it, or change it |
| everything else | The new build | Yours |

You may read the old code to learn a formula, and then only to write that formula down in `notes/b2_model.md` (B2, stage 0). A test in `tests/` must fail if any new code imports from `old/`. The old code does not run from `old/`, because its paths changed. If you need to run it to explain an oracle difference, check out the branch `autopilot` in a separate working copy (`git worktree add ../old-run autopilot`).

| You may use | Where | How |
|---|---|---|
| Raw session traces (100 sessions) and the frozen split | `data/traces/`, `data/traces/workload_main.tsv` | As input data |
| RTX 3080 measurements | `data/gpu/g1.jsonl`, `g2.jsonl` | As input data |
| Host CPU profile | `data/cpu/cpu_ops.tsv`, `data/host/i9_12900k.yaml` | As input data |
| Workload description | `data/workload/`, `data/case/` | As input data, after you check it against the model's `config.json` |
| Old library numbers | `data/oracle/tech/*.yaml`, `old/harness/space.py` (`LINKS`, `EFFORT`, the area constants) | As the starting inventory for B1. No number is adopted without the B1 checks |
| Old sweep | `data/oracle/sweep_main.tsv.gz` (575,400 rows: knobs, `time_s`, `speedup`, `feasible`, `area_mm2`, `effort_months`) | As the oracle in B2. Never as a result |
| Old design notes and reports | `old/docs/harness.md`, `old/docs/theory.md`, `old/docs/harness-dev/REPORT-*.md` | To learn what the old evaluator did and why |

**What the old work already showed.** These are inputs to your plan, not results to report.

1. Int8 KV is not a mechanism. The old optimum uses fp16 KV, so removing int8 leaves the ceiling unchanged. This plan removes two mechanisms, T and F, not three.
2. With sampling on the host, the old sweep gives these ceilings: full 9.47×, −T 8.04×, −F 3.27×, −TF 3.08×. Expect numbers near these. If yours differ by more than 1%, find out why before going on.
3. The old −F and −TF best designs sit at the largest number of digital units (256). The grid may limit those ceilings.
4. In knob-tuning runs, Sonnet and Haiku reached the optimum with or without readings, and mini-coder-4b could not drive the loop (best 1.19×). Adding options is a different task, so this does not settle X2. It does mean the 4B model gets a small trial first (X4).
5. Several old numbers were placeholders: the three links, the fixed-function unit's throughput and area, the transposable array's area, and the digital unit's width and clock. B1 exists to fix this.
6. One old evaluation of a design on the 45 search sessions took 0.66 s.

## 3. Terms

| Term | Meaning |
|---|---|
| Design | One value for each knob in §7 |
| Library | The numbers and options a design can use, as data files you build in B1 |
| Primitive | One number with its provenance, such as a crossbar read time |
| Option | Something built from primitives that a design can use: a mapping, a rewrite, a unit variant or a KV policy |
| Mechanism | One of two options the best known design depends on: transposable K writes (T) and the fixed-function softmax/norm unit (F) |
| Reduced library | The library with mechanisms removed: −T, −F, or −TF (both) |
| Ceiling | The best speedup possible in a reduced library, using only its listed options and no more area than the best design in the full library (X1) |
| Speedup | Session time on the RTX 3080 divided by session time on the design, over the 45 search sessions |
| Readings | The evaluator's outputs: binding term (the largest time term) and term ranking, headroom per term, Amdahl gap, wall crossings |
| Walk | The deterministic optimizer in §11.5 |
| Gates | The four checks on a proposal: G1 correct (Lean proof), G2 legal, G3 sourced costs, G4 true rationale |
| Oracle | The old sweep, used only to test the new evaluator |
| Run | One complete execution of a loop with one model, one setup and one run index (1, 2 or 3) |
| Arm | One setup being compared, such as "readings" or "latency only" |
| World | A copy of the library with some numbers changed on purpose |

## 4. Steps at a glance

There are no dates in this plan. The order is set by what each step needs. Work continuously, on as many steps at once as their needs allow.

| Step | What it does or answers | Needs | Priority |
|---|---|---|---|
| B1a | Inventory: every number the model needs | `data/` | must |
| **STOP 1** | Tomas reviews the inventory and fills the open source rows | B1a | |
| B1b | Source every number, group by group | STOP 1 for open groups; nothing for the others | must |
| **STOP 2** | Tomas approves each group, and every custom number | B1b, per group | |
| B1c | Options, the reduced libraries, freeze library v1 | STOP 2 for every group | must |
| B2 | The evaluator, tested against hand cases and the oracle | `data/` | must |
| **STOP 3** | Tomas reviews the oracle comparison | B2 | |
| B3 | Sweep the space with the new evaluator and library v1 | B1c, STOP 3 | must |
| X1 | Ceilings of the reduced libraries | B3 | must |
| X2 | Quick test: can agents find the mechanisms, and do readings help? | B1c, B2 | must |
| **STOP 4** | Report X1 and X2. Tomas picks the path in §6 | X1, X2 | |
| B4 | Gates and the full loop | B2; the Lean part needs nothing | must |
| Freeze | Freeze prompts (§5, rule 6) | B4 | must |
| X4 | Does PIMDA beat the ceiling? | Freeze, STOP 4 | must |
| X7 | Do the gates catch mistakes? | X4 | must |
| X3 | Does Roofline get the bottleneck wrong? | B3 | must |
| X5 | Which part of PIMDA does the work? | Freeze, STOP 4 | should |
| X8 | Is the final design good? | X4 | should |
| X6 | Do readings help when the LLM can't rely on memory? | Freeze, STOP 4 | if time |
| B1d | More devices, CAM and code search | X4 | if time |

The paper's notes use other names: X1 = E0, X2 = pilot, X3 = E1 and E2, X4 = E4, X5 = E5, X6 = E5b, X7 = E6, X8 = E7 and E9. Use the B and X names in your files and reports.

### Three tracks
The first three pieces of work do not depend on each other. Start all three at once, each in its own working copy.

| Track | Steps | Needs | Joins the others |
|---|---|---|---|
| A. Library | B1a, B1b, B1c; the NeuroSim and Eva-CAM runs | Tomas's answers at STOP 1 and STOP 2 | When library v1 is frozen |
| B. Evaluator | B2, stages 1–8, then the oracle check | Nothing at first: it uses `library/provisional.yaml` | When it loads library v1 for B3 |
| C. Lean | The Lean and Mathlib install, rules L1 and L2, then the rest of G1 | Nothing | In B4 |

### Keep running
Tomas answers when he can. Do not wait for him with nothing to do.

- **A stop blocks only what needs its answer.** At a stop, write the report, then go on with every step that does not need that answer.
- **Rehearse ahead.** You may run any later step on inputs that are not yet approved: the provisional library, an evaluator that has not passed STOP 3, prompts that are not frozen. A rehearsal finds the bugs in the pipeline before the real run. Its rules:
  - outputs go under `rehearsal/`, never under `results/` or `runs/`;
  - every output records the library hash and the word "rehearsal";
  - no rehearsal number is reported as a result, compared with a threshold in §6, or used to tune a prompt outside the development world;
  - when the approval arrives, rerun the step for real. Do not promote a rehearsal output.
- **What a stop does block:**

  | Stop | Until Tomas answers, do not |
  |---|---|
  | STOP 1 | choose a source for an open group |
  | STOP 2 | put that group's numbers, or any custom number, into library v1 |
  | STOP 3 | run the real sweep (B3), or report any evaluator number |
  | STOP 4 | run X4, X5 or X6 for real |

- **Rehearsals that call an LLM cost tokens.** Before STOP 4, limit them to one run per model per step.
- **A stop is cleared only by Tomas's written answer.** Silence is not approval.
- **If every step you could do is blocked or done,** use the time on checks: more hand-computed tests for B2, re-reading quotes against their sources, the seeded faults for X7, profiling the evaluator, proving the next Lean rule. Write down what you did.
- **Do not shrink or skip an experiment on your own** to save time. If something is taking much longer than expected, say so in the status file and keep going.

### Reports
- Keep `reports/status.md` current: one line per step with its state (not started, running, rehearsed, waiting at a stop, done), and at the top, the list of decisions waiting for Tomas, most important first.
- At each stop, write `reports/<stop>.md`: what finished, the question for Tomas, and the files to look at. Keep it to one page.
- Something that blocks every track goes at the top of the status file, marked as blocking.

### Machines
- **The virtual machine** runs everything except the two items below. It needs 16–32 cores, 64 GB of memory, 100 GB of disk, and Linux. At the start, write `notes/machine.md`: cores, memory, OS, tool versions, and the measured time of one evaluation once B2 has one.
- **Data:** it is already in `data/`. At the start, record a checksum of every file under `data/` in `notes/machine.md`, and check them again before B3.
- **Papers:** Tomas puts the PDFs in `library/sources/`. If one you need is missing, list it in the status file. Do not work from an abstract or a secondary source.
- **API access:** needed from X2 on. Check early that both Claude model IDs answer from this machine.
- **Not on the virtual machine:** the mini-coder-4b trial run (X4) needs a GPU, and the 3080 power measurement (X8) needs the 3080 machine. List them for Tomas when you reach them; do not block other work on them.

### If time runs short
Tomas decides, not you. The order of value is the Priority column: every "must" before any "should", every "should" before any "if time". If Tomas says to cut, the reduced plan is:
- finish B3, X1 and X3 on the library as it stands;
- finish B4 and run X4 on −TF only, with one model and 3 runs, plus X7;
- X8 on the walk's design from the full library;
- drop X5 and X6.

## 5. Rules for every step

1. **Do not invent numbers, and do not recall them.** A number enters the library only through B1b: you open the source, find the table or sentence, and record where it is. A number you remember from a paper is not checked until you have read it there.
2. **Do not change the model's time equations.** If a new cost is needed, add it as a new named term and tell Tomas.
3. **These thresholds are fixed.** Do not change them after seeing results.
4. **Report every run,** including failures. If a result contradicts the paper's hypothesis, report it. Do not tune until it matches.
5. **No leaks.** During any run, nothing the LLM can see (prompts, tools, library, logs) may contain a removed option or the best full-library design's placement. Both live in `library/removed/`, which the loop never loads.
6. **Prompt freeze.** Develop prompts only in the development world (`library/worlds/dev.yaml`). At the Freeze step, record the hash of `prompts/` and do not edit it again. Every run record stores that hash.
7. **Sessions.** Use the 45 search sessions everywhere except X8.
8. **Models.** `claude-sonnet-5-5` and `claude-haiku-5-5`, pinned by ID, and mini-coder-4b through llama.cpp (X4 only). Record the ID in every run record.
9. **Run index.** Runs 1, 2, 3 use seeds 1, 2, 3 where a seed applies. LLM calls are not exactly reproducible, so say so in the report.
10. **Run records.** Every run writes `runs/<step>/<arm>_<model>_<index>.json` with: config, seed, library hash, prompt hash, every proposal with its rationale, every gate result, every evaluation, accepted options with the reading behind each, final design, wall-clock, and input and output tokens per model.
11. **Cost check.** After the first full run of each model in X4, report its tokens and the projected total for X4 to X6. Wait for Tomas's go-ahead.
12. **Library versions.** Every result stores the library hash. A result computed with one library version is never mixed with another.
13. **Do not** commit or push unless asked, change raw traces or measurements, write paper text, or put author names in figures.

## 6. The stop point after X2

Write the STOP 4 report with the X1 and X2 tables. Tomas picks one path. Until he does, go on with B4, X3 and rehearsals:

| Path | Condition | What you run next |
|---|---|---|
| A | −TF ceiling ≤ 7.1×, and the readings arm finds at least 1 more mechanism on average or uses at least 30% fewer proposals for the same best speedup | Everything after STOP 4 |
| B | −TF ceiling ≤ 7.1×, and agents find mechanisms equally with or without readings | Everything after STOP 4, with X6 moved before X5 |
| C | −TF ceiling ≤ 7.1×, and agents rarely find any mechanism | X3 and X8 only, with X8 using the walk's design from the full library |
| D | −TF ceiling > 7.1× | Tomas chooses new removals from the B3 results. Repeat X1 and X2 |

## 7. The design space

The space is the old work's space with one change: embedding and sampling always run on the host. The agent reads every sampled token, and the model does not charge a token crossing, so a design that samples off the host would get a crossing for free.

| # | Knob | Values (walk's start in bold) | Active when | Term it relieves |
|---|---|---|---|---|
| 1 | `mvm`: weight MVMs of every layer | **crossbar**, host | always | Host, reads |
| 2 | `lm_head` | crossbar, **host** | always | Host, transfer |
| 3 | `attention`: q·Kᵀ, p·V, KV appends | **host**, crossbar, digital unit | always | Host, write, reads |
| 4 | `softmax` | **host**, digital unit, fixed-function unit | always | Host, transfer, launch |
| 5 | `vector`: norms, RoPE, SiLU, residuals | **host**, digital unit, fixed-function unit | always | Host, transfer, launch |
| 6 | Crossbar variant | **conventional**, transposable K writes | any of 1–3 on a crossbar | Write |
| 7 | Crossbar over-provisioning, relative to the need at the longest context | 0.5, 0.75, 1.0, **1.25**, 1.5, 2.0 | same | Capacity |
| 8 | Number of digital units | **1**, 4, 16, 64, 256 | any of 3–5 on the digital unit | Unit |
| 9 | Digital unit KV bandwidth, B/s | **64G**, 256G, 1T | attention on the digital unit | Unit |
| 10 | Digital unit KV memory, bytes | **4G**, 16G, 64G | attention on the digital unit | Capacity |
| 11 | Number of fixed-function units | **1**, 4, 16, 64 | softmax or vector on the fixed-function unit | Unit |
| 12 | Link | **CXL**, on-package, on-die | always | Transfer, launch |
| 13 | KV precision | **fp16**, int8 | always | Capacity, write |

- **Counts to reproduce:** 274,440 distinct designs after removing inactive knobs; −T 140,664; −F 48,120; −TF 24,648. These are the old sweep's rows with sampling on the host. If your enumeration gives other counts, stop and ask.
- **Held fixed:** mini-coder-4b at Q8_0 (36 layers, 8 KV heads); ISAAC's 128×128 arrays; the ISAAC tile (8 ADCs per IMA, 2-bit cells), except in the area study; 4-bit KV is excluded because its accuracy cost is not modeled.
- **Area study only (X1, X8):** tile (ISAAC, lean), ADCs per IMA (8, 4, 2), cell bits (2, 4). Together with knobs 6, 7, 11, 12 and 13 they form the 3,456-variant crossbar subspace.
- **Grid values of knobs 8–11** were chosen in the old work. B1a lists them as design ranges, not as sourced numbers, with a reason for each end of the range.

### Operator map: what runs where
One decode token of mini-coder-4b. Steps 1–13 repeat for each of the 36 layers. "?" means it is not decided yet: decide it from the unit's library entry in B1c and report it.

| Step | Operator | Crossbar | Digital unit | Fixed-function unit | Host | Timed by | Knob |
|---|---|---|---|---|---|---|---|
| Entry | Embedding | No | No | No | Yes | Profile | fixed |
| 1 | RMSNorm | No | Yes | Yes | Yes | Unit | 5 |
| 2 | Q, K, V projections | Yes | ? | No | Yes | VIPER t_comp | 1 |
| 3 | QK-Norm, RoPE | No | Yes | Yes | Yes | Unit | 5 |
| 4 | Append K and V | Yes (a write) | Yes | No | Yes | VIPER t_prog | 3 |
| 5 | Scores q·Kᵀ | Yes, after the write | Yes | No | Yes | VIPER t_comp | 3 |
| 6 | Softmax | No | Yes | Yes | Yes | Unit | 4 |
| 7 | Weighted sum p·V | Yes, after the write | Yes | No | Yes | VIPER t_comp | 3 |
| 8 | Output projection | Yes | ? | No | Yes | VIPER t_comp | 1 |
| 9 | Residual add, RMSNorm | No | Yes | Yes | Yes | Unit | 5 |
| 10 | Gate and up projections | Yes | ? | No | Yes | VIPER t_comp | 1 |
| 11 | SiLU(gate) · up | No | Yes | Yes | Yes | Unit | 5 |
| 12 | Down projection | Yes | ? | No | Yes | VIPER t_comp | 1 |
| 13 | Residual add | No | Yes | Yes | Yes | Unit | 5 |
| Exit | Final RMSNorm | No | Yes | Yes | Yes | Unit | 5 |
| Exit | LM head | Yes | ? | No | Yes | VIPER t_comp | 2 |
| Exit | Sampling | No | No | No | Yes | Profile | fixed |

**This map is not an assignment. The agents choose what runs where.**
- The four unit columns show what the library currently allows. They follow from each unit's `supports` field, so they change as options are added. The Knob column is for your checks only. The LLM never sees it.
- An agent changes placement by proposing a `mapping` option: one or more operators, moved together, to one unit, for all 36 layers or any chosen subset. A boundary between layers placed differently is charged like any other crossing in Eq. M2.
- The sweep can only place operators through knobs 1–5, which tie several operators together. Agents can place each step separately.
- "No" means no current option supports it. An agent may propose a new unit variant that makes it possible. The gates then decide. Embedding and sampling stay on the host whatever is proposed.
- In −F and −TF runs, the Fixed-function column is absent until an agent adds such a unit.
- A proposed option enters the library, then the walk decides whether to use it. It is kept only if time improves by more than 0.5%.

## 8. Steps

### B1: Build the library, carefully

Every result in the paper is a function of these numbers, so this step is slow on purpose. Work one group at a time and stop where the plan says.

#### Where a number can come from
A number has one of six origins. The library records the origin of every value.

| Origin | Meaning | What must be kept |
|---|---|---|
| `paper` | Read in a primary source | The PDF, the page, the table or section, the exact quote |
| `simulator` | Output of an architecture-level tool we ran, such as NeuroSim or Eva-CAM | The simulator rules below |
| `spice` | Output of a circuit simulation, ours or a collaborator's | The netlist or a pointer to it, the device model and its version, the corner, the testbench, the raw output, and who ran it |
| `measured` | Our own measurement on hardware | The data file and how it was taken |
| `derived` | A formula over other library values | The formula |
| `custom` | A value we provide | A reason, a low and high `range`, and Tomas's approval. Without approval it does not enter the library |

#### Several candidates, one choice
A number may have several candidate values from different origins: for example a crossbar read time from NeuroSim, from SPICE and from the ISAAC paper. The library keeps all of them. A separate file chooses which one is used.

- `library/primitives.yaml` holds, for each number, a list of `candidates`. Each candidate has its own origin and provenance.
- `library/profile.yaml` chooses one candidate per number. It sets a default origin per group and may override single numbers.
- The evaluator loads the library through a profile and never reads a candidate the profile did not choose.
- The library hash covers the profile. Changing a choice makes a new library version, and results from two versions are never mixed (§5, rule 12).
- Only Tomas changes `profile.yaml`. You may propose a change, with the effect on the affected numbers.
- A candidate that is not chosen is a cross-check. For every number with more than one candidate, report the ratio between them. If the chosen one differs from another by more than 2×, say so at the group's stop point. Never choose by which gives the better result.

#### Preferred sources
This table is the first `profile.yaml`. If the preferred origin can't give a number, stop and ask; do not fall back on your own.

| Group | Preferred | Other candidates to record |
|---|---|---|
| Crossbar: read path, write path, area, ADC | `simulator`: **NeuroSim**, run by us at the library's array size and cell | `paper`: ISAAC, ISCA 2016, Table I (the old values). `spice`: if Tomas provides results |
| CAM | `simulator`: **Eva-CAM**, the CAM evaluation tool from X. Sharon Hu's group, run by us | `paper`: CAM designs published by that group, which Tomas names. `spice`: if Tomas provides results |
| Transposable array | open. First check whether NeuroSim can model it. If not, Tomas names a published array or provides SPICE results | — |
| Fixed-function unit | open. Candidate: the STAR softmax engine paper | — |
| Digital unit | open. Candidate: PUMA, ASPLOS 2019 | — |
| Links (CXL, on-package, on-die) | open. Candidates: the specifications or measurement papers | — |
| Effort in months | `custom`: these are conventions, each with a stated reason | — |
| Host, GPU, workload | `measured`: our own data in `data/` | — |

"Open" rows are decided at STOP 1. Do not source an open group before then.

**Numbers from a simulator** follow these rules:
1. Pin the tool: name, version and commit. Keep it outside the library, and record how to rebuild it.
2. Keep every input file in `library/sim/<tool>/`, with the command that ran it and the raw output.
3. The simulator's own inputs need provenance too: the device parameters, the process node, the cell type, the ADC. Use the tool's shipped presets where they exist and record which. Any input you set yourself is listed for Tomas, with the reason.
4. Run at the library's conditions (128×128 arrays, 2-bit cells, the node the tool supports nearest to 32 nm). Do not rescale an output from another condition.
5. Each library entry points to the output line it was read from, in the `where` and `quote` fields.
6. Compare each simulated number with its cross-check value and report the ratio. If they differ by more than 2×, stop and ask before going on. Do not pick the one that gives the better result.

#### B1a: Inventory
- **Goal:** one list of every number the model needs, before any value is trusted.
- **Do:**
  1. From the equations in §11 and the formulas in `notes/b2_model.md` (B2, stage 0), list every symbol that needs a value, per unit type.
  2. From the old work (`data/oracle/tech/*.yaml`, `old/harness/space.py`, `data/host/`), list every number the old evaluator used: its ID, value, unit, the source the old file claims, and whether the old file marks it as checked, VERIFY, placeholder or assumption.
  3. Join the two lists. Flag any symbol with no number, and any number no equation uses.
  4. Mark each number's group: crossbar read path, crossbar write path, crossbar area, digital unit, fixed-function unit, transposable array, links, effort, host (measured), GPU (measured), workload.
  5. Add the sensitivity each number had in the old study (`data/oracle/sensitivity_main.json`), so the groups can be ordered by how much they move the result.
  6. For the crossbar group, list which rows NeuroSim can produce and which it can't. Do the same for the CAM group and Eva-CAM. For this, read each tool's manual and one example output; do not run anything yet.
- **Output:** `library/inventory.md`, one row per number.
- **Then STOP 1.** Tomas reviews the list, fills the "open" rows of the preferred-sources table, and says which SPICE results exist.

#### B1b: Source every number
- **Goal:** each number is read from its primary source, simulated, measured, or given by us and marked so.
- **Order of groups,** highest sensitivity first: crossbar read path; fixed-function unit; crossbar area; transposable array; crossbar write path; links; digital unit; effort.
- **For each number from a paper, do** (numbers from a simulator follow the simulator rules above):
  1. Get the primary source as a PDF into `library/sources/`. If you can't get it, say so. Do not use a secondary citation in its place.
  2. Find the value. Record the page, the table or section, and the sentence or table row, copied exactly.
  3. Record the conditions it was reported under: process node, array size, precision, measured silicon or simulation.
  4. If the library needs a different unit or condition, write the conversion as a formula, as a `derived` candidate. Do not round until the end.
  5. If two sources disagree, record both and ask.
- **Each entry in `library/primitives.yaml`:**

  ```yaml
  - id: xbar_read_s
    unit: s
    group: crossbar_read
    candidates:
      - name: neurosim
        origin: simulator
        value: null            # filled from the run
        source: "NeuroSim <version>, commit <hash>"
        where: "library/sim/neurosim/<run>/output.txt, line <n>"
        quote: >
          <the output line, verbatim>
        conditions: "<node>, 128x128, 2 b/cell"
      - name: isaac
        origin: paper
        value: 1.0e-7          # 100 ns
        source: "Shafiee et al., ISAAC, ISCA 2016"
        where: "Table I; Sec. V, p. 20"
        quote: >
          <the sentence or table row, verbatim>
        conditions: "32 nm, 128x128, 2 b/cell, simulated in the paper"
  ```

  And in `library/profile.yaml`:

  ```yaml
  default_origin:
    crossbar_read: simulator
    cam: simulator
    host: measured
  override:
    xbar_write_row_s: spice     # one number taken from another candidate
  ```

  A `derived` candidate adds `derivation`. A `custom` candidate adds `reason`, `range` and `approved_by`.

- **File formats.** The library files that Tomas reviews by hand are YAML: primitives, options, catalog, worlds. Everything a program or an LLM writes is JSON: agent proposals, run records, the sweep grid.
  - **Check types on load.** YAML reads `100e-9` as text, not as a number. Write values with a decimal point (`1.0e-7`). The loader rejects any `value` that is not a number; it never converts one.
  - **One schema.** The loader validates YAML library files and JSON proposals against the same schema, so an option an agent adds is the same object as one written by hand.
  - **Hash the parsed content,** not the file text. A comment or a reformat does not change the library version. Any value does.

- **A candidate is complete** when it has everything its origin requires in the "Where a number can come from" table. An incomplete candidate cannot be chosen by the profile.
- **Numbers that need the most care,** because the old values were placeholders:
  - **Fixed-function unit (STAR):** throughput, area and datapath width, from the STAR paper. This unit carries most of the speedup, so nothing here may be a custom value without Tomas seeing the paper's numbers first.
  - **Transposable array:** find a published transposable or bidirectional array (ReRAM or SRAM) and take its area overhead and write cost from it. The old +15% area was a guess.
  - **Links:** bandwidth and latency for CXL, on-package and on-die, each from a specification or a measurement paper.
  - **Digital unit (PUMA):** lane width and clock, from the paper's tables.
  - **Crossbar write:** row write time and how many arrays can be written in parallel.
- **Process node:** record each number at its own node. Do not scale between nodes in the library. Node scaling is a separate, documented step in X8.
- **Output:** `library/primitives.yaml`; `library/sources/`; `library/review_<group>.md` per group, a table of ID, every candidate with its origin, value, where and quote, the chosen one marked, and the ratio between candidates, written for Tomas to check against the source in a few minutes.
- **Then STOP 2,** once per group: write the review file. The group enters library v1 only after approval; go on with the next group meanwhile.

#### B1c: Options, reduced libraries, freeze
- **Part 1, options.** Write `library/options.yaml`. Every option has `id`, `kind`, `uses` (primitive IDs) and the fields for its kind:

  | Kind | Fields |
  |---|---|
  | `mapping` | `operators` (one or more IDs from the operator map, moved together), `unit`, `layers` (optional; default all 36) |
  | `rewrite` | `before`, `after` (operator graph terms) |
  | `unit_variant` | `supports` (operator classes), `precisions`, `costs` (latency, write, area, energy as expressions over primitive IDs and the state variables *L*, *n*) |
  | `kv_policy` | `precision_by_state` |

- **Part 2, the schema must express the mechanisms as data.** No field or flag may name a mechanism (`transposable: true` is forbidden), and the evaluator may not branch on one. A K append's write cost must follow from general fields: how the matrix is laid out in the array, and along which lines the array can write. Write T and F by hand in this schema, show both to Tomas, and only then build the rest. If the schema can't express one of them, stop and ask.
- **Part 3, reduced libraries.** Move the T and F options to `library/removed/`. Write a loader that takes a library name (`full`, `-T`, `-F`, `-TF`).
  - −T keeps the cell write latency, the array geometry, and every primitive T was built from.
  - −F keeps the catalog from part 4 and every primitive F was built from.
- **Part 4, catalog.** Write `library/catalog.yaml`: 5 published hardware blocks, each sourced by the B1b procedure. One is the STAR softmax engine. The other 4 are published blocks that do not compute softmax or norms. An agent may use a catalog block only by citing its `id`. If you can't source 4, use what you have and report the count.
- **Part 5, matching rules.** Write `scripts/match_mechanisms.py`, applied only after a run. An accepted option counts as:
  - **T** if it cuts a K append from *d_head* row writes to O(1) writes per token and declares an area cost;
  - **F** if it runs softmax and norms beside the arrays on the STAR catalog block.
- **Part 6, worlds.** Numeric overrides on the base file: `dev.yaml` (each `custom` number set to a random point inside its range, and each number with several candidates set to a random one of them, seed 0), `W1.yaml` (cell write latency ÷100), `W2.yaml` (link latency and bandwidth 10× better, host element-wise operators 10× faster).
- **Part 7, freeze.** Write `library/VERSION` with the hash of the library files and a one-page summary: how many chosen numbers have each origin, and the list of custom ones. This is library v1. A later change makes v2 and reruns everything that used v1.
- **Done when:** every number in `primitives.yaml` has a complete candidate chosen by `profile.yaml`, and no `custom` candidate is chosen without approval.

#### B1d: More devices, CAM and code search (if time)
Only after X4. Add `library/devices.yaml` (ReRAM, FeFET, PCM, SRAM-based compute: the VIPER inputs, energy and endurance, each sourced as in B1b) and a CAM entry. Classify traced tool commands as code search or not, report the Amdahl bound if all code-search time were removed, and add a `code_search` operator allowed on the host or a CAM with a new named term `cam_search`. On the host its time stays the traced time, so no earlier result changes. The CAM numbers come from Eva-CAM, as the preferred-sources table says.

### B2: Build the evaluator
- **Goal:** a new evaluator that implements the equations in §11, reads every number from the library, and is tested before anything depends on it.
- **Start at once,** in parallel with B1. Until a group is approved, load its old values from a file named `library/provisional.yaml`. Anything computed with provisional values is a rehearsal (§4).
- **Build in this order,** with tests at each stage. Do not start a stage before the previous one passes.
  0. **Write the model down first.** §11 gives the model's structure, but not how each operator's time, each unit's capacity, or area and effort are computed. Those formulas exist only in the old code (`old/harness/tech.py`, `integrator.py`, `session.py`, `space.py`) and partly in `old/docs/harness.md` and `old/docs/theory.md`. Read them and write `notes/b2_model.md`: every formula in mathematical notation, with the symbol IDs from the inventory, the old function it came from, and one worked example with numbers. Cover the list in §11.4. Where the old code does something §11 does not say, or contradicts it, list it under "Decisions for Tomas" and do not silently adopt it. Then write the new code from this document, not from the old code. Tomas reviews the document at STOP 3.
  1. **Workload.** The operator graph of one decode token and of a prefill step, built from the model's `config.json`: operators, op counts, operand sizes, the tensors each reads and writes. Test: op counts and bytes per token match `data/workload/` and a hand count for one layer.
  2. **One operator on one unit.** Crossbar by VIPER's terms; digital and fixed-function units by their own timing; host by the profile. Test: each matches a hand-computed value, and one operator offloaded to a crossbar matches Eq. M1.
  3. **One step.** Eq. M2: crossings, launches, the internal network, KV writes. Test: a hand-computed step for three small designs.
  4. **Capacity and the wall.** Resident data against capacity at a given context. Test: the context where a hand-sized design stops fitting.
  5. **Session replay.** Eq. M3 over a trace, with exact replay first, then sampled curves. Test: sampled within 0.3% of exact; GPU replay within 3% of measured decode time (§11.6).
  6. **Readings.** Time terms, binding term, headroom per term, Amdahl gap, wall crossings. Test: the terms sum to the total; headroom is zero for the binding term.
  7. **Area and effort.**
  8. **Options.** Price `unit_variant` cost expressions, accept new `mapping` options, accept a placement per layer.
- **One interface.** `evaluate(design, library, sessions)` returns `time_s`, `speedup`, the time terms, `binding_term`, the readings, `area_mm2`, `effort_months`, `feasible` and `infeasible_reason`. Every later step calls only this. The same inputs give identical outputs.
- **External checks:** the UPMEM prediction is 2.42× (§11.6), with the UPMEM numbers from the VIPER case study.
- **Oracle check.** Load the old numbers as a library (`library/oracle_old.yaml`, used for this check only). Evaluate 2,000 designs: 1,000 at random from the old sweep's rows with sampling on the host, the 100 fastest, and 900 at random from infeasible rows. Compare with the old `time_s`, `feasible` and `area_mm2`.
  - Target: feasibility agrees on every design, and time and area agree within 1%.
  - Write `notes/b2_oracle.md`: the distribution of differences, and for every design outside 1%, which term differs and whether §11 supports the old code or yours.
  - Do not change your equations to match the old code unless §11 says the old code is right. Known old behaviors that §11 does not state are listed in `old/docs/harness.md` and `old/docs/harness-dev/REPORT-M3.md` (crossbar prefill pipelined across layers; one buffer per fixed-function instance; weights past the wall reprogrammed every step; tool times net of container overhead). Report each one you adopt as a decision.
- **Batch runner.** Runs on 20 cores, writes shards, and can resume after a crash.
- **Keep `notes/b2_model.md` current:** next to each formula, add the new function that implements it and the test that checks it.
- **Then STOP 3:** write the report with `notes/b2_model.md` and `notes/b2_oracle.md`. The real sweep waits for the answer; rehearsals do not.

### B3: Sweep the space
- **Goal:** one file holding every design in §7's space, evaluated with the new evaluator and library v1. X1 and X3 query it.
- **Do:**
  1. Write the knobs and values to `sweep/grid.json`. Enumerate the designs, removing inactive knobs, and count them. The count must be 274,440.
  2. Evaluate every design, feasible or not. Evaluate the −TF designs first, then the rest of −T and −F, then everything else, so X1 can start early. `design_id` is a stable hash of the active knob values.
  3. Re-evaluate the 100 fastest feasible designs with exact replay. Report the largest difference from sampled replay.
  4. For the fastest design, list every numeric knob (7–11) that sits at the smallest or largest grid value.
- **Output:** `results/sweep_full.parquet`, one row per design:
  - `design_id`, every knob (inactive knobs empty), the library hash;
  - `feasible`, `infeasible_reason`;
  - `time_s`, `speedup`, `area_mm2`, `effort_months`;
  - one column per time term (host, transfer, launch, write, unit compute, reads, capacity passes), summed over the 45 sessions;
  - `binding_term` (largest term over the session total) and `binding_term_4k`, `_16k`, `_32k`, `_64k`;
  - for X3, summed over the 45 sessions: `total_ops`, `bytes_host_to_pim`, `bytes_device_memory`; and per design: `peak_compute_rate`, `memory_bandwidth`, `link_bandwidth`.
- **Done when:** 274,440 rows. Report the feasible count and the best speedup next to the old values (131,256 and 9.47×). They will differ, because library v1 replaces placeholder numbers. Explain the difference by term. Stop and report if the fastest design sits at the largest value of a numeric knob.

### X1: Ceilings of the reduced libraries
- **Question:** how much does removing each mechanism cost?
- **Do, for each of `full`, −T, −F and −TF:**
  1. **Filter** `sweep_full` to feasible designs in that library: −T, the crossbar variant is conventional; −F, no operator on the fixed-function unit; −TF, both.
  2. **Apply equal area.** Let *A_max* be the area of the fastest design in `full`. Keep only designs with area at or under *A_max*.
  3. **Take the fastest** remaining design.
  4. **Check the grid edge.** If the fastest design is at the largest grid value of a numeric knob (7–11), double that knob's largest value, evaluate only the new designs with the other knobs fixed, and repeat until the fastest design is inside the grid or the next step would exceed *A_max*. The result is the library's ceiling.
- **Then, under an area cap:** take the 10 fastest designs from step 3 for each library. For each, evaluate the 3,456 variants of the crossbar subspace (§7), and take the fastest at or under the cap. The cap is the area of the fastest `full` variant with the lean tile and 4-bit cells (3,692 mm² with the old numbers; recompute it).
- **Output:** `results/x1_ceilings.csv`: `library`, `ceiling_speedup`, `design_id`, `binding_term`, `area_mm2`, `extended_grid` (true or false, and which knobs), `capped_speedup`, `capped_design_id`.
- **Done when:** the file has 4 rows. Send it to Tomas at once.

### X2: Quick test
- **Question:** can agents find the mechanisms, and do readings change that? This is a signal for Tomas, not a paper result.
- **Setup:** −TF library, no gates. Sonnet and Haiku, runs 1–3, two arms = 12 runs.
- **Arms:** `readings`: the LLM sees the readings. `latency`: the LLM sees only total time and speedup.
- **Each round:**
  1. Run the walk on the current library to get the best design.
  2. Show the LLM the library, that design and the arm's information.
  3. The LLM returns one option as JSON. If it is malformed, return the error; it still counts as a proposal.
  4. Add the option, with the time term it says it relieves added to the walk's moves, and rerun the walk. Keep it if time improves by more than 0.5%. Otherwise remove it.
- **Stop a run** at 30 proposals, or after 10 proposals in a row with no kept option.
- **Output:** run records and `results/x2_summary.csv`: `arm`, `model`, `index`, `mechanisms_found`, `best_speedup`, `proposals_used`, `tokens`.
- **Then STOP 4** (§6).

### B4: Gates and the full loop
Start the Lean setup at once; it does not depend on the library.

- **G1 correct.**
  - Give each operator a meaning in Lean 4 + Mathlib (ℝ, `Matrix`, `Fin` vectors). A rewrite is a term: operator graph before → after.
  - Generate the statement ⟦before⟧ = ⟦after⟧ from the rewrite. The agent writes only the proof.
  - Prove the base rules in this order: L1 tiling an MVM across crossbars; L2 transposed storage; L5 split softmax with rescaling; L3 partitioning heads or layers; L4 commuting and fusing operators.
  - Prove the composition lemma: rewriting a subgraph preserves the whole graph.
  - A proof counts only if `lake build` succeeds with no `sorry`, `admit`, new `axiom` or `native_decide`, and `#print axioms` shows only `propext`, `Classical.choice` and `Quot.sound`.
  - Only `rewrite` options need G1. Quantization is a declared operator, checked by G2.
- **G2 legal.** A decidable checker. Each failure has a name:
  - `support`: the unit's `supports` lacks the operator's class;
  - `precision`: a bit width the operator or the unit does not allow. The allowed KV precisions are exactly 16 and 8;
  - `write`: a changing operand (the KV) on a unit with no write cost;
  - `host_only`: embedding or sampling off the host;
  - `dependency`: an operator reads a tensor nothing earlier produced;
  - `dataflow`: a KV appended on one unit and read on another;
  - `crossing`: a boundary crossing the evaluator does not charge;
  - `capacity`: resident data over capacity at some state of some session;
  - `cap`: over an area or effort cap.
- **G3 sourced costs.** Every primitive ID in an option's `uses` and `costs` exists, and every constant in a cost expression is a primitive ID or a catalog `id`. A bare number fails. An option that uses an `custom` primitive passes with a flag, and every result built on it carries that flag.
- **G4 true rationale.** The agent states its claims in fixed fields: the binding term, the term its option relieves, the expected gain. Each is checked against the readings: the binding term must match, and the gain must be within a factor of 2 of the measured one.
- **Loop.** Walk → readings → supervisor routes to the agent that owns the binding term (mapping, hardware or capacity) → proposal → gates → add to the library → walk again → log. A failed gate returns its reason to the agent, with 2 retries. A kept option must pass the gates and improve time by more than 0.5%. Stop at 30 proposals, after 10 in a row with none kept, or when time is within 2% of the Amdahl floor.
- **Done when:** with the `full` library and no LLM, the loop returns the walk's design; T and F written by hand pass every gate that applies; and the seeded faults of X7 are in place.
- **If the rewrite-to-Lean generator is the last thing holding B4 back:** ship L1 and L2 proved, apply G1 by hand to those two rewrites, and say so.

### X4: Rediscovery
- **Question:** does PIMDA beat the ceiling?
- **Setup:** full loop. Libraries −T, −F, −TF × Sonnet, Haiku × runs 1–3 = 18 runs.
- **The 4B model:** one run on −TF first. Add its other 8 runs only if that run keeps at least one option.
- **Output:** `results/x4_summary.csv`: `library`, `model`, `index`, `ceiling`, `final_speedup`, `mechanisms_found`, `proposals`, `evaluations`, `tokens`, `uses_custom_numbers`. Report the median and range per library and model.
- **Also:** for the best −TF run, write speedup after each proposal, with each kept option and its reading, to `results/x4_trace.json`.

### X7: Gate checks
- **Question:** do the gates catch mistakes and accept correct work?
- **Before X4 starts,** write at least 5 wrong proposals per gate in `tests/seeded_faults/`. Include:
  - G1: split softmax without rescaling; a tiling that drops a block;
  - G2: softmax on a plain crossbar; sampling on a digital unit; attention on a crossbar with no write cost; KV over capacity at 64K tokens; KV at 4 bits;
  - G3: a made-up constant; a cost citing a primitive that doesn't exist;
  - G4: a rationale naming the wrong binding term; an inflated expected gain.
- **Measure:**
  1. share of seeded faults caught per gate;
  2. false rejections: T and F written by hand, and L1–L5, must pass;
  3. pass and fail counts per gate across all X4 and X5 proposals, with the full text of each wrong one;
  4. from `gates_off` (X5): how many kept options would fail each gate, and how much that inflates the speedup;
  5. run every rewrite on random inputs, before and after, and flag any disagreement with the Lean result;
  6. how many kept options had a Lean proof, and how many were exempt (approximations, unit variants, catalog choices). Report exempt options as not proved.
- **For Tomas:** write every kept option from reported runs to `results/x7_audit.md`, with its cost expressions and the origin of every primitive it uses. Tomas reads these by hand, because G3 can pass a composition that is sourced but physically made up.
- **Output:** `results/x7_gates.csv` and `results/x7_audit.md`.

### X3: Roofline against the model
- **Question:** how often does Roofline name the wrong bottleneck? Start any time after B3.
- **Do,** for each feasible design, using session totals of operations and bytes:
  - **Plain Roofline:** rate = min(peak compute rate, operational intensity × device memory bandwidth). It names "compute" or "memory".
  - **Roofline plus link:** the same, with the host link's bandwidth as a third limit.
- **Compare with `binding_term`:** unit compute and reads count as compute; transfer and launch count as memory or link; write, capacity and host are "not expressible".
- **Output:** `results/x3_roofline.csv`, with these numbers per predictor:
  - share of designs whose binding term is not expressible, split by term;
  - disagreement among the rest;
  - Spearman rank correlation with the model's time;
  - the model's rank of Roofline's best design;
  - the same four numbers using `binding_term_4k` and `binding_term_64k`, because the binding term can change along a session.
- **Also:** pick one design from the most common disagreement type, not the most extreme, and write the data for a side-by-side figure to `results/x3_example.json`.
- **Expect:** in the old work, the host term caused nearly all the disagreement, and Roofline agreed with the model where it could name the term. If that holds, report it as it is.

### X5: Ablations
- **Question:** which part of PIMDA does the work?
- **Setup:** −TF. Sonnet and Haiku, runs 1–3. Change one thing per arm:

  | Arm | Change from the full loop |
  |---|---|
  | `full` | None; reuse the X4 runs |
  | `no_readings` | The LLM sees only total time and speedup |
  | `single_agent` | One agent owns every kind of option; no routing |
  | `random_routing` | The agent for each round is picked at random |
  | `eat_supervisor` | An LLM supervisor picks the agent and revises its instructions from the history, with no readings and no term-based routing, as in *Evolving Agent Teams* (arXiv 2610.36848) |
  | `gates_off` | Keep any schema-valid option that improves time |
  | `enumerator` | No LLM. Enumerate every schema-valid option built from at most 2 primitives or catalog blocks, plus every mapping of one operator or two adjacent operators to each unit, for all 36 layers. Send each through G2 and G3, then the walk |

- **Enumerator:** run it twice, once capped at the median number of evaluations `full` used, and once uncapped up to 12 hours on 20 cores. Report the total size of the enumeration.
- **Output:** `results/x5_summary.csv`, same columns as X4 plus `arm`.

### X8: The final design
- **Question:** is the design credible? Use the best −TF run from X4. On path C, use the walk's design from the full library.
- **Do:**
  1. Re-optimize the final library with a sweep if under 24 hours on 20 cores, otherwise with the walk from 3 starts. Report whether the design changes.
  2. Take the fastest variant at or under the X1 area cap, and also report the fastest with no cap.
  3. Report for both: speedup on the 44 held-out sessions and on the 11 later sessions, against the RTX 4090 and H100 (scaled as in §11.7), and speedup per unit area against the 3080.
  4. **Sensitivity to the library:** move every `custom` number to each end of its range, and switch every number that has several candidates to each other candidate, one at a time, and report the speedup. Then report the speedup with every custom number at its pessimistic end together.
  5. Energy per token and per session, if energy was sourced in B1b for every unit the design uses. Measure the 3080's power during decode; if you have no access to that machine, stop and ask.
  6. Scale area to 22 or 12 nm using a sourced ADC-area rule. If you can't find one, report 32 nm and say so.
- **Output:** `results/x8_design.json`, the full log, the term breakdown along the session, and `results/x8_summary.csv`.

### X6: Readings without memory (if time)
- **Question:** do readings help when the LLM's memory of the literature can't? Sonnet and Haiku, runs 1–3, arms `readings` and `latency` (as in X2), full gates.
- **Setups:**
  - **Anonymized:** the −TF library with every operator and unit renamed to an ID (`op_07`, `unit_B`) and descriptions reduced to numbers. Keep the name map outside the loop, and apply it only when scoring.
  - **W1:** −TF in world W1, where transposable writes should not pay.
  - **W2:** −TF in world W2, where the fixed-function unit should pay much less.
- **Check each world before any LLM run:** restore the relevant mechanism alone and run the walk (T for W1, F for W2). It must gain under 2%. If it doesn't, multiply that world's changes by 10 once. If it still doesn't, stop and ask.
- **World best:** run the walk with both mechanisms restored, in that world.
- **Output:** `results/x6_summary.csv`: `setup`, `arm`, `model`, `index`, `final_speedup`, `speedup_over_world_best`, `share_of_proposals_that_gain` (over 0.5%), `share_of_proposals_matching_a_mechanism_that_does_not_pay`, `tokens`.

## 9. Out of scope

Verilog, RTL or synthesis; options outside what the schema can express; per-layer knobs in the sweep; a second workload; calibrating the model's equations against a circuit simulator (NeuroSim supplies the crossbar's numbers in B1b; it does not replace the model); proving the performance model in Lean; node scaling inside the library; writing paper text.

## 10. Reporting and when to stop

**After each step,** report what ran, the numbers, what differs from what this plan expects, and any result that changes a claim. Keep it short.

**Stop that step, put the question at the top of the status file, and go on with other work, if:**
- a source can't be obtained, or its number differs from the old value by more than 2×;
- a number has no source and no approved custom value;
- the oracle check fails and §11 doesn't settle which side is right;
- a term, knob or count this brief names doesn't come out as stated;
- a step needs a decision this brief doesn't make;
- Lean setup blocks G1 for more than a day;
- every step you could do is blocked on an answer from Tomas.

Do not fill a gap by guessing.

## 11. The model, the walk and the facts from the paper

This section replaces the paper as your reference. It gives the structure of the model. The formulas below this level are written down in B2, stage 0.

### 11.1 VIPER's closed form (Eq. M1)
For one kernel offloaded from a host to a PIM device:

```
T = t_host + m * (t_in + t_prog + t_comp + t_out),    m = ceil(D / C)
```

- `t_host`: host time outside the kernel. `t_in`, `t_out`: moving data to and from the device. `t_prog`: writing data into the device. `t_comp`: computing in the device.
- `D`: the data the kernel holds in PIM. `C`: the device's capacity. `m` is the number of passes when the data does not fit.
- In this work, Eq. M1 describes crossbars (and a CAM, if one is added). Digital and fixed-function units use their own timing from the library.

### 11.2 Time of one step (Eq. M2)
Inputs:
- **Workload:** a graph `G(q)` of operations with their data dependencies. The state `q` holds the sizes that set each operation's work and storage. For the coding agent, `q = (n, L)`: `n` new tokens at context length `L`. Decode has `n = 1`.
- **Design `d`:** a mapping `pi` from each operation to a unit, the size of each unit, the host link, and the data precisions.
- **Library:** for each unit, the operations it supports, its execution and write costs, its storage geometry, its area and its effort.
- **Host costs:** profiled at the workload's operation sizes.

```
t(q, d) =   sum over operations o in G(q) of   t_o^{pi(o)}(q)
          + sum over host<->coprocessor transfers o in X(d) of   ( t_0 + alpha * N_{pi(o)} + B_o / BW )
          + t_rp(q, d)
          + B_noc / BW_noc
          + n_up * t_launch + n_down * t_sync
```

- The first sum is the operations, including writes to stored data.
- `X(d)` is the set of transfers across the host link. Each pays a fixed latency `t_0`, a scatter cost `alpha * N` where the unit's technology needs one (`N` = number of units), and `B_o / BW` for `B_o` bytes over a link of bandwidth `BW`.
- An operation on the host between two operations on the coprocessor costs a transfer in each direction.
- `t_rp` is reprogramming caused by capacity (§11.3).
- `B_noc / BW_noc` is data moved between units inside the coprocessor.
- `n_up` and `n_down` count launches (host to coprocessor) and synchronizations (back).
- Operations explicitly grouped as parallel contribute the latency of the group, not the sum. Everything else is serial.
- **Time terms.** Grouping these costs gives the terms `T_k`: host, compute per unit (reads on crossbars, unit compute on the others), transfer, launch, write, and capacity passes.
- **Check:** with one operation offloaded, Eq. M2 reduces to Eq. M1.

### 11.3 Capacity and the wall
At each state `q`, count the data that must stay on each unit: weights, the KV cache at that context length, and intermediate storage. The wall is where this exceeds the unit's capacity. Past the wall, a design is infeasible unless the unit supports partitioning and reprogramming, in which case it pays `t_rp`. In this plan a design is feasible only if its resident data fits at every state of every session in the trace.

### 11.4 Session replay and readings (Eqs. M3, M4)
A run `s` is a sequence of states `q_{s,i}` with host time `tau_{s,i}` outside the graph, which no design removes.

```
T_s(d) = sum over i of ( t(q_{s,i}, d) + tau_{s,i} )                      (M3)
S(d)   = ( sum over s of T_s^base ) / ( sum over s of T_s(d) )
```

For the coding agent, each traced turn `i` records prefilled tokens `p_i`, starting context `L_i`, generated tokens `g_i` and tool time `tau_i`:

```
T_s(d) = sum over i of ( t_pre(p_i, L_i, d) + sum_{j=1..g_i} t(L_i + j, d) + tau_i )     (M4)
```

- `t(L, d)` is Eq. M2 for one decode token. `t_pre` is prefill, which pipelines tokens across layers.
- The baseline is the measured RTX 3080, at its best KV configuration for each session: sessions that pass its fp16 limit were re-run with int8 KV at 64K. The GPU runs the same Q8_0 weights.
- The same states and the same `tau` are used for every design.
- Sampled curves over context length may replace the exact sum if they stay within 0.3% of it.

Readings, over the sessions being evaluated:
- **Binding term:** `b = argmax_k T_k`, and the ranking of the other terms.
- **Headroom of term k:** `T_b - T_k`.
- **Amdahl bound:** `S_max = (sum_s T_s^base) / (sum_s sum_i tau_{s,i})`. No design can pass it.
- A session that crosses the wall, or a design over its area cap, is also read as a binding term.

**What B2, stage 0 must write down,** because this section does not:
1. the time of an MVM on a crossbar: read cycles, input bits, the adder tree, how arrays work in parallel;
2. the time of an operation on a digital unit and on a fixed-function unit: operations, throughput, memory bandwidth, per-operation overhead, how several units share work;
3. the cost of a KV append on each unit, and how many arrays are written in parallel;
4. `t_rp`;
5. how weights and KV bytes become a number of arrays or bytes on each unit, and how over-provisioning sizes them;
6. which operations are grouped as parallel;
7. prefill, and how the sampled curves are built;
8. how tool time is taken from the traces, including the container overhead that is subtracted;
9. area and effort of a design;
10. the three area-study knobs: what each changes in time, capacity and area.

### 11.5 The walk
The walk is the deterministic optimizer. `T_tau` is the total host time no design removes. `epsilon = 0.5%`.

```
d <- d_0                                  # the start design: the bold values in §7
while T(d) - T_tau > 0.02 * T_tau:
    for each term k in readings(d), largest first:
        for each move m listed for k:
            d' <- m(d)
            if d' is faster but over the area cap: d' <- Repair(d')
            if d' is feasible and T(d') < (1 - epsilon) * T(d):
                log (m, k); d <- d'; restart the while loop
    stop                                  # no move on any term helps
Shrink(d)
```

Moves for each term. P moves the point, R raises the roof, W moves the wall:

| Binding term | Moves |
|---|---|
| Host | move the most expensive host operator off the host (P) |
| Transfer or launch | move the operators that cause round trips (P); a faster link (R) |
| Write | a cheaper write mechanism (R); remap the operations that write most (P); lower the data precision where allowed (P) |
| A unit | more units, more bandwidth, or a fixed-function unit (R) |
| Reads | more read parallelism (R); remap the operations that read most (P) |
| Capacity | more storage (W); a smaller representation (P, W) |

- The walk takes the first move that improves time by more than `epsilon`, and logs the term that proposed it.
- It stops when the remaining time is within 2% of `T_tau`, or when no move helps. If `T_tau` is zero, only the second rule applies.
- **Shrink:** step down every knob that costs area or effort while time stays within 1%.
- **Repair:** under an area cap, a faster move that breaks the cap is paid for with the area reduction that costs the least time: lean tile, 4-bit cells, fewer ADCs, less over-provisioning, int8 KV.
- An effort cap removes moves directly.
- When an agent's option is added to the library, the moves it makes possible are added under the term it says it relieves.

### 11.6 Validation targets
- **UPMEM:** with the UPMEM numbers from the VIPER case study, the model predicts 2.42× for a Llama2 layer, against 2.46× measured.
- **GPU replay:** replaying each trace through the 3080's measured decode curve predicts that session's measured decode time within 2–3%.
- **Sampled replay:** within 0.3% of exact replay.

### 11.7 Sessions and the stronger GPUs
- 100 sessions were traced. The first 89 form the frozen workload, split by a seeded shuffle into 45 search sessions and 44 held-out sessions. The split is the file `data/traces/workload_main.tsv`. 11 sessions collected later form a second held-out set.
- **RTX 4090 and H100:** scale the 3080's measured decode time by the ratio of memory bandwidths, and its prefill time by the ratio of tensor throughputs. Tool time is unchanged. Assume no memory limit on the larger GPUs. The bandwidth and throughput of each GPU are library numbers and are sourced in B1b like any other.
