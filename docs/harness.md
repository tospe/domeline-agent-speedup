# Harness design: optimizing an agentic workload on a PIM coprocessor

Oct 22, 2026. Working design; decisions marked **Open** are not settled.

## 1. What the paper shows

1. **Domeline**, the analytical model (VIPER's closed form, drawn as a chart). Checked against UPMEM.
2. **A harness** that uses Domeline to design a PIM coprocessor for a given workload. It is workload- and technology-agnostic.
3. **A design** for an agentic workload, found by the harness, that runs sessions faster than an RTX 3080.

UPMEM checks that the model is right. It is not a candidate design and is not in the harness loop.

## 2. System model

- A CPU runs the agent loop, the tools, and every op not offloaded.
- A PIM coprocessor sits next to the CPU and shares memory. It holds one or more technologies: crossbars, CAM, a digital vector unit, ASIC units taken from the literature.
- The coupling is VIPER's task-offloading mode: the CPU launches a kernel, data moves over the CPU–PIM link, and the result comes back.
- Each offload pays a launch cost plus transfer. Transfer grows with the number of units, `t0 + α·N`, a term the UPMEM case study found.
- Within a token every step depends on the one before, so times add up (no overlap).

## 3. Pipeline

```
workload spec ─► traces ─► per-token op graph(L) ─┐
CPU profile (measured once, per op) ──────────────┤
tech library (YAML, cited) ───────────────────────┼─► integrator ─► Domeline readings ─► search agent
design (mapping + sizing) ◄────────────────────────────────────────────────────────────────┘   │
                                                                     candidate ─► gem5 (budgeted)
```

### 3.1 Workload spec

The agent harness, the model, the task set, the step limit, and the engine settings. The harness only consumes the traces this produces.

Trace schema, one record per turn:

- `ctx_start`, the context length in tokens at the start of the turn
- `prefill_tokens`, `cached_tokens`, `gen_tokens`
- `t_prefill`, `t_decode` (measured on the GPU)
- `tool_cmd`, `t_tool`, `tool_out_tokens`
- `session_id`, `task`, `success`

### 3.2 Per-token op graph

Built from the model's `config.json`. For decode at context length L it lists every op with its class, FLOPs, bytes read and written, and output size. Op classes are `mvm_static`, `mvm_dynamic` (attention over the KV cache), `elementwise`, `reduction`, `kv_append` (a write), `search`, and `control`.

### 3.3 CPU profile

Measured once per host and model, with each op's CPU time recorded as a function of L. For any mapping, `T_stub` is the sum over the ops that stay on the CPU, so moving an op needs no re-measurement. Tool times come from the traces and are reused for every design.

### 3.4 Tech library

One YAML entry per technology or ASIC. Every number carries a citation. Users add entries without changing code.

```yaml
- name: reram_crossbar_isaac
  source: "Shafiee et al., ISCA 2016"
  kind: pim                      # pim | cam | digital | asic
  ops:                           # supported op classes and their cost
    mvm_static: {latency_ns: 100, bits_in: 8, bits_out: 8}
  array: {rows: 128, cols: 128, bits_per_cell: 2}
  write: {latency_ns_per_row: 1000, endurance: 1e8}
  area_mm2_per_array: 0.0003
  energy_pj_per_op: null
  effort_months: 0               # cost of adopting it in a design
  maturity: published-silicon    # published-silicon | simulated | projected
```

### 3.5 Design

A design is:

- the mapping from each op to a technology, or to the CPU;
- the sizing per technology (array count, and unit throughput for a digital unit or ASIC);
- the attach parameters (link bandwidth, launch cost, `t0`, `α`);
- the KV precision.

### 3.6 Integrator

VIPER's closed form, generalized to many ops, many technologies, and a whole session.

- `step_time(L, design)` adds up each op's time on its unit, transfer and launch at every CPU–PIM crossing, KV append writes, and programming when the weights don't fit.
- `session_time(trace, design)` sums the decode steps over the trace's context path and adds the tool times.
- It checks capacity, meaning where the wall falls against the longest session.
- It returns a breakdown by term: host, transfer, compute per technology, writes, capacity.

### 3.7 Domeline readings

These come from the integrator's breakdown:

- the binding term at each L;
- the headroom to every other term;
- the Amdahl bound (tool time plus the CPU ops that cannot be offloaded);
- the GPU sheet (3080 time for the same work).

### 3.8 Search agent

An LLM agent with these tools: `profile`, `domeline_eval(design)`, `gem5_eval(design)` (budgeted), `lookup_tech`, `lookup_asic`, `propose_move`.

- **Moves:** remap an op, resize a technology, add a technology or ASIC, change KV precision, change the attach parameters. Each move costs effort in months.
- **Budgets:** gem5 runs (this is M), effort months, area.
- **Hard constraints:** the wall lies past the longest session, and the design fits the area and effort caps.
- **Stopping rule:** stop within x% of the Amdahl bound, since no design can beat it.
- **Logging:** every move records the reading that justified it. The log is the co-design walk figure.

### 3.9 Verification

- gem5 runs one layer or one decode step per candidate, with each technology's timing as accelerator parameters, and scales up. One run takes 1–3 hours.
- UPMEM checks the model (E1). The existing LLaMA2 study gave 2.42× predicted against 2.46× measured. Add mini-coder-4b GEMV shapes if time allows.

## 4. Showing the search is not brute force

- **Pruning:** report the fraction of the space discarded with zero gem5 runs, because the wall falls before the session ends or the Amdahl bound sits below the GPU sheet.
- **Sample efficiency:** plot the best session time found against gem5 runs for a brute-force sweep (on a subspace), random search, Bayesian optimization, the agent with gem5 only, the agent with Domeline's time only, and the full harness (time plus readings).
- **Certificate:** the final design's distance to the Amdahl bound.
- **Scale:** about 7¹³ mappings before sizing; show convergence where a sweep is infeasible.
- **Generalization:** add a second workload or a new technology and show it converges in a few steps.

## 5. Choosing the agentic workload

Criteria:

1. It runs offline and reproducibly, with no network calls in tools.
2. A large share of its time can be offloaded. Tool time sets the Amdahl roof.
3. Its context grows over the session.
4. It uses an open model and a public benchmark.

| Candidate | For | Against |
|---|---|---|
| Coding agent: mini-SWE-agent + mini-coder-4b on SWE-bench Verified / Terminal-Bench 2.0 | Linear context, bash only, traces already planned | Builds and tests are CPU-heavy and cap the speedup |
| Retrieval / deep-research agent over a fixed local corpus (e.g. BrowseComp-Plus) | The tool is similarity search, which maps to CAM or crossbar, so the roof rises | Needs a local retriever setup |

Pick from data. Measure the tool share on about 5 sessions of each first. The workload with the higher Amdahl ceiling has more room for PIM to win.

## 6. Open decisions

- **Open:** what one invocation is, a decode step or a turn. Proposed: a decode step, summed per turn.
- **Open:** where the coprocessor attaches (on-die by the LLC, on-package, CXL). This sets `t0`, `α`, and link bandwidth.
- **Open:** where effort months come from, and the sensitivity plot that goes with them.
- **Open:** whether gem5 runs one layer or one decode step per candidate, and how it scales up.
- **Open:** the second workload.

## 7. Build order

1. **Integrator v0:** the model spec, the decode op graph(L), the tech library, the mapping, transfer, launch, writes, and capacity. Test it against `soc_inference_estimator.py` on BitNet under that script's assumptions.
2. **Trace schema and session replay.** Synthetic context paths until real traces arrive.
3. **Domeline readings:** the binding term, the headroom, the Amdahl bound, the GPU sheet.
4. **Agent loop:** budgets, moves, the decision log. Then the baselines: random search, Bayesian optimization, a sweep on a subspace.
5. **gem5 hook:** a design is turned into accelerator parameters and run for one layer.
