# Domeline theory: D, headroom, and the special cases

Oct 29, 2026. Plan items 1, 4 and 5. Every statement here is checked by a test:
`tests/test_readings.py` (D, headroom) and `tests/test_special_cases.py` (VIPER, Roofline).

## Symbols

| Symbol | Meaning | In code |
|---|---|---|
| d | a design: mapping π (op → tech or host), units per tech, link, weight bits w_b, KV bits k_b | `Design` |
| c | context length in tokens | `ctx` |
| G(n, c) | op graph of one forward step, n new tokens at context c | `ops.forward_ops` |
| π(o) | the tech that runs op o | `Design.tech_for` |
| M_o, N_o | output and input dims of a static MVM o | `Op.shape` |
| ℓ | layers that keep a KV cache; h_kv KV heads; d_h head dim | `ModelSpec` |
| R, K, b | rows, columns and bits per cell of one crossbar array | tech params |
| a(M, N, bits) | crossbar arrays to hold an M×N matrix: ⌈N/R⌉ · ⌈M·bits/(K·b)⌉ | `xbar_arrays_for` |
| A = R·K·b/8 | bytes one crossbar array stores | `_array_bytes` |
| u_t | units (arrays) of tech t in the design | `Design.units` |
| M_t | local memory of a digital or DRAM-PIM tech t (bytes) | `local_mem_bytes` |
| T_k | term k of the step time: host, compute per tech, transfer, launch, write | `StepResult.terms` |

## 1. D, the data held in PIM

**Definition.** D(d, c) is the number of bytes resident on the coprocessor at context c:

    D(d, c) = Σ_t D_t(d, c)
    D_t(d, c) = W_t + H_t + KV_t(c)

for every coprocessor tech t, with

- W_t = Σ over static MVMs o with π(o) = t of M_o · N_o · w_b / 8 (the layer weights), counted when t is a
  crossbar, or a digital/DRAM-PIM tech that keeps weights locally (`weights_local`, default true);
- H_t = vocab · hidden · w_b / 8 if the LM head is mapped to t (same rule; it is a static MVM), else 0;
- KV_t(c) = 2 · ℓ · h_kv · d_h · c · k_b / 8 if attention (q·Kᵀ and p·V) is mapped to t, else 0. K and V
  contribute half each; on a crossbar K is stored with d_h on the rows, V with c on the rows.

Not counted: activations (a few KB per layer, they stream), the embedding table, the KV cache or
weights of ops left on the host, and anything over the link. Bytes over the link stay a cost term
(transfer), not an axis.

**A session is a path along D.** Only KV_t grows with c, linearly: dD/dc = 2 · ℓ · h_kv · d_h · k_b / 8
when attention is offloaded, 0 otherwise. For mini-coder-4b (ℓ = 36, h_kv = 8, d_h = 128) that is
147,456 B per token at fp16 KV. A design with the weight MVMs on crossbars holds D = 3.63 GB at every
context; with the LM head and attention on crossbars too, D = 4.63 GB at 4K, 8.85 GB at 32K and
13.69 GB at 64K tokens.

**Occupied storage and the wall in D.** Crossbars hold whole arrays, so tech t occupies

    O_t(d, c) = A · Σ_o a_o        (crossbar: arrays each stored tensor needs, padding included)
    O_t(d, c) = D_t(d, c)          (digital, DRAM-PIM: bytes)

and provides

    C_t(d) = A · u_t               (crossbar)
    C_t(d) = M_t (· u_t if the memory scales with units)    (digital, DRAM-PIM)

The design is past its wall at c exactly when O_t(d, c) > C_t(d) for some t; this is the same test as
`Integrator.capacity` (array counts there, bytes here), checked over random designs in
`test_D_matches_capacity`. The scalar wall is D_wall(d) = Σ_t C_t(d), the bytes the design can hold.
With one tech holding data (every design in the 17-knob space except attention on the vector unit,
which puts the KV on the VFU and the weights on crossbars) the wall in D is exact up to padding; with
several, the per-tech test is the one that decides, and the scalar is their sum.

The wall context c_wall(d) is the smallest c with O_t(d, c) > C_t(d) for some t (`Eval.wall_ctx`,
found by bisection between grid points of the session replay).

**Code.** `Integrator.data_in_pim(design, ctx)` returns a `Residency`: `tensors[tech][tensor]` (logical
bytes: `w.<kind>`, `lm_head`, `kv.K`, `kv.V`), `occupied`, `capacity`, and `D`, `D_wall`, `headroom()`,
`binding_headroom()`, `over`. Both it and `capacity` read one itemization (`Integrator._resident`), so
they cannot drift apart.

## 2. Headroom

Two readings, both on `Eval` (`harness/evaluate.py`), both over the whole workload.

**Term headroom.** Let T_(1) ≥ T_(2) be the two largest nonzero terms of the workload's LLM-side
breakdown (each term summed over every step of every session). Then

    h_term = T_(1) / T_(2)

is the factor by which the binding term must fall before the second one binds. `Eval.binding_term`
and `Eval.second_term` name them; `Eval.term_headroom` = h_term (inf with fewer than two nonzero terms).
`evaluate.binding_and_next(terms)` computes it.

**Capacity headroom.** At the workload's longest context c_max = max over sessions of the session's
largest context,

    h_cap = min over t of ( C_t(d) − O_t(d, c_max) )      (bytes; < 0 means past the wall)

`Eval.capacity_headroom_bytes` = h_cap and `Eval.capacity_headroom_tech` names the tech that sets it.
h_cap < 0 exactly when `Eval.wall_ctx` is not None (checked in `test_eval_readings`). `Eval.D_bytes`,
`Eval.D_wall_bytes` and `Eval.D_by_tensor` give D, D_wall and the per-tensor breakdown at c_max.

The new readings are reported only: no searcher reads them yet. The walk, hill climbing and random
search give bit-identical logs before and after the change on the real traces (walk with and without
an area cap, hill climbing, random search; 1,200 capacity calls).

## 3. Special cases

The integrator's step time (main.tex Eq. step) is

    t(c, d) = Σ_{o ∈ G} t_o^{π(o)}(c)
            + Σ_{o ∈ X(d)} ( t_0 + α · N_{π(o)} · [π(o) is host-scattered] + B_o / BW )
            + B_noc / BW_noc + n_↑ · t_launch + n_↓ · t_sync + t_rp(c, d)

with X(d) the ops whose inputs were produced on the other side of the link, B_o the bytes they pull
across it, n_↑ and n_↓ the changes of side (host → coprocessor, and back), and t_rp the reprogramming
of weights past the wall. Ops in one group on crossbars (q/k/v, gate/up) add their maximum, not their
sum.

### 3.1 One offloaded kernel → VIPER

VIPER (main.tex Eq. viper): T = t_host + m · (t_in + t_prog + t_comp + t_out), m = ⌈D/C⌉.

Take one step (no session growth), and map one op o* (one kernel) to tech t; everything else stays on
the host. Then X(d) has two members, o* (its input comes from the host) and the first host op that
reads its output; n_↑ = n_↓ = 1; B_noc = 0. Below the wall (D ≤ C, m = 1) the step is

    t = Σ_{o ≠ o*} t_o^host                                   = t_host
      + t_launch + t_0 + α · N_t + B_in / BW                  = t_in
      + t_{o*}^t                                              = t_comp
      + t_sync + t_0 + B_out / BW                             = t_out

with t_prog = 0: the weights are programmed once before the session and stay resident. This is
VIPER's form with m = 1, exactly (`test_viper_crossbar_below_wall`, relative error < 1e-12). The launch
terms are not extra: t_launch belongs to t_in and t_sync to t_out.

Where it does not reduce exactly, and what the paper may claim:

1. **Past the wall (m > 1).** The integrator does not multiply by m. It times one pass and adds the
   reprogramming of the overflow once per step:
   t = t_host + t_in + t_comp + t_out + t_rp, t_rp = ⌈(a_need − u_t) / P_w⌉ · R · t_w
   (P_w arrays written in parallel, t_w per row; `test_viper_past_wall_is_additive`). If each of
   VIPER's m passes reprograms the u_t arrays, this is below VIPER, since a_need − u_t ≤ (m − 1) · u_t
   and t_in, t_comp, t_out are paid once instead of m times: the integrator is optimistic past the wall.
   For digital techs, a KV that outgrows local memory is read at min(BW_local, BW_link) instead; weights
   that outgrow a digital tech's memory cost no time at all. None of this reaches a reported result,
   because a design past its wall is infeasible and is never chosen. But main.tex Definition 1 writes
   ⌈D/C(h)⌉ · Σ_k T_k, which is not what the integrator computes. **Claim instead:** "Below the wall the
   step time reduces to VIPER's closed form exactly; past it, \sys{} marks the design infeasible (and
   times it optimistically, reprogramming only the overflow)." Or keep ⌈D/C⌉ in Definition 1 as the
   schematic of the wall in Fig. 2 and say so.
2. **α only on the scatter.** α · N is charged on transfers into a host-scattered tech (UPMEM DPUs), not
   on the gather back (`test_viper_alpha_only_on_scatter`). VIPER's T_transfer = t_0 + α N applies to
   both directions. Harmless for the ReRAM designs (α = 0), but the UPMEM reproduction relies on how
   t_out was fitted; say "α N on the host scatter".
3. **An op per layer is ℓ kernels.** Offloading o_proj offloads it in every layer, so the step is
   t_host + ℓ · (t_in + t_comp + t_out): VIPER's form once per invocation (`test_viper_per_layer_kernel`).
4. **Groups.** Offloading q, k and v together is one kernel whose t_comp is the maximum of the three
   (separate arrays run in parallel) and whose input crosses once.
5. **Prefill (n > 1) on crossbars** uses ISAAC's layer pipeline: an op costs (1 + (n − 1)/ℓ) passes, not
   n. The reduction is for decode (n = 1), where this does not apply.

### 3.2 Below the wall, no host, no link, no writes → per-op Roofline

Roofline for op o on a device with peak P and bandwidth BW: t_o = max(F_o / P, B_o / BW), F_o the
op's operations and B_o its bytes.

Map every op to one roofline-kind tech (`digital`, `dram_pim`, or a `cpu` without a measured
profile). Then X(d) is empty (no transfers), B_noc = 0, no weights are reprogrammed, and every term is
a roofline term, so

    t(c, d) = Σ_{o ∈ G(n, c)} max(F_o / P, B_o / BW) + n_↑ · t_launch + Σ_o t_fixed

with B_o the integrator's byte count for the op class: weights M·N·w_b/8 for a static MVM, the KV read
c · h_kv · d_h · k_b / 8 for attention, the KV written for an append, reads plus writes for vector and
gather ops. This is the E1 baseline (plan item 3: per op, `tech.roofline_time`, summed).

It is exact (`test_roofline_one_device`, decode at 1, 4K and 40K, and a 256-token prefill), with two
leftovers that vanish only when their parameters are zero:

- n_↑ = 1: a step that runs wholly on the coprocessor still pays one launch (and no sync, since it never
  returns). Exact Roofline needs t_launch = 0, or add the one launch.
- t_fixed = `op_overhead_s` per op (5 ns on the VFU, 1 µs on the host roofline). On the host's roofline
  model the step is Σ_o (max(F_o/P, B_o/BW) + 1 µs) (`test_roofline_on_host_roof`).

KV appends on a digital tech are booked under the write term, but their time is the roofline time of
the bytes they write (0 operations), so "no writes" holds for every roofline-kind tech: write cost that
is not a roofline appears only on crossbars.

Where it does not reduce, and what the paper may claim:

1. **Crossbars are not roofline devices.** Their op time is a latency model (⌈input bits / DAC bits⌉
   read cycles plus an adder tree), independent of F_o and B_o; K/V appends cost row writes. So the
   Domeline reduces to Roofline only when no op is on a crossbar.
2. **Several techs on the coprocessor** add B_noc / BW_noc between them; exact only with one tech or
   BW_noc = ∞.
3. **The measured host profile** is a lookup per op kind, not a roofline; host ops are a roofline only
   on the `cpu_roofline` model.
4. **Per-op, not per-kernel.** The reduction gives Σ_o max(F_o/P, B_o/BW), which is ≥ the whole-step
   Roofline max(Σ F_o / P, Σ B_o / BW). The paper should say "per-op Roofline".

**Claim:** "On one roofline-kind device, below the wall and with no host work or link costs, the
Domeline's step time is the per-op Roofline Σ_o max(F_o/P, B_o/BW) (plus one launch and the per-op
overhead when those are nonzero)."
