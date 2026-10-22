"""Op graph for one forward step of a decoder-only transformer.

`forward_ops(spec, n_new, ctx)` lists every op, in dependency order, for
processing `n_new` new tokens against a context of `ctx` tokens (the new
tokens included). Decode is n_new=1; a prefill of S tokens is n_new=ctx=S.

Each op names the activation tensors it reads and writes, so the integrator
can tell when data must cross between the CPU and the coprocessor. Weights
and the KV cache are not activations: they live with whatever unit runs the
op that uses them.

FLOP counts follow soc_inference_estimator.py (TERINT-GEMV) where it defines
them, so the two can be compared.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Op classes: what kind of hardware can run an op.
MVM_STATIC = "mvm_static"      # weight matrix fixed after programming
MVM_DYNAMIC = "mvm_dynamic"    # matrix is the KV cache, grows every token
KV_APPEND = "kv_append"        # write new K and V entries
VECTOR = "vector"              # elementwise / reduction (norms, softmax, RoPE, ...)
GATHER = "gather"              # embedding lookup

ACT_BYTES = 2  # activations in fp16 between ops


@dataclass
class Op:
    name: str                  # unique within a step, e.g. "L3.q_proj"
    kind: str                  # q_proj, softmax, rmsnorm, ... (what a mapping names)
    cls: str                   # one of the op classes above
    layer: Optional[int]
    flops: float
    reads: List[Tuple[str, float]] = field(default_factory=list)   # (tensor, bytes)
    writes: List[Tuple[str, float]] = field(default_factory=list)
    shape: Optional[Tuple[int, int]] = None   # (M out, N in) for static MVMs
    tokens: int = 1                           # rows processed (n_new, or 1 for lm_head)
    kv_read_elems: float = 0.0                # KV elements read (scaled by KV precision)
    kv_write_elems: float = 0.0               # KV elements appended
    group: Optional[str] = None               # ops in one group are independent of each other


def _rmsnorm_flops(S, dim):
    return 3 * S * dim


def forward_ops(spec, n_new=1, ctx=1, logits_tokens=None):
    """Ops for one step. logits_tokens: tokens that need logits (default n_new)."""
    S, L = n_new, ctx
    D, F = spec.hidden, spec.intermediate
    H, Hkv, dh = spec.heads, spec.kv_heads, spec.head_dim
    qd, kvd = spec.q_dim, spec.kv_dim
    a = ACT_BYTES
    ops: List[Op] = []

    def add(**kw):
        kw.setdefault("tokens", S)
        ops.append(Op(**kw))

    add(name="embed", kind="embed", cls=GATHER, layer=None, flops=0,
        writes=[("x", S * D * a)])

    kv_layer_ids = set(range(spec.layers))
    if spec.kv_layers is not None and spec.kv_layers < spec.layers:
        # Hybrid models: spread the cached layers evenly (e.g. 1 in 4).
        step = spec.layers / spec.kv_layers
        kv_layer_ids = {int(i * step + step - 1) for i in range(spec.kv_layers)}

    for l in range(spec.layers):
        p = f"L{l}."
        add(name=p + "attn_norm", kind="rmsnorm", cls=VECTOR, layer=l,
            flops=_rmsnorm_flops(S, D), reads=[("x", S * D * a)], writes=[(p + "h", S * D * a)])
        for nm, m, n in (("q_proj", qd, D), ("k_proj", kvd, D), ("v_proj", kvd, D)):
            add(name=p + nm, kind=nm, cls=MVM_STATIC, layer=l, flops=2 * S * m * n,
                reads=[(p + "h", S * n * a)], writes=[(p + nm[0], S * m * a)],
                shape=(m, n), tokens=S, group=p + "qkv")
        if spec.qk_norm:
            add(name=p + "qk_norm", kind="qk_norm", cls=VECTOR, layer=l,
                flops=_rmsnorm_flops(S, qd + kvd),
                reads=[(p + "q", S * qd * a), (p + "k", S * kvd * a)],
                writes=[(p + "q", S * qd * a), (p + "k", S * kvd * a)])
        add(name=p + "rope", kind="rope", cls=VECTOR, layer=l, flops=6 * S * dh * (H + Hkv),
            reads=[(p + "q", S * qd * a), (p + "k", S * kvd * a)],
            writes=[(p + "q", S * qd * a), (p + "k", S * kvd * a)])
        if l in kv_layer_ids:
            add(name=p + "kv_append", kind="kv_append", cls=KV_APPEND, layer=l, flops=0,
                reads=[(p + "k", S * kvd * a), (p + "v", S * kvd * a)],
                kv_write_elems=2 * S * kvd)
            add(name=p + "attn_score", kind="attn_score", cls=MVM_DYNAMIC, layer=l,
                flops=2 * S * L * dh * H, reads=[(p + "q", S * qd * a)],
                writes=[(p + "s", S * L * H * a)], kv_read_elems=L * kvd, tokens=S)
            add(name=p + "softmax", kind="softmax", cls=VECTOR, layer=l, flops=5 * S * L * H,
                reads=[(p + "s", S * L * H * a)], writes=[(p + "s", S * L * H * a)])
            add(name=p + "attn_value", kind="attn_value", cls=MVM_DYNAMIC, layer=l,
                flops=2 * S * L * dh * H, reads=[(p + "s", S * L * H * a)],
                writes=[(p + "o", S * qd * a)], kv_read_elems=L * kvd, tokens=S)
        else:
            # Non-attention mixer (e.g. linear attention): fixed-size state, no KV growth.
            add(name=p + "mixer", kind="mixer", cls=VECTOR, layer=l, flops=4 * S * qd * dh,
                reads=[(p + "q", S * qd * a), (p + "k", S * kvd * a), (p + "v", S * kvd * a)],
                writes=[(p + "o", S * qd * a)])
        if spec.sub_norms:
            add(name=p + "attn_sub_norm", kind="rmsnorm", cls=VECTOR, layer=l,
                flops=_rmsnorm_flops(S, D), reads=[(p + "o", S * qd * a)],
                writes=[(p + "o", S * qd * a)])
        add(name=p + "o_proj", kind="o_proj", cls=MVM_STATIC, layer=l, flops=2 * S * D * qd,
            reads=[(p + "o", S * qd * a)], writes=[(p + "ao", S * D * a)], shape=(D, qd), tokens=S)
        add(name=p + "attn_residual", kind="residual", cls=VECTOR, layer=l, flops=S * D,
            reads=[("x", S * D * a), (p + "ao", S * D * a)], writes=[("x", S * D * a)])
        add(name=p + "mlp_norm", kind="rmsnorm", cls=VECTOR, layer=l,
            flops=_rmsnorm_flops(S, D), reads=[("x", S * D * a)], writes=[(p + "h2", S * D * a)])
        for nm in ("gate_proj", "up_proj"):
            add(name=p + nm, kind=nm, cls=MVM_STATIC, layer=l, flops=2 * S * F * D,
                reads=[(p + "h2", S * D * a)], writes=[(p + nm[:2], S * F * a)],
                shape=(F, D), tokens=S, group=p + "gate_up")
        act_flops = (2 if spec.activation == "relu2" else 4) * S * F + S * F
        add(name=p + "act", kind="act", cls=VECTOR, layer=l, flops=act_flops,
            reads=[(p + "ga", S * F * a), (p + "up", S * F * a)], writes=[(p + "m", S * F * a)])
        if spec.sub_norms:
            add(name=p + "ffn_sub_norm", kind="rmsnorm", cls=VECTOR, layer=l,
                flops=_rmsnorm_flops(S, F), reads=[(p + "m", S * F * a)],
                writes=[(p + "m", S * F * a)])
        add(name=p + "down_proj", kind="down_proj", cls=MVM_STATIC, layer=l, flops=2 * S * D * F,
            reads=[(p + "m", S * F * a)], writes=[(p + "d", S * D * a)], shape=(D, F), tokens=S)
        add(name=p + "mlp_residual", kind="residual", cls=VECTOR, layer=l, flops=S * D,
            reads=[("x", S * D * a), (p + "d", S * D * a)], writes=[("x", S * D * a)])

    T = S if logits_tokens is None else logits_tokens
    add(name="final_norm", kind="rmsnorm", cls=VECTOR, layer=None,
        flops=_rmsnorm_flops(S, D), reads=[("x", S * D * a)], writes=[("xf", S * D * a)])
    add(name="lm_head", kind="lm_head", cls=MVM_STATIC, layer=None,
        flops=2 * T * D * spec.vocab, reads=[("xf", T * D * a)],
        writes=[("logits", T * spec.vocab * a)], shape=(spec.vocab, D), tokens=T)
    add(name="sample", kind="sample", cls=VECTOR, layer=None, flops=2 * T * spec.vocab, tokens=T,
        reads=[("logits", T * spec.vocab * a)], writes=[("token", T * 4)])
    return ops
