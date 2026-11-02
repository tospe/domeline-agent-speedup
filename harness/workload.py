"""The workload input as a file: op graph (per-layer templates) and trace (states + tau).

`load_workload(path)` reads lib/workload/<name>.yaml (format in its header). The result
is a `Workload`: it answers every ModelSpec attribute (it wraps one) and generates the op
list of a step from the YAML templates with `forward_ops(n_new, ctx, logits_tokens)`.
Integrator and Space take it in place of a ModelSpec (ops.ops_for picks the generator);
for the shipped file the ops equal ops.forward_ops exactly (tests/test_inputs_m1.py).

The trace is one TSV row per turn: the state the turn starts in (context, prefill and
generated tokens), the host tool time tau after it, and the GPU's measured times.
"""

import ast
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from . import ops as O
from .model import ModelSpec
from .session import Session, Turn

ROOT = Path(__file__).resolve().parent.parent

TRACE_COLS = ("session", "repo", "exit", "gpu_kv", "ctx_start", "prefill_tokens", "gen_tokens",
              "t_tool_s", "t_prefill_gpu_s", "t_decode_gpu_s")
_OP_FIELDS = {"name", "kind", "cls", "subcls", "when", "flops", "reads", "writes", "shape", "tokens",
              "kv_read_elems", "kv_write_elems", "group", "operands", "precisions"}
_NAMES = {"S", "L", "T", "D", "F", "H", "Hkv", "dh", "qd", "kvd", "V", "a", "l",
          "qk_norm", "sub_norms", "relu2", "kv_layer"}


def _expr(src):
    """Compile an arithmetic expression over the workload symbols (see the YAML header)."""
    src = str(src)
    tree = ast.parse(src, mode="eval")
    bad = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - _NAMES
    if bad:
        raise ValueError(f"unknown symbols {sorted(bad)} in {src!r}")
    return compile(tree, src, "eval")


def _ev(code, env):
    return eval(code, {"__builtins__": {}}, env)


@dataclass
class OpTemplate:
    """One op of the graph; `layer` templates are instantiated once per layer."""
    name: str
    kind: str
    cls: str
    flops: object
    subcls: Optional[str] = None
    when: object = None
    reads: List[Tuple[str, object]] = field(default_factory=list)
    writes: List[Tuple[str, object]] = field(default_factory=list)
    shape: Optional[Tuple[object, object]] = None
    tokens: object = None                      # default S
    kv_read_elems: object = None
    kv_write_elems: object = None
    group: Optional[str] = None
    operands: Optional[List[Tuple[str, str]]] = None
    precisions: Tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, d, class_precisions):
        extra = set(d) - _OP_FIELDS
        if extra:
            raise ValueError(f"unknown fields for op {d.get('name')!r}: {sorted(extra)}")
        c = lambda k: _expr(d[k]) if k in d else None   # noqa: E731
        operands = [(t, r) for t, r in d["operands"]] if "operands" in d else None
        for _, role in operands or ():
            if role not in (O.STATIC, O.DYNAMIC):
                raise ValueError(f"op {d['name']!r}: operand role {role!r}")
        return cls(name=d["name"], kind=d["kind"], cls=d["cls"], flops=_expr(d["flops"]),
                   subcls=d.get("subcls"), when=c("when"),
                   reads=[(t, _expr(b)) for t, b in d.get("reads", ())],
                   writes=[(t, _expr(b)) for t, b in d.get("writes", ())],
                   shape=tuple(_expr(x) for x in d["shape"]) if "shape" in d else None,
                   tokens=c("tokens"), kv_read_elems=c("kv_read_elems"),
                   kv_write_elems=c("kv_write_elems"), group=d.get("group"), operands=operands,
                   precisions=tuple(d.get("precisions", class_precisions.get(d["cls"], ()))))

    def instantiate(self, env, p, layer):
        if self.when is not None and not _ev(self.when, env):
            return None
        name = lambda t: t.format(p=p)   # noqa: E731
        return O.Op(
            name=p + self.name, kind=self.kind, cls=self.cls, layer=layer,
            flops=_ev(self.flops, env),
            reads=[(name(t), _ev(b, env)) for t, b in self.reads],
            writes=[(name(t), _ev(b, env)) for t, b in self.writes],
            shape=tuple(_ev(x, env) for x in self.shape) if self.shape else None,
            tokens=_ev(self.tokens, env) if self.tokens is not None else env["S"],
            kv_read_elems=_ev(self.kv_read_elems, env) if self.kv_read_elems is not None else 0.0,
            kv_write_elems=_ev(self.kv_write_elems, env) if self.kv_write_elems is not None else 0.0,
            group=name(self.group) if self.group else None, subcls=self.subcls,
            precisions=self.precisions,
            operands=[(name(t), r) for t, r in self.operands] if self.operands is not None else None)


@dataclass
class TraceRef:
    file: Path                       # per-turn state rows (TRACE_COLS)
    split_seed: int = 0
    raw: Optional[str] = None        # where the rows were exported from (provenance)
    frozen: Optional[str] = None

    def sessions(self) -> List[Session]:
        return load_trace_tsv(self.file)

    def split(self, sessions=None):
        """(search set, held-out set), the same rule as experiments/common.split."""
        s = sorted(sessions if sessions is not None else self.sessions(), key=lambda x: x.name)
        random.Random(self.split_seed).shuffle(s)
        return s[0::2], s[1::2]


@dataclass
class Workload:
    """A ModelSpec plus the op graph that replaces ops.forward_ops for it."""
    name: str
    spec: ModelSpec
    act_bytes: int
    pre: List[OpTemplate]
    layer: List[OpTemplate]
    post: List[OpTemplate]
    trace: Optional[TraceRef] = None
    source: str = ""

    def __getattr__(self, k):
        # Only called for names not on the Workload: answer as the ModelSpec.
        if k == "spec" or k.startswith("__"):
            raise AttributeError(k)
        return getattr(self.spec, k)

    def env(self, n_new, ctx, logits_tokens):
        m = self.spec
        return {"S": n_new, "L": ctx, "T": n_new if logits_tokens is None else logits_tokens,
                "D": m.hidden, "F": m.intermediate, "H": m.heads, "Hkv": m.kv_heads,
                "dh": m.head_dim, "qd": m.q_dim, "kvd": m.kv_dim, "V": m.vocab,
                "a": self.act_bytes, "qk_norm": m.qk_norm, "sub_norms": m.sub_norms,
                "relu2": m.activation == "relu2", "kv_layer": False, "l": -1}

    def forward_ops(self, n_new=1, ctx=1, logits_tokens=None) -> List[O.Op]:
        env = self.env(n_new, ctx, logits_tokens)
        out = [op for t in self.pre if (op := t.instantiate(env, "", None)) is not None]
        kv = O.kv_layer_ids(self.spec)
        for l in range(self.spec.layers):
            env["l"], env["kv_layer"] = l, l in kv
            out += [op for t in self.layer if (op := t.instantiate(env, f"L{l}.", l)) is not None]
        env["l"], env["kv_layer"] = -1, False
        out += [op for t in self.post if (op := t.instantiate(env, "", None)) is not None]
        return out


def _path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_workload(path) -> Workload:
    path = _path(path)
    d = yaml.safe_load(path.read_text())
    spec = ModelSpec(**d["model"])
    prec = {k: tuple(v) for k, v in (d.get("precisions") or {}).items()}
    g = d["graph"]
    tpl = {k: [OpTemplate.from_dict(o, prec) for o in g.get(k, ())] for k in ("pre", "layer", "post")}
    tr = d.get("trace")
    trace = TraceRef(file=_path(tr["file"]), split_seed=int(tr.get("split_seed", 0)),
                     raw=tr.get("raw"), frozen=tr.get("frozen")) if tr else None
    return Workload(name=d["name"], spec=spec, act_bytes=int(d.get("act_bytes", O.ACT_BYTES)),
                    trace=trace, source=str(path), **tpl)


# ---------------------------------------------------------------------------
# Trace file
# ---------------------------------------------------------------------------

def _num(x, typ):
    return None if x == "" else typ(x)


def write_trace_tsv(sessions: List[Session], path):
    """One row per turn; floats written with repr so they load back bit for bit."""
    rows = ["\t".join(TRACE_COLS)]
    for s in sessions:
        m = s.meta or {}
        for t in s.turns:
            vals = [s.name, m.get("repo", ""), m.get("exit", ""), m.get("gpu_kv", ""),
                    t.ctx_start, t.prefill_tokens, t.gen_tokens, t.t_tool_s,
                    t.t_prefill_gpu_s, t.t_decode_gpu_s]
            rows.append("\t".join("" if v is None else repr(v) if isinstance(v, float) else str(v)
                                  for v in vals))
    Path(path).write_text("\n".join(rows) + "\n")


def load_trace_tsv(path) -> List[Session]:
    lines = _path(path).read_text().splitlines()
    if tuple(lines[0].split("\t")) != TRACE_COLS:
        raise ValueError(f"{path}: header is not {TRACE_COLS}")
    out: Dict[str, Session] = {}
    for line in lines[1:]:
        r = dict(zip(TRACE_COLS, line.split("\t")))
        s = out.get(r["session"])
        if s is None:
            meta = {k: r[k] for k in ("repo", "exit", "gpu_kv") if r[k]}
            s = out[r["session"]] = Session(r["session"], [], meta)
        s.turns.append(Turn(ctx_start=int(r["ctx_start"]), prefill_tokens=int(r["prefill_tokens"]),
                            gen_tokens=int(r["gen_tokens"]), t_tool_s=float(r["t_tool_s"]),
                            t_prefill_gpu_s=_num(r["t_prefill_gpu_s"], float),
                            t_decode_gpu_s=_num(r["t_decode_gpu_s"], float)))
    return list(out.values())
