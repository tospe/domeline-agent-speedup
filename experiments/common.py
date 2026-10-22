"""Shared setup for experiments: model, library, real sessions, GPU baseline."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.model import ModelSpec  # noqa: E402
from harness.session import GpuBaseline, load_trace  # noqa: E402
from harness.tech import load_library  # noqa: E402

MODEL_DIR = "/localhome/tsousape/models/mini-coder-4b"
TRACE_DIR = ROOT / "results/traces/swebench"


def model():
    return ModelSpec.from_hf_config(MODEL_DIR)


def library():
    return load_library(ROOT / "lib/tech")


def sessions(trace_dir=TRACE_DIR, ok_only=True):
    """Real sessions with turns, podman exec overhead removed from tool times."""
    out = []
    for d in sorted(Path(trace_dir).iterdir()):
        res, tr = d / "result.json", d / "trace.jsonl"
        if not (res.exists() and tr.exists()):
            continue
        r = json.loads(res.read_text())
        if ok_only and r["exit_status"] in ("LimitsExceeded",) and r.get("n_turns", 0) == 0:
            continue
        s = load_trace(tr, name=d.name, exec_overhead_s=r.get("exec_overhead_s", 0.0))
        if s.turns:
            s.meta = {"repo": r["repo"], "exit": r["exit_status"]}
            out.append(s)
    return out


def split(sess, seed=0):
    """Search set / held-out set: alternate sessions after a seeded shuffle."""
    import random
    s = sorted(sess, key=lambda x: x.name)
    random.Random(seed).shuffle(s)
    return s[0::2], s[1::2]


def gpu():
    return GpuBaseline(ROOT / "results/gpu/g1.jsonl", weights="Q8_0", kv="f16")
