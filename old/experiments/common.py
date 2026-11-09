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


KV8_DIR = ROOT / "results/traces/swebench_kv8"
# The frozen workload every search result was produced on (89 sessions). Sessions
# collected later (the second held-out set) are loaded with which="extra".
WORKLOAD = ROOT / "traces/workload_main.tsv"


def sessions(trace_dir=TRACE_DIR, ok_only=True, which="main"):
    """Real sessions with turns, podman exec overhead removed from tool times.

    A session that hit the 3080's fp16 KV wall is taken from its int8-KV re-run
    (results/traces/swebench_kv8) when one exists, so it is complete; its GPU
    times are then those of the configuration the GPU had to use.
    which: "main" (the frozen workload), "extra" (collected after it) or "all".
    """
    frozen = set(WORKLOAD.read_text().split()[1:])
    out = []
    for d in sorted(Path(trace_dir).iterdir()):
        if not d.is_dir():
            continue
        if which != "all" and (d.name in frozen) != (which == "main"):
            continue
        if (KV8_DIR / d.name / "result.json").exists():
            d = KV8_DIR / d.name
        res, tr = d / "result.json", d / "trace.jsonl"
        if not (res.exists() and tr.exists()):
            continue
        r = json.loads(res.read_text())
        if ok_only and r["exit_status"] in ("LimitsExceeded",) and r.get("n_turns", 0) == 0:
            continue
        s = load_trace(tr, name=d.name, exec_overhead_s=r.get("exec_overhead_s", 0.0))
        if s.turns:
            s.meta = {"repo": r["repo"], "exit": r["exit_status"],
                      "gpu_kv": "q8_0" if d.parent == KV8_DIR else "f16"}
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
