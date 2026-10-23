"""Stronger-GPU sanity bar: the headline design against a 4090 and an H100, scaled from the 3080.

No measurement: each session's measured 3080 time is split into decode, prefill
and tools; decode is scaled by the memory-bandwidth ratio (batch-1 decode is
bandwidth-bound), prefill by the dense fp16 tensor-throughput ratio, and tools
are left unchanged. This is generous to the bigger GPUs: it assumes llama.cpp
reaches the same fraction of peak on them, and that no session hits a VRAM
wall (24 GB / 80 GB hold every session's KV). Spec numbers are vendor peaks.

  python experiments/gpu_scale.py
Output: results/search/gpu_scale.json
"""

import json

from common import ROOT, gpu, library, model, sessions, split

from harness.evaluate import Evaluator
from harness.space import Space

# GB/s, dense fp16 tensor TFLOPS (fp32 accumulate). VERIFY against vendor sheets.
GPUS = {
    "RTX 3080": (760.0, 59.5),
    "RTX 4090": (1008.0, 165.2),
    "H100 SXM": (3350.0, 989.0),
}


def main():
    spec, lib, g = model(), library(), gpu()
    train, test = split(sessions())
    extra = sessions(which="extra")
    space = Space(lib, spec, ref_ctx=max(s.max_ctx for s in train + test), host="cpu")
    point = json.loads((ROOT / "results/search/main.json").read_text())["runs"][0]["best"]["point"]
    bw0, fl0 = GPUS["RTX 3080"]
    out = {"point": point, "gpus": {k: {"bw_gb_s": v[0], "fp16_tflops": v[1]} for k, v in GPUS.items()}, "sets": {}}
    for name, ss in (("search", train), ("heldout", test), ("late", extra)):
        e = Evaluator(space, ss, g)(point)
        parts = [g.session_time(s) for s in ss]
        dec, pre, tools = (sum(p[k] for p in parts) for k in ("decode", "prefill", "tools"))
        row = {"design_s": e.time_s, "decode_s": dec, "prefill_s": pre, "tools_s": tools,
               "amdahl_vs_3080": (dec + pre + tools) / tools, "speedup": {}}
        for gname, (bw, fl) in GPUS.items():
            t = dec * bw0 / bw + pre * fl0 / fl + tools
            row["speedup"][gname] = t / e.time_s
        out["sets"][name] = row
        print(f"[{name}] {len(ss)} sessions: decode {dec:.0f}s prefill {pre:.0f}s tools {tools:.0f}s; "
              + ", ".join(f"vs {k} {v:.2f}x" for k, v in row["speedup"].items()))
    dst = ROOT / "results/search/gpu_scale.json"
    dst.write_text(json.dumps(out, indent=1))
    print(dst)


if __name__ == "__main__":
    main()
