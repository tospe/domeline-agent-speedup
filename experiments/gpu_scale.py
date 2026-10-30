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

# GB/s, dense (no sparsity) fp16 tensor TFLOPS with fp32 accumulate. Checked Oct 29, 2026:
#   RTX 3080: NVIDIA Ampere GA102 whitepaper v1, Table 2: 760 GB/s; FP16 Tensor 59.5/119 TFLOPS
#     (dense/sparse) with FP32 accumulate, 119/238 with FP16 accumulate; INT8 238/476 TOPS.
#     https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf
#   RTX 4090: NVIDIA Ada GPU architecture whitepaper, Table 2: 1008 GB/s; FP16 Tensor 165.2/330.4
#     with FP32 accumulate, 330.3/660.6 with FP16 accumulate; INT8 660.6/1321.2 TOPS.
#     https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
#   H100 SXM: https://www.nvidia.com/en-us/data-center/h100/ : 3.35 TB/s; FP16 Tensor 1,979 TFLOPS
#     "with sparsity", i.e. 989 dense; data-center parts run FP16 at the same rate with either
#     accumulator, so the GeForce halving for FP32 accumulate does not apply.
# The fp32-accumulate column is the consistent one across the three. If llama.cpp's prefill runs
# at the fp16-accumulate or int8 rate (MMQ for Q8_0), the 3080 and 4090 double and the H100 does
# not: its prefill ratio drops from 16.6x to 8.3x and the H100 bar rises from 2.52x to 2.54x on
# the search set, so the choice here is the one generous to the H100.
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
