"""Synthetic sessions for testing the harness before real traces exist. Not for results."""

import random

from .session import Session, Turn


def synthetic_sessions(n=5, seed=0, prompt=1800, turns=(15, 45), gen=(60, 400),
                       obs_median=350, tool_median_s=0.4, long_tool_p=0.1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ctx = prompt
        ts = []
        prev_gen = 0
        for t in range(rng.randint(*turns)):
            g = rng.randint(*gen)
            obs = int(rng.lognormvariate(0, 1.0) * obs_median)
            pre = (prompt if t == 0 else prev_gen + obs + 10)
            ctx_start = ctx if t == 0 else ctx + pre
            tool = rng.lognormvariate(0, 1.0) * tool_median_s
            if rng.random() < long_tool_p:
                tool += rng.uniform(5, 60)          # test suites, builds
            ts.append(Turn(ctx_start=ctx_start, prefill_tokens=pre, gen_tokens=g, t_tool_s=tool))
            ctx = ctx_start + g
            prev_gen = g
        out.append(Session(f"synthetic-{i}", ts, meta={"synthetic": True}))
    return out
