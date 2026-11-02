"""M0 regression: the paper's small-space numbers (docs/results.md) from the current code.

Every harness change must keep these passing: 575,400 designs, 280,296 feasible, optimum
9.48x, the walk within 0.7% of it, reaching the 1% band at eval 20. Uses the real 45 search
sessions (~1 min).
"""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

from m0_repro import PAPER, count_designs, parse, setup, sweep_rows  # noqa: E402

from harness.evaluate import Evaluator  # noqa: E402
from harness.search import domeline_walk  # noqa: E402

WALK_POINT = {"mvm": "xbar", "lm_head": "xbar", "attention": "xbar", "xbar_tech": "xbar_transpose",
              "softmax": "star", "vector": "star", "edge": "cpu", "xbar_scale": 1.0, "vfu_units": 1,
              "vfu_bw": 64e9, "vfu_mem": 4e9, "star_units": 64, "link": "cxl", "kv_bits": 8}


@pytest.fixture(scope="module")
def env():
    return setup()


@pytest.fixture(scope="module")
def rows():
    return sweep_rows()


def test_design_count(env):
    assert count_designs(env[0]) == PAPER["n_designs"]


def test_sweep_feasible_and_optimum(rows):
    feas = [r for r in rows if r["feasible"] == "1"]
    assert len(rows) == PAPER["n_designs"]
    assert len(feas) == PAPER["n_feasible"]
    assert max(float(r["speedup"]) for r in feas) == pytest.approx(PAPER["optimum"], abs=0.005)


def test_sweep_still_describes_the_evaluator(env, rows):
    """A seeded sample of sweep rows, plus the optimum, re-timed by today's code."""
    space, train, g = env
    feas = [r for r in rows if r["feasible"] == "1"]
    sample = random.Random(0).sample(rows, 15) + [max(feas, key=lambda r: float(r["speedup"]))]
    ev = Evaluator(space, train, g)
    for r in sample:
        e = ev(parse(r))
        assert int(e.feasible) == int(r["feasible"])
        assert e.time_s == pytest.approx(float(r["time_s"]), rel=1e-6)


def test_walk(env):
    space, train, g = env
    ev = Evaluator(space, train, g)
    tr = domeline_walk(ev)
    w = tr.steps[-1].eval
    assert {k: w.point[k] for k in WALK_POINT} == WALK_POINT
    assert 1 - w.speedup_vs_gpu / PAPER["optimum"] == pytest.approx(PAPER["walk_gap"], abs=0.0005)
    first = next(e.n for e in ev.log if e.feasible and e.speedup_vs_gpu >= 0.99 * PAPER["optimum"])
    assert first == PAPER["walk_eval"]
