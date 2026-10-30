"""The paper's numbers against the results (plan item 18).

main.tex (env DOMELINE_TEX, default /localhome/tsousape/Domeline/main.tex) is
checked against paper/claims.yaml and paper/numbers.json, which
experiments/make_numbers.py computes from the results files. The full list
with context is in docs/claims_report.md (python experiments/claims.py --report
docs/claims_report.md).
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import claims  # noqa: E402

needs_tex = pytest.mark.skipif(not claims.TEX.exists(), reason=f"{claims.TEX} not found (set DOMELINE_TEX)")


@pytest.fixture(scope="module")
def res():
    return claims.run()


def _fmt(nums):
    return "\n".join(f"  l.{n.line}: paper {n.text} vs {n.computed or n.source}  [...{n.context}...]" for n in nums)


def test_numbers_are_current(tmp_path):
    """paper/numbers.json matches what make_numbers.py computes now (integrator-free part)."""
    subprocess.run([sys.executable, str(ROOT / "experiments/make_numbers.py"), "--fast", "--out", str(tmp_path)],
                   check=True, capture_output=True)
    new = json.loads((tmp_path / "numbers.json").read_text())
    old = json.loads(claims.NUMBERS.read_text())
    stale = [f"{k}: {old.get(k, {}).get('text')} -> {v['text']}" for k, v in new.items()
             if old.get(k, {}).get("text") != v["text"]]
    assert not stale, "paper/numbers.json is stale; rerun experiments/make_numbers.py:\n" + "\n".join(stale)


@needs_tex
def test_registry_matches_paper(res):
    """Every registered phrase is still in the paper and lists one entry per number."""
    msg = res.errors + [f"phrase no longer in the paper: {s['t']}" for s in res.stale]
    assert not msg, "\n".join(msg)


@needs_tex
def test_no_mismatched_numbers(res):
    bad = [n for n in res.nums if n.status == "mismatched"]
    assert not bad, f"{len(bad)} numbers in main.tex disagree with the results:\n" + _fmt(bad)


@needs_tex
def test_no_unregistered_result_numbers(res):
    """Hard-coded result-like numbers must be registered in paper/claims.yaml (or be a numbers.tex macro)."""
    bad = [n for n in res.nums if n.status == "unregistered" and n.resultlike]
    assert not bad, f"{len(bad)} result-like numbers are not in paper/claims.yaml:\n" + _fmt(bad)


@needs_tex
def test_readings_are_computed(res):
    """Each Domeline reading the paper names is computed by harness/evaluate.py (pending ones warn)."""
    for r in res.readings:
        if r["status"] == "pending":
            warnings.warn(f"reading '{r['name']}' named by the paper is pending: {r['why']}")
    bad = [r for r in res.readings if r["status"] == "fail"]
    assert not bad, "\n".join(f"{r['name']}: {r['why']}" for r in bad)


def test_reading_names_from_code():
    names = claims.eval_fields()
    assert "Eval.terms" in names and "Evaluator.amdahl_bound_s" in names
