"""Claim checker: every number in the paper against the number the results give.

The registry paper/claims.yaml maps each claim of main.tex (a phrase copied
from the paper, with its numbers) to macros of paper/numbers.json, which
experiments/make_numbers.py computes from the results files. For every number
in a registered phrase the checker reports one status:

  matched     the paper's number equals the computed one at the printed precision
              (or satisfies the bound: "within 0.6%" holds if the value is <= 0.6)
  mismatched  it does not
  unsourced   no results file computes it (external citation, log entry, assumption)
  setup       a condition of the claim, not a result (a cap, a context length)

Numbers in main.tex that look like results (speedups, percentages, mm2, ms,
evaluation and design counts, K tokens) but are in no registered phrase are
reported as unregistered. It also checks that every Domeline reading the paper
names is a reading the code computes (fields of harness/evaluate.py).

  python experiments/claims.py                      # print the summary
  python experiments/claims.py --report docs/claims_report.md   # rewrite its generated section
  DOMELINE_TEX=/path/main.tex python experiments/claims.py
"""

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
TEX = Path(os.environ.get("DOMELINE_TEX", "/localhome/tsousape/Domeline/main.tex"))
REGISTRY = ROOT / "paper/claims.yaml"
NUMBERS = ROOT / "paper/numbers.json"
EVALUATE = ROOT / "harness/evaluate.py"

# A number as LaTeX prints it: 575,400 / 9.42 / 64K / 2.16M. Not part of a word (fp16, H100, Q8\_0, Qwen3).
NUM_RE = re.compile(r"(?<![A-Za-z0-9_.\\^])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?([KM](?![A-Za-z]))?(?![A-Za-z0-9])")
# What makes a number look like a result: its unit, or the word before it.
UNIT_RE = re.compile(
    r"^(?:\s*--\s*[\d.,]+[KM]?)?(?:\s*(?:\\X|×|\\%|%|\\times(?!\$?\d))"
    r"|\s*(?:\\,|~|\s)?\s*(?:mm|ms|µs|us\b|GB|TB|TF|nm\b|ns\b|h\b|tokens|CPU-years)"
    r"|\s+(?:evaluations|evals|designs|candidates|sessions|months|units|caps|knobs|seeds|runs|cases|layers"
    r"|arrays|moves|ADCs)\b"
    r"|\s+of\s+\d)")
BEFORE_RE = re.compile(r"(?:\beval|\bevals|\bopt\.\\\s|\bbest)\s*$")
# LaTeX whose arguments are not prose: their digits are masked out before numbers are extracted.
MASK_RE = re.compile(
    r"\\(?:cite|ref|eqref|label|input|includegraphics|bibliography|bibliographystyle|usetikzlibrary|"
    r"setlength|renewcommand|newcommand|begin\{tabular\}|begin\{algorithmic\}|begin\{figure\*?\}|"
    r"begin\{table\*?\}|begin\{equation\}|IEEEkeywords)"
    r"(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{[^{}]*\})*\})*(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})?")
# Numbers that are names, not quantities: GPU models, years.
NAMES = {"3080", "4090", "2016"}
MATH_ENV_RE = re.compile(r"\\begin\{(equation|aligned)\}.*?\\end\{\1\}", re.S)


@dataclass
class Num:
    pos: int                    # offset in the normalized text
    text: str                   # as printed, e.g. "575,400", "9.42", "64K"
    value: float
    decimals: int
    line: int
    context: str
    resultlike: bool
    status: str = "unregistered"
    macro: Optional[str] = None
    computed: Optional[str] = None
    source: str = ""
    note: str = ""
    claim: Optional[str] = None


@dataclass
class Result:
    nums: List[Num]
    stale: List[dict] = field(default_factory=list)       # registry phrases no longer in the paper
    errors: List[str] = field(default_factory=list)       # malformed registry entries
    flags: List[dict] = field(default_factory=list)       # phrasing/scope issues noted in the registry
    readings: List[dict] = field(default_factory=list)
    macros_used: List[str] = field(default_factory=list)  # \macros of numbers.tex the paper already uses


# --------------------------------------------------------------------------- text
def _blank(m):
    return re.sub(r"\S", " ", m.group(0))


def _mask(m):
    return re.sub(r"\S", "x", m.group(0))


def load_tex(path=TEX):
    """(normalized text, masked text, original line of each normalized char)."""
    raw = Path(path).read_text()
    i = raw.find("\\begin{document}")
    body = (re.sub(r"[^\n]", " ", raw[:i]) + raw[i:]) if i >= 0 else raw
    body = re.sub(r"(?<!\\)%[^\n]*", _blank, body)                        # comments
    masked = MASK_RE.sub(_mask, body)
    masked = MATH_ENV_RE.sub(_mask, masked)
    norm, mnorm, lines = [], [], []
    line, prev_ws = 1, True
    for c, mc in zip(body, masked):
        if c.isspace():
            if not prev_ws:
                norm.append(" ")
                mnorm.append(" ")
                lines.append(line)
            prev_ws = True
        else:
            norm.append(c)
            mnorm.append(mc)
            lines.append(line)
            prev_ws = False
        if c == "\n":
            line += 1
    return "".join(norm), "".join(mnorm), lines


def norm_phrase(s):
    return re.sub(r"\s+", " ", s.strip())


def table_spans(text):
    return [(m.start(), m.end()) for m in re.finditer(r"\\begin\{tabular\}.*?\\end\{tabular\}", text)]


def numbers_in(text, masked, lines, lo=0, hi=None):
    hi = len(text) if hi is None else hi
    tabs = table_spans(text)
    out = []
    for m in NUM_RE.finditer(masked, lo, hi):
        s = m.group(0)
        if s in NAMES:
            continue
        dec = len(m.group(2)) - 1 if m.group(2) else 0
        val = float((m.group(1) + (m.group(2) or "")).replace(",", ""))
        after = text[m.end(): m.end() + 30]
        before = text[max(0, m.start() - 12): m.start()]
        in_tab = any(a <= m.start() < b for a, b in tabs)
        rl = bool(m.group(3)) or bool(UNIT_RE.match(after)) or bool(BEFORE_RE.search(before)) or in_tab
        ctx = text[max(0, m.start() - 50): m.end() + 40]
        out.append(Num(m.start(), s, val, dec, lines[m.start()], ctx, rl))
    return out


# --------------------------------------------------------------------------- checks
def check_one(paper, dec, spec, numbers):
    """Compare one printed number with its macro(s). Returns (status, computed text, source, note)."""
    if spec == "-" or spec is None:
        return "setup", None, "", ""
    if isinstance(spec, str):
        spec = {"m": spec}
    if "setup" in spec:
        return "setup", None, spec["setup"], spec.get("note", "")
    if "src" in spec:
        return "unsourced", None, spec["src"], spec.get("note", "")
    macros = spec["m"] if isinstance(spec["m"], list) else [spec["m"]]
    texts, ok, srcs = [], True, []
    for mac in macros:
        if mac not in numbers:
            return "mismatched", None, f"macro {mac} not in numbers.json", spec.get("note", "")
        n = numbers[mac]
        v = n["value"]
        texts.append(f"\\{mac}={n['text']}")
        srcs.append(n["source"])
        if v is None:
            ok = False
            continue
        if spec.get("abs"):
            v = abs(v)
        v *= spec.get("scale", 1.0)
        half = 0.5 * 10 ** -dec
        tol = half + spec.get("tol", 0.0) + spec.get("rel", 0.0) * abs(v) + 1e-9
        chk = spec.get("check", "eq")
        if chk == "eq":
            ok &= abs(paper - v) <= tol
        elif chk == "le":        # the paper states an upper bound ("within 0.6%", "under 1%")
            ok &= v <= paper + tol
        elif chk == "ge":        # a lower bound ("just past 64K")
            ok &= v >= paper - tol
        else:
            raise ValueError(chk)
    return ("matched" if ok else "mismatched"), ", ".join(texts), "; ".join(dict.fromkeys(srcs)), \
        spec.get("note", "")


def eval_fields(path=EVALUATE):
    """Names the code exposes: Eval's fields and properties, Evaluator's methods and properties."""
    tree = ast.parse(Path(path).read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in ("Eval", "Evaluator"):
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    names.add(f"{node.name}.{b.target.id}")
                elif isinstance(b, ast.FunctionDef):
                    names.add(f"{node.name}.{b.name}")
    return names


def check_readings(text, reg):
    """Each reading the paper names must be computed by the code (pending ones only warn)."""
    code = eval_fields()
    out = []
    known = {r["name"]: r for r in reg.get("readings", [])}
    # Readings the paper defines: the \emph{} items of the "Readings." paragraph.
    named = set()
    m = re.search(r"\\textbf\{Readings\.\}(.*?)\\end\{itemize\}", text, re.S)
    if m:
        for e in re.findall(r"\\item the \\emph\{([^}]*)\}", m.group(1)):
            named.add(e)
    for name, r in known.items():
        pats = r.get("paper", [name])
        if any(re.search(p, text) for p in pats):
            named.add(name)
    for name in sorted(named):
        r = known.get(name)
        if r is None:
            out.append({"name": name, "status": "fail", "why": "paper names a reading the registry does not know"})
            continue
        have = [c for c in r.get("code", []) if c in code]
        if have:
            out.append({"name": name, "status": "ok", "why": "computed by " + ", ".join(have)})
        elif r.get("pending"):
            out.append({"name": name, "status": "pending",
                        "why": f"not in harness/evaluate.py yet (expected one of {r.get('code')}); {r.get('note', '')}"})
        else:
            out.append({"name": name, "status": "fail", "why": f"none of {r.get('code')} in harness/evaluate.py"})
    return out


def run(tex=TEX, registry=REGISTRY, numbers_path=NUMBERS):
    text, masked, lines = load_tex(tex)
    reg = yaml.safe_load(Path(registry).read_text())
    numbers = json.loads(Path(numbers_path).read_text())
    nums = numbers_in(text, masked, lines)
    by_pos = {n.pos: n for n in nums}
    res = Result(nums=nums)
    for c in reg["claims"]:
        phrase = norm_phrase(c["t"])
        starts = [m.start() for m in re.finditer(re.escape(phrase), text)]
        if not starts:
            res.stale.append(c)
            continue
        specs = c.get("n", [])
        for st in starts:
            inner = numbers_in(text, masked, lines, st, st + len(phrase))
            if len(inner) != len(specs):
                res.errors.append(f"claim {c.get('id')}: phrase has {len(inner)} numbers "
                                  f"({', '.join(x.text for x in inner)}), registry lists {len(specs)}")
                continue
            for x, spec in zip(inner, specs):
                n = by_pos[x.pos]
                n.status, n.computed, n.source, n.note = check_one(n.value, n.decimals, spec, numbers)
                if isinstance(spec, dict):
                    n.macro = spec.get("m") if isinstance(spec.get("m"), str) else None
                elif isinstance(spec, str) and spec != "-":
                    n.macro = spec
                n.claim = c.get("id")
        if c.get("flag"):
            res.flags.append({"id": c.get("id"), "line": lines[starts[0]], "flag": c["flag"], "t": phrase})
    res.readings = check_readings(text, reg)
    res.macros_used = sorted(m for m in set(re.findall(r"\\([A-Za-z]+)", text)) if m in numbers)
    return res


# --------------------------------------------------------------------------- report
def counts(res):
    c = {}
    for n in res.nums:
        if n.status == "unregistered" and not n.resultlike:
            continue
        c[n.status] = c.get(n.status, 0) + 1
    return c


def esc(s):
    return (s or "").replace("|", "\\|").replace("\n", " ")


def report(res):
    c = counts(res)
    L = [f"Generated by `python experiments/claims.py --report docs/claims_report.md` from `{TEX}`, "
         f"`paper/claims.yaml` and `paper/numbers.json`.", "",
         "| Status | Numbers |", "|---|---|"]
    for k in ("matched", "mismatched", "unsourced", "setup", "unregistered"):
        L.append(f"| {k} | {c.get(k, 0)} |")
    L.append("")
    if res.errors or res.stale:
        L += ["### Registry problems", ""]
        L += [f"- {e}" for e in res.errors]
        L += [f"- stale (phrase not in the paper): `{esc(s['t'])}`" for s in res.stale]
        L.append("")
    L += ["### Mismatched", "", "| Line | Paper | Computed | In context | Note |", "|---|---|---|---|---|"]
    for n in res.nums:
        if n.status == "mismatched":
            L.append(f"| {n.line} | {n.text} | {esc(n.computed)} | `{esc(n.context)}` | {esc(n.note)} |")
    L += ["", "### Phrasing and scope flags", "", "| Line | Claim | Flag |", "|---|---|---|"]
    for f in res.flags:
        L.append(f"| {f['line']} | `{esc(f['t'][:90])}` | {esc(f['flag'])} |")
    L += ["", "### Unsourced (no results file computes them)", "", "| Line | Paper | Source given | In context |",
          "|---|---|---|---|"]
    for n in res.nums:
        if n.status == "unsourced":
            L.append(f"| {n.line} | {n.text} | {esc(n.source)} | `{esc(n.context)}` |")
    L += ["", "### Unregistered result-like numbers", ""]
    un = [n for n in res.nums if n.status == "unregistered" and n.resultlike]
    if not un:
        L.append("None.")
    else:
        L += ["| Line | Paper | In context |", "|---|---|---|"]
        L += [f"| {n.line} | {n.text} | `{esc(n.context)}` |" for n in un]
    L += ["", "### Domeline readings named by the paper", "", "| Reading | Status | Why |", "|---|---|---|"]
    for r in res.readings:
        L.append(f"| {r['name']} | {r['status']} | {esc(r['why'])} |")
    L += ["", "### Every number, in paper order", "",
          "| Line | Paper | Status | Computed | Source | In context |", "|---|---|---|---|---|---|"]
    for n in res.nums:
        if n.status == "unregistered" and not n.resultlike:
            continue
        L.append(f"| {n.line} | {n.text} | {n.status} | {esc(n.computed)} | {esc(n.source)} | `{esc(n.context)}` |")
    return "\n".join(L) + "\n"


BEGIN, END = "<!-- BEGIN GENERATED (experiments/claims.py) -->", "<!-- END GENERATED -->"


def write_report(path, res):
    p = Path(path)
    body = report(res)
    old = p.read_text() if p.exists() else f"# Claims report\n\n{BEGIN}\n{END}\n"
    if BEGIN not in old:
        old += f"\n{BEGIN}\n{END}\n"
    a, rest = old.split(BEGIN, 1)
    _, b = rest.split(END, 1)
    p.write_text(f"{a}{BEGIN}\n{body}{END}{b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", help="rewrite the generated section of this markdown file")
    args = ap.parse_args()
    if not TEX.exists():
        sys.exit(f"{TEX} not found (set DOMELINE_TEX)")
    res = run()
    print(counts(res))
    for e in res.errors:
        print("registry:", e)
    for s in res.stale:
        print("stale:", s["t"])
    for n in res.nums:
        if n.status == "mismatched":
            print(f"MISMATCH l.{n.line}: paper {n.text} vs {n.computed}  [{n.context}]")
    for n in res.nums:
        if n.status == "unregistered" and n.resultlike:
            print(f"unregistered l.{n.line}: {n.text}  [{n.context}]")
    for r in res.readings:
        print(f"reading {r['name']}: {r['status']} ({r['why']})")
    if args.report:
        write_report(args.report, res)
        print("wrote", args.report)


if __name__ == "__main__":
    main()
