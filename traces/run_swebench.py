"""T1/T2: run mini-swe-agent 1.17.5 + mini-coder-4b on SWE-bench Verified and log per-turn traces.

The model is served by llama-server on this host's RTX 3080. Every turn logs:
the context length at the start of decode, the tokens reused from the KV
cache and the new tokens prefilled, the tokens generated, the prefill and
decode times (from llama-server's own timings), the command, its execution
time, and the tokens its output adds to the context.

Run with the agent venv (mini-swe-agent 1.17.5):
  /localhome/tsousape/tools/agentvenv/bin/python traces/run_swebench.py \
      --instances traces/swebench_sample20.tsv --out results/traces/swebench
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import requests
import yaml

os.environ.setdefault("MSWEA_DOCKER_EXECUTABLE", "podman")
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")

from datasets import load_dataset  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.run.extra.swebench import get_sb_environment  # noqa: E402


class ContextExceeded(Exception):
    pass


class LlamaServerModel:
    """Calls llama-server's OpenAI-compatible endpoint and keeps its per-request timings."""

    def __init__(self, base_url, model_name="mini-coder-4b", temperature=0.0, max_tokens=4096):
        self.base_url = base_url.rstrip("/")
        self.config = {"model_name": model_name, "temperature": temperature, "max_tokens": max_tokens}
        self.cost = 0.0
        self.n_calls = 0

    def query(self, messages, **kwargs):
        body = {"model": self.config["model_name"],
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "temperature": self.config["temperature"], "max_tokens": self.config["max_tokens"],
                "cache_prompt": True}
        t0 = time.perf_counter()
        r = requests.post(self.base_url + "/v1/chat/completions", json=body, timeout=3600)
        wall = time.perf_counter() - t0
        if r.status_code == 400 and "context" in r.text.lower():
            raise ContextExceeded(r.text[:500])
        r.raise_for_status()
        d = r.json()
        self.n_calls += 1
        return {"content": d["choices"][0]["message"]["content"] or "",
                "extra": {"timings": d.get("timings", {}), "usage": d.get("usage", {}),
                          "finish_reason": d["choices"][0].get("finish_reason"), "llm_wall_s": wall}}

    def n_tokens(self, text):
        r = requests.post(self.base_url + "/tokenize", json={"content": text}, timeout=60)
        r.raise_for_status()
        return len(r.json()["tokens"])

    def get_template_vars(self):
        return dict(self.config) | {"n_model_calls": self.n_calls, "model_cost": self.cost}


class TracingAgent(DefaultAgent):
    """DefaultAgent that logs one trace record per turn."""

    def __init__(self, *args, trace_path: Path, session: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_path = trace_path
        self.session = session
        self.turn = 0
        self._rec = None

    def query(self):
        t_start = time.time()
        response = super().query()
        t = response["extra"]["timings"]
        self.turn += 1
        self._rec = {
            "session": self.session, "turn": self.turn, "t_start": t_start,
            "cached_tokens": t.get("cache_n"), "prefill_tokens": t.get("prompt_n"),
            "ctx_start": (t.get("cache_n") or 0) + (t.get("prompt_n") or 0),
            "gen_tokens": t.get("predicted_n"),
            "t_prefill_s": (t.get("prompt_ms") or 0) / 1e3, "t_decode_s": (t.get("predicted_ms") or 0) / 1e3,
            "llm_wall_s": response["extra"]["llm_wall_s"], "finish_reason": response["extra"]["finish_reason"],
            "command": None, "t_tool_s": 0.0, "returncode": None, "obs_chars": 0, "obs_tokens": 0,
        }
        return response

    def execute_action(self, action):
        self._rec["command"] = action["action"]
        t0 = time.perf_counter()
        try:
            out = super().execute_action(action)
            self._rec["returncode"] = out.get("returncode")
            return out
        finally:
            self._rec["t_tool_s"] = time.perf_counter() - t0

    def add_message(self, role, content, **kwargs):
        super().add_message(role, content, **kwargs)
        if role == "user" and self._rec is not None:
            # The observation (or a format-error message) the next turn will prefill.
            self._rec["obs_chars"] = len(content)
            self._rec["obs_tokens"] = self.model.n_tokens(content)
            self._flush()

    def _flush(self):
        with self.trace_path.open("a") as f:
            f.write(json.dumps(self._rec) + "\n")
        self._rec = None


def exec_overhead(env, n=5):
    """Seconds podman exec takes for a no-op: the harness's own cost per tool call."""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        env.execute("true")
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[n // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=str(Path(__file__).with_name("mini_coder_v1.yaml")))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--wall-limit-s", type=int, default=2400)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    config["environment"]["executable"] = os.environ["MSWEA_DOCKER_EXECUTABLE"]
    ids = [l.split("\t")[0] for l in Path(args.instances).read_text().splitlines()[1:] if l.strip()]
    if args.limit:
        ids = ids[: args.limit]
    ds = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for iid in ids:
        # A fresh model per session: v1 checks step_limit against the model's call count.
        model = LlamaServerModel(args.base_url, temperature=config["model"]["model_kwargs"]["temperature"])
        sess_dir = out / iid
        if (sess_dir / "result.json").exists():
            continue
        sess_dir.mkdir(exist_ok=True)
        trace = sess_dir / "trace.jsonl"
        trace.unlink(missing_ok=True)
        inst = ds[iid]
        t0 = time.time()
        env = get_sb_environment(config, inst)
        t_env = time.time() - t0
        result = {"instance_id": iid, "repo": inst["repo"], "t_env_setup_s": t_env,
                  "exec_overhead_s": exec_overhead(env)}
        agent = TracingAgent(model, env, trace_path=trace, session=iid, **config["agent"])
        deadline = time.time() + args.wall_limit_s
        orig_step = agent.step

        def step_with_deadline():
            if time.time() > deadline:
                from minisweagent.agents.default import LimitsExceeded
                raise LimitsExceeded("wall limit")
            return orig_step()
        agent.step = step_with_deadline
        t1 = time.time()
        try:
            exit_status, submission = agent.run(inst["problem_statement"])
        except ContextExceeded as e:
            exit_status, submission = "ContextExceeded", str(e)
        except Exception as e:
            exit_status, submission = type(e).__name__, traceback.format_exc()
        finally:
            if agent._rec is not None:
                agent._flush()
            try:
                env.cleanup()
            except Exception:
                pass
        result.update({"exit_status": exit_status, "session_wall_s": time.time() - t1,
                       "n_turns": agent.turn, "submission": submission})
        (sess_dir / "trajectory.json").write_text(json.dumps(agent.messages, indent=1))
        (sess_dir / "result.json").write_text(json.dumps(result, indent=1))
        with (out / "preds.jsonl").open("a") as f:
            f.write(json.dumps({"instance_id": iid, "model_name_or_path": "mini-coder-4b-Q8_0",
                                "model_patch": submission if exit_status == "Submitted" else ""}) + "\n")
        print(f"{iid}: {exit_status} turns={agent.turn} wall={result['session_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
