# Agent interface (M5): C owns the loop (harness/team.py), A owns the LLM agents (harness/llm_agents.py)

Written by A first, as TASK-M5 asks. C: if something here doesn't fit the loop, change it and note the
change in your report; I'll follow what is on autopilot when it's merged. Read from my branch with
`git show h/m5-llm:docs/harness-dev/AGENT_IFACE.md`.

## 1. Ownership (both sides import it from `harness/team.py`, C defines it)

```python
ROLES = ("mapping", "hardware", "capacity")
def owner_of(knob) -> str                 # works for small knobs and LargeSpace slot names
def knobs_of(space, role) -> list[str]    # the knobs of `space` that role may change
```
My proposal (C decides; the agents only use `knobs_of`):

| role | small-space knobs | LargeSpace knobs |
|---|---|---|
| mapping (moves the point) | mvm, lm_head, attention, softmax, vector, edge | every placement slot `L<l>.<op>`, `L<l>.attention`, lm_head, final_norm, embed, sample |
| hardware (raises the roof) | xbar_tech, link, vfu_units, vfu_bw, star_units, xbar_tile, adc_per_ima, cell_bits | same |
| capacity (moves the wall) | xbar_scale, vfu_mem, kv_bits | xbar_scale, vfu_mem, every `L<l>.kv_bits` |

The term -> owner routing table (`ROUTE`) is C's; the agents don't depend on it.

## 2. What the loop passes an agent: `AgentContext` (dataclass, defined in harness/team.py)

| field | type | meaning |
|---|---|---|
| `role` | str | the agent's role |
| `point` | dict | current accepted point (as `Toolbox.read` saw it) |
| `reading` | dict \| None | `Toolbox.read(point)`; None in no-readings ablations |
| `knobs` | dict[str, list] | the agent's own knobs -> their options (only these) |
| `history` | list[dict] | recent proposals, oldest first: `{"role", "changes", "why", "outcome", "time_s", "speedup", "violations"}` with outcome in accepted / not_faster / illegal / not_owned / repeated |
| `feedback` | dict \| None | set on a retry: the last rejection of *this* agent (`outcome`, `violations` = list of `[rule, op, reason]`) |
| `toolbox` | Toolbox | agents may call `read`, `moves(point, term)` and `verify`; **never `evaluate`** (the loop owns the budget) |
| `record` | RunRecord | agents call `record.add_tokens(model_id, n_in, n_out)` on every LLM call |
| `budget_left` | int | evaluations left |
| `call_id` | int | increasing counter from the loop; LLM agents derive their seed from it |

## 3. What an agent returns: `Proposal` (dataclass, defined in harness/team.py)

```python
@dataclass
class Proposal:
    changes: dict          # knob -> value, concrete knob names only (the agent expands patterns)
    why: str               # one or two sentences; logged
    stop: bool = False     # True: nothing left to do in my knobs (then changes may be empty)
    meta: dict = field(default_factory=dict)   # model id, tokens, raw response, parse retries
```
An agent is any object with `role: str` and `propose(ctx: AgentContext) -> Proposal`. C's deterministic
agents (rule-table moves filtered by owner) and my LLM agents have this same signature.

The loop applies `{**ctx.point, **changes}`, rejects changes outside `ctx.knobs` (outcome not_owned), runs
`verify` (illegal -> violations back in `feedback`, retry within C's budget), evaluates, and accepts if
faster by more than eps. Group moves (one role in every layer, or a window of layers) arrive as many
concrete changes in one Proposal. The LLM writes them as patterns (`L*.rope`, `L10-20.attention`) and
`llm_agents.expand_changes(space, changes)` expands them before returning.

## 4. What A ships in harness/llm_agents.py

- `Backend` (call(system, prompt, schema, seed) -> (dict, usage)); `ClaudeCLI("claude-sonnet-5-5" |
  "claude-haiku-5-5")`, `LocalLlama()` (mini-coder-4b on llama-server :8080), `FakeBackend(responses)`.
- `LLMAgent(role, backend)`: implements `propose`.
- Baselines (own loops; they return a RunRecord): `single_agent_search(ev, backend, with_readings=False)`
  on Space or LargeSpace, and `eat_team_search(ev, backend)`, an Evolving-Agent-Teams-style supervisor
  team without readings or routing.
- Until team.py exists on autopilot, llm_agents.py defines fallback `AgentContext`/`Proposal` with these
  exact fields and imports C's when present.
