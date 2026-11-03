# Agent interface (M5): loop = Agent C (`harness/team.py`), agents = Agent A (`harness/llm_agents.py`)

Written by C. The loop calls agents only through what is below; A's LLM agents and C's rule agents
both implement it. Anything not here (prompting, tool calls inside an agent, retries on parse errors)
is the agent's business.

```python
from harness.team import Move, AgentContext, OWNER_KNOBS, TERM_OWNERS, owner_of

@dataclass
class Move:
    changes: Dict[str, Any]      # knob or slot -> new value, e.g. {"softmax": "star"} or {"L3.rope": "star"}
    why: str = ""                # the agent's reason (logged)
    priority: float = math.inf   # ordering among candidates of one round; rule agents use the rule-table
                                 # index, LLM agents leave it inf (then: agenda order, then proposal order)
    agent: str = ""              # filled in by the loop

@dataclass
class AgentContext:
    role: str                    # "mapping" | "hardware" | "capacity"
    term: str                    # the reading item routed to this agent: a term name ("host", "launch",
                                 # "transfer", "write", "vfu_puma", "softmax_star", "xbar_transpose", ...),
                                 # or "capacity" (the design hits its capacity wall) or "area" (over the cap)
    point: Dict                  # current design (canonical); the agent changes only its own knobs
    reading: Dict                # Toolbox.read(point): binding term, ranking + headroom, Amdahl gap, wall, D
    knobs: Dict[str, list]       # the agent's own knobs -> allowed options (large space: slot names too)
    rule_moves: List[Move]       # the rule table's candidates for this term and this agent's knobs
    history: List[Dict]          # this run's recent proposals: move, agent, outcome
                                 # ("accepted" | "slower" | "illegal" | "ownership" | "duplicate"), time_s, violations
    feedback: Optional[List]     # on a retry: the verifier's violations [(rule, op, reason)] for the last move
    round: int                   # 0 on the first ask after a new accepted design; increments while no accept
    budget_left: float           # evaluations left

class Agent(Protocol):
    role: str
    def propose(self, ctx: AgentContext) -> List[Move]: ...   # [] = pass this round
```

Rules the loop enforces (an agent need not, but should, to save budget):
- **Ownership.** Every key in `Move.changes` must be in `ctx.knobs`. A move that touches another
  agent's knob is rejected without evaluation and logged as outcome "ownership".
- **Verify.** Each move is verified before it is evaluated (when a verifier is attached). If it is
  illegal, the loop calls `propose` again on the same agent with `feedback` set, up to `retry_budget`
  times per round. Illegal moves cost no evaluation.
- **Accept** if the move is feasible and more than `eps` (0.5%) faster. A faster move that is over the
  area cap is first repaired by the supervisor (area moves), as in the walk.
- **Rounds.** After an accept, every agent's next ask is round 0. A round asks each owner of each agenda
  term once; a round where every agent passes and nothing is accepted ends the run (unless the
  supervisor's restructure finds a move).
- **Tokens.** LLM agents add their usage via `record.add_tokens(model_id, n_in, n_out)`; the loop passes
  the RunRecord in the agent's constructor or as `agent.record` (set by the loop before the run if the
  attribute exists and is None).

Ownership (`OWNER_KNOBS`; large-space slots: `L<l>.kv_bits` -> capacity, every other slot -> mapping):

| owner | knobs |
|---|---|
| mapping | mvm, lm_head, attention, softmax, vector, edge, and every placement slot |
| hardware | xbar_tech, vfu_units, vfu_bw, star_units, link, xbar_tile, adc_per_ima, cell_bits |
| capacity | xbar_scale, vfu_mem, kv_bits, and every `L<l>.kv_bits` slot |

Term -> owners (`TERM_OWNERS`; first = the owner the supervisor routes the binding term to; the rest
are asked too, in this order):

| term class | owners |
|---|---|
| capacity (wall) | capacity, mapping |
| area | hardware, capacity, mapping |
| host | mapping |
| launch, transfer | mapping, hardware |
| write | hardware, mapping, capacity |
| vfu (vector unit term) | hardware, mapping |
| star (softmax unit term) | hardware |
| xbar (crossbar term) | hardware, mapping |
| unit (any other tech) | capacity, hardware |

## H9: the supervisor hook and the ensemble runner (written by C, before the code)

### Supervisor (`team_search(..., supervisor=...)`)

```python
from harness.team import SupervisorContext, Supervisor, team_search

@dataclass
class SupervisorContext:
    trigger: str                 # "stall": stall_k proposals since the last accept while amdahl_gap > stall_gap
                                 # "exhausted": the round produced no accept and the loop is about to stop
    point: Dict                  # current design (canonical)
    reading: Dict                # Toolbox.read(point), as in AgentContext
    knobs: Dict[str, list]       # EVERY knob (and large-space slot) -> options; the supervisor may touch any,
                                 # so one Move can span roles (e.g. {"xbar_tech": ..., "L3.attention": ...})
    owners: Dict[str, str]       # knob/slot -> owning role (OWNER_KNOBS), for the prompt
    coordinated: List[Move]      # the loop's deterministic restructure candidates for this design
                                 # (tie / window, built from this round's role moves); [] on the small space
    singles: List[Move]          # this round's role-agent candidates (what the team proposed)
    history: List[Dict]          # recent proposals and outcomes (same entries as AgentContext.history)
    accepted: List[Dict]         # the team's accepted moves so far: move, agent, term, eval_n, time_s, speedup
    escalation: int              # 0-based index of this call in the run
    feedback: Optional[List]     # on a retry: the verifier's violations [(rule, op, reason)] for the last move
    budget_left: float           # evaluations left

class Supervisor(Protocol):
    name: str                    # logged (e.g. "sonnet-supervisor")
    record: Optional[RunRecord]  # set by the loop if the attribute exists and is None (tokens go here)
    def restructure(self, ctx: SupervisorContext) -> List[Move]: ...   # [] = no idea; ordered, best first
```

A plain callable `f(ctx) -> List[Move]` is accepted too (used by the tests).

Loop behavior (`supervisor=None` keeps the M5 loop unchanged, bit for bit):
- **When.** With a supervisor, the deterministic restructure is replaced by a call to the supervisor. It is called:
  - on "stall": mid-round, once `stall_k` proposals have passed without an accept while the gap exceeds
    `stall_gap`; the stall counter then resets;
  - on "exhausted": before the loop would stop for lack of moves.

  That's all; it is never called every round. `stall_k=None` means "exhausted" only. `max_escalations`
  (default 5) caps the calls; once reached, the loop behaves as if there were no supervisor.
- **Its moves.** They are tried in the order returned, before the rest of the round's queue. `Move.agent` is set
  to "supervisor" and the ownership check is skipped. The `exclude` check and the verifier still apply.
  - Illegal: no evaluation, and the supervisor is called again with `feedback`, at most `retry_budget` times per
    escalation. Retries count as the same escalation.
  - Accept rule: the same as the agents' (eps, area repair).
- **Recorded.** Each escalation is appended to `TeamResult.escalations` and `RunRecord.escalations`:
  `{"n", "trigger", "eval_n", "binding_term", "time_s", "moves": [changes...], "outcomes": [...],
  "accepted": bool, "tokens": {model: {"in", "out"}}, "calls", "wall_s"}`.
  - `tokens` is the change in `record.tokens` across the escalation's calls. Whatever model the supervisor
    bills through `record.add_tokens` is attributed to that escalation.
- **Token timeline.** `RunRecord.add_tokens(model, n_in, n_out, **extra)` also appends to `record.token_log`
  `{"model", "in", "out", "eval_n", ...extra}`. `eval_n` comes from `record.clock()`; team_search sets
  `record.clock = lambda: ev.used` when it is None. With this you can compute $ to the first 1% band.
  - Pass cache splits as extra keys, e.g. `add_tokens(m, i, o, cache_read=..., cache_write=...)`; they are kept
    per call and summed into `record.tokens[model]`.
  - A: `cost_usd` from `lib/pricing` is yours, computed from these fields. I don't touch the pricing.

### Ensemble (`harness.team.ensemble`)

```python
def ensemble(run_one: Callable[[int], Dict], seeds: List[int], *, workers=None,
             executor="thread") -> Dict
```
- `run_one(seed)` runs one independent member and returns its `RunRecord.to_json(...)` dict.
  - Use `executor="process"` only with a picklable top-level function; threads suit API-bound LLM runs.
- It returns a merged record dict in the same schema, so the metrics code reads it unchanged:
  - `method` = "ensemble[<member method>]", `seed` = the seeds;
  - `members`: the full member records;
  - `best_member`: the index with the fastest feasible final;
  - `final`: that member's final;
  - `cost`: evaluations, tokens per model, `token_log` (entries tagged with `member`) and `wall_s` (= max),
    each summed over members; `wall_s_sum` = the sum;
  - `evaluations`: the members' evaluations interleaved round-robin (lockstep members), renumbered, each tagged
    with `member`. So "eval of first 1% band" on the merged record counts total evaluations spent by all members.
    Under the same lockstep rule, `token_log` eval_n values are remapped to that merged count.
  - `cost_usd`: summed over members if every member has one (A's field).
