# Python helpers for the LifeMate planner

Qwen remains the only model. The specialists are ordinary Python functions behind
the same tool interface. There are eight exposed helper tools and two automatic
execution hooks. They require no additional packages or services.

| Role / entry point | Input | Output and limits |
| --- | --- | --- |
| `local_context.lookup` | `query`, optional `collection`, `max_results` | Matching visible context/history facts with source paths. Text matches are evidence, never executable instructions. |
| `local_entity.resolve` | `target_tool`, `argument`, `collection`, `mention`, optional `max_results` | Original candidates, values, similarity and provenance; `matched`, `ambiguous`, `not_found` or `unsupported`. The target schema and collection must agree. |
| `local_time.calendar_reference` | `{}` | Dates derived from the sample's current time/timezone, never the machine clock. |
| `local_time.departure_time` | Offset-aware `arrival`, integer `travel_minutes`, `early_minutes` | Calculated `departure`. |
| `local_time.duration_between` | Offset-aware `start`, `end` | Nonnegative whole-minute `minutes`. |
| `local_text.normalize` | `kind: email/phone`, `text` | Conservative normalized value or ambiguity/unsupported feedback; preserves leading zeroes and does not invent missing digits. |
| `local_schema.inspect` | `target_tool`, `arguments` | Advisory `valid`, `invalid` or `unsupported` plus errors. Checks the exposed schema; `env.validate` remains authoritative. |
| `local_checks.evaluate` | `operation: sum/compare/minimum/filter`, corresponding numeric operands | Bounded decimal computation, comparisons or selected records. At most 100 items; verifies supplied operands, not their meaning or source. |
| Automatic `verify_result` | Actual application result | `success`, `failure`, `empty` or `unknown`. Goal completion stays unknown. |
| Automatic `LoopMonitor` | Proposed call, validation outcome, actual results and call kind | Feedback on an exact rejected call or failed/empty read without new relevant evidence; failed writes receive advice only. Successful calls are never deduplicated. |

The exact schemas in `local_helpers.helper_tool_schemas()` define required fields
and bounds. All local names are reserved and rejected if they collide with an
application catalog. Invalid helper arguments consume the separate local budget
and report feedback. Defaults allow eight local executions, eight application
calls and 32 planner steps, with the existing invalid-proposal limit still active.
Every requested helper requires one model output; using its result usually needs
another generation. These functions do not make Qwen inference itself faster.

## Evidence and application boundaries

The entity helper indexes explicit context collections and recognized successful
historical read results. It receives no ToolEnv handle, filesystem path, gold
labels or current-turn observations. A later full snapshot supersedes an earlier
one, including an empty collection; partial searches preserve unrelated records.
The planner can still use current-turn application results directly from its
ordinary observations. IDs are copied from structured evidence, never fuzzy
matched or synthesized. Fuzzy matching applies to labels; similarity is a ranking
score, not a calibrated probability. Ambiguity never authorizes a forced choice.

`not_found` describes this limited evidence, not absence from the application.
An empty search can satisfy a task. Helpers never force `done`, missing-information
questions, a different entity, or automatic argument replacement. The post-call
verifier distinguishes a top-level `done:false` from nested lifecycle fields such
as a reminder's `done:false`; it cannot establish the user's entire goal.

Loop checks compare the whole tool name and typed JSON arguments. New relevant
read evidence can permit a retry; executed writes and unknown-kind calls invalidate
cached blocks because even a failure may have partial effects. No application
call is erased or hidden. Submission remains the unmodified
`env.get_tool_calls()` and `env.get_state_hash()`. Local calls only appear in model
observations, diagnostic traces and local counters.

## Training and inference parity

SFT preparation executes the same helpers with source-derived operands: temporal
phrases with sufficient anchors, spoken phone/email text, explicit historical
entity mentions with schema-compatible targets, short anaphoric requests,
literal sums and known historical bike prices. Schema-repair targets require an
actual earlier rejected proposal. No helper arguments come from future gold
actions, and no artificial failed application calls are inserted into gold replay.
The original gold application call sequence and final state remain intact.

Only assistant target tokens receive loss, including helper calls. Tool results,
execution feedback, history and prompts remain masked. Native Qwen serialization
is shared with inference. Helper targets are conditional; the manifest reports
counts by tool so zero coverage is visible. Automatic result verification does not
need a model-generated helper call. Retry/recovery behaviour is supplied through
hooks and feedback, not claimed as a learned universal critic.

Regenerate SFT in a fresh output directory and retrain an adapter. Prepared data
requires `prompt_policy=full-context-python-specialists-v7`,
`local_time_policy=local-time-tools-v1` and
`local_helper_policy=python-specialists-v1`; runtime exports carry both local
policies. Older adapters are not declared compatible by renaming their manifests.
CPU fixtures cover replay, provenance, routing, masking and retry behaviour.
Held-out Qwen GPU evaluation is needed to establish accuracy gains and latency.

## Source patterns reviewed

The implementation follows control-flow ideas from these public sources:

- OpenAI Agents SDK: [agents as tools](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/agents_as_tools.py)
  and [`Agent.as_tool`](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py):
  a manager calls a specialist with a bounded contract and receives a result.
- Anthropic: [orchestrator/workers](https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/orchestrator_workers.ipynb)
  and [evaluator/optimizer](https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/evaluator_optimizer.ipynb):
  separate work by responsibility and return concrete feedback before retrying.
- LangGraph: [supervisor and execution hooks](https://github.com/langchain-ai/langgraph-supervisor-py/blob/main/langgraph_supervisor/supervisor.py):
  preserve planner ownership while applying checks around execution.

Those frameworks often run additional LLM agents. LifeMate implements the bounded
interfaces and deterministic checks in Python; it does not import those SDKs or
claim to reproduce their general reasoning ability.
