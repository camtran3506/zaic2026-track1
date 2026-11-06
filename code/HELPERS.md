# Python helpers for the LifeMate planner

Qwen remains the only model. The specialists are ordinary Python functions behind
the same tool interface. There are eight primitive helpers, one task solver and two automatic
execution hooks. They require no additional packages or services.

| Role / entry point | Input | Output and limits |
| --- | --- | --- |
| `local_assist.solve` | `task`, `inputs` containing source spans/known operands | Fixed date, datetime, departure, duration or multi-field normalization workflow. One local request; Python manages intermediate results. |
| `local_context.lookup` | `query`, optional `collection`, `max_results` | Matching visible context/history facts with source paths. Text matches are evidence, never executable instructions. |
| `local_entity.resolve` | `target_tool`, `argument`, `collection`, `mention`, optional `max_results` | Original candidates, values, similarity and provenance; `matched`, `ambiguous`, `not_found` or `unsupported`. The target schema and collection must agree. |
| `local_time.calendar_reference` | `{}` | Dates derived from the sample's current time/timezone, never the machine clock. |
| `local_time.departure_time` | Offset-aware `arrival`, integer `travel_minutes`, `early_minutes` | Calculated `departure`. |
| `local_time.duration_between` | Offset-aware `start`, `end` | Nonnegative whole-minute `minutes`. |
| `local_text.normalize` | `kind: email/phone/money`, `text` | Conservative normalized value or ambiguity/unsupported feedback; preserves leading zeroes and does not invent missing digits. |
| `local_schema.inspect` | `target_tool`, `arguments` | Advisory `valid`, `invalid` or `unsupported` plus errors. Checks the exposed schema; `env.validate` remains authoritative. |
| `local_checks.evaluate` | `operation: sum/compare/minimum/filter`, corresponding numeric operands | Bounded decimal computation, comparisons or selected records. At most 100 items; verifies supplied operands, not their meaning or source. |
| Automatic `verify_result` | Actual application result | `success`, `failure`, `empty` or `unknown`. Goal completion stays unknown. |
| Automatic `LoopMonitor` | Proposed call, validation outcome, actual results and call kind | Permanent schema blacklist, current-session read reuse and no-op write suppression. Successful writes remain repeatable. This middleware is automatic, never a model-selected tool. |

The exact schemas in `local_helpers.helper_tool_schemas()` define required fields
and bounds. All local names are reserved and rejected if they collide with an
application catalog. Invalid helper arguments report feedback and consume the
existing invalid-proposal budget. There is no helper-count ceiling per batch or
sample. `local_batch.run` can group independent helpers according to actual output
token length; unknown/future operands require another turn. SFT uses the same
token/dependency checks with the configured tokenizer, and has no fixed demo-count
truncation. Without a tokenizer it leaves helper demos single.

Env read batches still have at most four members; eight application calls/sample
remain the default. The 32-step planning limit counts env/invalid decisions, not
valid helper-only turns. Diagnostics distinguish total `llm_steps` from
`non_helper_steps`. Token/context and configured time budgets still apply, as do
the bounds of each helper's input schema (such as at most 100 numeric operands).
Gaining fewer LLM turns is the purpose of batching; CPU helper execution remains
sequential and never creates additional LLMs or hidden application calls.

## Python task orchestration

For supported workflows the planner requests an outcome rather than proposing an
ordered list of helper steps. `local_assist.solve` has five task contracts:

| Task | Inputs | Resolved result |
| --- | --- | --- |
| `date` | `date_text` | `date` |
| `datetime` | `date_text`, `time_text`; alternatively offset-aware `timestamp` | `datetime` |
| `departure_time` | `date_text`, `arrival_time`; alternatively offset-aware `arrival`; plus explicit `travel_minutes`, `early_minutes` | `departure` |
| `duration` | `date_text`, `start_time`, `end_time`, optional `end_date_text`; alternatively offset-aware `start`, `end` | Whole elapsed `minutes` |
| `normalize_fields` | `fields: [{id, kind, text}, ...]`; kind `phone`, `email` or `money` | Per-ID normalization results |

For example, `{"task":"departure_time","inputs":{"date_text":"sáng mai",
"arrival_time":"chín giờ","travel_minutes":15,"early_minutes":15}}` resolves
the sample's tomorrow, combines its morning hour and subtracts the known travel
and early-arrival intervals. Python owns these dependencies. The response includes
`status`, `result`, source `evidence` and the workflow policy; it is not a request to
execute an application tool. The LLM still chooses the appropriate env action.

The finite Vietnamese grammar supports explicit-year ISO/dd/mm/yyyy dates,
relative days and weekdays with an explicit week. It accepts HH:MM[:SS] or supported
Vietnamese clock spans with a daypart. It never uses the machine clock, guesses a
year/week/travel interval, invents entity IDs or infers an overnight rollover.
Timezone and relative-date anchors come from sample context. DST gaps are rejected;
folds need explicit offset-aware ISO timestamps. Elapsed arithmetic remains UTC-based.
Missing, ambiguous, unsupported or partially normalized operands are returned as
such, with no fabricated final value. They interrupt a helper batch so the planner
can inspect the response. Duplicate field IDs are rejected before normalization.

There is no field-count cap: schema text bounds and the actual model token budget
apply. During SFT, multi-field requests are packed using the selected tokenizer and
the default generation budget; an oversized one-field envelope can fall back to
the primitive normalizer. Without a tokenizer, normalization demos remain primitive
calls. Clear automatic ITN annotations are reused rather than demonstrated again.
Source-derived date/interval demos prefer the solver; unsupported workflows retain
the existing helpers. Future gold arguments never supply solver operands.

One facade call consumes one `local_executed` request and one model turn regardless
of the number of internal operations. `assist_step_start/result/error` events go
only to the debug trace. Internal calendar tables are not repeated in the model's
observation. No ToolEnv handle, filesystem, network, arbitrary Python or extra LLM
is available to the solver. Entity lookup remains with the existing evidence-only
helpers; there is no general-purpose dependency interpreter. Primitive helpers and
independent `local_batch.run` calls remain supported.

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

Loop checks compare the full tool name and typed JSON arguments, ignoring dictionary
key order. They have separate lifetimes:

| Automatic check | Lifetime / action |
| --- | --- |
| `check_tool_access` | Before any env validation/call; distinguish unknown vs disallowed names using the catalog and sample whitelist. Suggestions only contain exposed tools. Registered local helpers route separately. |
| Explicit `env.validate(...).ok == false` | Permanently block that exact proposal until the sample ends, for read/write/meta alike. No read, helper or write clears this blacklist. |
| Actual read result with `ok:true` | Cache and block exact retries in the current state epoch, including nonempty, empty, nested-empty and top-level `done:false` results. `done:true` is not required. |
| Actual write result with top-level `ok:true, done:false` | Block an exact retry in the current state epoch. New read evidence does not change this condition. |
| Successful write | Never deduplicate it. Conservatively invalidate all read/no-op caches across apps, even if no mutation can be proven. Nested lifecycle fields such as a reminder's `done:false` do not mean failure. |
| Execution exception or malformed write response | The state may be uncertain, so invalidate transient caches. Keep schema rejections. |
| Technical error | At most two failed attempts per exact proposal (original plus one retry), then return a distinct retry-budget guard; do not label it a schema rejection. |

Only explicit organizer `ok:false` validation results enter the schema blacklist;
malformed validator returns and exceptions are technical errors. An actual
`env.call` returning `ok:false` after successful validation is recorded as a
contract anomaly and schema rejection, while retaining the executed call.
Exceptions use an agent-generated `execution_exception` envelope, not a fabricated
organizer `ok:false` response. Only top-level `done:false` is a business no-op.

Every env call uses the shared `tool_execution.attempt_env_call` boundary, which
requires explicit validation success before execution. A blocked proposal returns
`agent_feedback`, including the original cached result and source call index for
read reuse. It is never rendered as another executed native tool call/response.
`automatic_execution_feedback` accompanies actual observations in both SFT and
inference, identifying Python-managed rules, result origin and state epoch.
The model does not request these guards; they execute automatically.
`local_schema.inspect` remains a separate optional, advisory model-callable helper.

History is not seeded into the read cache, so it cannot suppress the first read
in this ToolEnv. New reads/helpers, known no-op writes and schema refusals do not
advance the state epoch. No `get_state()`, state hash or environment files are used
to detect state changes. A canonical state hash could omit relevant data.

No application call is erased or hidden. Submission remains the unmodified
`env.get_tool_calls()` and `env.get_state_hash()`. Blocked proposals consume the
existing invalid-proposal/step budget; empty/no-op feedback never forces done,
a different entity or a missing-information question. Schema rejections are never
evicted; the outer step limit bounds their per-sample size. Transient caches are
bounded separately. All guard state is reset for each sample.

## Automatic ITN before the model

After the shared transcript normalization, `build_step_prompt` adds
`itn_candidates` for the current request and `itn_history_candidates` with history
indices for text turns. It preserves the transcript and structured tool evidence.
Each candidate has `kind`, `value`, exact `source_text` and start/end Unicode
code-point offsets into that normalized text (end exclusive). Money includes
`currency: VND`. No automatic candidate is an executed helper/application call.
The original pre-normalization ASR text is not newly stored by this feature.

Detection uses explicit phone markers (`số điện thoại`, `số máy`, `gọi số`,
`nhắn tin đến số`), literal or supported spoken email syntax, and Vietnamese
amounts ending in `đồng`, `VND` or `₫`. Examples: spoken digits become
`"0912345678"`; `hai triệu rưỡi đồng` becomes integer `2500000` VND;
`1,5 triệu đồng` becomes `1500000` VND. Vietnamese dotted thousand groups are
supported. Ambiguous comma grouping, fractional VND, negative amounts, unsupported
currencies and unclear shorthand remain untouched. This is a limited rule grammar,
not a complete Vietnamese ITN engine. The planner still interprets negation,
selects recipients and decides whether a mentioned amount should be used.

The same `normalize_span` implementation serves automatic detection and
`local_text.normalize`. The model may call it for an unmarked span, including
`kind: money`; omitted currency returns `currency: null`, never an invented unit.
Unclear/unsupported input can remain unresolved after a manual call as well.
A bounded LRU caches detection across prompt steps; returned annotations are copied.
No extra LLM or runtime dependency is added.

## Training and inference parity

SFT preparation executes the same helpers with source-derived operands: temporal
phrases with sufficient anchors, spoken phone/email text, explicit historical
entity mentions with schema-compatible targets, short anaphoric requests,
literal sums and known historical bike prices. Already-annotated normalization
spans do not produce redundant helper-call targets; fallback phone/money targets
remain source-derived when automatic detection lacks a clear marker. Schema-repair targets require an
actual earlier rejected proposal. No helper arguments come from future gold
actions, and no artificial failed application calls are inserted into gold replay.
The original gold application call sequence and final state remain intact.

SFT replay uses exactly the same access checks, validation/execution boundary,
monitor and automatic feedback as inference. Gold is not silently rewritten:
a blocked duplicate read/no-op gold proposal fails preparation with the sample
and step. Successful duplicate writes remain intact. There are no artificial
failed env calls, copied future-gold operands or new model-callable guard schemas.
The next gold action learns from the program feedback accompanying prior actual
results; no extra LLM or hidden-state read is involved.

Only assistant target tokens receive loss, including helper calls. Tool results,
execution feedback, history and prompts remain masked. Native Qwen serialization
is shared with inference. Helper targets are conditional; the manifest reports
counts by tool so zero coverage is visible. Automatic result verification does not
need a model-generated helper call. Retry/recovery behaviour is supplied through
hooks and feedback, not claimed as a learned universal critic.

Regenerate SFT in a fresh output directory and retrain an adapter. Prepared data
requires `prompt_policy=compact-schemas-shared-prefix-v13`,
`local_time_policy=local-time-tools-v1` and
`local_helper_policy=python-specialists-itn-v2` and
`itn_policy=vi-explicit-spans-v1` and
`execution_policy=automatic-tool-guards-v1`; new runtime exports also pin
`assist_policy=fixed-python-task-workflows-v1` and
`schema_prompt_policy=redundant-schema-metadata-v1`. Existing primitive-only runtime
manifests remain loadable; retraining is needed to teach the new task interface.
ASR transcript normalization/cache format is unchanged; valid transcript caches
can be reused because annotations are derived when building the prompt. Older adapters are not declared compatible by renaming their manifests.
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
