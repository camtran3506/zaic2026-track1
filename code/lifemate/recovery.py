"""Recovery trajectories for empty initial search calls in ToolEnv."""
from __future__ import annotations

from copy import deepcopy
import difflib
import hashlib
import json
import re

from .agent import SYSTEM_PROMPT, _train_transcript, build_step_prompt, full_json
from .catalog import allowed_tools
from .execution_policy import DEFAULT_EMPTY_READ_LIMIT, LoopMonitor
from .history import collect_ids, extract_observed_ids
from .local_helpers import execute_helper, helper_tool_schemas
from .text import normalize_transcript
from .tool_execution import attempt_env_call, check_observed_xref, precheck_call, result_observation

GUARD_RECOVERY_POLICY = "guard-xref-validate-missing-info-recovery-v2"
DEFAULT_GUARD_RECOVERY_RATIO = 0.18


SEARCH_TOOLS = frozenset({
    "contacts.search",
    "news.search",
    "maps.search_place",
    "podcast_radio.search",
    "library.search",
})
HONORIFICS = frozenset({
    "anh", "chị", "em", "cô", "chú", "bác", "sếp", "thầy", "bạn", "ông", "bà",
    "dì", "cậu", "mợ", "cháu", "con", "vợ", "chồng", "mẹ", "bố", "ba", "má",
    "xếp", "chế", "o", "ôn",
})
STOP_WORDS = frozenset({
    "rồi", "cho", "của", "nha", "nhé", "với", "và", "là", "thì", "mà", "đi",
    "giúp", "giùm", "dùm", "em", "anh", "chị", "tôi", "mình", "nào", "đó",
    "này", "ấy", "nữa", "luôn",
})
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def search_hits(result: object) -> list[str]:
    """Extract entity IDs from a search response payload."""
    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, list) and value and isinstance(value[0], dict) and "id" in value[0]:
                return [entry["id"] for entry in value if isinstance(entry.get("id"), str)]
    return []


def aligned_span(gold_text: str, asr_text: str, phrase: str) -> str | None:
    """Locate ASR words aligned with `phrase` in clean `gold_text`."""
    gold_words, asr_words, phrase_words = _words(gold_text), _words(asr_text), _words(phrase)
    if not phrase_words:
        return None
    start = next(
        (i for i in range(len(gold_words) - len(phrase_words) + 1)
         if gold_words[i:i + len(phrase_words)] == phrase_words),
        None,
    )
    if start is None:
        return None
    end = start + len(phrase_words)
    matcher = difflib.SequenceMatcher(a=gold_words, b=asr_words, autojunk=False)
    aligned = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if i2 <= start or i1 >= end:
            continue
        if tag == "equal":
            aligned.extend(asr_words[j1 + max(0, start - i1):j1 + min(i2, end) - i1])
        else:
            aligned.extend(asr_words[j1:j2])
    return " ".join(aligned) or None


def _pick_wrong_query(fn: str, query: str, request: str, source_kind: str, gold_text: str | None):
    query_words = _words(query)
    if not query_words:
        return None, None
    req_words = _words(request)
    if fn == "contacts.search":
        if (source_kind == "asr" and isinstance(gold_text, str)
                and " ".join(query_words) not in " ".join(req_words)):
            span = aligned_span(gold_text, request, query)
            if span:
                span = " ".join(span.split()[-max(1, len(query_words)):])
            if span and span != " ".join(query_words):
                return "asr_name", span
        if query_words[0] not in HONORIFICS:
            prev = next(
                (req_words[i - 1] for i in range(1, len(req_words))
                 if req_words[i] == query_words[0] and req_words[i - 1] in HONORIFICS),
                None,
            )
            if prev:
                return "honorific", f"{prev} {query}"
        return None, None
    start = next(
        (i for i in range(len(req_words) - len(query_words) + 1)
         if req_words[i:i + len(query_words)] == query_words),
        None,
    )
    if start is None:
        return None, None
    left = req_words[max(0, start - 1):start]
    right = req_words[start + len(query_words):start + len(query_words) + 1]
    for cand in ([*left, query], [query, *right]):
        if len(cand) > 1 and cand[0] not in STOP_WORDS and cand[-1] not in STOP_WORDS:
            return "long_query", " ".join(cand)
    return None, None


def build_recovery_trajectory(record, prepared_transcript, steps, catalog, env_factory):
    """Build an extra step-wise trajectory where the first search fails then recovers.

    The failed attempt carries `train=False` so cross-entropy loss masks it out while
    keeping its genuine empty ToolEnv observation in context for the recovery step.
    Returns `(recovery_steps, recovery_meta)` or `None` if ineligible/reproduction fails.
    """
    if len(steps) < 2:
        return None
    context = record.get("context", {})
    enabled = allowed_tools(catalog, context.get("installed_apps", {}))
    local_names = set(helper_tool_schemas())
    search_idx = None
    for idx, step in enumerate(steps[:-1]):
        target = json.loads(step["messages"][-1]["content"])
        if (target.get("action") == "call" and target.get("name") in SEARCH_TOOLS
                and target.get("name") in enabled
                and isinstance((target.get("arguments") or {}).get("query"), str)):
            search_idx = idx
            break
    if search_idx is None or search_idx + 1 >= len(steps):
        return None

    target_search = json.loads(steps[search_idx]["messages"][-1]["content"])
    fn = target_search["name"]
    query = target_search["arguments"]["query"]
    next_obs = json.loads(steps[search_idx + 1]["messages"][1]["content"])["previous_actions_and_results"]
    if len(next_obs) <= search_idx or next_obs[search_idx].get("name") != fn:
        return None
    target_ids = set(search_hits(json.loads(next_obs[search_idx]["result"])))
    if not target_ids:
        return None

    raw_gold = _train_transcript(record)
    gold_clean = normalize_transcript(raw_gold) if isinstance(raw_gold, str) else None
    kind, wrong = _pick_wrong_query(
        fn, query, prepared_transcript["transcript"],
        prepared_transcript.get("transcript_source", "audio_text"), gold_clean,
    )
    if not kind or not wrong:
        return None

    transcript = prepared_transcript["transcript"]
    history = prepared_transcript["history"]

    def make_step(step_num, decision, observations, *, train=True):
        _, prompt = build_step_prompt(transcript, context, history, enabled, observations,
                                      transcript_source=prepared_transcript.get('prompt_transcript_source'))
        row = {
            "sample_id": record["id"],
            "step": step_num,
            "train": train,
            "transcript_source": prepared_transcript.get("transcript_source"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False, separators=(",", ":"))},
            ],
        }
        return row

    def guarded_call(env, name, args, monitor, call_index, observed_ids, observations):
        rejection = precheck_call(name, args, catalog, enabled, monitor, transcript=transcript,
                                  context=context, history=history, observations=observations)
        if rejection:
            return {'executed': False, 'feedback': rejection}
        return attempt_env_call(env, name, args, enabled[name].get('kind', 'unknown'), monitor,
                                call_index, spec=enabled[name], observed_ids=observed_ids)

    def replay_prefix(env, monitor):
        observations = deepcopy(
            json.loads(steps[search_idx]["messages"][1]["content"])["previous_actions_and_results"]
        )
        call_index = 0
        observed_ids = extract_observed_ids(context, history)
        for step in steps[:search_idx]:
            decision = json.loads(step["messages"][-1]["content"])
            if decision.get("action") != "call":
                continue
            name, args = decision["name"], decision["arguments"]
            if name in local_names:
                continue
            prior = json.loads(step['messages'][1]['content'])['previous_actions_and_results']
            outcome = guarded_call(env, name, args, monitor, call_index, observed_ids, prior)
            if not outcome["executed"]:
                return None, 0, observed_ids
            call_index += 1
        return observations, call_index, observed_ids

    probe_env, probe_monitor = env_factory(), LoopMonitor(empty_read_limit=DEFAULT_EMPTY_READ_LIMIT)
    probe_obs, call_index, probe_ids = replay_prefix(probe_env, probe_monitor)
    if probe_obs is None or precheck_call(fn, {"query": wrong}, catalog, enabled, probe_monitor) is not None:
        return None
    wrong_outcome = guarded_call(probe_env, fn, {'query': wrong}, probe_monitor, call_index, probe_ids, probe_obs)
    if not wrong_outcome["executed"] or search_hits(wrong_outcome["result"]):
        return None

    if kind == "asr_name":
        last_word = wrong.split()[-1]
        test_env, test_monitor = env_factory(), LoopMonitor(empty_read_limit=DEFAULT_EMPTY_READ_LIMIT)
        test_obs, test_idx, test_ids = replay_prefix(test_env, test_monitor)
        if test_obs is None:
            return None
        guarded_call(test_env, fn, {'query': wrong}, test_monitor, test_idx, test_ids, test_obs)
        last_outcome = guarded_call(test_env, fn, {'query': last_word}, test_monitor, test_idx + 1, test_ids, test_obs)
        recover_args = {"query": last_word} if (last_outcome["executed"] and (target_ids & set(search_hits(last_outcome["result"])))) else {}
    else:
        recover_args = {"query": query}

    env, monitor = env_factory(), LoopMonitor(empty_read_limit=DEFAULT_EMPTY_READ_LIMIT)
    observations, call_index, observed_ids = replay_prefix(env, monitor)
    if observations is None:
        return None
    new_steps = [deepcopy(row) for row in steps[:search_idx]]

    wrong_decision = {"action": "call", "name": fn, "arguments": {"query": wrong}}
    new_steps.append(make_step(len(new_steps), wrong_decision, observations, train=False))
    if precheck_call(fn, {"query": wrong}, catalog, enabled, monitor) is not None:
        return None
    wrong_outcome = guarded_call(env, fn, {'query': wrong}, monitor, call_index, observed_ids, observations)
    if not wrong_outcome["executed"] or search_hits(wrong_outcome["result"]):
        return None
    call_index += 1
    observations.append(result_observation(fn, {"query": wrong}, wrong_outcome, full_json))

    recover_decision = {"action": "call", "name": fn, "arguments": recover_args}
    new_steps.append(make_step(len(new_steps), recover_decision, observations, train=True))
    if precheck_call(fn, recover_args, catalog, enabled, monitor) is not None:
        return None
    recover_outcome = guarded_call(env, fn, recover_args, monitor, call_index, observed_ids, observations)
    if not recover_outcome["executed"] or not (target_ids & set(search_hits(recover_outcome["result"]))):
        return None
    call_index += 1
    observations.append(result_observation(fn, recover_args, recover_outcome, full_json))

    for offset, orig_step in enumerate(steps[search_idx + 1:], start=search_idx + 1):
        decision = json.loads(orig_step["messages"][-1]["content"])
        new_steps.append(make_step(len(new_steps), decision, observations, train=True))
        if decision.get("action") != "call":
            continue
        name, args = decision["name"], decision["arguments"]
        if name in local_names:
            local_res = execute_helper(name, args, context, history, enabled, observations)
            observations.append({"name": name, "arguments": args, "scope": "local", "result": full_json(local_res)})
            continue
        if offset + 1 >= len(steps):
            return None
        orig_after_obs = json.loads(steps[offset + 1]["messages"][1]["content"])["previous_actions_and_results"]
        expected_result = orig_after_obs[offset]["result"]
        if precheck_call(name, args, catalog, enabled, monitor) is not None:
            return None
        outcome = guarded_call(env, name, args, monitor, call_index, observed_ids, observations)
        if not outcome["executed"] or full_json(outcome["result"]) != expected_result:
            return None
        call_index += 1
        observations.append(result_observation(name, args, outcome, full_json))

    meta = {"kind": kind, "wrong": wrong, "recover": recover_args, "gold_query": query}
    for row in new_steps:
        row["recovery"] = deepcopy(meta)
    return new_steps, meta


def sample_selected_for_guard_recovery(
    sample_id: str,
    ratio: float = DEFAULT_GUARD_RECOVERY_RATIO,
    seed: int = 42,
) -> bool:
    """Deterministically select a fraction of sample IDs using SHA-256."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("guard_recovery_ratio must be between 0 and 1")
    if ratio == 0.0:
        return False
    if ratio == 1.0:
        return True
    digest = hashlib.sha256(f"{seed}:{sample_id}:guard_recovery".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0x100000000) < ratio


def _unobserved_xref_keys(args: dict, spec: dict, observed_ids: dict[str, set[str]]) -> list[tuple[str, str]]:
    props = (spec.get("parameters") or {}).get("properties") or {}
    unobserved: list[tuple[str, str]] = []
    for key, val in (args or {}).items():
        prop = props.get(key)
        if not isinstance(prop, dict) or not prop.get("x-ref"):
            continue
        ref_app = str(prop["x-ref"]).split(".", 1)[0]
        if not isinstance(val, str) or val not in observed_ids.get(ref_app, set()):
            unobserved.append((key, ref_app))
    return unobserved


def _xref_placeholder(read_args: dict, ref_app: str, observed_ids: dict[str, set[str]]) -> str:
    seen = observed_ids.get(ref_app, set())
    if isinstance(read_args, dict):
        q = read_args.get("query")
        if isinstance(q, str) and q.strip() and q.strip() not in seen:
            return q.strip()
        for val in read_args.values():
            if isinstance(val, str) and val.strip() and val.strip() not in seen:
                return val.strip()
    placeholder = "chua_xac_dinh"
    while placeholder in seen:
        placeholder += "_x"
    return placeholder


class _ValidateOnlyEnv:
    """Read-only wrapper that forbids `env.call` during guard probing."""

    def __init__(self, env):
        self._env = env

    def validate(self, name, arguments):
        return self._env.validate(name, arguments)

    def call(self, name, arguments):
        raise AssertionError(f"Guard probe must never execute env.call({name!r})")


def _probe_guard_rejection(env, name: str, bad_args: dict, catalog, enabled, spec: dict, observed_ids: dict[str, set[str]], call_index: int):
    if precheck_call(name, bad_args, catalog, enabled, LoopMonitor(empty_read_limit=DEFAULT_EMPTY_READ_LIMIT)) is not None:
        return None
    validation = env.validate(name, bad_args)
    accepted = isinstance(validation, dict) and validation.get("ok") is True
    if accepted and check_observed_xref(bad_args, spec, observed_ids) is None:
        return None
    outcome = attempt_env_call(
        _ValidateOnlyEnv(env),
        name,
        bad_args,
        spec.get("kind", "unknown"),
        LoopMonitor(empty_read_limit=DEFAULT_EMPTY_READ_LIMIT),
        call_index,
        spec=spec,
        observed_ids=observed_ids,
    )
    if outcome.get("executed"):
        return None
    return outcome.get("feedback")


def _parse_json_obj(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def build_guard_recovery_examples(
    record: dict,
    prepared_transcript: dict,
    steps: list[dict],
    catalog: dict[str, dict],
    env_or_factory,
    *,
    ratio: float = DEFAULT_GUARD_RECOVERY_RATIO,
    seed: int = 42,
    max_examples_per_trajectory: int = 1,
) -> list[dict]:
    """Build standalone 1-step `full_trajectory=False` recovery SFT rows from real guard rejections.

    Uses already-replayed step observations + read-only `env.validate` probing so no extra
    `ToolEnv` instances or `env.call` side effects occur.

    Covers genuine failure modes:
    1. `xref_guard`: LLM proposes a `write` call with an unobserved/raw `x-ref` argument before
       calling the prerequisite `read` tool -> teaches reading `agent_feedback` and calling the read tool.
    2. `validate`: LLM proposes an env call missing a required parameter or using an invalid enum
       -> teaches reading `agent_feedback` and immediately correcting the call.
    3. `missing_info_guard`: LLM asks for an already-resolved field -> teaches continuing the gold action.
    """
    if max_examples_per_trajectory < 1 or not steps:
        return []
    if not sample_selected_for_guard_recovery(str(record["id"]), ratio, seed):
        return []

    env = env_or_factory() if callable(env_or_factory) else env_or_factory
    context = record.get("context", {})
    enabled = allowed_tools(catalog, context.get("installed_apps", {}))
    local_names = set(helper_tool_schemas())
    transcript = prepared_transcript["transcript"]
    history = prepared_transcript["history"]
    observed_ids = extract_observed_ids(context, history)

    call_index = 0
    results: list[dict] = []

    for idx, step in enumerate(steps):
        decision = json.loads(step["messages"][-1]["content"])
        if decision.get("action") != "call":
            continue
        name = decision.get("name")
        args = decision.get("arguments") or {}
        if name in local_names or name not in enabled:
            continue

        spec = enabled[name]
        kind = spec.get("kind", "unknown")
        obs_at_k = json.loads(step["messages"][1]["content"])["previous_actions_and_results"]
        next_obs_result = None
        if idx + 1 < len(steps):
            next_obs = json.loads(steps[idx + 1]["messages"][1]["content"])["previous_actions_and_results"]
            if len(next_obs) > idx:
                next_obs_result = _parse_json_obj(next_obs[idx].get("result"))

        if len(results) < max_examples_per_trajectory and kind == "read":
            read_adds_ids: dict[str, set[str]] = deepcopy(observed_ids)
            read_app = name.split(".", 1)[0]
            if isinstance(next_obs_result, dict) and next_obs_result.get("ok") is True and next_obs_result.get("done") is not False:
                collect_ids(next_obs_result, read_adds_ids, read_app)

            for future in steps[idx + 1:]:
                f_dec = json.loads(future["messages"][-1]["content"])
                if f_dec.get("action") != "call":
                    continue
                f_name = f_dec.get("name")
                f_args = f_dec.get("arguments") or {}
                if f_name not in enabled or enabled[f_name].get("kind") != "write":
                    continue
                missing_xrefs = _unobserved_xref_keys(f_args, enabled[f_name], observed_ids)
                if not missing_xrefs:
                    continue
                relevant = any(
                    (f_args.get(rk) in read_adds_ids.get(ra, set())) or ra == read_app
                    for rk, ra in missing_xrefs
                )
                if not relevant:
                    continue
                bad_args = deepcopy(f_args)
                for ref_key, ref_app in missing_xrefs:
                    bad_args[ref_key] = _xref_placeholder(args, ref_app, observed_ids)
                feedback = _probe_guard_rejection(
                    env, f_name, bad_args, catalog, enabled, enabled[f_name], observed_ids, call_index
                )
                if feedback and feedback.get("stage") in ("xref_guard", "validate"):
                    obs_with_error = deepcopy(obs_at_k) + [feedback]
                    _, prompt = build_step_prompt(transcript, context, history, enabled, obs_with_error,
                                                  transcript_source=prepared_transcript.get('prompt_transcript_source'))
                    results.append({
                        "sample_id": record["id"],
                        "step": 0,
                        "full_trajectory": False,
                        "train": True,
                        "transcript_source": prepared_transcript.get("transcript_source"),
                        "recovery": {
                            "kind": "guard_xref",
                            "stage": feedback["stage"],
                            "failed_tool": f_name,
                            "target_tool": name,
                        },
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False, separators=(",", ":"))},
                        ],
                    })
                    break

        if len(results) < max_examples_per_trajectory and kind == 'write':
            props = (spec.get('parameters') or {}).get('properties') or {}
            for parameter in args:
                if parameter not in props:
                    continue
                proposed = {'tool_name': name, 'params': [parameter]}
                feedback = precheck_call('sys.ask_missing_info', proposed, catalog, enabled,
                                        LoopMonitor(empty_read_limit=DEFAULT_EMPTY_READ_LIMIT),
                                        transcript=transcript, context=context, history=history,
                                        observations=obs_at_k)
                if not feedback or feedback.get('feedback', {}).get('reason') != 'information_already_available':
                    continue
                _, prompt = build_step_prompt(transcript, context, history, enabled, deepcopy(obs_at_k) + [feedback],
                                              transcript_source=prepared_transcript.get('prompt_transcript_source'))
                results.append({
                    'sample_id': record['id'], 'step': 0, 'full_trajectory': False, 'train': True,
                    'transcript_source': prepared_transcript.get('transcript_source'),
                    'recovery': {'kind': 'guard_missing_info', 'stage': 'missing_info_guard',
                                 'failed_tool': 'sys.ask_missing_info', 'target_tool': name},
                    'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                                 {'role': 'user', 'content': prompt},
                                 {'role': 'assistant', 'content': json.dumps(decision, ensure_ascii=False, separators=(',', ':'))}],
                })
                break

        if len(results) < max_examples_per_trajectory and kind == "write" and name != "sys.ask_missing_info":
            params_schema = spec.get("parameters") or {}
            props = params_schema.get("properties") or {}
            required = list(params_schema.get("required") or [])
            bad_args = None
            mutation_kind = None
            non_xref_req = [
                r for r in required
                if r in args and not (isinstance(props.get(r), dict) and props[r].get("x-ref"))
            ]
            if non_xref_req:
                bad_args = deepcopy(args)
                bad_args.pop(non_xref_req[-1])
                mutation_kind = "missing_required"
            else:
                enum_keys = [
                    k for k, p in props.items()
                    if k in args and isinstance(p, dict) and isinstance(p.get("enum"), list) and p["enum"]
                ]
                if enum_keys:
                    bad_args = deepcopy(args)
                    bad_args[enum_keys[0]] = "__invalid_enum_value__"
                    mutation_kind = "invalid_enum"
            if bad_args is not None:
                feedback = _probe_guard_rejection(
                    env, name, bad_args, catalog, enabled, spec, observed_ids, call_index
                )
                if feedback and feedback.get("stage") == "validate":
                    obs_with_error = deepcopy(obs_at_k) + [feedback]
                    _, prompt = build_step_prompt(transcript, context, history, enabled, obs_with_error,
                                                  transcript_source=prepared_transcript.get('prompt_transcript_source'))
                    results.append({
                        "sample_id": record["id"],
                        "step": 0,
                        "full_trajectory": False,
                        "train": True,
                        "transcript_source": prepared_transcript.get("transcript_source"),
                        "recovery": {
                            "kind": "guard_validate",
                            "mutation": mutation_kind,
                            "stage": "validate",
                            "failed_tool": name,
                            "target_tool": name,
                        },
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": json.dumps(decision, ensure_ascii=False, separators=(",", ":"))},
                        ],
                    })

        # Update observed_ids cleanly using the already-replayed step output.
        if isinstance(next_obs_result, dict) and next_obs_result.get("ok") is True and next_obs_result.get("done") is not False:
            app = name.split(".", 1)[0]
            collect_ids(next_obs_result, observed_ids, app)
            if kind == "write":
                collect_ids(args, observed_ids, app)
        call_index += 1

    return results
