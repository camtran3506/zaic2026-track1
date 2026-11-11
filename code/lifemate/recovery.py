"""Recovery trajectories for empty initial search calls in ToolEnv."""
from __future__ import annotations

from copy import deepcopy
import difflib
import json
import re

from .agent import SYSTEM_PROMPT, _train_transcript, build_step_prompt, full_json
from .catalog import allowed_tools
from .execution_policy import LoopMonitor
from .local_helpers import execute_helper, helper_tool_schemas
from .text import normalize_transcript
from .tool_execution import attempt_env_call, precheck_call, result_observation


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
        _, prompt = build_step_prompt(transcript, context, history, enabled, observations)
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

    def replay_prefix(env, monitor):
        observations = deepcopy(
            json.loads(steps[search_idx]["messages"][1]["content"])["previous_actions_and_results"]
        )
        call_index = 0
        for step in steps[:search_idx]:
            decision = json.loads(step["messages"][-1]["content"])
            if decision.get("action") != "call":
                continue
            name, args = decision["name"], decision["arguments"]
            if name in local_names:
                continue
            outcome = attempt_env_call(env, name, args, enabled[name].get("kind", "unknown"), monitor, call_index)
            if not outcome["executed"]:
                return None, 0
            call_index += 1
        return observations, call_index

    probe_env, probe_monitor = env_factory(), LoopMonitor()
    probe_obs, call_index = replay_prefix(probe_env, probe_monitor)
    if probe_obs is None or precheck_call(fn, {"query": wrong}, catalog, enabled, probe_monitor) is not None:
        return None
    wrong_outcome = attempt_env_call(probe_env, fn, {"query": wrong}, enabled[fn].get("kind", "unknown"), probe_monitor, call_index)
    if not wrong_outcome["executed"] or search_hits(wrong_outcome["result"]):
        return None

    if kind == "asr_name":
        last_word = wrong.split()[-1]
        test_env, test_monitor = env_factory(), LoopMonitor()
        test_obs, test_idx = replay_prefix(test_env, test_monitor)
        if test_obs is None:
            return None
        attempt_env_call(test_env, fn, {"query": wrong}, enabled[fn].get("kind", "unknown"), test_monitor, test_idx)
        last_outcome = attempt_env_call(test_env, fn, {"query": last_word}, enabled[fn].get("kind", "unknown"), test_monitor, test_idx + 1)
        recover_args = {"query": last_word} if (last_outcome["executed"] and (target_ids & set(search_hits(last_outcome["result"])))) else {}
    else:
        recover_args = {"query": query}

    env, monitor = env_factory(), LoopMonitor()
    observations, call_index = replay_prefix(env, monitor)
    if observations is None:
        return None
    new_steps = [deepcopy(row) for row in steps[:search_idx]]

    wrong_decision = {"action": "call", "name": fn, "arguments": {"query": wrong}}
    new_steps.append(make_step(len(new_steps), wrong_decision, observations, train=False))
    if precheck_call(fn, {"query": wrong}, catalog, enabled, monitor) is not None:
        return None
    wrong_outcome = attempt_env_call(env, fn, {"query": wrong}, enabled[fn].get("kind", "unknown"), monitor, call_index)
    if not wrong_outcome["executed"] or search_hits(wrong_outcome["result"]):
        return None
    call_index += 1
    observations.append(result_observation(fn, {"query": wrong}, wrong_outcome, full_json))

    recover_decision = {"action": "call", "name": fn, "arguments": recover_args}
    new_steps.append(make_step(len(new_steps), recover_decision, observations, train=True))
    if precheck_call(fn, recover_args, catalog, enabled, monitor) is not None:
        return None
    recover_outcome = attempt_env_call(env, fn, recover_args, enabled[fn].get("kind", "unknown"), monitor, call_index)
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
        outcome = attempt_env_call(env, name, args, enabled[name].get("kind", "unknown"), monitor, call_index)
        if not outcome["executed"] or full_json(outcome["result"]) != expected_result:
            return None
        call_index += 1
        observations.append(result_observation(name, args, outcome, full_json))

    meta = {"kind": kind, "wrong": wrong, "recover": recover_args, "gold_query": query}
    for row in new_steps:
        row["recovery"] = deepcopy(meta)
    return new_steps, meta
