"""Train-only gold replay, sample-level splitting, and assistant-only SFT loss."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
import re
import time
import unicodedata

from .agent import (SYSTEM_PROMPT, AgentConfig, _train_transcript, audio_path, build_step_prompt,
                    full_json, history_text)
from .catalog import allowed_tools
from .function_calling import native_chat, native_observation_messages
from .history import extract_observed_ids, history_entries
from .llm import chat_token_ids
from .local_time import LOCAL_TIME_POLICY
from .local_helpers import LOCAL_HELPER_POLICY, execute_helper
from .execution_policy import EXECUTION_POLICY, LoopMonitor
from .tool_execution import (precheck_call, attempt_env_call, check_observed_xref,
                             result_observation, update_observed_ids)
from .specialist_training import specialist_demonstrations
from .temporal_training import helper_demonstrations
from .assist_training import task_demonstrations
from .local_assist import ASSIST_TOOL, ASSIST_POLICY
from .context import DEFAULT_MAX_NEW_TOKENS
from .reasoning import REASONING_POLICY
from .recovery import GUARD_RECOVERY_POLICY
from .text import TEXT_NORMALIZATION, normalize_transcript
from .itn import ITN_POLICY
from .schema_prompt import SCHEMA_PROMPT_POLICY
from .numerics import DEBUG_BATCH_KEY


SFT_PROMPT_POLICY = "full-trajectory-date-apps-v17-dual-transcript-source"
VALIDATION_ROLLOUT_POLICY = "asr-only-paired-clean-v1"
DEFAULT_ASR_TRANSCRIPT_RATIO = 0.5
SYNTHETIC_SFT_POLICY = "recorded-tool-results-audio-text-only-v1"
_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "\n</think>\n\n"
_ASSISTANT_SPAN = re.compile(
    r"<\|im_start\|>assistant\n(?:<think>\n(?:\n</think>\n\n)?)?(.*?<\|im_end\|>)(?=\n|$)",
    re.S,
)
_FUNC_RE = re.compile(r"<function=([^>\n]+)>")
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n(.*?)\n</parameter>", re.S)


def asr_transcript_ratio(value):
    ratio = float(value)
    if not 0 <= ratio <= 1:
        raise ValueError("ASR transcript ratio must be between 0 and 1")
    return ratio


def is_synthetic_sample(item):
    """Use explicit provenance, never a missing WAV, to select offline replay."""
    return (item.get("synthetic") is True
            or str(item.get("id", "")).startswith("synthetic_"))


def assign_transcript_sources(records, ratio=DEFAULT_ASR_TRANSCRIPT_RATIO, seed=42, *, count=None):
    """Choose one source per sample, independently of order and trajectory length."""
    ratio = asr_transcript_ratio(ratio)
    ids = sorted(record["id"] for record in records)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    random.Random(seed).shuffle(ids)
    count = int(len(ids) * ratio + 0.5) if count is None else count
    if not isinstance(count, int) or not 0 <= count <= len(ids):
        raise ValueError("ASR sample count is out of range")
    asr_ids = set(ids[:count])
    return {sample_id: "asr" if sample_id in asr_ids else "audio_text" for sample_id in ids}


def validate_prompt_policy(manifest):
    if (manifest.get("prompt_policy") != SFT_PROMPT_POLICY
            or manifest.get("text_normalization") != TEXT_NORMALIZATION
            or manifest.get("local_time_policy") != LOCAL_TIME_POLICY
            or manifest.get("local_helper_policy") != LOCAL_HELPER_POLICY
            or manifest.get("assist_policy") != ASSIST_POLICY
            or manifest.get("schema_prompt_policy") != SCHEMA_PROMPT_POLICY
            or manifest.get("itn_policy") != ITN_POLICY
            or manifest.get("execution_policy") != EXECUTION_POLICY
            or manifest.get("reasoning_policy", REASONING_POLICY) != REASONING_POLICY
            or manifest.get("guard_recovery_policy", GUARD_RECOVERY_POLICY) != GUARD_RECOVERY_POLICY):
        raise ValueError("Prepared SFT data uses an older prompt/text-normalization/evaluation policy; "
                         "rerun prepare_sft.py in a new output directory")


class UnexposedToolError(ValueError):
    """A gold tool is absent from the actual prompt's complete schemas."""


class GoldGuardConflict(ValueError):
    """An automatic guard rejects a gold target; never train that contradiction."""

    def __init__(self, sample_id, step, feedback):
        self.feedback = deepcopy(feedback)
        super().__init__(f"{sample_id} step {step}: gold blocked by automatic guard: {full_json(feedback)}")


def split_records(records, validation_fraction=0.0, seed=42):
    if not 0 <= validation_fraction < 1 or not records:
        raise ValueError("Need nonempty samples and 0 <= validation_fraction < 1")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    ordered = sorted(records, key=lambda r: str(r["id"]))
    random.Random(seed).shuffle(ordered)
    if validation_fraction == 0:
        return ordered, []
    if len(ordered) < 2:
        raise ValueError("Need >=2 unique samples for a validation split")
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    return ordered[count:], ordered[:count]


def _fold_asr_text(value: str) -> str:
    text = unicodedata.normalize("NFD", str(value)).lower().replace("đ", "d")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return " ".join(re.findall(r"[^\W_]+", text))


def is_asr_argument_damaged(record, prepared_transcript, catalog) -> bool:
    """True when an exact-match spoken gold parameter disappeared from the ASR transcript."""
    clean = _train_transcript(record)
    if not clean or not isinstance(prepared_transcript, dict):
        return False
    clean_folded = _fold_asr_text(clean)
    asr_folded = _fold_asr_text(prepared_transcript.get("transcript", ""))
    if not clean_folded:
        return False
    seen_folded = " ".join(
        [_fold_asr_text(json.dumps(record.get("context", {}), ensure_ascii=False)),
         _fold_asr_text(json.dumps(prepared_transcript.get("history", []), ensure_ascii=False)),
         asr_folded]
    )
    for call in record.get("label", {}).get("tool_calls", []):
        if not isinstance(call, dict):
            continue
        spec = catalog.get(call.get("name"), {}) if isinstance(catalog, dict) else {}
        props = spec.get("parameters", {}).get("properties", {}) if isinstance(spec, dict) else {}
        for key, val in (call.get("arguments") or {}).items():
            if not isinstance(val, str) or len(val.strip()) < 2:
                continue
            prop = props.get(key, {}) if isinstance(props, dict) else {}
            if prop.get("x-match") == "fuzzy" or prop.get("x-ref"):
                continue
            if re.fullmatch(r"[a-z]+_\w+", val.strip()):
                continue
            val_folded = _fold_asr_text(val)
            if not val_folded or len(val_folded) < 2:
                continue
            if f" {val_folded} " in f" {clean_folded} " and f" {val_folded} " not in f" {seen_folded} ":
                return True
    return False


def _current_train_transcript(item, split_dir, asr, transcript_source):
    if transcript_source not in (None, "audio_text", "asr"):
        raise ValueError(f"Unknown transcript source: {transcript_source}")
    transcript = None if transcript_source == "asr" else _train_transcript(item)
    if transcript_source == "audio_text" and transcript is None:
        raise ValueError(f"{item['id']}: selected audio_text source is missing")
    if transcript is None:
        if asr is None:
            raise ValueError(f"{item['id']}: ASR transcript required; pass --asr-model to transcribe train audio")
        if not isinstance(item.get("audio"), str):
            raise ValueError(f"{item['id']}: selected ASR source requires an audio path")
        from .asr_biasing import transcribe_with_item
        transcript = transcribe_with_item(asr, audio_path(split_dir, item["audio"]), item)
        transcript_source = "asr"
    else:
        transcript_source = "audio_text"
    if not isinstance(transcript, str):
        raise ValueError(f"{item['id']}: transcript must be text")
    return normalize_transcript(transcript), transcript_source


def prepare_train_transcript(item, split_dir, asr=None, *, transcript_source=None):
    """Resolve one selected source for all SFT steps, without silent fallback."""
    if is_synthetic_sample(item):
        if not isinstance(item.get("audio_text"), str) or not item["audio_text"].strip():
            raise ValueError(f"{item['id']}: synthetic samples require nonempty audio_text")
        # Synthetic audio fields can contain placeholders. Never decode them,
        # including audio-only history; history must supply its own text.
        asr, transcript_source = None, "audio_text"
    started = time.monotonic()
    transcript, transcript_source = _current_train_transcript(item, split_dir, asr, transcript_source)
    class MissingASR:
        def transcribe(self, path):
            raise ValueError("History audio needs --asr-model or a train history transcript")
    from .asr_biasing import history_asr_for_item
    hist_asr = history_asr_for_item(asr, item, split_dir) if asr is not None else MissingASR()
    history = history_text(item.get("history", []), split_dir, hist_asr, started, float("inf"))
    return {"id": item["id"], "transcript": transcript, "history": history,
            "transcript_source": transcript_source}


def prepare_validation_transcripts(item, split_dir, asr, prepared_transcript):
    """Use ASR for selection and preserve an optional clean view of the same IDs.

    History is resolved once by the SFT preparation. Only the current request
    differs between views, so paired rollouts isolate its recognition errors.
    """
    if asr is None:
        raise ValueError("ASR validation requires --asr-model, including clean-only SFT ablations")
    asr_prepared = deepcopy(prepared_transcript)
    if asr_prepared["transcript_source"] != "asr":
        transcript, source = _current_train_transcript(item, split_dir, asr, "asr")
        asr_prepared.update(transcript=transcript, transcript_source=source)
    clean = _train_transcript(item)
    clean_prepared = None
    if clean is not None:
        clean_prepared = deepcopy(asr_prepared)
        clean_prepared.update(transcript=normalize_transcript(clean), transcript_source="audio_text")
    return asr_prepared, clean_prepared


def pretranscribe_sft_partition(records, split_dir, asr, partition, sources, batch_size=16):
    """Pre-transcribe uncached partition audios using batched decode streams before sample replay."""
    if asr is None or not hasattr(asr, "transcribe_batch"):
        return
    from .asr_biasing import collect_batch_audio_hotwords, hotwords_enabled
    if hotwords_enabled(asr):
        active = [r for r in records if not is_synthetic_sample(r)]
        batch_paths, batch_hws = collect_batch_audio_hotwords(
            active,
            split_dir,
            getattr(asr, "vocab_chars", None),
            skip_main_fn=lambda r: not ((partition != "train") or (sources.get(r["id"]) == "asr")),
        )
        if batch_paths:
            asr.transcribe_batch(batch_paths, batch_size=batch_size, hotwords_list=batch_hws)
        return
    needed_audios = []
    for record in records:
        if is_synthetic_sample(record):
            continue
        needs_current = (partition != "train") or (sources.get(record["id"]) == "asr")
        if needs_current and record.get("audio"):
            needed_audios.append(audio_path(split_dir, record["audio"]))
        for _role, text, audio in history_entries(record.get("history", [])):
            if text is None and audio:
                needed_audios.append(audio_path(split_dir, audio))
    unique_needed = list(dict.fromkeys(needed_audios))
    if unique_needed:
        asr.transcribe_batch(unique_needed, batch_size=batch_size)


def replay_sample(item, catalog, env, split_dir, asr=None, prepared_transcript=None, *, tokenizer=None,
                  include_date_helpers=True, config=None, _recorded_results=None):
    """Replay TRAIN gold in order; never expose future calls in earlier prompts."""
    calls = item.get("label", {}).get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{item['id']}: missing label.tool_calls list (empty list is valid)")
    prepared = prepared_transcript or prepare_train_transcript(item, split_dir, asr)
    transcript, history = prepared["transcript"], prepared["history"]
    prompt_source = None if is_synthetic_sample(item) else prepared.get('prompt_transcript_source')
    context = item.get("context", {})
    enabled = allowed_tools(catalog, context.get("installed_apps", {}))
    observations, examples = [], []
    config = config or AgentConfig()
    monitor = LoopMonitor(empty_read_limit=config.empty_read_limit)
    observed_ids = extract_observed_ids(context, history) if config.enforce_observed_xref else None

    def example(target, step):
        _, prompt = build_step_prompt(transcript, context, history, enabled, observations,
                                      transcript_source=prompt_source)
        exposed = json.loads(prompt)["available_tools"]["tool_schemas"]
        if target["action"] == "call" and target["name"] not in {t["name"] for t in exposed}:
            raise UnexposedToolError(
                f"{item['id']} step {step}: target {target['name']} not exposed in prompt "
                f"({len(exposed)} schemas); check catalog prompt construction")
        row = {"sample_id": item["id"], "step": step, "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ]}
        if _recorded_results is not None:
            row["trajectory_source"] = "synthetic_recorded"
        return row

    def helper_rows(helper):
        """Pack field normalization by actual completion tokens, never a count cap."""
        if helper['name'] != ASSIST_TOOL or helper['arguments']['task'] != 'normalize_fields':
            yield helper
            return
        remaining = list(helper['arguments']['inputs']['fields'])
        while remaining:
            _, prompt = build_step_prompt(transcript, context, history, enabled, observations,
                                          transcript_source=prompt_source)
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
            prefix = chat_token_ids(tokenizer, messages, True)

            def fits(candidate):
                target = {'action': 'call', **candidate}
                full = chat_token_ids(tokenizer, messages+[{'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)}], False)
                return full[:len(prefix)] == prefix and 0 < len(full)-len(prefix) <= min(DEFAULT_MAX_NEW_TOKENS, 384)

            chunk = []
            for field in remaining:
                candidate = {'name': ASSIST_TOOL, 'arguments': {'task': 'normalize_fields', 'inputs': {'fields': chunk+[field]}}}
                if not fits(candidate):
                    break
                chunk.append(field)
            if chunk:
                yield {'name': ASSIST_TOOL, 'arguments': {'task': 'normalize_fields', 'inputs': {'fields': chunk}}}
                remaining = remaining[len(chunk):]
            else:
                # A one-field facade can exceed the generation budget due to its
                # envelope. Preserve the supported primitive interface as fallback.
                field = remaining.pop(0)
                candidate = {'name': 'local_text.normalize', 'arguments': {k: field[k] for k in ('kind', 'text')}}
                if not fits(candidate):
                    raise ValueError(f"{item['id']}: normalization source exceeds the generation token budget; use a shorter source span")
                yield candidate

    def _filter_date_helpers(helpers):
        if include_date_helpers:
            return helpers
        return [
            h for h in helpers
            if h.get("name") != "local_time.calendar_reference"
            and not (h.get("name") == ASSIST_TOOL and (h.get("arguments") or {}).get("task") in ("date", "datetime"))
        ]

    if not calls:
        helpers = helper_demonstrations(transcript, context, history, enabled, observations)
        helpers += specialist_demonstrations(transcript, context, history, enabled, observations, pending=helpers)
        helpers = _filter_date_helpers(task_demonstrations(
            transcript, context, enabled, observations, helpers, normalize_fields=tokenizer is not None))
        for helper in helpers:
            for packed in helper_rows(helper):
                examples.append(example({"action": "call", **packed}, len(examples)))
                local_result = execute_helper(packed["name"], packed["arguments"], context, history, enabled, observations)
                observations.append({**packed, "scope": "local", "result": full_json(local_result)})
        examples.append(example({"action": "done"}, len(examples)))
        return examples

    for step, call in enumerate(calls):
        name, args = call.get("name"), call.get("arguments")
        rejection = precheck_call(name, args, catalog, enabled, monitor, transcript=transcript,
                                  context=context, history=history, observations=observations)
        if rejection:
            if rejection.get('access', {}).get('allowed') is False:
                raise UnexposedToolError(f"{item['id']} step {step}: target {name!r} not exposed: {full_json(rejection)}")
            raise GoldGuardConflict(item['id'], step, rejection)
        if name == "sys.ask_missing_info" and step != len(calls) - 1:
            raise ValueError(f"{item['id']}: invalid or non-terminal ask_missing_info")
        # Local demonstrations are derived exclusively from the visible request,
        # history and previous results, never from this gold action's operands.
        helpers = helper_demonstrations(transcript, context, history, enabled, observations)
        helpers += specialist_demonstrations(transcript, context, history, enabled, observations, pending=helpers)
        helpers = _filter_date_helpers(task_demonstrations(
            transcript, context, enabled, observations, helpers, normalize_fields=tokenizer is not None))
        for helper in helpers:
            for packed in helper_rows(helper):
                examples.append(example({"action": "call", **packed}, len(examples)))
                local_result = execute_helper(packed["name"], packed["arguments"], context, history, enabled, observations)
                observations.append({**packed, "scope": "local", "result": full_json(local_result)})
        examples.append(example({"action": "call", "name": name, "arguments": args}, len(examples)))
        kind = enabled[name].get("kind", "unknown")
        if _recorded_results is None:
            outcome = attempt_env_call(
                env, name, args, kind, monitor, step,
                spec=enabled[name], observed_ids=observed_ids,
                transcript=transcript, context=context, history=history,
                enabled=enabled, observations=observations,
            )
        else:
            from .local_checks import inspect_call
            inspection = inspect_call(name, args, enabled)
            if inspection["status"] == "invalid":
                raise ValueError(f"{item['id']} step {step}: synthetic schema invalid: {inspection['errors']}")
            xref_error = check_observed_xref(args, enabled[name], observed_ids)
            if xref_error:
                raise GoldGuardConflict(item['id'], step, {'stage': 'xref_guard', 'name': name,
                                        'arguments': args, 'error': xref_error['error']})
            result = deepcopy(_recorded_results[step])
            update_observed_ids(name, args, result, kind, observed_ids)
            verification = monitor.record_result(name, args, result, kind, call_index=step)
            verification["origin"] = "synthetic_recorded"
            outcome = {"executed": True, "result": result, "verification": verification,
                       "automatic_execution_feedback": monitor.automatic_feedback(kind, verification, step)}
        if not outcome["executed"]:
            if outcome['feedback'].get('stage') == 'xref_guard':
                raise GoldGuardConflict(item['id'], step, outcome['feedback'])
            raise ValueError(f"{item['id']} step {step}: gold schema invalid or validation failed: {outcome['feedback']}")
        observations.append(result_observation(name, args, outcome, full_json))
    if not calls or calls[-1]["name"] != "sys.ask_missing_info":
        examples.append(example({"action": "done"}, len(examples)))
    return examples


def replay_synthetic_sample(item, catalog, split_dir, *, prepared_transcript=None, tokenizer=None,
                            include_date_helpers=True):
    """Build SFT from aligned recorded calls/results without constructing ToolEnv.

    Schema checks are local; recorded results and final_state are never treated
    as proof of execution. Only the current call's recorded result is appended
    after its target, so neither future results nor final_state enter the prefix.
    """
    label = item.get("label") or {}
    calls, results = label.get("tool_calls"), label.get("tool_results")
    if not isinstance(calls, list) or not isinstance(results, list) or len(calls) != len(results):
        raise ValueError(f"{item['id']}: synthetic label.tool_calls/tool_results must be aligned lists")
    for step, (call, result) in enumerate(zip(calls, results)):
        if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments"), dict) or not isinstance(result, dict)):
            raise ValueError(f"{item['id']} step {step}: synthetic calls need name/arguments and object tool_results")
        try:
            json.dumps([call, result], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item['id']} step {step}: synthetic call/result must be finite JSON") from exc
    prepared = prepared_transcript or prepare_train_transcript(item, split_dir, transcript_source="audio_text")
    return replay_sample(item, catalog, None, split_dir, prepared_transcript=prepared,
                         tokenizer=tokenizer, include_date_helpers=include_date_helpers,
                         _recorded_results=results)


def collapse_trajectory(steps, *, extra=None):
    """Collapse a trajectory's step-wise SFT rows into one multi-turn SFT row."""
    if not steps:
        raise ValueError("Cannot collapse an empty trajectory")
    last = deepcopy(steps[-1])
    payload = json.loads(last["messages"][1]["content"])
    observations = payload.get("previous_actions_and_results", [])
    if len(observations) != len(steps) - 1:
        raise ValueError(
            f"{last.get('sample_id')}: expected {len(steps) - 1} cumulative observations, got {len(observations)}"
        )
    reasonings = [
        (row.get("messages", [{}])[-1].get("reasoning_content") or row.get("reasoning") or "").strip()
        for row in steps
    ]
    collapsed = {
        "sample_id": last["sample_id"],
        "step": 0,
        "full_trajectory": True,
        "assistant_train_mask": [row.get("train", True) is not False for row in steps],
        "messages": last["messages"],
    }
    if any(reasonings):
        collapsed["assistant_reasonings"] = reasonings
    for key in ("transcript_source", "augmentation", "recovery", "reasoning", "trajectory_source"):
        if key in last:
            collapsed[key] = deepcopy(last[key])
    if extra:
        collapsed.update(deepcopy(extra))
    return collapsed


def _thinking_ranges(text, spans):
    ranges = []
    for s0, e0 in spans:
        if s0 >= len(_THINK_OPEN) and text.startswith(_THINK_OPEN, s0 - len(_THINK_OPEN)):
            body = text[s0:e0]
            close = body.find(_THINK_CLOSE)
            if close > 0:
                ranges.append((s0, s0 + close))
    return ranges


def _fuzzy_ranges(text, spans, tools):
    fuzzy_by_fn = {
        t["function"]["name"]: {
            k for k, p in (t["function"].get("parameters") or {}).get("properties", {}).items()
            if isinstance(p, dict) and p.get("x-match") == "fuzzy"
        }
        for t in tools or []
    }
    ranges = []
    for s0, e0 in spans:
        body = text[s0:e0]
        offset = 0
        if s0 >= len(_THINK_OPEN) and text.startswith(_THINK_OPEN, s0 - len(_THINK_OPEN)):
            close = body.find(_THINK_CLOSE)
            if close >= 0:
                offset = close + len(_THINK_CLOSE)
        fn_body = body[offset:]
        fn = _FUNC_RE.search(fn_body)
        keys = fuzzy_by_fn.get(fn.group(1), set()) if fn else set()
        if not keys:
            continue
        for param in _PARAM_RE.finditer(fn_body):
            if param.group(1) in keys:
                ranges.append((s0 + offset + param.start(2), s0 + offset + param.end(2)))
    return ranges


def _encode_native_with_spans(example, tokenizer, max_length, *, fuzzy_weight=None, reasoning_weight=None):
    chat, tools = native_chat(example["messages"])
    payload = json.loads(example["messages"][1]["content"])
    obs_count = len(native_observation_messages(payload.get("previous_actions_and_results", [])))
    pre_obs_chat = chat[:-obs_count - 1]
    kwargs = dict(getattr(tokenizer, "lifemate_template_kwargs", {}))
    kwargs.pop("tools", None)
    pre_text = tokenizer.apply_chat_template(
        pre_obs_chat, tokenize=False, add_generation_prompt=False, tools=tools, **kwargs
    )
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=False, tools=tools, **kwargs
    )
    if not text.startswith(pre_text):
        raise ValueError("Rendered chat diverges from pre-observation request prefix")
    if example.get("full_trajectory"):
        trained = list(example.get("assistant_train_mask") or [])
        matches = list(_ASSISTANT_SPAN.finditer(text, pos=len(pre_text)))
        if not matches or len(matches) != len(trained):
            raise ValueError(
                f"{example.get('sample_id')}: assistant spans ({len(matches)}) do not match steps ({len(trained)})"
            )
        raw_reasonings = example.get("assistant_reasonings")
        if isinstance(raw_reasonings, list) and len(raw_reasonings) == len(trained):
            reasonings = [(r or "").strip() for r in raw_reasonings]
        else:
            last_reason = (example["messages"][-1].get("reasoning_content") or example.get("reasoning") or "").strip()
            reasonings = [""] * (len(trained) - 1) + ([last_reason] if trained else [])
        parts = [text[:len(pre_text)]]
        cursor = len(pre_text)
        for m, reason in zip(matches, reasonings):
            parts.append(text[cursor:m.start(0)])
            body = m.group(1)
            tool_part = body.split(_THINK_CLOSE, 1)[1] if _THINK_CLOSE in body else body
            parts.append(f"<|im_start|>assistant\n{_THINK_OPEN}{reason}{_THINK_CLOSE}{tool_part}")
            cursor = m.end(0)
        parts.append(text[cursor:])
        text = "".join(parts)
        spans = [(m.start(1), m.end(1)) for m in _ASSISTANT_SPAN.finditer(text, pos=len(pre_text))]
    else:
        spans = [(m.start(1), m.end(1)) for m in _ASSISTANT_SPAN.finditer(text, pos=len(pre_text))]
        trained = [False] * (len(spans) - 1) + [example.get("train", True) is not False]
    if not spans or len(spans) != len(trained):
        raise ValueError(
            f"{example.get('sample_id')}: assistant spans ({len(spans)}) do not match steps ({len(trained)})"
        )
    for (s0, _), is_trained in zip(spans, trained):
        if not is_trained:
            continue
        prefix_before_target = text[:s0]
        if not (prefix_before_target.endswith(_THINK_OPEN) or prefix_before_target.endswith(_THINK_CLOSE)):
            raise ValueError("Supervised assistant span must be preceded by a <think> block for train/inference parity")
    fuzzy = _fuzzy_ranges(text, spans, tools) if fuzzy_weight is not None else []
    thinking = _thinking_ranges(text, spans) if reasoning_weight is not None else []
    use_weights = (fuzzy_weight is not None) or (reasoning_weight is not None)
    enc = tokenizer(
        text, add_special_tokens=False, truncation=False,
        return_offsets_mapping=True, return_token_type_ids=False,
    )
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if len(ids) > max_length:
        raise ValueError(
            f"{example['sample_id']} step {example['step']}: {len(ids)} tokens > max_length={max_length}; "
            "increase length, never truncate prompt or gold JSON"
        )
    labels = [-100] * len(ids)
    weights = [0.0] * len(ids)
    j = 0
    for t, (s, e) in enumerate(offs):
        while j < len(spans) and s >= spans[j][1]:
            j += 1
        if j < len(spans) and trained[j] and s >= spans[j][0] and e <= spans[j][1] and e > s:
            labels[t] = ids[t]
            if use_weights:
                if reasoning_weight is not None and any(a <= s < b for a, b in thinking):
                    weights[t] = float(reasoning_weight)
                elif fuzzy_weight is not None and any(a <= s < b for a, b in fuzzy):
                    weights[t] = float(fuzzy_weight)
                else:
                    weights[t] = 1.0
    if not any(label != -100 for label in labels):
        raise ValueError(f"{example.get('sample_id')}: trajectory produced zero supervised tokens")
    row = {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    if use_weights:
        row["loss_weights"] = weights
    return row


def encode_example(example, tokenizer, max_length, *, fuzzy_weight=None, reasoning_weight=None):
    messages = example["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("SFT expects system/user/assistant messages")
    native = getattr(tokenizer, "lifemate_native_fc", False) is True
    use_weights = (fuzzy_weight is not None) or (reasoning_weight is not None)
    if example.get("full_trajectory"):
        if not native or not getattr(tokenizer, "is_fast", False):
            raise ValueError("Full-trajectory SFT encoding requires a fast native-function-calling tokenizer")
        return _encode_native_with_spans(
            example, tokenizer, max_length, fuzzy_weight=fuzzy_weight, reasoning_weight=reasoning_weight
        )
    if use_weights and native and getattr(tokenizer, "is_fast", False):
        return _encode_native_with_spans(
            example, tokenizer, max_length, fuzzy_weight=fuzzy_weight, reasoning_weight=reasoning_weight
        )
    has_reasoning = native and bool((messages[2].get("reasoning_content") or "").strip())
    prefix = chat_token_ids(
        tokenizer, messages[:-1], add_generation_prompt=True,
        enable_thinking=(True if has_reasoning else None),
    )
    full = chat_token_ids(tokenizer, messages, add_generation_prompt=False)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template cannot isolate assistant tokens safely")
    if len(full) > max_length:
        raise ValueError(f"{example['sample_id']} step {example['step']}: {len(full)} tokens > max_length={max_length}; increase length, never truncate prompt or gold JSON")
    labels = [-100] * len(prefix) + full[len(prefix):]
    row = {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}
    if use_weights:
        row["loss_weights"] = [0.0] * len(prefix) + [1.0] * (len(full) - len(prefix))
    return row


class SupervisedDataset:
    def __init__(self, examples, tokenizer, max_length, *, include_debug_metadata=False, source_path=None,
                 fuzzy_weight=None, reasoning_weight=None):
        self.rows = [
            encode_example(e, tokenizer, max_length, fuzzy_weight=fuzzy_weight, reasoning_weight=reasoning_weight)
            for e in examples
        ]
        if include_debug_metadata:
            for index, (row, example) in enumerate(zip(self.rows, examples)):
                row[DEBUG_BATCH_KEY] = {'sample_id': example['sample_id'], 'sft_step': example['step'],
                    'row_index': index, 'source_line': index+1, 'source_path': str(source_path) if source_path else None,
                    'transcript_source': example.get('transcript_source'),
                    'augmentation_variant': example.get('augmentation', {}).get('variant')}
        if not self.rows:
            raise ValueError("Empty SFT dataset")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        batch = {key: torch.tensor([r[key] + [padding] * (length - len(r[key])) for r in rows])
                 for key, padding in (("input_ids", self.pad), ("attention_mask", 0), ("labels", -100))}
        if any("loss_weights" in r for r in rows):
            batch["loss_weights"] = torch.tensor(
                [
                    r.get("loss_weights", [1.0 if lbl != -100 else 0.0 for lbl in r["labels"]])
                    + [0.0] * (length - len(r["input_ids"]))
                    for r in rows
                ],
                dtype=torch.float32,
            )
        if any(DEBUG_BATCH_KEY in row for row in rows):
            batch[DEBUG_BATCH_KEY] = [deepcopy(row.get(DEBUG_BATCH_KEY, {})) for row in rows]
        return batch


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
