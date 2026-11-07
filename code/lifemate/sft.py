"""Train-only gold replay, sample-level splitting, and assistant-only SFT loss."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
import time

from .agent import (SYSTEM_PROMPT, _train_transcript, audio_path, build_step_prompt,
                    full_json, history_text)
from .catalog import allowed_tools
from .llm import chat_token_ids
from .local_time import LOCAL_TIME_POLICY
from .local_helpers import LOCAL_HELPER_POLICY, execute_helper
from .execution_policy import EXECUTION_POLICY, LoopMonitor
from .tool_execution import precheck_call, attempt_env_call, result_observation
from .specialist_training import specialist_demonstrations
from .temporal_training import helper_demonstrations
from .assist_training import task_demonstrations
from .local_assist import ASSIST_TOOL, ASSIST_POLICY
from .context import DEFAULT_MAX_NEW_TOKENS
from .text import TEXT_NORMALIZATION, normalize_transcript
from .itn import ITN_POLICY
from .schema_prompt import SCHEMA_PROMPT_POLICY
from .numerics import DEBUG_BATCH_KEY


SFT_PROMPT_POLICY = "compact-schemas-shared-prefix-v13"
VALIDATION_ROLLOUT_POLICY = "asr-only-paired-clean-v1"
DEFAULT_ASR_TRANSCRIPT_RATIO = 0.5


def asr_transcript_ratio(value):
    ratio = float(value)
    if not 0 <= ratio <= 1:
        raise ValueError("ASR transcript ratio must be between 0 and 1")
    return ratio


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
            or manifest.get("execution_policy") != EXECUTION_POLICY):
        raise ValueError("Prepared SFT data uses an older prompt/text-normalization/evaluation policy; "
                         "rerun prepare_sft.py in a new output directory")


class UnexposedToolError(ValueError):
    """A gold tool is absent from the actual prompt's complete schemas."""


def split_records(records, validation_fraction=0.1, seed=42):
    if not 0 < validation_fraction < 1 or len(records) < 2:
        raise ValueError("Need >=2 unique samples and 0 < validation_fraction < 1")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")
    ordered = sorted(records, key=lambda r: str(r["id"]))
    random.Random(seed).shuffle(ordered)
    count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    return ordered[count:], ordered[:count]


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
        transcript = asr.transcribe(audio_path(split_dir, item["audio"]))
        transcript_source = "asr"
    else:
        transcript_source = "audio_text"
    if not isinstance(transcript, str):
        raise ValueError(f"{item['id']}: transcript must be text")
    return normalize_transcript(transcript), transcript_source


def prepare_train_transcript(item, split_dir, asr=None, *, transcript_source=None):
    """Resolve one selected source for all SFT steps, without silent fallback."""
    started = time.monotonic()
    transcript, transcript_source = _current_train_transcript(item, split_dir, asr, transcript_source)
    class MissingASR:
        def transcribe(self, path):
            raise ValueError("History audio needs --asr-model or a train history transcript")
    history = history_text(item.get("history", []), split_dir, asr or MissingASR(), started, float("inf"))
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


def replay_sample(item, catalog, env, split_dir, asr=None, prepared_transcript=None, *, tokenizer=None):
    """Replay TRAIN gold in order; never expose future calls in earlier prompts."""
    calls = item.get("label", {}).get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{item['id']}: missing label.tool_calls list (empty list is valid)")
    prepared = prepared_transcript or prepare_train_transcript(item, split_dir, asr)
    transcript, history = prepared["transcript"], prepared["history"]
    context = item.get("context", {})
    enabled = allowed_tools(catalog, context.get("installed_apps", {}))
    observations, examples = [], []
    monitor = LoopMonitor()

    def example(target, step):
        _, prompt = build_step_prompt(transcript, context, history, enabled, observations)
        exposed = json.loads(prompt)["available_tools"]["tool_schemas"]
        if target["action"] == "call" and target["name"] not in {t["name"] for t in exposed}:
            raise UnexposedToolError(
                f"{item['id']} step {step}: target {target['name']} not exposed in prompt "
                f"({len(exposed)} schemas); check catalog prompt construction")
        return {"sample_id": item["id"], "step": step, "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ]}

    def helper_rows(helper):
        """Pack field normalization by actual completion tokens, never a count cap."""
        if helper['name'] != ASSIST_TOOL or helper['arguments']['task'] != 'normalize_fields':
            yield helper
            return
        remaining = list(helper['arguments']['inputs']['fields'])
        while remaining:
            _, prompt = build_step_prompt(transcript, context, history, enabled, observations)
            messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
            prefix = chat_token_ids(tokenizer, messages, True)

            def fits(candidate):
                target = {'action': 'call', **candidate}
                full = chat_token_ids(tokenizer, messages+[{'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)}], False)
                return full[:len(prefix)] == prefix and 0 < len(full)-len(prefix) <= DEFAULT_MAX_NEW_TOKENS

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

    for step, call in enumerate(calls):
        name, args = call.get("name"), call.get("arguments")
        rejection = precheck_call(name, args, catalog, enabled, monitor)
        if rejection:
            reason = rejection.get("feedback", {}).get("reason", rejection.get("access", {}).get("status", rejection["stage"]))
            raise ValueError(f"{item['id']} step {step}: gold blocked by automatic guard: {name!r}, {reason}")
        if name == "sys.ask_missing_info" and step != len(calls) - 1:
            raise ValueError(f"{item['id']}: invalid or non-terminal ask_missing_info")
        # Local demonstrations are derived exclusively from the visible request,
        # history and previous results, never from this gold action's operands.
        helpers = helper_demonstrations(transcript, context, history, enabled, observations)
        helpers += specialist_demonstrations(transcript, context, history, enabled, observations, pending=helpers)
        helpers = task_demonstrations(transcript, context, enabled, observations, helpers,
                                      normalize_fields=tokenizer is not None)
        for helper in helpers:
            for packed in helper_rows(helper):
                examples.append(example({"action": "call", **packed}, len(examples)))
                local_result = execute_helper(packed["name"], packed["arguments"], context, history, enabled, observations)
                observations.append({**packed, "scope": "local", "result": full_json(local_result)})
        examples.append(example({"action": "call", "name": name, "arguments": args}, len(examples)))
        outcome = attempt_env_call(env, name, args, enabled[name].get("kind", "unknown"), monitor, step)
        if not outcome["executed"]:
            raise ValueError(f"{item['id']} step {step}: gold schema invalid or validation failed: {outcome['feedback']}")
        observations.append(result_observation(name, args, outcome, full_json))
    if not calls or calls[-1]["name"] != "sys.ask_missing_info":
        examples.append(example({"action": "done"}, len(examples)))
    return examples


def encode_example(example, tokenizer, max_length):
    messages = example["messages"]
    if [m["role"] for m in messages] != ["system", "user", "assistant"]:
        raise ValueError("SFT expects system/user/assistant messages")
    prefix = chat_token_ids(tokenizer, messages[:-1], add_generation_prompt=True)
    full = chat_token_ids(tokenizer, messages, add_generation_prompt=False)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template cannot isolate assistant tokens safely")
    if len(full) > max_length:
        raise ValueError(f"{example['sample_id']} step {example['step']}: {len(full)} tokens > max_length={max_length}; increase length, never truncate prompt or gold JSON")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


class SupervisedDataset:
    def __init__(self, examples, tokenizer, max_length, *, include_debug_metadata=False, source_path=None):
        self.rows = [encode_example(e, tokenizer, max_length) for e in examples]
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
        if any(DEBUG_BATCH_KEY in row for row in rows):
            batch[DEBUG_BATCH_KEY] = [deepcopy(row.get(DEBUG_BATCH_KEY, {})) for row in rows]
        return batch


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
