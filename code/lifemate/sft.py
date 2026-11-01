"""Train-only gold replay, sample-level splitting, and assistant-only SFT loss."""
from __future__ import annotations

import hashlib
import json
import random
import time

from .agent import (SYSTEM_PROMPT, _train_transcript, audio_path, build_step_prompt,
                    history_text, short_json)
from .catalog import allowed_tools
from .guard import CallGuard, GuardError
from .llm import chat_token_ids


class UnexposedToolError(ValueError):
    """A valid gold tool is absent from the actual prompt's selected schemas."""


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


def prepare_train_transcript(item, split_dir, asr=None):
    """Resolve train text/audio once; reuse the same input for SFT and rollouts."""
    started = time.monotonic()
    transcript = _train_transcript(item)
    if transcript is None:
        if asr is None:
            raise ValueError(f"{item['id']}: audio_text missing; pass --asr-model to transcribe train audio")
        transcript = asr.transcribe(audio_path(split_dir, item["audio"]))
    class MissingASR:
        def transcribe(self, path):
            raise ValueError("History audio needs --asr-model or a train history transcript")
    history = history_text(item.get("history", []), split_dir, asr or MissingASR(), started, float("inf"))
    return {"id": item["id"], "transcript": transcript, "history": history}


def replay_sample(item, catalog, env, split_dir, asr=None, max_tool_details=35,
                  prepared_transcript=None):
    """Replay TRAIN gold in order; never expose future calls in earlier prompts."""
    calls = item.get("label", {}).get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{item['id']}: missing label.tool_calls list (empty list is valid)")
    prepared = prepared_transcript or prepare_train_transcript(item, split_dir, asr)
    transcript, history = prepared["transcript"], prepared["history"]
    context = item.get("context", {})
    enabled = allowed_tools(catalog, context.get("installed_apps", {}))
    observations, examples = [], []
    last_tool = None
    inspected_tool = None
    guard = CallGuard(enabled, history)

    def example(target, step):
        tools, prompt = build_step_prompt(transcript, context, history, enabled, observations,
                                          last_tool, max_tool_details, inspected_tool)
        exposed = json.loads(prompt)["available_tools"]["tool_schemas"]
        if target["action"] == "call" and target["name"] not in {t["name"] for t in exposed}:
            raise UnexposedToolError(
                f"{item['id']} step {step}: target {target['name']} not exposed in prompt "
                f"({len(exposed)} schemas) after schema inspection; check catalog prompt construction")
        return {"sample_id": item["id"], "step": step, "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
        ]}

    for step, call in enumerate(calls):
        name, args = call.get("name"), call.get("arguments")
        if name not in enabled or not isinstance(args, dict):
            raise ValueError(f"{item['id']} step {step}: unavailable tool or non-object arguments")
        if name == "sys.ask_missing_info":
            if step != len(calls) - 1 or args.get("tool_name") not in enabled or args.get("tool_name") == name:
                raise ValueError(f"{item['id']}: invalid or non-terminal ask_missing_info")
        validation = env.validate(name, args)
        if not isinstance(validation, dict) or not validation.get("ok"):
            raise ValueError(f"{item['id']} step {step}: gold schema invalid: {validation}")
        try:
            guard.check(name, args)
        except GuardError as exc:
            raise ValueError(f"{item['id']} step {step}: gold rejected by inference guard: {exc}") from exc
        _, prompt = build_step_prompt(transcript, context, history, enabled, observations,
                                       last_tool, max_tool_details, inspected_tool)
        exposed = {spec["name"] for spec in json.loads(prompt)["available_tools"]["tool_schemas"]}
        if name not in exposed:
            # Supervise an explicit schema request, just as inference can request it.
            # The first prompt contains no gold args/schema injected from the label.
            examples.append(example({"action": "inspect", "name": name}, len(examples)))
            inspected_tool = name
        examples.append(example({"action": "call", "name": name, "arguments": args}, len(examples)))
        try:
            result = env.call(name, args)
        except Exception as exc:
            # Match the inference agent's observation exactly, including failed tool calls.
            result = {"ok": False, "error": repr(exc)}
        observations.append({"name": name, "arguments": args, "result": short_json(result, 4500)})
        last_tool = name
        guard.observe(name, args, result)
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
        raise ValueError(f"{example['sample_id']} step {example['step']}: {len(full)} tokens > max_length={max_length}; increase length, never truncate gold JSON")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


class SupervisedDataset:
    def __init__(self, examples, tokenizer, max_length):
        self.rows = [encode_example(e, tokenizer, max_length) for e in examples]
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
        return {key: torch.tensor([r[key] + [padding] * (length - len(r[key])) for r in rows])
                for key, padding in (("input_ids", self.pad), ("attention_mask", 0), ("labels", -100))}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
