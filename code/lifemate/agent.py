"""Observe -> plan a single tool call -> validate -> execute -> observe."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import allowed_tools, prompt_catalog

SYSTEM_PROMPT = """Bạn là AI Agent LifeMate. Chỉ xuất ĐÚNG MỘT JSON object, KHÔNG markdown hoặc diễn giải.
Bạn có transcript yêu cầu nói tiếng Việt, context, history, các tool được cài và kết quả các calls trước.
Mục tiêu: hoàn tất đúng trạng thái ứng dụng với ít calls hợp lý nhất.

Một trong hai dạng output:
{"action":"call","name":"app.tool","arguments":{"key":"value"}}
{"action":"done"}

QUY TẮC BẮT BUỘC:
- Mỗi lần chỉ đề xuất 1 tool call; bạn sẽ được xem kết quả và quyết định tiếp.
- Chỉ gọi tools trong tool_schemas, đúng tên và JSON types/enum/required/x-one-of.
- Với x-ref, trước tiên dùng read tool tương ứng để tìm ID thực; KHÔNG bịa ID.
- Nếu cần kết hợp nhiều hành động, hoàn tất từng hành động theo thứ tự phụ thuộc.
- Dùng user context (current_time/timezone/location/profile) và lịch sử để giải quyết tham chiếu.
- Đừng thay đổi trạng thái ngoài mục tiêu của người dùng; đừng gọi read tools lặp vô ích.
- Nếu thiếu thông tin bắt buộc không suy ra được qua context/read tools, gọi
  sys.ask_missing_info với {"tool_name":"tool dự kiến", "params":["tham số còn thiếu"]}, rồi dừng.
- Nếu không ứng dụng được cài nào có khả năng đáp ứng yêu cầu: xuất {"action":"done"} ngay, KHÔNG gọi ask_missing_info.
- Nếu đã hoàn tất: {"action":"done"}. Không tự nói là đã gọi tool.
- Arguments must be a JSON object. Output exactly one JSON object.
"""


def parse_decision(raw: str) -> dict:
    """Find first JSON object, including model outputs with harmless surrounding text."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "tool_calls" in obj and isinstance(obj["tool_calls"], list) and obj["tool_calls"]:
            obj = obj["tool_calls"][0]
        if obj.get("action") in ("done", "finish", "stop", "none"):
            return {"action": "done"}
        if obj.get("action") == "call" or ("name" in obj and "arguments" in obj):
            return {"action": "call", "name": obj.get("name"), "arguments": obj.get("arguments")}
    raise ValueError(f"Model did not produce a valid tool decision: {raw[:300]}")


def short_json(obj: Any, limit: int = 5000) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def fit_prompt_payload(info: dict, max_chars: int = 20000) -> str:
    """Fit large catalogs while preserving syntactically valid JSON at all times."""
    # The ranker puts likely-relevant schemas first. Always retain meta schema.
    from copy import deepcopy
    payload = deepcopy(info)
    cat = payload["available_tools"]

    def dumped():
        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))

    s = dumped()
    while len(s) > max_chars and len(cat["tool_schemas"]) > 2:
        # Remove lowest-ranked non-meta schema.
        i = next((i for i in range(len(cat["tool_schemas"]) - 1, -1, -1)
                  if cat["tool_schemas"][i]["name"] != "sys.ask_missing_info"), None)
        if i is None:
            break
        cat["tool_schemas"].pop(i)
        s = dumped()
    if len(s) > max_chars:
        for desc in cat["tool_index"]:
            desc["description"] = desc.get("description", "")[:35]
        s = dumped()
    if len(s) > max_chars:
        # Preserve tool names but make previous observations concise.
        for o in payload["previous_actions_and_results"]:
            if isinstance(o.get("result"), str):
                o["result"] = o["result"][:1200]
        s = dumped()
    if len(s) > max_chars:
        payload["conversation_history"] = payload["conversation_history"][-3:]
        s = dumped()
    if len(s) > max_chars:
        # Extremely large catalogs: collapse descriptions in index first.
        cat["tool_index"] = [{"name": x["name"]} for x in cat["tool_index"]]
        s = dumped()
    # Do not truncate into an invalid JSON document; if the request itself
    # is enormous, the tokenizer max_input_tokens remains a last-resort limit.
    return s


def _train_transcript(item: dict) -> str | None:
    """Opt-in, TRAIN ONLY. Never read labels or transcripts from public/private test."""
    txt = item.get("audio_text")
    if isinstance(txt, str):
        return txt
    if isinstance(txt, dict):
        for key in ("latest", "current", "text", "transcript", "audio_text"):
            if isinstance(txt.get(key), str):
                return txt[key]
    if isinstance(txt, list):
        for entry in reversed(txt):
            if isinstance(entry, str):
                return entry
            if isinstance(entry, dict):
                for k in ("audio_text", "text", "transcript", "content"):
                    if isinstance(entry.get(k), str):
                        return entry[k]
    return None


def audio_path(split_dir: Path, relative: str) -> Path:
    path = (split_dir / relative).resolve()
    if not path.is_relative_to(split_dir.resolve()):
        raise ValueError(f"Audio path escapes split directory: {relative}")
    return path


def history_text(history: Any, split_dir: Path, asr: Any, start: float, time_limit: float) -> list[dict]:
    result = []
    if not isinstance(history, list):
        return result
    for turn in history[-12:]:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("speaker", "user")))
        text = next((turn[k] for k in ("audio_text", "text", "content", "transcript")
                     if isinstance(turn.get(k), str) and turn[k].strip()), "")
        if not text and role.lower() in ("user", "human"):
            path = turn.get("audio", turn.get("audio_path"))
            if isinstance(path, str) and time.monotonic() - start < time_limit - 5:
                text = asr.transcribe(audio_path(split_dir, path))
        if text:
            result.append({"role": role, "text": text[:1700]})
    return result


@dataclass
class AgentConfig:
    max_calls: int = 8
    max_invalid: int = 3
    max_tool_details: int = 35
    sample_seconds: float = 28.0
    # Reserving time is best effort, not a hard CUDA generation timeout.
    reserve_seconds: float = 1.0
    use_train_transcript: bool = False


def build_step_prompt(transcript, context, history, enabled, observations, last_tool, max_tool_details):
    """Identical candidate selection and prompt payload for SFT and inference."""
    search_query = transcript + " " + " ".join(
        str(o.get("name", "")) + " " + short_json(o.get("result", ""), 900)
        for o in observations[-2:]
    )
    tools = prompt_catalog(enabled, search_query, last_tool, max_tool_details)
    info = {
        "request_transcript": transcript,
        "user_context": {k: v for k, v in context.items() if k != "installed_apps"},
        "conversation_history": history,
        "available_tools": tools,
        "previous_actions_and_results": observations[-7:],
    }
    return tools, fit_prompt_payload(info)


class LifeMateAgent:
    def __init__(self, asr: Any, llm: Any, catalog: dict[str, dict], cfg: AgentConfig):
        self.asr = asr
        self.llm = llm
        self.catalog = catalog
        self.cfg = cfg

    def run(self, item: dict, env: Any, split_dir: Path, split: str) -> dict:
        """env is created exactly once by caller; never instantiate another here."""
        t0 = time.monotonic()
        context = item.get("context", {})
        enabled = allowed_tools(self.catalog, context.get("installed_apps", {}))
        if len(enabled) == 1:  # only sys.ask_missing_info; no app can act
            return {"transcript": "", "elapsed_seconds": 0, "executed": 0, "invalid_proposals": 0}
        transcript = _train_transcript(item) if self.cfg.use_train_transcript and split == "train" else None
        if transcript is None:
            transcript = self.asr.transcribe(audio_path(split_dir, item["audio"]))
        history = history_text(item.get("history", []), split_dir, self.asr, t0, self.cfg.sample_seconds)
        observations: list[dict] = []
        invalid = 0
        executed = 0
        last_tool: str | None = None
        epoch = 0
        recent_reads: set[tuple[int, str, str]] = set()
        while executed < self.cfg.max_calls and invalid < self.cfg.max_invalid:
            if time.monotonic() - t0 > self.cfg.sample_seconds - self.cfg.reserve_seconds:
                break
            tools, user_prompt = build_step_prompt(
                transcript, context, history, enabled, observations, last_tool, self.cfg.max_tool_details)
            raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            try:
                decision = parse_decision(raw)
            except ValueError as exc:
                invalid += 1
                observations.append({"error": str(exc)[:450], "instruction": "Trả về JSON hợp lệ."})
                continue
            if decision["action"] == "done":
                break
            name, args = decision.get("name"), decision.get("arguments")
            if not isinstance(name, str) or name not in enabled or name not in {x["name"] for x in tools["tool_schemas"]}:
                invalid += 1
                observations.append({"error": f"Tool {name!r} not exposed in tool_schemas; choose only from listed schemas."})
                continue
            if not isinstance(args, dict):
                invalid += 1
                observations.append({"error": "arguments must be a JSON object"})
                continue
            if name == "sys.ask_missing_info":
                if args.get("tool_name") not in enabled or args.get("tool_name") == name:
                    invalid += 1
                    observations.append({"error": "Missing-info tool_name must be another available tool."})
                    continue
            try:
                validation = env.validate(name, args)
            except Exception as exc:
                validation = {"ok": False, "error": repr(exc)}
            if not isinstance(validation, dict) or not validation.get("ok", False):
                invalid += 1
                observations.append({"error": "Schema invalid: " + short_json(validation, 1300), "name": name})
                continue
            read_key = (epoch, name, short_json(args))
            if enabled[name].get("kind") == "read" and read_key in recent_reads:
                invalid += 1
                observations.append({"error": "Duplicate read without intervening state change", "name": name})
                continue
            # This is the *only* place execution takes place. Calls that fail are logged by ToolEnv.
            try:
                result = env.call(name, args)
            except Exception as exc:
                result = {"ok": False, "error": repr(exc)}
            executed += 1
            last_tool = name
            if enabled[name].get("kind") == "read":
                recent_reads.add(read_key)
            elif enabled[name].get("kind") == "write":
                epoch += 1
                recent_reads.clear()
            observations.append({"name": name, "arguments": args, "result": short_json(result, 4500)})
            if name == "sys.ask_missing_info":
                break
        return {"transcript": transcript, "elapsed_seconds": round(time.monotonic() - t0, 3),
                "executed": executed, "invalid_proposals": invalid}
