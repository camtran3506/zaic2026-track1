"""Observe -> plan a single tool call -> validate -> execute -> observe."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import allowed_tools, prompt_catalog
from .history import prompt_history, resolve_history

SYSTEM_PROMPT = """Bạn là AI Agent LifeMate. Chỉ xuất ĐÚNG MỘT JSON object, KHÔNG markdown hoặc diễn giải.
Bạn có transcript yêu cầu nói tiếng Việt, context, history, các tool được cài và kết quả các calls trước.
Mục tiêu: hoàn tất đúng trạng thái ứng dụng với ít calls hợp lý nhất.

Một trong các dạng output:
{"action":"call","name":"app.tool","arguments":{"key":"value"}}
{"action":"done"}

QUY TẮC BẮT BUỘC:
- Mỗi lần chỉ đề xuất 1 tool call; bạn sẽ được xem kết quả và quyết định tiếp.
- Chỉ gọi tools trong tool_schemas, đúng tên và JSON types/enum/required/x-one-of.
- tool_index và tool_schemas chứa toàn bộ tools được phép cùng schema đầy đủ.
  Dùng action call với arguments đúng schema; không đoán tham số.
- Với x-ref, dùng ID từ kết quả read tool thành công trong history hoặc lượt hiện tại;
  nếu chưa có thì gọi read tool tương ứng để tìm ID thực; KHÔNG bịa ID.
- Nếu cần kết hợp nhiều hành động, hoàn tất từng hành động theo thứ tự phụ thuộc.
- Dùng user context (current_time/timezone/location/profile) và lịch sử để giải quyết tham chiếu.
- Đừng thay đổi trạng thái ngoài mục tiêu của người dùng; đừng gọi read tools lặp vô ích.
- Nếu thiếu thông tin bắt buộc không suy ra được qua context/read tools, gọi
  sys.ask_missing_info với {"tool_name":"tool dự kiến", "params":["tham số còn thiếu"]}, rồi dừng.
- Nếu không ứng dụng được cài nào có khả năng đáp ứng yêu cầu: xuất {"action":"done"} ngay, KHÔNG gọi ask_missing_info.
- Nếu đã hoàn tất: {"action":"done"}. Không tự nói là đã gọi tool.
- Arguments must be a JSON object. Output exactly one JSON object.
"""


def parse_decision(raw: str, catalog=None) -> dict:
    """Find first JSON object, including model outputs with harmless surrounding text."""
    from .function_calling import parse_native_call, visible_output
    raw = visible_output(raw)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(raw):
        if raw.startswith("<tool_call>", idx):
            return parse_native_call(raw[idx:], catalog or {})
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
        if obj.get("action") == "inspect":
            return {"action": "inspect", "name": obj.get("name")}
        if obj.get("action") == "call" or ("name" in obj and "arguments" in obj):
            return {"action": "call", "name": obj.get("name"), "arguments": obj.get("arguments")}
    raise ValueError(f"Model did not produce a valid tool decision: {raw}")


def full_json(obj: Any) -> str:
    """Serialize prompt data without character limits or lossy size fitting."""
    return json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))


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
    def transcribe(relative):
        if time.monotonic() - start < time_limit - 5:
            return asr.transcribe(audio_path(split_dir, relative))
        return None

    return prompt_history(resolve_history(history, transcribe))


@dataclass
class AgentConfig:
    max_calls: int = 8
    max_invalid: int = 3
    sample_seconds: float = float("inf")
    # Reserving time is best effort, not a hard CUDA generation timeout.
    reserve_seconds: float = 1.0
    use_train_transcript: bool = False


def build_step_prompt(transcript, context, history, enabled, observations):
    """Use the same complete catalog, history and observations in SFT/inference."""
    tools = prompt_catalog(enabled)
    info = {
        "request_transcript": transcript,
        "user_context": {k: v for k, v in context.items() if k != "installed_apps"},
        "conversation_history": history,
        "available_tools": tools,
        "previous_actions_and_results": observations,
    }
    # Token limits are checked by LocalLLM/encode_example, never by dropping data.
    return tools, full_json(info)


class LifeMateAgent:
    def __init__(self, asr: Any, llm: Any, catalog: dict[str, dict], cfg: AgentConfig):
        self.asr = asr
        self.llm = llm
        self.catalog = catalog
        self.cfg = cfg

    def run(self, item: dict, env: Any, split_dir: Path, split: str,
            prepared_transcript: dict | None = None) -> dict:
        """env is created exactly once by caller; never instantiate another here."""
        t0 = time.monotonic()
        context = item.get("context", {})
        enabled = allowed_tools(self.catalog, context.get("installed_apps", {}))
        if len(enabled) == 1:  # only sys.ask_missing_info; no app can act
            return {"transcript": prepared_transcript["transcript"] if prepared_transcript else "",
                    "elapsed_seconds": 0, "executed": 0, "invalid_proposals": 0,
                    "stop_reason": "no_available_tools"}
        if prepared_transcript is not None:
            if prepared_transcript.get("id") != item["id"]:
                raise ValueError("Prepared transcript ID does not match current sample")
            transcript = prepared_transcript["transcript"]
            history = prompt_history(prepared_transcript["history"])
        else:
            transcript = _train_transcript(item) if self.cfg.use_train_transcript and split == "train" else None
            if transcript is None:
                transcript = self.asr.transcribe(audio_path(split_dir, item["audio"]))
            history = history_text(item.get("history", []), split_dir, self.asr, t0, self.cfg.sample_seconds)
        observations: list[dict] = []
        invalid = 0
        executed = 0
        inspected_tool = None
        inspections = 0
        stop_reason = None
        while executed < self.cfg.max_calls and invalid < self.cfg.max_invalid:
            if time.monotonic() - t0 > self.cfg.sample_seconds - self.cfg.reserve_seconds:
                stop_reason = "time_limit"
                break
            tools, user_prompt = build_step_prompt(
                transcript, context, history, enabled, observations)
            raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            try:
                decision = parse_decision(raw, enabled)
            except ValueError as exc:
                invalid += 1
                observations.append({"error": str(exc), "instruction": "Trả về JSON hợp lệ."})
                continue
            if decision["action"] == "done":
                stop_reason = "done"
                break
            if decision["action"] == "inspect":
                # Compatibility with older adapters: all schemas are already exposed.
                name = decision.get("name")
                if (not isinstance(name, str) or name not in enabled or name == inspected_tool
                        or inspections >= self.cfg.max_calls):
                    invalid += 1
                    observations.append({"error": "Invalid, duplicate or excessive schema inspection"})
                else:
                    inspected_tool = name
                    inspections += 1
                continue
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
                if not isinstance(validation, dict) or not validation.get("ok"):
                    raise ValueError(f"Schema invalid: {validation!r}")
            except Exception as exc:
                invalid += 1
                observations.append({"error": str(exc), "name": name})
                continue
            # This is the *only* place execution takes place. Calls that fail are logged by ToolEnv.
            try:
                result = env.call(name, args)
            except Exception as exc:
                result = {"ok": False, "error": repr(exc)}
            executed += 1
            observations.append({"name": name, "arguments": args, "result": full_json(result)})
            if name == "sys.ask_missing_info":
                stop_reason = "ask_missing_info"
                break
        if stop_reason is None:
            stop_reason = "max_calls" if executed >= self.cfg.max_calls else "max_invalid"
        return {"transcript": transcript, "elapsed_seconds": round(time.monotonic() - t0, 3),
                "executed": executed, "invalid_proposals": invalid, "schema_inspections": inspections,
                "stop_reason": stop_reason}
