"""Observe -> plan a single tool call -> validate -> execute -> observe."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import allowed_tools, prompt_catalog
from .text import history_turn_text, normalize_transcript

SYSTEM_PROMPT = """Bạn là AI Agent LifeMate. Chỉ xuất ĐÚNG MỘT JSON object, KHÔNG markdown hoặc diễn giải.
Bạn có transcript yêu cầu nói tiếng Việt, context, history, các tool được cài và kết quả các calls trước.
Mục tiêu: hoàn tất đúng trạng thái ứng dụng với ít calls hợp lý nhất.

Một trong các dạng output:
{"action":"call","name":"app.tool","arguments":{"key":"value"}}
{"action":"done"}
{"action":"inspect","name":"app.tool"}

QUY TẮC BẮT BUỘC:
- Mỗi lần chỉ đề xuất 1 tool call; bạn sẽ được xem kết quả và quyết định tiếp.
- Chỉ gọi tools trong tool_schemas, đúng tên và JSON types/enum/required/x-one-of.
- tool_index liệt kê toàn bộ tools được phép. Nếu tool cần dùng có trong tool_index
  nhưng chưa có schema, xuất {"action":"inspect","name":"app.tool"} để xem schema.
  inspect là thao tác nội bộ, không thực thi tool và không thay đổi trạng thái ứng dụng.
  Sau khi xem schema, dùng action call với arguments đúng schema; không đoán tham số.
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
        if obj.get("action") == "inspect":
            return {"action": "inspect", "name": obj.get("name")}
        if obj.get("action") == "call" or ("name" in obj and "arguments" in obj):
            return {"action": "call", "name": obj.get("name"), "arguments": obj.get("arguments")}
    raise ValueError(f"Model did not produce a valid tool decision: {raw[:300]}")


def short_json(obj: Any, limit: int = 5000) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


def fit_prompt_payload(info: dict, max_chars: int = 20000, protected_tools=()) -> str:
    """Fit large catalogs while preserving syntactically valid JSON at all times."""
    # Retain the explicitly inspected schema and meta schema with full parameters.
    from copy import deepcopy
    payload = deepcopy(info)
    cat = payload["available_tools"]

    def dumped():
        return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))

    s = dumped()
    while len(s) > max_chars and len(cat["tool_schemas"]) > 2:
        # Remove lowest-ranked non-meta schema.
        i = next((i for i in range(len(cat["tool_schemas"]) - 1, -1, -1)
                  if cat["tool_schemas"][i]["name"] not in {"sys.ask_missing_info", *protected_tools}), None)
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
        cat["tool_index"] = [{"name": x["name"], "description": x.get("description", "")[:35]}
                             for x in cat["tool_index"]]
        s = dumped()
    # Never cut JSON/schema definitions. LocalLLM checks the actual token budget
    # and raises if this complete payload still does not fit.
    return s


def _train_transcript(item: dict) -> str | None:
    """Opt-in, TRAIN ONLY. Never read labels or transcripts from public/private test."""
    txt = item.get("audio_text")
    if isinstance(txt, str):
        return normalize_transcript(txt)
    if isinstance(txt, dict):
        for key in ("latest", "current", "text", "transcript", "audio_text"):
            if isinstance(txt.get(key), str):
                return normalize_transcript(txt[key])
    if isinstance(txt, list):
        for entry in reversed(txt):
            if isinstance(entry, str):
                return normalize_transcript(entry)
            if isinstance(entry, dict):
                for k in ("audio_text", "text", "transcript", "content"):
                    if isinstance(entry.get(k), str):
                        return normalize_transcript(entry[k])
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
        text = history_turn_text(turn)
        if text is None and role.lower() in ("user", "human"):
            path = turn.get("audio", turn.get("audio_path"))
            if isinstance(path, str) and time.monotonic() - start < time_limit - 5:
                text = normalize_transcript(asr.transcribe(audio_path(split_dir, path)))
        if text:
            result.append({"role": role, "text": text[:1700]})
    return result


@dataclass
class AgentConfig:
    max_calls: int = 8
    max_invalid: int = 3
    max_tool_details: int = 35
    sample_seconds: float = float("inf")
    # Reserving time is best effort, not a hard CUDA generation timeout.
    reserve_seconds: float = 1.0
    use_train_transcript: bool = False


def build_step_prompt(transcript, context, history, enabled, observations, last_tool, max_tool_details,
                      inspected_tool=None):
    """Identical candidate selection and prompt payload for SFT and inference."""
    search_query = transcript + " " + " ".join(
        str(o.get("name", "")) + " " + short_json(o.get("result", ""), 900)
        for o in observations[-2:]
    )
    tools = prompt_catalog(enabled, search_query, last_tool, max_tool_details, inspected_tool)
    info = {
        "request_transcript": transcript,
        "user_context": {k: v for k, v in context.items() if k != "installed_apps"},
        "conversation_history": history,
        "available_tools": tools,
        "previous_actions_and_results": observations[-7:],
    }
    payload = fit_prompt_payload(info, protected_tools=(inspected_tool,) if inspected_tool else ())
    # Validate calls against schemas actually visible after size fitting.
    return json.loads(payload)["available_tools"], payload


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
                    "elapsed_seconds": 0, "executed": 0, "invalid_proposals": 0}
        if prepared_transcript is not None:
            if prepared_transcript.get("id") != item["id"]:
                raise ValueError("Prepared transcript ID does not match current sample")
            transcript = normalize_transcript(prepared_transcript["transcript"])
            history = [{"role": turn["role"], "text": turn["text"][:1700]}
                       for turn in prepared_transcript["history"][-12:]]
        else:
            transcript = _train_transcript(item) if self.cfg.use_train_transcript and split == "train" else None
            if transcript is None:
                transcript = normalize_transcript(self.asr.transcribe(audio_path(split_dir, item["audio"])))
            history = history_text(item.get("history", []), split_dir, self.asr, t0, self.cfg.sample_seconds)
        observations: list[dict] = []
        invalid = 0
        executed = 0
        last_tool: str | None = None
        inspected_tool = None
        inspections = 0
        guard = CallGuard(enabled)
        while executed < self.cfg.max_calls and invalid < self.cfg.max_invalid:
            if time.monotonic() - t0 > self.cfg.sample_seconds - self.cfg.reserve_seconds:
                break
            tools, user_prompt = build_step_prompt(
                transcript, context, history, enabled, observations, last_tool, self.cfg.max_tool_details,
                inspected_tool)
            raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            try:
                decision = parse_decision(raw)
            except ValueError as exc:
                invalid += 1
                observations.append({"error": str(exc)[:450], "instruction": "Trả về JSON hợp lệ."})
                continue
            if decision["action"] == "done":
                break
            if decision["action"] == "inspect":
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
                args = validated_arguments(env, name, args, enabled[name].get("parameters", {}), context)
                guard.check(name, args)
            except (GuardError, ValueError, TypeError) as exc:
                invalid += 1
                observations.append({"error": str(exc)[:1300], "name": name})
                continue
            # This is the *only* place execution takes place. Calls that fail are logged by ToolEnv.
            try:
                result = env.call(name, args)
            except Exception as exc:
                result = {"ok": False, "error": repr(exc)}
            executed += 1
            last_tool = name
            guard.observe(name, args, result)
            observations.append({"name": name, "arguments": args, "result": short_json(result, 4500)})
            if name == "sys.ask_missing_info":
                break
        return {"transcript": transcript, "elapsed_seconds": round(time.monotonic() - t0, 3),
                "executed": executed, "invalid_proposals": invalid, "schema_inspections": inspections}
