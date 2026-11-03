"""Observe -> plan a single tool call -> validate -> execute -> observe."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import allowed_tools, prompt_catalog
from .eval_trace import exception_details
from .history import prompt_history, resolve_history
from .local_helpers import (DEFAULT_MAX_LOCAL_CALLS, execute_helper,
                            helper_tool_schemas, with_local_tools)
from .local_checks import inspect_call
from .execution_policy import LoopMonitor
from .text import normalize_transcript

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
- local_time.* là các hàm Python nội bộ: dùng calendar_reference để tra ngày,
  departure_time để trừ thời gian đi/đến sớm, duration_between để tính số phút.
  Lấy toán hạng từ yêu cầu/context/kết quả đã có; đọc kết quả rồi mới gọi tool ứng dụng.
  Các helper thời gian không đổi trạng thái ứng dụng, không cung cấp ID x-ref, không tự hoàn tất tác vụ.
- local_context.lookup và local_entity.resolve chỉ tra context/history đã có; dùng nguồn
  được trả về để đối chiếu đúng collection. Không tìm thấy không chứng minh entity không tồn tại.
- local_text.normalize chuẩn hóa email/số điện thoại; local_checks.evaluate tính/so sánh
  các toán hạng đã biết. local_schema.inspect chỉ góp ý schema, không thay thế ToolEnv.
- Helper ambiguous/unsupported không cho phép tự bịa giá trị hoặc mặc định done.
  Khi kết quả tool thất bại/rỗng, xem phản hồi, sửa kế hoạch; không lặp nguyên call vô ích.
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
        if obj.get("action") == "call" or ("action" not in obj and "name" in obj and "arguments" in obj):
            return {"action": "call", "name": obj.get("name"), "arguments": obj.get("arguments")}
        if "action" in obj:
            raise ValueError(f"Unsupported agent action: {obj['action']!r}")
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
    max_local_calls: int = DEFAULT_MAX_LOCAL_CALLS
    max_total_steps: int = 32


def build_step_prompt(transcript, context, history, enabled, observations):
    """Use the same complete catalog, history and observations in SFT/inference."""
    tools = prompt_catalog(with_local_tools(enabled))
    info = {
        "request_transcript": normalize_transcript(transcript),
        "user_context": {k: v for k, v in context.items() if k != "installed_apps"},
        "conversation_history": prompt_history(history),
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
            prepared_transcript: dict | None = None, *, trace=None) -> dict:
        """env is created exactly once by caller; never instantiate another here."""
        t0 = time.monotonic()
        step = 0
        def emit(event, **data):
            if trace is not None:
                trace({"event": event, "step": step, **data})

        context = item.get("context", {})
        enabled = allowed_tools(self.catalog, context.get("installed_apps", {}))
        planner_tools = with_local_tools(enabled)
        local_names = helper_tool_schemas().keys()
        monitor = LoopMonitor()
        if len(enabled) == 1:  # only sys.ask_missing_info; no app can act
            emit("stop", reason="no_available_tools", executed=0, invalid_proposals=0)
            return {"transcript": normalize_transcript(prepared_transcript["transcript"]) if prepared_transcript else "",
                    "elapsed_seconds": 0, "executed": 0, "invalid_proposals": 0,
                    "local_executed": 0,
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
        transcript = normalize_transcript(transcript)
        observations: list[dict] = []
        invalid = 0
        executed = 0
        local_executed = 0
        stop_reason = None

        def reject(error, stage, **feedback):
            nonlocal invalid
            invalid += 1
            # Keep rejected proposals available until a call is executed.
            # A rejection is feedback, not an executed call:
            # deliberately avoid the `result` key used for actual tool results.
            observation = {"error": error, "stage": stage, **feedback}
            observations.append(observation)
            emit("decision_rejected", **observation)

        while (executed < self.cfg.max_calls and invalid < self.cfg.max_invalid
               and step < self.cfg.max_total_steps):
            if time.monotonic() - t0 > self.cfg.sample_seconds - self.cfg.reserve_seconds:
                stop_reason = "time_limit"
                break
            tools, user_prompt = build_step_prompt(
                transcript, context, history, enabled, observations)
            step += 1
            emit("llm_start", request={"system": SYSTEM_PROMPT, "user": user_prompt})
            try:
                raw = self.llm.generate(SYSTEM_PROMPT, user_prompt)
            except Exception as exc:
                emit("llm_error", exception=exception_details(exc))
                raise
            emit("llm_output", output=raw)
            try:
                decision = parse_decision(raw, planner_tools)
            except ValueError as exc:
                reject(str(exc), "parse", raw_output=raw,
                       instruction="Trả về quyết định đúng định dạng đã được hướng dẫn bởi system prompt và chat template; không markdown hoặc diễn giải.")
                continue
            emit("decision_parsed", decision=decision)
            if decision["action"] == "done":
                warnings = monitor.pending_warnings()
                if warnings:
                    emit("completion_advisory", warnings=warnings)
                stop_reason = "done"
                break
            name, args = decision.get("name"), decision.get("arguments")
            if not isinstance(name, str) or name not in planner_tools or name not in {x["name"] for x in tools["tool_schemas"]}:
                reject(f"Tool {name!r} not exposed in tool_schemas; choose only from listed schemas.",
                       "precheck", name=name, arguments=args)
                continue
            if not isinstance(args, dict):
                reject("arguments must be a JSON object", "precheck", name=name, arguments=args)
                continue
            if name in local_names:
                if local_executed >= self.cfg.max_local_calls:
                    reject("Local helper budget exhausted; use existing results or an available application tool.",
                           "local_budget", name=name, arguments=args)
                    continue
                local_executed += 1
                emit("local_call_start", name=name, arguments=args, local_call_index=local_executed - 1)
                try:
                    result = execute_helper(name, args, context, history, enabled, observations)
                    if result.get("status") == "invalid_request":
                        reject(str(result.get("warnings", "Invalid helper request")),
                               "local_validate", name=name, arguments=args)
                except ValueError as exc:
                    result = {"ok": False, "error": str(exc)}
                    reject(str(exc), "local_validate", name=name, arguments=args)
                emit("local_call_result", name=name, arguments=args, result=result,
                     local_call_index=local_executed - 1)
                observations.append({"name": name, "arguments": args,
                                     "result": full_json(result), "scope": "local"})
                # Helpers cannot call ToolEnv. Entity IDs only reference existing evidence.
                continue
            if name == "sys.ask_missing_info":
                if args.get("tool_name") not in enabled or args.get("tool_name") == name:
                    reject("Missing-info tool_name must be another available tool.",
                           "precheck", name=name, arguments=args)
                    continue
            advisory = inspect_call(name, args, enabled)
            emit("schema_inspection", name=name, inspection=advisory)
            feedback = monitor.check(name, args, enabled[name].get("kind", "unknown"))
            if feedback:
                emit("loop_advisory", name=name, arguments=args, feedback=feedback)
                if feedback.get("block"):
                    reject(feedback["instruction"], "loop_guard", name=name, arguments=args,
                           feedback=feedback)
                    continue
            emit("validate_start", name=name, arguments=args)
            try:
                validation = env.validate(name, args)
            except Exception as exc:
                emit("validate_error", name=name, arguments=args, exception=exception_details(exc))
                reject(str(exc), "validate", name=name, arguments=args, exception=exception_details(exc))
                continue
            accepted = isinstance(validation, dict) and bool(validation.get("ok"))
            emit("validate_result", name=name, arguments=args, result=validation, accepted=accepted)
            if not accepted:
                monitor.record_validation_failure(name, args, validation)
                reject(f"Schema invalid: {validation!r}", "validate", name=name,
                       arguments=args, validation=validation)
                continue
            # This is the *only* place execution takes place. Calls that fail are logged by ToolEnv.
            emit("call_start", name=name, arguments=args, call_index=executed)
            try:
                result = env.call(name, args)
            except Exception as exc:
                result = {"ok": False, "error": repr(exc)}
                emit("call_error", name=name, arguments=args, call_index=executed,
                     exception=exception_details(exc), agent_result=result)
            else:
                emit("call_result", name=name, arguments=args, call_index=executed, result=result)
            executed += 1
            verification = monitor.record_result(name, args, result, enabled[name].get("kind", "unknown"))
            emit("execution_verified", name=name, verification=verification)
            # Move on using actual calls/results once validation led to execution.
            # Executed failures remain evidence; only unexecuted feedback expires.
            # The debug trace and cumulative invalid counter retain earlier errors.
            observations[:] = [observation for observation in observations if "result" in observation]
            observation = {"name": name, "arguments": args, "result": full_json(result)}
            if verification["status"] in ("failure", "empty"):
                observation["verification"] = verification
            observations.append(observation)
            if name == "sys.ask_missing_info":
                stop_reason = "ask_missing_info"
                break
        if stop_reason is None:
            stop_reason = ("max_calls" if executed >= self.cfg.max_calls else
                           "max_invalid" if invalid >= self.cfg.max_invalid else "max_total_steps")
        emit("stop", reason=stop_reason, executed=executed, local_executed=local_executed,
             invalid_proposals=invalid)
        return {"transcript": transcript, "elapsed_seconds": round(time.monotonic() - t0, 3),
                "executed": executed, "local_executed": local_executed, "invalid_proposals": invalid,
                "stop_reason": stop_reason}
