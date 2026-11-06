"""Observe -> plan a call/read-only batch -> validate -> execute -> observe."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import allowed_tools
from .eval_trace import exception_details
from .history import prompt_history, resolve_history
from .local_helpers import execute_helper, helper_tool_schemas, with_local_tools
from .local_assist import ASSIST_INSTRUCTIONS
from .local_checks import inspect_call
from .execution_policy import AUTOMATIC_GUARD_INSTRUCTIONS, LoopMonitor
from .tool_execution import precheck_call, attempt_env_call, result_observation
from .text import normalize_transcript
from .prompt_session import InferenceSession, fixed_prompt_payload
from .batching import (BATCH_TOOL, BATCH_INSTRUCTIONS, batch_error, batch_observation,
                       observation_children, successful)

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
- local_text.normalize chuẩn hóa email/số điện thoại/số tiền; local_checks.evaluate tính/so sánh
  các toán hạng đã biết. local_schema.inspect chỉ góp ý schema, không thay thế ToolEnv.
- itn_candidates và itn_history_candidates là các giá trị Python đã chuẩn hóa, kèm đoạn nguồn và vị trí.
  Dùng ngay khi phù hợp, không gọi lại helper cho cùng chuyển đổi đã có. Transcript vẫn quyết định
  ý định/phủ định và chọn người nhận; ITN không cấp phép thực thi. Với đoạn chưa được ITN xử lý,
  có thể gọi local_text.normalize theo kind email/phone/money; không tự đoán phần còn thiếu.
- Helper ambiguous/unsupported không cho phép tự bịa giá trị hoặc mặc định done.
  Khi kết quả tool thất bại/rỗng, xem phản hồi, sửa kế hoạch; không lặp nguyên call vô ích.
- Đừng thay đổi trạng thái ngoài mục tiêu của người dùng; đừng gọi read tools lặp vô ích.
- Nếu thiếu thông tin bắt buộc không suy ra được qua context/read tools, gọi
  sys.ask_missing_info với {"tool_name":"tool dự kiến", "params":["tham số còn thiếu"]}, rồi dừng.
- Nếu không ứng dụng được cài nào có khả năng đáp ứng yêu cầu: xuất {"action":"done"} ngay, KHÔNG gọi ask_missing_info.
- Nếu đã hoàn tất: {"action":"done"}. Không tự nói là đã gọi tool.
- Arguments must be a JSON object. Output exactly one JSON object.
"""


SYSTEM_PROMPT += AUTOMATIC_GUARD_INSTRUCTIONS + BATCH_INSTRUCTIONS + ASSIST_INSTRUCTIONS


def parse_decision(raw: str, catalog=None) -> dict:
    """Find first JSON object, including model outputs with harmless surrounding text."""
    from .function_calling import parse_native_call, visible_output
    raw = visible_output(raw)
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(raw):
        char = raw[idx]
        if raw.startswith("<tool_call>", idx):
            return parse_native_call(raw[idx:], catalog or {})
        if char in '["':
            try:
                _, end = decoder.raw_decode(raw, idx)
            except json.JSONDecodeError as exc:
                raise ValueError('Incomplete JSON envelope; do not execute nested calls') from exc
            if char == '[':
                raise ValueError('Use local_batch.run, not a top-level call array')
            idx = end
            continue
        if char != "{":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError as exc:
            raise ValueError('Incomplete JSON envelope; do not execute nested calls') from exc
        idx = end
        if not isinstance(obj, dict):
            continue
        if "tool_calls" in obj and isinstance(obj["tool_calls"], list) and obj["tool_calls"]:
            if len(obj['tool_calls']) != 1:
                raise ValueError('Use local_batch.run for independent helpers/reads; do not emit multiple top-level calls')
            obj = obj["tool_calls"][0]
            if not isinstance(obj, dict):
                raise ValueError('Tool call must be a JSON object')
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
    # Only env/invalid planning decisions spend this budget. Valid helper-only
    # turns have no count ceiling; model input/output token limits still apply.
    max_total_steps: int = 32


def build_step_prompt(transcript, context, history, enabled, observations):
    """Use the same complete catalog, history and observations in SFT/inference."""
    tools, info = fixed_prompt_payload(transcript, context, history, enabled)
    info["previous_actions_and_results"] = observations
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
        begin = getattr(self.llm, "begin_sample", None)
        end = getattr(self.llm, "end_sample", None)
        session = InferenceSession() if getattr(self.llm, "prompt_cache_mode", "auto") == "auto" else None
        if callable(begin):
            begin()
        try:
            report = self._run(item, env, split_dir, split, prepared_transcript, trace=trace, session=session)
            stats = getattr(self.llm, "prefix_cache_stats", None)
            if callable(stats):
                report["prefix_cache"] = stats()
            if session is not None:
                report["prompt_cache"] = session.report()
            return report
        finally:
            if session is not None:
                session.close()
            if callable(end):
                end()

    def _run(self, item: dict, env: Any, split_dir: Path, split: str,
             prepared_transcript: dict | None = None, *, trace=None, session=None) -> dict:
        t0 = time.monotonic()
        step = 0
        non_helper_steps = 0
        batch_child_index = None
        def emit(event, **data):
            if trace is not None:
                if batch_child_index is not None:
                    data['batch_child_index'] = batch_child_index
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
                    'llm_steps': 0, 'non_helper_steps': 0, 'batches': 0,
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
        if session is not None:
            session.prepare(transcript, context, history, enabled)
        observations: list[dict] = []
        invalid = 0
        executed = 0
        local_executed = 0
        batches = 0
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

        def execute_one(name, args, *, in_batch=False):
            nonlocal executed, local_executed, stop_reason
            rejection = precheck_call(name, args, self.catalog, enabled, monitor, local_names)
            if rejection:
                if rejection['stage'] == 'loop_guard':
                    emit('loop_advisory', name=name, arguments=args, feedback=rejection['feedback'])
                reject(**rejection)
                return False
            if name in local_names:
                local_executed += 1
                emit('local_call_start', name=name, arguments=args, local_call_index=local_executed-1)
                try:
                    result = execute_helper(name, args, context, history, enabled,
                                            observation_children(observations), trace=emit)
                    if result.get('status') == 'invalid_request':
                        reject(str(result.get('warnings', 'Invalid helper request')), 'local_validate', name=name, arguments=args)
                except ValueError as exc:
                    result = {'ok': False, 'error': str(exc)}
                    reject(str(exc), 'local_validate', name=name, arguments=args)
                emit('local_call_result', name=name, arguments=args, result=result, local_call_index=local_executed-1)
                observation = {'name': name, 'arguments': args, 'result': full_json(result), 'scope': 'local'}
                observations.append(observation)
                return successful(observation)
            advisory = inspect_call(name, args, enabled)
            emit('schema_inspection', name=name, inspection=advisory)
            outcome = attempt_env_call(env, name, args, enabled[name].get('kind', 'unknown'), monitor, executed, emit=emit)
            if not outcome['executed']:
                reject(**outcome['feedback'])
                return False
            executed += 1
            if not in_batch:
                observations[:] = [o for o in observations if 'result' in o]
            observation = result_observation(name, args, outcome, full_json)
            observations.append(observation)
            if name == 'sys.ask_missing_info':
                stop_reason = 'ask_missing_info'
            return successful(observation)

        while (executed < self.cfg.max_calls and invalid < self.cfg.max_invalid
               and non_helper_steps < self.cfg.max_total_steps):
            if time.monotonic() - t0 > self.cfg.sample_seconds - self.cfg.reserve_seconds:
                stop_reason = "time_limit"
                break
            if session is not None:
                user_prompt = session.prompt(observations)
            else:
                _, user_prompt = build_step_prompt(transcript, context, history, enabled, observations)
            step += 1
            non_helper_steps += 1
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
            if name == BATCH_TOOL:
                payload = json.loads(user_prompt)
                error = batch_error(args, planner_tools, local_names, payload,
                    env_remaining=self.cfg.max_calls-executed)
                if error:
                    reject(error, 'batch_precheck', name=name, arguments=args)
                    continue
                # All access/repetition guards run before executing any child.
                rejection = next((r for c in args['calls'] if
                    (r := precheck_call(c['name'], c['arguments'], self.catalog, enabled, monitor, local_names))), None)
                if rejection:
                    reject(**rejection)
                    continue
                batches += 1
                start, before_executed, before_invalid, results = len(observations), executed, invalid, []
                emit('batch_start', calls=args['calls'])
                for batch_child_index, child in enumerate(args['calls']):
                    if time.monotonic()-t0 > self.cfg.sample_seconds-self.cfg.reserve_seconds:
                        stop_reason = 'time_limit'
                        break
                    child_start = len(observations)
                    ok = execute_one(child['name'], child['arguments'], in_batch=True)
                    child_observations = observations[child_start:]
                    result = dict(child_observations[-1])
                    if len(child_observations) > 1:
                        result['agent_feedback'] = child_observations[:-1]
                    results.append(result)
                    if not ok:
                        break
                batch_child_index = None
                del observations[start:]
                if executed > before_executed:
                    observations[:] = [o for o in observations if 'result' in o]
                response = batch_observation(args['calls'], results)
                observations.append(response)
                if invalid == before_invalid and all(c['name'] in local_names for c in args['calls']):
                    non_helper_steps -= 1
                emit('batch_result', response=response)
                if stop_reason is not None:
                    break
                continue
            before_invalid = invalid
            execute_one(name, args)
            if isinstance(name, str) and name in local_names and invalid == before_invalid:
                non_helper_steps -= 1
            if stop_reason is not None:
                break
        if stop_reason is None:
            stop_reason = ("max_calls" if executed >= self.cfg.max_calls else
                           "max_invalid" if invalid >= self.cfg.max_invalid else "max_total_steps")
        emit("stop", reason=stop_reason, executed=executed, local_executed=local_executed,
             invalid_proposals=invalid, batches=batches, non_helper_steps=non_helper_steps)
        return {"transcript": transcript, "elapsed_seconds": round(time.monotonic() - t0, 3),
                "executed": executed, "local_executed": local_executed, "invalid_proposals": invalid,
                'llm_steps': step, 'non_helper_steps': non_helper_steps, 'batches': batches,
                "stop_reason": stop_reason}
