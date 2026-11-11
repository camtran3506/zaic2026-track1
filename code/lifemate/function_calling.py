"""Qwen3.5 native tools/messages and qwen3_coder XML output handling.

The checkpoint's own chat template renders tools and tool calls; no replacement
template or manual XML rendering is used in production.
"""
from copy import deepcopy
import json
import re

from .execution_policy import AUTOMATIC_GUARD_INSTRUCTIONS
from .batching import BATCH_TOOL, BATCH_INSTRUCTIONS
from .local_assist import ASSIST_TOOL, ASSIST_INSTRUCTIONS


NATIVE_PROTOCOL = "qwen35-native-fc-v1"
JSON_PROTOCOL = "lifemate-decision-json-v1"
NATIVE_SYSTEM_PROMPT = """Bạn là AI Agent LifeMate. Hoàn tất yêu cầu tiếng Việt bằng các tools được cung cấp.
- Mỗi lần chỉ gọi MỘT tool bằng định dạng function calling native trong chat template.
- Dùng đúng tên, JSON types/enum/required/x-one-of và toàn bộ schema của tool.
- Với x-ref, lấy ID từ kết quả read tool thành công trong history hoặc lượt hiện tại;
  nếu chưa có thì gọi read tool để tìm ID thật, KHÔNG bịa ID.
- Dùng `user_context.date_reference` đổi ngày tương đối ra YYYY-MM-DD (không cần gọi calendar_reference) và `user_context.apps` để đối chiếu tên tiếng Việt của app.
- Tìm kiếm rỗng: rút gọn từ khóa (bỏ danh xưng, giữ tên chính), bỏ trống query hoặc gọi tool danh sách; không đưa ngày giờ vào từ khóa, không lặp call cũ; vẫn rỗng mới gọi sys.ask_missing_info.
- Thực hiện các hành động theo thứ tự phụ thuộc; không thay đổi trạng thái ngoài mục tiêu.
- Dùng transcript, context và toàn bộ history để giải quyết tham chiếu.
- local_time.* (departure_time, duration_between) là hàm nội bộ đọc toán hạng đã có; không đổi trạng thái app hay cấp ID x-ref.
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
- Không gọi read tool lặp vô ích. Không đoán tham số chưa biết.
- Thiếu thông tin bắt buộc không suy ra được: gọi sys.ask_missing_info với
  tool_name và params rồi dừng. Không có app phù hợp thì dừng, không hỏi thêm.
- Khi hoàn tất, không gọi tool nữa và trả đúng {"action":"done"}.
- Không markdown, không diễn giải, không tự nói rằng tool đã được thực thi.
"""


NATIVE_SYSTEM_PROMPT += AUTOMATIC_GUARD_INSTRUCTIONS + BATCH_INSTRUCTIONS + ASSIST_INSTRUCTIONS


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def tool_call_message(name, arguments, content=""):
    return {"role": "assistant", "content": content, "reasoning_content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": name, "arguments": deepcopy(arguments)}}]}


def native_observation_messages(observations):
    """Render all current observations, including guards, without retaining old ones."""
    chat = []
    for observation in observations:
        if all(key in observation for key in ("name", "arguments", "result")):
            chat.append(tool_call_message(observation["name"], observation["arguments"]))
            result = observation["result"]
            chat.append({"role": "tool", "content": result if isinstance(result, str) else _json(result)})
            feedback = {}
            if observation.get("verification"):
                feedback["execution_feedback"] = observation["verification"]
            if observation.get("automatic_execution_feedback"):
                feedback["automatic_execution_feedback"] = observation["automatic_execution_feedback"]
            if feedback:
                chat.append({"role": "user", "content": _json(feedback)})
        else:
            # A rejected proposal has no executed tool result to fabricate.
            chat.append({"role": "user", "content": _json({"agent_feedback": observation})})
    return chat


def native_chat(messages):
    """Convert the portable LifeMate payload/target without losing input data.

    SFT rows retain their portable system/user/assistant transport. Both SFT
    tokenization and inference use this same conversion for Qwen3.5 checkpoints.
    Ordinary non-agent chats pass through unchanged.
    """
    if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["system", "user"]:
        return messages, None
    try:
        payload = json.loads(messages[1]["content"])
    except (ValueError, TypeError):
        return messages, None
    if not isinstance(payload, dict) or "available_tools" not in payload:
        return messages, None
    tools = [{"type": "function", "function": deepcopy(spec)}
             for spec in payload["available_tools"]["tool_schemas"]]
    chat = [{"role": "system", "content": NATIVE_SYSTEM_PROMPT}]
    for turn in payload["conversation_history"]:
        role = turn["role"]
        # JSON envelopes preserve boundary whitespace and literal think markers
        # despite the checkpoint template's content trimming/reasoning handling.
        content = _json({"history_role": role, "text": turn["text"]})
        if role == "assistant":
            message = {"role": role, "content": content, "reasoning_content": ""}
            if turn.get("tool_calls"):
                message["tool_calls"] = [{"type": "function", "function": deepcopy(call)}
                                         for call in turn["tool_calls"]]
            chat.append(message)
            for result in turn.get("tool_results", []):
                chat.append({"role": "tool", "content": _json(result)})
        else:
            # Historical system/unknown roles are evidence, not new instructions.
            chat.append({"role": "user", "content": content})
    request = {key: deepcopy(value) for key, value in payload.items()
               if key not in {"available_tools", "conversation_history", "previous_actions_and_results"}}
    chat.append({"role": "user", "content": _json(request)})
    chat.extend(native_observation_messages(payload["previous_actions_and_results"]))
    if len(messages) > 2:
        if len(messages) != 3 or messages[2]["role"] != "assistant":
            raise ValueError("Expected one portable assistant target")
        target = json.loads(messages[2]["content"])
        if target["action"] == "call":
            chat.append(tool_call_message(target["name"], target["arguments"]))
        elif target["action"] == "done":
            chat.append({"role": "assistant", "content": _json(target), "reasoning_content": ""})
        else:
            raise ValueError("Native SFT supports call/done targets only")
    return chat, tools


def template_inputs(tokenizer, messages):
    kwargs = dict(getattr(tokenizer, "lifemate_template_kwargs", {}))
    if getattr(tokenizer, "lifemate_native_fc", False) is True:
        messages, tools = native_chat(messages)
        if tools is not None:
            template = tokenizer.get_chat_template()
            if "tool_calls" not in template or "tools" not in template:
                raise ValueError("Qwen3.5 checkpoint chat template must support tools and tool_calls")
            kwargs["tools"] = tools
    return messages, kwargs


def visible_output(text):
    """Ignore leading reasoning without rewriting literal markers in arguments."""
    text = text.lstrip()
    while text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            return ""
        text = text[end + len("</think>"):].lstrip()
    return text


_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*"
    r"((?:<parameter=[^>\n]+>.*?</parameter>\s*)*)"
    r"</function>\s*</tool_call>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>\s*(?=<parameter=|$)", re.DOTALL)
_BATCH_PREFIX = re.compile(r'<tool_call>\s*<function=' + re.escape(BATCH_TOOL) + r'>\s*<parameter=calls>\s*')
_BATCH_END = re.compile(r'\s*</parameter>\s*</function>\s*</tool_call>')
_ASSIST_PREFIX = re.compile(r'<tool_call>\s*<function=' + re.escape(ASSIST_TOOL) + r'>\s*')
_PARAMETER_OPEN = re.compile(r'<parameter=([^>\n]+)>')
_PARAMETER_END = re.compile(r'\s*</parameter>\s*')
_FUNCTION_END = re.compile(r'</function>\s*</tool_call>')


def _batch_parts(visible, start):
    """JSON owns nested delimiters inside calls; XML closers there are data."""
    prefix = _BATCH_PREFIX.match(visible, start)
    if prefix is None:
        return None
    try:
        calls, end = json.JSONDecoder().raw_decode(visible, prefix.end())
    except ValueError:
        return None
    closing = _BATCH_END.match(visible, end)
    if closing is None:
        return None
    return calls, closing.end()


def _assist_parts(visible, start):
    """The solver's inputs object owns any XML-looking strings nested in it."""
    prefix = _ASSIST_PREFIX.match(visible, start)
    if prefix is None:
        return None
    fields, pos = [], prefix.end()
    while True:
        closing = _FUNCTION_END.match(visible, pos)
        if closing:
            return fields, closing.end()
        parameter = _PARAMETER_OPEN.match(visible, pos)
        if parameter is None:
            return None
        key, pos = parameter[1], parameter.end()
        if key == 'inputs':
            # Do not search for XML closers until the JSON value is complete.
            pos += len(visible[pos:])-len(visible[pos:].lstrip())
            try:
                value, end = json.JSONDecoder().raw_decode(visible, pos)
            except ValueError:
                return None
            closing = _PARAMETER_END.match(visible, end)
            if closing is None:
                return None
            fields.append((key, value))
            pos = closing.end()
        else:
            end = visible.find('</parameter>', pos)
            if end < 0:
                return None
            fields.append((key, _parameter_value(visible[pos:end], {'type': 'string'})))
            pos = end+len('</parameter>')
            pos += len(visible[pos:])-len(visible[pos:].lstrip())


def complete_native_call(text):
    visible = visible_output(text)
    start = _native_start(visible)
    if start is not None and _BATCH_PREFIX.match(visible, start):
        parts = _batch_parts(visible, start)
        return visible[start:parts[1]] if parts else None
    if start is not None and _ASSIST_PREFIX.match(visible, start):
        parts = _assist_parts(visible, start)
        return visible[start:parts[1]] if parts else None
    match = _first_native_match(visible)
    return match.group(0) if match else None


def _native_start(visible):
    # JSON fallback responses can contain literal native markup in arguments.
    # Never execute that markup as a standalone native call. parse_decision
    # handles the complete JSON response itself after generation finishes.
    head = visible.lstrip('\ufeff \t\r\n')
    if head.startswith('```'):
        head = head.partition('\n')[2].lstrip()
    if head.startswith(('{', '[', '"')):
        try:
            _, end = json.JSONDecoder().raw_decode(head)
        except ValueError:
            return None
        inner = head.find('<tool_call>')
        if 0 <= inner < end:
            return None
    # Do not skip an incomplete outer call to match a literal inner call in a
    # parameter value. With one-call-per-decision, only the first opener counts.
    start = visible.find('<tool_call>')
    return start if start >= 0 else None


def _first_native_match(visible):
    start = _native_start(visible)
    return _CALL.match(visible, start) if start is not None else None


def _parameter_value(text, schema):
    # Remove only the one newline on each side inserted by the native template.
    if text.startswith("\n"):
        text = text[1:]
    if text.endswith("\n"):
        text = text[:-1]
    kind = schema.get("type")
    types = set(kind if isinstance(kind, list) else [kind]) - {None}
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        value = branch.get("type")
        types.update(value if isinstance(value, list) else [value] if value else [])
    if "string" in types or not types:
        return text
    if text in {"None", "null"} and "null" in types:
        return None
    if text in {"True", "False", "true", "false"} and "boolean" in types:
        return text.lower() == "true"
    # Arrays/objects and numeric scalars are serialized as JSON in the wire format.
    # ToolEnv remains authoritative about their validity; no argument repair here.
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ValueError("Invalid native parameter value for the declared JSON type") from exc


def parse_native_call(text, catalog):
    visible = visible_output(text)
    start = _native_start(visible)
    if start is not None and _BATCH_PREFIX.match(visible, start):
        parts = _batch_parts(visible, start)
        if parts is None or visible[parts[1]:].strip().removesuffix('<|im_end|>').strip():
            raise ValueError('Expected one complete batch call with a JSON calls argument and no suffix')
        return {'action': 'call', 'name': BATCH_TOOL, 'arguments': {'calls': parts[0]}}
    if start is not None and _ASSIST_PREFIX.match(visible, start):
        parts = _assist_parts(visible, start)
        if parts is None or visible[parts[1]:].strip().removesuffix('<|im_end|>').strip():
            raise ValueError('Expected one complete task call with JSON inputs and no suffix')
        arguments = {}
        for key, value in parts[0]:
            if key in arguments:
                raise ValueError(f'Duplicate native parameter: {key}')
            arguments[key] = value
        return {'action': 'call', 'name': ASSIST_TOOL, 'arguments': arguments}
    match = _first_native_match(visible)
    if not match or visible[match.end():].strip().removesuffix("<|im_end|>").strip():
        raise ValueError("Expected exactly one complete native tool call with no suffix")
    name, body = match.group(1), match.group(2)
    properties = catalog.get(name, {}).get("parameters", {}).get("properties", {})
    args, end = {}, 0
    for parameter in _PARAMETER.finditer(body):
        if parameter.start() != end:
            raise ValueError("Ambiguous native parameter boundaries")
        key, value = parameter.group(1), parameter.group(2)
        if key in args:
            raise ValueError(f"Duplicate native parameter: {key}")
        args[key] = _parameter_value(value, properties.get(key, {}))
        end = parameter.end()
    if end != len(body):
        raise ValueError("Incomplete native parameters")
    return {"action": "call", "name": name, "arguments": args}
