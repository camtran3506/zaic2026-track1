"""Qwen3.5 native tools/messages and qwen3_coder XML output handling.

The checkpoint's own chat template renders tools and tool calls; no replacement
template or manual XML rendering is used in production.
"""
from copy import deepcopy
import json
import re


NATIVE_PROTOCOL = "qwen35-native-fc-v1"
JSON_PROTOCOL = "lifemate-decision-json-v1"
NATIVE_SYSTEM_PROMPT = """Bạn là AI Agent LifeMate. Hoàn tất yêu cầu tiếng Việt bằng các tools được cung cấp.
- Mỗi lần chỉ gọi MỘT tool bằng định dạng function calling native trong chat template.
- Dùng đúng tên, JSON types/enum/required/x-one-of và toàn bộ schema của tool.
- Với x-ref, lấy ID từ kết quả read tool thành công trong history hoặc lượt hiện tại;
  nếu chưa có thì gọi read tool để tìm ID thật, KHÔNG bịa ID.
- Thực hiện các hành động theo thứ tự phụ thuộc; không thay đổi trạng thái ngoài mục tiêu.
- Dùng transcript, context và toàn bộ history để giải quyết tham chiếu.
- Không gọi read tool lặp vô ích. Không đoán tham số chưa biết.
- Thiếu thông tin bắt buộc không suy ra được: gọi sys.ask_missing_info với
  tool_name và params rồi dừng. Không có app phù hợp thì dừng, không hỏi thêm.
- Khi hoàn tất, không gọi tool nữa và trả đúng {"action":"done"}.
- Không markdown, không diễn giải, không tự nói rằng tool đã được thực thi.
"""


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def tool_call_message(name, arguments, content=""):
    return {"role": "assistant", "content": content, "reasoning_content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": name, "arguments": deepcopy(arguments)}}]}


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
    for observation in payload["previous_actions_and_results"]:
        if all(key in observation for key in ("name", "arguments", "result")):
            chat.append(tool_call_message(observation["name"], observation["arguments"]))
            result = observation["result"]
            chat.append({"role": "tool", "content": result if isinstance(result, str) else _json(result)})
        else:
            # A rejected proposal has no executed tool result to fabricate.
            chat.append({"role": "user", "content": _json({"agent_feedback": observation})})
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


def complete_native_call(text):
    match = _CALL.search(visible_output(text))
    return match.group(0) if match else None


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
    match = _CALL.search(visible)
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
