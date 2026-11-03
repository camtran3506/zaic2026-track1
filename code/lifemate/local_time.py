"""Deterministic, offline date/time tools; never interact with ToolEnv state."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


LOCAL_TIME_POLICY = "local-time-tools-v1"
_MAX_MINUTES = 7 * 24 * 60
_WEEKDAYS = ("thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật")


def _spec(name, description, properties, required):
    return {"name": name, "kind": "local", "description": description,
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False}}


_TIMESTAMP = {"type": "string", "description": "ISO-8601 có ngày, giờ và múi giờ, ví dụ 2026-09-22T09:00:00+07:00"}
_MINUTES = {"type": "integer", "minimum": 0, "maximum": _MAX_MINUTES,
            "description": "Số phút nguyên không âm, tối đa 10080 (7 ngày)"}
_TOOLS = {
    "local_time.calendar_reference": _spec(
        "local_time.calendar_reference",
        "Tra hôm qua/hôm nay/ngày mai/ngày kia và lịch tuần trước/này/sau. "
        "Mốc thời gian và múi giờ lấy từ context của yêu cầu hiện tại, không phải giờ máy. "
        "Chỉ là dữ liệu tính toán nội bộ; không thực hiện hành động ứng dụng.", {}, []),
    "local_time.departure_time": _spec(
        "local_time.departure_time",
        "Tính giờ xuất phát = giờ hẹn - thời gian đi - khoảng đến sớm. "
        "Dùng đúng các khoảng phút từ yêu cầu và kết quả tool đã biết; không đoán. "
        "Chỉ tính toán, không tự tạo nhắc việc hay thay đổi trạng thái ứng dụng.",
        {"arrival": _TIMESTAMP, "travel_minutes": _MINUTES, "early_minutes": _MINUTES},
        ["arrival", "travel_minutes", "early_minutes"]),
    "local_time.duration_between": _spec(
        "local_time.duration_between",
        "Tính số phút thực tế từ start đến end (0 đến 7 ngày), hai mốc phải có múi giờ. "
        "Qua nửa đêm phải truyền đúng ngày của end; không tự suy đoán sang ngày hôm sau. "
        "Chỉ tính toán, không ghi tiến độ hoặc thay đổi trạng thái ứng dụng.",
        {"start": _TIMESTAMP, "end": _TIMESTAMP}, ["start", "end"]),
}


def local_tool_schemas():
    """Return independent schemas suitable for both prompts and native parsing."""
    return deepcopy(_TOOLS)


def with_local_tools(env_tools):
    """Keep official tools intact and reject collisions with the reserved namespace."""
    if any(name.startswith("local_time.") for name in env_tools):
        raise ValueError("ToolEnv catalog collides with reserved local_time namespace")
    return {**env_tools, **local_tool_schemas()}


def _timestamp(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 timestamp with timezone")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp with timezone") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit timezone offset")
    return result


def _context_zone(context, fallback):
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    zone_name = context.get("timezone")
    if zone_name is None:
        return fallback
    if not isinstance(zone_name, str) or not zone_name:
        raise ValueError("context.timezone must be a valid IANA timezone")
    try:
        return ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError("context.timezone must be a valid IANA timezone") from exc


def _checked_arguments(name, arguments):
    if not isinstance(name, str) or name not in _TOOLS:
        raise ValueError(f"Unknown local time tool: {name}")
    if not isinstance(arguments, dict):
        raise ValueError("Local time arguments must be an object")
    parameters = _TOOLS[name]["parameters"]
    missing = set(parameters["required"]) - arguments.keys()
    unexpected = arguments.keys() - parameters["properties"].keys()
    if missing or unexpected:
        raise ValueError(f"Invalid local time arguments: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in arguments.items():
        schema = parameters["properties"][key]
        if schema["type"] == "integer" and (type(value) is not int or not 0 <= value <= _MAX_MINUTES):
            raise ValueError(f"{key} must be an integer between 0 and {_MAX_MINUTES}")


def execute_local_tool(name, arguments, context):
    """Execute a whitelisted pure helper, raising ValueError on invalid inputs.

    Arithmetic uses UTC elapsed time, then renders in the sample timezone. This
    avoids wall-clock subtraction errors across daylight saving transitions.
    Dates in the calendar use local date arithmetic instead of fixed 24h shifts.
    """
    _checked_arguments(name, arguments)
    if not isinstance(context, dict):
        raise ValueError("context must be an object")
    try:
        if name == "local_time.calendar_reference":
            current = _timestamp(context.get("current_time"), "context.current_time")
            zone = _context_zone(context, current.tzinfo)
            current = current.astimezone(zone)
            today = current.date()
            monday = today - timedelta(days=today.weekday())
            return {"ok": True, "as_of": current.isoformat(),
                    "timezone": context.get("timezone") or current.strftime("%z"),
                    "week_starts_on": _WEEKDAYS[0],
                    "relative_dates": {label: (today + timedelta(days=days)).isoformat()
                                       for label, days in (("hôm qua", -1), ("hôm nay", 0),
                                                           ("ngày mai", 1), ("ngày kia", 2))},
                    "weeks": {label: {weekday: (monday + timedelta(days=offset + index)).isoformat()
                                      for index, weekday in enumerate(_WEEKDAYS)}
                              for label, offset in (("tuần trước", -7), ("tuần này", 0), ("tuần sau", 7))}}
        if name == "local_time.departure_time":
            arrival = _timestamp(arguments["arrival"], "arrival")
            zone = _context_zone(context, arrival.tzinfo)
            minutes = arguments["travel_minutes"] + arguments["early_minutes"]
            if minutes > _MAX_MINUTES:
                raise ValueError("Total travel and early-arrival minutes must not exceed 10080")
            departure = arrival.astimezone(timezone.utc) - timedelta(minutes=minutes)
            return {"ok": True, "departure": departure.astimezone(zone).isoformat()}
        start = _timestamp(arguments["start"], "start")
        end = _timestamp(arguments["end"], "end")
        _context_zone(context, start.tzinfo)  # Validate context consistently, even without date rendering.
        seconds = (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()
        if seconds < 0 or seconds > _MAX_MINUTES * 60 or seconds % 60:
            raise ValueError("Interval must be 0 to 10080 whole elapsed minutes; supply correct dates across midnight")
        return {"ok": True, "minutes": int(seconds // 60)}
    except (OverflowError, OSError) as exc:
        raise ValueError("Date/time calculation is outside the supported calendar range") from exc
