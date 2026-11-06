"""Conservative SFT helper demonstrations from observable inputs, never labels.

This is not an inference date parser: the model chooses helper calls at runtime.
Preparation only demonstrates calls whose operands can be read unambiguously
from the request and earlier tool responses. Unsupported wording is skipped.
"""
from datetime import datetime, time, timedelta, timezone
import json
import re
from zoneinfo import ZoneInfo

from .local_time import execute_local_tool


TEMPORAL_TRAINING_POLICY = "observable-temporal-helpers-v1"
_UNITS = {"không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
          "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9}
_UNIT_PATTERN = "(?:" + "|".join(_UNITS) + ")"
_NUMBER = rf"(?:\d{{1,2}}|mười(?:\s+{_UNIT_PATTERN})?|{_UNIT_PATTERN}\s+mươi(?:\s+{_UNIT_PATTERN})?|{_UNIT_PATTERN})"
_CLOCK = re.compile(rf"(?<!\w)(?P<hour>{_NUMBER})\s*(?:(?P<half>rưỡi)|(?:giờ|:|h)\s*(?P<minute>{_NUMBER})?)(?!\w)")
_DATE_WORDS = re.compile(r"\b(?:hôm|mai|mốt|ngày|tuần|thứ|chủ nhật)\b")
_EARLY = re.compile(rf"(?:đến|tới)\s+sớm\s+(?P<minutes>{_NUMBER})\s+phút")
_ALARM_BEFORE = re.compile(rf"báo thức trước đó\s+(?P<offset>một tiếng|nửa tiếng|{_NUMBER}\s+phút)")
_TEMPORAL_REQUEST = re.compile(r"nhắc|báo thức|đặt.*(?:vé|chuyến|lịch)|lịch|hẹn|giờ|thời gian|học|tiến độ|đi bộ|chạy bộ|thời tiết|dự báo|mưa")
_TEMPORAL_ENV = {"calendar", "alarm", "reminders", "weather", "flights",
                 "learning", "fitness", "maps", "medication", "tv"}


def _number(value):
    if value.isdigit():
        return int(value)
    words = value.split()
    if words[0] == "mười":
        return 10 + (_UNITS[words[1]] if len(words) == 2 else 0)
    if len(words) > 1 and words[1] == "mươi":
        return _UNITS[words[0]] * 10 + (_UNITS[words[2]] if len(words) == 3 else 0)
    return _UNITS[words[0]]


def _clocks(text):
    clocks = []
    for match in _CLOCK.finditer(text):
        hour = _number(match["hour"])
        minute = 30 if match["half"] else _number(match["minute"]) if match["minute"] else 0
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            return []
        clocks.append((hour, minute, match.start(), match.end()))
    # A global period is only safe when the utterance names one period.
    afternoon = bool(re.search(r"\b(?:chiều|tối)\b", text))
    morning = bool(re.search(r"\bsáng\b", text))
    if afternoon and morning:
        return []
    if afternoon:
        clocks = [(hour + 12 if 1 <= hour < 12 else hour, minute, start, end)
                  for hour, minute, start, end in clocks]
    return clocks


def _request_day(text, today):
    explicit = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    relative = []
    if re.search(r"\b(?:ngày kia|mốt|hai ngày nữa|2 ngày nữa)\b", text):
        relative.append(2)
    if re.search(r"\b(?:ngày|sáng|trưa|chiều|tối) mai\b|^mai\s*[,.:]", text):
        relative.append(1)
    if re.search(r"\b(?:hôm|sáng|trưa|chiều|tối|đêm) qua\b", text):
        relative.append(-1)
    if re.search(r"\bhôm kia\b", text):
        relative.append(-2)
    if len(set(explicit)) > 1 or len(set(relative)) > 1 or (explicit and relative):
        return None
    if explicit:
        return datetime.fromisoformat(explicit[0]).date()
    if relative:
        return today + timedelta(days=relative[0])
    # Do not guess a date for an unsupported explicit date or weekday phrase.
    if re.search(r"\b(?:tuần|thứ|chủ nhật|mai|tháng|kia)\b|\d{1,2}/\d{1,2}|ngày\s+(?:\d|mùng)|ngày nữa|ngày khác|(?:ngày|hôm|bữa)\s+(?:đó|ấy|đấy|trước)", text):
        return None
    if re.search(r"\bngày\b", text) and not re.search(r"\b(?:hôm nay|ngày nay|ngày hôm nay)\b", text):
        return None
    return today


def _at_clock(day, clock, zone):
    result = datetime.combine(day, time(clock[0], clock[1]), tzinfo=zone)
    roundtrip = result.astimezone(timezone.utc).astimezone(zone)
    if roundtrip.replace(tzinfo=None) != result.replace(tzinfo=None):
        return None  # Nonexistent local wall-clock time at a DST transition.
    if result.utcoffset() != result.replace(fold=1).utcoffset():
        return None  # Ambiguous wall time needs an explicit offset from the user.
    return result.isoformat()


def _route_minutes(observations):
    durations = []
    for observation in observations:
        if observation.get("name") != "maps.get_route":
            continue
        result = observation.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                continue
        if not isinstance(result, dict) or result.get("ok") is not True:
            continue
        route = result.get("route")
        if not isinstance(route, dict):
            continue
        duration = route.get("duration_min")
        if type(duration) is int and duration >= 0:
            durations.append(duration)
    return durations[0] if durations and len(set(durations)) == 1 else None


def _updated_event_start(observations):
    starts = set()
    for observation in observations:
        if observation.get("name") != "calendar.update_event":
            continue
        result = observation.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                continue
        if isinstance(result, dict) and result.get("ok") is True and isinstance(result.get("item"), dict):
            start = result["item"].get("start")
            if isinstance(start, str):
                starts.add(start)
    return next(iter(starts)) if len(starts) == 1 else None


def _rescheduled_event_interval(text, observations):
    """Use an observed event only when its full title identifies it uniquely."""
    if not re.search(r"\bdời\b", text):
        return None
    for observation in reversed(observations):
        if observation.get("name") != "calendar.list_events":
            continue
        result = observation.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                return None
        if not isinstance(result, dict) or result.get("ok") is not True or result.get("done") is False:
            return None
        events = result.get("events")
        if not isinstance(events, list):
            return None
        matched = []
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("title"), str):
                continue
            title = " ".join(event["title"].lower().split())
            if title and re.search(r"(?<!\w)" + re.escape(title) + r"(?!\w)", text):
                matched.append(event)
        if len(matched) != 1:
            return None
        event = matched[0]
        if isinstance(event.get("start"), str) and isinstance(event.get("end"), str):
            return {"start": event["start"], "end": event["end"]}
        return None
    return None


def helper_demonstrations(transcript, context, history, enabled, observations):
    """Return eligible helper calls without reading current/future gold actions.

    Idempotence is based on executed local observations; callers invoke this
    before each gold action as more genuine read results become observable.
    """
    if not any(name.split(".")[0] in _TEMPORAL_ENV for name in enabled):
        return []
    text = transcript.lower()
    history_text = " ".join(turn.get("text", "") for turn in history).lower()
    current_reference = _DATE_WORDS.search(text) or _CLOCK.search(text)
    history_reference = _DATE_WORDS.search(history_text) or _CLOCK.search(history_text)
    if not (current_reference or (history_reference and _TEMPORAL_REQUEST.search(text))):
        return []
    try:
        reference = execute_local_tool("local_time.calendar_reference", {}, context)
        current = datetime.fromisoformat(reference["as_of"])
        zone = ZoneInfo(context["timezone"]) if context.get("timezone") else current.tzinfo
        day = _request_day(text, current.date())
        clocks = [] if re.search(r"\bkém\b", text) else _clocks(text)
    except (ValueError, KeyError, TypeError):
        return []
    candidates = [{"name": "local_time.calendar_reference", "arguments": {}}]
    interval = _rescheduled_event_interval(text, observations)
    if interval is not None:
        candidates.append({"name": "local_time.duration_between", "arguments": interval})
    if day is not None and len(clocks) == 2:
        a, b = clocks
        if re.search(r"\btừ\s*$", text[:a[2]]) and re.fullmatch(r"\s*(?:tới|đến|-)\s*", text[a[3]:b[2]]):
            start, end = _at_clock(day, a, zone), _at_clock(day, b, zone)
            if start and end:
                candidates.append({"name": "local_time.duration_between", "arguments": {"start": start, "end": end}})
    early = list(_EARLY.finditer(text))
    travel = _route_minutes(observations)
    on_time = re.search(r"(?:có mặt|tới nơi|đến nơi)", text)
    if day is not None and len(clocks) == 1 and travel is not None and (len(early) == 1 or (not early and on_time)):
        arrival = _at_clock(day, clocks[0], zone)
        if arrival:
            candidates.append({"name": "local_time.departure_time", "arguments": {
                "arrival": arrival, "travel_minutes": travel,
                "early_minutes": _number(early[0]["minutes"]) if early else 0}})
    # A rescheduled event returned by a completed update is observable evidence.
    # Do not take the future alarm target's time as an operand.
    alarm_before = list(_ALARM_BEFORE.finditer(text))
    updated = _updated_event_start(observations)
    if len(alarm_before) == 1 and updated and not re.search(r"cả hai|mỗi lịch|các lịch|hai lịch", text):
        offset = alarm_before[0]["offset"]
        minutes = 60 if offset == "một tiếng" else 30 if offset == "nửa tiếng" else _number(offset.removesuffix(" phút"))
        candidates.append({"name": "local_time.departure_time", "arguments": {
            "arrival": updated, "travel_minutes": 0, "early_minutes": minutes}})
    previous = {(row.get("name"), json.dumps(row.get("arguments"), sort_keys=True, ensure_ascii=False))
                for row in observations if row.get("scope") == "local"}
    result = []
    for candidate in candidates:
        key = (candidate["name"], json.dumps(candidate["arguments"], sort_keys=True, ensure_ascii=False))
        if key in previous:
            continue
        try:
            execute_local_tool(candidate["name"], candidate["arguments"], context)
        except ValueError:
            continue
        result.append(candidate)
        previous.add(key)
    return result
