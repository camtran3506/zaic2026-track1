"""Conservative helper-use targets from visible inputs, never future labels.

This planner runs only during SFT preparation. Inference decisions stay with the
main model. It does not fetch new data, invent error trajectories, or inspect the
gold call that will follow a demonstration.
"""
import json
import re
import unicodedata

from .local_helpers import DEFAULT_MAX_LOCAL_CALLS, execute_helper
from .local_knowledge import knowledge_records


SPECIALIST_TRAINING_POLICY = "observable-python-specialists-v1"
MAX_TRAINING_HELPERS = DEFAULT_MAX_LOCAL_CALLS
_ANAPHORA = re.compile(r"^(?:ừ|ờ|rồi|thế|đúng rồi)|\b(?:lúc nãy|hồi nãy|vừa rồi|cái đó|người đó|bác ấy|quán đó)\b")
_EMAIL = re.compile(
    r"(?<![\w@])(?:[a-z0-9_+\-]+(?:\s*(?:chấm|\.)\s*[a-z0-9_+\-]+)*)"
    r"\s*(?:a\s+còng|a\s+cong|@)\s*"
    r"(?:[a-z0-9\-]+\s*(?:chấm|\.)\s*)+(?:test|com|vn|org|net)\b", re.IGNORECASE)
_DIGITS = r"(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|\d)"
_PHONE = re.compile(
    rf"(?:số điện thoại|số)\s+(?P<phone>{_DIGITS}(?:[\s,.-]*{_DIGITS}){{7,14}})"
    rf"(?!\w|[\s,.-]*{_DIGITS}\b)", re.IGNORECASE)
_ENTITY_ROUTES = (
    (r"đánh dấu|đã coi|đã xem", "home_security.acknowledge_alert", "alert_id", "home_security.alerts"),
    (r"gọi", "phone.call", "contact_id", "contacts.contacts"),
    (r"nhắn|gửi tin", "sms.send", "contact_id", "contacts.contacts"),
    (r"dời|đổi lịch", "calendar.update_event", "event_id", "calendar.events"),
    (r"hủy lịch|huỷ lịch", "calendar.cancel_event", "event_id", "calendar.events"),
    (r"bật đèn|tắt đèn", "smart_home.light_set", "device_id", "smart_home.devices"),
    (r"khóa|khoá", "door_lock.set_lock", "lock_id", "door_lock.locks"),
    (r"lưu", "maps.save_place", "place_id", "maps.places"),
)


def _normalize(text):
    text = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    return " ".join(re.findall(r"\w+", "".join(c for c in text if unicodedata.category(c) != "Mn")))


def _present(mention, text):
    mention, text = _normalize(mention), _normalize(text)
    return bool(mention and re.search(r"(?<!\w)" + re.escape(mention) + r"(?!\w)", text))


def _mentions(row):
    for key in ("name", "title", "location"):
        if isinstance(row.get(key), str) and row[key].strip():
            yield row[key]
    message = row.get("message", "")
    if isinstance(message, str):
        match = re.search(r"Khách\s+(.+?)\s+bấm chuông", message, re.IGNORECASE)
        if match:
            yield match[1]


def _known_entity_targets(text, context, history, enabled):
    records = knowledge_records(context, history)
    for pattern, tool, argument, collection in _ENTITY_ROUTES:
        if not re.search(pattern, text) or tool not in enabled:
            continue
        prop = enabled[tool].get("parameters", {}).get("properties", {}).get(argument, {})
        if prop.get("x-ref") != collection:
            continue
        mentions = {mention for entry in records if entry["collection"] == collection
                    for mention in _mentions(entry["record"]) if _present(mention, text)}
        for mention in sorted(mentions):
            yield {"name": "local_entity.resolve", "arguments": {
                "target_tool": tool, "argument": argument, "collection": collection,
                "mention": mention, "max_results": 3}}


def specialist_demonstrations(transcript, context, history, enabled, observations, *, pending=()):
    """Suggest bounded, source-derived local targets independent of future gold.

    Missing or ambiguous source data is never replaced using a label. Schema
    inspection is demonstrated only for a genuinely observed rejected proposal,
    not redundantly after every successful tool call.
    """
    text = transcript.casefold()
    candidates = []
    for match in _EMAIL.finditer(transcript):
        candidates.append({"name": "local_text.normalize", "arguments": {"kind": "email", "text": match.group()}})
    for match in _PHONE.finditer(transcript):
        candidates.append({"name": "local_text.normalize", "arguments": {"kind": "phone", "text": match["phone"]}})
    candidates.extend(_known_entity_targets(text, context, history, enabled))
    if history and len(text) <= 160 and _ANAPHORA.search(text):
        candidates.append({"name": "local_context.lookup", "arguments": {"query": transcript, "max_results": 3}})
    # A literal arithmetic expression is observable; arbitrary prose numbers are
    # not automatically assumed to be amounts that should be summed.
    for match in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?:\s*\+\s*\d+(?:\.\d+)?)+(?!\w|\.\d)", text):
        values = [float(value.strip()) for value in match.group().split("+")]
        candidates.append({"name": "local_checks.evaluate", "arguments": {"operation": "sum", "values": values}})
    if re.search(r"rẻ nhất|rẻ nhứt", text):
        bikes = [entry["record"] for entry in knowledge_records(context, history)
                 if entry["collection"] == "bike_rental.bikes"
                 and entry["record"].get("available") is True
                 and type(entry["record"].get("price_per_hour")) in (int, float)]
        if 1 <= len(bikes) <= 100:
            candidates.append({"name": "local_checks.evaluate", "arguments": {
                "operation": "minimum", "values": [row["price_per_hour"] for row in bikes]}})
    for observation in observations:
        if (observation.get("stage") in ("validate", "precheck")
                and observation.get("name") in enabled
                and isinstance(observation.get("arguments"), dict)
                and "result" not in observation):
            candidates.append({"name": "local_schema.inspect", "arguments": {
                "target_tool": observation["name"], "arguments": observation["arguments"]}})
    previous = {(row.get("name"), json.dumps(row.get("arguments"), ensure_ascii=False, sort_keys=True))
                for row in observations if row.get("scope") == "local"}
    previous.update((row["name"], json.dumps(row["arguments"], ensure_ascii=False, sort_keys=True)) for row in pending)
    remaining = MAX_TRAINING_HELPERS - sum(row.get("scope") == "local" for row in observations) - len(pending)
    selected = []
    for candidate in candidates:
        if len(selected) >= remaining:
            break
        key = (candidate["name"], json.dumps(candidate["arguments"], ensure_ascii=False, sort_keys=True))
        if key in previous:
            continue
        try:
            output = execute_helper(candidate["name"], candidate["arguments"], context, history, enabled, observations)
        except ValueError:
            continue
        if output.get("status") in ("not_found", "unsupported", "invalid_request"):
            continue
        selected.append(candidate)
        previous.add(key)
    return selected
