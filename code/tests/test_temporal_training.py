"""Helper SFT uses visible operands and preserves official environment replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.sft import replay_sample
from lifemate.temporal_training import helper_demonstrations


CONTEXT = {"current_time": "2026-10-13T20:50:00+07:00", "timezone": "Asia/Ho_Chi_Minh"}
CATALOG = {
    "learning.log_progress": {"name": "learning.log_progress", "kind": "write", "parameters": {
        "type": "object", "properties": {"course_id": {"type": "string"}, "minutes": {"type": "integer"}},
        "required": ["course_id", "minutes"]}},
    "maps.get_route": {"name": "maps.get_route", "kind": "read", "parameters": {
        "type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
        "required": ["origin", "destination"]}},
    "reminders.create": {"name": "reminders.create", "kind": "write", "parameters": {
        "type": "object", "properties": {"content": {"type": "string"}, "time": {"type": "string"}},
        "required": ["content", "time"]}},
    "sys.ask_missing_info": {"name": "sys.ask_missing_info", "parameters": {
        "type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}},
        "required": ["tool_name", "params"]}},
}


class Env:
    def __init__(self):
        self.calls = []

    def validate(self, name, args):
        assert name in CATALOG  # A helper must never touch ToolEnv validation.
        return {"ok": set(CATALOG[name]["parameters"]["required"]) <= set(args)}

    def call(self, name, args):
        assert name in CATALOG
        self.calls.append({"name": name, "arguments": deepcopy(args)})
        if name == "maps.get_route":
            return {"ok": True, "route": {"duration_min": 15}}
        return {"ok": True}

    def get_state_hash(self):
        return json.dumps(self.calls, sort_keys=True)


def entry(transcript, calls, context=None):
    return {"id": "train_fixture", "audio_text": transcript, "history": [],
            "context": {**(context or CONTEXT), "installed_apps": {
                "learning": {"tools": ["learning.log_progress"]},
                "maps": {"tools": ["maps.get_route"]},
                "reminders": {"tools": ["reminders.create"]}}},
            "label": {"tool_calls": calls}}


def targets(examples):
    return [json.loads(row["messages"][-1]["content"]) for row in examples]


def local_targets(examples):
    return [target for target in targets(examples) if target.get("name", "").startswith("local_time.")]


@pytest.mark.parametrize("text", [
    "Tối ni cô học online từ bảy rưỡi tới tám giờ mười, ghi vô tiến độ giùm cô.",
    "Tối nay học từ 19:30 đến 20:10, ghi tiến độ.",
])
def test_duration_demonstration_with_numeric_or_spoken_times(text):
    record = entry(text, [{"name": "learning.log_progress", "arguments": {"course_id": "course_a", "minutes": 40}}])
    env = Env()
    examples = replay_sample(record, CATALOG, env, Path("/unused"))
    helpers = local_targets(examples)
    assert [h["name"] for h in helpers] == ["local_time.calendar_reference", "local_time.duration_between"]
    assert helpers[1]["arguments"] == {"start": "2026-10-13T19:30:00+07:00", "end": "2026-10-13T20:10:00+07:00"}
    env_target = next(row for row in examples if json.loads(row["messages"][-1]["content"]).get("name") == "learning.log_progress")
    observations = json.loads(env_target["messages"][1]["content"])["previous_actions_and_results"]
    assert json.loads(observations[-1]["result"]) == {"ok": True, "minutes": 40}
    assert env.calls == record["label"]["tool_calls"]


def test_future_gold_arguments_do_not_determine_helper_operands_or_prompt():
    record = entry("Tối nay học từ 19:30 tới 20:10.", [
        {"name": "learning.log_progress", "arguments": {"course_id": "secret_course_a", "minutes": 40}}])
    changed = deepcopy(record)
    changed["label"]["tool_calls"][0]["arguments"] = {"course_id": "secret_course_b", "minutes": 999}
    left = replay_sample(record, CATALOG, Env(), Path("/unused"))
    right = replay_sample(changed, CATALOG, Env(), Path("/unused"))
    assert local_targets(left) == local_targets(right)
    assert left[0]["messages"][1] == right[0]["messages"][1]
    assert "secret_course" not in left[0]["messages"][1]["content"]


def test_departure_demonstration_waits_for_observed_route_and_keeps_env_hash():
    record = entry(
        "Chín giờ sáng mai anh phải có mặt ở bệnh viện, xem đi xe máy mất bao lâu rồi nhắc anh đến sớm mười lăm phút.",
        [{"name": "maps.get_route", "arguments": {"origin": "nhà", "destination": "bệnh viện"}},
         {"name": "reminders.create", "arguments": {"content": "Rời nhà", "time": "2026-09-22T08:30:00+07:00"}}],
        {"current_time": "2026-09-21T09:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh"})
    env, baseline = Env(), Env()
    examples = replay_sample(record, CATALOG, env, Path("/unused"))
    names = [target.get("name", "done") for target in targets(examples)]
    assert names == ["local_time.calendar_reference", "maps.get_route", "local_time.departure_time", "reminders.create", "done"]
    departure = next(row for row in examples if json.loads(row["messages"][-1]["content"]).get("name") == "local_time.departure_time")
    args = json.loads(departure["messages"][-1]["content"])["arguments"]
    assert args == {"arrival": "2026-09-22T09:00:00+07:00", "travel_minutes": 15, "early_minutes": 15}
    for call in record["label"]["tool_calls"]:
        baseline.call(call["name"], call["arguments"])
    assert env.get_state_hash() == baseline.get_state_hash()
    assert "08:30" not in departure["messages"][1]["content"]  # It has not been calculated yet.


@pytest.mark.parametrize("text", [
    "Ngày 24 tháng 9 học từ 19:30 tới 20:10.",
    "Ngày mùng ba học từ 19:30 tới 20:10.",
    "Thứ sáu tuần sau học từ 19:30 tới 20:10.",
    "Tối nay học từ 23:30 tới 00:10.",  # Do not infer an unspoken overnight rollover.
    "Sáng học từ 9 giờ tới 10 giờ, tối học từ 19 giờ tới 20 giờ.",
    "Tối qua học từ chín giờ tới mười giờ kém mười lăm.",
])
def test_unsupported_or_ambiguous_arithmetic_is_not_guessed(text):
    result = helper_demonstrations(text, CONTEXT, [], CATALOG, [])
    assert [row["name"] for row in result] == ["local_time.calendar_reference"]


def test_unrelated_history_does_not_add_calendar_to_nontemporal_task():
    assert helper_demonstrations("Bật đèn phòng khách.", CONTEXT,
                                 [{"role": "user", "text": "Ngày mai học lúc 9 giờ."}], CATALOG, []) == []
    assert helper_demonstrations("Ừ, đặt giúp em một vé nhé.", CONTEXT,
                                 [{"role": "user", "text": "Thứ sáu tuần sau bay Đồng Hới."}], CATALOG, [])


def test_absent_anchor_malformed_route_and_repeat_deduplication_are_safe():
    assert helper_demonstrations("Ngày mai học từ 19:30 tới 20:10.", {}, [], CATALOG, []) == []
    text = "Chín giờ sáng mai đến bệnh viện sớm mười lăm phút."
    bad_route = [{"name": "maps.get_route", "result": {"ok": True, "route": None}}]
    assert [row["name"] for row in helper_demonstrations(text, CONTEXT, [], CATALOG, bad_route)] == ["local_time.calendar_reference"]
    local = [{"name": "local_time.calendar_reference", "arguments": {}, "scope": "local", "result": "{}"}]
    assert helper_demonstrations(text, CONTEXT, [], CATALOG, local) == []
    other_helpers = [{'name': 'local_checks.evaluate', 'arguments': {'values': [i]}, 'scope': 'local'} for i in range(40)]
    result = helper_demonstrations('Tối nay học từ 19:30 tới 20:10.', CONTEXT, [], CATALOG, other_helpers)
    assert [r['name'] for r in result] == ['local_time.calendar_reference', 'local_time.duration_between']


@pytest.mark.parametrize("day", ["Mốt", "Hai ngày nữa", "2 ngày nữa"])
def test_day_after_tomorrow_is_not_today(day):
    route = [{"name": "maps.get_route", "result": {"ok": True, "route": {"duration_min": 15}}}]
    result = helper_demonstrations(f"{day} chín giờ sáng có hẹn bệnh viện, nhắc đến sớm mười lăm phút.", CONTEXT, [], CATALOG, route)
    departure = next(row for row in result if row["name"] == "local_time.departure_time")
    assert departure["arguments"]["arrival"] == "2026-10-15T09:00:00+07:00"


@pytest.mark.parametrize("day", ["Ba ngày nữa", "Thứ sáu tuần sau", "Ngày mười sáu tháng mười", "Hôm đó", "Ngày đó", "Bữa đó", "Hôm ấy", "Hôm đấy"])
def test_unsupported_relative_day_does_not_fall_back_to_today(day):
    route = [{"name": "maps.get_route", "result": {"ok": True, "route": {"duration_min": 15}}}]
    result = helper_demonstrations(f"{day} chín giờ sáng có hẹn bệnh viện, nhắc đến sớm mười lăm phút.", CONTEXT, [], CATALOG, route)
    assert all(row["name"] != "local_time.departure_time" for row in result)


def test_alarm_offset_uses_completed_update_result_not_future_gold():
    text = "Dời lịch khám răng mai muộn 60 phút, đặt báo thức trước đó một tiếng."
    assert all(row["name"] != "local_time.departure_time"
               for row in helper_demonstrations(text, CONTEXT, [], CATALOG, []))
    observed = [{"name": "calendar.update_event", "result": json.dumps({
        "ok": True, "item": {"start": "2026-10-14T10:00:00+07:00"}})}]
    result = helper_demonstrations(text, CONTEXT, [], CATALOG, observed)
    departure = next(row for row in result if row["name"] == "local_time.departure_time")
    assert departure["arguments"] == {"arrival": "2026-10-14T10:00:00+07:00", "travel_minutes": 0, "early_minutes": 60}


def test_arrive_at_time_without_early_buffer_uses_explicit_zero():
    text = "Chiều nay xem đường từ nhà tới công viên, nhắc anh đi sao cho bốn giờ có mặt."
    observed = [{"name": "maps.get_route", "result": {"ok": True, "route": {"duration_min": 27}}}]
    result = helper_demonstrations(text, CONTEXT, [], CATALOG, observed)
    departure = next(row for row in result if row["name"] == "local_time.departure_time")
    assert departure["arguments"] == {"arrival": "2026-10-13T16:00:00+07:00", "travel_minutes": 27, "early_minutes": 0}


def test_reschedule_duration_uses_unique_observed_event_before_future_update():
    text = "Dời lịch khám răng mai muộn 60 phút, đặt báo thức trước đó một tiếng."
    first = {"id": "evt_a", "title": "khám răng", "start": "2026-10-14T09:00:00+07:00", "end": "2026-10-14T09:45:00+07:00"}
    other = {"id": "evt_b", "title": "họp team", "start": "2026-10-14T15:00:00+07:00", "end": "2026-10-14T16:00:00+07:00"}
    assert all(row["name"] != "local_time.duration_between" for row in helper_demonstrations(text, CONTEXT, [], CATALOG, []))
    observed = [{"name": "calendar.list_events", "result": json.dumps({"ok": True, "events": [first, other]})}]
    result = helper_demonstrations(text, CONTEXT, [], CATALOG, observed)
    duration = next(row for row in result if row["name"] == "local_time.duration_between")
    assert duration["arguments"] == {"start": first["start"], "end": first["end"]}
    assert all(row["name"] != "local_time.departure_time" for row in result)
    # Another event with the exact same title cannot be selected by guessing ID.
    ambiguous = [{"name": "calendar.list_events", "result": {"ok": True, "events": [first, dict(first, id="evt_c")]}}]
    assert all(row["name"] != "local_time.duration_between" for row in helper_demonstrations(text, CONTEXT, [], CATALOG, ambiguous))
    for bad in (None, {}, {"ok": False, "events": [first]}, {"ok": True, "events": None}):
        observations = [{"name": "calendar.list_events", "result": bad}]
        assert all(row["name"] != "local_time.duration_between" for row in helper_demonstrations(text, CONTEXT, [], CATALOG, observations))
