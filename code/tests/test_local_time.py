from copy import deepcopy

import pytest

from lifemate.local_time import execute_local_tool, local_tool_schemas, with_local_tools


def test_calendar_uses_sample_clock_and_crosses_month_and_year():
    context = {"current_time": "2026-12-31T23:30:00+07:00", "timezone": "Asia/Ho_Chi_Minh"}
    original = deepcopy(context)
    result = execute_local_tool("local_time.calendar_reference", {}, context)
    assert result["relative_dates"]["ngày mai"] == "2027-01-01"
    assert result["weeks"]["tuần này"]["thứ hai"] == "2026-12-28"
    assert result["weeks"]["tuần sau"]["chủ nhật"] == "2027-01-10"
    assert context == original


def test_calendar_converts_zone_before_selecting_day_and_handles_leap_year():
    result = execute_local_tool("local_time.calendar_reference", {}, {
        "current_time": "2024-02-28T18:00:00+00:00", "timezone": "Asia/Ho_Chi_Minh"})
    assert result["relative_dates"]["hôm nay"] == "2024-02-29"
    assert result["relative_dates"]["ngày mai"] == "2024-03-01"
    assert result["as_of"] == "2024-02-29T01:00:00+07:00"


def test_calendar_supports_aware_offset_without_iana_zone():
    result = execute_local_tool("local_time.calendar_reference", {}, {"current_time": "2026-09-25T09:00:00+07:00"})
    assert result["weeks"]["tuần sau"]["thứ sáu"] == "2026-10-02"
    assert result["relative_dates"]["hôm nay"] == "2026-09-25"


def test_departure_subtracts_both_intervals_and_can_cross_midnight():
    result = execute_local_tool("local_time.departure_time", {
        "arrival": "2026-09-22T09:00:00+07:00", "travel_minutes": 15, "early_minutes": 15}, {})
    assert result == {"ok": True, "departure": "2026-09-22T08:30:00+07:00"}
    result = execute_local_tool("local_time.departure_time", {
        "arrival": "2027-01-01T00:10:00+07:00", "travel_minutes": 20, "early_minutes": 5}, {})
    assert result["departure"] == "2026-12-31T23:45:00+07:00"


def test_elapsed_arithmetic_across_dst_uses_real_time_not_wall_clock():
    context = {"timezone": "America/New_York"}
    result = execute_local_tool("local_time.departure_time", {
        "arrival": "2026-03-08T03:30:00-04:00", "travel_minutes": 60, "early_minutes": 0}, context)
    assert result["departure"] == "2026-03-08T01:30:00-05:00"
    assert execute_local_tool("local_time.duration_between", {
        "start": "2026-03-08T01:30:00-05:00", "end": "2026-03-08T03:30:00-04:00"}, context)["minutes"] == 60
    assert execute_local_tool("local_time.duration_between", {
        "start": "2026-11-01T01:30:00-04:00", "end": "2026-11-01T01:30:00-05:00"}, context)["minutes"] == 60


def test_duration_40_minutes_and_explicit_overnight_dates():
    assert execute_local_tool("local_time.duration_between", {
        "start": "2026-10-13T19:30:00+07:00", "end": "2026-10-13T20:10:00+07:00"}, {}) == {"ok": True, "minutes": 40}
    assert execute_local_tool("local_time.duration_between", {
        "start": "2026-10-13T23:50:00+07:00", "end": "2026-10-14T00:10:00+07:00"}, {})["minutes"] == 20


@pytest.mark.parametrize("name,args,context", [
    ("local_time.unknown", {}, {}),
    ("local_time.calendar_reference", [], {}),
    ("local_time.calendar_reference", {"now": "2026-01-01"}, {}),
    ("local_time.calendar_reference", {}, {}),
    ("local_time.calendar_reference", {}, {"current_time": "2026-01-01T12:00:00"}),
    ("local_time.calendar_reference", {}, {"current_time": "bad"}),
    ("local_time.calendar_reference", {}, {"current_time": "2026-01-01T12:00:00Z", "timezone": "bad/zone"}),
    ("local_time.departure_time", {"arrival": "2026-01-01T12:00:00Z", "travel_minutes": True, "early_minutes": 0}, {}),
    ("local_time.departure_time", {"arrival": "2026-01-01T12:00:00Z", "travel_minutes": -1, "early_minutes": 0}, {}),
    ("local_time.departure_time", {"arrival": "2026-01-01T12:00:00Z", "travel_minutes": "15", "early_minutes": 0}, {}),
    ("local_time.departure_time", {"arrival": "2026-01-01T12:00:00Z", "travel_minutes": 10080, "early_minutes": 1}, {}),
    ("local_time.departure_time", {"arrival": "2026-01-01T12:00:00", "travel_minutes": 15, "early_minutes": 0}, {}),
    ("local_time.duration_between", {"start": "2026-01-01T12:00:00Z", "end": "2026-01-01T11:00:00Z"}, {}),
    ("local_time.duration_between", {"start": "2026-01-01T12:00:00Z", "end": "2026-01-01T12:00:30Z"}, {}),
    ("local_time.duration_between", {"start": "2026-01-01T12:00:00Z", "end": "2026-01-09T12:00:00Z"}, {}),
])
def test_invalid_inputs_are_errors_instead_of_guessed_dates(name, args, context):
    with pytest.raises(ValueError):
        execute_local_tool(name, args, context)


def test_registry_does_not_mutate_official_tools_and_returns_independent_schemas():
    official = {"app.read": {"name": "app.read", "kind": "read"}}
    original = deepcopy(official)
    merged = with_local_tools(official)
    assert official == original and len(merged) == 4
    merged["local_time.calendar_reference"]["parameters"]["properties"]["bad"] = {}
    assert local_tool_schemas()["local_time.calendar_reference"]["parameters"]["properties"] == {}
    with pytest.raises(ValueError, match="reserved"):
        with_local_tools({"local_time.calendar_reference": {}})
