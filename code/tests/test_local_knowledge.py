"""Entity/history specialists preserve provenance and never call external tools."""
from copy import deepcopy

import pytest

from lifemate.local_knowledge import execute, knowledge_records, schemas


TOOLS = {
    "home_security.acknowledge_alert": {"parameters": {"properties": {
        "alert_id": {"type": "string", "x-ref": "home_security.alerts"}}, "required": ["alert_id"]}},
    "contacts.search": {"parameters": {"properties": {"query": {"type": "string"}}}},
    "phone.call": {"parameters": {"properties": {
        "contact_id": {"type": "string", "x-ref": "contacts.contacts"}}}},
    "loyalty.redeem": {"parameters": {"properties": {
        "reward_id": {"type": "string", "x-ref": "loyalty.rewards"}}}},
}


def history_read(name, field, rows):
    return [{"role": "assistant", "text": "Read result",
             "tool_calls": [{"name": name, "arguments": {}}],
             "tool_results": [{"ok": True, field: rows}]}]


def resolve(history, mention, collection="home_security.alerts", target="home_security.acknowledge_alert", arg="alert_id", context=None):
    return execute("local_entity.resolve", {"mention": mention, "collection": collection,
        "target_tool": target, "argument": arg}, context or {}, history, TOOLS)


def test_exact_entity_and_door_are_scoped_to_alert_collection():
    history = history_read("home_security.list_alerts", "alerts", [
        {"id": "alert_hale", "message": "Khách Peter Hale bấm chuông", "location": "cổng"},
        {"id": "alert_hall", "message": "Khách Peter Hall bấm chuông", "location": "cổng"},
        {"id": "alert_door", "message": "Cửa sau mở quá 10 phút", "location": "cửa sau"},
    ]) + history_read("door_lock.list_locks", "locks", [{"id": "lock_door", "name": "cửa sau"}])
    original = deepcopy(history)
    result = resolve(history, "peter hale")
    assert result["status"] == "matched"
    assert result["candidates"][0]["value"] == "alert_hale"
    result = resolve(history, "CUA SAU")
    assert result["status"] == "matched"
    assert [r["value"] for r in result["candidates"]] == ["alert_door"]
    assert result["candidates"][0]["evidence"] == {
        "source": "history", "tool": "home_security.list_alerts", "path": "history[0].tool_results[0].alerts[2]"}
    assert history == original


def test_close_name_competitors_are_ambiguous_but_exact_name_wins():
    history = history_read("contacts.search", "contacts", [
        {"id": "c_1", "name": "Michael Chen", "relation": "bạn trao đổi"},
        {"id": "c_2", "name": "Michael Cheng", "relation": "bạn cùng lớp"},
    ])
    result = resolve(history, "Michael Che", "contacts.contacts", "phone.call", "contact_id")
    assert result["status"] == "ambiguous" and len(result["candidates"]) == 2
    result = resolve(history, "Michael Chen", "contacts.contacts", "phone.call", "contact_id")
    assert result["status"] == "matched" and [r["value"] for r in result["candidates"]] == ["c_1"]


def test_missing_collection_never_uses_assistant_claims_or_schema_examples_for_ids():
    history = [{"role": "assistant", "text": "Bà Hồng có ID c_fake"}]
    result = resolve(history, "Bà Hồng", "contacts.contacts", "phone.call", "contact_id")
    assert result["status"] == "not_found" and result["candidates"] == []
    context = {"installed_apps": {"contacts": {"example": {"id": "c_fake", "name": "Bà Hồng"}}},
               "label": {"contacts": [{"id": "c_fake", "name": "Bà Hồng"}]}}
    assert resolve(history, "Bà Hồng", "contacts.contacts", "phone.call", "contact_id", context)["status"] == "not_found"


def test_failed_malformed_and_nonread_results_do_not_become_grounded_entities():
    history = history_read("contacts.search", "contacts", [{"id": "c_fake", "name": "Bà Hồng"}])
    for result in [{"ok": False, "contacts": [{"id": "c_fake", "name": "Bà Hồng"}]},
                   {"ok": True, "done": False, "contacts": [{"id": "c_fake", "name": "Bà Hồng"}]}]:
        history[0]["tool_results"] = [result]
        assert resolve(history, "Bà Hồng", "contacts.contacts", "phone.call", "contact_id")["status"] == "not_found"
    history = history_read("contacts.update", "contacts", [{"id": "c_fake", "name": "Bà Hồng"}])
    assert knowledge_records({}, history) == []
    history[0]["tool_calls"].append({"name": "contacts.search", "arguments": {}})
    assert knowledge_records({}, history) == []


def test_actual_schema_ref_and_target_tool_are_authoritative():
    history = history_read("contacts.search", "contacts", [{"id": "c_1", "name": "Hồng"}])
    assert resolve(history, "Hồng", "contacts.contacts")["status"] == "invalid_request"
    assert resolve(history, "Hồng", "contacts.contacts", "missing.tool", "contact_id")["status"] == "invalid_request"
    assert resolve(history, "Hồng", "unregistered.contacts", "phone.call", "contact_id")["status"] == "unsupported"


def test_id_is_not_fuzzy_matched_and_return_type_is_not_coerced():
    history = history_read("contacts.search", "contacts", [{"id": "c_123", "name": "Someone"}])
    assert resolve(history, "c_124", "contacts.contacts", "phone.call", "contact_id")["status"] == "not_found"
    history = history_read("contacts.search", "contacts", [{"id": 123, "name": "Someone"}])
    assert resolve(history, "Someone", "contacts.contacts", "phone.call", "contact_id")["status"] == "not_found"


def test_context_snapshot_overrides_stale_read_and_nested_rewards_preserve_source():
    history = history_read("contacts.search", "contacts", [{"id": "c_1", "name": "Tên cũ"}])
    context = {"collections": {"contacts.contacts": [{"id": "c_1", "name": "Tên mới"}]}}
    result = resolve(history, "Tên mới", "contacts.contacts", "phone.call", "contact_id", context)
    assert result["status"] == "matched" and result["candidates"][0]["evidence"]["source"] == "context"
    history = history_read("loyalty.get_points", "programs", [
        {"id": "p_1", "name": "Cinema", "rewards": [{"id": "r_1", "name": "Bắp nước", "cost": 400}]}])
    result = resolve(history, "Bắp nước", "loyalty.rewards", "loyalty.redeem", "reward_id")
    candidate = result["candidates"][0]
    assert candidate["value"] == "r_1" and candidate["parent"] == {"id": "p_1", "name": "Cinema"}
    assert candidate["evidence"]["path"].endswith("programs[0].rewards[0]")


def test_lookup_preserves_history_fact_and_origin_without_mutation():
    history = [{"role": "user", "text": "Thứ sáu tuần sau đi Đồng Hới chuyến RV305."}]
    context = {"user_profile": {"city": "Hà Nội"}}
    original = deepcopy((context, history))
    result = execute("local_context.lookup", {"query": "Đồng Hới"}, context, history, TOOLS)
    assert result["status"] == "matched"
    assert result["result"][0]["record"]["text"] == history[0]["text"]
    assert result["evidence"][0]["path"] == "history[0].text"
    assert (context, history) == original


def test_context_name_without_id_cannot_invent_an_id():
    context = {"collections": {"contacts.contacts": [{"name": "Bà Hồng"}]}}
    assert resolve([], "Bà Hồng", "contacts.contacts", "phone.call", "contact_id", context)["status"] == "not_found"
    result = resolve([], "Bà Hồng", "contacts.contacts", "contacts.search", "query", context)
    assert result["candidates"][0]["value"] == "Bà Hồng"


@pytest.mark.parametrize("name,args", [
    ("local_context.lookup", {"query": "Hồng", "max_results": True}),
    ("local_context.lookup", {"query": "Hồng", "max_results": 11}),
    ("local_context.lookup", {"query": ""}),
    ("local_entity.resolve", {"mention": "Hồng"}),
    ("local_context.lookup", {"query": "Hồng", "hidden_path": "/tmp/gold"}),
])
def test_argument_validation_is_bounded_and_rejects_unknown_fields(name, args):
    with pytest.raises(ValueError):
        execute(name, args, {}, [], TOOLS)


def test_schemas_are_independent():
    first = schemas()
    first["local_entity.resolve"]["parameters"]["required"].append("gold")
    assert "gold" not in schemas()["local_entity.resolve"]["parameters"]["required"]


def test_phone_projection_uses_recorded_phone_and_unsupported_arguments_do_not_receive_name():
    history = history_read("contacts.search", "contacts", [{"id": "c_1", "name": "Vợ Hồng", "phone": "0900123456"}])
    tools = deepcopy(TOOLS)
    tools["phone.call"]["parameters"]["properties"]["phone_number"] = {"type": "string"}
    args = {"mention": "Vợ Hồng", "collection": "contacts.contacts", "target_tool": "phone.call", "argument": "phone_number"}
    result = execute("local_entity.resolve", args, {}, history, tools)
    assert result["status"] == "matched"
    assert result["candidates"][0]["value"] == "0900123456"
    tools["contacts.search"]["parameters"]["properties"]["time"] = {"type": "string"}
    args.update(target_tool="contacts.search", argument="time")
    assert execute("local_entity.resolve", args, {}, history, tools)["status"] == "unsupported"
    tools["phone.call"]["parameters"]["properties"]["phone_number"]["enum"] = ["0911000000"]
    args.update(target_tool="phone.call", argument="phone_number")
    assert execute("local_entity.resolve", args, {}, history, tools)["status"] == "not_found"


def test_later_empty_full_read_invalidates_old_entities_but_partial_reads_preserve_others():
    old = history_read("contacts.search", "contacts", [{"id": "c_1", "name": "Hồng"}])
    empty = history_read("contacts.search", "contacts", [])
    assert resolve(old + empty, "Hồng", "contacts.contacts", "phone.call", "contact_id")["status"] == "not_found"
    empty[0]["tool_calls"][0]["arguments"] = {"query": "Minh"}
    assert resolve(old + empty, "Hồng", "contacts.contacts", "phone.call", "contact_id")["status"] == "matched"
    assert resolve(old, "Hồng", "contacts.contacts", "phone.call", "contact_id",
                   {"collections": {"contacts.contacts": []}})["status"] == "not_found"


def test_union_schema_does_not_coerce_integer_id_into_string():
    tools = deepcopy(TOOLS)
    tools["phone.call"]["parameters"]["properties"]["contact_id"] = {
        "anyOf": [{"type": "string"}, {"type": "null"}], "x-ref": "contacts.contacts"}
    history = history_read("contacts.search", "contacts", [{"id": 123, "name": "Someone"}])
    result = execute("local_entity.resolve", {"mention": "Someone", "collection": "contacts.contacts",
                      "target_tool": "phone.call", "argument": "contact_id"}, {}, history, tools)
    assert result["status"] == "not_found"


def test_actual_contact_get_and_medication_schedule_response_shapes_are_indexed():
    history = history_read("contacts.get", "contacts", [{"id": "c_1", "name": "Hồng"}])
    result = resolve(history, "Hồng", "contacts.contacts", "phone.call", "contact_id")
    assert result["status"] == "matched" and result["candidates"][0]["value"] == "c_1"
    history = history_read("medication.list", "schedules", [{"id": "sch_1", "name": "Vitamin D", "time": "08:30"}])
    tools = {"medication.remove": {"parameters": {"properties": {
        "schedule_id": {"type": "string", "x-ref": "medication.schedules"}}}}}
    result = execute("local_entity.resolve", {"mention": "Vitamin D", "collection": "medication.schedules",
                     "target_tool": "medication.remove", "argument": "schedule_id"}, {}, history, tools)
    assert result["status"] == "matched" and result["candidates"][0]["value"] == "sch_1"
