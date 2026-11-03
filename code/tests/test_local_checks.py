from copy import deepcopy

import pytest

from lifemate.local_checks import execute, inspect_call, schemas


def call(operation, **kwargs):
    return execute("local_checks.evaluate", {"operation": operation, **kwargs}, {})


@pytest.mark.parametrize("raw,expected", [
    ("team chấm data a còng example chấm test", "team.data@example.test"),
    ("team chấm qa a còng example chấm test", "team.qa@example.test"),
    ("Team.Data@Example.test", "Team.Data@Example.test"),
    ("user gạch dưới qa a còng example chấm test", "user_qa@example.test"),
])
def test_email_preserves_identity_parts(raw, expected):
    assert execute("local_text.normalize", {"kind": "email", "text": raw}, {})["value"] == expected


def test_normalizer_does_not_guess_from_sentences_or_two_addresses():
    assert execute("local_text.normalize", {"kind": "email", "text": "gửi cho team a còng example chấm test"}, {})["status"] == "unsupported"
    assert execute("local_text.normalize", {"kind": "email", "text": "a@x.test hoặc b@y.test"}, {})["status"] == "ambiguous"
    assert execute("local_text.normalize", {"kind": "phone", "text": "0987 hoặc 0912"}, {})["status"] == "ambiguous"
    assert execute("local_text.normalize", {"kind": "phone", "text": "không chín tám"}, {})["status"] == "unsupported"


def test_phone_preserves_leading_zero_and_never_repairs_missing_digits():
    raw = "không chín tám bảy, không không một, một hai hai không"
    assert execute("local_text.normalize", {"kind": "phone", "text": raw}, {})["value"] == "09870011220"
    assert execute("local_text.normalize", {"kind": "phone", "text": "+84 (98) 700-1122"}, {})["value"] == "+84987001122"


def test_decimal_sum_compare_minimum_and_boundary():
    assert call("sum", values=[0.1, 0.2])["result"]["value"] == 0.3
    assert call("sum", values=[])["result"]["value"] == 0
    assert call("compare", left=430000, right=430000, comparison="ge")["result"]["value"] is True
    assert call("minimum", values=[15000, 10000, 10000])["result"] == {"value": 10000, "indices": [1, 2]}


def test_filter_preserves_records_and_does_not_mutate():
    rows = [{"id": "a", "cost": 300}, {"id": "b", "cost": 400}]
    original = deepcopy(rows)
    result = call("filter", records=rows, field="cost", comparison="le", right=350)
    assert result["result"] == {"records": [rows[0]], "indices": [0]}
    result["result"]["records"][0]["id"] = "changed"
    assert rows == original and "supplied operands" in result["warnings"][0]


@pytest.mark.parametrize("args", [
    {"operation": "sum", "values": [True]},
    {"operation": "sum", "values": [float("nan")]},
    {"operation": "sum", "values": [float("inf")]},
    {"operation": "sum", "values": list(range(101))},
    {"operation": "sum", "values": ["100"]},
    {"operation": "sum", "values": [10**19]},
    {"operation": "minimum", "values": []},
    {"operation": "compare", "left": 1, "right": 2, "comparison": "eval"},
    {"operation": "sum", "values": [1], "left": 2},
    {"operation": "filter", "records": [{}], "field": "cost", "comparison": "le", "right": 1},
])
def test_math_invalid_inputs_do_not_coerce_or_execute(args):
    with pytest.raises(ValueError):
        execute("local_checks.evaluate", args, {})


TOOLS = {
    "app.send": {"name": "app.send", "parameters": {
        "type": "object", "properties": {
            "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
            "phone": {"type": "string"},
            "count": {"type": "integer", "minimum": 1, "maximum": 3},
            "mode": {"type": "string", "enum": ["sms", "chat"]},
            "days": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 7}, "maxItems": 2},
            "config": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"], "additionalProperties": False}},
        "required": ["count"], "x-one-of": [["contact_id"], ["phone"]], "additionalProperties": False}}
}


def test_schema_inspection_checks_actual_schema_not_entity_or_goal():
    result = inspect_call("app.send", {"contact_id": "never_seen_id", "count": 1}, TOOLS)
    assert result["status"] == "valid" and result["scope"] == "declared_schema_only"
    assert "Entity existence" in result["warnings"][0]
    result = execute("local_schema.inspect", {"target_tool": "app.send", "arguments": {
        "contact_id": "x", "count": 2, "mode": "sms", "days": [1, 7], "config": {"enabled": True}}}, TOOLS)
    assert result["status"] == "valid"


@pytest.mark.parametrize("args,code", [
    ({"count": 1}, "x-one-of"),
    ({"phone": "00123"}, "required"),
    ({"phone": "00123", "count": True}, "type"),
    ({"phone": "00123", "count": 0}, "minimum"),
    ({"phone": "00123", "count": 4}, "maximum"),
    ({"phone": "00123", "count": 1, "mode": "email"}, "enum"),
    ({"phone": "00123", "count": 1, "typo": "x"}, "additionalProperties"),
    ({"phone": "00123", "count": 1, "days": [1, 8]}, "maximum"),
    ({"phone": "00123", "count": 1, "config": {"enabled": "true"}}, "type"),
])
def test_schema_errors_have_field_paths(args, code):
    result = inspect_call("app.send", args, TOOLS)
    assert result["status"] == "invalid"
    assert any(e["code"] == code and e["path"].startswith("$.arguments") for e in result["errors"])


def test_unsupported_rules_are_not_claimed_valid_or_used_for_network_refs():
    tools = {"app.unknown": {"parameters": {"type": "object", "$ref": "https://private.invalid/schema"}}}
    assert inspect_call("app.unknown", {}, tools)["status"] == "unsupported"
    assert inspect_call("app.not_allowed", {}, TOOLS)["status"] == "invalid"
    assert inspect_call("app.send", [], TOOLS)["status"] == "invalid"
    # Unknown patternProperties could authorize the x_ field; do not falsely reject.
    tools["app.unknown"]["parameters"] = {"type": "object", "patternProperties": {"^x_": {}}, "additionalProperties": False}
    assert inspect_call("app.unknown", {"x_a": 1}, tools)["status"] == "unsupported"


def test_date_formats_and_combinators_are_inspected_conservatively():
    tools = {"app.date": {"parameters": {"type": "object", "properties": {
        "date": {"type": "string", "format": "date"},
        "value": {"oneOf": [{"type": "integer"}, {"type": "string"}]}}}}}
    assert inspect_call("app.date", {"date": "2024-02-29", "value": 1}, tools)["status"] == "valid"
    assert inspect_call("app.date", {"date": "2026-02-29", "value": True}, tools)["status"] == "invalid"
    tools["app.date"]["parameters"]["properties"]["value"] = {"oneOf": [{"type": "integer"}, {"pattern": "^a"}]}
    assert inspect_call("app.date", {"value": 1}, tools)["status"] == "unsupported"


def test_schema_copies_and_strict_helper_arguments():
    first = schemas()
    first["local_text.normalize"]["parameters"]["properties"].clear()
    assert "kind" in schemas()["local_text.normalize"]["parameters"]["properties"]
    with pytest.raises(ValueError):
        execute("local_text.normalize", {"kind": "phone", "text": "09870011220", "repair": True}, {})


@pytest.mark.parametrize("value", [".team@example.test", "team..data@example.test", "team.@example.test"])
def test_malformed_email_is_not_certified(value):
    assert execute("local_text.normalize", {"kind": "email", "text": value}, {})["status"] == "unsupported"


def test_malformed_schema_and_helper_name_are_bounded():
    assert inspect_call("app.bad", {}, {"app.bad": None})["status"] == "unsupported"
    with pytest.raises(ValueError):
        execute([], {}, {})
