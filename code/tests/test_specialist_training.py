"""Specialist demonstrations are grounded in visible inputs, not next labels."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.local_helpers import LOCAL_HELPER_POLICY, execute_helper, helper_tool_schemas
from lifemate.local_time import LOCAL_TIME_POLICY
from lifemate.runtime import load_runtime_manifest, RUNTIME_MANIFEST
from lifemate.sft import SFT_PROMPT_POLICY, replay_sample, validate_prompt_policy
from lifemate.specialist_training import specialist_demonstrations
from lifemate.text import TEXT_NORMALIZATION
from lifemate.execution_policy import EXECUTION_POLICY
from lifemate.itn import ITN_POLICY, detect_candidates
from test_local_time_runtime import temporal_artifact


CONTEXT = {"current_time": "2026-10-20T22:10:00+07:00", "timezone": "Asia/Ho_Chi_Minh"}
TOOLS = {
    "home_security.acknowledge_alert": {"name": "home_security.acknowledge_alert", "parameters": {
        "type": "object", "properties": {"alert_id": {"type": "string", "x-ref": "home_security.alerts"}},
        "required": ["alert_id"]}},
    "notes.create": {"name": "notes.create", "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "content": {"type": "string"}}, "required": ["title", "content"]}},
    "sys.ask_missing_info": {"name": "sys.ask_missing_info", "parameters": {"type": "object", "properties": {
        "tool_name": {"type": "string"}, "params": {"type": "array"}}, "required": ["tool_name", "params"]}},
}
HISTORY = [{"role": "assistant", "text": "Cửa sau đang mở.", "tool_calls": [
    {"name": "home_security.list_alerts", "arguments": {}}], "tool_results": [{"ok": True, "alerts": [
        {"id": "alert_back", "location": "cửa sau", "message": "Cửa sau mở quá 10 phút"},
        {"id": "alert_hall", "location": "cổng", "message": "Khách Peter Hall bấm chuông"}]}]}]


def names(calls):
    return [call["name"] for call in calls]


def test_historical_entity_demonstration_keeps_collection_and_original_identity():
    original = deepcopy((CONTEXT, HISTORY, TOOLS))
    calls = specialist_demonstrations("Đánh dấu đã coi cửa sau.", CONTEXT, HISTORY, TOOLS, [])
    helper = next(call for call in calls if call["name"] == "local_entity.resolve")
    assert helper["arguments"] == {"target_tool": "home_security.acknowledge_alert", "argument": "alert_id",
                                    "collection": "home_security.alerts", "mention": "cửa sau", "max_results": 3}
    result = execute_helper(helper["name"], helper["arguments"], CONTEXT, HISTORY, TOOLS)
    assert result["candidates"][0]["value"] == "alert_back"
    assert (CONTEXT, HISTORY, TOOLS) == original


@pytest.mark.parametrize("text,kind,value", [
    ("Soạn nháp gửi team chấm data a còng example chấm test, báo dừng gói.", "email", "team.data@example.test"),
    ("Soạn nháp gửi Team chấm Data a còng example chấm test.", "email", "Team.Data@example.test"),
    ("Soạn nháp gửi o'connor@example.com.", "email", "o'connor@example.com"),
    ("Soạn nháp gửi Lan chấm Data@example.com.", "email", "Lan.Data@example.com"),
    ("Gọi số không chín tám bảy không không một một hai hai không.", "phone", "09870011220"),
])
def test_normalizer_demonstrations_copy_only_the_observable_span(text, kind, value):
    helpers = specialist_demonstrations(text, CONTEXT, [], TOOLS, [])
    assert "local_text.normalize" not in names(helpers)
    row = next(row for row in detect_candidates(text) if row["kind"] == kind)
    assert row["source_text"] in text
    assert row["value"] == value
    assert execute_helper("local_text.normalize", {"kind": kind, "text": row["source_text"]}, CONTEXT, [], TOOLS)["value"] == value



@pytest.mark.parametrize("text", [
    "Gọi số không chín tám bảy không không một một hai hai không không không không không không.",
    "Gọi số 0987001122000000.",
])
def test_phone_demonstration_never_truncates_an_overlong_number(text):
    calls = specialist_demonstrations(text, CONTEXT, [], TOOLS, [])
    assert "local_text.normalize" not in names(calls)


def test_retrieval_demonstration_uses_history_for_anaphoric_request():
    history = [{"role": "user", "text": "Đồng Hới là điểm đến của chuyến RV305."}]
    calls = specialist_demonstrations("Đồng Hới lúc nãy.", CONTEXT, history, TOOLS, [])
    assert "local_context.lookup" in names(calls)


def test_comparison_demonstrations_use_literal_expression_or_known_prices():
    calls = specialist_demonstrations("Tính tổng 125000 + 200000.", CONTEXT, [], TOOLS, [])
    summed = next(c for c in calls if c["name"] == "local_checks.evaluate")
    assert execute_helper(summed["name"], summed["arguments"], CONTEXT, [], TOOLS)["result"]["value"] == 325000
    history = [{"role": "assistant", "text": "Có ba xe.", "tool_calls": [
        {"name": "bike_rental.find_bikes", "arguments": {}}], "tool_results": [{"ok": True, "bikes": [
            {"id": "a", "available": True, "price_per_hour": 10000},
            {"id": "b", "available": True, "price_per_hour": 15000},
            {"id": "c", "available": False, "price_per_hour": 1000}]}]}]
    calls = specialist_demonstrations("Giữ chiếc xe rẻ nhất.", CONTEXT, history, TOOLS, [])
    minimum = next(c for c in calls if c["name"] == "local_checks.evaluate")
    assert minimum["arguments"] == {"operation": "minimum", "values": [10000, 15000]}


def test_schema_inspection_only_uses_a_previously_rejected_proposal():
    text = "Ghi chú giúp tôi."
    assert "local_schema.inspect" not in names(specialist_demonstrations(text, CONTEXT, [], TOOLS, []))
    proposal = {"name": "notes.create", "arguments": {"title": "Ghi chú"}, "stage": "validate", "error": "missing content"}
    calls = specialist_demonstrations(text, CONTEXT, [], TOOLS, [proposal])
    inspect = next(c for c in calls if c["name"] == "local_schema.inspect")
    assert inspect["arguments"] == {"target_tool": "notes.create", "arguments": {"title": "Ghi chú"}}


def test_helper_budget_and_executed_helper_deduplication():
    text = "Soạn nháp gửi team chấm qa a còng example chấm test."
    first = specialist_demonstrations(text, CONTEXT, [], TOOLS, [])
    observed = [{**h, "scope": "local", "result": "{}"} for h in first]
    assert specialist_demonstrations(text, CONTEXT, [], TOOLS, observed) == []
    assert specialist_demonstrations(text, CONTEXT, [], TOOLS, observed * 8) == []


class Env:
    def __init__(self): self.calls = []
    def validate(self, name, args):
        assert name in TOOLS
        return {"ok": True}
    def call(self, name, args):
        assert name in TOOLS
        self.calls.append({"name": name, "arguments": deepcopy(args)})
        return {"ok": True}


def record(arguments):
    return {"id": "train_fixture", "audio_text": "Đánh dấu đã coi cửa sau.", "history": HISTORY,
            "context": {**CONTEXT, "installed_apps": {"home_security": {"tools": ["home_security.acknowledge_alert"]}}},
            "label": {"tool_calls": [{"name": "home_security.acknowledge_alert", "arguments": arguments}]}}


def test_sft_specialists_do_not_see_future_gold_and_keep_env_sequence():
    env = Env()
    left = replay_sample(record({"alert_id": "future_a"}), TOOLS, env, Path("/unused"))
    right = replay_sample(record({"alert_id": "future_b"}), TOOLS, Env(), Path("/unused"))
    a, b = [json.loads(x["messages"][-1]["content"]) for x in left], [json.loads(x["messages"][-1]["content"]) for x in right]
    helpers_a = [x for x in a if x.get("name") in helper_tool_schemas()]
    helpers_b = [x for x in b if x.get("name") in helper_tool_schemas()]
    assert helpers_a == helpers_b and helpers_a
    assert "future_a" not in left[0]["messages"][1]["content"]
    assert env.calls == record({"alert_id": "future_a"})["label"]["tool_calls"]
    assert [x for x in a if x.get("name") not in helper_tool_schemas()] == [
        {"action": "call", "name": "home_security.acknowledge_alert", "arguments": {"alert_id": "future_a"}}, {"action": "done"}]


@pytest.mark.parametrize("result,status", [
    ({"ok": False, "error": "Entity does not exist"}, "failure"),
    ({"ok": True, "done": False, "message": "Cannot complete action"}, "failure"),
    ({"ok": True, "items": []}, "empty"),
    ({"ok": True, "done": True}, None),
    ({"ok": True, "item": {"done": False}}, None),
])
def test_sft_feedback_distinguishes_reported_failure_empty_and_success(result, status):
    class ResultEnv(Env):
        def call(self, name, args):
            super().call(name, args)
            return deepcopy(result)
    examples = replay_sample(record({"alert_id": "alert_back"}), TOOLS, ResultEnv(), Path("/unused"))
    final_prompt = json.loads(examples[-1]["messages"][1]["content"])
    observation = final_prompt["previous_actions_and_results"][-1]
    assert json.loads(observation["result"]) == result
    if status is None:
        assert "verification" not in observation
    else:
        assert observation["verification"]["status"] == status
        assert observation["verification"]["goal_completion"] == "unknown"


def test_old_specialist_manifest_is_rejected_without_guessing_compatibility():
    manifest = {"prompt_policy": SFT_PROMPT_POLICY, "text_normalization": TEXT_NORMALIZATION,
                "local_time_policy": LOCAL_TIME_POLICY, "local_helper_policy": LOCAL_HELPER_POLICY, "itn_policy": ITN_POLICY, "execution_policy": EXECUTION_POLICY}
    validate_prompt_policy(manifest)
    manifest.pop("local_helper_policy")
    with pytest.raises(ValueError, match="rerun prepare_sft"):
        validate_prompt_policy(manifest)


def test_runtime_manifest_requires_specialist_capability(temporal_artifact):
    base, artifact, tokenizer, manifest = temporal_artifact
    assert manifest["local_helper_policy"] == LOCAL_HELPER_POLICY
    manifest.pop("local_helper_policy")
    (artifact / RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Python specialist helper policy.*regenerate SFT"):
        load_runtime_manifest(base, tokenizer, artifact)


@pytest.mark.parametrize("text,kind,value", [
    ("Đó là số không chín một hai ba bốn năm sáu bảy tám.", "phone", "0912345678"),
    ("Chuyển hai triệu rưỡi cho bà Hồng.", "money", 2500000),
])
def test_manual_fallback_demonstrations_remain_when_auto_itn_has_no_clear_marker(text, kind, value):
    assert detect_candidates(text) == []
    calls = specialist_demonstrations(text, CONTEXT, [], TOOLS, [])
    call = next(row for row in calls if row["name"] == "local_text.normalize")
    assert call["arguments"]["kind"] == kind
    assert call["arguments"]["text"] in text
    assert execute_helper(call["name"], call["arguments"], CONTEXT, [], TOOLS)["value"] == value


@pytest.mark.parametrize("text", ["Chuyển một kênh.", "Trả hai quyển sách."])
def test_fallback_does_not_treat_generic_counts_as_money(text):
    calls = specialist_demonstrations(text, CONTEXT, [], TOOLS, [])
    assert "local_text.normalize" not in names(calls)
