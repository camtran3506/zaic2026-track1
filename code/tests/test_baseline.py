import json
from pathlib import Path

from lifemate.agent import AgentConfig, LifeMateAgent, parse_decision
from lifemate.catalog import allowed_tools
from main import resolve_data_layout
from score_train import call_f1

CATALOG = {
    "smart_home.get_status": {
        "name": "smart_home.get_status", "kind": "read", "description": "Liệt kê thiết bị nhà",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    "smart_home.light_set": {
        "name": "smart_home.light_set", "kind": "write", "description": "Bật/tắt đèn",
        "parameters": {"type": "object", "properties": {
            "device_id": {"type": "string", "x-ref": "smart_home.devices"},
            "power": {"type": "string", "enum": ["on", "off"]},
        }, "required": ["device_id"]},
    },
    "sys.ask_missing_info": {
        "name": "sys.ask_missing_info", "kind": "meta", "description": "Hỏi người dùng",
        "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array"}},
                       "required": ["tool_name", "params"]},
    },
}


class StubASR:
    def transcribe(self, path):
        return "Bật đèn phòng khách"


class StubLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
    def generate(self, system, user):
        return next(self.outputs)


class FakeEnv:
    def __init__(self):
        self.calls = []
    def validate(self, name, args):
        spec = CATALOG[name]["parameters"]
        missing = set(spec.get("required", [])) - set(args)
        return {"ok": not missing, "error": "missing required" if missing else ""}
    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        if name == "smart_home.get_status":
            return {"ok": True, "devices": [{"id": "light_living", "name": "đèn phòng khách"}]}
        return {"ok": True}
    def get_tool_calls(self):
        return self.calls


def item():
    return {"id": "train_0001", "audio": "audio/train_0001/train_0001.wav",
            "context": {"timezone": "Asia/Ho_Chi_Minh", "installed_apps": {
                "smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]}}}, "history": []}


def run(outputs, installed=None):
    entry = item()
    if installed is not None:
        entry["context"]["installed_apps"] = installed
    env = FakeEnv()
    result = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig()).run(
        entry, env, Path("/tmp/lifemate_test_no_access_to_env"), "train")
    return env.calls, result


def test_multiple_steps_read_write():
    calls, stats = run([
        '{"action":"call","name":"smart_home.get_status","arguments":{}}',
        '{"action":"call","name":"smart_home.light_set","arguments":{"device_id":"light_living","power":"on"}}',
        '{"action":"done"}',
    ])
    assert [c["name"] for c in calls] == ["smart_home.get_status", "smart_home.light_set"]
    assert stats["executed"] == 2


def test_schema_invalid_proposal_not_executed():
    calls, stats = run([
        '{"action":"call","name":"smart_home.light_set","arguments":{"power":"on"}}',
        '{"action":"call","name":"smart_home.get_status","arguments":{}}',
        '{"action":"done"}',
    ])
    assert len(calls) == 1
    assert stats["invalid_proposals"] == 1


def test_unsupported_app_rejected():
    calls, _ = run(['{"action":"call","name":"smart_home.get_status","arguments":{}}',
                    '{"action":"done"}'], installed={})
    assert calls == []


def test_missing_info():
    calls, _ = run([
        '{"action":"call","name":"sys.ask_missing_info","arguments":{"tool_name":"smart_home.light_set","params":["device_id"]}}'
    ])
    assert calls[0]["name"] == "sys.ask_missing_info"


def test_json_parser_and_out_of_scope():
    assert parse_decision('```json\n{"action":"done"}\n```') == {"action": "done"}
    assert allowed_tools(CATALOG, {}).keys() == {"sys.ask_missing_info"}


def test_proxy_matching():
    gold = [{"name": "smart_home.get_status", "arguments": {}},
            {"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "on"}}]
    assert call_f1(list(reversed(gold)), gold, CATALOG)[0] == 1.0
    assert call_f1([], [], CATALOG)[0] == 1.0
    assert call_f1(gold[:1], gold, CATALOG)[0] == 2 / 3


def test_direct_split_layout(tmp_path):
    (tmp_path / "public_test.jsonl").write_text("", encoding="utf-8")
    split_dir, root, temp = resolve_data_layout(tmp_path, "public_test")
    assert split_dir == tmp_path
    assert (root / "public_test").resolve() == tmp_path.resolve()
    temp.cleanup()


def test_parent_split_layout(tmp_path):
    (tmp_path / "public_test").mkdir()
    (tmp_path / "public_test" / "public_test.jsonl").write_text("", encoding="utf-8")
    split_dir, root, temp = resolve_data_layout(tmp_path, "public_test")
    assert split_dir == tmp_path / "public_test"
    assert root == tmp_path
    assert temp is None
