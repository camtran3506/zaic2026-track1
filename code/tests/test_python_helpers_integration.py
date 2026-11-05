"""Exercise Python specialists through the planner, ToolEnv and submission path."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.function_calling import native_chat
from lifemate.local_helpers import execute_helper


CATALOG = {
    "notes.search": {
        "name": "notes.search", "kind": "read", "description": "Search notes",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    },
    "notes.append": {
        "name": "notes.append", "kind": "write", "description": "Append a note",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]},
    },
    "sys.ask_missing_info": {
        "name": "sys.ask_missing_info", "kind": "meta", "description": "Ask user",
        "parameters": {"type": "object", "properties": {
            "tool_name": {"type": "string"}, "params": {"type": "array"}},
            "required": ["tool_name", "params"]},
    },
}


def call(name, **arguments):
    return {"action": "call", "name": name, "arguments": arguments}


DONE = {"action": "done"}


class Env:
    """Public ToolEnv surface only; rejects one sentinel, accepts other types."""
    def __init__(self, *args, **kwargs):
        self.calls = []
        self.validations = []
        self.notes = []

    def validate(self, name, arguments):
        assert name in CATALOG  # Any accidental helper forwarding fails here.
        self.validations.append({"name": name, "arguments": deepcopy(arguments)})
        return {"ok": arguments.get("text") != "INVALID", "error": "reserved note"}

    def call(self, name, arguments):
        assert name in CATALOG
        self.calls.append({"name": name, "arguments": deepcopy(arguments)})
        if name == "notes.search":
            return {"ok": True, "notes": [deepcopy(note) for note in self.notes
                                           if arguments["query"] in str(note["text"])]}
        if name == "notes.append":
            note = {"text": deepcopy(arguments["text"])}
            self.notes.append(note)
            return {"ok": True, "done": True, "note": deepcopy(note)}
        return {"ok": True}

    def get_tool_calls(self):
        return deepcopy(self.calls)

    def get_state_hash(self):
        return hashlib.sha256(json.dumps(self.notes, sort_keys=True).encode()).hexdigest()


class Planner:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate(self, system, user):
        self.requests.append((system, user))
        output = next(self.outputs)
        return output if isinstance(output, str) else json.dumps(output)


def sample():
    return {"id": "public_1", "context": {"installed_apps": {
        "notes": {"tools": ["notes.search", "notes.append"]}}}, "history": []}


def prepared():
    return {"id": "public_1", "transcript": "Ghi các ghi chú.", "history": [], "asr_seconds": 0}


def run(outputs, env=None):
    env = Env() if env is None else env
    planner, events = Planner(outputs), []
    report = LifeMateAgent(None, planner, CATALOG, AgentConfig()).run(
        sample(), env, Path("/unused"), "public_test", prepared(), trace=events.append)
    return env, report, events, planner


def test_helpers_stay_out_of_env_log_and_hash_but_reach_native_planner():
    application = call("notes.append", text="team@example.test")
    env, report, events, planner = run([
        call("local_text.normalize", kind="email", text="team a còng example chấm test"),
        call("local_checks.evaluate", operation="sum", values=[2, 3]), application, DONE,
    ])
    baseline = Env()
    baseline.call(application["name"], application["arguments"])
    assert env.get_tool_calls() == baseline.get_tool_calls()
    assert env.get_state_hash() == baseline.get_state_hash()
    assert env.validations == baseline.get_tool_calls()
    assert report["local_executed"] == 2 and report["executed"] == 1
    assert [event["name"] for event in events if event["event"] == "call_start"] == ["notes.append"]
    system, user = planner.requests[-1]
    chat, tools = native_chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    assert [turn["tool_calls"][0]["function"]["name"] for turn in chat if turn.get("tool_calls")] == [
        "local_text.normalize", "local_checks.evaluate", "notes.append"]
    assert len([turn for turn in chat if turn["role"] == "tool"]) == 3
    assert "local_text.normalize" in {tool["function"]["name"] for tool in tools}


def test_validation_rejection_is_cached_without_calling_env_again():
    rejected, corrected = call("notes.append", text="INVALID"), call("notes.append", text="valid")
    env, report, events, planner = run([rejected, rejected, corrected, DONE])
    assert env.validations == [{key: value for key, value in proposal.items() if key != "action"}
                               for proposal in [rejected, corrected]]
    assert env.get_tool_calls() == [{"name": "notes.append", "arguments": {"text": "valid"}}]
    assert report["invalid_proposals"] == 2 and report["stop_reason"] == "done"
    assert [event["stage"] for event in events if event["event"] == "decision_rejected"] == [
        "validate", "loop_guard"]
    system, user = planner.requests[2]
    chat, _ = native_chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    assert not any(turn["role"] == "tool" or turn.get("tool_calls") for turn in chat)
    assert any(json.loads(turn["content"]).get("agent_feedback", {}).get("stage") == "loop_guard"
               for turn in chat if turn["role"] == "user")


def test_empty_read_repeat_is_blocked_and_feedback_does_not_force_completion():
    query = call("notes.search", query="absent")
    env, report, events, planner = run([query, query, DONE])
    assert env.get_tool_calls() == [{"name": "notes.search", "arguments": {"query": "absent"}}]
    assert len(env.validations) == 1 and report["invalid_proposals"] == 1
    assert report["stop_reason"] == "done"
    assert any(event["event"] == "completion_advisory" for event in events)
    system, user = planner.requests[1]
    chat, _ = native_chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    feedback = [json.loads(turn["content"])["execution_feedback"] for turn in chat
                if turn["role"] == "user" and "execution_feedback" in json.loads(turn["content"])]
    assert feedback[0]["status"] == "empty" and feedback[0]["goal_completion"] == "unknown"


def test_local_calculation_does_not_reset_empty_read_guard():
    query = call("notes.search", query="absent")
    env, report, _, _ = run([query, call("local_checks.evaluate", operation="sum", values=[1, 1]), query, DONE])
    assert len(env.get_tool_calls()) == 1
    assert report["executed"] == 1 and report["local_executed"] == 1 and report["invalid_proposals"] == 1


def test_successful_repeated_writes_are_preserved_and_duplicate_read_is_blocked():
    write, read = call("notes.append", text="same"), call("notes.search", query="same")
    env, report, _, _ = run([write, write, read, read, DONE])
    assert [entry["name"] for entry in env.get_tool_calls()] == [
        "notes.append", "notes.append", "notes.search"]
    assert env.notes == [{"text": "same"}, {"text": "same"}]
    assert report["executed"] == 3 and report["invalid_proposals"] == 1


def test_write_between_empty_queries_allows_fresh_read():
    query = call("notes.search", query="new")
    env, report, _, _ = run([query, call("notes.append", text="new"), query, DONE])
    assert [entry["name"] for entry in env.get_tool_calls()] == ["notes.search", "notes.append", "notes.search"]
    assert report["invalid_proposals"] == 0


def test_schema_helper_and_automatic_inspection_remain_advisory():
    env, report, events, planner = run([
        call("local_schema.inspect", target_tool="notes.append", arguments={"text": 2}),
        call("notes.append", text=2), DONE,
    ])
    assert env.get_tool_calls() == [{"name": "notes.append", "arguments": {"text": 2}}]
    assert env.validations == env.get_tool_calls()
    assert report["invalid_proposals"] == 0 and report["executed"] == 1
    inspections = [event["inspection"] for event in events if event["event"] == "schema_inspection"]
    assert inspections[0]["status"] == "invalid"
    observation = json.loads(planner.requests[1][1])["previous_actions_and_results"][0]
    assert json.loads(observation["result"])["status"] == "invalid"


@pytest.mark.parametrize("name, arguments, context, history, tools", [
    (None, {}, {}, [], CATALOG),
    ("local_checks.evaluate", [], {}, [], CATALOG),
    ("local_checks.evaluate", {}, None, [], CATALOG),
    ("local_checks.evaluate", {}, {}, {}, CATALOG),
    ("local_checks.evaluate", {}, {}, [], None),
    ("unregistered.helper", {}, {}, [], CATALOG),
])
def test_registry_rejects_malformed_inputs(name, arguments, context, history, tools):
    with pytest.raises(ValueError):
        execute_helper(name, arguments, context, history, tools)


def test_llm_submission_uses_only_env_calls_and_env_hash(tmp_path, monkeypatch):
    import main

    planner = Planner([call("local_checks.evaluate", operation="sum", values=[2, 3]),
                       call("notes.append", text="five"), DONE])
    envs = []

    class ToolEnv(Env):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            envs.append(self)

    environment = ModuleType("zaic_tools")
    environment.ToolEnv = ToolEnv
    models = ModuleType("lifemate.models")
    models.LocalLLM = lambda *args, **kwargs: planner
    monkeypatch.setitem(sys.modules, "zaic_tools", environment)
    monkeypatch.setitem(sys.modules, "lifemate.models", models)
    monkeypatch.setattr(main, "seed_everything", lambda seed: None)
    # run_llm_stage prepends the dataset directory for the official ToolEnv import.
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "public_test_tools_catalog.json").write_text(
        json.dumps({"apps": [{"tools": list(CATALOG.values())}]}), encoding="utf-8")
    args = SimpleNamespace(
        split="public_test", seed=42, llm_model=tmp_path / "unused", device="cpu", llm_dtype="float32",
        max_input_tokens=65536, max_new_tokens=512, llm_adapter=None, llm_device_map="single",
        llm_max_memory_gib=6, prefix_cache="auto", max_calls=8, max_invalid=3, sample_seconds=float("inf"),
        use_train_transcript=False, output=tmp_path / "submission.jsonl", log_path=tmp_path / "log.jsonl",
        fail_fast=True)
    main.run_llm_stage(args, [sample()], [prepared()], tmp_path, tmp_path)
    row = json.loads(args.output.read_text())
    assert len(envs) == 1
    assert row == {"id": "public_1", "tool_calls": envs[0].get_tool_calls(),
                   "state_hash": envs[0].get_state_hash()}
    assert row["tool_calls"] == [{"name": "notes.append", "arguments": {"text": "five"}}]
    assert json.loads(args.log_path.read_text())["local_executed"] == 1
