"""Internal helpers use native messages but never enter official call logs."""
import json
from pathlib import Path

from lifemate.agent import AgentConfig, LifeMateAgent, build_step_prompt, parse_decision
from lifemate.function_calling import native_chat
from test_baseline import CATALOG, FakeEnv, StubASR, StubLLM, item


def decision(name, **arguments):
    return json.dumps({"action": "call", "name": name, "arguments": arguments})


DURATION = decision("local_time.duration_between",
                    start="2026-10-13T19:30:00+07:00", end="2026-10-13T20:10:00+07:00")


def test_helper_result_is_observed_but_only_application_calls_reach_env():
    entry = item()
    env = FakeEnv()
    events = []

    class LLM:
        def generate(self, system, user):
            payload = json.loads(user)
            observations = payload["previous_actions_and_results"]
            if not observations:
                # Exercise the native parser's integer handling with local schemas.
                return ("<tool_call>\n<function=local_time.departure_time>\n"
                        "<parameter=arrival>2026-09-22T09:00:00+07:00</parameter>\n"
                        "<parameter=travel_minutes>15</parameter>\n"
                        "<parameter=early_minutes>15</parameter>\n</function>\n</tool_call>")
            if len(observations) == 1:
                assert observations[0]["scope"] == "local"
                assert json.loads(observations[0]["result"])["departure"] == "2026-09-22T08:30:00+07:00"
                chat, _ = native_chat([{"role": "system", "content": system},
                                       {"role": "user", "content": user}])
                assert chat[-1]["role"] == "tool"
                assert json.loads(chat[-1]["content"])["ok"] is True
                return decision("smart_home.get_status")
            return '{"action":"done"}'

    stats = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "train", trace=events.append)
    assert env.get_tool_calls() == [{"name": "smart_home.get_status", "arguments": {}}]
    assert stats["executed"] == stats["local_executed"] == 1
    assert stats["invalid_proposals"] == 0 and stats["stop_reason"] == "done"
    assert [e["name"] for e in events if e["event"] == "validate_start"] == ["smart_home.get_status"]
    assert [e["result"]["departure"] for e in events if e["event"] == "local_call_result"] == [
        "2026-09-22T08:30:00+07:00"]


def test_helpers_do_not_consume_official_call_budget():
    env = FakeEnv()
    outputs = [DURATION, decision("smart_home.get_status"),
               decision("smart_home.light_set", device_id="light_living", power="on")]
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig(max_calls=2)).run(
        item(), env, Path("/unused"), "train")
    assert len(env.calls) == stats["executed"] == 2
    assert stats["local_executed"] == 1 and stats["stop_reason"] == "max_calls"


def test_local_budget_and_total_step_limit_bound_helper_loops():
    class Loop:
        def generate(self, system, user): return DURATION

    for cfg, expected, calls in [(AgentConfig(max_local_calls=1), "max_invalid", 1),
                                 (AgentConfig(max_total_steps=2), "max_total_steps", 2)]:
        env = FakeEnv()
        stats = LifeMateAgent(StubASR(), Loop(), CATALOG, cfg).run(item(), env, Path("/unused"), "train")
        assert stats["stop_reason"] == expected and stats["local_executed"] == calls
        assert not env.calls


def test_invalid_helper_returns_feedback_without_falling_through_to_env():
    env = FakeEnv()
    outputs = [decision("local_time.duration_between", start="not a date", end="bad"),
               decision("smart_home.get_status"), '{"action":"done"}']
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig()).run(
        item(), env, Path("/unused"), "train")
    assert stats["invalid_proposals"] == 1 and stats["local_executed"] == 1
    assert env.calls == [{"name": "smart_home.get_status", "arguments": {}}]


def test_reserved_namespace_does_not_authorize_unknown_helpers_or_missing_info_targets():
    env = FakeEnv()
    outputs = [decision("local_time.eval", code="1+1"),
               decision("sys.ask_missing_info", tool_name="local_time.departure_time", params=["arrival"]),
               '{"action":"done"}']
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig()).run(
        item(), env, Path("/unused"), "train")
    assert stats["invalid_proposals"] == 2 and stats["local_executed"] == 0
    assert not env.calls


def test_no_available_application_does_not_start_helper_only_agent():
    entry = item()
    entry["context"]["installed_apps"] = {}
    stats = LifeMateAgent(StubASR(), StubLLM([]), CATALOG, AgentConfig()).run(
        entry, FakeEnv(), Path("/unused"), "train")
    assert stats["stop_reason"] == "no_available_tools" and stats["local_executed"] == 0


def test_local_numeric_schema_used_for_native_parser():
    tools, _ = build_step_prompt("test", {}, [], CATALOG, [])
    schemas = {t["name"]: t for t in tools["tool_schemas"]}
    raw = ("<tool_call><function=local_time.departure_time>"
           "<parameter=arrival>2026-09-22T09:00:00+07:00</parameter>"
           "<parameter=travel_minutes>15</parameter><parameter=early_minutes>15</parameter>"
           "</function></tool_call>")
    assert type(parse_decision(raw, schemas)["arguments"]["travel_minutes"]) is int
