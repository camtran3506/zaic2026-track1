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


def test_valid_helper_turns_have_no_local_or_hidden_planner_count_limit():
    env = FakeEnv()
    outputs = [DURATION]*40 + ['{"action":"done"}']
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig(max_total_steps=1)).run(
        item(), env, Path('/unused'), 'train')
    assert stats['stop_reason'] == 'done' and stats['local_executed'] == 40
    assert stats['llm_steps'] == 41 and stats['non_helper_steps'] == 1
    assert not env.calls


def test_non_helper_decisions_still_have_the_planner_step_limit():
    env = FakeEnv()
    outputs = [decision('smart_home.get_status'), '{"action":"done"}']
    stats = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig(max_total_steps=1)).run(
        item(), env, Path('/unused'), 'train')
    assert stats['stop_reason'] == 'max_total_steps'
    assert stats['executed'] == stats['llm_steps'] == stats['non_helper_steps'] == 1


def test_malformed_names_remain_invalid_proposals_without_breaking_step_accounting():
    for name in ({'tool': 'local_time.duration_between'}, ['local_time.duration_between'], None):
        env = FakeEnv()
        outputs = [decision(name), '{"action":"done"}']
        stats = LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig()).run(item(), env, Path('/unused'), 'train')
        assert stats['invalid_proposals'] == 1 and stats['non_helper_steps'] == 2
        assert stats['local_executed'] == 0 and stats['stop_reason'] == 'done' and not env.calls


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


def test_graceful_stop_when_token_budget_exceeded_on_subsequent_steps():
    entry = item()
    env = FakeEnv()
    events = []

    class ExplodingLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, system, user):
            self.calls += 1
            if self.calls == 1:
                return DURATION
            raise ValueError("Prompt has 16500 tokens, exceeds --max-input-tokens=16000; full schemas, history and tool results are never truncated")

    stats = LifeMateAgent(StubASR(), ExplodingLLM(), CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "train", trace=events.append)
    assert stats["stop_reason"] == "max_input_tokens"
    assert stats["local_executed"] == 1
    assert any(e["event"] == "context_limit_reached" for e in events)

    # Initial step 1 exceeding budget re-raises as configuration error
    class InitialExplodingLLM:
        def generate(self, system, user):
            raise ValueError("Prompt has 16500 tokens, exceeds --max-input-tokens=16000; full schemas, history and tool results are never truncated")

    import pytest
    with pytest.raises(ValueError, match="exceeds --max-input-tokens"):
        LifeMateAgent(StubASR(), InitialExplodingLLM(), CATALOG, AgentConfig()).run(
            entry, FakeEnv(), Path("/unused"), "train")


def test_identical_local_helper_observations_are_deduplicated():
    entry = item()
    env = FakeEnv()
    prompts = []

    class CapturingLLM:
        def __init__(self):
            self.outputs = [DURATION, DURATION, '{"action":"done"}']
            self.idx = 0

        def generate(self, system, user):
            prompts.append(json.loads(user))
            out = self.outputs[self.idx]
            self.idx += 1
            return out

    stats = LifeMateAgent(StubASR(), CapturingLLM(), CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "train")
    assert stats["stop_reason"] == "done"
    assert stats["local_executed"] == 2
    assert len(prompts) == 3
    # Step 3 prompt (after 2 DURATION calls) only has 1 local observation entry
    obs = prompts[2]["previous_actions_and_results"]
    assert len(obs) == 1
    assert obs[0]["name"] == "local_time.duration_between"


def test_successful_helper_cleans_up_prior_rejections():
    entry = item()
    env = FakeEnv()
    prompts = []

    class RecoveringLLM:
        def __init__(self):
            # Step 1: malformed/rejected decision
            # Step 2: successful helper
            # Step 3: done
            self.outputs = [
                '{"action":"call","name":"local_batch.run","arguments":{"calls":[{"name":"news.search","arguments":{"query":"test"}}]}}',
                DURATION,
                '{"action":"done"}'
            ]
            self.idx = 0

        def generate(self, system, user):
            prompts.append(json.loads(user))
            out = self.outputs[self.idx]
            self.idx += 1
            return out

    stats = LifeMateAgent(StubASR(), RecoveringLLM(), CATALOG, AgentConfig()).run(
        entry, env, Path("/unused"), "train")
    assert stats["stop_reason"] == "done"
    assert stats["invalid_proposals"] == 1
    assert stats["local_executed"] == 1
    # Step 3 prompt (after helper succeeded) should have cleaned up the rejected batch observation
    obs = prompts[2]["previous_actions_and_results"]
    assert len(obs) == 1
    assert "error" not in obs[0]
    assert obs[0]["name"] == "local_time.duration_between"


def test_max_llm_steps_hard_ceiling():
    entry = item()
    env = FakeEnv()

    class EndlessHelperLLM:
        def generate(self, system, user):
            return DURATION

    stats = LifeMateAgent(StubASR(), EndlessHelperLLM(), CATALOG, AgentConfig(max_llm_steps=10)).run(
        entry, env, Path("/unused"), "train")
    assert stats["stop_reason"] == "max_llm_steps"
    assert stats["llm_steps"] == 10
    assert stats["local_executed"] == 10
