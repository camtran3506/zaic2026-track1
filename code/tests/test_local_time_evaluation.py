"""Scores come from actual ToolEnv logs, not internal helper observations."""
import json

from lifemate.evaluation import StateToolEvaluator
from test_evaluation import prepared


def test_evaluator_keeps_helper_out_of_official_calls_and_state_score(prepared, tmp_path):
    out, _, _, _ = prepared
    evaluator = StateToolEvaluator(out, json.loads((out / "manifest.json").read_text()))

    class Planner:
        def generate(self, system, user):
            observations = json.loads(user)["previous_actions_and_results"]
            local = [r for r in observations if r.get("scope") == "local"]
            official = [r for r in observations if r.get("scope") != "local"]
            if not local:
                return json.dumps({"action": "call", "name": "local_time.duration_between", "arguments": {
                    "start": "2026-10-13T19:30:00+07:00", "end": "2026-10-13T20:10:00+07:00"}})
            assert json.loads(local[0]["result"]) == {"ok": True, "minutes": 40}
            if not official:
                return '{"action":"call","name":"smart_home.get_status","arguments":{}}'
            if len(official) == 1:
                return ('{"action":"call","name":"smart_home.light_set",'
                        '"arguments":{"device_id":"light_living","power":"on"}}')
            return '{"action":"done"}'

    try:
        metrics = evaluator.evaluate(Planner(), tmp_path / "with-helper.jsonl")
        assert metrics["eval_state_tool_score"] == 100
        assert metrics["eval_tool_call_f1"] == 100
        row = json.loads((tmp_path / "with-helper.jsonl").read_text())
        assert row["tool_calls"] == row["gold_tool_calls"]
        assert row["agent_stats"]["local_executed"] == 1
        assert row["agent_stats"]["executed"] == 2
        trace = [json.loads(l) for l in (tmp_path / "with-helper.trace.jsonl").read_text().splitlines()]
        assert sum(e["event"] == "local_call_result" for e in trace) == 1
        assert all(not e["name"].startswith("local_time.") for e in trace if e["event"] == "call_start")
    finally:
        evaluator.close()
