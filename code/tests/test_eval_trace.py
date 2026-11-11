"""Durable eval trace with unchanged prompts, calls and recovery behavior."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.eval_trace import EvalTraceWriter
from test_baseline import CATALOG, FakeEnv, StubASR, item


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_writer_flushes_each_event_and_snapshots_full_mutable_results(tmp_path):
    path = tmp_path / 'trace.jsonl'
    text = '  GIỮ NGUYÊN!\nId_X <tag> ' * 500
    result = {'ok': True, 'rows': [{'text': text}], 'tail': 'Không cắt!'}
    with path.open('w', encoding='utf-8') as stream:
        writer = EvalTraceWriter(stream)
        emit = writer.for_sample('train_1')
        emit({'event': 'call_result', 'step': 1, 'result': result})
        first = read_events(path)[0]  # Still inside with: the writer must flush.
        assert first['result'] == result and first['result']['rows'][0]['text'] == text
        result['rows'][0]['text'] = 'mutated later'
        writer.for_sample('train_2')({'event': 'sample_start'})
        emit({'event': 'call_result', 'step': 2, 'result': result})
    events = read_events(path)
    assert events[0]['result']['rows'][0]['text'] == text
    assert events[-1]['result']['rows'][0]['text'] == 'mutated later'
    assert [e['event_index'] for e in events] == [1, 2, 3]
    assert [e['sample_id'] for e in events] == ['train_1', 'train_2', 'train_1']
    assert all(e['version'] == 1 and e['elapsed_seconds'] >= 0 for e in events)


def test_tracing_keeps_prompts_outputs_validation_and_execution_identical(tmp_path):
    outputs = ['not a decision',
        '{"action":"call","name":"smart_home.light_set","arguments":{"power":"on"}}',
        '<tool_call>\n<function=smart_home.get_status>\n</function>\n</tool_call>',
        '{"action":"call","name":"smart_home.light_set","arguments":{"device_id":"Id_X","power":"on"}}',
        '{"action":"done"}']
    result_text = '  GIỮ Chữ HOA, 08:30!\n ' * 500
    class Env(FakeEnv):
        def __init__(self): super().__init__(); self.validations = []
        def validate(self, name, args):
            self.validations.append({'name': name, 'arguments': deepcopy(args)})
            return super().validate(name, args)
        def call(self, name, args):
            super().call(name, args)
            return {'ok': True, 'text': result_text}
    class LLM:
        def __init__(self): self.requests = []
        def generate(self, system, user):
            self.requests.append({'system': system, 'user': user})
            return outputs[len(self.requests) - 1]
    def run(trace=None):
        llm, env = LLM(), Env()
        stats = LifeMateAgent(StubASR(), llm, CATALOG, AgentConfig(enforce_observed_xref=False)).run(
            item(), env, Path('/unused'), 'train', trace=trace)
        return llm.requests, env, stats
    untraced_requests, untraced_env, untraced_stats = run()
    path = tmp_path / 'trace.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        traced_requests, env, stats = run(EvalTraceWriter(stream).for_sample('train_0001'))
    assert traced_requests == untraced_requests
    # Both errors survive retrying; after the next actual call only executed
    # calls/results remain visible, while the trace below retains the errors.
    pending = json.loads(traced_requests[2]['user'])['previous_actions_and_results']
    assert [observation['stage'] for observation in pending] == ['parse', 'validate']
    for request, expected_names in zip(traced_requests[3:], [
            ['smart_home.get_status'], ['smart_home.get_status', 'smart_home.light_set']]):
        observations = json.loads(request['user'])['previous_actions_and_results']
        assert [observation['name'] for observation in observations] == expected_names
        assert all('result' in observation and 'stage' not in observation for observation in observations)
    assert env.calls == untraced_env.calls and env.validations == untraced_env.validations
    assert {k: v for k, v in stats.items() if k != 'elapsed_seconds'} == {
        k: v for k, v in untraced_stats.items() if k != 'elapsed_seconds'}
    assert stats['executed'] == 2 and stats['invalid_proposals'] == 2
    events = read_events(path)
    assert [e['request'] for e in events if e['event'] == 'llm_start'] == traced_requests
    assert [e['output'] for e in events if e['event'] == 'llm_output'] == outputs
    rejected = [e for e in events if e['event'] == 'decision_rejected']
    assert [e['stage'] for e in rejected] == ['parse', 'validate']
    assert rejected[0]['raw_output'] == outputs[0]
    assert rejected[1]['name'] == 'smart_home.light_set'
    assert rejected[1]['arguments'] == {'power': 'on'}
    assert rejected[1]['validation'] == {'ok': False, 'error': 'missing required'}
    validations = [e for e in events if e['event'] == 'validate_result']
    assert [e['accepted'] for e in validations] == [False, True, True]
    assert validations[0]['arguments'] == {'power': 'on'}
    assert validations[0]['result'] == {'ok': False, 'error': 'missing required'}
    results = [e for e in events if e['event'] == 'call_result']
    assert [e['call_index'] for e in results] == [0, 1]
    assert all(e['result'] == {'ok': True, 'text': result_text} for e in results)
    assert results[-1]['arguments']['device_id'] == 'Id_X'
    assert [e['step'] for e in events if e['event'] == 'llm_start'] == list(range(1, 6))
    assert events[-1]['event'] == 'stop' and events[-1]['reason'] == 'done'


@pytest.mark.parametrize('validation', [None, True, {}, {'ok': False, 'error': 'Sai tham số!'}])
def test_rejected_validator_returns_are_saved_without_execution(tmp_path, validation):
    class Env(FakeEnv):
        def validate(self, name, args): return validation
    class LLM:
        def generate(self, system, user):
            return '{"action":"call","name":"smart_home.get_status","arguments":{}}'
    path = tmp_path / 'trace.jsonl'
    env = Env()
    with path.open('w', encoding='utf-8') as stream:
        stats = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig(max_invalid=1)).run(
            item(), env, Path('/unused'), 'train', trace=EvalTraceWriter(stream).for_sample('sample'))
    events = read_events(path)
    result = next(e for e in events if e['event'] == 'validate_result')
    assert result['result'] == validation and result['accepted'] is False
    assert not any(e['event'] in ('validate_error', 'call_start', 'call_result') for e in events)
    assert not env.calls and stats['stop_reason'] == 'max_invalid'


@pytest.mark.parametrize('phase', ['validate', 'call'])
def test_validator_and_execution_exceptions_are_distinguished_from_returns(tmp_path, phase):
    class Env(FakeEnv):
        def validate(self, name, args):
            if phase == 'validate': raise RuntimeError('Validator unavailable')
            return {'ok': True}
        def call(self, name, args):
            super().call(name, args)
            raise ValueError('Call failed')
    class LLM:
        count = 0
        def generate(self, system, user):
            self.count += 1
            return ('{"action":"call","name":"smart_home.get_status","arguments":{}}'
                    if self.count == 1 else '{"action":"done"}')
    path = tmp_path / 'trace.jsonl'
    llm, env = LLM(), Env()
    with path.open('w', encoding='utf-8') as stream:
        stats = LifeMateAgent(StubASR(), llm, CATALOG, AgentConfig()).run(
            item(), env, Path('/unused'), 'train', trace=EvalTraceWriter(stream).for_sample('sample'))
    events = read_events(path)
    error = next(e for e in events if e['event'] == f'{phase}_error')
    assert error['arguments'] == {} and error['name'] == 'smart_home.get_status'
    assert error['exception']['type'] == ('RuntimeError' if phase == 'validate' else 'ValueError')
    assert not any(e['event'] == f'{phase}_result' for e in events)
    assert llm.count == 2
    assert len(env.calls) == stats['executed'] == (1 if phase == 'call' else 0)
    assert stats['invalid_proposals'] == (1 if phase == 'validate' else 0)
    if phase == 'call':
        assert error['agent_result'] == {'execution_exception': error['exception']}
        assert 'ok' not in error['agent_result']


def test_execution_returning_ok_false_is_logged_as_a_return(tmp_path):
    class Env(FakeEnv):
        def call(self, name, args):
            super().call(name, args)
            return {'ok': False, 'error': 'Tool execution rejected'}
    class LLM:
        def generate(self, system, user):
            return '{"action":"call","name":"smart_home.get_status","arguments":{}}'
    env = Env()
    path = tmp_path / 'trace.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        stats = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig(max_calls=1)).run(
            item(), env, Path('/unused'), 'train', trace=EvalTraceWriter(stream).for_sample('sample'))
    events = read_events(path)
    assert next(e for e in events if e['event'] == 'call_result')['result'] == {
        'ok': False, 'error': 'Tool execution rejected'}
    assert not any(e['event'] == 'call_error' for e in events)
    assert stats['executed'] == 1 and stats['invalid_proposals'] == 0


@pytest.mark.parametrize('decision', [
    {'action': 'call', 'name': 'unknown.tool', 'arguments': {}},
    {'action': 'call', 'name': 'smart_home.get_status', 'arguments': []},
    {'action': 'call', 'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'unknown.tool'}},
])
def test_precheck_rejection_is_logged_without_invoking_toolenv(tmp_path, decision):
    class Env(FakeEnv):
        def validate(self, name, args): pytest.fail('Precheck must reject before validate')
        def call(self, name, args): pytest.fail('Precheck must reject before execution')
    class LLM:
        def generate(self, system, user): return json.dumps(decision)
    path = tmp_path / 'trace.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig(max_invalid=1)).run(
            item(), Env(), Path('/unused'), 'train', trace=EvalTraceWriter(stream).for_sample('sample'))
    events = read_events(path)
    assert next(e for e in events if e['event'] == 'decision_parsed')['decision'] == decision
    assert next(e for e in events if e['event'] == 'decision_rejected')['stage'] == 'precheck'
    assert not any(e['event'].startswith(('validate_', 'call_')) for e in events)
