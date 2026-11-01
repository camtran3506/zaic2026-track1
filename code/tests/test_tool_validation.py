"""ToolEnv alone decides argument validity in SFT and inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.sft import replay_sample
from test_baseline import CATALOG, FakeEnv, StubASR, StubLLM, item


def run_agent(calls, env):
    outputs = [json.dumps({'action': 'call', **call}) for call in calls]
    outputs.append('{"action":"done"}')
    return LifeMateAgent(StubASR(), StubLLM(outputs), CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')


def replay(calls, env):
    entry = dict(item(), audio_text='Bật đèn', label={'tool_calls': calls})
    return replay_sample(entry, CATALOG, env, Path('/unused'))


def test_toolenv_accepted_ids_and_duplicate_reads_are_executed_in_both_paths():
    calls = [
        {'name': 'smart_home.light_set', 'arguments': {'device_id': 'not_previously_read', 'power': 'on'}},
        {'name': 'smart_home.get_status', 'arguments': {}},
        {'name': 'smart_home.get_status', 'arguments': {}},
    ]
    for runner in (run_agent, replay):
        env = FakeEnv()
        runner(calls, env)
        assert env.calls == calls


def test_arguments_rejected_by_toolenv_are_never_coerced_or_executed():
    class Env(FakeEnv):
        def __init__(self):
            super().__init__()
            self.validations = []
        def validate(self, name, args):
            self.validations.append(deepcopy(args))
            return {'ok': type(args['device_id']) is int}
    rejected = {'name': 'smart_home.light_set', 'arguments': {'device_id': '2'}}
    corrected = {'name': 'smart_home.light_set', 'arguments': {'device_id': 2}}
    env = Env()
    report = run_agent([rejected, corrected], env)
    assert report['invalid_proposals'] == 1
    assert env.calls == [corrected]
    assert env.validations == [rejected['arguments'], corrected['arguments']]
    env = Env()
    with pytest.raises(ValueError, match='gold schema invalid'):
        replay([rejected], env)
    assert not env.calls
    assert env.validations == [rejected['arguments']]


@pytest.mark.parametrize('response', [None, True, {}, {'ok': False}, {'error': 'invalid'}])
def test_validation_must_explicitly_accept_before_either_path_executes(response):
    class Env(FakeEnv):
        def validate(self, name, args): return response
    call = {'name': 'smart_home.get_status', 'arguments': {}}
    env = Env()
    assert run_agent([call], env)['invalid_proposals'] == 1
    assert not env.calls
    with pytest.raises(ValueError, match='gold schema invalid'):
        replay([call], env)
    assert not env.calls


def test_validate_exception_is_returned_to_model_without_execution():
    class Env(FakeEnv):
        def validate(self, name, args): raise RuntimeError('validator unavailable')
    class LLM:
        def __init__(self): self.steps = 0
        def generate(self, system, user):
            self.steps += 1
            if self.steps == 1:
                return '{"action":"call","name":"smart_home.get_status","arguments":{}}'
            assert 'validator unavailable' in user
            return '{"action":"done"}'
    env = Env()
    report = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert report['invalid_proposals'] == 1
    assert not env.calls


def test_execution_error_is_returned_to_model_after_successful_validation():
    class Env(FakeEnv):
        def call(self, name, args):
            self.calls.append({'name': name, 'arguments': args})
            raise ValueError('ID does not exist')
    class LLM:
        def __init__(self): self.steps = 0
        def generate(self, system, user):
            self.steps += 1
            if self.steps == 1:
                return '{"action":"call","name":"smart_home.light_set","arguments":{"device_id":"unknown"}}'
            assert 'ID does not exist' in user
            return '{"action":"done"}'
    env = Env()
    report = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert report['executed'] == 1 and report['invalid_proposals'] == 0
    assert len(env.calls) == 1
