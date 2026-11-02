"""ToolEnv alone decides argument validity in SFT and inference."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.function_calling import native_chat
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


def rejection_feedback(system, user):
    """Inspect both portable and Qwen-native inputs at the next planning step."""
    feedback, = json.loads(user)['previous_actions_and_results']
    chat, _ = native_chat([{'role': 'system', 'content': system},
                           {'role': 'user', 'content': user}])
    assert chat[-1]['role'] == 'user'
    assert json.loads(chat[-1]['content']) == {'agent_feedback': feedback}
    assert not any(turn['role'] == 'tool' or turn.get('tool_calls') for turn in chat)
    assert 'result' not in feedback  # No tool was executed for this proposal.
    return feedback


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


@pytest.mark.parametrize('validation', [None, True, {}, {
    'ok': False,
    'error': 'Sai device_id!',
    'details': {'received': '  HALLUCINATED_123\n', 'expected': ['Device_A', 'Device_B'],
                'explanation': ' Giữ NGUYÊN văn bản!\n' * 300},
}])
def test_full_rejected_arguments_and_validation_reach_next_native_planning_step(validation):
    proposed = {'device_id': '  HALLUCINATED_123\n', 'power': 'on'}
    corrected = {'device_id': 'Device_A', 'power': 'on'}
    class Env(FakeEnv):
        def __init__(self): super().__init__(); self.validations = []
        def validate(self, name, args):
            self.validations.append(deepcopy(args))
            return validation if len(self.validations) == 1 else {'ok': True}
    class LLM:
        count = 0
        def generate(self, system, user):
            self.count += 1
            if self.count == 2:
                feedback = rejection_feedback(system, user)
                assert feedback['stage'] == 'validate'
                assert feedback['name'] == 'smart_home.light_set'
                assert feedback['arguments'] == proposed
                assert feedback['validation'] == validation
            if self.count == 3:
                observations = json.loads(user)['previous_actions_and_results']
                assert len(observations) == 1
                assert observations[0]['arguments'] == corrected
                assert json.loads(observations[0]['result'])['ok'] is True
                chat, _ = native_chat([{'role': 'system', 'content': system},
                                      {'role': 'user', 'content': user}])
                assert not any('agent_feedback' in turn.get('content', '') for turn in chat)
            return (json.dumps({'action': 'call', 'name': 'smart_home.light_set',
                                'arguments': proposed if self.count == 1 else corrected})
                    if self.count < 3 else '{"action":"done"}')
    env, llm = Env(), LLM()
    report = LifeMateAgent(StubASR(), llm, CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert report['invalid_proposals'] == 1 and report['executed'] == 1
    assert env.validations == [proposed, corrected]
    assert env.calls == [{'name': 'smart_home.light_set', 'arguments': corrected}]


@pytest.mark.parametrize('outcome', ['ok', 'error_return', 'exception'])
def test_rejection_expires_after_execution_but_call_failure_remains_visible(outcome):
    proposed = {'device_id': 'wrong_id', 'power': 'on'}
    corrected = {'device_id': 'Device_A', 'power': 'on'}
    result = ({'ok': True} if outcome == 'ok' else
              {'ok': False, 'error': 'Call failed'} if outcome == 'error_return' else
              {'ok': False, 'error': "RuntimeError('Call failed')"})
    class Env(FakeEnv):
        def validate(self, name, args):
            return {'ok': args == corrected, 'error': 'Wrong ID' if args == proposed else None}
        def call(self, name, args):
            self.calls.append({'name': name, 'arguments': deepcopy(args)})
            if outcome == 'exception':
                raise RuntimeError('Call failed')
            return result
    class LLM:
        count = 0
        def generate(self, system, user):
            self.count += 1
            if self.count == 2:
                assert rejection_feedback(system, user)['arguments'] == proposed
            if self.count == 3:
                assert json.loads(user)['previous_actions_and_results'] == [{
                    'name': 'smart_home.light_set', 'arguments': corrected,
                    'result': json.dumps(result, ensure_ascii=False, separators=(',', ':'))}]
                chat, _ = native_chat([{'role': 'system', 'content': system},
                                      {'role': 'user', 'content': user}])
                assert chat[-2]['tool_calls'][0]['function']['arguments'] == corrected
                assert chat[-1]['role'] == 'tool' and json.loads(chat[-1]['content']) == result
                assert not any('agent_feedback' in turn.get('content', '') for turn in chat)
                return '{"action":"done"}'
            return json.dumps({'action': 'call', 'name': 'smart_home.light_set',
                               'arguments': proposed if self.count == 1 else corrected})
    env = Env()
    report = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert env.calls == [{'name': 'smart_home.light_set', 'arguments': corrected}]
    assert report['executed'] == 1 and report['invalid_proposals'] == 1


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
            feedback = rejection_feedback(system, user)
            assert feedback['stage'] == 'validate'
            assert feedback['name'] == 'smart_home.get_status' and feedback['arguments'] == {}
            assert feedback['exception'] == {'type': 'RuntimeError', 'message': 'validator unavailable',
                                              'repr': "RuntimeError('validator unavailable')"}
            assert 'validation' not in feedback  # The validator raised instead of returning.
            return '{"action":"done"}'
    env = Env()
    report = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert report['invalid_proposals'] == 1
    assert not env.calls


@pytest.mark.parametrize('decision', [
    {'action': 'call', 'name': 'unknown.tool', 'arguments': {'target': 'Tên Gốc'}},
    {'action': 'call', 'name': 'smart_home.get_status', 'arguments': ['unchanged', 1]},
    {'action': 'call', 'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'unknown.tool'}},
])
def test_precheck_rejections_preserve_attempted_call_in_next_prompt(decision):
    class Env(FakeEnv):
        def validate(self, name, args): pytest.fail('Precheck must reject before validate')
        def call(self, name, args): pytest.fail('Precheck must reject before execution')
    class LLM:
        count = 0
        def generate(self, system, user):
            self.count += 1
            if self.count == 1: return json.dumps(decision)
            feedback = rejection_feedback(system, user)
            assert feedback['stage'] == 'precheck'
            assert feedback['name'] == decision['name']
            assert feedback['arguments'] == decision['arguments']
            assert 'validation' not in feedback
            return '{"action":"done"}'
    report = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        item(), Env(), Path('/unused'), 'train')
    assert report['invalid_proposals'] == 1 and report['executed'] == 0


def test_parse_repair_preserves_raw_output_and_respects_native_call_protocol():
    malformed = '<tool_call>\n<function=smart_home.get_status>\nBROKEN '
    class LLM:
        count = 0
        def generate(self, system, user):
            self.count += 1
            if self.count == 1: return malformed
            if self.count == 2:
                feedback = rejection_feedback(system, user)
                assert feedback['stage'] == 'parse' and feedback['raw_output'] == malformed
                assert 'chat template' in feedback['instruction']
                assert 'JSON' not in feedback['instruction']
                return '<tool_call>\n<function=smart_home.get_status>\n</function>\n</tool_call>'
            return '{"action":"done"}'
    env = FakeEnv()
    report = LifeMateAgent(StubASR(), LLM(), CATALOG, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert report['invalid_proposals'] == 1 and report['executed'] == 1
    assert env.calls == [{'name': 'smart_home.get_status', 'arguments': {}}]


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
