"""Every allowed schema stays complete in SFT and real agent decisions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent, build_step_prompt, parse_decision
from lifemate.sft import replay_sample
from lifemate.local_helpers import with_local_tools
from test_baseline import CATALOG, FakeEnv, StubASR, item


def large_catalog():
    catalog = dict(CATALOG)
    for i in range(60):
        name = f'smart_home.read_{i:02d}'
        catalog[name] = {'name': name, 'kind': 'read', 'description': 'Thiết bị',
                         'parameters': {'type': 'object', 'properties': {}}}
    name = 'smart_home.zz_custom'
    catalog[name] = {'name': name, 'kind': 'write', 'description': 'Tùy chỉnh',
        'parameters': {'type': 'object', 'properties': {
            'mode': {'type': 'string', 'enum': ['on', 'off']}}, 'required': ['mode']}}
    return catalog, name


def sample(catalog, target):
    entry = item()
    entry['context']['installed_apps']['smart_home']['tools'] = list(catalog)
    entry['audio_text'] = 'Thực hiện yêu cầu'
    entry['label'] = {'tool_calls': [{'name': target, 'arguments': {'mode': 'on'}}]}
    return entry


class CatalogEnv(FakeEnv):
    def validate(self, name, args):
        return {'ok': name == 'smart_home.zz_custom' and args == {'mode': 'on'}}


def test_sft_exposes_every_schema_from_the_first_step():
    catalog, target = large_catalog()
    env = CatalogEnv()
    rows = replay_sample(sample(catalog, target), catalog, env, Path('/unused'))
    decisions = [json.loads(row['messages'][-1]['content']) for row in rows]
    assert decisions == [{'action': 'call', 'name': target, 'arguments': {'mode': 'on'}},
                         {'action': 'done'}]
    for row in rows:
        available = json.loads(row['messages'][1]['content'])['available_tools']
        assert {t['name']: t for t in available['tool_schemas']} == with_local_tools(catalog)
        assert {t['name'] for t in available['tool_index']} == set(with_local_tools(catalog))
    assert env.calls == [{'name': target, 'arguments': {'mode': 'on'}}]
    assert [r['step'] for r in rows] == [0, 1]


def test_inference_can_execute_any_allowed_tool():
    catalog, target = large_catalog()
    entry = sample(catalog, target)
    class Planner:
        def __init__(self): self.prompts = []
        def generate(self, system, user):
            self.prompts.append(user)
            available = json.loads(user)['available_tools']
            assert {t['name']: t for t in available['tool_schemas']} == with_local_tools(catalog)
            if len(self.prompts) == 1:
                return json.dumps({'action': 'call', 'name': target, 'arguments': {'mode': 'on'}})
            return '{"action":"done"}'
    llm, env = Planner(), CatalogEnv()
    stats = LifeMateAgent(StubASR(), llm, catalog, AgentConfig()).run(entry, env, Path('/unused'), 'train')
    assert stats['executed'] == 1
    assert stats['invalid_proposals'] == 0
    assert env.calls == [{'name': target, 'arguments': {'mode': 'on'}}]


def test_large_payload_preserves_all_fields_without_size_fitting():
    catalog, target = large_catalog()
    catalog[target].update({
        'description': 'Mô tả chi tiết ' * 500,
        'returns': {'type': 'object', 'properties': {'note': {
            'type': 'string', 'description': 'Kết quả đầy đủ ' * 500}}},
        'x-custom': {'nested': ['giữ nguyên', {'value': 42}]},
        'additionalProperties': False,
    })
    catalog[target]['parameters']['properties']['mode']['description'] = 'Thông tin ' * 500
    history = [{'role': 'user', 'text': f'Lượt {i}: ' + 'Lịch sử! ' * 1000} for i in range(20)]
    observations = [{'name': target, 'arguments': {'mode': 'on'},
                     'result': 'Kết quả ' * 1000 + str(i)} for i in range(12)]
    original = deepcopy((catalog, history, observations))
    tools, prompt = build_step_prompt('Thiết bị', {'profile': {'notes': 'x' * 6000}},
                                      history, catalog, observations)
    payload = json.loads(prompt)
    assert len(prompt) > 20000
    assert tools == payload['available_tools']
    assert {t['name']: t for t in tools['tool_schemas']} == with_local_tools(catalog)
    assert next(t['description'] for t in tools['tool_index'] if t['name'] == target) == catalog[target]['description']
    assert payload['user_context'] == {'profile': {'notes': 'x' * 6000}}
    assert payload['conversation_history'] == [
        {'role': 'user', 'text': f'Lượt {i}: ' + 'lịch sử! ' * 999 + 'lịch sử!'} for i in range(20)]
    assert payload['previous_actions_and_results'] == observations
    assert (catalog, history, observations) == original


@pytest.mark.parametrize('arguments', [None, {}, {'action': 'done'}])
def test_removed_action_is_invalid_even_with_arguments(arguments):
    catalog, target = large_catalog()
    decision = {'action': 'inspect', 'name': target}
    if arguments is not None:
        decision['arguments'] = arguments
    raw = json.dumps(decision)
    with pytest.raises(ValueError, match='Unsupported agent action'):
        parse_decision(raw)
    class Planner:
        def __init__(self): self.prompts = []
        def generate(self, system, user):
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return raw
            feedback = json.loads(user)['previous_actions_and_results']
            assert 'Unsupported agent action' in feedback[-1]['error']
            return '{"action":"done"}'
    llm, env = Planner(), CatalogEnv()
    stats = LifeMateAgent(StubASR(), llm, catalog, AgentConfig()).run(
        sample(catalog, target), env, Path('/unused'), 'train')
    assert stats['invalid_proposals'] == 1 and not env.calls
    assert stats['stop_reason'] == 'done'
    assert 'schema_inspections' not in stats


def test_repeated_removed_action_uses_invalid_budget_and_never_executes():
    catalog, target = large_catalog()
    class Planner:
        count = 0
        def generate(self, system, user):
            self.count += 1
            return json.dumps({'action': 'inspect', 'name': target})
    llm, env = Planner(), CatalogEnv()
    stats = LifeMateAgent(StubASR(), llm, catalog, AgentConfig()).run(
        sample(catalog, target), env, Path('/unused'), 'train')
    assert llm.count == stats['invalid_proposals'] == 3
    assert stats['stop_reason'] == 'max_invalid'
    assert not env.calls
