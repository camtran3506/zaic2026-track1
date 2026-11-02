"""Every allowed schema stays complete in SFT and real agent decisions."""
from copy import deepcopy
import json
from pathlib import Path

from lifemate.agent import AgentConfig, LifeMateAgent, build_step_prompt, parse_decision
from lifemate.sft import replay_sample
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


def test_sft_exposes_every_schema_from_the_first_step_without_inspection():
    catalog, target = large_catalog()
    env = CatalogEnv()
    rows = replay_sample(sample(catalog, target), catalog, env, Path('/unused'))
    decisions = [json.loads(row['messages'][-1]['content']) for row in rows]
    assert decisions == [{'action': 'call', 'name': target, 'arguments': {'mode': 'on'}},
                         {'action': 'done'}]
    for row in rows:
        available = json.loads(row['messages'][1]['content'])['available_tools']
        assert {t['name']: t for t in available['tool_schemas']} == catalog
        assert {t['name'] for t in available['tool_index']} == set(catalog)
    assert env.calls == [{'name': target, 'arguments': {'mode': 'on'}}]
    assert [r['step'] for r in rows] == [0, 1]


def test_inference_can_execute_any_allowed_tool_without_inspection():
    catalog, target = large_catalog()
    entry = sample(catalog, target)
    class Planner:
        def __init__(self): self.prompts = []
        def generate(self, system, user):
            self.prompts.append(user)
            available = json.loads(user)['available_tools']
            assert {t['name']: t for t in available['tool_schemas']} == catalog
            if len(self.prompts) == 1:
                return json.dumps({'action': 'call', 'name': target, 'arguments': {'mode': 'on'}})
            return '{"action":"done"}'
    llm, env = Planner(), CatalogEnv()
    stats = LifeMateAgent(StubASR(), llm, catalog, AgentConfig()).run(entry, env, Path('/unused'), 'train')
    assert stats['executed'] == 1 and stats['schema_inspections'] == 0
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
    assert {t['name']: t for t in tools['tool_schemas']} == catalog
    assert next(t['description'] for t in tools['tool_index'] if t['name'] == target) == catalog[target]['description']
    assert payload['user_context'] == {'profile': {'notes': 'x' * 6000}}
    assert payload['conversation_history'] == history
    assert payload['previous_actions_and_results'] == observations
    assert (catalog, history, observations) == original


def test_legacy_inspection_does_not_hide_or_replace_schemas():
    catalog, target = large_catalog()
    class Planner:
        def __init__(self): self.prompts = []
        def generate(self, system, user):
            self.prompts.append(user)
            if len(self.prompts) == 1:
                return json.dumps({'action': 'inspect', 'name': target})
            assert user == self.prompts[0]
            return '{"action":"done"}'
    llm, env = Planner(), CatalogEnv()
    stats = LifeMateAgent(StubASR(), llm, catalog, AgentConfig()).run(
        sample(catalog, target), env, Path('/unused'), 'train')
    assert stats['schema_inspections'] == 1 and not env.calls
    assert parse_decision('{"action":"inspect","name":"x.y"}') == {'action': 'inspect', 'name': 'x.y'}
