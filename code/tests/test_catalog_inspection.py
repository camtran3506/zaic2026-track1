"""Catalog recall and schema preservation across SFT and real agent decisions."""
import json
from pathlib import Path

from lifemate.agent import (AgentConfig, LifeMateAgent, build_step_prompt,
                            fit_prompt_payload, parse_decision)
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


def test_low_ranked_gold_is_retained_via_supervised_inspection():
    catalog, target = large_catalog()
    env = CatalogEnv()
    rows = replay_sample(sample(catalog, target), catalog, env, Path('/unused'), max_tool_details=1)
    decisions = [json.loads(row['messages'][-1]['content']) for row in rows]
    assert [d['action'] for d in decisions] == ['inspect', 'call', 'done']
    assert decisions[0] == {'action': 'inspect', 'name': target}
    before = json.loads(rows[0]['messages'][1]['content'])['available_tools']
    after = json.loads(rows[1]['messages'][1]['content'])['available_tools']
    assert target in {t['name'] for t in before['tool_index']}
    assert target not in {t['name'] for t in before['tool_schemas']}
    assert catalog[target]['parameters'] == next(t['parameters'] for t in after['tool_schemas'] if t['name'] == target)
    assert len(env.calls) == 1  # inspect never executes an organizer tool
    assert [r['step'] for r in rows] == [0, 1, 2]


def test_inference_can_inspect_then_execute_outside_initial_candidates():
    catalog, target = large_catalog()
    entry = sample(catalog, target)
    class Planner:
        def __init__(self): self.prompts = []
        def generate(self, system, user):
            self.prompts.append(user)
            available = json.loads(user)['available_tools']
            if target not in {t['name'] for t in available['tool_schemas']}:
                return json.dumps({'action': 'inspect', 'name': target})
            if len(self.prompts) == 2:
                return json.dumps({'action': 'call', 'name': target, 'arguments': {'mode': 'on'}})
            return '{"action":"done"}'
    llm, env = Planner(), CatalogEnv()
    stats = LifeMateAgent(StubASR(), llm, catalog, AgentConfig(max_tool_details=1)).run(entry, env, Path('/unused'), 'train')
    assert stats['executed'] == 1 and stats['schema_inspections'] == 1
    assert stats['invalid_proposals'] == 0
    assert env.calls == [{'name': target, 'arguments': {'mode': 'on'}}]
    assert parse_decision('{"action":"inspect","name":"x.y"}') == {'action': 'inspect', 'name': 'x.y'}


def test_size_fitting_protects_requested_schema_and_retains_complete_index():
    catalog, target = large_catalog()
    catalog[target]['parameters']['properties']['mode']['description'] = 'Thông tin ' * 500
    tools, prompt = build_step_prompt('Thiết bị', {}, [], catalog, [], None, 1, target)
    assert tools == json.loads(prompt)['available_tools']
    info = json.loads(prompt)
    info['available_tools']['tool_schemas'] += [
        {'name': f'smart_home.big_{i}', 'parameters': {'description': 'x' * 8000}} for i in range(5)]
    fitted = json.loads(fit_prompt_payload(info, max_chars=11000, protected_tools=(target,)))
    specs = {t['name']: t for t in fitted['available_tools']['tool_schemas']}
    assert specs[target]['parameters'] == catalog[target]['parameters']
    assert {t['name'] for t in fitted['available_tools']['tool_index']} == set(catalog)
    assert len(json.dumps(fitted, ensure_ascii=False, separators=(',', ':'))) <= 11000
