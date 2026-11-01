"""Guard checks must prevent calls, not merely produce valid-looking JSON."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.guard import CallGuard, GuardError, canonical_arguments, repair_arguments, validated_arguments
from lifemate.sft import replay_sample
from test_baseline import CATALOG, FakeEnv, StubASR, StubLLM, item


@pytest.mark.parametrize(('value', 'schema', 'expected'), [
    ('2', {'type': 'integer'}, 2),
    ('2.0', {'type': 'integer'}, 2),
    ('2.5', {'type': 'integer'}, '2.5'),
    (True, {'type': 'integer'}, True),
    ('1,000', {'type': 'number'}, '1,000'),
    ('NaN', {'type': 'number'}, 'NaN'),
    ('2.5', {'type': 'number'}, 2.5),
    (' FALSE ', {'type': 'boolean'}, False),
    ('yes', {'type': 'boolean'}, 'yes'),
    (123, {'type': 'string'}, '123'),
    ('ON', {'type': 'string', 'enum': ['on', 'off']}, 'on'),
    ('on', {'type': 'string', 'enum': ['ON', 'On']}, 'on'),
    ('9h30', {'type': 'string', 'pattern': r'^\d{2}:\d{2}$'}, '09:30'),
    ('9 giờ 30', {'type': 'string', 'format': 'time'}, '09:30:00'),
    ('25:30', {'type': 'string', 'format': 'time'}, '25:30'),
    ('09', {'type': 'string', 'format': 'time'}, '09'),
    ('2026-10-31', {'type': 'string', 'format': 'date-time'}, '2026-10-31'),
    ('31/10/2026', {'type': 'string', 'format': 'date'}, '2026-10-31'),
    ('31/02/2026', {'type': 'string', 'format': 'date'}, '31/02/2026'),
    ('03/04/2026', {'type': 'string', 'format': 'date'}, '03/04/2026'),
    ('ngày mai', {'type': 'string', 'format': 'date'}, 'ngày mai'),
])
def test_only_unambiguous_schema_directed_repairs(value, schema, expected):
    actual = repair_arguments(value, schema)
    assert actual == expected and type(actual) is type(expected)


def test_nested_types_preserve_unknown_fields_and_do_not_mutate_input():
    value = {'rows': [{'count': '3', 'on': 'true'}], 'unknown': 'keep me'}
    schema = {'type': 'object', 'properties': {'rows': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'count': {'type': 'integer'}, 'on': {'type': 'boolean'}}}}}}
    assert repair_arguments(value, schema) == {'rows': [{'count': 3, 'on': True}], 'unknown': 'keep me'}
    assert value['rows'][0]['count'] == '3'
    assert repair_arguments('1', {'oneOf': [{'type': 'integer'}, {'type': 'string'}]}) == '1'


def test_datetime_requires_explicit_or_unambiguous_context_timezone():
    schema = {'type': 'string', 'format': 'date-time'}
    text = '2026-10-31 09:30:00'
    assert repair_arguments(text, schema) == text
    assert repair_arguments(text, schema, {'timezone': 'Asia/Ho_Chi_Minh'}) == '2026-10-31T09:30:00+07:00'
    assert repair_arguments(text, schema, {'timezone': 'missing'}) == text
    for local in ('2026-11-01 01:30:00', '2026-03-08 02:30:00'):
        assert repair_arguments(local, schema, {'timezone': 'America/New_York'}) == local
    assert repair_arguments('2026-10-31T09:30:00Z', schema) == '2026-10-31T09:30:00+00:00'
    assert repair_arguments('2026-10-31', schema, {'timezone': 'Asia/Ho_Chi_Minh'}) == '2026-10-31'


def test_valid_arguments_preserved_and_repair_must_pass_authoritative_validator():
    class Env:
        def __init__(self): self.validations = []
        def validate(self, name, args):
            self.validations.append(deepcopy(args))
            return {'ok': args == {'n': 2} and type(args['n']) is int}
    env = Env()
    schema = {'type': 'object', 'properties': {'n': {'type': 'integer'}}}
    assert validated_arguments(env, 'app.write', {'n': '2'}, schema, {}) == {'n': 2}
    assert env.validations == [{'n': '2'}, {'n': 2}]
    assert type(validated_arguments(env, 'app.write', {'n': 2.0}, schema, {})['n']) is int
    with pytest.raises(GuardError):
        validated_arguments(env, 'app.write', {'n': '3'}, schema, {})
    accepted = {'n': 2}
    assert validated_arguments(env, 'app.write', accepted, schema, {}) is accepted


def test_read_provenance_is_resource_scoped_and_failed_results_do_not_authorize():
    catalog = deepcopy(CATALOG)
    catalog['smart_home.other'] = {'kind': 'read', 'parameters': {}}
    catalog['elsewhere.devices'] = {'kind': 'read', 'parameters': {}}
    guard = CallGuard(catalog)
    args = {'device_id': 'lamp', 'power': 'on'}
    for name, result in [
        ('smart_home.get_status', {'ok': False, 'devices': [{'id': 'lamp'}]}),
        ('smart_home.get_status', {'status': 'error', 'devices': [{'id': 'lamp'}]}),
        ('smart_home.get_status', {'contacts': [{'id': 'lamp'}]}),
        ('smart_home.get_status', {'devices': [{'name': 'lamp'}]}),
        ('smart_home.get_status', {'devices': [{'id': 'other', 'contacts': [{'id': 'lamp'}]}]}),
        ('elsewhere.devices', {'devices': [{'id': 'lamp'}]}),
    ]:
        guard.observe(name, {}, result)
        with pytest.raises(GuardError, match='Unread ID'):
            guard.check('smart_home.light_set', args)
    guard.observe('smart_home.get_status', {}, {'data': {'devices': [{'id': 'lamp'}]}})
    guard.check('smart_home.light_set', args)


def test_read_dedup_is_canonical_untruncated_and_only_successful_writes_reset_it():
    guard = CallGuard(CATALOG)
    a = {'a': 1, 'b': {'n': 2}}
    b = {'b': {'n': 2.0}, 'a': 1.0}
    assert canonical_arguments(a) == canonical_arguments(b)
    guard.observe('smart_home.get_status', a, {'ok': True})
    with pytest.raises(GuardError, match='Duplicate read'):
        guard.check('smart_home.get_status', b)
    guard.observe('smart_home.light_set', {}, {'ok': False})
    with pytest.raises(GuardError, match='Duplicate read'):
        guard.check('smart_home.get_status', b)
    guard.observe('smart_home.light_set', {}, {'ok': True})
    guard.check('smart_home.get_status', b)
    guard.observe('smart_home.get_status', {'q': 'x' * 6000 + 'a'}, {'ok': True})
    guard.check('smart_home.get_status', {'q': 'x' * 6000 + 'b'})


def test_unref_ids_arrays_and_unwrapped_detail_results():
    catalog = deepcopy(CATALOG)
    catalog['smart_home.light_set']['parameters'] = {'type': 'object', 'properties': {
        'device_ids': {'type': 'array', 'items': {'type': 'string'}}}}
    catalog['smart_home.get_device'] = {'kind': 'read'}
    guard = CallGuard(catalog)
    guard.observe('smart_home.get_device', {}, [{'id': 'one'}, {'id': 'two'}])
    guard.check('smart_home.light_set', {'device_ids': ['one', 'two']})
    with pytest.raises(GuardError, match='Unread ID'):
        guard.check('smart_home.light_set', {'device_ids': ['one', 'three']})


def test_composed_reference_schema_cannot_bypass_provenance():
    catalog = deepcopy(CATALOG)
    catalog['smart_home.light_set']['parameters'] = {'type': 'object', 'properties': {
        'target': {'anyOf': [{'type': 'null'}, {'type': 'string', 'x-ref': 'smart_home.devices'}]}}}
    guard = CallGuard(catalog)
    with pytest.raises(GuardError, match='Unread ID'):
        guard.check('smart_home.light_set', {'target': 'lamp'})
    guard.observe('smart_home.get_status', {}, {'devices': [{'id': 'lamp'}]})
    guard.check('smart_home.light_set', {'target': 'lamp'})


def test_agent_blocks_unread_write_and_executes_only_validated_repaired_arguments():
    catalog = deepcopy(CATALOG)
    props = catalog['smart_home.light_set']['parameters']['properties']
    props.update({'level': {'type': 'integer'}, 'enabled': {'type': 'boolean'},
                  'at': {'type': 'string', 'pattern': r'^\d{2}:\d{2}$'}})
    accepted = {'device_id': 'light_living', 'power': 'on', 'level': 2, 'enabled': False, 'at': '09:30'}
    class Env(FakeEnv):
        def validate(self, name, args):
            if name.endswith('get_status'):
                return {'ok': True}
            valid = dict(args, device_id='light_living') == accepted
            valid = valid and type(args.get('level')) is int and type(args.get('enabled')) is bool
            return {'ok': valid}
    proposals = [
        {'action': 'call', 'name': 'smart_home.light_set', 'arguments': dict(accepted, device_id='invented')},
        {'action': 'call', 'name': 'smart_home.get_status', 'arguments': {}},
        {'action': 'call', 'name': 'smart_home.get_status', 'arguments': {}},
        {'action': 'call', 'name': 'smart_home.light_set', 'arguments': dict(accepted, level='2', enabled='false', at='9h30')},
        {'action': 'done'},
    ]
    env = Env()
    report = LifeMateAgent(StubASR(), StubLLM(map(json.dumps, proposals)), catalog, AgentConfig()).run(
        item(), env, Path('/unused'), 'train')
    assert report['invalid_proposals'] == 2
    assert env.calls == [{'name': 'smart_home.get_status', 'arguments': {}},
                         {'name': 'smart_home.light_set', 'arguments': accepted}]


def test_sft_replay_rejects_gold_using_unread_id():
    entry = dict(item(), audio_text='Bật đèn', label={'tool_calls': [
        {'name': 'smart_home.light_set', 'arguments': {'device_id': 'invented', 'power': 'on'}}]})
    env = FakeEnv()
    with pytest.raises(ValueError, match='Unread ID'):
        replay_sample(entry, CATALOG, env, Path('/unused'))
    assert env.calls == []
