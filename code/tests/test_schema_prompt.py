"""Prompt compaction preserves action semantics and official validation inputs."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.catalog import prompt_catalog
from lifemate.local_checks import inspect_call
from lifemate.schema_prompt import SCHEMA_PROMPT_POLICY, compact_tool_schema, _compiled
from lifemate.sft import replay_sample, validate_prompt_policy
from test_local_time_runtime import temporal_artifact


def tool():
    return {'name': 'contacts.create', 'app': 'contacts', 'kind': 'write',
        'description': 'Tạo liên hệ. Giữ nguyên số điện thoại, lấy group_id từ contacts.list_groups.',
        'returns': '', 'x-extension': {'app': 'contacts', 'returns': '', 'description': 'Literal data'},
        'parameters': {'type': 'object', 'required': ['name'], 'additionalProperties': False,
            'x-one-of': [['phone'], ['email']], 'properties': {
                'name': {'type': 'string'},
                'phone': {'type': 'string', 'description': 'Giữ số 0 đầu; không thêm mã vùng.', 'x-match': 'fuzzy'},
                'email': {'type': 'string'},
                'group_id': {'type': 'string', 'x-ref': 'contacts.groups', 'description': 'ID từ contacts.list_groups'},
                'priority': {'type': 'integer', 'minimum': 0, 'maximum': 3,
                    'description': 'Số nguyên, không dấu chấm hay phẩy, ví dụ 2'},
                'time': {'type': 'string', 'format': 'date-time', 'description': 'ISO-8601 đầy đủ giây và offset múi giờ'},
                'mode': {'type': 'string', 'enum': ['app', 'description', 'returns']},
                'description': {'type': 'string', 'default': 'Số nguyên, không dấu chấm hay phẩy, ví dụ 2'},
                'children': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0,
                    'description': 'Số nguyên, không dấu chấm hay phẩy, ví dụ 2'}}}}}


def test_compaction_removes_only_redundancy_and_is_immutable_idempotent():
    raw = tool()
    before = deepcopy(raw)
    compact = compact_tool_schema(raw)
    assert raw == before
    expected = deepcopy(raw)
    del expected['app'], expected['returns']
    del expected['parameters']['properties']['priority']['description']
    del expected['parameters']['properties']['children']['items']['description']
    assert compact == expected
    assert compact_tool_schema(compact) == compact
    compact['parameters']['required'].clear()
    assert compact_tool_schema(raw) == expected  # Cached views are not shared mutable dicts.
    assert len(json.dumps(expected, ensure_ascii=False)) < len(json.dumps(raw, ensure_ascii=False))


@pytest.mark.parametrize('arguments,status', [
    ({'name': 'Hồng', 'phone': '0901234567'}, 'valid'),
    ({'name': 'Hồng', 'email': 'a@example.com', 'priority': 2}, 'valid'),
    ({'name': 'Hồng'}, 'invalid'),
    ({'phone': '0901234567'}, 'invalid'),
    ({'name': 'Hồng', 'phone': 901234567}, 'invalid'),
    ({'name': 'Hồng', 'phone': '0901234567', 'priority': True}, 'invalid'),
    ({'name': 'Hồng', 'phone': '0901234567', 'priority': 4}, 'invalid'),
    ({'name': 'Hồng', 'phone': '0901234567', 'mode': 'unknown'}, 'invalid'),
    ({'name': 'Hồng', 'phone': '0901234567', 'unknown': 1}, 'invalid'),
    ({'name': 'Hồng', 'phone': '0901234567', 'children': [-1]}, 'invalid'),
])
def test_valid_and_invalid_argument_outcomes_are_identical(arguments, status):
    raw = tool()
    for schema in (raw, compact_tool_schema(raw)):
        assert inspect_call(raw['name'], arguments, {raw['name']: schema})['status'] == status


def test_nonempty_returns_mismatched_metadata_business_descriptions_and_literals_survive():
    raw = tool()
    raw['returns'] = {'type': 'object', 'properties': {'app': {'type': 'string'}}}
    raw['app'] = 'custom-metadata'
    raw['parameters']['properties']['priority']['description'] += '; đơn vị: phút'
    compact = compact_tool_schema(raw)
    assert compact['returns'] == raw['returns'] and compact['app'] == raw['app']
    assert compact['parameters']['properties']['priority'] == raw['parameters']['properties']['priority']
    assert compact['x-extension'] == raw['x-extension']


def test_cpu_compilation_hits_across_samples_and_invalidates_on_catalog_change():
    _compiled.cache_clear()
    catalog = {'contacts.create': tool()}
    first = prompt_catalog(catalog)
    second = prompt_catalog(catalog)
    assert first == second and _compiled.cache_info().hits == 1
    catalog['contacts.create']['parameters']['properties']['priority']['maximum'] = 8
    assert prompt_catalog(catalog) != first
    assert _compiled.cache_info().misses == 2


def test_sft_prompt_is_compact_but_env_receives_original_gold_and_catalog():
    raw = tool()
    catalog = {raw['name']: raw, 'sys.ask_missing_info': {'name': 'sys.ask_missing_info', 'parameters': {}}}
    original = deepcopy(catalog)
    gold = {'name': raw['name'], 'arguments': {'name': 'Hồng', 'phone': '0901234567'}}
    calls = []
    class Env:
        def validate(self, name, args):
            assert catalog == original
            assert catalog[name]['app'] == 'contacts' and 'returns' in catalog[name]
            return {'ok': True}
        def call(self, name, args):
            calls.append({'name': name, 'arguments': args})
            return {'ok': True}
    record = {'id': 'train_compact', 'audio_text': 'Tạo liên hệ Hồng.', 'history': [],
              'context': {'installed_apps': {'contacts': {}}}, 'label': {'tool_calls': [gold]}}
    rows = replay_sample(record, catalog, Env(), Path('/unused'))
    for row in rows:
        schemas = json.loads(row['messages'][1]['content'])['available_tools']['tool_schemas']
        assert next(s for s in schemas if s['name'] == gold['name']) == compact_tool_schema(raw)
    assert calls == [gold] and catalog == original


def test_runtime_policy_and_legacy_adapter_compatibility(temporal_artifact):
    from lifemate.runtime import load_runtime_manifest, RUNTIME_MANIFEST
    base, adapter, tok, manifest = temporal_artifact
    assert manifest['schema_prompt_policy'] == SCHEMA_PROMPT_POLICY
    manifest.pop('schema_prompt_policy')
    (adapter/RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    assert load_runtime_manifest(base, tok, adapter) == manifest
    manifest['schema_prompt_policy'] = 'unknown-compaction'
    (adapter/RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Schema prompt policy'):
        load_runtime_manifest(base, tok, adapter)
