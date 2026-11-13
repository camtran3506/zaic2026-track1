"""Canonical replay, alias consistency, train-only noise and strict unseen apps."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lifemate.data_generalization import (AppAugmenter, POLICY, assert_no_heldout, exposed_apps,
    load_groups, split_generalization, validate_split_manifest)
from lifemate.agent import build_step_prompt
from lifemate.catalog import load_catalog
from lifemate.sft import replay_sample
from test_baseline import CATALOG, FakeEnv
from test_sft import labeled
from test_prompt_session import bpe_tokenizer


def options(**kwargs):
    values = dict(rename_variants=2, distractor_probability=.5, heldout_apps='none', heldout_groups='',
                  heldout_fraction=.1, validation_fraction=.1, seed=42)
    return SimpleNamespace(**{**values, **kwargs})


def extra_catalog():
    catalog = deepcopy(CATALOG)
    for app in ('weather', 'banking', 'flights', 'library'):
        name = app + '.read'
        catalog[name] = {'name': name, 'kind': 'read', 'description': 'Read '+app,
                         'parameters': {'type': 'object', 'properties': {}, 'required': []}}
    for name, spec in catalog.items():
        spec['app'] = name.split('.')[0]
    return catalog


def records():
    return [{'id': f'{app}_{i}', 'audio': f'{app}_{i}.wav', 'audio_text': 'Yêu cầu đọc dữ liệu',
        'context': {'installed_apps': {app: {'tools': [tool]}}}, 'history': [],
        'label': {'tool_calls': [{'name': tool, 'arguments': {}}]}}
        for app, tool in [('smart_home', 'smart_home.get_status'), ('weather', 'weather.read'),
                          ('banking', 'banking.read'), ('flights', 'flights.read'), ('library', 'library.read')]
        for i in range(4)]


def test_app_disjoint_split_includes_installed_unused_history_and_schema_dependencies():
    catalog, rows = extra_catalog(), records()
    # Weather is unused in this target, but its schema was shown in the prompt.
    rows[0]['context']['installed_apps']['library'] = {'tools': ['library.read']}
    rows[1]['history'] = [{'role': 'assistant', 'text': '', 'tool_calls': [
        {'name': 'library.read', 'arguments': {}}], 'tool_results': [{'ok': True}]}]
    catalog['banking.read']['parameters']['properties']['account_id'] = {'type': 'string', 'x-ref': 'library.accounts'}
    args = options(heldout_apps='library')
    partitions, info = split_generalization(rows, catalog, args, load_groups(None, catalog))
    expected = {r['id'] for r in rows if 'library' in exposed_apps(r, catalog)}
    assert {r['id'] for r in partitions['app_heldout']} == expected
    assert rows[0]['id'] in expected and rows[1]['id'] in expected
    assert all(not exposed_apps(r, catalog) & {'library'} for p in ('train', 'validation') for r in partitions[p])
    again, repeated = split_generalization(rows[::-1], catalog, args, load_groups(None, catalog))
    assert partitions == again and info == repeated


def test_auto_holdout_and_tiny_dataset_unavailable_are_explicit():
    catalog, rows = extra_catalog(), records()
    partitions, info = split_generalization(rows, catalog, options(heldout_apps='auto'), load_groups(None, catalog))
    assert info['status'] == 'enabled' and len(info['apps']) == 1 and partitions['app_heldout']
    tiny = [dict(labeled(), id=str(i)) for i in range(4)]
    partitions, info = split_generalization(tiny, CATALOG, options(heldout_apps='auto'), load_groups(None, CATALOG))
    assert info['status'] == 'unavailable' and info['apps'] == [] and not partitions['app_heldout']


def test_group_holdout_and_infeasible_or_unknown_selection(tmp_path):
    catalog, rows = extra_catalog(), records()
    path = tmp_path / 'groups.json'
    path.write_text(json.dumps({'chosen': ['library', 'flights']}))
    groups = load_groups(path, catalog)
    partitions, info = split_generalization(rows, catalog, options(heldout_groups='chosen'), groups)
    assert info['apps'] == ['flights', 'library'] and len(partitions['app_heldout']) == 8
    with pytest.raises(ValueError, match='Unknown'):
        split_generalization(rows, catalog, options(heldout_groups='typo'), groups)
    with pytest.raises(ValueError):
        split_generalization(rows, catalog, options(heldout_apps='unknown'), groups)
    with pytest.raises(ValueError, match='no apps'):
        split_generalization(rows, catalog, options(heldout_groups='communication'), groups)
    with pytest.raises(ValueError, match='insufficient'):
        split_generalization(rows, catalog, options(heldout_apps='smart_home,weather,banking,flights,library'), groups)


def test_app_names_inside_argument_data_do_not_count_as_tool_targets():
    from lifemate.data_generalization import target_apps
    entry = {'label': {'tool_calls': [{'name': 'sms.send', 'arguments': {'content': 'Gửi literal library.read'}}]}}
    assert target_apps(entry, {'sms', 'library'}) == {'sms'}
    entry['label']['tool_calls'] = [{'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'library.read'}}]
    assert target_apps(entry, {'sms', 'library'}) == {'library'}


def test_two_alias_trajectories_preserve_arguments_history_guards_and_sys_target(bpe_tokenizer):
    entry = labeled()
    entry['context']['collections'] = {'smart_home.devices': [{'id': 'light_living', 'name': 'Đèn'}]}
    entry['history'] = [{'role': 'assistant', 'text': 'Trạng thái trước.',
        'tool_calls': [{'name': 'smart_home.get_status', 'arguments': {}}],
        'tool_results': [{'ok': True, 'devices': [{'id': 'light_living', 'name': 'Đèn'}]}]}]
    examples = replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))
    augmentation = AppAugmenter(CATALOG, [entry], [], load_groups(None, CATALOG), tokenizer=bpe_tokenizer,
                                distractor_probability=0)
    rows = augmentation.augment(entry, examples)
    base = [e for e in examples if json.loads(e['messages'][-1]['content']).get('name')
            not in {'local_entity.resolve', 'local_context.lookup'}]
    assert len(rows) == len(base)*2
    assert json.loads(base[0]['messages'][-1]['content'])['name'] == 'smart_home.get_status'
    assert rows[0]['augmentation']['tool_aliases'] != rows[len(base)]['augmentation']['tool_aliases']
    for row in rows:
        meta = row['augmentation']; tools = meta['tool_aliases']; apps = meta['app_aliases']
        payload = json.loads(row['messages'][1]['content']); target = json.loads(row['messages'][-1]['content'])
        assert row['sample_id'] == entry['id']
        assert payload['request_transcript'] == json.loads(examples[0]['messages'][1]['content'])['request_transcript']
        assert apps['smart_home']+'.devices' in payload['user_context']['collections']
        assert payload['conversation_history'][0]['tool_calls'][0]['name'] == tools['smart_home.get_status']
        assert payload['conversation_history'][0]['tool_results'] == entry['history'][0]['tool_results']
        if target['action'] == 'call':
            original = next(name for name, alias in tools.items() if alias == target['name'])
            assert target['arguments'] == next(json.loads(e['messages'][-1]['content'])['arguments'] for e in examples
                if json.loads(e['messages'][-1]['content']).get('name') == original)
        for observation in payload['previous_actions_and_results']:
            if 'automatic_execution_feedback' in observation:
                assert not any(name in json.dumps(observation['automatic_execution_feedback']) for name in tools)
    missing = deepcopy(entry)
    # Keep the sys target aliasing case genuinely unresolved under the shared
    # missing-info guard; context/history preservation is checked above.
    missing['context']['collections'] = {}
    missing['history'] = []
    missing['label']['tool_calls'] = [{'name': 'sys.ask_missing_info', 'arguments': {
        'tool_name': 'smart_home.light_set', 'params': ['device_id']}}]
    examples = replay_sample(missing, CATALOG, FakeEnv(), Path('/unused'))
    row = augmentation.augment(missing, examples)[0]
    target = json.loads(row['messages'][-1]['content'])
    assert target['name'] == 'sys.ask_missing_info'
    assert target['arguments']['tool_name'] == row['augmentation']['tool_aliases']['smart_home.light_set']
    assert target['arguments']['params'] == ['device_id']


def test_aliases_omit_only_generated_name_bound_helper_demos(bpe_tokenizer):
    entry = labeled()
    examples = replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))
    demo = deepcopy(examples[0])
    demo['messages'][-1]['content'] = json.dumps({'action': 'call', 'name': 'local_entity.resolve', 'arguments': {
        'target_tool': 'smart_home.light_set', 'argument': 'device_id', 'collection': 'smart_home.devices', 'mention': 'Đèn'}})
    helper_obs = {'name': 'local_entity.resolve', 'arguments': {}, 'result': '{"status":"matched"}', 'scope': 'local'}
    for row in examples:
        payload = json.loads(row['messages'][1]['content'])
        payload['previous_actions_and_results'].insert(0, helper_obs)
        row['messages'][1]['content'] = json.dumps(payload)
    generator = AppAugmenter(CATALOG, [entry], [], load_groups(None, CATALOG), tokenizer=bpe_tokenizer, distractor_probability=0)
    rows = generator.augment(entry, [demo, *examples])
    assert len(rows) == 2*len(examples)
    assert generator.stats['generated_knowledge_demos_removed'] == 2
    assert all('local_entity.resolve' != obs.get('name') for row in rows
               for obs in json.loads(row['messages'][1]['content'])['previous_actions_and_results'])
    assert any('local_entity.resolve' == obs.get('name') for obs in json.loads(examples[0]['messages'][1]['content'])['previous_actions_and_results'])


def test_noise_is_one_to_three_unrelated_seen_apps_and_never_heldout(bpe_tokenizer):
    catalog, source = extra_catalog(), records()
    train = [r for r in source if not r['id'].startswith('library_')]
    generator = AppAugmenter(catalog, train, ['library'], load_groups(None, catalog),
                             tokenizer=bpe_tokenizer, distractor_probability=1)
    entry = source[0]
    _, prompt = build_step_prompt(entry['audio_text'], entry['context'], [], {'smart_home.get_status': catalog['smart_home.get_status']}, [])
    example = {'sample_id': entry['id'], 'step': 0, 'messages': [
        {'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': '{"action":"call","name":"smart_home.get_status","arguments":{}}'}]}
    rows = generator.augment(entry, [example])
    assert len(rows) == 2
    for row in rows:
        meta = row['augmentation']
        assert 1 <= len(meta['distractor_apps']) <= 3
        assert set(meta['distractor_apps']) <= {'weather', 'banking', 'flights'}
        for spec in json.loads(row['messages'][1]['content'])['available_tools']['tool_schemas']:
            if 'app' in spec:
                assert spec['app'] == spec['name'].split('.')[0]
        assert_no_heldout(row, ['library'])
        assert json.loads(row['messages'][-1]['content'])['arguments'] == {}


def test_noise_only_keeps_canonical_calls_and_empty_app_requests_are_not_duplicated(bpe_tokenizer):
    catalog, source = extra_catalog(), records()
    entry = source[0]
    _, prompt = build_step_prompt(entry['audio_text'], entry['context'], [], {'smart_home.get_status': catalog['smart_home.get_status']}, [])
    example = {'sample_id': entry['id'], 'step': 0, 'messages': [
        {'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': '{"action":"call","name":"smart_home.get_status","arguments":{}}'}]}
    generator = AppAugmenter(catalog, source, [], load_groups(None, catalog), rename_variants=0,
                             distractor_probability=1, tokenizer=bpe_tokenizer)
    rows = generator.augment(entry, [example])
    assert len(rows) == 1 and not rows[0]['augmentation']['tool_aliases']
    assert rows[0]['messages'][-1] == example['messages'][-1]
    assert rows[0]['augmentation']['distractor_apps']
    empty = deepcopy(entry); empty['context']['installed_apps'] = {}; empty['label']['tool_calls'] = []
    generator = AppAugmenter(catalog, [empty], [], load_groups(None, catalog), tokenizer=bpe_tokenizer)
    assert generator.augment(empty, [example]) == []
    assert generator.stats['variants_skipped_no_apps'] == 2


def test_exact_token_budget_drops_whole_variant_and_does_not_clip_original(bpe_tokenizer):
    entry = labeled(); examples = replay_sample(entry, CATALOG, FakeEnv(), Path('/unused')); saved = deepcopy(examples)
    generator = AppAugmenter(CATALOG, [entry], [], load_groups(None, CATALOG), tokenizer=bpe_tokenizer, max_length=10)
    assert generator.augment(entry, examples) == []
    assert generator.stats['variants_dropped_token_budget'] == 2 and examples == saved
    entry['audio_text'] = 'Giữ nguyên literal smart_home.light_set'
    examples = replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))
    assert generator.augment(entry, examples) == []
    assert generator.stats['variants_skipped_literal_tool_reference'] == 2


@pytest.fixture
def prepared_generalization(tmp_path, monkeypatch, bpe_tokenizer):
    import prepare_sft
    import lifemate.models
    import lifemate.transcripts
    source = tmp_path/'train'; source.mkdir()
    catalog, source_rows = extra_catalog(), records()
    (source/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in source_rows))
    (source/'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(catalog.values())}]}))
    instances = []
    class Env:
        def __init__(self, split, id, root): self.id=id; self.calls=[]; instances.append(self)
        def validate(self, name, args):
            assert name in catalog and not name.startswith('app_')
            return {'ok': True}
        def call(self, name, args):
            self.calls.append({'name': name, 'arguments': args}); return {'ok': True, 'rows': []}
        def get_tool_calls(self): return deepcopy(self.calls)
        def get_state_hash(self): return 'calls:'+str(len(self.calls))
    monkeypatch.setitem(sys.modules, 'zaic_tools', SimpleNamespace(ToolEnv=Env))
    class ASR:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, path): return 'ASR yêu cầu đọc dữ liệu'
    monkeypatch.setattr(lifemate.models, 'LocalASR', ASR)
    monkeypatch.setattr(lifemate.transcripts, 'asr_identity', lambda *args: {'model': 'stub'})
    tokenizer_path = tmp_path/'tokenizer'; bpe_tokenizer.save_pretrained(tokenizer_path)
    (tokenizer_path/'config.json').write_text('{"model_type":"qwen3_5"}')
    output = tmp_path/'sft'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source), '--output-dir', str(output),
        '--asr-model', 'stub', '--transcript-mode', 'mix', '--validation-fraction', '0.1',
        '--heldout-apps', 'library', '--rename-variants', '2', '--distractor-probability', '1',
        '--tokenizer', str(tokenizer_path)])
    prepare_sft.main()
    return output, catalog, source_rows, instances


def test_full_preparation_replays_once_and_augments_only_train(prepared_generalization):
    output, catalog, source_rows, instances = prepared_generalization
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['app_holdout']['status'] == 'enabled'
    assert len(instances) == len(source_rows) == len({e.id for e in instances})
    assert all(len(e.calls) == 1 for e in instances)
    validate_split_manifest(manifest, catalog)
    train_rows = [json.loads(line) for line in (output/'train.jsonl').read_text().splitlines()]
    assert any(row.get('augmentation') for row in train_rows)
    ids = {row['sample_id'] for row in train_rows}
    assert ids == set(manifest['partitions']['train']['sample_ids'])
    for row in train_rows: assert_no_heldout(row, ['library'])
    for partition in ('validation', 'app_heldout'):
        rows = [json.loads(line) for line in (output/f'{partition}.jsonl').read_text().splitlines()]
        assert not any(row.get('augmentation') for row in rows)
        assert not ids & {row['sample_id'] for row in rows}
        rollouts = [json.loads(line) for line in (output/f'{partition}_rollouts.jsonl').read_text().splitlines()]
        assert all(row['prepared_transcript']['transcript_source'] == 'asr' for row in rollouts)
    fake = deepcopy(train_rows[-1]); fake.setdefault('augmentation', {})['distractor_apps'] = ['library']
    with pytest.raises(ValueError, match='held-out app'):
        assert_no_heldout(fake, ['library'])


def test_heldout_metrics_are_separate_and_use_canonical_toolenv(prepared_generalization):
    from lifemate.evaluation import StateToolEvaluator
    output, _, _, instances = prepared_generalization
    manifest = json.loads((output/'manifest.json').read_text())
    class LLM:
        def generate(self, system, user):
            payload = json.loads(user)
            assert 'gold_tool_calls' not in payload
            if payload['previous_actions_and_results']: return '{"action":"done"}'
            tool = next(s['name'] for s in payload['available_tools']['tool_schemas']
                        if not s['name'].startswith(('local_', 'sys.')))
            return json.dumps({'action': 'call', 'name': tool, 'arguments': {}})
    normal = StateToolEvaluator(output, manifest)
    heldout = StateToolEvaluator(output, manifest, partition='app_heldout', require_clean_transcripts=True)
    try:
        one = normal.evaluate(LLM(), output/'normal.jsonl')
        two = heldout.evaluate(LLM(), output/'heldout.jsonl')
        clean = heldout.evaluate(LLM(), output/'heldout-clean.jsonl', transcript_source='audio_text')
        assert one['eval_state_tool_score'] == two['eval_app_heldout_state_tool_score'] == 100
        assert clean['eval_app_heldout_clean_state_tool_score'] == 100
        assert not one.keys() & two.keys() and not one.keys() & clean.keys()
    finally:
        normal.close(); heldout.close()
    assert all(not call['name'].startswith('app_') for env in instances for call in env.calls)
    bad = deepcopy(manifest)
    bad['partitions']['train']['sample_ids'].append(bad['partitions']['app_heldout']['sample_ids'][0])
    with pytest.raises(ValueError, match='leaked'):
        validate_split_manifest(bad, load_catalog(Path(bad['source']).parent/'train_tools_catalog.json'))
