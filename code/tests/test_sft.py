import json
import sys
import types
from pathlib import Path

import pytest

from lifemate.sft import UnexposedToolError, encode_example, replay_sample, split_records, AssistantCollator
from test_baseline import CATALOG, FakeEnv, item


def labeled():
    entry = item()
    entry['audio_text'] = 'Bật đèn phòng khách'
    entry['label'] = {'tool_calls': [
        {'name': 'smart_home.get_status', 'arguments': {}},
        {'name': 'smart_home.light_set', 'arguments': {'device_id': 'light_living', 'power': 'on'}},
    ]}
    return entry


def test_gold_replay_observations_do_not_leak_future_targets():
    env = FakeEnv()
    steps = replay_sample(labeled(), CATALOG, env, Path('/unused'))
    assert len(steps) == 3
    payloads = [json.loads(s['messages'][1]['content']) for s in steps]
    assert payloads[0]['previous_actions_and_results'] == []
    assert 'light_living' not in steps[0]['messages'][1]['content']
    assert 'light_living' in payloads[1]['previous_actions_and_results'][0]['result']
    assert json.loads(steps[-1]['messages'][-1]['content']) == {'action': 'done'}
    assert len(env.calls) == 2


def test_schema_and_missing_labels_fail_instead_of_becoming_done():
    entry = labeled()
    entry.pop('label')
    with pytest.raises(ValueError, match='missing label'):
        replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))
    entry = labeled()
    entry['label']['tool_calls'][1]['arguments'] = {}
    with pytest.raises(ValueError, match='gold schema invalid'):
        replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))


def test_terminal_missing_info_does_not_add_done():
    entry = labeled()
    entry['label']['tool_calls'] = [{'name': 'sys.ask_missing_info',
        'arguments': {'tool_name': 'smart_home.light_set', 'params': ['device_id']}}]
    assert len(replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))) == 1


def test_empty_calls_with_temporal_words_demonstrates_helpers_then_done():
    entry = item()
    entry['context']['current_time'] = '2026-10-14T08:00:00+07:00'
    entry['context']['installed_apps']['reminders'] = {'tools': ['reminders.create']}
    entry['audio_text'] = 'Nhắc tôi ngày mai có việc gì không?'
    entry['label'] = {'tool_calls': []}
    catalog = {
        **CATALOG,
        'reminders.create': {
            'name': 'reminders.create', 'kind': 'write', 'description': 'Tạo nhắc nhở',
            'parameters': {'type': 'object', 'properties': {'content': {'type': 'string'}}, 'required': ['content']},
        }
    }
    steps = replay_sample(entry, catalog, FakeEnv(), Path('/unused'))
    assert len(steps) >= 2
    step0 = json.loads(steps[0]['messages'][-1]['content'])
    assert step0['action'] == 'call'
    assert step0['name'] in ('local_time.calendar_reference', 'local_assist.solve')
    step_last = json.loads(steps[-1]['messages'][-1]['content'])
    assert step_last == {'action': 'done'}
    last_prompt = json.loads(steps[-1]['messages'][1]['content'])
    obs = last_prompt['previous_actions_and_results']
    assert len(obs) == 1
    assert obs[0]['name'] in ('local_time.calendar_reference', 'local_assist.solve')


def test_gold_tool_must_exist_in_complete_prompt(monkeypatch):
    import lifemate.sft as sft
    monkeypatch.setattr(sft, 'build_step_prompt', lambda *args, **kwargs: (
        {'tool_schemas': [{'name': 'smart_home.get_status'}]},
        json.dumps({'available_tools': {'tool_schemas': []}})))
    with pytest.raises(UnexposedToolError, match='smart_home.get_status not exposed'):
        replay_sample(labeled(), CATALOG, FakeEnv(), Path('/unused'))


def test_prepare_skip_reports_whole_samples_and_retains_partition_ids(tmp_path, monkeypatch):
    import prepare_sft
    import lifemate.models
    import lifemate.transcripts
    source = tmp_path / 'data' / 'train'
    source.mkdir(parents=True)
    records = [dict(labeled(), id=f'train_{i}') for i in range(8)]
    train, valid = split_records(records, 0.3, 42)
    rejected = {train[0]['id']}
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(CATALOG.values())}]}))
    class HashEnv(FakeEnv):
        def get_state_hash(self): return 'fixture-gold-hash'
    monkeypatch.setitem(sys.modules, 'zaic_tools', types.SimpleNamespace(ToolEnv=lambda *a, **k: HashEnv()))
    class ASR:
        def __init__(self, *a, **kw): pass
        def transcribe(self, path): return 'Bật đèn phòng khách'
    monkeypatch.setattr(lifemate.models, 'LocalASR', ASR)
    monkeypatch.setattr(lifemate.transcripts, 'asr_identity', lambda *a: {'model': 'stub-asr'})
    def replay(record, *args, **kwargs):
        if record['id'] in rejected:
            raise UnexposedToolError(f"{record['id']}: target not exposed")
        return replay_sample(record, *args, **kwargs)
    monkeypatch.setattr(prepare_sft, 'replay_sample', replay)
    out = tmp_path / 'sft'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--asr-model', 'stub-asr', '--asr-transcript-ratio', '0', '--data-root', str(source),
        '--output-dir', str(out), '--validation-fraction', '0.3', '--missing-tool-policy', 'skip'])
    prepare_sft.main()
    manifest = json.loads((out / 'manifest.json').read_text())
    retained = set()
    for part in ('train', 'validation'):
        metadata = manifest['partitions'][part]
        rows = [json.loads(line) for line in (out / f'{part}.jsonl').read_text().splitlines()]
        ids = {r['sample_id'] for r in rows}
        assert ids == set(metadata['sample_ids'])
        assert not ids & rejected
        assert not retained & ids
        retained |= ids
        assert metadata['source_samples'] == metadata['samples'] + len(metadata['skipped_samples'])
        if part == 'train':
            assert len((out / f'{part}_skipped.jsonl').read_text().splitlines()) == 1
        else:
            assert not metadata['skipped_samples']
    assert len(retained) == 7


def test_split_is_grouped_deterministic_and_order_independent():
    records = [{'id': str(i)} for i in range(20)]
    train, valid = split_records(records, 0.1)
    assert not {r['id'] for r in train} & {r['id'] for r in valid}
    assert (train, valid) == split_records(records[::-1], 0.1)
    with pytest.raises(ValueError, match='Duplicate'):
        split_records([{'id': 'x'}, {'id': 'x'}])


@pytest.mark.parametrize('count', [1, 20])
def test_default_split_keeps_all_samples_in_train(count):
    records = [{'id': str(i)} for i in range(count)]
    train, valid = split_records(records)
    assert valid == []
    assert {row['id'] for row in train} == {row['id'] for row in records}
    assert (train, valid) == split_records(records[::-1])


class TemplateTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 7, 8, 9]


def test_loss_masks_prompt_and_padding_and_rejects_truncated_json():
    example = {'sample_id': 'train_1', 'step': 0, 'messages': [
        {'role': role, 'content': ''} for role in ('system', 'user', 'assistant')]}
    row = encode_example(example, TemplateTokenizer(), 6)
    assert row['labels'] == [-100, -100, -100, 7, 8, 9]
    with pytest.raises(ValueError, match='never truncate'):
        encode_example(example, TemplateTokenizer(), 5)
    short = {k: v[:-1] for k, v in row.items()}
    batch = AssistantCollator(0)([row, short])
    assert batch['labels'][1, -1].item() == -100
    assert batch['attention_mask'][1, -1].item() == 0
