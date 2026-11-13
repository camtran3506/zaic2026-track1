"""Recorded synthetic trajectories never depend on ASR, ToolEnv or state hashes."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lifemate.function_calling import native_chat
from lifemate.reasoning import attach_reasoning_to_trajectory
from lifemate.sft import (collapse_trajectory, encode_example, is_synthetic_sample,
                          prepare_train_transcript, pretranscribe_sft_partition,
                          replay_sample, replay_synthetic_sample)
from test_baseline import CATALOG, FakeEnv
from test_prompt_session import bpe_tokenizer
from test_sft import labeled


TOOLS = {
    **CATALOG,
    'new_app.search': {
        'name': 'new_app.search', 'kind': 'read', 'description': 'Tìm sản phẩm',
        'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}},
                       'required': ['query']},
    },
    'new_app.order': {
        'name': 'new_app.order', 'kind': 'write', 'description': 'Đặt sản phẩm',
        'parameters': {'type': 'object', 'properties': {
            'product_id': {'type': 'string', 'x-ref': 'new_app.products'},
            'quantity': {'type': 'integer', 'minimum': 1}}, 'required': ['product_id', 'quantity']},
    },
}


def synthetic_record():
    return {
        'id': 'synthetic_0001', 'audio': 'audio/placeholder_does_not_exist.wav',
        'audio_text': 'Đặt một sản phẩm sữa', 'history': [],
        'context': {'installed_apps': {'new_app': {'tools': ['new_app.search', 'new_app.order']}}},
        'label': {
            'tool_calls': [
                {'name': 'new_app.search', 'arguments': {'query': 'sữa'}},
                {'name': 'new_app.order', 'arguments': {'product_id': 'product_123', 'quantity': 1}},
            ],
            'tool_results': [
                {'ok': True, 'products': [{'id': 'product_123', 'name': 'sữa'}]},
                {'ok': True, 'order': {'id': 'order_secret_future'}},
            ],
            'final_state': {'orders': ['final_state_must_not_leak']},
        },
    }


def fail_if_called(*args, **kwargs):
    raise AssertionError('Synthetic samples must not execute ASR/ToolEnv/recovery')


def test_recorded_trajectory_is_causal_and_encodes_without_execution(monkeypatch, bpe_tokenizer):
    import lifemate.sft
    monkeypatch.setattr(lifemate.sft, 'attempt_env_call', fail_if_called)
    record = synthetic_record()
    before = deepcopy(record)
    rows = replay_synthetic_sample(record, TOOLS, Path('/unused'), include_date_helpers=False)
    assert record == before
    assert [json.loads(r['messages'][-1]['content']).get('name') for r in rows] == [
        'new_app.search', 'new_app.order', None]
    prompts = [json.loads(r['messages'][1]['content']) for r in rows]
    assert prompts[0]['previous_actions_and_results'] == []
    assert 'product_123' not in rows[0]['messages'][1]['content']
    assert 'product_123' in rows[1]['messages'][1]['content']
    assert 'order_secret_future' not in rows[1]['messages'][1]['content']
    assert 'order_secret_future' in rows[2]['messages'][1]['content']
    assert all('final_state_must_not_leak' not in r['messages'][1]['content'] for r in rows)
    assert all(r['trajectory_source'] == 'synthetic_recorded' for r in rows)
    result = prompts[1]['previous_actions_and_results'][0]
    assert json.loads(result['result']) == record['label']['tool_results'][0]
    assert result['automatic_execution_feedback']['origin'] == 'synthetic_recorded'
    for steps in (rows, attach_reasoning_to_trajectory(rows)):
        collapsed = collapse_trajectory(steps)
        assert collapsed['trajectory_source'] == 'synthetic_recorded'
        chat, _ = native_chat(collapsed['messages'])
        assert any(m['role'] == 'tool' and 'product_123' in m['content'] for m in chat)
        encoded = encode_example(collapsed, bpe_tokenizer, 16384, fuzzy_weight=0.3, reasoning_weight=0.25)
        assert any(label != -100 for label in encoded['labels'])


@pytest.mark.parametrize('failure', ['missing_results', 'misaligned', 'not_object', 'nan',
                                     'missing_required', 'wrong_type'])
def test_invalid_recorded_trajectories_fail(failure):
    record = synthetic_record()
    label = record['label']
    if failure == 'missing_results':
        label.pop('tool_results')
    elif failure == 'misaligned':
        label['tool_results'].pop()
    elif failure == 'not_object':
        label['tool_results'][0] = 'not a result object'
    elif failure == 'nan':
        label['tool_results'][0]['value'] = float('nan')
    elif failure == 'missing_required':
        label['tool_calls'][1]['arguments'].pop('product_id')
    else:
        label['tool_calls'][1]['arguments']['quantity'] = 'one'
    with pytest.raises(ValueError, match='synthetic'):
        replay_synthetic_sample(record, TOOLS, Path('/unused'))


def test_synthetic_empty_calls_and_terminal_missing_info():
    record = synthetic_record()
    record['label'] = {'tool_calls': [], 'tool_results': []}
    rows = replay_synthetic_sample(record, TOOLS, Path('/unused'), include_date_helpers=False)
    assert len(rows) == 1 and json.loads(rows[0]['messages'][-1]['content']) == {'action': 'done'}
    record['label'] = {
        'tool_calls': [{'name': 'sys.ask_missing_info',
                        'arguments': {'tool_name': 'new_app.order', 'params': ['quantity']}}],
        'tool_results': [{'ok': True, 'asked': True}],
    }
    rows = replay_synthetic_sample(record, TOOLS, Path('/unused'), include_date_helpers=False)
    assert len(rows) == 1
    assert json.loads(rows[0]['messages'][-1]['content'])['name'] == 'sys.ask_missing_info'


def test_synthetic_text_source_and_history_never_transcribe():
    record = synthetic_record()
    record.update(id='explicit_source', synthetic=True)
    record['history'] = [{'role': 'user', 'audio': 'history_missing.wav', 'audio_text': 'Mua giúp tôi'}]
    asr = SimpleNamespace(transcribe=fail_if_called)
    prepared = prepare_train_transcript(record, Path('/unused'), asr, transcript_source='asr')
    assert is_synthetic_sample(record)
    assert prepared['transcript_source'] == 'audio_text'
    assert prepared['history'][0]['text'] == 'Mua giúp tôi.'
    record['history'][0].pop('audio_text')
    with pytest.raises(ValueError, match='History audio'):
        prepare_train_transcript(record, Path('/unused'), asr)
    record['audio_text'] = ''
    with pytest.raises(ValueError, match='nonempty audio_text'):
        prepare_train_transcript(record, Path('/unused'), asr)


def test_batch_pretranscription_excludes_all_synthetic_audio():
    original = dict(labeled(), audio='real.wav')
    synthetic = synthetic_record()
    synthetic['history'] = [{'role': 'user', 'audio': 'synthetic_history.wav'}]
    batches = []
    asr = SimpleNamespace(transcribe_batch=lambda paths, **kw: batches.append(paths))
    pretranscribe_sft_partition([original, synthetic], Path('/unused'), asr, 'train',
                               {original['id']: 'asr', synthetic['id']: 'asr'})
    assert batches == [[Path('/unused/real.wav')]]


def test_original_replay_still_reads_env_instead_of_recorded_results():
    record = labeled()
    record['label']['tool_results'] = [{'fake': True}, {'fake': True}]
    env = FakeEnv()
    rows = replay_sample(record, CATALOG, env, Path('/unused'), include_date_helpers=False)
    assert len(env.calls) == 2
    assert 'light_living' in rows[1]['messages'][1]['content']
    assert 'fake' not in rows[1]['messages'][1]['content']


@pytest.mark.parametrize('mode,ratio', [('dual', 0.5), ('mix', 0.5), ('mix', 1.0),
                                      ('dual', 0.0), ('dual', 1.0)])
def test_mixed_preparation_keeps_original_asr_and_excludes_synthetic_env(tmp_path, monkeypatch, mode, ratio):
    import prepare_sft
    import lifemate.models
    import lifemate.transcripts
    actual_dual = mode == 'dual' and 0 < ratio < 1
    source = tmp_path / 'train'
    source.mkdir()
    originals = [dict(labeled(), id=f'train_{i}', audio=f'train_{i}.wav') for i in range(4)]
    synthetic = synthetic_record()
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [*originals, synthetic]))
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(TOOLS.values())}]}))
    decoded, predecoded, env_ids, recovery_ids, guard_ids = [], [], [], [], []

    class ASR:
        def __init__(self, *a, **kw): pass
        def transcribe(self, path):
            assert path.stem.startswith('train_')
            decoded.append(path.stem)
            return 'Bật đèn phòng khách'
        def transcribe_batch(self, paths, **kwargs):
            assert all(p.stem.startswith('train_') for p in paths)
            predecoded.extend(p.stem for p in paths)

    def env_factory(split, sample_id, **kwargs):
        assert sample_id.startswith('train_')
        env_ids.append(sample_id)
        return FakeEnv()  # No get_state_hash: training must never request one.

    def recovery(record, *a, **kw):
        assert not is_synthetic_sample(record)
        recovery_ids.append(record['id'])

    def guard(record, *a, **kw):
        assert not is_synthetic_sample(record)
        guard_ids.append(record['id'])
        return []

    monkeypatch.setattr(lifemate.models, 'LocalASR', ASR)
    monkeypatch.setattr(lifemate.transcripts, 'asr_identity', lambda *a, **kw: {'model': 'stub-asr'})
    monkeypatch.setitem(sys.modules, 'zaic_tools', SimpleNamespace(ToolEnv=env_factory))
    monkeypatch.setattr(prepare_sft, 'build_recovery_trajectory', recovery)
    monkeypatch.setattr(prepare_sft, 'build_guard_recovery_examples', guard)
    output = tmp_path / 'sft'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
        '--output-dir', str(output), '--asr-model', 'stub-asr', '--transcript-mode', mode,
        '--asr-transcript-ratio', str(ratio)])
    prepare_sft.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    info = manifest['partitions']['train']
    assert info['synthetic_samples'] == 1
    assert info['transcript_sources'][synthetic['id']] == 'audio_text'
    assert info['trajectory_sources'][synthetic['id']] == 'synthetic_recorded'
    asr_count = 4 if actual_dual else round(4 * ratio)
    assert info['transcript_source_counts'] == {'audio_text': 5 if actual_dual else 5-asr_count,
                                               'asr': asr_count}
    assert len(env_ids) == (8 if actual_dual else 4)
    assert len(decoded) == len(predecoded) == asr_count
    assert len(recovery_ids) == len(guard_ids) == len(env_ids)
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert manifest['sft_format']['dual_transcript_source_prompt'] is actual_dual
    for row in rows:
        payload = json.loads(row['messages'][1]['content'])
        if actual_dual and row['sample_id'] != synthetic['id']:
            assert payload['request_transcript_source'] == row['transcript_source']
            assert payload['request_transcript_note']
        else:
            assert 'request_transcript_source' not in payload
            assert 'request_transcript_note' not in payload
    syn_rows = [r for r in rows if r['sample_id'] == synthetic['id']]
    assert len(syn_rows) == 3
    assert all(r['transcript_source'] == 'audio_text' and r['trajectory_source'] == 'synthetic_recorded'
               for r in syn_rows)


def test_synthetic_only_preparation_never_imports_toolenv_or_loads_asr(tmp_path, monkeypatch):
    import prepare_sft
    import lifemate.models
    source = tmp_path / 'train'
    source.mkdir()
    record = synthetic_record()
    (source / 'train.jsonl').write_text(json.dumps(record) + '\n')
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(TOOLS.values())}]}))
    monkeypatch.setitem(sys.modules, 'zaic_tools', None)
    monkeypatch.setattr(lifemate.models, 'LocalASR', fail_if_called)
    monkeypatch.setattr(prepare_sft, 'build_recovery_trajectory', fail_if_called)
    monkeypatch.setattr(prepare_sft, 'build_guard_recovery_examples', fail_if_called)
    output = tmp_path / 'sft'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
        '--output-dir', str(output), '--asr-transcript-ratio', '0', '--asr-model', 'unused'])
    prepare_sft.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['asr'] is None
    assert manifest['partitions']['train']['synthetic_samples'] == 1
    assert manifest['partitions']['train']['recovery_trajectories'] == 0
    assert manifest['partitions']['train']['guard_recovery_examples'] == 0


def test_synthetic_stays_out_of_diagnostic_splits():
    from prepare_sft import split_sft_records
    from lifemate.data_generalization import load_groups
    args = SimpleNamespace(rename_variants=0, distractor_probability=0, heldout_fraction=.1,
                           heldout_apps='none', heldout_groups='', validation_fraction=.25, seed=42)
    records = [dict(labeled(), id=f'train_{i}') for i in range(8)]
    syn = synthetic_record()
    partitions, holdout = split_sft_records([*records, syn], TOOLS, args, load_groups(None, TOOLS))
    assert len(partitions['validation']) == 2
    assert syn in partitions['train'] and syn not in partitions['validation']
    assert syn['id'] in holdout['exposed_apps_by_id']
    with pytest.raises(ValueError, match='Duplicate'):
        split_sft_records([syn, syn], TOOLS, args, load_groups(None, TOOLS))
