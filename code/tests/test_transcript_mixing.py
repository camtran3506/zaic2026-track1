"""Seeded 50:50 sample mixing, real preparation/replay and source failures."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lifemate.evaluation import StateToolEvaluator
from lifemate.sft import assign_transcript_sources, prepare_train_transcript
from test_baseline import CATALOG, FakeEnv
from test_sft import labeled


def test_sources_are_balanced_seeded_and_independent_of_input_order_and_gold():
    records = [dict(labeled(), id=f'train_{i}') for i in range(20)]
    sources = assign_transcript_sources(records)
    assert Counter(sources.values()) == {'audio_text': 10, 'asr': 10}
    assert sources == assign_transcript_sources(records[::-1])
    assert sources != assign_transcript_sources(records, seed=123)
    mutated = deepcopy(records)
    for record in mutated:
        record['label']['tool_calls'] *= 5
        record['audio_text'] = 'Changed clean transcript'
    assert sources == assign_transcript_sources(mutated)
    assert Counter(assign_transcript_sources(records[:5]).values()) == {'asr': 3, 'audio_text': 2}


@pytest.mark.parametrize('ratio', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_ratios_fail(ratio):
    with pytest.raises(ValueError):
        assign_transcript_sources([{'id': 'train_1'}], ratio)


def test_explicit_sources_normalize_without_substituting_clean_text(tmp_path):
    entry = dict(labeled(), audio='request.wav')
    calls = []
    raw = '  ASR sai dấu, lúc 08:30!  '
    class ASR:
        def transcribe(self, path):
            calls.append(path)
            return raw
    human = prepare_train_transcript(entry, tmp_path, ASR(), transcript_source='audio_text')
    assert human['transcript'] == entry['audio_text'] + '.' and not calls
    noisy = prepare_train_transcript(entry, tmp_path, ASR(), transcript_source='asr')
    assert noisy['transcript'] == 'Asr sai dấu, lúc 08:30!' and noisy['transcript_source'] == 'asr'
    assert calls == [tmp_path / 'request.wav']
    entry.pop('audio_text')
    with pytest.raises(ValueError, match='selected audio_text source is missing'):
        prepare_train_transcript(entry, tmp_path, ASR(), transcript_source='audio_text')


def test_asr_error_is_not_replaced_by_existing_audio_text(tmp_path):
    entry = dict(labeled(), audio='request.wav')
    class ASR:
        def transcribe(self, path): raise RuntimeError('ASR failed')
    with pytest.raises(RuntimeError, match='ASR failed'):
        prepare_train_transcript(entry, tmp_path, ASR(), transcript_source='asr')
    with pytest.raises(ValueError, match='ASR transcript required'):
        prepare_train_transcript(entry, tmp_path, transcript_source='asr')


@pytest.mark.parametrize('script', ['finetune.py', 'prepare_sft.py'])
@pytest.mark.parametrize('ratio', [0.5, 1])
def test_asr_mix_requires_asr_before_creating_output(script, ratio, tmp_path):
    app = Path(__file__).resolve().parents[1]
    output = tmp_path / 'run'
    command = [sys.executable, app / script, '--data-root', 'unused', '--output-dir', output,
               '--asr-transcript-ratio', str(ratio)]
    if script == 'finetune.py':
        command += ['--base-model', 'unused']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and '--asr-model is required' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('samples,validation_fraction', [(20, 0), (1, 0), (20, 0.2), (10, 0.1)])
def test_preparation_keeps_sft_mix_and_uses_asr_for_every_validation_rollout(
        tmp_path, monkeypatch, samples, validation_fraction):
    import prepare_sft
    import lifemate.models
    import lifemate.transcripts
    source = tmp_path / 'train'
    source.mkdir()
    records = [dict(labeled(), id=f'train_{i}', audio=f'train_{i}.wav',
                    audio_text=f'  HUMAN {i}!  ', history=[{'role': 'user', 'text': ' Full history! '}])
               for i in range(samples)]
    # Different trajectory lengths must not affect transcript assignments.
    for i, record in enumerate(records):
        record['label']['tool_calls'] = record['label']['tool_calls'][:1] + [
            {'name': 'smart_home.light_set', 'arguments': {'device_id': 'light_living', 'power': 'on'}}
            for _ in range(i % 3)]
        (source / record['audio']).write_bytes(b'fixture: recognizer is stubbed')
    manifest_path = source / 'train.jsonl'
    manifest_path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(CATALOG.values())}]}))
    transcribed = []
    class ASR:
        def __init__(self, model, **kwargs): pass
        def transcribe(self, path):
            transcribed.append(path.stem)
            return f'  ASR {path.stem}: lỗi nhận dạng!  '
    class Env(FakeEnv):
        def __init__(self, *a, **kw): super().__init__()
        def get_state_hash(self): return 'fixture-state'
    monkeypatch.setattr(lifemate.models, 'LocalASR', ASR)
    monkeypatch.setattr(lifemate.transcripts, 'asr_identity', lambda *a: {'model': 'stub-asr', 'precision': 'int8'})
    monkeypatch.setitem(sys.modules, 'zaic_tools', SimpleNamespace(ToolEnv=Env))
    golds = {r['id']: r['label']['tool_calls'] for r in records}
    assignments = []
    for attempt in range(2):
        transcribed_before = len(transcribed)
        output = tmp_path / f'sft-{attempt}'
        monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
            '--output-dir', str(output), '--asr-model', 'stub-asr',
            '--transcript-mode', 'mix',
            '--validation-fraction', str(validation_fraction)])
        prepare_sft.main()
        saved = json.loads((output / 'manifest.json').read_text())
        assert saved['transcript_mixing']['asr_ratio'] == 0.5
        assert saved['transcript_mixing']['unit'] == 'sample'
        assert saved['asr']['model'] == 'stub-asr'
        sources = {}
        for partition in saved['partitions']:
            info = saved['partitions'][partition]
            assert not set(sources) & set(info['sample_ids'])
            sources.update(info['transcript_sources'])
            counts = Counter(info['transcript_sources'].values())
            assert all(info['transcript_source_counts'][key] == counts[key] for key in ('audio_text', 'asr'))
            assert abs(counts['audio_text'] - counts['asr']) <= 1
            steps = [json.loads(line) for line in (output / f'{partition}.jsonl').read_text().splitlines()]
            for step in steps:
                sample_id = step['sample_id']
                expected_source = info['transcript_sources'][sample_id]
                assert step['transcript_source'] == expected_source
                prompt = json.loads(step['messages'][1]['content'])
                expected = f'Human {sample_id.removeprefix("train_")}!' if expected_source == 'audio_text' else f'Asr {sample_id}: lỗi nhận dạng!'
                assert prompt['request_transcript'] == expected
                assert prompt['conversation_history'] == [{'role': 'user', 'text': 'Full history!'}]
                # Gold targets and previous real observations are source-independent.
                decision = json.loads(step['messages'][-1]['content'])
                if decision['action'] == 'call':
                    assert decision['name'] == golds[sample_id][step['step']]['name']
        asr_count = int(samples * 0.5 + 0.5)
        assert Counter(sources.values()) == Counter({'asr': asr_count, 'audio_text': samples - asr_count})
        validation_ids = set(saved['partitions'].get('validation', {}).get('sample_ids', []))
        expected_asr_ids = {sid for sid, source in sources.items() if source == 'asr'} | validation_ids
        # Selected SFT ASR + every held-out request, each transcribed once.
        assert Counter(transcribed[transcribed_before:]) == {sid: 1 for sid in expected_asr_ids}
        transcribed_after_preparation = len(transcribed)
        assignments.append(sources)
        if validation_fraction == 0:
            assert set(saved['partitions']) == {'train'}
            assert set(saved['partitions']['train']['sample_ids']) == {row['id'] for row in records}
            assert not list(output.glob('*rollouts*'))
            assert not (output / 'validation.jsonl').exists()
            manifest_path.write_text(''.join(json.dumps(r) + '\n' for r in reversed(records)))
            continue
        evaluator = StateToolEvaluator(output, saved)
        rollout_transcripts = []
        class LLM:
            def generate(self, system, user):
                rollout_transcripts.append(json.loads(user)['request_transcript'])
                return '{"action":"done"}'
        try:
            metrics = evaluator.evaluate(LLM(), output / 'scores.jsonl')
            assert metrics['eval_rollout_samples'] == len(evaluator.rows)
            assert rollout_transcripts == [row['prepared_transcript']['transcript'] for row in evaluator.rows]
            for row in evaluator.rows:
                assert row['prepared_transcript']['transcript_source'] == 'asr'
                assert row['prepared_transcript']['transcript'] == f'Asr {row["item"]["id"]}: lỗi nhận dạng!'
                assert row['clean_prepared_transcript']['transcript_source'] == 'audio_text'
                assert row['clean_prepared_transcript']['history'] == row['prepared_transcript']['history']
        finally:
            evaluator.close()
        assert len(transcribed) == transcribed_after_preparation  # Evaluation never decodes audio again.
        manifest_path.write_text(''.join(json.dumps(r) + '\n' for r in reversed(records)))
    assert assignments[0] == assignments[1]


@pytest.mark.parametrize('ratio,source', [(0, 'audio_text'), (1, 'asr')])
def test_explicit_single_source_ablation(ratio, source):
    records = [{'id': f'train_{i}'} for i in range(6)]
    assert set(assign_transcript_sources(records, ratio).values()) == {source}


def test_default_dual_mode_replays_both_gold_and_asr(tmp_path, monkeypatch):
    import lifemate.models
    import lifemate.transcripts
    import prepare_sft
    source = tmp_path / 'train'
    source.mkdir()
    records = [dict(labeled(), id=f'train_{i}', audio=f'train_{i}.wav',
                    audio_text=f'Human {i}') for i in range(4)]
    for i in range(4):
        (source / f'train_{i}.wav').write_bytes(b'audio')
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(CATALOG.values())}]}))
    (source / 'train.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    class ASR:
        def __init__(self, *a, **kw): pass
        def transcribe(self, path): return f'ASR {path.stem}'
    class Env(FakeEnv):
        def __init__(self, *a, **kw): super().__init__()
        def get_state_hash(self): return 'fixture-state'
    monkeypatch.setattr(lifemate.models, 'LocalASR', ASR)
    monkeypatch.setattr(lifemate.transcripts, 'asr_identity', lambda *a: {'model': 'stub-asr', 'precision': 'int8'})
    monkeypatch.setitem(sys.modules, 'zaic_tools', SimpleNamespace(ToolEnv=Env))
    output = tmp_path / 'sft-dual'
    monkeypatch.setattr(sys, 'argv', ['prepare_sft.py', '--data-root', str(source),
        '--output-dir', str(output), '--asr-model', 'stub-asr', '--validation-fraction', '0'])
    prepare_sft.main()
    saved = json.loads((output / 'manifest.json').read_text())
    info = saved['partitions']['train']
    assert saved['sft_format']['transcript_mode'] == 'dual'
    assert info['transcript_source_counts'] == {'audio_text': 4, 'asr': 4}
    rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
    assert {(r['sample_id'], r['transcript_source']) for r in rows} == {
        (f'train_{i}', src) for i in range(4) for src in ('audio_text', 'asr')}
