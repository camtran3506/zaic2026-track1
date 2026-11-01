"""Numeric-preserving transcript normalization across the pipeline."""
import json
from types import SimpleNamespace

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent, _train_transcript, history_text
from lifemate.models import LocalASR
from lifemate.sft import prepare_train_transcript, replay_sample
from lifemate.text import TEXT_NORMALIZATION_VERSION, normalize_transcript
from lifemate.transcripts import cache_paths, dataset_identity, load_transcripts, prepare_transcripts
from test_baseline import CATALOG, FakeEnv, item


@pytest.mark.parametrize(('raw', 'expected'), [
    ('  BA\u0323\u0302T ĐÈN, PHÒNG KHÁCH!\n', 'bật đèn phòng khách'),
    ('“Xin—chào”… (Hà_Nội); a/b\\c', 'xin chào hà nội a b c'),
    ('08:30 ngày 29/10/2026; 3.14, 1.000₫', '08:30 ngày 29/10/2026 3.14 1.000₫'),
    ('  Ấm\t\tđun\r\nnước\u00a0  2  ', 'ấm đun nước 2'),
    ('ĐỪNG bỏ dấu, đừng lặp lặp + 20% = 2 😊', 'đừng bỏ dấu đừng lặp lặp + 20% = 2 😊'),
    ('', ''),
    ('\t...!!!\n', ''),
])
def test_normalize_transcript_unicode_and_whitespace(raw, expected):
    assert normalize_transcript(raw) == expected
    assert normalize_transcript(expected) == expected


@pytest.mark.parametrize(('raw', 'expected'), [
    (' Nhắc lúc 08:30, ngày 29/10/2026! ', 'nhắc lúc 08:30 ngày 29/10/2026'),
    ('Ngày 2026-10-29; lúc 8:05:09.', 'ngày 2026-10-29 lúc 8:05:09'),
    ('Giá 1.234,56 ₫; 1,234.56 USD.', 'giá 1.234,56 ₫ 1,234.56 usd'),
    ('Giảm -5%; tăng +2,5 %; còn 0.75.', 'giảm -5% tăng +2,5 % còn 0.75'),
    ('Nhiệt độ -5 độ; tăng +3.', 'nhiệt độ -5 độ tăng +3'),
    ('BẬT ĐÈN!!!\nPhòng_khách.', 'bật đèn phòng khách'),
    ('tám giờ rưỡi ngày mai', 'tám giờ rưỡi ngày mai'),
    ('99:99 và 31/02/2026', '99:99 và 31/02/2026'),
    ('08 30; 3 14', '08 30 3 14'),
])
def test_llm_normalization_preserves_numeric_structure_without_guessing(raw, expected):
    assert normalize_transcript(raw) == expected
    assert normalize_transcript(expected) == expected


def test_local_asr_normalizes_joined_segments_before_caching(tmp_path, monkeypatch):
    import faster_whisper
    calls = []
    class Whisper:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, path, **kwargs):
            calls.append(path)
            return iter([SimpleNamespace(text=' BẬT đèn, '),
                         SimpleNamespace(text=' lúc 08:30!\n')]), None
    monkeypatch.setattr(faster_whisper, 'WhisperModel', Whisper)
    audio = tmp_path / 'speech.wav'
    audio.write_bytes(b'fixture; decoding stubbed')
    asr = LocalASR(tmp_path, device='cpu')
    assert asr.transcribe(audio) == 'bật đèn lúc 08:30'
    assert asr.transcribe(audio) == 'bật đèn lúc 08:30'
    assert len(calls) == 1


@pytest.mark.parametrize('audio_text', [
    ' BẬT ĐÈN! ', {'latest': ' BẬT ĐÈN! '}, {'current': ' BẬT ĐÈN! '},
    ['cũ', {'audio_text': ' BẬT ĐÈN! '}], ['cũ', ' BẬT ĐÈN! '],
])
def test_train_audio_text_formats_are_normalized(audio_text):
    assert _train_transcript({'audio_text': audio_text}) == 'bật đèn'


class RawASR:
    def transcribe(self, path): return ' BẬT đèn, lúc 08:30! '


def test_cache_direct_inference_and_sft_share_normalized_spoken_text(tmp_path):
    entry = dict(item(), audio='current.wav', audio_text=' BẬT đèn, lúc 08:30! ',
                 label={'tool_calls': []}, history=[
                     {'role': 'user', 'audio_text': ' TẮT ĐÈN! '},
                     {'role': 'human', 'content': ' Ở ĐÂU? '},
                     {'role': 'assistant', 'text': 'ID: light_living; 08:30'},
                     {'role': 'user', 'audio': 'history.wav'},
                 ])
    for name in ('current.wav', 'history.wav'):
        (tmp_path / name).write_bytes(b'fixture')
    identity = dataset_identity([entry], tmp_path, 'train', use_train_transcript=True)
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, RawASR)
    rows, manifest = load_transcripts(cache, identity)
    expected = 'bật đèn lúc 08:30'
    assert rows[0]['transcript'] == expected
    assert rows[0]['history'] == [
        {'role': 'user', 'text': 'tắt đèn'},
        {'role': 'human', 'text': 'ở đâu'},
        {'role': 'assistant', 'text': 'ID: light_living; 08:30'},
        {'role': 'user', 'text': expected},
    ]
    assert manifest['text_normalization'] == TEXT_NORMALIZATION_VERSION
    prepared = prepare_train_transcript(entry, tmp_path, RawASR())
    assert prepared['transcript'] == rows[0]['transcript']
    assert prepared['history'] == rows[0]['history']
    payload = json.loads(replay_sample(entry, CATALOG, FakeEnv(), tmp_path,
                                      prepared_transcript=prepared)[0]['messages'][1]['content'])
    assert payload['request_transcript'] == expected
    class LLM:
        def generate(self, system, user):
            actual = json.loads(user)
            assert actual['request_transcript'] == payload['request_transcript']
            assert actual['conversation_history'] == payload['conversation_history']
            return '{"action":"done"}'
    for use_train in (False, True):
        agent = LifeMateAgent(RawASR(), LLM(), CATALOG, AgentConfig(use_train_transcript=use_train))
        assert agent.run(entry, FakeEnv(), tmp_path, 'train')['transcript'] == expected
        assert agent.run(entry, FakeEnv(), tmp_path, 'train', prepared_transcript=rows[0])['transcript'] == expected
    # Explicit audio text is normalized even when the speaker is not user/human.
    assert history_text([{'role': 'assistant', 'audio_text': 'CHÀO!'}],
                        tmp_path, None, 0, float('inf')) == [{'role': 'assistant', 'text': 'chào'}]


@pytest.mark.parametrize('marker', [None, 'outdated-normalization'])
def test_old_complete_cache_requires_refresh(tmp_path, marker):
    (tmp_path / 'current.wav').write_bytes(b'fixture')
    entries = [dict(item(), audio='current.wav')]
    identity = dataset_identity(entries, tmp_path, 'public_test')
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR)
    manifest_path = cache_paths(cache)[0]
    manifest = json.loads(manifest_path.read_text())
    manifest.pop('text_normalization')
    if marker is not None:
        manifest['text_normalization'] = marker
    manifest_path.write_text(json.dumps(manifest))
    def must_not_load(): raise AssertionError('Old cache must be rejected before ASR loads')
    with pytest.raises(ValueError, match='normalization changed'):
        load_transcripts(cache, identity)
    with pytest.raises(ValueError, match='normalization changed'):
        prepare_transcripts(entries, tmp_path, cache, identity, {}, must_not_load)
    prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR, refresh=True)
    assert load_transcripts(cache, identity)[0][0]['transcript'] == 'bật đèn lúc 08:30'


def test_old_partial_cache_cannot_mix_normalization_rules(tmp_path):
    entries = [dict(item(), id=f'public_{i}', audio=f'{i}.wav') for i in range(2)]
    for name in ('0.wav', '1.wav'):
        (tmp_path / name).write_bytes(b'fixture')
    identity = dataset_identity(entries, tmp_path, 'public_test')
    cache = tmp_path / 'transcripts.jsonl'
    class FailingASR(RawASR):
        def transcribe(self, path):
            if path.name == '1.wav':
                raise RuntimeError('fixture ASR failure')
            return super().transcribe(path)
    with pytest.raises(RuntimeError, match='fixture ASR failure'):
        prepare_transcripts(entries, tmp_path, cache, identity, {}, FailingASR)
    _, partial, partial_manifest = cache_paths(cache)
    header = json.loads(partial_manifest.read_text())
    header.pop('text_normalization')
    partial_manifest.write_text(json.dumps(header))
    # Simulate a raw prefix produced before normalization was introduced.
    row = json.loads(partial.read_text())
    row['transcript'] = 'BẬT ĐÈN!'
    partial.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Partial ASR cache settings changed'):
        prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR)
    prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR, refresh=True)
    rows, _ = load_transcripts(cache, identity)
    assert [r['transcript'] for r in rows] == ['bật đèn lúc 08:30'] * 2


def test_punctuation_only_text_does_not_trigger_asr_fallback(tmp_path):
    entry = dict(item(), audio_text='?!', history=[
        {'role': 'user', 'audio_text': '...', 'audio': 'missing-history.wav'},
    ])
    def must_not_load(): raise AssertionError('Available text must not require ASR')
    # Neither current audio nor history audio exists; available train text suffices.
    identity = dataset_identity([entry], tmp_path, 'train', use_train_transcript=True)
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, must_not_load)
    row = load_transcripts(cache, identity)[0][0]
    assert row['transcript'] == ''
    assert row['history'] == []
    assert prepare_train_transcript(entry, tmp_path)['history'] == []
