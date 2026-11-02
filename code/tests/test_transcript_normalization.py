"""Shared normalization across clean/ASR text, history, caching and SFT."""
import json

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent, _train_transcript, history_text
from lifemate.sft import prepare_train_transcript, replay_sample
from lifemate.transcripts import cache_paths, dataset_identity, load_transcripts, prepare_transcripts
from test_baseline import CATALOG, FakeEnv, item
from lifemate.text import TEXT_NORMALIZATION


RAW = '  BA\u0323\u0302T ĐÈN!\tEmail: A_B@Example.com; 08:30, -5%\n'

NORMALIZED = 'Bật đèn! email: a_b@example.com; 08:30, -5%.'


class RawASR:
    def transcribe(self, path):
        return RAW


@pytest.mark.parametrize('audio_text', [RAW, {'latest': RAW}, {'current': RAW},
    ['cũ', {'audio_text': RAW}], ['cũ', RAW], {'transcript': RAW}])
def test_train_transcript_formats_preserve_original_text(audio_text):
    assert _train_transcript({'audio_text': audio_text}) == RAW


@pytest.mark.parametrize('use_train', [False, True])
def test_cache_direct_inference_and_sft_share_normalized_text(tmp_path, use_train):
    entry = dict(item(), audio='current.wav', audio_text=RAW,
                 label={'tool_calls': []}, history=[
                     {'role': 'user', 'audio_text': ' TẮT ĐÈN! '},
                     {'role': 'human', 'content': ' Ở ĐÂU? '},
                     {'role': 'assistant', 'text': 'Id: light_living; 08:30.'},
                     {'role': 'user', 'audio': 'history.wav'},
                 ])
    for name in ('current.wav', 'history.wav'):
        (tmp_path / name).write_bytes(b'fixture audio; ASR stubbed')
    identity = dataset_identity([entry], tmp_path, 'train', use_train_transcript=use_train)
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, RawASR)
    rows, manifest = load_transcripts(cache, identity)
    assert rows[0]['transcript'] == NORMALIZED
    assert rows[0]['history'] == [
        {'role': 'user', 'text': 'Tắt đèn!'},
        {'role': 'human', 'text': 'Ở đâu?'},
        {'role': 'assistant', 'text': 'Id: light_living; 08:30.'},
        {'role': 'user', 'text': NORMALIZED},
    ]
    assert manifest['version'] == 4
    assert manifest['text_normalization'] == TEXT_NORMALIZATION
    # Include both provided train text and missing-text fallback to actual ASR output.
    train_entry = entry if use_train else {k: v for k, v in entry.items() if k != 'audio_text'}
    prepared = prepare_train_transcript(train_entry, tmp_path, RawASR())
    assert prepared['transcript'] == NORMALIZED
    assert prepared['history'] == rows[0]['history']
    payload = json.loads(replay_sample(entry, CATALOG, FakeEnv(), tmp_path,
                                      prepared_transcript=prepared)[0]['messages'][1]['content'])
    assert payload['request_transcript'] == NORMALIZED

    class LLM:
        def generate(self, system, user):
            actual = json.loads(user)
            assert actual['request_transcript'] == NORMALIZED
            assert actual['conversation_history'] == payload['conversation_history']
            return '{"action":"done"}'

    agent = LifeMateAgent(RawASR(), LLM(), CATALOG, AgentConfig(use_train_transcript=use_train))
    assert agent.run(entry, FakeEnv(), tmp_path, 'train')['transcript'] == NORMALIZED
    assert agent.run(entry, FakeEnv(), tmp_path, 'train', prepared_transcript=rows[0])['transcript'] == NORMALIZED
    assert history_text([{'role': 'assistant', 'audio_text': RAW}],
                        tmp_path, None, 0, float('inf')) == [{'role': 'assistant', 'text': NORMALIZED}]


@pytest.mark.parametrize('old_version', [1, 2, 3])
def test_old_complete_cache_requires_refresh(tmp_path, old_version):
    (tmp_path / 'current.wav').write_bytes(b'fixture')
    entries = [dict(item(), audio='current.wav')]
    identity = dataset_identity(entries, tmp_path, 'public_test')
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR)
    manifest_path = cache_paths(cache)[0]
    manifest = json.loads(manifest_path.read_text())
    manifest['version'] = old_version
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='cache format changed'):
        load_transcripts(cache, identity)
    with pytest.raises(ValueError, match='cache format changed'):
        prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR)
    prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR, refresh=True)
    assert load_transcripts(cache, identity)[0][0]['transcript'] == NORMALIZED


def test_old_partial_prefix_is_rebuilt_instead_of_mixing_text(tmp_path):
    entries = [dict(item(), id=f'public_{i}', audio=f'{i}.wav') for i in range(2)]
    for i in range(2):
        (tmp_path / f'{i}.wav').write_bytes(b'fixture')
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
    header['version'] = 1
    partial_manifest.write_text(json.dumps(header))
    row = json.loads(partial.read_text())
    row['transcript'] = 'previously transformed text'
    partial.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='Partial ASR cache settings changed'):
        prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR)
    prepare_transcripts(entries, tmp_path, cache, identity, {}, RawASR, refresh=True)
    rows, _ = load_transcripts(cache, identity)
    assert [r['transcript'] for r in rows] == [NORMALIZED] * 2


def test_punctuation_only_text_is_preserved_without_asr(tmp_path):
    entry = dict(item(), audio_text='?!', history=[
        {'role': 'user', 'audio_text': '...', 'audio': 'missing-history.wav'},
    ])

    def must_not_load():
        raise AssertionError('Available text must not require ASR')

    identity = dataset_identity([entry], tmp_path, 'train', use_train_transcript=True)
    cache = tmp_path / 'transcripts.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, must_not_load)
    row = load_transcripts(cache, identity)[0][0]
    assert row['transcript'] == '?!'
    assert row['history'] == [{'role': 'user', 'text': '...'}]
    assert prepare_train_transcript(entry, tmp_path)['history'] == row['history']


@pytest.mark.parametrize('policy', [None, 'old-normalization'])
def test_cache_with_different_normalization_requires_refresh(tmp_path, policy):
    entry = dict(item(), audio='request.wav')
    (tmp_path / 'request.wav').write_bytes(b'fixture')
    identity = dataset_identity([entry], tmp_path, 'train')
    cache = tmp_path / 'cache.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, RawASR)
    path = cache_paths(cache)[0]
    manifest = json.loads(path.read_text())
    if policy is None:
        manifest.pop('text_normalization')
    else:
        manifest['text_normalization'] = policy
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='normalization changed'):
        load_transcripts(cache, identity)


def test_structured_tool_data_and_sources_are_unchanged_by_text_normalization(tmp_path):
    from copy import deepcopy
    from test_sft import labeled
    entry = labeled()
    entry['history'] = [{'role': 'assistant', 'text': '', 'tool_calls': [
        {'name': 'smart_home.get_status', 'arguments': {}}], 'tool_results': [{'ok': True}]}]
    entry['audio_text'] = '  GỬI	TIN NHẮN  '
    entry['history'][0]['text'] = '  ĐÃ GỬI  '
    entry['history'][0]['tool_calls'][0]['arguments'] = {'literal': ' A_B@Example.COM\n  Id_X '}
    entry['history'][0]['tool_results'][0]['literal'] = ' A_B@Example.COM\n  Id_X '
    original = deepcopy(entry)
    prepared = prepare_train_transcript(entry, tmp_path)
    steps = replay_sample(entry, CATALOG, FakeEnv(), tmp_path, prepared_transcript=prepared)
    payload = json.loads(steps[0]['messages'][1]['content'])
    turn = payload['conversation_history'][0]
    assert payload['request_transcript'] == 'Gửi tin nhắn.'
    assert turn['text'] == 'Đã gửi.'
    assert turn['tool_calls'] == original['history'][0]['tool_calls']
    assert turn['tool_results'] == original['history'][0]['tool_results']
    assert payload['available_tools']['tool_schemas'] == [CATALOG[name] for name in sorted(CATALOG)]
    assert entry == original
    assert [json.loads(s['messages'][-1]['content'])['arguments'] for s in steps[:-1]] == [
        call['arguments'] for call in original['label']['tool_calls']]
