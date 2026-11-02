"""Historical tool results reach the planner in SFT, direct and cached inference."""
from copy import deepcopy
import json

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.history import prompt_history, resolve_history
from lifemate.sft import prepare_train_transcript, replay_sample
from lifemate.transcripts import dataset_identity, load_transcripts, prepare_transcripts
from test_baseline import CATALOG as BASE_CATALOG


CATALOG = {
    'sys.ask_missing_info': BASE_CATALOG['sys.ask_missing_info'],
    'chat_ott.list_chats': {'name': 'chat_ott.list_chats', 'kind': 'read',
                           'parameters': {'type': 'object', 'properties': {}}},
    'chat_ott.send_message': {'name': 'chat_ott.send_message', 'kind': 'write',
        'parameters': {'type': 'object', 'properties': {
            'chat_id': {'type': 'string', 'x-ref': 'chat_ott.chats'},
            'content': {'type': 'string'}}}},
}
HISTORY = [{'role': 'assistant', 'text': 'Đã gửi giá dầu.', 'tool_calls': [
    {'name': 'chat_ott.list_chats', 'arguments': {}},
    {'name': 'chat_ott.send_message', 'arguments': {'chat_id': 'chat_3071', 'content': '20.600'}},
], 'tool_results': [
    {'ok': True, 'chats': [{'id': 'chat_918', 'name': 'Grace Nakamura'},
                           {'id': 'chat_3071', 'name': 'Grace Nakamoto'}]},
    {'ok': True},
]}]


class Env:
    def __init__(self): self.calls = []
    def validate(self, name, args): return {'ok': True}
    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return {'ok': True}


class ASR:
    def transcribe(self, path): return 'Gửi lại đúng người rồi nhắn người kia xin lỗi.'


def record():
    return {'id': 'train_1484', 'audio': 'current.wav', 'audio_text': ASR().transcribe(None),
            'context': {'installed_apps': {'chat_ott': {'tools': list(CATALOG)}}},
            'history': deepcopy(HISTORY), 'label': {'tool_calls': [
                {'name': 'chat_ott.send_message', 'arguments': {'chat_id': target, 'content': text}}
                for target, text in [('chat_918', '20.600'), ('chat_3071', 'Xin lỗi, gửi nhầm.')]]}}


def test_sft_direct_and_cached_inference_share_history_evidence_without_replaying_it(tmp_path):
    entry = record()
    prepared = prepare_train_transcript(entry, tmp_path)
    env = Env()
    examples = replay_sample(entry, CATALOG, env, tmp_path, prepared_transcript=prepared)
    assert env.calls == entry['label']['tool_calls']
    payload = json.loads(examples[0]['messages'][1]['content'])
    assert payload['previous_actions_and_results'] == []
    assert payload['conversation_history'][0]['tool_results'] == HISTORY[0]['tool_results']
    assert 'Xin lỗi, gửi nhầm.' not in examples[0]['messages'][1]['content']

    (tmp_path / 'current.wav').write_bytes(b'fixture; ASR stubbed')
    identity = dataset_identity([entry], tmp_path, 'train')
    cache = tmp_path / 'cache.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, ASR)
    cached = load_transcripts(cache, identity)[0][0]
    assert cached['history'] == prepared['history']
    for transcript in (None, cached):
        class LLM:
            def __init__(self): self.steps = iter(examples)
            def generate(self, system, user):
                step = next(self.steps)
                assert json.loads(user)['conversation_history'] == payload['conversation_history']
                return step['messages'][-1]['content']
        inference_env = Env()
        stats = LifeMateAgent(ASR(), LLM(), CATALOG, AgentConfig()).run(
            entry, inference_env, tmp_path, 'train', prepared_transcript=transcript)
        assert stats['invalid_proposals'] == 0
        assert inference_env.calls == entry['label']['tool_calls']


@pytest.mark.parametrize('mutation', ['user', 'mismatch'])
def test_unpaired_or_user_tool_metadata_is_not_exposed_as_assistant_results(mutation):
    history = deepcopy(HISTORY)
    if mutation == 'user':
        history[0]['role'] = 'user'
    else:
        history[0]['tool_results'].pop()
    resolved = prompt_history(resolve_history(history, None))
    assert 'tool_results' not in resolved[0]


def test_tool_only_history_survives():
    history = deepcopy(HISTORY)
    history[0].pop('text')
    resolved = prompt_history(resolve_history(history, None))
    assert resolved[0]['text'] == ''
    assert resolved[0]['tool_results'] == HISTORY[0]['tool_results']


def test_old_tool_only_history_is_retained_with_all_later_turns():
    history = deepcopy(HISTORY) + [{'role': 'user', 'text': str(i)} for i in range(20)]
    history[0].pop('text')
    resolved = prompt_history(resolve_history(history, None))
    assert len(resolved) == 21
    assert resolved[0]['tool_results'] == HISTORY[0]['tool_results']
    assert resolved[-1]['text'] == '19'


@pytest.mark.parametrize('result_kind', ['object', 'string', 'error'])
def test_large_current_results_and_all_actions_match_sft_direct_and_cached_inference(tmp_path, result_kind):
    entry = record()
    entry['label']['tool_calls'] = [entry['label']['tool_calls'][0]] * 10
    entry['history'] += [{'role': 'user', 'text': f'History {i}: ' + 'giữ! ' * 500}
                         for i in range(20)]
    result_text = 'Kết quả cần giữ nguyên: ' * 1000
    class LargeEnv(Env):
        def call(self, name, args):
            super().call(name, args)
            if result_kind == 'error':
                raise RuntimeError(result_text)
            if result_kind == 'string':
                return result_text
            return {'ok': True, 'rows': [{'id': 'target', 'text': result_text}],
                    'tail': {'important': 'Không được mất'}}
    prepared = prepare_train_transcript(entry, tmp_path)
    examples = replay_sample(entry, CATALOG, LargeEnv(), tmp_path, prepared_transcript=prepared)
    assert len(examples) == 11
    final = json.loads(examples[-1]['messages'][1]['content'])
    assert len(final['previous_actions_and_results']) == 10
    for action in final['previous_actions_and_results']:
        result = json.loads(action['result'])
        if result_kind == 'error':
            assert result == {'ok': False, 'error': repr(RuntimeError(result_text))}
        elif result_kind == 'string':
            assert result == result_text
        else:
            assert result == {'ok': True, 'rows': [{'id': 'target', 'text': result_text}],
                              'tail': {'important': 'Không được mất'}}
    (tmp_path / 'current.wav').write_bytes(b'fixture; ASR stubbed')
    identity = dataset_identity([entry], tmp_path, 'train')
    cache = tmp_path / 'cache.jsonl'
    prepare_transcripts([entry], tmp_path, cache, identity, {}, ASR)
    cached = load_transcripts(cache, identity)[0][0]
    for transcript in (None, cached):
        class LLM:
            def __init__(self): self.steps = iter(examples); self.count = 0
            def generate(self, system, user):
                step = next(self.steps)
                assert system == step['messages'][0]['content']
                assert user == step['messages'][1]['content']
                self.count += 1
                return step['messages'][-1]['content']
        llm, inference_env = LLM(), LargeEnv()
        stats = LifeMateAgent(ASR(), llm, CATALOG, AgentConfig(max_calls=11)).run(
            entry, inference_env, tmp_path, 'train', prepared_transcript=transcript)
        assert llm.count == 11 and stats['executed'] == 10
        assert stats['invalid_proposals'] == 0
        assert inference_env.calls == entry['label']['tool_calls']
