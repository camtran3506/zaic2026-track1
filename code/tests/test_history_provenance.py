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


def test_history_prompt_window():
    history = HISTORY + [{'role': 'user', 'text': str(i)} for i in range(12)]
    resolved = prompt_history(resolve_history(history, None))
    assert len(resolved) == 12
    assert all('tool_results' not in turn for turn in resolved)
