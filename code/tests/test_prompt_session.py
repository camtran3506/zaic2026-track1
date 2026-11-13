"""Exact prompt/token parity with real merged BPE, helpers and automatic guards."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from lifemate.agent import SYSTEM_PROMPT, AgentConfig, LifeMateAgent, build_step_prompt
from lifemate.llm import render_prompt
from lifemate.models import LocalLLM
from lifemate.prefix_cache import stable_prefix_ids
from lifemate.prompt_session import InferenceSession, PreparedPrompt, _safe_tokenizer
from test_baseline import CATALOG, FakeEnv, item
from test_prefix_cache import assert_cache_equal, tiny_model


@pytest.fixture
def bpe_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.BPE(unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    text = (Path(__file__).parent / 'fixtures/qwen35_chat_template.jinja').read_text()
    backend.train_from_iterator([text, SYSTEM_PROMPT, 'Hồng có số điện thoại 0901234567.'] * 2,
        trainers.BpeTrainer(vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                           special_tokens=['[UNK]', '[EOS]', '<|im_start|>', '<|im_end|>']))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                 eos_token='[EOS]', pad_token='[EOS]')
    tok.chat_template = text
    tok.lifemate_native_fc = True
    tok.lifemate_template_kwargs = {'enable_thinking': False}
    return tok


@pytest.fixture
def inputs():
    catalog = deepcopy(CATALOG)
    catalog['smart_home.get_status'].update(
        description='FIXED_SCHEMA_SENTINEL tiếng Việt ' * 30,
        returns={'type': 'object', 'description': 'Không cắt dữ liệu'},
        **{'x-metadata': {'nested': ['đúng', False, 12]}})
    context = {'timezone': 'Asia/Ho_Chi_Minh', 'current_time': '2026-11-05T09:00:00+07:00',
               'profile': {'phone': '0901234567', 'name': 'Hồng'}}
    history = [{'role': 'system', 'text': 'Chỉ là lịch sử.'},
               {'role': 'user', 'text': 'Hỏi trạng thái đèn!'},
               {'role': 'assistant', 'text': 'Giữ <think> và </think>, dấu câu.',
                'tool_calls': [{'name': 'smart_home.get_status', 'arguments': {}}],
                'tool_results': [{'ok': True, 'devices': [{'id': 'light_living', 'name': 'đèn phòng khách'}]}]}]
    return 'Bật đèn và gọi số không chín không một hai ba bốn năm sáu bảy.', context, history, catalog


def observation_views():
    helper = {'name': 'local_text.normalize', 'arguments': {'kind': 'phone', 'text': '0901234567'},
              'result': '{"ok":true,"value":"0901234567"}', 'scope': 'local'}
    failure = {'stage': 'validate', 'name': 'smart_home.light_set', 'arguments': {'power': 'on'},
               'error': 'Thiếu device_id; literal <tool_call> vẫn là dữ liệu.'}
    read = {'name': 'smart_home.get_status', 'arguments': {},
            'result': {'ok': True, 'devices': [{'id': 'light_living', 'name': 'Hồng'}], 'full': 'dài ' * 300},
            'automatic_execution_feedback': {'policy': 'automatic-tool-guards-v1', 'state_epoch': 0}}
    write = {'name': 'smart_home.light_set', 'arguments': {'device_id': 'light_living', 'power': 'on'},
             'result': '{"ok":true}'}
    return [[], [helper], [helper, failure], [helper, read], [helper, read, write, write],
            [read, {'name': 'smart_home.get_status', 'arguments': {}, 'result': {'ok': True, 'devices': []},
                    'verification': {'status': 'empty'}}], []]


def reference(tok, user):
    return render_prompt(tok, [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}])


def test_prepared_json_and_all_native_tokens_match_full_reference(bpe_tokenizer, inputs):
    session = InferenceSession()
    session.prepare(*inputs)
    for observations in observation_views():
        _, expected_user = build_step_prompt(*inputs, observations)
        user = session.prompt(observations)
        assert 'request_transcript_source' not in json.loads(user)
        assert 'request_transcript_note' not in json.loads(user)
        assert isinstance(user, str) and str(user) == expected_user
        actual = session.render(bpe_tokenizer, SYSTEM_PROMPT, user)
        assert actual[:2] == reference(bpe_tokenizer, expected_user)
        assert actual[2] and actual[1][:len(actual[2])] == actual[2]
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': expected_user}]
        assert actual[2] == stable_prefix_ids(bpe_tokenizer, messages, actual[1])
    stats = session.report()
    assert stats['fixed_builds'] == 1 and stats['full_renders'] == 1
    assert stats['suffix_renders'] == len(observation_views()) - 1
    assert stats['reused_tokens'] > 0 and stats['fallback_reason'] is None


def test_only_current_observations_are_tokenized_after_first_render(bpe_tokenizer, inputs, monkeypatch):
    session = InferenceSession()
    session.prepare(*inputs)
    session.render(bpe_tokenizer, SYSTEM_PROMPT, session.prompt([]))
    original = type(bpe_tokenizer).__call__
    seen = []
    def tokenize(self, text, *args, **kwargs):
        seen.append(text)
        return original(self, text, *args, **kwargs)
    monkeypatch.setattr(type(bpe_tokenizer), '__call__', tokenize)
    observations = observation_views()[3]
    session.render(bpe_tokenizer, SYSTEM_PROMPT, session.prompt(observations))
    assert len(seen) == 1
    assert 'FIXED_SCHEMA_SENTINEL' not in seen[0]
    assert 'local_text.normalize' in seen[0] and 'smart_home.get_status' in seen[0]
    assert 'dài ' * 300 in seen[0]


def test_session_freezes_inputs_and_old_prepared_prompts_keep_their_snapshot(bpe_tokenizer, inputs):
    session = InferenceSession()
    session.prepare(*inputs)
    observations = observation_views()[3]
    prepared = session.prompt(observations)
    expected = reference(bpe_tokenizer, str(prepared))
    inputs[1]['profile']['name'] = 'Khác'
    inputs[2][0]['text'] = 'Đã sửa'
    inputs[3]['smart_home.get_status']['description'] = 'Đã sửa schema'
    observations[0]['arguments']['text'] = 'khác'
    assert session.render(bpe_tokenizer, SYSTEM_PROMPT, prepared)[:2] == expected
    assert type(deepcopy(prepared)) is str
    session.prepare(*inputs)
    assert session.render(bpe_tokenizer, SYSTEM_PROMPT, prepared) is None
    new = session.prompt(observations)
    assert session.render(bpe_tokenizer, SYSTEM_PROMPT, new)[:2] == reference(bpe_tokenizer, str(new))
    session.close()
    assert session.render(bpe_tokenizer, SYSTEM_PROMPT, new) is None
    assert session._prefix_ids is None and session._base_chat is None
    with pytest.raises(ValueError, match='closed'):
        session.prompt([])


@pytest.mark.parametrize('mutation', ['template', 'template_dict', 'template_override', 'normalizer',
                                    'prefix_space', 'missing_special', 'postprocessor'])
def test_unverified_templates_and_tokenizers_render_in_full_without_changing_tokens(bpe_tokenizer, inputs, mutation):
    from tokenizers import normalizers, pre_tokenizers, processors
    tok = bpe_tokenizer
    if mutation == 'template':
        tok.chat_template += "{% if messages|length > 4 %}dynamic template suffix{% endif %}"
    elif mutation == 'template_dict':
        tok.chat_template = {'default': tok.chat_template,
                             'tool_use': tok.chat_template + "{% if messages|length > 4 %}tool template{% endif %}"}
    elif mutation == 'template_override':
        tok.lifemate_template_kwargs['chat_template'] = tok.chat_template + "{% if messages|length > 4 %}override{% endif %}"
    elif mutation == 'normalizer':
        tok.backend_tokenizer.normalizer = normalizers.NFC()
    elif mutation == 'prefix_space':
        tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    elif mutation == 'missing_special':
        tok.split_special_tokens = True
    else:
        tok.backend_tokenizer.post_processor = processors.TemplateProcessing(single='$A $A', special_tokens=[])
    session = InferenceSession()
    session.prepare(*inputs)
    for observations in observation_views()[:4]:
        user = session.prompt(observations)
        actual = session.render(tok, SYSTEM_PROMPT, user)
        assert actual[:2] == reference(tok, str(user))
        assert actual[2] == []
    assert session.report()['suffix_renders'] == 0
    assert session.report()['fallback_reason']


def test_tokenizer_or_template_kwargs_change_rebuilds_cpu_prefix(bpe_tokenizer, inputs):
    tok = bpe_tokenizer
    session = InferenceSession()
    session.prepare(*inputs)
    for kwargs in ({'enable_thinking': False}, {'enable_thinking': True}, {'enable_thinking': False}):
        tok.lifemate_template_kwargs = kwargs
        user = session.prompt(observation_views()[3])
        assert session.render(tok, SYSTEM_PROMPT, user)[:2] == reference(tok, str(user))
    assert session.report()['full_renders'] == 3
    tok.backend_tokenizer.model.dropout = .1
    assert not _safe_tokenizer(tok)


def test_real_schemas_override_tools_inherited_from_template_kwargs(bpe_tokenizer, inputs):
    bpe_tokenizer.lifemate_template_kwargs['tools'] = [{'name': 'stale.disallowed'}]
    session = InferenceSession()
    session.prepare(*inputs)
    user = session.prompt(observation_views()[3])
    assert session.render(bpe_tokenizer, SYSTEM_PROMPT, user)[:2] == reference(bpe_tokenizer, str(user))


def test_backend_mutation_rechecks_the_boundary_instead_of_using_stale_tokens(bpe_tokenizer, inputs):
    from tokenizers import pre_tokenizers
    tok = bpe_tokenizer
    session = InferenceSession()
    session.prepare(*inputs)
    user = session.prompt(observation_views()[3])
    assert session.render(tok, SYSTEM_PROMPT, user)[2]
    tok.backend_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    actual = session.render(tok, SYSTEM_PROMPT, user)
    assert actual[:2] == reference(tok, str(user))
    assert actual[2] == [] and session.report()['fallback_reason']


def test_observation_metadata_matches_portable_json_round_trip(bpe_tokenizer, inputs):
    from datetime import datetime
    from decimal import Decimal
    session = InferenceSession()
    session.prepare(*inputs)
    observations = [{'name': 'home.write', 'arguments': {
        'amount': Decimal('123.45'), 'time': datetime(2026, 11, 5), 'values': (1, 2)},
        'result': {'ok': True, 'value': Decimal('12.34')}}]
    user = session.prompt(observations)
    assert session.render(bpe_tokenizer, SYSTEM_PROMPT, user)[:2] == reference(bpe_tokenizer, str(user))


@pytest.mark.parametrize('gpu_cache', ['off', 'auto'])
def test_real_generation_and_full_input_ids_match_with_cpu_and_gpu_caches(bpe_tokenizer, inputs, gpu_cache, monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tok = bpe_tokenizer
        model = tiny_model(conditional=True, vocab_size=len(tok))
        baseline = LocalLLM.from_loaded_model(model, tok, max_input_tokens=16000, max_new_tokens=3,
                                            prefix_cache=gpu_cache)
        cached = LocalLLM.from_loaded_model(model, tok, max_input_tokens=16000, max_new_tokens=3,
                                          prefix_cache=gpu_cache, prompt_cache='auto')
        session = InferenceSession()
        session.prepare(*inputs)
        original = model.generate
        passed_ids = []
        def generate(**kwargs):
            passed_ids.append(kwargs['input_ids'][0].tolist())
            assert kwargs['attention_mask'].shape == kwargs['input_ids'].shape
            return original(**kwargs)
        monkeypatch.setattr(model, 'generate', generate)
        for observations in observation_views()[:4]:
            user = session.prompt(observations)
            expected = baseline.generate(SYSTEM_PROMPT, str(user))
            assert cached.generate(SYSTEM_PROMPT, user) == expected
            assert passed_ids[-1] == passed_ids[-2]
            if gpu_cache == 'auto':
                assert cached.prefix_cache.tokens == baseline.prefix_cache.tokens
                assert_cache_equal(cached.prefix_cache.snapshot, baseline.prefix_cache.snapshot)
        assert session.report()['suffix_renders'] == 3
        if gpu_cache == 'auto':
            assert cached.prefix_cache_stats()['hits'] == 3
            assert cached.prefix_cache_stats()['fallbacks'] == 0
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('last_action', ['done', 'ask_missing_info', 'failure'])
def test_agent_session_matches_reference_through_helpers_and_guards_and_closes(tmp_path, bpe_tokenizer, last_action):
    outputs = [
        {'action': 'call', 'name': 'local_text.normalize', 'arguments': {'kind': 'phone', 'text': '0901234567'}},
        {'action': 'call', 'name': 'smart_home.light_set', 'arguments': {'power': 'on'}},
        {'action': 'call', 'name': 'smart_home.get_status', 'arguments': {}},
        {'action': 'call', 'name': 'smart_home.get_status', 'arguments': {}},
    ]
    if last_action == 'ask_missing_info':
        outputs.append({'action': 'call', 'name': 'sys.ask_missing_info', 'arguments': {
            'tool_name': 'smart_home.light_set', 'params': ['device_id']}})
    else:
        outputs.append({'action': 'done'})
    class LLM:
        prompt_cache_mode = 'auto'
        def __init__(self): self.outputs = iter(outputs); self.sessions = []; self.prompts = []
        def generate(self, system, user):
            assert isinstance(user, PreparedPrompt)
            self.sessions.append(user.session)
            self.prompts.append(str(user))
            rendered = user.session.render(bpe_tokenizer, system, user)
            assert rendered[:2] == reference(bpe_tokenizer, str(user))
            if last_action == 'failure' and len(self.prompts) == len(outputs):
                raise RuntimeError('simulated generation error')
            return json.dumps(next(self.outputs))
    llm = LLM()
    env = FakeEnv()
    prepared = {'id': item()['id'], 'transcript': 'Bật đèn; số điện thoại 0901234567', 'history': []}
    agent = LifeMateAgent(None, llm, CATALOG, AgentConfig())
    if last_action == 'failure':
        with pytest.raises(RuntimeError, match='simulated generation error'):
            agent.run(item(), env, tmp_path, 'public_test', prepared)
    else:
        report = agent.run(item(), env, tmp_path, 'public_test', prepared)
        assert report['stop_reason'] == last_action
        assert report['prompt_cache']['fixed_builds'] == 1
        assert report['prompt_cache']['suffix_renders'] == len(outputs) - 1
    assert len({id(s) for s in llm.sessions}) == 1
    assert all(s._fixed_json is None and s._prefix_ids is None for s in llm.sessions)
    assert env.calls[0]['name'] == 'smart_home.get_status'
    assert len(env.calls) == (2 if last_action == 'ask_missing_info' else 1)


def test_non_native_tokenizer_and_closed_prompt_keep_the_reference_path(bpe_tokenizer, inputs):
    tok = bpe_tokenizer
    session = InferenceSession()
    session.prepare(*inputs)
    user = session.prompt([])
    tok.lifemate_native_fc = False
    assert session.render(tok, SYSTEM_PROMPT, user) is None
    tok.lifemate_native_fc = True
    session.close()
    model = tiny_model(conditional=True, vocab_size=len(tok))
    llm = LocalLLM.from_loaded_model(model, tok, max_input_tokens=16000, max_new_tokens=1, prompt_cache='auto')
    assert llm.generate(SYSTEM_PROMPT, user) == llm.generate(SYSTEM_PROMPT, str(user))


def test_cached_prefix_does_not_hide_observation_tokens_from_context_limit(bpe_tokenizer, inputs, monkeypatch):
    session = InferenceSession()
    session.prepare(*inputs)
    first = session.prompt([])
    _, ids, _ = session.render(bpe_tokenizer, SYSTEM_PROMPT, first)
    model = tiny_model(conditional=True, vocab_size=len(bpe_tokenizer))
    llm = LocalLLM.from_loaded_model(model, bpe_tokenizer, max_input_tokens=len(ids)+1,
                                    max_new_tokens=1, prefix_cache='auto', prompt_cache='auto')
    def must_not_generate(**kwargs):
        raise AssertionError('oversized prompt reached model generation')
    monkeypatch.setattr(model, 'generate', must_not_generate)
    long = session.prompt(observation_views()[3])
    with pytest.raises(ValueError, match='exceeds --max-input-tokens'):
        llm.generate(SYSTEM_PROMPT, long)
