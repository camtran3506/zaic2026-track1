"""Real hybrid-state/logit parity; no downloads, GPU or organiser state needed."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from lifemate.agent import SYSTEM_PROMPT, AgentConfig, LifeMateAgent
from lifemate.llm import render_prompt
from lifemate.models import LocalLLM
from lifemate.prefix_cache import PrefixCache, stable_prefix_ids


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(*, conditional=False, vocab_size=32, value_heads=2):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5ForCausalLM
    torch.manual_seed(7)
    text = dict(vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        layer_types=['linear_attention', 'full_attention', 'linear_attention'],
        linear_num_key_heads=1, linear_num_value_heads=value_heads,
        linear_key_head_dim=8, linear_value_head_dim=8,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1., 'mrope_section': [1, 1, 2]},
        max_position_embeddings=8192, eos_token_id=1, pad_token_id=1)
    if conditional:
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text,
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                out_hidden_size=16, patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
                num_position_embeddings=16),
            image_token_id=vocab_size+1, video_token_id=vocab_size+2,
            vision_start_token_id=vocab_size+3, vision_end_token_id=vocab_size+4))
    else:
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**text))
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update
    for m in model.modules():
        if not any(p.is_cuda for p in m.parameters()):
            if hasattr(m, 'causal_conv1d_fn'):
                m.causal_conv1d_fn = None
            if hasattr(m, 'causal_conv1d_update'):
                m.causal_conv1d_update = torch_causal_conv1d_update
    return model.eval()


def assert_cache_equal(left, right):
    for name in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states'):
        for a, b in zip(getattr(left, name), getattr(right, name)):
            if a is None:
                assert b is None
            else:
                assert torch.equal(a, b), name


@pytest.mark.parametrize('prefix_length,suffix_length', [(1, 1), (3, 2), (13, 11), (65, 67)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('conditional,value_heads', [(False, 1), (True, 2)])
def test_hybrid_continuation_logits_and_immutable_snapshot(prefix_length, suffix_length, dtype, conditional, value_heads):
    model = tiny_model(conditional=conditional, value_heads=value_heads).to(dtype=dtype)
    ids = torch.randint(2, 32, (1, prefix_length + suffix_length))
    cache = PrefixCache()
    assert cache.eligible(model)
    original = [m.forward for m in cache._layers]
    with torch.inference_mode():
        expected = model(input_ids=ids).logits[:, prefix_length:]
        for attempt in range(2):
            with cache.restore(model, ids, ids[0, :prefix_length].tolist()) as kwargs:
                saved = deepcopy(cache.snapshot)
                assert kwargs['past_key_values'] is not cache.snapshot
                actual = model(input_ids=ids[:, prefix_length:], attention_mask=torch.ones_like(ids),
                               past_key_values=kwargs['past_key_values'],
                               position_ids=kwargs['position_ids'][..., prefix_length:]).logits
                tolerance = 2e-3 if dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                assert_cache_equal(cache.snapshot, saved)
            assert [m.forward for m in cache._layers] == original
    assert cache.stats['hits'] == 1
    assert cache.stats['reused_tokens'] == prefix_length


def test_snapshot_replaced_on_prefix_change_and_shim_restored_on_error():
    model = tiny_model()
    cache = PrefixCache()
    assert cache.eligible(model)
    ids = torch.randint(2, 32, (1, 20))
    before = [m.forward for m in cache._layers]
    with torch.inference_mode():
        with cache.restore(model, ids, ids[0, :10].tolist()):
            snapshot = cache.snapshot
        ids[0, 0] = (ids[0, 0] + 1) % 32
        with pytest.raises(RuntimeError, match='simulated failure'):
            with cache.restore(model, ids, ids[0, :10].tolist()):
                assert cache.snapshot is not snapshot
                raise RuntimeError('simulated failure')
    assert [m.forward for m in cache._layers] == before
    assert cache.stats['misses'] == 2 and cache.stats['hits'] == 0
    cache.begin_sample()
    assert cache.snapshot is None and cache.stats['misses'] == 0


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {'[UNK]': 0, '[EOS]': 1}
    vocab.update({s: i + 2 for i, s in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]')
    tok.chat_template = (Path(__file__).parent / 'fixtures/qwen35_chat_template.jinja').read_text()
    tok.lifemate_native_fc = True
    tok.lifemate_template_kwargs = {'enable_thinking': False}
    return tok


def request(observations=(), *, transcript='Bật đèn', history=(), context=None):
    payload = {'available_tools': {'tool_schemas': [{'name': 'home.status', 'parameters': {
        'type': 'object', 'properties': {}}}]},
        'request_transcript': transcript, 'conversation_history': list(history),
        'user_context': context or {}, 'previous_actions_and_results': list(observations)}
    user = json.dumps(payload, ensure_ascii=False)
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': user}]


def test_exact_native_prefix_including_history_survives_helper_and_rejection_changes():
    tok = tokenizer()
    history = [{'role': 'assistant', 'text': 'Lần trước', 'tool_calls': [
        {'name': 'home.status', 'arguments': {}}], 'tool_results': [{'ok': True}]}]
    base = request(history=history)
    _, ids = render_prompt(tok, base)
    prefix = stable_prefix_ids(tok, base, ids)
    assert prefix and len(prefix) < len(ids)
    for observations in [
        [{'error': 'invalid arguments'}],
        [{'name': 'local_time.calendar_reference', 'arguments': {}, 'result': '{"ok":true}'}],
        [{'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"devices":[]}'}],
        [],  # Rejected feedback can disappear; do not reuse its recurrent state.
    ]:
        changed = request(observations, history=history)
        _, changed_ids = render_prompt(tok, changed)
        assert stable_prefix_ids(tok, changed, changed_ids) == prefix
    for changed in [request(transcript='Tắt đèn', history=history),
                    request(history=history, context={'timezone': 'UTC'}), request()]:
        _, changed_ids = render_prompt(tok, changed)
        assert stable_prefix_ids(tok, changed, changed_ids) != prefix
    assert stable_prefix_ids(tok, base, [999] + ids) == []


@pytest.mark.parametrize('peft', [False, True])
def test_real_generate_matches_full_prefill_across_agent_turns(peft):
    tok = tokenizer()
    model = tiny_model(conditional=True, vocab_size=len(tok))
    if peft:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=2, task_type='CAUSAL_LM', target_modules=['in_proj_qkv']))
        # Exercise nonzero adapter weights, not just an initially neutral LoRA.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if 'lora_B' in name:
                    parameter.normal_(std=.05)
        model.eval()
    cached = LocalLLM.from_loaded_model(model, tok, max_input_tokens=8192, max_new_tokens=4, prefix_cache='auto')
    baseline = LocalLLM.from_loaded_model(model, tok, max_input_tokens=8192, max_new_tokens=4)
    seen_lengths = []
    handle = model.get_input_embeddings().register_forward_pre_hook(lambda module, args: seen_lengths.append(args[0].shape[1]))
    try:
        for observations in [[], [{'error': 'schema invalid'}],
                [{'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"data":[1]}'}], []]:
            messages = request(observations)
            seen_lengths.clear()
            expected = baseline.generate(SYSTEM_PROMPT, messages[1]['content'])
            full_length = seen_lengths[0]
            seen_lengths.clear()
            assert cached.generate(SYSTEM_PROMPT, messages[1]['content']) == expected
            assert seen_lengths[0] < full_length
        assert cached.prefix_cache_stats()['hits'] == 3
        assert cached.prefix_cache_stats()['fallbacks'] == 0
    finally:
        handle.remove()


def test_cache_failure_retries_original_path_and_restores_forwards(monkeypatch):
    tok = tokenizer()
    model = tiny_model(vocab_size=len(tok))
    llm = LocalLLM.from_loaded_model(model, tok, max_input_tokens=8192, max_new_tokens=3, prefix_cache='auto')
    baseline = LocalLLM.from_loaded_model(model, tok, max_input_tokens=8192, max_new_tokens=3)
    messages = request()
    expected = baseline.generate(SYSTEM_PROMPT, messages[1]['content'])
    original_generate = model.generate
    attempts = []
    def generate(**kwargs):
        attempts.append('past_key_values' in kwargs)
        guard = kwargs['logits_processor'][0]
        if 'past_key_values' in kwargs:
            # A cached attempt can fail after starting an action. Its output
            # framing state must not leak into the fresh full-prompt retry.
            begun = torch.cat([kwargs['input_ids'], torch.tensor([[tok.get_vocab()['{']]])], dim=1)
            guard(begun, torch.zeros(1, len(tok)))
            raise RuntimeError('simulated incompatible cached generate')
        scores = guard(kwargs['input_ids'], torch.zeros(1, len(tok)))
        assert torch.isneginf(scores[0, tok.get_vocab()['R']])
        return original_generate(**kwargs)
    monkeypatch.setattr(model, 'generate', generate)
    assert llm.generate(SYSTEM_PROMPT, messages[1]['content']) == expected
    assert attempts == [True, False]
    assert llm.prefix_cache.snapshot is None
    assert llm.prefix_cache_stats()['fallbacks'] == 1
    assert all('forward' not in vars(layer) for layer in llm.prefix_cache._layers)


def test_cache_gates_other_versions_and_patched_training_runtime(monkeypatch):
    import transformers
    model = tiny_model()
    cache = PrefixCache()
    monkeypatch.setattr(transformers, '__version__', '5.17.0')
    assert not cache.eligible(model)
    assert cache.disabled_reason
    monkeypatch.setattr(transformers, '__version__', '5.4.0')
    cache = PrefixCache()
    layer = model.model.layers[0].linear_attn
    monkeypatch.setattr(layer, 'forward', lambda **kwargs: None)
    assert not cache.eligible(model)
    model.train()
    assert not PrefixCache().eligible(model)


@pytest.mark.parametrize('raises', [False, True])
def test_agent_clears_snapshot_at_sample_end_even_after_failure(tmp_path, raises):
    from test_baseline import CATALOG, FakeEnv, item
    class LLM:
        def __init__(self): self.events = []
        def begin_sample(self): self.events.append('begin')
        def end_sample(self): self.events.append('end')
        def prefix_cache_stats(self): return {'hits': 2}
        def generate(self, *args):
            if raises:
                raise RuntimeError('sample failure')
            return '{"action":"done"}'
    llm = LLM()
    agent = LifeMateAgent(None, llm, CATALOG, AgentConfig())
    prepared = {'id': item()['id'], 'transcript': 'Bật đèn', 'history': []}
    if raises:
        with pytest.raises(RuntimeError, match='sample failure'):
            agent.run(item(), FakeEnv(), tmp_path, 'public_test', prepared)
    else:
        assert agent.run(item(), FakeEnv(), tmp_path, 'public_test', prepared)['prefix_cache'] == {'hits': 2}
    assert llm.events == ['begin', 'end']


def test_incremental_multiturn_prefix_cache_chaining():
    tok = tokenizer()
    model = tiny_model(conditional=True, vocab_size=len(tok))
    cached = LocalLLM.from_loaded_model(model, tok, max_input_tokens=8192, max_new_tokens=4, prefix_cache='auto')
    baseline = LocalLLM.from_loaded_model(model, tok, max_input_tokens=8192, max_new_tokens=4)

    obs_turn1 = []
    obs_turn2 = [{'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"data":[1]}'}]
    obs_turn3 = [
        {'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"data":[1]}'},
        {'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"data":[2]}'},
    ]
    obs_turn4 = [
        {'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"data":[1]}'},
        {'name': 'home.status', 'arguments': {}, 'result': '{"ok":true,"data":[2]}'},
        {'error': 'replan tool call'},
    ]

    for obs in (obs_turn1, obs_turn2, obs_turn3, obs_turn4):
        messages = request(obs)
        expected = baseline.generate(SYSTEM_PROMPT, messages[1]['content'])
        actual = cached.generate(SYSTEM_PROMPT, messages[1]['content'])
        assert actual == expected

    stats = cached.prefix_cache_stats()
    assert stats['hits'] == 3
    assert stats['chain_advanced_tokens'] > 0
    assert stats['chain_hits'] >= 1
    assert stats['fallbacks'] == 0
