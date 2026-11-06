"""Cross-sample isolation, bounded admission and real Qwen hybrid-state parity."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from lifemate.agent import SYSTEM_PROMPT
from lifemate.llm import render_prompt
from lifemate.models import LocalLLM
from lifemate.prefix_cache import PrefixCache, stable_prefix_ids
from lifemate.schema_cache import SchemaCache, snapshot_bytes
from test_prefix_cache import tiny_model, assert_cache_equal, request
from test_prompt_session import bpe_tokenizer


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def decoded_prefix(tokenizer, ids):
    # The fixture deliberately has no decoder; reconstruct ByteLevel pieces for
    # content/isolation assertions rather than inspecting a space-joined decode.
    from tokenizers.decoders import ByteLevel
    return ByteLevel().decode(tokenizer.convert_ids_to_tokens(list(ids)))


@pytest.mark.parametrize('mode', ['off', 'auto', 'shared'])
def test_admission_retains_no_unique_gpu_snapshots_in_auto(mode):
    cache = SchemaCache(mode)
    built = []
    def build():
        built.append(1)
        return SimpleNamespace(state=torch.zeros(2))
    first = cache.snapshot_for([1, 2], build)
    assert len(cache.entries) == (1 if mode == 'shared' else 0)
    cache.snapshot_for([1, 2], build)
    assert len(built) == (1 if mode == 'shared' else 0)  # Same sample is not a repeated-schema workload.
    cache.begin_sample()
    second = cache.snapshot_for([1, 2], build)
    assert len(built) == (0 if mode == 'off' else 1)
    cache.begin_sample()
    third = cache.snapshot_for([1, 2], build)
    if mode != 'off':
        assert third is second
        assert cache.report()['hits'] == 1 and cache.report()['reused_tokens'] == 2
    cache.clear()
    assert not cache.entries and not cache.seen and not cache.rendered


def test_unique_catalogs_use_bounded_cpu_admission_metadata_only():
    cache = SchemaCache('auto')
    for sample in range(400):
        cache.begin_sample()
        assert cache.snapshot_for([sample], lambda: pytest.fail('Unique schemas must not be admitted')) is None
    assert not cache.entries and len(cache.seen) == 256


def test_pool_uses_lru_and_accounts_for_all_hybrid_storages():
    tensor = torch.zeros(4)
    snapshot = SimpleNamespace(key_cache=[tensor], value_cache=[tensor[:2]],
                               conv_states=[torch.zeros(2)], recurrent_states=[torch.zeros(3)])
    assert snapshot_bytes(snapshot) == 36
    cache = SchemaCache('shared', max_mb=80/1024**2, max_entries=2)
    for tokens in ([1], [2], [1], [3]):
        cache.snapshot_for(tokens, lambda: deepcopy(snapshot))
    assert [entry[0] for entry in cache.entries.values()] == [(1,), (3,)]
    assert cache.report()['bytes'] == 72 and cache.report()['evictions'] == 1
    tiny = SchemaCache('shared', max_mb=1/1024**2)
    assert tiny.snapshot_for([1], lambda: snapshot) is snapshot
    assert not tiny.entries
    assert tiny.snapshot_for([1], lambda: pytest.fail('Do not rebuild an oversized entry')) is None


@pytest.mark.parametrize('mode,mb', [('invalid', 1), ('auto', 0), ('auto', -1), ('auto', float('nan')), ('auto', float('inf'))])
def test_invalid_modes_and_budgets_fail_early(mode, mb):
    with pytest.raises(ValueError):
        SchemaCache(mode, mb)


def test_prefix_material_has_no_sample_state_and_uses_exact_token_check(bpe_tokenizer):
    tok = bpe_tokenizer
    cache = SchemaCache('shared')
    messages = request(transcript='PRIVATE_REQUEST_123', history=[{'role': 'user', 'text': 'PRIVATE_HISTORY_456'}],
                       context={'secret': 'PRIVATE_CONTEXT_789'})
    _, ids = render_prompt(tok, messages)
    prefix = cache.prefix_ids(tok, messages, ids)
    assert prefix and tuple(ids[:len(prefix)]) == prefix
    decoded = decoded_prefix(tok, prefix)
    assert 'home.status' in decoded and 'PRIVATE_' not in decoded and 'schema boundary' not in decoded
    assert decoded.endswith('<|im_end|>')
    original_source = cache.prefix_source
    other = request(transcript='different request', history=[], context={})
    _, other_ids = render_prompt(tok, other)
    assert cache.prefix_ids(tok, other, other_ids) == prefix
    assert cache.prefix_source == original_source
    assert cache.prefix_ids(tok, other, [999]+other_ids) == ()
    changed = json.loads(other[1]['content'])
    changed['available_tools']['tool_schemas'][0]['description'] = 'A different tool contract.'
    other[1]['content'] = json.dumps(changed)
    _, changed_ids = render_prompt(tok, other)
    assert cache.prefix_ids(tok, other, changed_ids) != prefix
    assert cache.prefix_source != original_source


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('conditional', [False, True])
def test_two_prefill_boundaries_preserve_hybrid_logits_and_immutable_shared_snapshot(dtype, conditional):
    model = tiny_model(conditional=conditional).to(dtype=dtype)
    pool = SchemaCache('shared')
    cache = PrefixCache()
    assert cache.eligible(model)
    base = torch.randint(2, 32, (1, 40))
    with torch.inference_mode():
        for sample in range(3):
            ids = base.clone()
            ids[:, 10:] = torch.randint(2, 32, (1, 30))
            expected = model(input_ids=ids).logits[:, 25:]
            cache.begin_sample()
            pool.begin_sample()
            with cache.restore(model, ids, ids[0, :25].tolist(), schema_cache=pool, schema_prefix=ids[0, :10].tolist()) as kwargs:
                shared = next(iter(pool.entries.values()))[1]
                saved = deepcopy(shared)
                actual = model(input_ids=ids[:, 25:], attention_mask=torch.ones_like(ids),
                               past_key_values=kwargs['past_key_values'],
                               position_ids=kwargs['position_ids'][..., 25:]).logits
                tolerance = 3e-3 if dtype == torch.bfloat16 else 3e-6
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                assert shared.get_seq_length() == 10
                assert_cache_equal(shared, saved)
            cache.clear()
    assert pool.total_hits == 2 and pool.total_builds == 1


@pytest.mark.parametrize('mode', ['auto', 'shared'])
@pytest.mark.parametrize('prompt_cache', ['auto', 'off'])
def test_real_multi_sample_generate_matches_uncached_and_keeps_only_schema_between_samples(bpe_tokenizer, mode, prompt_cache):
    from lifemate.prompt_session import InferenceSession
    from lifemate.agent import build_step_prompt
    tok = bpe_tokenizer
    model = tiny_model(conditional=True, vocab_size=len(tok))
    cached = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1,
                                       prefix_cache='auto', prompt_cache=prompt_cache, schema_cache=mode)
    baseline = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1)
    schema = {'home.status': {'name': 'home.status', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}}}
    lengths = []
    hook = model.get_input_embeddings().register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    try:
        for i in range(4):
            cached.begin_sample()
            session = InferenceSession()
            transcript, context, history = f'PRIVATE_REQUEST_{i}', {'private': f'PRIVATE_CONTEXT_{i}'}, []
            session.prepare(transcript, context, history, schema)
            try:
                for observations in ([], [{'error': 'repair'}], []):
                    prompt = session.prompt(observations)
                    assert str(prompt) == build_step_prompt(transcript, context, history, schema, observations)[1]
                    lengths.clear()
                    actual = cached.generate(SYSTEM_PROMPT, prompt)
                    if not observations and cached.prefix_cache_stats()['misses'] == 1 and cached.prefix_cache_stats()['hits'] == 0 and i >= 2:
                        # A warm schema pool must reduce actual model input work,
                        # not merely report a hit while recomputing the prefix.
                        assert lengths[0] == (cached.prefix_cache_stats()['snapshot_tokens']
                                              - cached.schema_cache_stats()['prefix_tokens'])
                    assert actual == baseline.generate(SYSTEM_PROMPT, str(prompt))
                assert cached.prefix_cache_stats()['hits'] == 2
                if i >= 2:
                    assert cached.schema_cache_stats()['hits'] == 1
                if mode == 'auto' and i == 0:
                    assert not cached.schema_cache.entries
                assert cached.prefix_cache_stats()['disabled_reason'] is None
            finally:
                session.close()
                cached.end_sample()
            assert cached.prefix_cache.snapshot is None
            for tokens, snapshot, size in cached.schema_cache.entries.values():
                assert 'PRIVATE_' not in decoded_prefix(tok, tokens)
                assert snapshot.get_seq_length() == len(tokens)
        assert cached.schema_cache.total_builds == 1
    finally:
        hook.remove()
        cached.close()
    assert not cached.schema_cache.entries


def test_weight_updates_and_adapter_switches_invalidate_all_snapshots(bpe_tokenizer):
    from peft import LoraConfig, get_peft_model
    tok = bpe_tokenizer
    config = LoraConfig(r=2, task_type='CAUSAL_LM', target_modules=['in_proj_qkv'])
    model = get_peft_model(tiny_model(conditional=True, vocab_size=len(tok)), config)
    model.add_adapter('second', config)
    model.eval()
    cached = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1, prefix_cache='auto', schema_cache='shared')
    baseline = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1)
    user = request()[1]['content']
    snapshots = []
    for change in ('initial', 'weights', 'adapter', 'disabled_adapter'):
        if change == 'weights':
            with torch.no_grad():
                next(model.parameters()).add_(.01)
        elif change == 'adapter':
            model.set_adapter('second')
        context = model.disable_adapter() if change == 'disabled_adapter' else __import__('contextlib').nullcontext()
        with context:
            assert cached.generate(SYSTEM_PROMPT, user) == baseline.generate(SYSTEM_PROMPT, user)
            assert len(cached.schema_cache.entries) == 1
            snapshots.append(next(iter(cached.schema_cache.entries.values()))[1])
    assert len({id(s) for s in snapshots}) == 4
    assert cached.schema_cache.invalidations == 3
    cached.close()


def test_failure_discards_shared_snapshots_and_retries_full_prompt(bpe_tokenizer, monkeypatch):
    tok = bpe_tokenizer
    model = tiny_model(vocab_size=len(tok))
    cached = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1, prefix_cache='auto', schema_cache='shared')
    baseline = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1)
    user = request()[1]['content']
    expected = baseline.generate(SYSTEM_PROMPT, user)
    original = model.generate
    attempts = []
    def generate(**kwargs):
        attempts.append('past_key_values' in kwargs)
        if 'past_key_values' in kwargs:
            raise RuntimeError('simulated cached generation failure')
        return original(**kwargs)
    monkeypatch.setattr(model, 'generate', generate)
    assert cached.generate(SYSTEM_PROMPT, user) == expected
    assert attempts == [True, False]
    assert cached.prefix_cache.snapshot is None and not cached.schema_cache.entries
    assert all('forward' not in vars(layer) for layer in cached.prefix_cache._layers)


def test_prefix_cache_off_disables_gpu_schema_pool(bpe_tokenizer):
    model = tiny_model(vocab_size=len(bpe_tokenizer))
    cached = LocalLLM.from_loaded_model(model, bpe_tokenizer, max_new_tokens=1, schema_cache='shared')
    cached.generate(SYSTEM_PROMPT, request()[1]['content'])
    assert not cached.schema_cache.entries and cached.schema_cache.total_builds == 0


def test_sequence_length_dependent_rope_disables_snapshot_reuse():
    model = tiny_model()
    model.config.rope_parameters['rope_type'] = 'dynamic'
    cache = PrefixCache()
    assert not cache.eligible(model)
    assert 'RoPE' in cache.disabled_reason
