"""Cache isolation across rollout views, optimizer updates and training failures."""
from copy import deepcopy
from functools import wraps
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import TrainerControl, TrainerState

from lifemate.agent import SYSTEM_PROMPT
from lifemate.evaluation import score_callback
from lifemate.models import LocalLLM
from lifemate.prefix_cache import PrefixCache
from lifemate.prompt_session import InferenceSession
from test_prefix_cache import tiny_model
from test_prompt_session import bpe_tokenizer


@pytest.mark.parametrize('failure', [None, 'asr', 'clean', 'app_asr', 'app_clean', 'training_switch'])
@pytest.mark.parametrize('mode', ['auto', 'off'])
def test_callback_cache_modes_cleanup_and_all_training_state_restored(tmp_path, bpe_tokenizer, failure, mode):
    tok = bpe_tokenizer
    model = torch.nn.Embedding(len(tok), 2)
    model.get_input_embeddings = lambda: model
    model.config = SimpleNamespace(use_cache=False, text_config=SimpleNamespace(use_cache=False))
    seen, switches = [], []
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def consume_rng():
        random.random(); np.random.rand(); torch.rand(1)
    def evaluate(llm, path, *, transcript_source='asr', app=False):
        assert not model.training and model.config.use_cache and model.config.text_config.use_cache
        assert llm.prefix_cache.mode == mode and llm.prompt_cache_mode == mode
        assert llm.prefix_cache.snapshot is None
        seen.append(llm)
        llm.prefix_cache.snapshot = object()  # Even a custom evaluator retaining a cache must be cleaned up.
        tok.padding_side = tok.truncation_side = 'left'
        consume_rng()
        view = ('app_' if app else '') + ('asr' if transcript_source == 'asr' else 'clean')
        if failure == view:
            raise RuntimeError('evaluation failed')
        prefix = 'eval_app_heldout_' if app else 'eval_'
        if transcript_source == 'audio_text':
            prefix += 'clean_'
        return {prefix+'state_tool_score': 98 if app else 75}
    def for_inference(model):
        switches.append('inference'); consume_rng()
    def for_training(model):
        assert all(llm.prefix_cache.snapshot is None for llm in seen)
        switches.append('training'); consume_rng()
        model.config.text_config.use_cache = True
        if failure == 'training_switch':
            raise RuntimeError('switch failed')
    fast = SimpleNamespace(for_inference=for_inference, for_training=for_training)
    callback = score_callback(SimpleNamespace(evaluate=evaluate), tok, tmp_path, fast,
        eval_clean_transcripts=True, eval_prefix_cache=mode, eval_prompt_cache=mode,
        app_evaluator=SimpleNamespace(evaluate=lambda *a, **kw: evaluate(*a, **kw, app=True)))
    args = (None, TrainerState(global_step=1), TrainerControl())
    if failure:
        with pytest.raises(RuntimeError):
            callback.on_evaluate(*args, model=model, metrics={})
    else:
        metrics = {}
        callback.on_evaluate(*args, model=model, metrics=metrics)
        assert metrics['eval_state_tool_score'] == metrics['eval_clean_state_tool_score'] == 75
        assert metrics['eval_app_heldout_state_tool_score'] == metrics['eval_app_heldout_clean_state_tool_score'] == 98
        assert len(seen) == 4
        logged = json.loads((tmp_path/'validation_scores.jsonl').read_text())
        assert logged['app_heldout_clean_rollouts'].endswith('.app-heldout.clean.jsonl')
    assert model.training and not model.config.use_cache and not model.config.text_config.use_cache
    assert tok.padding_side == tok.truncation_side == 'right'
    assert switches == ['inference', 'training']
    assert all(llm.prefix_cache.snapshot is None for llm in seen)
    assert random.getstate() == py_rng
    restored_np = np.random.get_state()
    assert restored_np[0] == np_rng[0] and np.array_equal(restored_np[1], np_rng[1]) and restored_np[2:] == np_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_fresh_lora_weights_and_gradients_after_each_cached_rollout(tmp_path, bpe_tokenizer):
    from peft import LoraConfig, get_peft_model
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tok = bpe_tokenizer
        model = get_peft_model(tiny_model(conditional=True, vocab_size=len(tok)),
            LoraConfig(r=2, task_type='CAUSAL_LM', target_modules=['in_proj_qkv', 'k_proj']))
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora_B' in name:
                    param.normal_(std=.03)
        model.train()
        model.config.use_cache = model.config.text_config.use_cache = False
        wrappers, snapshots = [], []
        def evaluate(llm, path, *, transcript_source='asr'):
            assert llm.prefix_cache.snapshot is None
            assert llm.prefix_cache.mode == llm.prompt_cache_mode == 'auto'
            llm.max_new_tokens = 1  # Bound the tiny random model's rollout in this test.
            reference = LocalLLM.from_loaded_model(model, tok, max_new_tokens=1)
            session = InferenceSession()
            session.prepare('Bật đèn', {}, [], {})
            try:
                for observations in ([], [{'error': 'replan'}]):
                    prompt = session.prompt(observations)
                    assert llm.generate(SYSTEM_PROMPT, prompt) == reference.generate(SYSTEM_PROMPT, str(prompt))
                assert llm.prefix_cache_stats()['hits'] == 1
                assert llm.prefix_cache_stats()['disabled_reason'] is None
                snapshots.append(deepcopy(llm.prefix_cache.snapshot))
                wrappers.append(llm)
            finally:
                session.close()
            return {'eval_state_tool_score': 75}
        callback = score_callback(SimpleNamespace(evaluate=evaluate), tok, tmp_path)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.2)
        for step in (1, 2):
            callback.on_evaluate(None, TrainerState(global_step=step), TrainerControl(), model=model, metrics={})
            assert model.training and not model.config.use_cache and not model.config.text_config.use_cache
            assert wrappers[-1].prefix_cache.snapshot is None
            inputs = torch.tensor([[5, 6, 7, 8]])
            loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
            loss.backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
            optimizer.step(); optimizer.zero_grad()
        assert wrappers[0] is not wrappers[1]
        assert any(not torch.equal(a, b) for a, b in zip(snapshots[0].key_cache, snapshots[1].key_cache)
                   if a is not None and b is not None)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('component', ['linear', 'text', 'decoder'])
@pytest.mark.parametrize('global_patch', [False, True])
def test_patched_qwen_forward_even_with_wraps_disables_gpu_cache(monkeypatch, component, global_patch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    cls = {'linear': q.Qwen3_5GatedDeltaNet, 'text': q.Qwen3_5TextModel, 'decoder': q.Qwen3_5DecoderLayer}[component]
    original = cls.forward
    @wraps(original)
    def patch(self, *args, **kwargs):
        return original(self, *args, **kwargs)
    model = tiny_model()
    if global_patch:
        monkeypatch.setattr(cls, 'forward', patch)
    else:
        module = next(m for m in model.modules() if type(m) is cls)
        monkeypatch.setattr(module, 'forward', lambda *args, **kwargs: original(module, *args, **kwargs))
    cache = PrefixCache()
    assert not cache.eligible(model)
    assert 'runtime' in cache.disabled_reason
