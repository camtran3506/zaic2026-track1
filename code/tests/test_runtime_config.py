"""Portable policy, artifact identity and BF16 defaults without an engine change."""
import json
import shutil
from types import SimpleNamespace

import pytest

from lifemate.llm import configure_tokenizer, render_prompt
from lifemate.runtime import (DEFAULT_LLM_DTYPE, RUNTIME_MANIFEST, generation_context,
    load_runtime_manifest, make_runtime_manifest, save_runtime_manifest,
    select_tokenizer_path, training_dtype, transformers_generation_config)


@pytest.fixture
def artifact(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text('{"model_type":"qwen3_5"}')
    (base / 'model.safetensors').write_bytes(b'fingerprint fixture, not loaded as a model')
    (base / 'generation_config.json').write_text(json.dumps({
        'do_sample': True, 'temperature': 0.8, 'repetition_penalty': 1.7,
        'eos_token_id': [2, 4], 'forced_bos_token_id': 3}))
    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2,
                                         '[BOS]': 3, '[STOP]': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
        pad_token='[PAD]', eos_token='[EOS]', bos_token='[BOS]', additional_special_tokens=['[STOP]'])
    tok.chat_template = "{% for m in messages %}{{ m['role'] + ':' + m['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}"
    configure_tokenizer(tok, base)
    adapter = tmp_path / 'adapter'
    tok.save_pretrained(adapter)
    manifest = make_runtime_manifest(base, tok)
    save_runtime_manifest(adapter, manifest)
    return base, adapter, tok, manifest


def test_portable_contract_survives_tokenizer_reload_and_model_relocation(artifact, tmp_path):
    from transformers import AutoTokenizer, GenerationConfig

    base, adapter, tok, manifest = artifact
    moved = tmp_path / 'relocated'
    shutil.copytree(base, moved)
    reloaded = configure_tokenizer(AutoTokenizer.from_pretrained(adapter, local_files_only=True), moved)
    actual = load_runtime_manifest(moved, reloaded, adapter)
    assert actual == manifest
    assert actual['inference_dtype'] == DEFAULT_LLM_DTYPE == 'bfloat16'
    assert actual['chat_template_kwargs'] == {'enable_thinking': False}
    messages = [{'role': 'user', 'content': ' BẬT đèn, 08:30! '}]
    assert render_prompt(tok, messages) == render_prompt(reloaded, messages)
    # Portable greedy is temperature=0; HF uses do_sample=False, without sampling-only flags.
    assert actual['generation']['temperature'] == 0
    hf = GenerationConfig.from_pretrained(adapter)
    assert not hf.do_sample and hf.repetition_penalty == 1
    assert hf.eos_token_id == [2, 4] and hf.pad_token_id == 1
    assert hf.forced_bos_token_id is None


@pytest.mark.parametrize('change,match', [
    ('weights', 'checkpoint differs'), ('config', 'checkpoint differs'),
    ('vocabulary', 'Tokenizer'), ('template', 'Tokenizer'), ('backend', 'Tokenizer'),
    ('kwargs', 'Chat template kwargs'),
])
def test_runtime_rejects_wrong_base_or_tokenizer(artifact, change, match):
    base, adapter, tok, _ = artifact
    if change == 'weights':
        (base / 'model.safetensors').write_bytes(b'replaced model weights')
    elif change == 'config':
        (base / 'config.json').write_text('{"model_type":"qwen3_5","vocab_size":999}')
    elif change == 'vocabulary':
        tok.add_tokens(['NEW_TOKEN'])
    elif change == 'template':
        tok.chat_template += 'different prompt'
    elif change == 'backend':
        from tokenizers.processors import TemplateProcessing
        tok.backend_tokenizer.post_processor = TemplateProcessing(single='[BOS] $A', special_tokens=[('[BOS]', 3)])
    else:
        tok.lifemate_template_kwargs['enable_thinking'] = True
    with pytest.raises(ValueError, match=match):
        load_runtime_manifest(base, tok, adapter)


def test_inference_rejects_wrong_model_before_loading_weights(artifact, monkeypatch):
    from lifemate.models import LocalLLM
    import lifemate.models

    base, adapter, _, _ = artifact
    (base / 'model.safetensors').write_bytes(b'wrong base weights')
    monkeypatch.setattr(lifemate.models, 'model_loader',
                        lambda *a: pytest.fail('Must verify identity before loading a model'))
    with pytest.raises(ValueError, match='checkpoint differs'):
        LocalLLM(base, adapter_path=adapter, device='cpu')


def test_missing_adapter_tokenizer_never_falls_back_to_base(artifact):
    base, adapter, _, _ = artifact
    (adapter / 'tokenizer_config.json').unlink()
    with pytest.raises(FileNotFoundError, match='Adapter tokenizer missing'):
        select_tokenizer_path(base, adapter)


def test_legacy_artifact_reports_unverified_identity(artifact, caplog):
    base, adapter, tok, _ = artifact
    (adapter / RUNTIME_MANIFEST).unlink()
    assert load_runtime_manifest(base, tok, adapter) is None
    assert 'identity cannot be verified' in caplog.text


@pytest.mark.parametrize('cuda,bf16,expected', [(True, True, 'BF16'), (False, False, 'FP32')])
def test_training_precision(cuda, bf16, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
        is_bf16_supported=lambda: bf16), bfloat16='BF16', float32='FP32')
    assert training_dtype(torch) == expected


def test_unsupported_gpu_does_not_silently_fall_back_to_fp16():
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
        is_bf16_supported=lambda: False), bfloat16='BF16', float32='FP32')
    with pytest.raises(RuntimeError, match='no automatic FP16 fallback'):
        training_dtype(torch)


def test_generation_does_not_inherit_model_sampling_defaults(artifact):
    import torch
    from transformers import GenerationConfig
    from lifemate.models import LocalLLM

    _, _, tok, _ = artifact

    class Model:
        generation_config = GenerationConfig(do_sample=True, temperature=0.8,
            repetition_penalty=1.7, forced_bos_token_id=3)

        def get_input_embeddings(self):
            return SimpleNamespace(weight=torch.zeros(1))

        def generate(self, input_ids, generation_config, **kwargs):
            assert not generation_config.do_sample
            assert generation_config.repetition_penalty == 1
            assert generation_config.forced_bos_token_id is None
            assert generation_config.eos_token_id == [2]
            assert generation_config.max_new_tokens == 2
            return torch.cat([input_ids, torch.tensor([[tok.eos_token_id]])], dim=1)

    llm = LocalLLM.from_loaded_model(Model(), tok, max_input_tokens=512, max_new_tokens=2)
    llm.generate('JSON only', 'Bật đèn')


def test_saved_policy_validated_before_use(artifact):
    _, adapter, _, manifest = artifact
    manifest['generation']['temperature'] = 0.8
    with pytest.raises(ValueError, match='neutral greedy'):
        transformers_generation_config(manifest['generation'])


def test_hf_does_not_fill_omitted_fields_from_base_model_and_restores_on_error(artifact):
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    _, _, _, manifest = artifact
    poisoned = GenerationConfig(do_sample=True, repetition_penalty=1.8,
        forced_bos_token_id=3, suppress_tokens=[2], no_repeat_ngram_size=3,
        token_healing=True, return_dict_in_generate=True)
    base = SimpleNamespace(generation_config=poisoned)
    wrapper = SimpleNamespace(get_base_model=lambda: base)
    policy = manifest['generation']
    # Actual Transformers preparation inherits otherwise-None fields.
    inherited, _ = GenerationMixin._prepare_generation_config(base, transformers_generation_config(policy))
    assert inherited.forced_bos_token_id == 3
    with pytest.raises(RuntimeError, match='decode failed'):
        with generation_context(wrapper, policy, 384) as config:
            resolved, _ = GenerationMixin._prepare_generation_config(base, config)
            assert not resolved.do_sample
            assert resolved.repetition_penalty == 1
            assert resolved.forced_bos_token_id is None
            assert resolved.suppress_tokens is None
            assert resolved.no_repeat_ngram_size == 0
            assert not resolved.token_healing and not resolved.return_dict_in_generate
            raise RuntimeError('decode failed')
    assert base.generation_config is poisoned
    assert 'generation_config' not in vars(wrapper)


def test_native_protocol_is_persisted_and_legacy_json_adapter_rejected(artifact):
    base, adapter, tok, manifest = artifact
    assert manifest['output_protocol'] == 'qwen35-native-fc-v1'
    manifest['output_protocol'] = 'lifemate-decision-json-v1'
    (adapter / RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='regenerate SFT and retrain'):
        load_runtime_manifest(base, tok, adapter)


@pytest.mark.parametrize('policy', [None, 'old-normalization'])
def test_runtime_rejects_different_text_normalization(artifact, policy):
    from lifemate.text import TEXT_NORMALIZATION
    base, adapter, tok, manifest = artifact
    assert manifest['text_normalization'] == TEXT_NORMALIZATION
    if policy is None:
        manifest.pop('text_normalization')
    else:
        manifest['text_normalization'] = policy
    (adapter / RUNTIME_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Text normalization differs'):
        load_runtime_manifest(base, tok, adapter)
