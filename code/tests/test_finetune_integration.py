"""Actual tiny random Qwen + LoRA training; no downloads or competition data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


def test_lora_accepts_modern_torchao_api(monkeypatch):
    """Modern TorchAO no longer exports LinearActivationQuantizedTensor."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from peft.tuners.lora import torchao as dispatch_module
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    fake_utils = types.ModuleType('torchao.utils')
    fake_utils.TorchAOBaseTensor = type('TorchAOBaseTensor', (), {})
    monkeypatch.setitem(sys.modules, 'torchao.utils', fake_utils)
    monkeypatch.setattr(dispatch_module, 'is_torchao_available', lambda: True)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    model = get_peft_model(model, LoraConfig(r=2, task_type='CAUSAL_LM', target_modules=['q_proj']))
    assert any('lora_B' in name for name, _ in model.named_parameters())


@pytest.mark.parametrize('architecture', ['qwen3', 'qwen3_5'])
def test_train_adapter_reload_and_merge(tmp_path, architecture):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM
    from tokenizers import Tokenizer, models, pre_tokenizers
    from lifemate.models import LocalLLM
    base = tmp_path / 'tiny-base'
    base.mkdir()
    tok = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2,
        '<s>': 3, '</s>': 4, 'system': 5, 'user': 6, 'assistant': 7, '{': 8}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]',
        pad_token='[PAD]', eos_token='[EOS]', additional_special_tokens=['<s>', '</s>'])
    tokenizer.chat_template = "{% for m in messages %}{{ '<s>' + m['role'] + '\n' + m['content'] + '</s>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<s>assistant\n' }}{% endif %}"
    if architecture == 'qwen3_5':
        tokenizer.chat_template = (Path(__file__).parent / 'fixtures/qwen35_chat_template.jinja').read_text()
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.save_pretrained(base)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=4096, eos_token_id=2, pad_token_id=1))
    if architecture == 'qwen3_5':
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=dict(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                layer_types=['linear_attention', 'full_attention'], linear_num_key_heads=1,
                linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8, max_position_embeddings=4096, eos_token_id=2, pad_token_id=1),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                out_hidden_size=16, patch_size=2, spatial_merge_size=1, temporal_patch_size=1,
                num_position_embeddings=16),
            image_token_id=100, video_token_id=101, vision_start_token_id=102, vision_end_token_id=103))
    model.save_pretrained(base)
    data = tmp_path / 'data' / 'train'
    data.mkdir(parents=True)
    tools = [{'name': name, 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}}
             for name in ('home.read', 'home.other_read')]
    source_rows = [{
        'id': f'train_{i}', 'audio': f'train_{i}.wav', 'audio_text': 'Kiểm tra thiết bị',
        'context': {'installed_apps': {'home': {'tools': ['home.read', 'home.other_read']}}}, 'history': [],
        'label': {'tool_calls': [{'name': name, 'arguments': {}} for name in ('home.read', 'home.other_read')]},
    } for i in range(4)]
    # Every app stays in train, including the single-sample library app.
    if architecture == 'qwen3_5':
        # Real assistant-only SFT contains a >8-helper batch, not just executor mocks.
        for row in source_rows:
            row['audio_text'] += '; '+ '; '.join(f'{i}.0 + {i+1}.0' for i in range(20))
            # Source-derived facade targets must survive preparation, aliases,
            # native batch serialization, assistant masking and real LoRA SFT.
            row['audio_text'] += '; Chuyển hai triệu rưỡi cho bà Hồng.'
        tools.append({'name': 'library.read', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}})
        source_rows.append({'id': 'train_4', 'audio': 'train_4.wav', 'audio_text': 'Kiểm tra thư viện',
            'context': {'installed_apps': {'library': {'tools': ['library.read']}}}, 'history': [],
            'label': {'tool_calls': [{'name': 'library.read', 'arguments': {}}]}})
    for spec in tools:
        spec['app'] = spec['name'].split('.')[0]
        spec['returns'] = ''
        spec['parameters']['properties']['limit'] = {'type': 'integer', 'minimum': 1,
            'description': 'Số nguyên, không dấu chấm hay phẩy, ví dụ 2'}
    (data / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': tools}]}))
    (data / 'train.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in source_rows))
    (data / 'zaic_tools.py').write_text('''class ToolEnv:
    def __init__(self, split, id, root):
        assert split == 'train'
        self.calls = []
    def validate(self, name, args): return {'ok': True}
    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return {'ok': True, 'devices': ['living_room']}
    def get_tool_calls(self): return self.calls
    def get_state_hash(self): return 'fixture-unchanged-state'
''')
    # Local ASR fixtures do not require model downloads.
    import numpy as np
    import soundfile as sf
    for i in range(len(source_rows)):
        sf.write(data / f'train_{i}.wav', np.zeros(1600, dtype=np.float32), 16000)
    asr = tmp_path / 'asr'
    asr.mkdir()
    for filename in ('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt'):
        (asr / filename).write_bytes(b'fixture ASR export')
    stubs = tmp_path / 'stub-runtime'
    stubs.mkdir()
    (stubs / 'sherpa_onnx.py').write_text('''from types import SimpleNamespace
class OfflineRecognizer:
    @staticmethod
    def from_transducer(**kwargs):
        class Stream:
            result = SimpleNamespace(text='ASR kiểm tra thiết bị')
            def accept_waveform(self, rate, samples):
                assert rate == 16000 and len(samples) == 1600
        return SimpleNamespace(create_stream=Stream, decode_streams=lambda streams: None)
''')
    app = Path(__file__).resolve().parents[1]
    run = tmp_path / 'run'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONPATH': str(stubs) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    result = subprocess.run([sys.executable, app / 'finetune.py', '--data-root', data.parent,
        '--base-model', base, '--output-dir', run, '--backend', 'transformers',
        '--asr-transcript-ratio', '0',
        '--max-steps', '2', '--gradient-accumulation', '1', '--save-strategy', 'steps', '--save-steps', '1',
        '--learning-rate', '0.01', '--export-merged'],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
    config = json.loads((run / 'training/training_config.json').read_text())
    assert config['quantization'] == 'none'
    assert config['max_length'] == 16384
    assert config['batch_size'] == 1
    assert config['eval_strategy'] == 'no'
    assert config['validation_steps'] == 0
    assert config['checkpoint_selection'] == 'last'
    assert config['numerics_guard'] == 'finite-loss-and-gradients-v1'
    assert not (run/'training/debug_numerics').exists()
    assert config['fuzzy_weight'] == 0.3
    assert config['epochs'] == 3
    prepared_manifest = json.loads((run/'sft/manifest.json').read_text())
    assert prepared_manifest['training_augmentation']['rename_variants'] == 0
    assert prepared_manifest['training_augmentation']['variants_kept'] == 0
    assert prepared_manifest['training_augmentation']['distractor_probability'] == 0.0
    assert set(prepared_manifest['partitions']) == {'train'}
    assert set(prepared_manifest['partitions']['train']['sample_ids']) == {row['id'] for row in source_rows}
    assert prepared_manifest['partitions']['train']['batch_examples'] >= (8 if architecture == 'qwen3_5' else 4)
    assert prepared_manifest['batching']['max_helper_calls'] is None
    assert prepared_manifest['batching']['max_helpers_per_sample'] is None
    assert prepared_manifest['batching']['max_read_calls'] == 4
    assert prepared_manifest['sft_format']['short_reasoning'] is False
    assert prepared_manifest['sft_format']['full_trajectory'] is False
    if architecture == 'qwen3_5':
        assert prepared_manifest['partitions']['train']['trajectories'] == len(source_rows)
        assert prepared_manifest['partitions']['train']['examples'] > len(source_rows)
        assert prepared_manifest['partitions']['train']['reasoning_stepwise_examples'] == 0
        assert prepared_manifest['partitions']['train']['local_helper_examples'] >= 84
        assert prepared_manifest['partitions']['train']['local_helper_examples_by_tool']['local_assist.solve'] >= 4
    assert prepared_manifest['assist_policy'] == 'fixed-python-task-workflows-v1'
    assert prepared_manifest['schema_prompt_policy'] == 'redundant-schema-metadata-v1'
    # Real preparation/training sees compact schemas, while the file consumed by
    # the organizer/env retains its original app/returns/parameter annotations.
    original_names = {spec['name'] for spec in tools}
    first = json.loads((run/'sft/train.jsonl').read_text().splitlines()[0])
    assert 'reasoning_content' not in first['messages'][-1]
    prompt_schemas = json.loads(first['messages'][1]['content'])['available_tools']['tool_schemas']
    for spec in prompt_schemas:
        if spec['name'] in original_names:
            assert 'app' not in spec and 'returns' not in spec
            assert spec['parameters']['properties']['limit'] == {'type': 'integer', 'minimum': 1}
    assert prepared_manifest['partitions']['train']['ungrouped_examples'] > prepared_manifest['partitions']['train']['examples']
    assert prepared_manifest['app_holdout']['status'] == 'off'
    assert not (run / 'sft/validation.jsonl').exists()
    assert not (run / 'sft/validation_rollouts.jsonl').exists()
    adapter = run / 'training' / 'adapter'
    assert (adapter / 'adapter_model.safetensors').is_file()
    runtime = json.loads((adapter / 'lifemate_runtime.json').read_text())
    assert runtime['inference_dtype'] == 'bfloat16'
    assert runtime['assist_policy'] == prepared_manifest['assist_policy']
    assert runtime['schema_prompt_policy'] == prepared_manifest['schema_prompt_policy']
    expected_protocol = 'qwen35-native-fc-v1' if architecture == 'qwen3_5' else 'lifemate-decision-json-v1'
    assert runtime['output_protocol'] == expected_protocol
    assert runtime['generation']['temperature'] == 0
    assert runtime['generation']['max_input_tokens'] == 16000
    assert runtime['generation']['max_new_tokens'] == 384
    assert runtime['generation']['thinking_budget'] == 0
    assert (run / 'training/checkpoint-2/lifemate_runtime.json').is_file()
    from safetensors.torch import load_file
    weights = load_file(str(adapter / 'adapter_model.safetensors'))
    if architecture == 'qwen3_5':
        assert all('visual' not in name and 'language_model' in name for name in weights)
    # The exported adapter contains the last optimizer step's weights.
    latest_weights = load_file(str(run / 'training/checkpoint-2/adapter_model.safetensors'))
    assert any(torch.count_nonzero(v).item() > 0 for k, v in latest_weights.items() if 'lora_B' in k)
    assert weights.keys() == latest_weights.keys()
    assert all(torch.equal(weights[k], latest_weights[k]) for k in weights)
    assert config['selection_metric'] is None
    assert config['final_step'] == 2
    assert not (run / 'training/eval_results.json').exists()
    assert not (run / 'training/validation_scores.jsonl').exists()
    assert not (run / 'training/validation_rollouts').exists()
    trainer_state = json.loads((run / 'training/trainer_state.json').read_text())
    assert trainer_state['best_metric'] is None
    assert trainer_state['best_model_checkpoint'] is None
    assert all(not any(key.startswith('eval_') for key in row) for row in trainer_state['log_history'])
    metrics = json.loads((run / 'training/train_results.json').read_text())
    assert 0 < metrics['train_loss'] < 100
    llm = LocalLLM(base, device='cpu', dtype='float32', adapter_path=adapter, max_new_tokens=2)
    merged_runtime = json.loads((run / 'merged/lifemate_runtime.json').read_text())
    assert merged_runtime['generation'] == runtime['generation']
    assert merged_runtime['chat_template_kwargs'] == runtime['chat_template_kwargs']
    merged_weights = load_file(str(run / 'merged/model.safetensors'))
    assert merged_weights['lm_head.weight'].dtype == torch.bfloat16
    merged = LocalLLM(run / 'merged', device='cpu', dtype='float32').model
    llm.model.eval()
    merged.eval()
    with torch.no_grad():
        tokens = torch.tensor([[3, 6, 0, 4]])
        a = llm.model(tokens).logits
        b = merged(tokens).logits
    assert torch.allclose(a, b, atol=0.003, rtol=0.01)
    assert isinstance(llm.generate('system', 'user'), str)
    assert (run / 'train_lora.log').is_file()
    assert (run / 'training/checkpoint-2/trainer_state.json').is_file()
    resume = subprocess.run([sys.executable, app / 'train_lora.py', '--base-model', base,
        '--sft-dir', run / 'sft', '--output-dir', run / 'training', '--backend', 'transformers',
        '--max-steps', '3', '--gradient-accumulation', '1', '--save-strategy', 'steps', '--save-steps', '1',
        '--learning-rate', '0.01',
        '--resume-from-checkpoint', run / 'training/checkpoint-2'],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert resume.returncode == 0, resume.stdout
    assert (run / 'training/checkpoint-3/trainer_state.json').is_file()
