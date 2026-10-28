"""Actual tiny random Qwen + LoRA training; no downloads or competition data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_train_adapter_reload_and_merge(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('peft')
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast, AutoModelForCausalLM
    from tokenizers import Tokenizer, models, pre_tokenizers
    from lifemate.models import LocalLLM
    base = tmp_path / 'tiny-base'
    base.mkdir()
    tok = Tokenizer(models.WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2,
        '<s>': 3, '</s>': 4, 'system': 5, 'user': 6, 'assistant': 7}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token='[UNK]',
        pad_token='[PAD]', eos_token='[EOS]', additional_special_tokens=['<s>', '</s>'])
    tokenizer.chat_template = "{% for m in messages %}{{ '<s>' + m['role'] + '\n' + m['content'] + '</s>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<s>assistant\n' }}{% endif %}"
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.save_pretrained(base)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=4096, eos_token_id=2, pad_token_id=1))
    model.save_pretrained(base)
    data = tmp_path / 'data' / 'train'
    data.mkdir(parents=True)
    (data / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': [
        {'name': 'home.read', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}}]}]}))
    (data / 'train.jsonl').write_text(''.join(json.dumps({
        'id': f'train_{i}', 'audio_text': 'Kiểm tra thiết bị',
        'context': {'installed_apps': {'home': {'tools': ['home.read']}}}, 'history': [],
        'label': {'tool_calls': [{'name': 'home.read', 'arguments': {}}]},
    }) + '\n' for i in range(4)))
    (data / 'zaic_tools.py').write_text('''class ToolEnv:
    def __init__(self, split, id, root): assert split == 'train'
    def validate(self, name, args): return {'ok': True}
    def call(self, name, args): return {'ok': True, 'devices': ['living_room']}
''')
    app = Path(__file__).resolve().parents[1]
    run = tmp_path / 'run'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    result = subprocess.run([sys.executable, app / 'finetune.py', '--data-root', data.parent,
        '--base-model', base, '--output-dir', run, '--backend', 'transformers', '--quantization', 'none',
        '--max-steps', '2', '--gradient-accumulation', '1', '--save-steps', '1',
        '--max-length', '4096', '--learning-rate', '0.01', '--export-merged'],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert result.returncode == 0, result.stdout
    adapter = run / 'training' / 'adapter'
    assert (adapter / 'adapter_model.safetensors').is_file()
    from safetensors.torch import load_file
    weights = load_file(str(adapter / 'adapter_model.safetensors'))
    assert any(torch.count_nonzero(v).item() > 0 for k, v in weights.items() if 'lora_B' in k)
    metrics = json.loads((run / 'training/eval_results.json').read_text())
    assert 0 < metrics['eval_loss'] < 100
    llm = LocalLLM(base, device='cpu', dtype='float32', adapter_path=adapter, max_new_tokens=2)
    merged = AutoModelForCausalLM.from_pretrained(run / 'merged', local_files_only=True, torch_dtype=torch.float32)
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
        '--sft-dir', run / 'sft', '--output-dir', run / 'training', '--backend', 'transformers', '--quantization', 'none',
        '--max-steps', '3', '--gradient-accumulation', '1', '--save-steps', '1',
        '--max-length', '4096', '--learning-rate', '0.01',
        '--resume-from-checkpoint', run / 'training/checkpoint-2'],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    assert resume.returncode == 0, resume.stdout
    assert (run / 'training/checkpoint-3/trainer_state.json').is_file()
