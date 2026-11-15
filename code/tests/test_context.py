"""Training and generation must respect the same total context budget."""
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from lifemate.context import MAX_CONTEXT_TOKENS
from lifemate.models import LocalLLM

APP = Path(__file__).resolve().parents[1]


def test_inference_defaults_match_training_context(monkeypatch):
    from main import parse_args
    monkeypatch.setattr(sys, 'argv', ['main.py', '--data-root', 'data'])
    args = parse_args()
    assert (args.max_input_tokens, args.max_new_tokens) == (16000, 384)
    assert args.thinking_budget == 0
    assert args.max_input_tokens + args.max_new_tokens == MAX_CONTEXT_TOKENS


@pytest.mark.parametrize('budgets', [(15872, 513), (16000, 385), (16384, 1), (0, 512), (15872, -1)])
def test_invalid_budget_fails_before_dataset_or_model_loading(budgets):
    result = subprocess.run([sys.executable, APP / 'main.py', '--data-root', 'missing',
        '--max-input-tokens', str(budgets[0]), '--max-new-tokens', str(budgets[1])],
        capture_output=True, text=True)
    assert result.returncode != 0
    assert 'token' in result.stderr
    assert 'FileNotFoundError' not in result.stderr
    with pytest.raises(ValueError):
        LocalLLM(Path('missing'), max_input_tokens=budgets[0], max_new_tokens=budgets[1])


def test_reallocated_output_budget_is_preserved(tmp_path):
    # Longer JSON arguments can reserve more output tokens by reducing input.
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=Mock()), \
         patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=Mock()):
        llm = LocalLLM(tmp_path, device='cpu', max_input_tokens=15360, max_new_tokens=1024)
    assert llm.max_input_tokens == 15360
    assert llm.max_new_tokens == 1024


def test_training_cannot_exceed_challenge_context():
    result = subprocess.run([sys.executable, APP / 'training_code' / 'train_lora.py',
        '--base-model', 'model', '--sft-dir', 'sft', '--output-dir', 'run',
        '--max-length', '16385'], capture_output=True, text=True)
    assert result.returncode != 0
    assert '--max-length must be between 1 and 16384' in result.stderr


def test_complete_prompt_over_budget_fails_in_sft_and_inference_before_generation():
    import torch
    from types import SimpleNamespace
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from lifemate.agent import SYSTEM_PROMPT, build_step_prompt
    from lifemate.llm import chat_token_ids
    from lifemate.sft import encode_example
    from test_baseline import CATALOG

    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[EOS]': 1}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       eos_token='[EOS]', pad_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    history = [{'role': 'user', 'text': 'preserve all these words ' * 500}]
    _, prompt = build_step_prompt('request', {}, history, CATALOG, [])
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': '{"action":"done"}'}]
    full_length = len(chat_token_ids(tokenizer, messages, add_generation_prompt=False))
    example = {'sample_id': 'train_long', 'step': 0, 'messages': messages}
    row = encode_example(example, tokenizer, full_length)
    assert len(row['input_ids']) == full_length
    with pytest.raises(ValueError, match='never truncate prompt or gold JSON'):
        encode_example(example, tokenizer, full_length - 1)
    model = Mock()
    model.get_input_embeddings.return_value = SimpleNamespace(weight=torch.zeros(1))
    model.lifemate_generation_policy = None
    llm = LocalLLM.from_loaded_model(model, tokenizer, max_input_tokens=128, max_new_tokens=8)
    with pytest.raises(ValueError, match='full schemas, history and tool results are never truncated'):
        llm.generate(SYSTEM_PROMPT, prompt)
    model.generate.assert_not_called()
