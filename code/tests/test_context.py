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
    assert args.max_input_tokens + args.max_new_tokens == MAX_CONTEXT_TOKENS


@pytest.mark.parametrize('budgets', [(16000, 385), (16384, 1), (0, 384), (16000, -1)])
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
    result = subprocess.run([sys.executable, APP / 'train_lora.py',
        '--base-model', 'model', '--sft-dir', 'sft', '--output-dir', 'run',
        '--max-length', '16385'], capture_output=True, text=True)
    assert result.returncode != 0
    assert '--max-length must be between 1 and 16384' in result.stderr
