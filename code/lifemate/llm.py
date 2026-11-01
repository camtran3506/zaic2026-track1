"""Select the native offline loader for text-only use of Qwen3.5 checkpoints."""
import json
from pathlib import Path


def is_qwen35(model_path):
    config = Path(model_path) / 'config.json'
    return config.is_file() and json.loads(config.read_text()).get('model_type') == 'qwen3_5'


def model_loader(model_path):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if is_qwen35(model_path) else AutoModelForCausalLM


def lora_targets(model_path):
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    if is_qwen35(model_path):
        # Exclude the vision tower; retain the same attention/MLP projection set.
        return r'.*language_model\..*\.(' + '|'.join(targets) + ')'
    return targets


def configure_tokenizer(tokenizer, model_path):
    # Preserve direct JSON responses rather than spending the existing output
    # budget on Qwen3.5's optional reasoning mode. SFT must use the same template.
    tokenizer.lifemate_template_kwargs = {"enable_thinking": False} if is_qwen35(model_path) else {}
    return tokenizer
