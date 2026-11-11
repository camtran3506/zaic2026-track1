"""Select the native offline loader for text-only use of Qwen3.5 checkpoints."""
import json
from pathlib import Path

from .function_calling import template_inputs


def is_qwen35(model_path):
    config = Path(model_path) / 'config.json'
    return config.is_file() and json.loads(config.read_text()).get('model_type') == 'qwen3_5'


def model_loader(model_path):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    return AutoModelForImageTextToText if is_qwen35(model_path) else AutoModelForCausalLM


def lora_targets(model_path):
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    if is_qwen35(model_path):
        # Exclude the vision tower; include Qwen3.5 GatedDeltaNet linear_attn projections.
        qwen35_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'in_proj_qkv', 'in_proj_z', 'out_proj',
                          'gate_proj', 'up_proj', 'down_proj']
        return r'.*language_model\..*\.(' + '|'.join(qwen35_targets) + ')'
    return targets


def configure_tokenizer(tokenizer, model_path):
    # Use direct native tool calls without Qwen3.5's optional reasoning mode.
    # SFT and inference must use the same checkpoint template and kwargs.
    tokenizer.lifemate_native_fc = is_qwen35(model_path)
    tokenizer.lifemate_template_kwargs = {"enable_thinking": False} if tokenizer.lifemate_native_fc else {}
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def chat_token_ids(tokenizer, messages, add_generation_prompt):
    messages, kwargs = template_inputs(tokenizer, messages)
    encoded = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=add_generation_prompt, **kwargs)
    return encoded["input_ids"] if hasattr(encoded, "keys") else encoded


def render_prompt(tokenizer, messages, *, add_generation_prompt=True):
    """Portable prompt text and exact input IDs for any inference engine."""
    messages, kwargs = template_inputs(tokenizer, messages)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
    ids = tokenizer(text, add_special_tokens=False, truncation=False,
                    return_token_type_ids=False)["input_ids"]
    return text, ids
