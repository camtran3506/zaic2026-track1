"""Shared challenge context budget for training and inference (including chat tokens)."""
MAX_CONTEXT_TOKENS = 16384
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - DEFAULT_MAX_NEW_TOKENS


def validate_inference_budget(max_input_tokens, max_new_tokens):
    if max_input_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("--max-input-tokens and --max-new-tokens must be positive")
    if max_input_tokens + max_new_tokens > MAX_CONTEXT_TOKENS:
        raise ValueError(
            f"Input + output token budgets must be <= {MAX_CONTEXT_TOKENS}; "
            f"got {max_input_tokens} + {max_new_tokens}"
        )


def validate_training_length(max_length):
    if not 0 < max_length <= MAX_CONTEXT_TOKENS:
        raise ValueError(f"--max-length must be between 1 and {MAX_CONTEXT_TOKENS}")
