"""Shared challenge context budget for training and inference (including chat tokens)."""
MAX_CONTEXT_TOKENS = 16384
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - DEFAULT_MAX_NEW_TOKENS
DEFAULT_THINKING_BUDGET = 0
MIN_TOOL_CALL_TOKENS = 192


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


def adaptive_thinking_budget(
    prompt_tokens: int,
    max_new_tokens: int,
    requested_budget: int,
    *,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    min_tool_call_tokens: int = MIN_TOOL_CALL_TOKENS,
) -> tuple[int, int]:
    """Compute effective (max_new_tokens, thinking_budget) within the total context limit.

    Always preserves at least `min_tool_call_tokens` for the final decision /
    `<tool_call>` block before allocating any remainder to `<think>...</think>`.
    """
    room_for_generation = max(0, min(int(max_new_tokens), int(max_context_tokens) - int(prompt_tokens)))
    if requested_budget <= 0:
        return room_for_generation, 0
    thinking = min(int(requested_budget), max(0, room_for_generation - int(min_tool_call_tokens)))
    return room_for_generation, thinking
