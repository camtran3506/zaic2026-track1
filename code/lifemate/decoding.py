"""Stop greedy generation at a complete decision object, not an inner brace."""
from __future__ import annotations

import json

from transformers import StoppingCriteria


def complete_decision_json(text: str) -> str | None:
    # Reasoning markers are special only outside JSON, not inside argument strings.
    skip_until = 0
    start, depth, in_string, escaped = None, 0, False, False
    for index, char in enumerate(text):
        if index < skip_until:
            continue
        if start is None:
            if text.startswith('<think>', index):
                end_think = text.find('</think>', index + len('<think>'))
                if end_think < 0:
                    return None
                skip_until = end_think + len('</think>')
                continue
            if char == '{':
                start, depth = index, 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth:
                continue
            candidate = text[start:index + 1]
            start = None
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            decision = obj
            if isinstance(obj.get('tool_calls'), list) and obj['tool_calls']:
                decision = obj['tool_calls'][0]
            if not isinstance(decision, dict):
                continue
            action = decision.get('action')
            if action in ('done', 'finish', 'stop', 'none'):
                return candidate
            if action == 'inspect' and isinstance(decision.get('name'), str):
                return candidate
            if (action == 'call' or ('name' in decision and 'arguments' in decision)):
                if isinstance(decision.get('name'), str) and isinstance(decision.get('arguments'), dict):
                    return candidate
    return None


class DecisionJsonStoppingCriteria(StoppingCriteria):
    """Single-request agent decoding. Never inspect JSON from the input prompt."""
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.completed_json = None

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('Decision JSON stopping supports one agent request at a time')
        generated = input_ids[0, self.prompt_length:]
        # Preserve think markers for the structural check; decode only generated tokens.
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        self.completed_json = complete_decision_json(text)
        return self.completed_json is not None
