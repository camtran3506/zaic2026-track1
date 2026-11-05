"""Stop greedy generation at a complete decision object, not an inner brace."""
from __future__ import annotations

import json
from weakref import WeakKeyDictionary

from transformers import StoppingCriteria


_CLOSING_TOKEN_TABLES = WeakKeyDictionary()


def _closing_token_table(tokenizer):
    """Conservative ASCII-closing filter for verified, monotone fast decoders.

    ByteLevel and the default space-join decoder cannot invent ASCII '}' or '>'
    unless a new vocabulary piece contains that character. Do not assume that a
    closing tag has one fixed tokenization. Unknown/custom decoders use the full
    check, including byte-fallback/CTC decoders that can rewrite previous text.
    Tokenizers are immutable during a single greedy generate() call.
    """
    from transformers import PreTrainedTokenizerFast
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return None
    public_decode = getattr(tokenizer.decode, "__func__", None)
    backend_decode = getattr(tokenizer._decode, "__func__", None)
    public_source = getattr(getattr(public_decode, "__code__", None), "co_filename", "").replace('\\', '/')
    backend_source = getattr(getattr(backend_decode, "__code__", None), "co_filename", "").replace('\\', '/')
    if (public_decode is not PreTrainedTokenizerBase.decode
            or backend_decode is not PreTrainedTokenizerFast._decode
            or not public_source.endswith('transformers/tokenization_utils_base.py')
            or not backend_source.endswith(('transformers/tokenization_utils_fast.py',
                                            'transformers/tokenization_utils_tokenizers.py'))
            or tokenizer.clean_up_tokenization_spaces):
        # Check the implementation too: class-wide patches (even functools.wraps
        # wrappers) would otherwise compare equal to the patched class method.
        return None
    backend = tokenizer.backend_tokenizer
    decoder = backend.decoder
    if type(backend.model).__name__ not in {"BPE", "WordLevel"}:
        return None
    if decoder is not None and type(decoder).__name__ != "ByteLevel":
        return None
    key = (len(tokenizer), type(backend.model).__name__,
           decoder.__getstate__() if decoder is not None else None)
    try:
        cached = _CLOSING_TOKEN_TABLES.get(tokenizer)
    except TypeError:
        return None
    if cached is not None and cached[0] == key:
        return cached[1]
    vocab = tokenizer.get_vocab()
    if not vocab or any(type(idx) is not int or idx < 0 for idx in vocab.values()):
        return None
    largest = max(vocab.values())
    if largest > 4 * len(vocab):
        return None  # Do not allocate an unbounded sparse ID table.
    # Unknown IDs stay candidates. Aliases containing a closer also win over
    # aliases without one, so the filter can only cause extra full checks.
    table = bytearray(b'\x01') * (largest + 1)
    for idx in vocab.values():
        table[idx] = 0
    for piece, idx in vocab.items():
        if '}' in piece or '>' in piece:
            table[idx] = 1
    table = bytes(table)
    _CLOSING_TOKEN_TABLES[tokenizer] = (key, table)
    return table


class _FilteredStoppingCriteria(StoppingCriteria):
    """Only transfer newly appended token IDs until a full parse may succeed."""
    def __init__(self, tokenizer, prompt_length, *, fast_check=True):
        self.tokenizer, self.prompt_length = tokenizer, prompt_length
        self._last_length = prompt_length
        self._closing_tokens = _closing_token_table(tokenizer) if fast_check else None
        self.full_decodes = 0
        self.skipped_checks = 0

    def _needs_full_check(self, input_ids):
        length = input_ids.shape[1]
        previous, self._last_length = self._last_length, length
        if self._closing_tokens is None or length <= previous:
            # Repeated/rewound sequences are checked in full. Production greedy
            # generation appends tokens and creates a fresh criterion per call.
            return True
        new_ids = input_ids[0, max(previous, self.prompt_length):].tolist()
        table = self._closing_tokens
        if any(type(idx) is not int or idx < 0 or idx >= len(table) or table[idx] for idx in new_ids):
            return True
        self.skipped_checks += 1
        return False

    def _decode(self, input_ids):
        self.full_decodes += 1
        # Never decode input-prompt JSON, and never drop literal special markers.
        return self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=False)


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
            if (action == 'call' or ('action' not in decision and 'name' in decision and 'arguments' in decision)):
                if isinstance(decision.get('name'), str) and isinstance(decision.get('arguments'), dict):
                    return candidate
    return None


class DecisionJsonStoppingCriteria(_FilteredStoppingCriteria):
    """Single-request agent decoding. Never inspect JSON from the input prompt."""
    def __init__(self, tokenizer, prompt_length: int, *, fast_check=True):
        super().__init__(tokenizer, prompt_length, fast_check=fast_check)
        self.completed_json = None

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('Decision JSON stopping supports one agent request at a time')
        if not self._needs_full_check(input_ids):
            return self.completed_json is not None
        text = self._decode(input_ids)
        self.completed_json = complete_decision_json(text)
        return self.completed_json is not None


class NativeToolStoppingCriteria(_FilteredStoppingCriteria):
    """Wait for the entire native XML call, never a JSON object inside a parameter."""
    def __init__(self, tokenizer, prompt_length, *, fast_check=True):
        super().__init__(tokenizer, prompt_length, fast_check=fast_check)
        self.completed_output = None

    def __call__(self, input_ids, scores, **kwargs):
        from .function_calling import complete_native_call, visible_output
        if input_ids.shape[0] != 1:
            raise ValueError('Native tool stopping supports one agent request at a time')
        if not self._needs_full_check(input_ids):
            return self.completed_output is not None
        text = self._decode(input_ids)
        visible = visible_output(text)
        if '<tool_call>' in visible:
            self.completed_output = complete_native_call(visible)
        else:
            # A native tool conversation still uses the explicit terminal done marker.
            candidate = complete_decision_json(visible)
            self.completed_output = candidate if candidate and json.loads(candidate) == {'action': 'done'} else None
        return self.completed_output is not None
