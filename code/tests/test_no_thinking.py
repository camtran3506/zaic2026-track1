"""Zero-budget generation blocks reasoning before decoding, preserving payloads."""
import json
from types import SimpleNamespace

import pytest
import torch

from lifemate.decoding import DirectActionLogitsProcessor
from lifemate.models import LocalLLM
from test_stopping_filters import native_call, raw_bytes, tokenizer_and_ids


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('width', [1, 2, 7, 10000])
def test_all_valid_token_boundaries_and_literal_think_payloads(native, width):
    text = (native_call('Hồng: <think>literal</think>') if native else json.dumps({
        'action': 'call', 'name': 'home.write', 'arguments': {'text': '<think>Hồng</think>'}}, ensure_ascii=False))
    raw = raw_bytes(text)
    pieces = [raw[i:i + width] for i in range(0, len(raw), width)]
    forbidden = ['<think>', '</think>', '<thi', 'Let me think', 'Reasoning:', '```json']
    tok, all_ids = tokenizer_and_ids([*pieces, *forbidden])
    ids, bad = all_ids[:len(pieces)], all_ids[len(pieces):]
    prompt = [bad[0], bad[1]]  # Input markers are irrelevant to the new output.
    processor = DirectActionLogitsProcessor(tok, len(prompt), native=native)
    generated = []
    for next_id in ids:
        scores = torch.zeros(1, len(tok))
        before = tok.decode(generated, skip_special_tokens=False).lstrip()
        committed = before.startswith('{') or (native and before.startswith('<tool_call>'))
        result = processor(torch.tensor([prompt + generated]), scores)
        assert torch.isfinite(result[0, next_id])
        if not committed:
            assert torch.isneginf(result[0, bad]).all()
        generated.append(next_id)
    assert tok.decode(generated, skip_special_tokens=False) == text


def test_merged_prefix_tokens_are_allowed_and_json_mode_blocks_native_opener():
    pieces = [' {"action":"done"}', '<tool_call>\n<function=home.read>', '<think>analysis</think>', '\n']
    tok, ids = tokenizer_and_ids([raw_bytes(p) for p in pieces])
    for native in (False, True):
        processor = DirectActionLogitsProcessor(tok, 1, native=native)
        result = processor(torch.tensor([[2]]), torch.zeros(1, len(tok)))
        assert torch.isfinite(result[0, ids[0]])
        assert bool(torch.isfinite(result[0, ids[1]])) is native
        assert torch.isneginf(result[0, ids[2:]]).all()


def test_impossible_logits_fail_closed_instead_of_generating_masked_thinking():
    tok, ids = tokenizer_and_ids(['<think>'])
    scores = torch.full((1, len(tok)), -float('inf'))
    scores[0, ids[0]] = 10
    with pytest.raises(ValueError, match='No finite direct-action'):
        DirectActionLogitsProcessor(tok, 1)(torch.tensor([[2]]), scores)


def test_vocabulary_decode_is_cached_across_turns(monkeypatch):
    tok, _ = tokenizer_and_ids(['<think>', '<tool_call>'])
    original = tok.batch_decode
    calls = []
    def decode(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(tok, 'batch_decode', decode)
    DirectActionLogitsProcessor(tok, 1, native=True)(torch.tensor([[2]]), torch.zeros(1, len(tok)))
    count = len(calls)
    assert count
    DirectActionLogitsProcessor(tok, 1, native=True)(torch.tensor([[2]]), torch.zeros(1, len(tok)))
    assert len(calls) == count


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('budget', [0, 256])
def test_local_llm_real_generation_blocks_reasoning_and_keeps_argument_markers(native, budget):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerFast,
                              Qwen3Config, Qwen3ForCausalLM)
    tokens = ['[UNK]', '[EOS]', 'PROMPT', '<think>', '</think>', 'literal',
              '<tool_call>', '<function=home.write>', '<parameter=text>', '</parameter>',
              '</function>', '</tool_call>', '{"action":"done"}']
    backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]',
                                 pad_token='[EOS]', additional_special_tokens=['<think>', '</think>', '<tool_call>', '</tool_call>'])
    tok.lifemate_native_fc = native
    tok.lifemate_template_kwargs = {'enable_thinking': False} if native else {}
    tok.chat_template = ('{% for m in messages %}PROMPT {% endfor %}{% if add_generation_prompt %}'
                         '{% if enable_thinking %}<think>{% else %}<think> </think>{% endif %}{% endif %}')
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, eos_token_id=1, pad_token_id=1)).eval()
    sequence = [6, 7, 8, 3, 5, 4, 9, 10, 11, 1] if native else [12, 1]
    captured = []

    class TemptThinking(LogitsProcessor):
        def __init__(self, prompt_length): self.prompt_length = prompt_length
        def __call__(self, ids, scores):
            step = ids.shape[1] - self.prompt_length
            scores.fill_(-float('inf'))
            scores[:, sequence[min(step, len(sequence) - 1)]] = 0
            if step == 0:
                scores[:, 3] = 100  # Without the production guard, <think> wins.
            return scores

    def generate(**kwargs):
        start = kwargs['input_ids'].shape[1]
        guards = list(kwargs.pop('logits_processor'))
        assert [type(p).__name__ for p in guards] == ['DirectActionLogitsProcessor']
        # budget=256 is deliberately clamped to 0 by max_new_tokens=32.
        if native:
            assert tok.decode(kwargs['input_ids'][0], skip_special_tokens=False).rstrip().endswith('</think>')
        out = model.generate(**kwargs, logits_processor=LogitsProcessorList([TemptThinking(start), *guards]))
        captured.append(tok.decode(out[0, start:], skip_special_tokens=False))
        return out

    llm = LocalLLM.from_loaded_model(SimpleNamespace(generate=generate,
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1))), tok,
        max_input_tokens=100, max_new_tokens=32)
    result = llm.generate('system', 'request', thinking_budget=budget)
    assert not captured[0].lstrip().startswith('<think>')
    if native:
        from lifemate.function_calling import parse_native_call
        assert parse_native_call(result, {'home.write': {'parameters': {
            'type': 'object', 'properties': {'text': {'type': 'string'}}}}}) == {
                'action': 'call', 'name': 'home.write', 'arguments': {'text': ' <think> literal </think> '}}
    else:
        assert json.loads(result) == {'action': 'done'}
