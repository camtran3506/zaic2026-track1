"""Differential stopping tests across byte boundaries, payloads and real generate()."""
import json
import random

import pytest
import torch

from lifemate.decoding import (DecisionJsonStoppingCriteria, NativeToolStoppingCriteria,
                              _closing_token_table, complete_decision_json)
from lifemate.function_calling import complete_native_call, parse_native_call, visible_output


def raw_bytes(text):
    from tokenizers.pre_tokenizers import ByteLevel
    return ''.join(piece for piece, _ in ByteLevel(add_prefix_space=False).pre_tokenize_str(text))


def tokenizer_and_ids(pieces):
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    vocabulary = {'[UNK]': 0, '[EOS]': 1, 'PROMPT': 2, '{"action":"done"}': 3, 'TRAILING': 4}
    for piece in pieces:
        vocabulary.setdefault(piece, len(vocabulary))
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token='[UNK]'))
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]',
                                       pad_token='[EOS]', clean_up_tokenization_spaces=False)
    return tokenizer, [vocabulary[piece] for piece in pieces]


def native_call(value='hello', *, parameter='text'):
    return (f'<tool_call>\n<function=home.write>\n<parameter={parameter}>\n'
            f'{value}\n</parameter>\n</function>\n</tool_call>')


JSON_TEXTS = [
    '{"action":"done"}',
    '{"action":"finish"}',
    json.dumps({'action': 'call', 'name': 'home.write', 'arguments': {
        'nested': {'items': [1, {'action': 'done'}, False, None]},
        'text': 'Hồng nói: } { "quote" \\ slash; </tool_call> <think>literal</think>'}}, ensure_ascii=False),
    '{"tool_calls":[{"name":"home.write","arguments":{"x":{"y":[]}}}]}',
    '<think>{"action":"done"}</think>\n<think>more</think>\n{"action":"done"}',
    '```json\n{"other":true}\n{"action":"done"}\n``` TRAILING',
    '{"action":"call","name":"x","arguments":{"text":"unfinished } escaped \\"',
    '{"action":"inspect","arguments":{}}',
]
NATIVE_TEXTS = [
    native_call(),
    '<tool_call><function=home.read></function></tool_call>TRAILING',
    native_call('{"nested":[{"action":"done"}],"text":"} {"}', parameter='payload'),
    native_call('Hồng: literal </tool_call>, </think>, \\ and "quote".'),
    native_call('literal <tool_call><function=inner.read></function></tool_call> stays inside text'),
    '<think>' + native_call() + ' {"action":"done"}</think>\n' + native_call('actual'),
    '{"action":"done"} TRAILING',
    '<think>{"action":"done"}</think> {"action":"done"}',
    native_call()[:-1],  # Missing final > must never be completed by a guess.
    '<tool_call><function=home.write><parameter=text>{"action":"done"}',
    '{"action":"call","name":"home.write","arguments":{}}',  # Native mode accepts only JSON done.
    json.dumps({'action': 'call', 'name': 'home.write', 'arguments': {
        'text': '<tool_call><function=inner.read></function></tool_call>'}}),
    '```json\n' + json.dumps({'action': 'call', 'name': 'home.write', 'arguments': {
        'text': '<tool_call><function=inner.read></function></tool_call>'}}) + '\n```',
    '{"analysis":"standalone metadata, not an argument"} ' + native_call(),
    '"quoted preamble" ' + native_call(),
]


def full_reference(kind, text):
    if kind == 'json':
        return complete_decision_json(text)
    visible = visible_output(text)
    if '<tool_call>' in visible:
        return complete_native_call(visible)
    candidate = complete_decision_json(visible)
    return candidate if candidate and json.loads(candidate) == {'action': 'done'} else None


def criterion_class(kind):
    return DecisionJsonStoppingCriteria if kind == 'json' else NativeToolStoppingCriteria


def output_of(criterion, kind):
    return criterion.completed_json if kind == 'json' else criterion.completed_output


def compare_every_step(kind, tokenizer, ids, *, groups=1):
    # A complete decision in the INPUT prompt must not affect stopping.
    prompt = [3, 2]
    cls = criterion_class(kind)
    fast, slow = cls(tokenizer, len(prompt)), cls(tokenizer, len(prompt), fast_check=False)
    assert fast._closing_tokens is not None
    stop_at = None
    ends = list(range(groups, len(ids) + 1, groups))
    if not ends or ends[-1] != len(ids):
        ends.append(len(ids))
    for end in ends:
        tensor = torch.tensor([prompt + ids[:end]])
        text = tokenizer.decode(ids[:end], skip_special_tokens=False)
        expected = full_reference(kind, text)
        assert fast(tensor, None) == slow(tensor, None) == (expected is not None), (kind, end, text)
        assert output_of(fast, kind) == output_of(slow, kind) == expected
        if expected is not None:
            stop_at = end
            break
    return fast, stop_at


@pytest.mark.parametrize('kind,text', [('json', t) for t in JSON_TEXTS] + [('native', t) for t in NATIVE_TEXTS])
@pytest.mark.parametrize('width', [1, 2, 7, 31, 10000])
def test_every_token_boundary_matches_full_decode_reference(kind, text, width):
    raw = raw_bytes(text)
    pieces = [raw[i:i+width] for i in range(0, len(raw), width)]
    tok, ids = tokenizer_and_ids(pieces)
    assert tok.decode(ids, skip_special_tokens=False) == text
    compare_every_step(kind, tok, ids)


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_random_partitions_preserve_first_stop_and_completed_output(kind):
    text = JSON_TEXTS[2] if kind == 'json' else NATIVE_TEXTS[4]
    raw = raw_bytes(text + ' TRAILING')
    for seed in range(60):
        rng, pieces, offset = random.Random(seed), [], 0
        while offset < len(raw):
            width = rng.randint(1, 25)
            pieces.append(raw[offset:offset+width])
            offset += width
        tok, ids = tokenizer_and_ids(pieces)
        _, stopped = compare_every_step(kind, tok, ids)
        assert stopped is not None


def test_real_merged_byte_bpe_encode_matches_full_checks_in_both_protocols():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.BPE(unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(JSON_TEXTS + NATIVE_TEXTS,
        trainers.BpeTrainer(vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                           special_tokens=['[UNK]', '[EOS]']))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]',
                                 pad_token='[EOS]', clean_up_tokenization_spaces=False)
    for kind, texts in [('json', JSON_TEXTS), ('native', NATIVE_TEXTS)]:
        for text in texts:
            ids = tok(text, add_special_tokens=False)['input_ids']
            assert tok.decode(ids, skip_special_tokens=False) == text
            compare_every_step(kind, tok, ids)


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_every_split_of_the_terminal_marker_is_recognized_at_the_same_step(kind):
    head = '{"action":"done"' if kind == 'json' else '<tool_call><function=home.read></function>'
    marker = '}' if kind == 'json' else '</tool_call>'
    for split in range(len(marker) + 1):
        pieces = [head + marker[:split], marker[split:] + ' trailing words']
        tok, ids = tokenizer_and_ids(pieces)
        fast, stopped = compare_every_step(kind, tok, ids)
        assert stopped == (1 if split == len(marker) else 2)
        assert output_of(fast, kind) == head + marker


@pytest.mark.parametrize('groups', [2, 5, 10000])
@pytest.mark.parametrize('kind', ['json', 'native'])
def test_check_all_new_tokens_when_callback_observes_more_than_one_token(kind, groups):
    text = ('{"action":"done"}' if kind == 'json' else native_call()) + ' TRAILING without closers'
    tok, ids = tokenizer_and_ids(list(raw_bytes(text)))
    compare_every_step(kind, tok, ids, groups=groups)


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_marker_completed_in_middle_of_token_is_not_missed(kind):
    if kind == 'json':
        pieces = ['{"action":', '"done"', '} TRAILING has no closer at the end']
        expected = '{"action":"done"}'
    else:
        pieces = ['<tool_call><function=home.read>', '</function>', '</tool_call>TRAILING']
        expected = '<tool_call><function=home.read></function></tool_call>'
    tok, ids = tokenizer_and_ids(pieces)
    fast, stopped = compare_every_step(kind, tok, ids)
    assert stopped == 3 and output_of(fast, kind) == expected


def test_literal_inner_native_call_never_ends_the_outer_parameter_early():
    value = 'literal <tool_call><function=inner.read></function></tool_call> not executed'
    text = native_call(value)
    for end in range(len(text)):
        assert complete_native_call(text[:end]) is None
    incomplete = text[:text.index(' not executed')]
    with pytest.raises(ValueError, match='complete native tool call'):
        parse_native_call(incomplete, {})
    target = parse_native_call(text, {})
    assert target == {'action': 'call', 'name': 'home.write', 'arguments': {'text': value}}


@pytest.mark.parametrize('fenced', [False, True])
def test_json_fallback_argument_with_native_markup_is_not_misread_as_a_call(fenced):
    from lifemate.agent import parse_decision
    target = {'action': 'call', 'name': 'home.write', 'arguments': {
        'text': '<tool_call><function=inner.read></function></tool_call>'}}
    text = json.dumps(target)
    if fenced:
        text = '```json\n' + text + '\n```'
    for end in range(len(text) + 1):
        assert complete_native_call(text[:end]) is None
    assert parse_decision(text) == target


def test_native_call_after_complete_json_preamble_is_not_delayed():
    expected = native_call()
    for prefix in ('{"analysis":"standalone metadata"} ', '"quoted preamble" '):
        assert complete_native_call(prefix + expected) == expected


def test_no_fixed_tail_length_assumption_even_with_empty_tokens_between_marker_bytes():
    pieces = ['<tool_call><function=home.read></function></tool', *([''] * 150), '_call>']
    tok, ids = tokenizer_and_ids(pieces)
    fast, stopped = compare_every_step('native', tok, ids)
    assert stopped == len(ids)
    assert fast.completed_output == '<tool_call><function=home.read></function></tool_call>'


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_repeated_or_rewound_sequences_do_not_reuse_a_stale_completed_decision(kind):
    text = '{"action":"done"}' if kind == 'json' else native_call()
    tok, ids = tokenizer_and_ids(list(raw_bytes(text)))
    criterion = criterion_class(kind)(tok, 1)
    assert criterion(torch.tensor([[2] + ids]), None)
    assert not criterion(torch.tensor([[2] + ids[:3]]), None)
    assert output_of(criterion, kind) is None
    assert not criterion(torch.tensor([[2] + ids[:3]]), None)
    assert criterion(torch.tensor([[2] + ids]), None)


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_unsupported_decoder_that_invents_a_closing_brace_uses_full_checks(kind):
    from tokenizers import decoders
    tok, ids = tokenizer_and_ids(['{"action":', '"done"', 'X'])
    tok.backend_tokenizer.decoder = decoders.Sequence([decoders.ByteLevel(), decoders.Replace('X', '}')])
    criterion = criterion_class(kind)(tok, 1)
    assert criterion._closing_tokens is None
    for end in range(1, 4):
        actual = criterion(torch.tensor([[2] + ids[:end]]), None)
        assert actual == (end == 3)
    assert criterion.full_decodes == 3 and criterion.skipped_checks == 0


def test_custom_python_decode_falls_back_instead_of_assuming_standard_backend(monkeypatch):
    tok, ids = tokenizer_and_ids(['{"action":', '"done"', 'X'])
    original = tok.decode
    monkeypatch.setattr(tok, 'decode', lambda *args, **kwargs: original(*args, **kwargs).replace('X', '}'))
    criterion = DecisionJsonStoppingCriteria(tok, 1)
    assert criterion._closing_tokens is None
    assert criterion(torch.tensor([[2] + ids]), None)


@pytest.mark.parametrize('method', ['decode', '_decode'])
@pytest.mark.parametrize('kind', ['json', 'native'])
def test_class_wide_decode_patch_with_wraps_also_uses_full_checks(monkeypatch, method, kind):
    from functools import wraps
    from transformers import PreTrainedTokenizerFast
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    tok, ids = tokenizer_and_ids(['{"action":', '"done"', 'X'])
    cls = PreTrainedTokenizerBase if method == 'decode' else PreTrainedTokenizerFast
    original = getattr(cls, method)
    @wraps(original)
    def replacement(self, *args, **kwargs):
        return original(self, *args, **kwargs).replace('X', '}')
    monkeypatch.setattr(cls, method, replacement)
    criterion = criterion_class(kind)(tok, 1)
    assert criterion._closing_tokens is None
    for end in range(1, 4):
        assert criterion(torch.tensor([[2] + ids[:end]]), None) == (end == 3)


def test_vocabulary_table_is_shared_and_rebuilt_after_added_token_or_decoder_change(monkeypatch):
    from tokenizers import decoders
    tok, _ = tokenizer_and_ids(['plain'])
    original, calls = tok.get_vocab, []
    def vocab():
        calls.append(True)
        return original()
    monkeypatch.setattr(tok, 'get_vocab', vocab)
    first = DecisionJsonStoppingCriteria(tok, 1)
    second = NativeToolStoppingCriteria(tok, 1)
    assert first._closing_tokens is second._closing_tokens
    assert len(calls) == 1
    tok.add_tokens(['new}>token'])
    third = DecisionJsonStoppingCriteria(tok, 1)
    new_id = tok.convert_tokens_to_ids('new}>token')
    assert third._closing_tokens is not first._closing_tokens and third._closing_tokens[new_id]
    assert len(calls) == 2
    tok.backend_tokenizer.decoder = decoders.Replace('plain', '}')
    assert _closing_token_table(tok) is None


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_full_decode_count_is_reduced_for_long_plain_arguments(kind):
    value = 'a' * 700
    text = json.dumps({'action': 'call', 'name': 'home.write', 'arguments': {'text': value}}) if kind == 'json' else native_call(value)
    tok, ids = tokenizer_and_ids(list(raw_bytes(text)))
    fast, stopped = compare_every_step(kind, tok, ids)
    assert stopped == len(ids)
    assert fast.full_decodes < 15 and fast.skipped_checks > 650


@pytest.mark.parametrize('kind', ['json', 'native'])
def test_batch_size_other_than_one_is_still_rejected(kind):
    tok, _ = tokenizer_and_ids(['text'])
    criterion = criterion_class(kind)(tok, 1)
    with pytest.raises(ValueError, match='one agent request'):
        criterion(torch.tensor([[2, 2], [2, 2]]), None)


@pytest.mark.parametrize('kind', ['json', 'native'])
@pytest.mark.parametrize('bundled', [False, True])
def test_real_qwen35_generate_has_identical_tokens_and_stops_before_trailing_output(kind, bundled):
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
    from test_prefix_cache import tiny_model
    from lifemate.runtime import generation_context, generation_policy

    value = '{"inner":{"action":"done"},"literal":"</tool_call>","unicode":"Hồng"}'
    text = json.dumps({'action': 'call', 'name': 'home.write', 'arguments': {'text': value}}, ensure_ascii=False) if kind == 'json' else native_call(value)
    raw = raw_bytes(text)
    pieces = [raw[i:i+7] for i in range(0, len(raw), 7)] if not bundled else [raw[:-2], raw[-2:] + 'TRAILING']
    tok, ids = tokenizer_and_ids(pieces)
    steps = ids + [4, 1]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = tiny_model(conditional=True, vocab_size=len(tok))
        prompt = torch.tensor([[3, 2]])
        class Script(LogitsProcessor):
            def __call__(self, input_ids, scores):
                index = min(input_ids.shape[1]-prompt.shape[1], len(steps)-1)
                scores.fill_(-float('inf'))
                scores[:, steps[index]] = 0
                return scores
        outputs, criteria = [], []
        for fast in (False, True):
            criterion = criterion_class(kind)(tok, prompt.shape[1], fast_check=fast)
            with torch.inference_mode(), generation_context(model, generation_policy(tok), len(steps)) as cfg:
                output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                    generation_config=cfg, logits_processor=LogitsProcessorList([Script()]),
                    stopping_criteria=StoppingCriteriaList([criterion]))
            outputs.append(output)
            criteria.append(criterion)
        assert torch.equal(*outputs)
        assert outputs[0][0, prompt.shape[1]:].tolist() == ids
        assert output_of(criteria[0], kind) == output_of(criteria[1], kind) == text
        if not bundled:
            assert criteria[1].full_decodes < criteria[0].full_decodes
    finally:
        torch.set_num_threads(previous_threads)
