"""Native Qwen3.5 template, loss mask, wire parser and generation boundaries."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifemate.agent import SYSTEM_PROMPT, AgentConfig, LifeMateAgent, build_step_prompt, parse_decision
from lifemate.function_calling import complete_native_call, native_chat, parse_native_call
from lifemate.llm import chat_token_ids, configure_tokenizer, render_prompt
from lifemate.sft import encode_example, replay_sample
from test_baseline import CATALOG, FakeEnv, StubASR, item


TEMPLATE = Path(__file__).parent / 'fixtures/qwen35_chat_template.jinja'


@pytest.fixture
def native_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(models.WordLevel({'[UNK]': 0, '[EOS]': 1}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       eos_token='[EOS]', pad_token='[EOS]')
    tokenizer.chat_template = TEMPLATE.read_text()
    (tmp_path / 'config.json').write_text('{"model_type":"qwen3_5"}')
    return configure_tokenizer(tokenizer, tmp_path)


def portable(prompt, target=None):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]
    if target is not None:
        messages.append({'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)})
    return messages


def test_native_template_preserves_tools_history_and_results(native_tokenizer):
    catalog = deepcopy(CATALOG)
    catalog['smart_home.light_set']['description'] = 'Description! ' * 1000
    catalog['smart_home.light_set']['returns'] = {'type': 'object', 'x-details': 'metadata ' * 500}
    catalog['smart_home.light_set']['x-custom'] = [1, {'extra': True}]
    text = '  Giữ: </think> và <think>, 08:30!  ' * 100
    result = {'ok': True, 'data': 'Kết quả! ' * 1000}
    history = [{'role': 'assistant', 'text': text, 'tool_calls': [
        {'name': 'smart_home.get_status', 'arguments': {}}], 'tool_results': [result]}]
    history += [{'role': role, 'text': text} for role in ['system', 'user'] * 10]
    observations = [{'name': 'smart_home.get_status', 'arguments': {}, 'result': json.dumps(result)}] * 10
    observations.append({'name': 'smart_home.light_set', 'error': 'validation error ' * 1000})
    _, prompt = build_step_prompt('  BẬT đèn!  ', {'timezone': 'Asia/Ho_Chi_Minh'}, history, catalog, observations)
    original = deepcopy((catalog, history, observations))
    chat, tools = native_chat(portable(prompt))
    assert {t['function']['name']: t['function'] for t in tools} == catalog
    historical = [m for m in chat[1:] if m['role'] != 'tool'][:len(history)]
    assert [json.loads(m['content']) for m in historical] == [
        {'history_role': turn['role'], 'text': turn['text']} for turn in history]
    assert sum(m['role'] == 'tool' for m in chat) == 11
    assert all(json.loads(m['content']) == result for m in chat if m['role'] == 'tool')
    assert json.loads(chat[-1]['content']) == {'agent_feedback': observations[-1]}
    assert (catalog, history, observations) == original
    rendered, _ = render_prompt(native_tokenizer, portable(prompt))
    assert '<tools>' in rendered and '<tool_response>' in rendered
    assert '<function=smart_home.get_status>' in rendered
    assert '<|im_start|>tool' not in rendered  # Native template wraps results as tool_response.
    assert rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert '"action":"call"' not in rendered


@pytest.mark.parametrize('target', [
    {'action': 'call', 'name': 'smart_home.get_status', 'arguments': {}},
    {'action': 'call', 'name': 'smart_home.light_set', 'arguments': {'device_id': '001', 'power': 'on'}},
    {'action': 'done'},
])
def test_sft_native_target_and_inference_token_prefix_match(native_tokenizer, target):
    _, prompt = build_step_prompt('Bật đèn', {}, [], CATALOG, [])
    messages = portable(prompt, target)
    row = encode_example({'sample_id': 'train_native', 'step': 0, 'messages': messages}, native_tokenizer, 16384)
    prefix_length = row['labels'].count(-100)
    _, inference = render_prompt(native_tokenizer, messages[:-1])
    assert row['input_ids'][:prefix_length] == inference
    chat, tools = native_chat(messages)
    full = native_tokenizer.apply_chat_template(chat, tools=tools, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    prefix, _ = render_prompt(native_tokenizer, messages[:-1])
    assert full.startswith(prefix)
    completion = full[len(prefix):].removesuffix('<|im_end|>\n')
    if target['action'] == 'call':
        assert '<tool_call>' in completion and '<function=' in completion
        assert parse_decision(completion, CATALOG) == target
    else:
        assert json.loads(completion) == target
    assert row['input_ids'] == chat_token_ids(native_tokenizer, messages, False)
    assert all(label != -100 for label in row['labels'][prefix_length:])


def test_sft_and_rollout_share_native_messages_after_real_tool_results(native_tokenizer):
    entry = dict(item(), audio_text='Bật đèn', label={'tool_calls': [
        {'name': 'smart_home.get_status', 'arguments': {}},
        {'name': 'smart_home.light_set', 'arguments': {'device_id': 'light_living', 'power': 'on'}},
    ]})
    steps = replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))
    class NativePlanner:
        def __init__(self): self.count = 0
        def generate(self, system, user):
            step = steps[self.count]
            self.count += 1
            assert portable(user) == step['messages'][:-1]
            prefix, _ = render_prompt(native_tokenizer, portable(user))
            chat, tools = native_chat(step['messages'])
            full = native_tokenizer.apply_chat_template(chat, tools=tools, tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
            assert full.startswith(prefix)
            return full[len(prefix):].removesuffix('<|im_end|>\n')
    llm, env = NativePlanner(), FakeEnv()
    stats = LifeMateAgent(StubASR(), llm, CATALOG, AgentConfig(use_train_transcript=True)).run(
        entry, env, Path('/unused'), 'train')
    assert llm.count == 3 and stats['invalid_proposals'] == 0
    assert env.calls == entry['label']['tool_calls']


def test_native_argument_types_preserve_strings_and_nested_values(native_tokenizer):
    args = {'string': ' 001\n</think> & <literal> ', 'integer': 42, 'number': 2.5,
            'boolean': True, 'null': None, 'object': {'nested': ['x', 1, False]}, 'array': [1, {'x': 'y'}]}
    catalog = {'home.write': {'name': 'home.write', 'parameters': {
        'type': 'object', 'properties': {name: {'type': name} for name in args}}}}
    _, prompt = build_step_prompt('request', {}, [], catalog, [])
    target = {'action': 'call', 'name': 'home.write', 'arguments': args}
    chat, tools = native_chat(portable(prompt, target))
    text = native_tokenizer.apply_chat_template(chat, tools=tools, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    output = text[len(render_prompt(native_tokenizer, portable(prompt))[0]):].removesuffix('<|im_end|>\n')
    assert parse_native_call(output, catalog) == target
    for end in range(len(output)):
        assert complete_native_call(output[:end]) is None
    assert complete_native_call('<think>do not call yet</think>' + output) == output


@pytest.mark.parametrize('output', [
    '<tool_call><function=x><parameter=a>1</parameter></function>',
    '<tool_call><function=x><parameter=a>1</parameter><parameter=a>2</parameter></function></tool_call>',
    '<tool_call><function=x></function></tool_call><tool_call><function=y></function></tool_call>',
])
def test_malformed_duplicate_and_multiple_calls_fail_without_partial_execution(output):
    with pytest.raises(ValueError):
        parse_decision(output, {})


def test_native_stopping_waits_past_inner_json_and_keeps_special_tags(native_tokenizer):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, LogitsProcessor, LogitsProcessorList
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from lifemate.models import LocalLLM

    tokens = ['[UNK]', '[EOS]', '<tool_call>', '<function=home.write>', '<parameter=payload>',
              '{"action":"done","items":[1,2]}', '</parameter>', '</function>', '</tool_call>', 'TRAILING']
    backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', eos_token='[EOS]',
        pad_token='[EOS]', additional_special_tokens=['<tool_call>', '</tool_call>'])
    tokenizer.chat_template = native_tokenizer.chat_template
    tokenizer.lifemate_native_fc = True
    tokenizer.lifemate_template_kwargs = {'enable_thinking': False}
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokens), hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=4096, eos_token_id=1, pad_token_id=1)).eval()
    lengths = []
    class Script(LogitsProcessor):
        def __init__(self, prefix): self.prefix = prefix
        def __call__(self, ids, scores):
            sequence = [2, 3, 4, 5, 6, 7, 8, 9, 1]
            scores.fill_(-float('inf'))
            scores[:, sequence[min(ids.shape[1] - self.prefix, len(sequence) - 1)]] = 0
            return scores
    def generate(**kwargs):
        prefix = kwargs['input_ids'].shape[1]
        output = model.generate(**kwargs, logits_processor=LogitsProcessorList([Script(prefix)]))
        lengths.append(output.shape[1] - prefix)
        return output
    llm = LocalLLM.from_loaded_model(SimpleNamespace(generate=generate,
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1))), tokenizer,
        max_input_tokens=3000, max_new_tokens=16)
    catalog = {'home.write': {'name': 'home.write', 'parameters': {
        'type': 'object', 'properties': {'payload': {'type': 'object'}}}}}
    _, prompt = build_step_prompt('request', {}, [], catalog, [])
    raw = llm.generate(SYSTEM_PROMPT, prompt)
    assert lengths == [7] and raw.endswith('</tool_call>')
    assert parse_decision(raw, catalog)['arguments'] == {'payload': {'action': 'done', 'items': [1, 2]}}


def test_json_argument_literal_native_markers_are_not_treated_as_markup():
    target = {'action': 'call', 'name': 'home.write', 'arguments': {
        'text': '<tool_call><function=literal></function></tool_call>'}}
    assert parse_decision(json.dumps(target)) == target


def test_missing_fc_template_is_rejected_instead_of_dropping_tools(native_tokenizer):
    native_tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    _, prompt = build_step_prompt('request', {}, [], CATALOG, [])
    with pytest.raises(ValueError, match='must support tools and tool_calls'):
        render_prompt(native_tokenizer, portable(prompt))
