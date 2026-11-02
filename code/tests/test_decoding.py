"""Validate complete decision boundaries and exercise real HF early stopping."""
import json

import pytest

from lifemate.decoding import complete_decision_json, DecisionJsonStoppingCriteria


@pytest.mark.parametrize('obj', [
    {'action': 'done'},
    {'action': 'call', 'name': 'home.write', 'arguments': {
        'nested': {'list': [1, {'text': 'a } { brace, "quote", \\ slash, tiếng Việt'}]},
        'literal': '<think>this is an argument, not a reasoning block'}},
    {'tool_calls': [{'name': 'home.write', 'arguments': {'x': {}}}]},
])
def test_only_stops_on_complete_outer_decision(obj):
    text = json.dumps(obj, ensure_ascii=False)
    for length in range(len(text)):
        assert complete_decision_json(text[:length]) is None
    assert complete_decision_json('```json\n' + text + '\n``` explanation') == text


def test_skips_reasoning_and_non_decision_objects():
    assert complete_decision_json('<think>{"action":"done"}') is None
    assert complete_decision_json('<think>{"action":"done"}</think> {"action":"call","name":"a.b","arguments":{}}') == '{"action":"call","name":"a.b","arguments":{}}'
    assert complete_decision_json('{"reason":"not a decision"} {"action":"done"}') == '{"action":"done"}'
    assert complete_decision_json('{"arguments":{"action":"done"}}') is None
    assert complete_decision_json('{"action":"call","name":"x","arguments":') is None


def test_hf_generate_stops_before_trailing_tokens_and_ignores_prompt_json():
    import torch
    from tokenizers import Tokenizer, models
    from transformers import (Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast,
                              LogitsProcessor, LogitsProcessorList, StoppingCriteriaList)
    vocabulary = {'[PAD]': 0, '[EOS]': 1, '{"action":': 2, '"done"': 3, '}': 4, 'TRAILING': 5}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocabulary, unk_token='[PAD]')),
                                       pad_token='[PAD]', eos_token='[EOS]')
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=6, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, eos_token_id=1, pad_token_id=0)).eval()
    prompt = torch.tensor([[2, 3, 4]])  # A complete JSON in the prompt must NOT stop output.
    class ScriptedTokens(LogitsProcessor):
        def __init__(self, prefix_length=prompt.shape[1]): self.prefix_length = prefix_length
        def __call__(self, input_ids, scores):
            step = input_ids.shape[1] - self.prefix_length
            token = [2, 3, 4, 5, 5, 1][min(step, 5)]
            scores.fill_(-float('inf'))
            scores[:, token] = 0
            return scores
    def generate(limit, criterion=None):
        return model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
            max_new_tokens=limit, pad_token_id=0,
            logits_processor=LogitsProcessorList([ScriptedTokens()]),
            stopping_criteria=StoppingCriteriaList([criterion]) if criterion else None)
    normal = generate(8)
    criterion = DecisionJsonStoppingCriteria(tokenizer, prompt.shape[1])
    early = generate(8, criterion)
    assert early[0, prompt.shape[1]:].tolist() == [2, 3, 4]
    assert normal[0, prompt.shape[1]:].tolist() == [2, 3, 4, 5, 5, 1]
    assert json.loads(criterion.completed_json) == {'action': 'done'}
    assert complete_decision_json(tokenizer.decode(normal[0, prompt.shape[1]:])) == criterion.completed_json
    incomplete = DecisionJsonStoppingCriteria(tokenizer, prompt.shape[1])
    assert generate(2, incomplete).shape[1] == prompt.shape[1] + 2
    assert incomplete.completed_json is None  # max_new_tokens remains the safety limit
    # Exercise the production wrapper too, not just HF's stopping API in isolation.
    from types import SimpleNamespace
    from lifemate.models import LocalLLM
    tokenizer.chat_template = "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
    produced = []
    def actual_generate(**kwargs):
        length = kwargs['input_ids'].shape[1]
        result = model.generate(**kwargs, logits_processor=LogitsProcessorList([ScriptedTokens(length)]))
        produced.append(result.shape[1] - length)
        return result
    llm = LocalLLM.from_loaded_model(SimpleNamespace(generate=actual_generate,
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(1))), tokenizer,
        max_input_tokens=6000, max_new_tokens=8)
    assert json.loads(llm.generate('system', 'user')) == {'action': 'done'}
    assert produced == [3]


@pytest.mark.parametrize('arguments', [None, {}, {'action': 'done'}])
def test_removed_action_does_not_trigger_decision_stopping(arguments):
    decision = {'action': 'inspect', 'name': 'home.read'}
    if arguments is not None:
        decision['arguments'] = arguments
    text = json.dumps(decision)
    for end in range(len(text) + 1):
        assert complete_decision_json(text[:end]) is None
