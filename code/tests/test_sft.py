import json
from pathlib import Path

import pytest

from lifemate.sft import encode_example, replay_sample, split_records, AssistantCollator
from test_baseline import CATALOG, FakeEnv, item


def labeled():
    entry = item()
    entry['audio_text'] = 'Bật đèn phòng khách'
    entry['label'] = {'tool_calls': [
        {'name': 'smart_home.get_status', 'arguments': {}},
        {'name': 'smart_home.light_set', 'arguments': {'device_id': 'light_living', 'power': 'on'}},
    ]}
    return entry


def test_gold_replay_observations_do_not_leak_future_targets():
    env = FakeEnv()
    steps = replay_sample(labeled(), CATALOG, env, Path('/unused'))
    assert len(steps) == 3
    payloads = [json.loads(s['messages'][1]['content']) for s in steps]
    assert payloads[0]['previous_actions_and_results'] == []
    assert 'light_living' not in steps[0]['messages'][1]['content']
    assert 'light_living' in payloads[1]['previous_actions_and_results'][0]['result']
    assert json.loads(steps[-1]['messages'][-1]['content']) == {'action': 'done'}
    assert len(env.calls) == 2


def test_schema_and_missing_labels_fail_instead_of_becoming_done():
    entry = labeled()
    entry.pop('label')
    with pytest.raises(ValueError, match='missing label'):
        replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))
    entry = labeled()
    entry['label']['tool_calls'][1]['arguments'] = {}
    with pytest.raises(ValueError, match='gold schema invalid'):
        replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))


def test_terminal_missing_info_does_not_add_done():
    entry = labeled()
    entry['label']['tool_calls'] = [{'name': 'sys.ask_missing_info',
        'arguments': {'tool_name': 'smart_home.light_set', 'params': ['device_id']}}]
    assert len(replay_sample(entry, CATALOG, FakeEnv(), Path('/unused'))) == 1


def test_split_is_grouped_deterministic_and_order_independent():
    records = [{'id': str(i)} for i in range(20)]
    train, valid = split_records(records)
    assert not {r['id'] for r in train} & {r['id'] for r in valid}
    assert (train, valid) == split_records(records[::-1])
    with pytest.raises(ValueError, match='Duplicate'):
        split_records([{'id': 'x'}, {'id': 'x'}])


class TemplateTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [1, 2, 3] if add_generation_prompt else [1, 2, 3, 7, 8, 9]


def test_loss_masks_prompt_and_padding_and_rejects_truncated_json():
    example = {'sample_id': 'train_1', 'step': 0, 'messages': [
        {'role': role, 'content': ''} for role in ('system', 'user', 'assistant')]}
    row = encode_example(example, TemplateTokenizer(), 6)
    assert row['labels'] == [-100, -100, -100, 7, 8, 9]
    with pytest.raises(ValueError, match='never truncate'):
        encode_example(example, TemplateTokenizer(), 5)
    short = {k: v[:-1] for k, v in row.items()}
    batch = AssistantCollator(0)([row, short])
    assert batch['labels'][1, -1].item() == -100
    assert batch['attention_mask'][1, -1].item() == 0
