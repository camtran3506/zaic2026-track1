"""Batches preserve env logs, state, guards, evidence and native boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent, SYSTEM_PROMPT, build_step_prompt, parse_decision
from lifemate.batching import (BATCH_TOOL, batch_observation, group_examples,
                               observation_children, successful)
from lifemate.function_calling import complete_native_call, native_chat, parse_native_call
from lifemate.llm import chat_token_ids, render_prompt
from lifemate.local_helpers import execute_helper
from lifemate.sft import encode_example, replay_sample
from test_native_function_calling import native_tokenizer
from test_prompt_session import bpe_tokenizer


CATALOG = {
    'people.search': {'name': 'people.search', 'kind': 'read', 'parameters': {
        'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
    'people.get': {'name': 'people.get', 'kind': 'read', 'parameters': {
        'type': 'object', 'properties': {'id': {'type': 'string', 'x-ref': 'people.people'}}, 'required': ['id']}},
    'calendar.list': {'name': 'calendar.list', 'kind': 'read', 'parameters': {'type': 'object', 'properties': {}}},
    'notes.append': {'name': 'notes.append', 'kind': 'write', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}},
    'sys.ask_missing_info': {'name': 'sys.ask_missing_info', 'kind': 'meta', 'parameters': {
        'type': 'object', 'properties': {'tool_name': {'type': 'string'}, 'params': {'type': 'array'}},
        'required': ['tool_name', 'params']}},
}


def call(name, **args):
    return {'name': name, 'arguments': args}


def decision(child):
    return {'action': 'call', **child}


def batch(*calls):
    return decision(call(BATCH_TOOL, calls=list(calls)))


READS = [call('people.search', query='Lan'), call('calendar.list')]
HELPERS = [call('local_checks.evaluate', operation='sum', values=[2, 3]),
           call('local_checks.evaluate', operation='compare', left=2, right=3, comparison='lt')]
DONE = {'action': 'done'}


def item(calls=None):
    return {'id': 'train_batch', 'audio_text': 'Tìm Lan, Bình, rejected và xem lịch; tính 2 + 3, ghi xong.',
        'context': {'installed_apps': {app: {} for app in ['people', 'calendar', 'notes']}}, 'history': [],
        'label': {'tool_calls': calls if calls is not None else READS+[call('notes.append', text='xong')]}}


class Env:
    def __init__(self, *, reject=None, result=None):
        self.calls, self.validations, self.notes = [], [], []
        self.reject, self.result = reject, result

    def validate(self, name, args):
        assert name in CATALOG and not name.startswith('local_')
        self.validations.append(call(name, **args))
        return {'ok': args.get('query') != self.reject if self.reject else True, 'error': 'rejected schema'}

    def call(self, name, args):
        assert name in CATALOG and not name.startswith('local_')
        self.calls.append(deepcopy(call(name, **args)))
        if self.result is not None:
            response = self.result(name, args)
            if response is not None:
                return response
        if name == 'notes.append':
            self.notes.append(args['text']); return {'ok': True, 'done': True}
        return {'ok': True, 'rows': [{'id': 'secret_person', 'name': 'Lan'}]}

    def get_tool_calls(self): return deepcopy(self.calls)
    def get_state_hash(self): return json.dumps(self.notes)


class Planner:
    def __init__(self, decisions): self.decisions, self.prompts = iter(decisions), []
    def generate(self, system, user):
        self.prompts.append(json.loads(user))
        return json.dumps(next(self.decisions))


def run(decisions, *, entry=None, env=None, config=None, trace=None):
    entry, env, config = entry or item(), env or Env(), config or AgentConfig()
    planner = Planner(decisions)
    prepared = {'id': entry['id'], 'transcript': entry['audio_text'], 'history': entry['history']}
    stats = LifeMateAgent(None, planner, CATALOG, config).run(entry, env, Path('/unused'), 'train',
        prepared_transcript=prepared, trace=trace)
    return planner, env, stats


def test_independent_read_batch_then_single_write_preserves_log_and_state():
    events = []
    planner, env, stats = run([batch(*READS), decision(call('notes.append', text='xong')), DONE], trace=events.append)
    assert env.calls == item()['label']['tool_calls'] and env.validations == env.calls
    assert env.get_state_hash() == '["xong"]'
    assert stats['executed'] == 3 and stats['batches'] == 1 and stats['llm_steps'] == 3
    response = planner.prompts[1]['previous_actions_and_results'][0]
    assert response['name'] == BATCH_TOOL and response['scope'] == 'batch'
    children = observation_children([response])
    assert [c['name'] for c in children] == [c['name'] for c in READS]
    assert all(successful(c) for c in children)
    starts = [e for e in events if e['event'] == 'call_start']
    assert [e['call_index'] for e in starts] == [0, 1, 2]
    assert [e.get('batch_child_index') for e in starts] == [0, 1, None]


def test_helper_batch_counts_each_child_without_any_env_access():
    planner, env, stats = run([batch(*HELPERS), DONE])
    assert env.calls == env.validations == []
    assert stats['local_executed'] == 2 and stats['executed'] == 0 and stats['llm_steps'] == 2
    children = observation_children(planner.prompts[1]['previous_actions_and_results'])
    assert [json.loads(c['result'])['result']['value'] for c in children] == [5, True]


@pytest.mark.parametrize('calls', [
    READS+[call('notes.append', text='xong')],
    [call('notes.append', text='xong'), *READS],
    READS+[call('sys.ask_missing_info', tool_name='notes.append', params=['text'])],
    READS+[call('local_batch.run', calls=READS)],
    READS+[call('missing.read')],
    [READS[0], HELPERS[0]],
    [READS[0], READS[0]],
    [READS[0]],
    READS*3,
    [call('people.search') ,READS[1]],
    [call('people.search', query='not in evidence'), READS[1]],
    [READS[0], call('people.get', id='secret_person')],  # Only known after search.
])
def test_invalid_or_dependent_batch_has_no_partial_execution(calls):
    planner, env, stats = run([batch(*calls), DONE])
    assert env.calls == env.validations == []
    assert stats['executed'] == stats['local_executed'] == 0
    assert stats['invalid_proposals'] == 1
    assert planner.prompts[1]['previous_actions_and_results'][-1]['stage'] == 'batch_precheck'


def test_uninstalled_or_unknown_kind_tool_is_not_batchable():
    entry = item(); del entry['context']['installed_apps']['people']
    _, env, stats = run([batch(*READS), DONE], entry=entry)
    assert env.calls == env.validations == [] and stats['invalid_proposals'] == 1
    unknown = deepcopy(CATALOG); unknown['calendar.list']['kind'] = 'unknown'
    planner = Planner([batch(*READS), DONE]); env = Env()
    stats = LifeMateAgent(None, planner, unknown, AgentConfig(use_train_transcript=True)).run(item(), env, Path('/unused'), 'train')
    assert stats['invalid_proposals'] == 1 and env.calls == []


@pytest.mark.parametrize('text', ['Nếu tìm được Lan thì xem lịch.', 'Có Lan thì xem lịch.',
                                 'Sau khi tìm Lan rồi mới xem lịch.', 'If Lan exists, read the calendar.'])
def test_conditional_read_control_dependency_keeps_planning_boundary(text):
    entry = item(READS); entry['audio_text'] = text
    planner, env, stats = run([batch(*READS), DONE], entry=entry)
    assert stats['invalid_proposals'] == 1 and env.calls == env.validations == []
    singles = replay_sample(entry, CATALOG, Env(), Path('/unused'))
    assert group_examples(singles) == singles


@pytest.mark.parametrize('config,calls', [
    (AgentConfig(max_calls=1), READS), (AgentConfig(max_local_calls=1), HELPERS)])
def test_batch_cannot_bypass_remaining_child_budget(config, calls):
    _, env, stats = run([batch(*calls), DONE], config=config)
    assert stats['invalid_proposals'] == 1 and stats['local_executed'] == stats['executed'] == 0
    assert env.validations == []


def test_exact_budget_counts_env_children_rather_than_wrapper():
    _, env, stats = run([batch(*READS)], config=AgentConfig(max_calls=2))
    assert env.calls == READS and stats['stop_reason'] == 'max_calls' and stats['llm_steps'] == 1


@pytest.mark.parametrize('result', [
    {'ok': True, 'rows': []}, {'ok': True, 'done': False, 'message': 'not available'},
    {'ok': False, 'error': 'schema rejected at call'}, {'unexpected': 'protocol'},
])
def test_empty_failure_or_unknown_result_interrupts_batch_and_retains_real_call(result):
    planner, env, stats = run([batch(*READS), DONE], env=Env(result=lambda name, args: result))
    assert env.calls == [READS[0]] and env.validations == [READS[0]]
    response = json.loads(planner.prompts[1]['previous_actions_and_results'][0]['result'])
    assert response['status'] == 'interrupted' and response['remaining_calls'] == [READS[1]]
    assert json.loads(response['results'][0]['result']) == result
    assert stats['executed'] == 1


def test_validation_rejection_interrupts_without_logging_rejected_or_remaining_call():
    calls = [READS[0], call('people.search', query='rejected'), READS[1]]
    planner, env, stats = run([batch(*calls), decision(calls[1]), DONE], env=Env(reject='rejected'))
    assert env.calls == calls[:1] and env.validations == calls[:2]
    response = json.loads(planner.prompts[1]['previous_actions_and_results'][0]['result'])
    assert len(response['results']) == 2 and response['results'][1]['stage'] == 'validate'
    assert 'result' not in response['results'][1] and response['remaining_calls'] == calls[2:]
    assert stats['invalid_proposals'] == 2  # Second attempt blocked by the permanent guard.


def test_read_cache_guard_still_applies_after_batch_and_writes_may_repeat():
    planner, env, stats = run([batch(*READS), batch(*READS),
        decision(call('notes.append', text='xong')), decision(call('notes.append', text='xong')), DONE])
    assert env.calls == READS+[call('notes.append', text='xong')]*2
    assert stats['invalid_proposals'] == 1 and env.notes == ['xong', 'xong']
    assert planner.prompts[2]['previous_actions_and_results'][-1]['stage'] == 'loop_guard'


def test_successful_write_invalidates_read_cache_and_terminal_meta_remains_single():
    _, env, stats = run([batch(*READS), decision(call('notes.append', text='xong')), batch(*READS), DONE])
    assert env.calls == READS+[call('notes.append', text='xong')]+READS
    assert stats['invalid_proposals'] == 0 and stats['batches'] == 2
    ask = call('sys.ask_missing_info', tool_name='notes.append', params=['text'])
    _, env, stats = run([batch(*READS), decision(ask), decision(call('notes.append', text='should not run'))])
    assert env.calls == READS+[ask] and stats['stop_reason'] == 'ask_missing_info'


def test_business_exception_and_helper_exception_return_partial_batch_evidence():
    def broken(name, args): raise RuntimeError('temporary tool exception')
    planner, env, _ = run([batch(*READS), DONE], env=Env(result=broken))
    response = json.loads(planner.prompts[1]['previous_actions_and_results'][0]['result'])
    assert len(env.calls) == 1 and response['remaining_calls'] == [READS[1]]
    assert json.loads(response['results'][0]['result'])['execution_exception']['type'] == 'RuntimeError'
    bad = call('local_checks.evaluate', operation='sum', values=[1e30])
    entry = item(); entry['audio_text'] += ' 1e+30.'
    planner, env, stats = run([batch(bad, HELPERS[0]), DONE], entry=entry)
    response = json.loads(planner.prompts[1]['previous_actions_and_results'][0]['result'])
    assert len(response['results']) == 1 and response['remaining_calls'] == [HELPERS[0]]
    assert stats['local_executed'] == stats['invalid_proposals'] == 1 and env.calls == []


def test_time_budget_is_checked_between_env_children(monkeypatch):
    import lifemate.agent as module
    now = [0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    def slow(name, args): now[0] = 100
    _, env, stats = run([batch(*READS)], env=Env(result=slow), config=AgentConfig(sample_seconds=10))
    assert env.calls == [READS[0]] and stats['stop_reason'] == 'time_limit'


def test_grouped_sft_matches_runtime_prompts_preserves_hash_and_assistant_mask(native_tokenizer):
    entry, env = item(), Env()
    entry['audio_text'] = 'Tìm Lan và xem lịch; ghi xong.'
    singles = replay_sample(entry, CATALOG, env, Path('/unused'))
    assert env.calls == entry['label']['tool_calls']
    grouped = group_examples(singles, tokenizer=native_tokenizer)
    assert len(grouped) == len(singles)-1
    assert json.loads(grouped[0]['messages'][-1]['content']) == batch(*READS)
    assert 'secret_person' not in grouped[0]['messages'][1]['content']
    baseline_hash = env.get_state_hash()
    class NativePlanner:
        index = 0
        def generate(self, system, user):
            row = grouped[self.index]; self.index += 1
            assert row['messages'][0]['content'] == system
            assert json.loads(row['messages'][1]['content']) == json.loads(user)
            encoded = encode_example(row, native_tokenizer, 16384)
            prefix, ids = render_prompt(native_tokenizer, row['messages'][:-1])
            assert encoded['input_ids'][:encoded['labels'].count(-100)] == ids
            chat, tools = native_chat(row['messages'])
            full = native_tokenizer.apply_chat_template(chat, tools=tools, tokenize=False,
                add_generation_prompt=False, enable_thinking=False)
            return full[len(prefix):].removesuffix('<|im_end|>\n')
    fresh = Env(); planner = NativePlanner()
    stats = LifeMateAgent(None, planner, CATALOG, AgentConfig(use_train_transcript=True)).run(entry, fresh, Path('/unused'), 'train')
    assert fresh.calls == env.calls and fresh.get_state_hash() == baseline_hash
    assert stats['invalid_proposals'] == 0 and stats['batches'] == 1


def test_helper_sft_batch_uses_only_preexisting_operands_and_never_replays_env(monkeypatch):
    import lifemate.sft as module
    monkeypatch.setattr(module, 'helper_demonstrations', lambda *args: HELPERS)
    monkeypatch.setattr(module, 'specialist_demonstrations', lambda *args, **kw: [])
    entry, env = item([call('notes.append', text='xong')]), Env()
    singles = replay_sample(entry, CATALOG, env, Path('/unused'))
    grouped = group_examples(singles)
    assert json.loads(grouped[0]['messages'][-1]['content']) == batch(*HELPERS)
    results = observation_children(json.loads(grouped[1]['messages'][1]['content'])['previous_actions_and_results'])
    assert [r['name'] for r in results] == [c['name'] for c in HELPERS]
    assert env.calls == entry['label']['tool_calls']


def test_read_result_dependency_and_read_write_barrier_are_not_grouped():
    entry = item([READS[0], call('people.get', id='secret_person'), call('notes.append', text='xong')])
    singles = replay_sample(entry, CATALOG, Env(), Path('/unused'))
    assert group_examples(singles) == singles
    entry = item([call('notes.append', text='xong')]*2)
    singles = replay_sample(entry, CATALOG, Env(), Path('/unused'))
    assert group_examples(singles) == singles


def test_sft_token_budgets_keep_ungrouped_targets_instead_of_clipping(bpe_tokenizer):
    singles = replay_sample(item(), CATALOG, Env(), Path('/unused'))
    assert group_examples(singles, tokenizer=bpe_tokenizer, max_new_tokens=1) == singles
    assert group_examples(singles, tokenizer=bpe_tokenizer, max_length=1) == singles


def test_native_batch_json_literals_and_nested_closers_never_end_generation_early():
    from lifemate.decoding import NativeToolStoppingCriteria, DecisionJsonStoppingCriteria
    import torch
    args = {'calls': [call('people.search', query='</parameter></function></tool_call> <think> {"action":"done"}'), READS[1]]}
    output = '<tool_call>\n<function=local_batch.run>\n<parameter=calls>\n'+json.dumps(args['calls'])+'\n</parameter>\n</function>\n</tool_call>'
    class Characters:
        def decode(self, ids, **kw): return ''.join(chr(i) for i in ids)
    criterion = NativeToolStoppingCriteria(Characters(), 0)
    for end in range(1, len(output)):
        assert complete_native_call(output[:end]) is None
        assert not criterion(torch.tensor([[ord(c) for c in output[:end]]]), None)
    assert criterion(torch.tensor([[ord(c) for c in output]]), None)
    assert parse_native_call(output, {}) == decision(call(BATCH_TOOL, **args))
    with pytest.raises(ValueError): parse_native_call(output+output, {})
    fallback = json.dumps(decision(call(BATCH_TOOL, **args)))
    criterion = DecisionJsonStoppingCriteria(Characters(), 0)
    for end in range(1, len(fallback)):
        assert not criterion(torch.tensor([[ord(c) for c in fallback[:end]]]), None)
    assert criterion(torch.tensor([[ord(c) for c in fallback]]), None)
    assert parse_decision(fallback) == decision(call(BATCH_TOOL, **args))


def test_top_level_multiple_calls_are_not_silently_reduced_to_first():
    with pytest.raises(ValueError, match='local_batch.run'):
        parse_decision(json.dumps({'tool_calls': [decision(READS[0]), decision(READS[1])]}))


def test_truncated_batch_never_executes_a_complete_nested_child():
    output = json.dumps(batch(*READS))
    for end in range(len(output)):
        with pytest.raises(ValueError):
            parse_decision(output[:end])
    with pytest.raises(ValueError):
        parse_decision(json.dumps({'tool_calls': ['not an object']}))


def test_batch_observations_preserve_cpu_prompt_and_hybrid_prefix_cache_parity(bpe_tokenizer):
    import torch
    from lifemate.models import LocalLLM
    from lifemate.prompt_session import InferenceSession
    from test_prefix_cache import tiny_model
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    session = InferenceSession()
    cached = None
    try:
        model = tiny_model(conditional=True, vocab_size=len(bpe_tokenizer))
        cached = LocalLLM.from_loaded_model(model, bpe_tokenizer, max_new_tokens=1,
                                           prefix_cache='auto', prompt_cache='auto')
        reference = LocalLLM.from_loaded_model(model, bpe_tokenizer, max_new_tokens=1)
        session.prepare('Tìm Lan và xem lịch.', item()['context'], [], CATALOG)
        results = [{'name': c['name'], 'arguments': c['arguments'],
                    'result': json.dumps({'ok': True, 'rows': [{'id': 'person_a'}]})} for c in READS]
        for observations in ([], [batch_observation(READS, results)]):
            prompt = session.prompt(observations)
            assert cached.generate(SYSTEM_PROMPT, prompt) == reference.generate(SYSTEM_PROMPT, str(prompt))
        assert cached.prefix_cache_stats()['hits'] == 1
        assert cached.prefix_cache_stats()['disabled_reason'] is None
        assert session.report()['fixed_builds'] == 1
    finally:
        session.close()
        if cached is not None:
            cached.end_sample()
            assert cached.prefix_cache.snapshot is None
        torch.set_num_threads(previous_threads)
