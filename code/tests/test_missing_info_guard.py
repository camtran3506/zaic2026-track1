"""Visible-evidence ask guard and parity of training/inference execution guards."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.execution_policy import LoopMonitor
from lifemate.function_calling import native_chat
from lifemate.local_checks import inspect_call
from lifemate.missing_info_guard import check_missing_info
from lifemate.recovery import build_guard_recovery_examples
from lifemate.sft import GoldGuardConflict, encode_example, replay_sample, replay_synthetic_sample
from lifemate.tool_execution import precheck_call
from test_baseline import CATALOG, StubLLM
from test_prompt_session import bpe_tokenizer


def tool(name, props, required=(), kind='write', **schema):
    return {'name': name, 'kind': kind, 'parameters': {
        'type': 'object', 'properties': props, 'required': list(required), **schema}}


TOOLS = {
    **CATALOG,
    'alarm.create': tool('alarm.create', {'time': {'type': 'string', 'format': 'date-time'},
                                         'label': {'type': 'string'}}, ['time']),
    'contacts.search': tool('contacts.search', {'query': {'type': 'string'}}, kind='read'),
    'phone.call': tool('phone.call', {'contact_id': {'type': 'string', 'x-ref': 'contacts.contacts'},
                                    'phone': {'type': 'string'}}, **{'x-one-of': [['contact_id'], ['phone']]}),
    'ewallet.transfer': tool('ewallet.transfer', {
        'contact_id': {'type': 'string', 'x-ref': 'contacts.contacts'}, 'amount': {'type': 'integer'}},
        ['contact_id', 'amount']),
    'new_app.search': tool('new_app.search', {'query': {'type': 'string'}}, ['query'], kind='read'),
    'new_app.order': tool('new_app.order', {'product_id': {'type': 'string', 'x-ref': 'new_app.products'},
                                          'quantity': {'type': 'integer', 'minimum': 1}}, ['product_id', 'quantity']),
}
TIME = {'current_time': '2026-11-27T07:30:00+07:00', 'timezone': 'Asia/Ho_Chi_Minh'}
REQUEST = 'Gọi cho Minh luôn đi rồi đặt báo thức sáu giờ sáng mai giúp tôi với.'


def check(target, params, **kwargs):
    return check_missing_info({'tool_name': target, 'params': params}, TOOLS, **kwargs)


def record(calls, transcript, context=None):
    return {'id': 'train_guard', 'audio_text': transcript, 'history': [],
            'context': {**TIME, **(context or {}),
                        'installed_apps': {n.split('.')[0]: {} for n in TOOLS if not n.startswith('sys.')}},
            'label': {'tool_calls': calls}}


class Env:
    def __init__(self, result=None):
        self.calls, self.validations = [], []
        self.result = result or {'ok': True, 'done': True}

    def validate(self, name, args):
        self.validations.append({'name': name, 'arguments': deepcopy(args)})
        return {'ok': inspect_call(name, args, TOOLS)['status'] == 'valid'}

    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': deepcopy(args)})
        return deepcopy(self.result)

    def get_tool_calls(self):
        return self.calls


@pytest.mark.parametrize('params', [[], ['time', 'time'], 'time', [None]])
def test_malformed_missing_params_rejected(params):
    assert check('alarm.create', params)['reason'] == 'invalid_missing_info_params'


def test_wrong_parameter_names_rejected_before_environment_validation():
    args = {'tool_name': 'alarm.create', 'params': ['alarm_time']}
    feedback = precheck_call('sys.ask_missing_info', args, TOOLS, TOOLS, LoopMonitor())
    assert feedback['stage'] == 'missing_info_guard'
    assert feedback['feedback']['unknown_params'] == ['alarm_time']
    assert 'time' in feedback['feedback']['available_params']
    assert 'result' not in feedback


def test_known_time_is_bound_to_its_own_task_and_partial_asks_are_identified():
    feedback = check('alarm.create', ['time', 'label'], transcript=REQUEST, context=TIME)
    assert feedback['known_arguments'] == {'time': '2026-11-28T06:00:00+07:00'}
    assert feedback['unresolved_params'] == ['label']


@pytest.mark.parametrize('transcript', [
    'Đặt báo thức giúp tôi.', 'Đặt báo thức sáu giờ sáng.',
    'Đặt báo thức sáng mai lúc sáu hoặc bảy giờ.',
    'Đặt báo thức nếu mai sáu giờ trời mưa.',
    'Đặt báo thức sáu giờ sáng mai và cuộc họp tám giờ sáng ngày kia.',
    'Mai gọi cho Minh rồi đặt báo thức.',
])
def test_unknown_ambiguous_and_other_tasks_times_can_still_be_asked(transcript):
    assert check('alarm.create', ['time'], transcript=transcript, context=TIME) is None


def test_no_context_time_is_invented():
    assert check('alarm.create', ['time'], transcript='Đặt báo thức sáu giờ sáng mai') is None


def test_schema_values_are_known_but_optional_fields_are_not_blanket_rejected():
    tools = {**TOOLS, 'new_app.test': tool('new_app.test', {
        'a': {'type': 'string', 'const': 'a'}, 'b': {'type': 'integer', 'default': 1},
        'c': {'type': 'string', 'enum': ['single']}, 'd': {'type': 'string', 'enum': ['x', 'y']},
        'optional': {'type': 'string'}, 'example': {'type': 'string', 'examples': ['not_evidence']}})}
    ask = {'tool_name': 'new_app.test', 'params': ['a', 'b', 'c']}
    feedback = check_missing_info(ask, tools, transcript='Dùng giá trị mặc định')
    assert feedback['known_arguments'] == {'a': 'a', 'b': 1, 'c': 'single'}
    assert check_missing_info({'tool_name': 'new_app.test', 'params': ['b']}, tools) is None
    assert check_missing_info({'tool_name': 'new_app.test', 'params': ['d', 'optional', 'example']}, tools) is None


def contact_result(*rows, **status):
    return {'ok': True, 'contacts': list(rows), **status}


def test_exact_entity_can_come_from_context_history_or_current_successful_read():
    result = contact_result({'id': 'c_lan', 'name': 'Lan', 'phone': '0901234567'})
    call = {'name': 'contacts.search', 'arguments': {}}
    for evidence in (
        {'context': {'collections': {'contacts.contacts': result['contacts']}}},
        {'history': [{'role': 'assistant', 'tool_calls': [call], 'tool_results': [result]}]},
        {'observations': [{**call, 'result': json.dumps(result)}]},
    ):
        feedback = check('phone.call', ['contact_id'], transcript='Gọi cho Lan', **evidence)
        assert feedback['known_arguments']['contact_id'] == 'c_lan'
        # A known one-of alternative also prevents asking for the other member.
        feedback = check('phone.call', ['phone'], transcript='Gọi cho Lan', **evidence)
        assert feedback['unresolved_params'] == []


@pytest.mark.parametrize('rows, transcript', [
    ([{'id': '1', 'name': 'Lan'}, {'id': '2', 'name': 'Lan'}], 'Gọi Lan'),
    ([{'id': '1', 'name': 'Lộc'}, {'id': '2', 'name': 'Anh Lộc'}], 'Gọi Lộc'),
    ([{'id': '1', 'name': 'Sarah Phan'}, {'id': '2', 'name': 'Sarah Lâm'}], 'Chuyển trả Sarah phần của chị'),
    ([{'id': '1', 'name': 'Lan'}], 'Đừng gọi Lan'),
    ([{'id': '1', 'name': 'Lan'}], 'Gọi Nam'),
])
def test_ambiguous_negated_or_unrelated_entities_are_not_resolved(rows, transcript):
    assert check('phone.call', ['contact_id'], transcript=transcript,
                 context={'collections': {'contacts.contacts': rows}}) is None


def test_name_diacritics_do_not_turn_ordinary_words_into_entity_matches():
    rows = [{'id': '1', 'name': 'Sarah Phan'}, {'id': '2', 'name': 'Sarah Lâm'}]
    assert check('ewallet.transfer', ['contact_id'], transcript='Chuyển ví trả Sarah phần của chị',
                 context={'collections': {'contacts.contacts': rows}}) is None


def test_contact_from_another_task_does_not_resolve_phone_recipient():
    rows = [{'id': '1', 'name': 'Lan'}]
    assert check('phone.call', ['contact_id'], transcript='Nhắn Lan rồi gọi chú đó',
                 context={'collections': {'contacts.contacts': rows}}) is None


@pytest.mark.parametrize('result', [contact_result({'id': 'c_lan', 'name': 'Lan'}, ok=False),
                                    contact_result({'id': 'c_lan', 'name': 'Lan'}, done=False)])
def test_failed_or_rejected_read_is_not_entity_evidence(result):
    obs = [{'name': 'contacts.search', 'arguments': {}, 'result': json.dumps(result)},
           {'stage': 'validate', 'name': 'phone.call', 'arguments': {'contact_id': 'c_lan'}}]
    assert check('phone.call', ['contact_id'], transcript='Gọi Lan', observations=obs) is None


def test_current_full_read_supersedes_older_history_snapshot():
    old = {'role': 'assistant', 'tool_calls': [{'name': 'contacts.search', 'arguments': {}}],
           'tool_results': [contact_result({'id': 'c_lan', 'name': 'Lan'})]}
    current = {'name': 'contacts.search', 'arguments': {}, 'result': json.dumps(contact_result())}
    assert check('phone.call', ['contact_id'], transcript='Gọi Lan', history=[old],
                 observations=[current]) is None


def test_inference_rejects_redundant_ask_and_continues_without_executing_it():
    ask = {'action': 'call', 'name': 'sys.ask_missing_info',
           'arguments': {'tool_name': 'alarm.create', 'params': ['time']}}
    write = {'action': 'call', 'name': 'alarm.create', 'arguments': {'time': '2026-11-28T06:00:00+07:00'}}
    prompts = []

    class Planner(StubLLM):
        def generate(self, system, user):
            prompts.append(user)
            return super().generate(system, user)

    entry = record([], REQUEST)
    prepared = {'id': entry['id'], 'transcript': REQUEST, 'history': []}
    env = Env()
    report = LifeMateAgent(None, Planner([json.dumps(ask), json.dumps(write), '{"action":"done"}']),
                          TOOLS, AgentConfig()).run(entry, env, Path('/unused'), 'train', prepared)
    assert report['invalid_proposals'] == 1 and report['executed'] == 1
    assert report['stop_reason'] == 'done'
    assert env.calls == env.validations == [{'name': write['name'], 'arguments': write['arguments']}]
    chat, _ = native_chat([{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompts[1]}])
    feedback = json.loads(chat[-1]['content'])['agent_feedback']
    assert feedback['stage'] == 'missing_info_guard'
    assert not any(m.get('tool_calls') for m in chat)


def test_sft_rejects_same_redundant_ask_and_keeps_genuine_missing_info():
    ask = {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'alarm.create', 'params': ['time']}}
    env = Env()
    with pytest.raises(GoldGuardConflict, match='information_already_available'):
        replay_sample(record([ask], REQUEST), TOOLS, env, Path('/unused'), include_date_helpers=False)
    assert not env.calls and not env.validations
    rows = replay_sample(record([ask], 'Đặt báo thức'), TOOLS, env, Path('/unused'), include_date_helpers=False)
    assert json.loads(rows[-1]['messages'][-1]['content'])['name'] == ask['name']
    assert env.calls == [ask]


@pytest.mark.parametrize('result', [
    {'ok': True, 'products': []},
    {'ok': True, 'done': False, 'message': 'not found'},
])
def test_sft_and_inference_allow_distinct_empty_reads_but_block_exact_repeats(result):
    calls = [{'name': 'new_app.search', 'arguments': {'query': q}}
             for q in ('milk', 'rice', 'eggs', 'bread')]
    entry = record(calls, 'Tìm sản phẩm')
    env = Env(result)
    replay_sample(entry, TOOLS, env, Path('/unused'), include_date_helpers=False)
    assert env.calls == env.validations == calls
    duplicate_entry = record(calls + [calls[0]], 'Tìm sản phẩm')
    duplicate_env = Env(result)
    with pytest.raises(GoldGuardConflict, match='repeated_(empty|failed)_read'):
        replay_sample(duplicate_entry, TOOLS, duplicate_env, Path('/unused'), include_date_helpers=False)
    assert duplicate_env.calls == duplicate_env.validations == calls
    synthetic_entry = deepcopy(entry)
    synthetic_entry['id'] = 'synthetic_guard'
    synthetic_entry['label']['tool_results'] = [deepcopy(result) for _ in calls]
    replay_synthetic_sample(synthetic_entry, TOOLS, Path('/unused'), include_date_helpers=False)
    inference_env = Env(result)
    outputs = [json.dumps({'action': 'call', **call}) for call in calls + [calls[0]]]
    outputs += ['{"action":"done"}']
    prepared = {'id': entry['id'], 'transcript': entry['audio_text'], 'history': []}
    report = LifeMateAgent(None, StubLLM(outputs), TOOLS, AgentConfig()).run(
        entry, inference_env, Path('/unused'), 'train', prepared)
    assert inference_env.calls == inference_env.validations == calls
    assert report['executed'] == len(calls) and report['invalid_proposals'] == 1
    assert report['stop_reason'] == 'done'


@pytest.mark.parametrize('result', [{'ok': True, 'products': []},
                                   {'ok': False, 'products': [{'id': 'p1'}]}])
def test_recorded_synthetic_cannot_use_future_or_failed_result_ids(result):
    entry = record([{'name': 'new_app.search', 'arguments': {'query': 'sữa'}},
                    {'name': 'new_app.order', 'arguments': {'product_id': 'p1', 'quantity': 1}}], 'Đặt sữa')
    entry.update(id='synthetic_guard')
    entry['label']['tool_results'] = [result, {'ok': True, 'product_id': 'p1'}]
    with pytest.raises(GoldGuardConflict, match='xref_guard'):
        replay_synthetic_sample(entry, TOOLS, Path('/unused'), include_date_helpers=False)


def test_redundant_ask_recovery_supervises_only_the_correct_action(bpe_tokenizer):
    call = {'name': 'alarm.create', 'arguments': {'time': '2026-11-28T06:00:00+07:00'}}
    entry = record([call], REQUEST)
    prepared = {'id': entry['id'], 'transcript': REQUEST, 'history': [], 'transcript_source': 'audio_text'}
    env = Env()
    steps = replay_sample(entry, TOOLS, env, Path('/unused'), prepared_transcript=prepared,
                          include_date_helpers=False)
    rows = build_guard_recovery_examples(entry, prepared, steps, TOOLS, env, ratio=1.0)
    row, = rows
    assert row['recovery']['kind'] == 'guard_missing_info'
    assert json.loads(row['messages'][-1]['content']) == {'action': 'call', **call}
    chat, _ = native_chat(row['messages'])
    assert json.loads(chat[-2]['content'])['agent_feedback']['stage'] == 'missing_info_guard'
    assistants = [m for m in chat if m.get('tool_calls')]
    assert len(assistants) == 1
    assert assistants[0]['tool_calls'][0]['function']['name'] == call['name']
    encoded = encode_example(row, bpe_tokenizer, 16384)
    assert any(label != -100 for label in encoded['labels'])
    assert env.calls == [call]  # Recovery does not execute a fake ask or another write.


@pytest.mark.parametrize('policy', ['skip', 'error'])
def test_prepare_reports_whole_guard_conflicts_without_changing_gold(tmp_path, monkeypatch, policy):
    import prepare_sft
    good = record([{'name': 'alarm.create', 'arguments': {'time': '2026-11-28T06:00:00+07:00'}}], REQUEST)
    bad = record([{'name': 'new_app.order', 'arguments': {'product_id': 'unseen', 'quantity': 1}}], 'Đặt sữa')
    bad['id'] = 'train_conflict'
    source = tmp_path / 'train'
    source.mkdir()
    data = ''.join(json.dumps(r) + '\n' for r in (good, bad))
    (source / 'train.jsonl').write_text(data)
    (source / 'train_tools_catalog.json').write_text(json.dumps({'apps': [{'tools': list(TOOLS.values())}]}))
    monkeypatch.setitem(sys.modules, 'zaic_tools', SimpleNamespace(ToolEnv=lambda *a, **kw: Env()))
    output = tmp_path / 'sft'
    args = ['prepare_sft.py', '--data-root', str(source), '--output-dir', str(output),
            '--asr-transcript-ratio', '0', '--no-inject-recovery', '--no-inject-guard-recovery']
    if policy == 'error':
        args += ['--guard-conflict-policy', 'error']
    monkeypatch.setattr(sys, 'argv', args)
    if policy == 'error':
        with pytest.raises(GoldGuardConflict, match='xref_guard'):
            prepare_sft.main()
    else:
        prepare_sft.main()
        rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
        assert rows and {r['sample_id'] for r in rows} == {good['id']}
        skipped, = [json.loads(line) for line in (output / 'train_skipped.jsonl').read_text().splitlines()]
        assert skipped['sample_id'] == bad['id']
        assert skipped['category'] == 'automatic_guard_conflict'
        assert skipped['guard_feedback']['stage'] == 'xref_guard'
        manifest = json.loads((output / 'manifest.json').read_text())
        assert manifest['training_guards']['enforce_observed_xref'] is True
        assert manifest['training_guards']['empty_read_limit'] == 0
    assert (source / 'train.jsonl').read_text() == data
