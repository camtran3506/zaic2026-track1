from copy import deepcopy

import pytest

from lifemate.execution_policy import LoopMonitor, verify_result, check_tool_access


REJECTION = {'ok': False, 'error': 'missing required parameter'}


def test_verifier_distinguishes_business_noop_and_nested_lifecycle():
    assert verify_result('banking.transfer', {'ok': True, 'done': False})['status'] == 'failure'
    assert verify_result('app.call', {'ok': False, 'error': 'schema'})['status'] == 'failure'
    assert verify_result('reminders.create', {'ok': True, 'item': {'done': False}})['status'] == 'success'
    assert verify_result('app.call', {'ok': 1})['status'] == 'unknown'
    assert verify_result('app.call', None)['status'] == 'unknown'


def test_empty_is_not_proof_of_incomplete_goal_and_zero_is_valid_data():
    result = verify_result('contacts.search', {'ok': True, 'contacts': []})
    assert result['status'] == 'empty' and result['goal_completion'] == 'unknown'
    assert verify_result('banking.get_balance', {'ok': True, 'balance': 0})['status'] == 'success'
    assert verify_result('app.read', {'ok': True, 'rows': [], 'balance': 50})['status'] == 'success'


@pytest.mark.parametrize('response', [
    {'ok': True, 'contacts': []}, {'ok': True, 'contacts': [{'id': 'c_1'}]},
    {'ok': True, 'done': True, 'contacts': []}, {'ok': True, 'done': False, 'message': 'not found'},
    {'ok': True, 'nested': {'items': []}}, {'ok': True, 'balance': 0},
])
def test_every_ok_true_read_is_cached_without_requiring_done(response):
    monitor = LoopMonitor()
    args = {'query': 'Hồng'}
    original = deepcopy(response)
    monitor.record_result('contacts.search', args, response, 'read', call_index=4)
    response['mutated'] = True
    feedback = monitor.check('contacts.search', args, 'read')
    assert feedback['block'] is True and feedback['cached_result'] == original
    assert feedback['source_call_index'] == 4 and feedback['state_epoch'] == 0
    feedback['cached_result']['outside'] = True
    assert monitor.check('contacts.search', args, 'read')['cached_result'] == original
    assert monitor.check('contacts.search', {'query': 'Bà Hồng'}, 'read') is None


@pytest.mark.parametrize('kind', ['read', 'write', 'meta'])
def test_schema_rejection_survives_reads_helpers_success_and_uncertain_writes(kind):
    monitor = LoopMonitor()
    args = {'missing': 'field'}
    monitor.record_validation_failure('app.tool', args, REJECTION)
    for name, result, execution_kind in [
        ('local_time.calendar_reference', {'ok': True}, 'local'),
        ('app.list', {'ok': True, 'items': [{'id': 'new'}]}, 'read'),
        ('other.list', {'ok': True, 'items': [1]}, 'read'),
        ('app.update', {'ok': True, 'done': True}, 'write'),
        ('other.update', None, 'write'),
    ]:
        monitor.record_result(name, {}, result, execution_kind)
        assert monitor.check('app.tool', args, kind)['scope'] == 'sample'
    assert monitor.check('app.tool', {'fixed': 'field'}, kind) is None


@pytest.mark.parametrize('response', [None, True, {}, {'ok': 0}, {'ok': True}, 'failed'])
def test_only_explicit_false_validation_enters_schema_blacklist(response):
    monitor = LoopMonitor()
    monitor.record_validation_failure('app.tool', {}, response)
    assert monitor.check('app.tool', {}, 'write') is None


def test_new_read_data_does_not_unlock_cached_reads_or_noop_writes():
    monitor = LoopMonitor()
    monitor.record_result('contacts.search', {'query': 'Hồng'}, {'ok': True, 'contacts': []}, 'read')
    monitor.record_result('banking.transfer', {'amount': 20}, {'ok': True, 'done': False}, 'write')
    for name in ['contacts.list', 'banking.get_balance', 'weather.read', 'local_time.calendar_reference']:
        monitor.record_result(name, {}, {'ok': True, 'data': [123]}, 'local' if name.startswith('local_') else 'read')
    assert monitor.state_epoch == 0
    assert monitor.check('contacts.search', {'query': 'Hồng'}, 'read')['block']
    assert monitor.check('banking.transfer', {'amount': 20}, 'write')['block']


def test_failed_write_does_not_clear_reads_and_successful_write_does():
    monitor = LoopMonitor()
    monitor.record_result('contacts.search', {}, {'ok': True, 'contacts': []}, 'read')
    monitor.record_result('banking.transfer', {'amount': 20}, {'ok': True, 'done': False}, 'write')
    assert monitor.state_epoch == 0 and monitor.check('contacts.search', {}, 'read')['block']
    monitor.record_result('banking.transfer', {'amount': 10}, {'ok': True, 'done': True}, 'write')
    assert monitor.state_epoch == 1
    assert monitor.check('contacts.search', {}, 'read') is None
    assert monitor.check('banking.transfer', {'amount': 20}, 'write') is None


def test_successful_repeated_writes_are_never_deduplicated():
    monitor = LoopMonitor()
    args = {'program': 'ABC', 'reward_id': 'r_1'}
    for index in range(3):
        assert monitor.check('loyalty.redeem', args, 'write') is None
        monitor.record_result('loyalty.redeem', args, {'ok': True, 'done': True}, 'write', call_index=index)
    assert monitor.state_epoch == 3


def test_nested_done_false_on_successful_write_invalidates_read_cache():
    monitor = LoopMonitor()
    monitor.record_result('reminders.list', {}, {'ok': True, 'items': []}, 'read')
    monitor.record_result('reminders.create', {'content': 'mua hoa'}, {'ok': True, 'item': {'done': False}}, 'write')
    assert monitor.check('reminders.list', {}, 'read') is None


def test_actual_call_schema_rejection_is_permanent_and_does_not_change_state_epoch():
    monitor = LoopMonitor()
    outcome = monitor.record_result('app.tool', {}, {'ok': False, 'error': 'missing'}, 'write')
    assert outcome['contract_violation'] is True and monitor.state_epoch == 0
    monitor.record_result('other.write', {}, {'ok': True}, 'write')
    assert monitor.check('app.tool', {}, 'write')['scope'] == 'sample'


@pytest.mark.parametrize('kind', ['write', 'unknown', None])
def test_unknown_or_exception_write_invalidates_cache_but_is_not_schema_failure(kind):
    monitor = LoopMonitor()
    monitor.record_result('contacts.search', {}, {'ok': True, 'contacts': []}, 'read')
    exc = {'type': 'RuntimeError', 'message': 'backend unavailable'}
    outcome = monitor.record_result('contacts.update', {}, {'execution_exception': exc}, kind, execution_exception=exc)
    assert outcome['origin'] == 'agent_exception' and monitor.state_epoch == 1
    assert monitor.check('contacts.search', {}, 'read') is None
    assert monitor.check('contacts.update', {}, kind) is None
    monitor.record_result('contacts.update', {}, {}, kind, execution_exception=exc)
    assert monitor.check('contacts.update', {}, kind)['reason'] == 'technical_retry_budget'


def test_read_exception_does_not_authorize_a_read_cache_hit():
    monitor = LoopMonitor()
    monitor.record_result('app.read', {}, None, 'read', execution_exception={'type': 'ValueError'})
    assert monitor.check('app.read', {}, 'read') is None


def test_typed_signature_ignores_key_order_but_not_scalar_types():
    monitor = LoopMonitor()
    monitor.record_validation_failure('app.tool', {'value': '2', 'other': 0}, REJECTION)
    assert monitor.check('app.tool', {'other': 0, 'value': '2'}, 'read')['block']
    for value in (2, 2.0, True, None):
        assert monitor.check('app.tool', {'other': 0, 'value': value}, 'read') is None


def test_schema_rejections_are_not_evicted_by_transient_memory_limit():
    monitor = LoopMonitor(max_entries=2)
    for value in (1, 2, 3):
        monitor.record_validation_failure('app.tool', {'value': value}, REJECTION)
    assert len(monitor.pending_warnings()) == 3
    for value in range(5):
        monitor.record_result('app.read', {'value': value}, {'ok': True, 'items': []}, 'read')
    assert monitor.check('app.read', {'value': 0}, 'read') is None
    assert monitor.check('app.tool', {'value': 1}, 'write')['block']
    warnings = monitor.pending_warnings()
    warnings[0]['arguments']['value'] = 99
    assert monitor.pending_warnings()[0]['arguments']['value'] == 1


@pytest.mark.parametrize('limit', [0, -1, 101, True, 1.5])
def test_invalid_memory_limit_is_rejected(limit):
    with pytest.raises(ValueError): LoopMonitor(limit)


def test_non_json_arguments_cannot_be_confused_with_a_valid_signature():
    monitor = LoopMonitor()
    monitor.record_validation_failure('app.tool', {'bad': float('nan')}, REJECTION)
    assert monitor.check('app.tool', {'bad': float('nan')}, 'read') is None


def test_access_check_distinguishes_denied_and_unknown_and_suggests_only_exposed_tools():
    catalog = {'contacts.search': {}, 'banking.transfer': {}, 'sys.ask_missing_info': {}}
    enabled = {name: catalog[name] for name in ('contacts.search', 'sys.ask_missing_info')}
    assert check_tool_access('banking.transfer', catalog, enabled)['status'] == 'tool_not_allowed'
    missing = check_tool_access('contacts.serch', catalog, enabled)
    assert missing['status'] == 'unknown_tool' and missing['suggested_tools'] == ['contacts.search']
    assert set(missing['suggested_tools']) <= set(enabled)
    assert check_tool_access('sys.ask_missing_info', catalog, enabled)['allowed']
    assert check_tool_access('local_text.normalize', catalog, enabled, ['local_text.normalize'])['scope'] == 'local'
    assert check_tool_access(['bad'], catalog, enabled)['status'] == 'invalid_tool_name'
