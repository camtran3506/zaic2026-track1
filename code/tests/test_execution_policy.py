from copy import deepcopy

import pytest

from lifemate.execution_policy import LoopMonitor, verify_result


def test_verifier_distinguishes_transport_success_from_business_failure():
    assert verify_result("banking.transfer", {"ok": True, "done": False,
                                               "message": "số dư không đủ"})["status"] == "failure"
    assert verify_result("app.call", {"ok": False, "error": "failed"})["status"] == "failure"
    result = {"ok": True, "item": {"done": False, "content": "Mua hoa"}}
    assert verify_result("reminders.create", result)["status"] == "success"
    assert verify_result("app.call", {"ok": 1})["status"] == "unknown"
    assert verify_result("app.call", None)["status"] == "unknown"


def test_empty_is_a_factual_outcome_not_proof_of_incomplete_user_goal():
    result = verify_result("contacts.search", {"ok": True, "contacts": []})
    assert result["status"] == "empty" and result["goal_completion"] == "unknown"
    assert verify_result("app.read", {"ok": True, "rows": [], "balance": 50})["status"] == "success"
    assert verify_result("app.call", {"ok": True, "done": True, "rows": []})["status"] == "success"


def test_exact_empty_read_retry_is_blocked_without_changing_arguments():
    monitor = LoopMonitor()
    args = {"query": "Bà Hồng"}
    original = deepcopy(args)
    assert monitor.check("contacts.search", args, "read") is None
    monitor.record_result("contacts.search", args, {"ok": True, "contacts": []}, "read")
    feedback = monitor.check("contacts.search", args, "read")
    assert feedback["block"] is True and feedback["reason"] == "repeated_empty_read"
    feedback["arguments"]["query"] = "edited outside monitor"
    assert args == original
    assert monitor.check("contacts.search", original, "read")["arguments"] == original
    assert monitor.check("contacts.search", {"query": "Hồng"}, "read") is None


def test_successful_read_refresh_and_successful_writes_are_never_deduplicated():
    monitor = LoopMonitor()
    for _ in range(3):
        assert monitor.check("contacts.search", {}, "read") is None
        monitor.record_result("contacts.search", {}, {"ok": True, "contacts": [{"id": "c_1"}]}, "read")
        assert monitor.check("sms.send", {"content": "again"}, "write") is None
        monitor.record_result("sms.send", {"content": "again"}, {"ok": True}, "write")
    assert monitor.state_epoch == 3


def test_validation_rejection_blocks_only_same_call_at_current_evidence_epoch():
    monitor = LoopMonitor()
    args = {"contact_id": "c_unknown"}
    monitor.record_validation_failure("contacts.get", args, {"error": "unknown id"})
    assert monitor.check("contacts.get", args, "read")["block"] is True
    assert monitor.check("contacts.get", args, "read")["previous_error"] == {"error": "unknown id"}
    # A helper or an unrelated app read cannot launder the rejected proposal.
    monitor.record_result("local_time.calendar_reference", {}, {"ok": True, "today": "2026-11-04"}, "local")
    monitor.record_result("weather.get_current", {}, {"ok": True, "current": [{"city": "Huế"}]}, "read")
    assert monitor.check("contacts.get", args, "read")["block"] is True
    monitor.record_result("contacts.search", {}, {"ok": True, "contacts": [{"id": "c_unknown"}]}, "read")
    assert monitor.check("contacts.get", args, "read") is None


def test_repeated_identical_read_evidence_does_not_reset_rejection():
    monitor = LoopMonitor()
    known = {"ok": True, "contacts": [{"id": "c_1"}]}
    monitor.record_result("contacts.search", {}, known, "read")
    monitor.record_validation_failure("contacts.get", {"contact_id": "missing"}, "not found")
    monitor.record_result("contacts.search", {}, known, "read")
    assert monitor.check("contacts.get", {"contact_id": "missing"}, "read")["block"] is True


def test_state_change_permits_retry_but_failed_write_is_only_advisory():
    monitor = LoopMonitor()
    args = {"amount": 20}
    monitor.record_result("banking.transfer", args, {"ok": True, "done": False}, "write")
    assert monitor.check("banking.transfer", args, "write")["block"] is False
    monitor.record_result("banking.deposit", {"amount": 100}, {"ok": True}, "write")
    assert monitor.check("banking.transfer", args, "write") is None


def test_unknown_write_cannot_leave_stale_absence_cache_authoritative():
    monitor = LoopMonitor()
    monitor.record_result("contacts.search", {}, {"ok": True, "contacts": []}, "read")
    assert monitor.check("contacts.search", {}, "read")["block"] is True
    monitor.record_result("contacts.update", {"contact_id": "c_1"}, None, "write")
    assert monitor.check("contacts.search", {}, "read") is None


@pytest.mark.parametrize("kind,result", [
    ("write", {"ok": False, "error": "failed after possible partial update"}),
    (None, {"ok": True}),
    ("unknown", None),
])
def test_partial_or_unknown_execution_invalidates_prior_read_rejections(kind, result):
    monitor = LoopMonitor()
    monitor.record_result("contacts.search", {}, {"ok": True, "contacts": []}, "read")
    monitor.record_result("contacts.custom", {}, result, kind)
    assert monitor.check("contacts.search", {}, "read") is None


def test_typed_canonical_signature_does_not_confuse_values_or_dict_order():
    monitor = LoopMonitor()
    monitor.record_validation_failure("app.call", {"value": "2", "other": 0}, "type")
    assert monitor.check("app.call", {"other": 0, "value": "2"}, "read")["block"] is True
    for value in (2, 2.0, True, None):
        assert monitor.check("app.call", {"other": 0, "value": value}, "read") is None


def test_pending_warnings_are_advisory_copies_and_memory_is_bounded():
    monitor = LoopMonitor(max_entries=2)
    for value in (1, 2, 3):
        monitor.record_validation_failure("app.call", {"value": value}, "invalid")
    warnings = monitor.pending_warnings()
    assert len(warnings) == 2 and monitor.check("app.call", {"value": 1}, "read") is None
    warnings[0]["arguments"]["value"] = 99
    assert monitor.pending_warnings()[0]["arguments"]["value"] == 2


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_invalid_memory_limit_is_rejected(limit):
    with pytest.raises(ValueError):
        LoopMonitor(limit)


def test_non_json_arguments_do_not_turn_into_an_accidental_match():
    monitor = LoopMonitor()
    monitor.record_validation_failure("app.call", {"bad": float("nan")}, "invalid")
    assert monitor.check("app.call", {"bad": float("nan")}, "read") is None
