"""Bounded factual feedback around execution; no model, ToolEnv or gold access.

Inspired by pre/post execution hooks and evaluator feedback loops. These checks
describe observable outcomes, not whether the user's entire task is complete.
"""
from collections import OrderedDict
from copy import deepcopy
import json


_META_KEYS = {"ok", "done", "error", "message", "count", "total", "cursor", "next_cursor"}


def _canonical(value):
    """Preserve JSON scalar types and list order; uncertain inputs fail open."""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError):
        return None


def verify_result(name, result):
    """Classify a reported result, never infer goal completion from it.

    In particular, an empty search can be a valid answer, and nested `done:false`
    on a newly-created reminder is its lifecycle state rather than call failure.
    """
    outcome = {"tool": name, "status": "unknown", "goal_completion": "unknown"}
    if not isinstance(result, dict):
        return {**outcome, "reason": "unrecognized_result_protocol"}
    if result.get("ok") is False or result.get("done") is False:
        reason = "business_not_completed" if result.get("done") is False else "reported_error"
        return {**outcome, "status": "failure", "reason": reason,
                "message": deepcopy(result.get("message", result.get("error"))),
                "instruction": "Hành động chưa được xác nhận thành công; đọc lỗi và đánh giá lại kế hoạch."}
    if result.get("ok") is not True:
        return {**outcome, "reason": "unrecognized_result_protocol"}
    if result.get("done") is not True:
        payload = [value for key, value in result.items() if key not in _META_KEYS]
        if payload and all(isinstance(value, list) and not value for value in payload):
            return {**outcome, "status": "empty", "reason": "empty_result",
                    "instruction": "Kết quả rỗng có thể hợp lệ. Không lặp nguyên truy vấn khi chưa có dữ kiện mới; "
                                   "không ép chọn một entity và không tự suy ra phải hỏi lại."}
    return {**outcome, "status": "success", "reason": "reported_success"}


class LoopMonitor:
    """Reject only known non-progressing retries, with bounded per-sample memory.

    Relevant evidence is tracked by app namespace. An executed write may change
    application state; new, distinct nonempty read evidence can justify a
    retry in that same app. Helpers and unrelated reads never reset a block.
    Successful calls are not deduplicated. Failed writes remain advisory because
    this observer cannot establish transactional rollback or absent side effects.
    """

    def __init__(self, max_entries=100):
        if type(max_entries) is not int or not 1 <= max_entries <= 100:
            raise ValueError("max_entries must be an integer between 1 and 100")
        self.max_entries = max_entries
        self.state_epoch = 0
        self._serial = 0
        self._versions = OrderedDict()
        self._seen_evidence = OrderedDict()
        self._records = OrderedDict()

    @staticmethod
    def _domain(name):
        return name.split(".", 1)[0]

    def _key(self, name, arguments):
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        arguments_json = _canonical(arguments)
        return (name, arguments_json) if arguments_json is not None else None

    def _version(self, name):
        return self.state_epoch, self._versions.get(self._domain(name), 0)

    def _trim(self, mapping):
        while len(mapping) > self.max_entries:
            mapping.popitem(last=False)

    def _new_evidence(self, name, result):
        encoded = _canonical(result)
        if encoded is None:
            return
        domain = self._domain(name)
        key = domain, encoded
        if key in self._seen_evidence:
            self._seen_evidence.move_to_end(key)
            return
        self._seen_evidence[key] = None
        self._trim(self._seen_evidence)
        self._serial += 1
        self._versions[domain] = self._serial
        self._versions.move_to_end(domain)
        while len(self._versions) > self.max_entries:
            retired, _ = self._versions.popitem(last=False)
            # Evicted evidence must not make an old rejection appear current.
            for record_key in list(self._records):
                if self._domain(record_key[0]) == retired:
                    del self._records[record_key]

    def _remember(self, name, arguments, reason, *, block, detail=None):
        key = self._key(name, arguments)
        if key is None:
            return
        self._records[key] = {"name": name, "arguments": deepcopy(arguments),
                              "reason": reason, "block": block,
                              "version": self._version(name), "detail": deepcopy(detail)}
        self._records.move_to_end(key)
        self._trim(self._records)

    def check(self, name, arguments, kind):
        """Return a block/advisory for an exact retry, or None.

        A caller must inspect `block`; an advisory is not permission to suppress
        an otherwise valid call. Nothing here executes or rewrites arguments.
        """
        if kind == "local":
            return None
        key = self._key(name, arguments)
        record = self._records.get(key) if key is not None else None
        if record is None or record["version"] != self._version(name):
            return None
        self._records.move_to_end(key)
        return {"block": record["block"], "reason": record["reason"],
                "name": name, "arguments": deepcopy(arguments),
                "previous_error": deepcopy(record["detail"]),
                "instruction": "Call giống hệt đã bị từ chối hoặc chưa tạo tiến triển. "
                               "Dùng lỗi/kết quả đã có để sửa đúng tham số hoặc kế hoạch; "
                               "không bắt buộc đổi tên entity hay ép tìm được kết quả."}

    def record_validation_failure(self, name, arguments, reason):
        """Remember the full call, not a permanent ban on a parameter/name."""
        if isinstance(name, str) and name.startswith("local_"):
            return
        self._remember(name, arguments, "repeated_rejected_call", block=True, detail=reason)

    def record_result(self, name, arguments, result, kind):
        outcome = verify_result(name, result)
        if kind == "local":
            return outcome
        key = self._key(name, arguments)
        if kind not in {"read", "meta"}:
            # Missing kind metadata cannot prove read-only execution either.
            # Even an error/unknown outcome does not prove the call rolled back.
            self.state_epoch += 1
        if outcome["status"] == "success":
            if kind == "read":
                self._new_evidence(name, result)
            if key is not None:
                self._records.pop(key, None)
        elif outcome["status"] in {"empty", "failure"}:
            self._remember(name, arguments,
                           "repeated_empty_read" if outcome["status"] == "empty" else "repeated_failed_call",
                           block=kind == "read", detail=outcome)
        return outcome

    def pending_warnings(self):
        """Facts still current enough to surface before done; never veto done."""
        return [{"name": row["name"], "arguments": deepcopy(row["arguments"]),
                 "reason": row["reason"], "block": row["block"]}
                for row in self._records.values()
                if row["version"] == self._version(row["name"])]
