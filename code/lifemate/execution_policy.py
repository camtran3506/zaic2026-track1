"""Automatic Python guards over public catalog/proposals/results, per sample.

No model, ToolEnv construction, hidden state or gold access. Schema rejections
last for the sample; successful reads and no-op writes last for a state epoch.
"""
from collections import OrderedDict
from copy import deepcopy
from difflib import get_close_matches
import json


EXECUTION_POLICY = "automatic-tool-guards-v1"
# Distinct read arguments stay allowed after empty results; exact repeats are
# still blocked by the state-scoped read cache. Zero disables the empty-read cap.
DEFAULT_EMPTY_READ_LIMIT = 0
AUTOMATIC_GUARD_INSTRUCTIONS = """- Chương trình tự kiểm tên/quyền tool, validate và chống lặp bằng Python; không cần gọi helper để bật các kiểm tra này.
- sys.ask_missing_info chỉ báo phần thực sự chưa xác định. Python chặn tên tham số sai hoặc hỏi lại giá trị đã xác định từ schema/context/transcript/history/kết quả đọc; dùng bằng chứng trong phản hồi chặn để tiếp tục.
- Tool + arguments bị validate từ chối bị cấm suốt sample, kể cả sau read/write khác; sửa đề xuất theo lỗi thật.
- Read đã trả ok:true (có dữ liệu, rỗng hoặc done:false) được giữ lại trong phiên hiện tại.
  Không gọi lại cùng read + arguments khi chưa có write có thể đổi trạng thái; dùng kết quả/feedback đã có.
- Write trả done:false ở cấp ngoài cùng cũng không được thử lại nguyên call khi trạng thái chưa đổi.
  Read/helper không đổi trạng thái. Write thành công làm kết quả read cũ hết hiệu lực.
- Write thành công có thể được lặp đúng số lần yêu cầu, ví dụ đổi hai voucher; không tự gộp các call cần thiết.
- Phản hồi chặn và automatic_execution_feedback do chương trình tạo, không phải tool env mới thực thi.
  Kết quả rỗng không ép đổi entity, hỏi lại hoặc done. Lỗi kỹ thuật không đồng nghĩa bị từ chối schema.
"""
_META_KEYS = {"ok", "done", "error", "message", "count", "total", "cursor", "next_cursor"}


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return None


def check_tool_access(name, catalog, enabled, local_names=()):
    """Automatic exact-name access check. Fuzzy results are suggestions only."""
    if isinstance(name, str) and name in local_names:
        return {"allowed": True, "status": "allowed_local_helper", "scope": "local"}
    if isinstance(name, str) and name in enabled:
        return {"allowed": True, "status": "allowed_env_tool", "scope": "env"}
    status = ("invalid_tool_name" if not isinstance(name, str) else
              "tool_not_allowed" if name in catalog else "unknown_tool")
    exposed = sorted(set(enabled) | set(local_names))
    suggestions = get_close_matches(name, exposed, n=3, cutoff=0.45) if isinstance(name, str) else []
    return {"allowed": False, "status": status, "scope": "none", "suggested_tools": suggestions,
            "instruction": "Tool không tồn tại hoặc không được sample cho phép. Chọn đúng tên trong tool_schemas; "
                           "gợi ý chỉ để đối chiếu, không tự thay tool hoặc đoán mục tiêu."}


def verify_result(name, result):
    """Classify the response, never claim the user's entire goal is complete."""
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
                    "instruction": "Kết quả rỗng có thể hợp lệ. Dùng kết quả đã có; không ép chọn entity, hỏi lại hoặc done."}
    return {**outcome, "status": "success", "reason": "reported_success"}


class LoopMonitor:
    """Automatically gate exact proposals; never execute or rewrite a call.

    max_entries bounds state-scoped caches. Schema rejections are NEVER evicted:
    the outer agent's total-step budget bounds their per-sample size. No history
    read is seeded into this cache. Only actual calls in this ToolEnv authorize
    reuse. A successful/potentially mutating write invalidates all transient caches
    conservatively across apps; reads and known no-op writes do not.
    """
    def __init__(self, max_entries=100, empty_read_limit=DEFAULT_EMPTY_READ_LIMIT):
        if type(max_entries) is not int or not 1 <= max_entries <= 100:
            raise ValueError("max_entries must be an integer between 1 and 100")
        if type(empty_read_limit) is not int or empty_read_limit < 0:
            raise ValueError("empty_read_limit must be a nonnegative integer")
        self.max_entries = max_entries
        self.empty_read_limit = empty_read_limit
        self.state_epoch = 0
        self._schema_rejections = {}
        self._records = OrderedDict()
        self._technical_errors = {}
        self._empty_read_counts = {}

    def _key(self, name, arguments):
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        encoded = _canonical(arguments)
        return (name, encoded) if encoded is not None else None

    def _advance_state(self):
        self.state_epoch += 1
        self._records.clear()
        self._empty_read_counts.clear()

    def check(self, name, arguments, kind):
        if kind == "local":
            return None
        key = self._key(name, arguments)
        if key is None:
            return None
        if key in self._schema_rejections:
            return {"block": True, "reason": "repeated_rejected_call", "scope": "sample",
                    "name": name, "arguments": deepcopy(arguments),
                    "previous_error": deepcopy(self._schema_rejections[key]),
                    "instruction": "Đúng tool + arguments này đã bị từ chối schema trong sample. "
                                   "Read/write/helper không mở lại đề xuất này; sửa đúng arguments theo lỗi đã có."}
        technical = self._technical_errors.get(key)
        if technical and technical["count"] >= 2:
            return {"block": True, "reason": "technical_retry_budget", "scope": "sample_retry_budget",
                    "name": name, "arguments": deepcopy(arguments), "technical_error": deepcopy(technical),
                    "instruction": "Đề xuất này đã gặp lỗi kỹ thuật hai lần; không thử lại nguyên call. "
                                   "Đây không phải kết luận schema hay bằng chứng entity không tồn tại."}
        record = self._records.get(key)
        if record is not None:
            self._records.move_to_end(key)
            return {"block": True, "reason": record["reason"], "scope": "current_state",
                    "name": name, "arguments": deepcopy(arguments), "state_epoch": self.state_epoch,
                    "cached_result": deepcopy(record["result"]),
                    "source_call_index": record["call_index"],
                    "previous_error": deepcopy(record["verification"]),
                    "instruction": "Call giống hệt đã có kết quả trong phiên hiện tại và chưa có write có thể đổi trạng thái. "
                                   "Dùng cached_result từ source_call_index; không có env call mới. "
                                   "Không tự đổi entity, hỏi lại hoặc done chỉ vì kết quả rỗng/chưa hoàn tất."}
        if self.empty_read_limit and kind == "read" and self._empty_read_counts.get(name, 0) >= self.empty_read_limit:
            count = self._empty_read_counts[name]
            return {"block": True, "reason": "read_empty_retry_cap", "scope": "current_state",
                    "name": name, "arguments": deepcopy(arguments), "state_epoch": self.state_epoch,
                    "empty_count": count,
                    "instruction": f"{name} đã trả về kết quả rỗng {count} lần; không gọi lại tool này. "
                                   "Nếu còn thiếu thông tin bắt buộc thì gọi sys.ask_missing_info, nếu không thì làm bước tiếp theo hoặc kết thúc."}
        return None

    def record_validation_failure(self, name, arguments, reason):
        """Only an explicit organizer rejection is a permanent schema rejection."""
        key = self._key(name, arguments)
        if key is not None and isinstance(reason, dict) and reason.get("ok") is False:
            self._schema_rejections.setdefault(key, deepcopy(reason))

    def record_technical_failure(self, name, arguments, error, *, stage):
        key = self._key(name, arguments)
        if key is not None:
            count = self._technical_errors.get(key, {}).get("count", 0) + 1
            self._technical_errors[key] = {"count": count, "stage": stage, "error": deepcopy(error)}

    def record_result(self, name, arguments, result, kind, *, call_index=None, execution_exception=None):
        key = self._key(name, arguments)
        if kind == "local":
            return verify_result(name, result)
        may_write = kind not in {"read", "meta"}
        if execution_exception is not None:
            if may_write:
                self._advance_state()
            self.record_technical_failure(name, arguments, execution_exception, stage="call")
            return {"tool": name, "status": "unknown", "goal_completion": "unknown",
                    "reason": "execution_exception", "origin": "agent_exception",
                    "exception": deepcopy(execution_exception),
                    "instruction": "Tool call đã được thử nhưng ném exception; không coi đây là lỗi schema hay kết quả nghiệp vụ."}
        if not isinstance(result, dict) or type(result.get("ok")) is not bool:
            if may_write:
                self._advance_state()
            self.record_technical_failure(name, arguments, result, stage="call_protocol")
            return {"tool": name, "status": "unknown", "goal_completion": "unknown",
                    "reason": "unrecognized_result_protocol", "origin": "env.call"}
        if result["ok"] is False:
            # Under the organizer contract this can only be a schema rejection.
            # Actual exception envelopes are handled above, never put here.
            self.record_validation_failure(name, arguments, result)
            return {**verify_result(name, result), "origin": "env.call",
                    "reason": "schema_rejected_at_call", "contract_violation": True}
        if key is not None:
            self._technical_errors.pop(key, None)
        outcome = verify_result(name, result)
        if may_write and result.get("done") is not False:
            self._advance_state()
        elif key is not None and (kind == "read" or (may_write and result.get("done") is False)):
            if kind == "read" and outcome["status"] in ("empty", "failure"):
                self._empty_read_counts[name] = self._empty_read_counts.get(name, 0) + 1
            reason = ("repeated_empty_read" if outcome["status"] == "empty" else
                      "repeated_failed_read" if kind == "read" and outcome["status"] == "failure" else
                      "repeated_successful_read" if kind == "read" else "repeated_no_progress_write")
            self._records[key] = {"name": name, "arguments": deepcopy(arguments), "reason": reason,
                                  "result": deepcopy(result), "verification": deepcopy(outcome),
                                  "call_index": call_index}
            self._records.move_to_end(key)
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)
        return outcome

    def automatic_feedback(self, kind, verification, call_index):
        """Same factual post-call feedback for SFT and inference, no model tool."""
        return {"policy": EXECUTION_POLICY, "managed_by": "python", "state_epoch": self.state_epoch,
                "source_call_index": call_index, "origin": verification.get("origin", "env.call"),
                "repeat_policy": ("blocked_for_sample" if verification.get("contract_violation") else
                                  "technical_retry_budget" if verification["status"] == "unknown" else
                                  "reuse_read_until_state_change" if kind == "read" else
                                  "blocked_until_state_change" if verification["status"] == "failure" else
                                  "allowed"),
                "instruction": "Các chặn được chương trình quản lý tự động. Read/helper không làm kết quả cũ hết hiệu lực; "
                               "write thành công có thể đổi trạng thái. Write thành công được phép lặp đúng mục tiêu."}

    def pending_warnings(self):
        warnings = [{"name": key[0], "arguments": json.loads(key[1]),
                     "reason": "repeated_rejected_call", "block": True}
                    for key in self._schema_rejections]
        warnings += [{"name": row["name"], "arguments": deepcopy(row["arguments"]),
                      "reason": row["reason"], "block": True}
                     for row in self._records.values() if row["verification"]["status"] != "success"]
        return warnings
