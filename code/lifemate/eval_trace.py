"""Streaming debug events for validation; never supplied to the model."""
import json
import time


def exception_details(exc):
    return {"type": type(exc).__name__, "message": str(exc), "repr": repr(exc)}


class EvalTraceWriter:
    """Write and flush each event immediately, snapshotting mutable API results."""
    def __init__(self, stream):
        self.stream = stream
        self.event_index = 0
        self.started = time.monotonic()

    def for_sample(self, sample_id):
        def emit(event):
            record = {**event, "version": 1, "event_index": self.event_index + 1,
                      "sample_id": sample_id,
                      "elapsed_seconds": round(time.monotonic() - self.started, 6)}
            # No truncation or text normalization: arguments/results retain their
            # original values. Non-JSON objects use the agent's existing str fallback.
            self.stream.write(json.dumps(record, ensure_ascii=False, default=str,
                                         separators=(",", ":")) + "\n")
            self.stream.flush()
            self.event_index += 1
        return emit
