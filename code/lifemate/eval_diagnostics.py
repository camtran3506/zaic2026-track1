"""Post-rollout error analysis using gold labels; never called by the planner."""
from collections import Counter, defaultdict, deque

from .scoring import calls_match, match_call_indices


ERROR_TYPES = ("state_hash_mismatch", "missing_calls", "extra_calls", "wrong_arguments", "early_stop")
STOP_REASONS = ("done", "ask_missing_info", "max_calls", "max_invalid", "time_limit",
                "max_total_steps", "no_available_tools", "unknown")


def _argument_differences(predicted, gold, catalog):
    actual, expected = predicted.get("arguments", {}), gold.get("arguments", {})
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return [{"argument": None, "kind": "arguments_not_object",
                 "predicted": actual, "expected": expected}]
    differences = []
    for key in sorted(actual.keys() | expected.keys()):
        if key not in actual:
            differences.append({"argument": key, "kind": "missing", "expected": expected[key]})
        elif key not in expected:
            differences.append({"argument": key, "kind": "unexpected", "predicted": actual[key]})
        elif not calls_match({"name": gold["name"], "arguments": {key: actual[key]}},
                             {"name": gold["name"], "arguments": {key: expected[key]}}, catalog):
            differences.append({"argument": key,
                "kind": "type_mismatch" if type(actual[key]) is not type(expected[key]) else "value_mismatch",
                "predicted": actual[key], "expected": expected[key]})
    return differences


def analyze_rollout(predicted, gold, predicted_hash, gold_hash, catalog, stop_reason="unknown"):
    """Describe observable discrepancies, not unproven causal explanations.

    Missing/extra counts use the scorer's maximum full-call matching. Remaining
    same-name calls are paired in their original order to diagnose arguments;
    this pairing is advisory when a tool occurs multiple times and never changes
    the score. Wrong-argument calls remain missing/extra in the full-call counts.
    """
    pairs = match_call_indices(predicted, gold, catalog)
    matched_pred = {i for i, _ in pairs}
    matched_gold = {j for _, j in pairs}
    extra = [i for i in range(len(predicted)) if i not in matched_pred]
    missing = [j for j in range(len(gold)) if j not in matched_gold]
    by_name = defaultdict(deque)
    for j in missing:
        by_name[gold[j].get("name")].append(j)
    wrong_arguments, attempted = [], set()
    for i in extra:
        candidates = by_name[predicted[i].get("name")]
        if candidates:
            j = candidates.popleft()
            attempted.add(j)
            wrong_arguments.append({"prediction_index": i, "gold_index": j,
                "tool_name": gold[j].get("name"),
                "differences": _argument_differences(predicted[i], gold[j], catalog)})
    unattempted = [j for j in missing if j not in attempted]
    hash_match = predicted_hash == gold_hash
    stop_reason = stop_reason if stop_reason in STOP_REASONS else "unknown"
    # A missing read can leave the correct state: that alone is not early stopping.
    # Likewise, trying the right tool with wrong args is not proof of stopping early.
    flags = {"state_hash_mismatch": not hash_match, "missing_calls": bool(missing),
             "extra_calls": bool(extra), "wrong_arguments": bool(wrong_arguments),
             "early_stop": stop_reason == "done" and not hash_match and bool(unattempted)}
    return {"version": 1, "state_hash_match": hash_match, "stop_reason": stop_reason,
            "error_types": [name for name in ERROR_TYPES if flags[name]],
            "matched_call_indices": [{"prediction_index": i, "gold_index": j} for i, j in pairs],
            "missing_gold_call_indices": missing, "extra_prediction_call_indices": extra,
            "wrong_argument_pairs": wrong_arguments, "unattempted_gold_call_indices": unattempted,
            "call_counts": {"predicted": len(predicted), "gold": len(gold), "matched": len(pairs),
                            "missing": len(missing), "extra": len(extra), "wrong_arguments": len(wrong_arguments)}}


class DiagnosticSummary:
    """Aggregate sample errors separately from numbers of individual calls."""
    def __init__(self):
        self.samples = 0
        self.errors = Counter()
        self.calls = Counter()
        self.stops = Counter()
        self.sources = {}

    def add(self, diagnostic, source="unknown"):
        self.samples += 1
        self.errors.update(diagnostic["error_types"])
        self.calls.update(diagnostic["call_counts"])
        self.stops.update([diagnostic["stop_reason"]])
        source = source if source in ("audio_text", "asr") else "unknown"
        if source not in self.sources:
            self.sources[source] = DiagnosticSummary()
        # Avoid recursive source grouping in the per-source aggregate.
        group = self.sources[source]
        group.samples += 1
        group.errors.update(diagnostic["error_types"])
        group.calls.update(diagnostic["call_counts"])
        group.stops.update([diagnostic["stop_reason"]])

    def as_dict(self):
        return {"version": 1, "samples": self.samples,
                "error_samples": {name: self.errors[name] for name in ERROR_TYPES},
                "error_rates": {name: self.errors[name] / self.samples if self.samples else 0.0
                                for name in ERROR_TYPES},
                "call_counts": {name: self.calls[name] for name in
                                ("predicted", "gold", "matched", "missing", "extra", "wrong_arguments")},
                "stop_reasons": {name: self.stops[name] for name in STOP_REASONS},
                "by_transcript_source": {name: group.as_dict() for name, group in sorted(self.sources.items())}}

    def metrics(self):
        return {**{f"eval_{name}_samples": self.errors[name] for name in ERROR_TYPES},
                **{f"eval_{name}_calls": self.calls[name] for name in ("missing", "extra", "wrong_arguments")},
                **{f"eval_stop_{name}_samples": self.stops[name] for name in STOP_REASONS}}
