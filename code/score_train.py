#!/usr/bin/env python3
"""Train-only proxy F1 for tool calls; NOT the official state-hash-gated score.

This script reads labeled TRAIN JSONL and submitted predictions, NOT app state.
It does not instantiate any ToolEnv or access *_environment/.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from lifemate.catalog import load_catalog


def token_f1(a, b) -> float:
    aa = unicodedata.normalize("NFC", str(a).lower()).split()
    bb = unicodedata.normalize("NFC", str(b).lower()).split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    common = sum((Counter(aa) & Counter(bb)).values())
    return 2 * common / (len(aa) + len(bb))


def calls_match(pred: dict, gold: dict, catalog: dict[str, dict]) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    p, g = pred.get("arguments", {}), gold.get("arguments", {})
    if not isinstance(p, dict) or not isinstance(g, dict) or set(p) != set(g):
        return False
    props = catalog.get(pred["name"], {}).get("parameters", {}).get("properties", {})
    for key in p:
        if props.get(key, {}).get("x-match") == "fuzzy":
            if token_f1(p[key], g[key]) <= 0.6:
                return False
        elif p[key] != g[key] or type(p[key]) is not type(g[key]):
            return False
    return True


def max_matches(preds: list, golds: list, catalog: dict) -> int:
    # Maximum one-to-one bipartite matching (order-independent).
    edges = [[j for j, gold in enumerate(golds) if calls_match(pred, gold, catalog)]
             for pred in preds]
    gold_to_pred: dict[int, int] = {}

    def augment(pred_idx: int, seen: set[int]) -> bool:
        for j in edges[pred_idx]:
            if j in seen:
                continue
            seen.add(j)
            if j not in gold_to_pred or augment(gold_to_pred[j], seen):
                gold_to_pred[j] = pred_idx
                return True
        return False

    return sum(augment(i, set()) for i in range(len(preds)))


def call_f1(preds: list, golds: list, catalog: dict) -> tuple[float, int]:
    if not golds:
        return (1.0 if not preds else 0.0), 0
    matched = max_matches(preds, golds, catalog)
    return (2 * matched / (len(preds) + len(golds)) if matched else 0.0), matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    predictions = {}
    for line in args.submission.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            predictions[item["id"]] = item.get("tool_calls", [])
    count = 0
    total = 0.0
    exact = 0
    covered = 0
    for line in args.train_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["id"] not in predictions:
            continue
        g = item.get("label", {}).get("tool_calls", [])
        p = predictions[item["id"]]
        score, matched = call_f1(p, g, catalog)
        count += 1
        total += score
        exact += int(score == 1.0)
        covered += matched
    if not count:
        raise SystemExit("No overlapping TRAIN sample ids.")
    print(f"Samples: {count} | Tool-call F1 (no hash gate): {100 * total / count:.2f}%")
    print(f"Exact tool-call samples: {exact}/{count} | Matched gold calls: {covered}")
    print("WARNING: This is NOT the official score. Cannot infer state-hash match from train labels.")


if __name__ == "__main__":
    main()
