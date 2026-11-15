#!/usr/bin/env python3
"""Train-only proxy F1 for tool calls; NOT the official state-hash-gated score.

This script reads labeled TRAIN JSONL and submitted predictions, NOT app state.
It does not instantiate any ToolEnv or access *_environment/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lifemate.catalog import load_catalog
from lifemate.scoring import token_f1, calls_match, max_matches, call_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=REPO_ROOT / "data" / "train" / "train.jsonl")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data" / "train" / "train_tools_catalog.json")
    parser.add_argument("--submission", type=Path, default=CODE_DIR / "result" / "submission.jsonl")
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
