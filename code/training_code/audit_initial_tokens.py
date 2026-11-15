import argparse
import csv
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from transformers import AutoTokenizer

import lifemate.agent as agent
from lifemate.catalog import allowed_tools, load_catalog
from lifemate.history import prompt_history
from lifemate.llm import configure_tokenizer, render_prompt
from lifemate.sft import prepare_train_transcript


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def serialize_without_shortening(payload, *args, **kwargs):
    return json.dumps(
        payload, ensure_ascii=False, default=str, separators=(",", ":")
    )


def count_tokens(tokenizer, user_prompt):
    _, ids = render_prompt(tokenizer, [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ])
    return len(ids)


def statistics(rows, key, budget):
    ordered = sorted(rows, key=lambda row: row[key])
    values = [row[key] for row in ordered]
    exceeded = sum(value > budget for value in values)

    def percentile(p):
        return values[max(0, math.ceil(p * len(values)) - 1)]

    return {
        "samples": len(values),
        "min_tokens": values[0],
        "mean_tokens": round(sum(values) / len(values), 2),
        "p50_tokens": percentile(0.50),
        "p90_tokens": percentile(0.90),
        "p95_tokens": percentile(0.95),
        "p99_tokens": percentile(0.99),
        "max_tokens": values[-1],
        "max_sample_ids": [
            row["sample_id"] for row in ordered if row[key] == values[-1]
        ],
        "over_budget_samples": exceeded,
        "over_budget_percent": round(100 * exceeded / len(values), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, default=REPO_ROOT / "data" / "train" / "train.jsonl")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "data" / "train" / "train_tools_catalog.json")
    parser.add_argument("--model", type=Path, default=CODE_DIR / "saved_models" / "llm")
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--budget", type=int, default=16000)
    parser.add_argument("--max-tool-details", type=int, default=35)
    parser.add_argument(
        "--output-dir", type=Path, default=CODE_DIR / "result" / "token_audit"
    )
    args = parser.parse_args()

    records = read_jsonl(args.train_jsonl)
    if not records:
        raise ValueError("Dataset is empty")
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate sample IDs")

    catalog = load_catalog(args.catalog)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    configure_tokenizer(tokenizer, args.model)

    cached = {}
    if args.transcripts:
        cache_rows = read_jsonl(args.transcripts)
        cached = {row["id"]: row for row in cache_rows}
        if len(cached) != len(cache_rows):
            raise ValueError("Duplicate IDs in transcript cache")

    rows = []
    for index, record in enumerate(records, 1):
        context = record.get("context", {})
        enabled = allowed_tools(catalog, context.get("installed_apps", {}))

        # Match the runtime: only the universal meta tool means no LLM call.
        if len(enabled) == 1:
            rows.append({
                "sample_id": record["id"],
                "allowed_tools_including_meta": 1,
                "current_schemas": 0,
                "current_tokens": 0,
                "all_schemas_tokens": 0,
                "llm_skipped": True,
            })
            continue

        try:
            prepared = cached.get(record["id"])
            if prepared is None:
                prepared = prepare_train_transcript(
                    record, args.train_jsonl.parent
                )
            transcript = prepared["transcript"]
            history = prompt_history(prepared["history"])
        except Exception as exc:
            raise RuntimeError(
                f"{record['id']}: cannot resolve request/history text. "
                "If audio-only, supply --transcripts with a newly generated "
                "runtime transcript cache. No text is silently omitted."
            ) from exc

        # Initial step: no calls/results from the current request yet.
        build_args = (transcript, context, history, enabled, [], None)

        _, current = agent.build_step_prompt(
            *build_args, args.max_tool_details
        )

        # Disable character-based pruning for the full-schema experiment.
        with patch.object(
            agent, "fit_prompt_payload", serialize_without_shortening
        ):
            _, full = agent.build_step_prompt(*build_args, len(enabled))

        full_schemas = json.loads(full)["available_tools"]["tool_schemas"]
        exposed = {schema["name"] for schema in full_schemas}
        if exposed != set(enabled):
            raise RuntimeError(
                f"{record['id']}: full-schema prompt still omits tools: "
                f"{sorted(set(enabled) - exposed)}"
            )

        rows.append({
            "sample_id": record["id"],
            "allowed_tools_including_meta": len(enabled),
            "current_schemas": len(
                json.loads(current)["available_tools"]["tool_schemas"]
            ),
            "current_tokens": count_tokens(tokenizer, current),
            "all_schemas_tokens": count_tokens(tokenizer, full),
            "llm_skipped": False,
        })

        if index % 100 == 0:
            print(f"Processed {index}/{len(records)}", file=sys.stderr)

    summary = {
        "model": str(args.model.resolve()),
        "input_budget": args.budget,
        "comparison": "strictly greater than budget",
        "initial_step_only": True,
        "history_policy": "repository policy; no future gold calls/results",
        "text_source": "transcript cache or TRAIN-provided text",
        "llm_skipped_samples": sum(row["llm_skipped"] for row in rows),
        "current": statistics(rows, "current_tokens", args.budget),
        "all_schemas": statistics(rows, "all_schemas_tokens", args.budget),
        "top_10_all_schemas": sorted(
            rows, key=lambda row: row["all_schemas_tokens"], reverse=True
        )[:10],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = json.dumps(summary, ensure_ascii=False, indent=2)
    (args.output_dir / "summary.json").write_text(
        report + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()