#!/usr/bin/env python3
"""Convert TRAIN gold calls to observed next-action dialogues; split by sample ID."""
import argparse
import json
from pathlib import Path
import sys

from main import resolve_data_layout
from lifemate.catalog import load_catalog
from lifemate.sft import SFT_PROMPT_POLICY, UnexposedToolError, file_sha256, prepare_train_transcript, replay_sample, split_records
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(p)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing-tool-policy", choices=["error", "skip"], default="error",
                   help="Skip/report whole samples whose gold tools are absent from prompts, or stop")
    p.add_argument("--asr-model", type=Path)
    p.add_argument("--device", choices=["cpu"], default="cpu", help="ASR is CPU-only")
    p.add_argument("--asr-precision", choices=["int8", "fp32"], default="int8")
    p.add_argument("--asr-threads", type=int, default=4)
    args = p.parse_args()
    configure_cuda_device(args)
    split_dir, env_root, holder = resolve_data_layout(args.data_root, "train")
    source = split_dir / "train.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    train, validation = split_records(records, args.validation_fraction, args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory; prepared data is never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    catalog_path = split_dir / "train_tools_catalog.json"
    catalog = load_catalog(catalog_path)
    asr = None
    if args.asr_model:
        from lifemate.models import LocalASR
        asr = LocalASR(args.asr_model, precision=args.asr_precision, num_threads=args.asr_threads)
    metadata = dict(source=str(source), source_sha256=file_sha256(source),
                    catalog_sha256=file_sha256(catalog_path), seed=args.seed,
                    prompt_policy=SFT_PROMPT_POLICY, missing_tool_policy=args.missing_tool_policy,
                    partitions={})
    rollout_rows = []
    for partition, rows in (("train", train), ("validation", validation)):
        path = args.output_dir / f"{partition}.jsonl"
        count = 0
        retained_ids = []
        skipped = []
        # Partial preparations stay marked .partial and are never consumed by the trainer.
        with path.with_suffix(".partial").open("w", encoding="utf-8") as f:
            for i, record in enumerate(rows):
                env = ToolEnv("train", record["id"], root=str(env_root))
                try:
                    prepared = prepare_train_transcript(record, split_dir, asr)
                    examples = replay_sample(record, catalog, env, split_dir, asr,
                                             prepared_transcript=prepared)
                except UnexposedToolError as exc:
                    if args.missing_tool_policy == "error":
                        raise
                    reason = {"sample_id": record["id"], "reason": str(exc)}
                    skipped.append(reason)
                    with (args.output_dir / f"{partition}_skipped.jsonl").open("a", encoding="utf-8") as report:
                        report.write(json.dumps(reason, ensure_ascii=False) + "\n")
                    print(f"WARNING skipped entire {partition} sample: {exc}", flush=True)
                    continue
                if partition == "validation":
                    gold_hash = env.get_state_hash()
                    if not isinstance(gold_hash, str) or not gold_hash:
                        raise ValueError(f"{record['id']}: ToolEnv returned an invalid gold state hash")
                    supplied_hash = record.get("label", {}).get("state_hash")
                    if supplied_hash is not None and supplied_hash != gold_hash:
                        raise ValueError(f"{record['id']}: replayed gold hash differs from the provided gold hash")
                    rollout_rows.append({
                        "item": {k: v for k, v in record.items() if k not in ("label", "audio_text")},
                        "prepared_transcript": prepared,
                        "gold_tool_calls": record["label"]["tool_calls"], "gold_state_hash": gold_hash,
                    })
                retained_ids.append(record["id"])
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                print(f"{partition}: {i+1}/{len(rows)} samples, {count} SFT steps", flush=True)
        if not retained_ids:
            raise ValueError(f"No usable {partition} samples; inspect skipped report and candidate selection")
        path.with_suffix(".partial").replace(path)
        metadata["partitions"][partition] = dict(sample_ids=retained_ids,
            samples=len(retained_ids), source_samples=len(rows), skipped_samples=skipped,
            examples=count, sha256=file_sha256(path))
        print(f"{partition} summary: kept {len(retained_ids)}/{len(rows)}, skipped {len(skipped)}, {count} steps", flush=True)
    if metadata["partitions"]["validation"]["skipped_samples"]:
        raise ValueError("State/tool checkpoint selection requires every held-out validation sample; do not skip them")
    rollout_path = args.output_dir / "validation_rollouts.jsonl"
    rollout_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rollout_rows), encoding="utf-8")
    metadata["validation_rollouts"] = {"path": rollout_path.name, "sha256": file_sha256(rollout_path)}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared TRAIN-only SFT data:", args.output_dir)
    if holder is not None:
        holder.cleanup()


if __name__ == "__main__":
    main()
