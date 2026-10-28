#!/usr/bin/env python3
"""Convert TRAIN gold calls to observed next-action dialogues; split by sample ID."""
import argparse
import json
from pathlib import Path
import sys

from main import resolve_data_layout
from lifemate.catalog import load_catalog
from lifemate.sft import file_sha256, replay_sample, split_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tool-details", type=int, default=35)
    p.add_argument("--asr-model", type=Path)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--asr-compute-type", default="float16")
    args = p.parse_args()
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
        asr = LocalASR(args.asr_model, device=args.device, compute_type=args.asr_compute_type)
    metadata = dict(source=str(source), source_sha256=file_sha256(source),
                    catalog_sha256=file_sha256(catalog_path), seed=args.seed,
                    max_tool_details=args.max_tool_details, partitions={})
    for partition, rows in (("train", train), ("validation", validation)):
        path = args.output_dir / f"{partition}.jsonl"
        count = 0
        # Partial preparations stay marked .partial and are never consumed by the trainer.
        with path.with_suffix(".partial").open("w", encoding="utf-8") as f:
            for i, record in enumerate(rows):
                env = ToolEnv("train", record["id"], root=str(env_root))
                for example in replay_sample(record, catalog, env, split_dir, asr, args.max_tool_details):
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                print(f"{partition}: {i+1}/{len(rows)} samples, {count} SFT steps", flush=True)
        path.with_suffix(".partial").replace(path)
        metadata["partitions"][partition] = dict(sample_ids=[r["id"] for r in rows],
            samples=len(rows), examples=count, sha256=file_sha256(path))
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared TRAIN-only SFT data:", args.output_dir)
    if holder is not None:
        holder.cleanup()


if __name__ == "__main__":
    main()
