#!/usr/bin/env python3
"""Convert TRAIN gold calls to observed next-action dialogues; split by sample ID."""
import argparse
import json
from pathlib import Path
import sys

from main import resolve_data_layout
from lifemate.catalog import load_catalog
from lifemate.sft import (DEFAULT_ASR_TRANSCRIPT_RATIO, SFT_PROMPT_POLICY, VALIDATION_ROLLOUT_POLICY, UnexposedToolError,
                         asr_transcript_ratio, assign_transcript_sources, file_sha256,
                         prepare_train_transcript, prepare_validation_transcripts, replay_sample)
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device
from lifemate.text import TEXT_NORMALIZATION
from lifemate.itn import ITN_POLICY
from lifemate.execution_policy import EXECUTION_POLICY
from lifemate.local_time import LOCAL_TIME_POLICY
from lifemate.local_helpers import LOCAL_HELPER_POLICY, helper_tool_schemas
from lifemate.specialist_training import SPECIALIST_TRAINING_POLICY
from lifemate.temporal_training import TEMPORAL_TRAINING_POLICY
from lifemate.data_generalization import (POLICY as GENERALIZATION_POLICY, AppAugmenter, add_arguments,
    load_groups, split_generalization, assert_no_heldout, validate_options)
from lifemate.context import MAX_CONTEXT_TOKENS, DEFAULT_MAX_NEW_TOKENS, validate_training_length
from lifemate.batching import BATCH_POLICY, BATCH_TOOL, MAX_READ_BATCH_CALLS, group_examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(p)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing-tool-policy", choices=["error", "skip"], default="error",
                   help="Skip/report whole samples whose gold tools are absent from prompts, or stop")
    p.add_argument("--asr-transcript-ratio", type=asr_transcript_ratio, default=DEFAULT_ASR_TRANSCRIPT_RATIO,
                   help="Fraction of SFT sample IDs using ASR instead of audio_text (default 0.5); rollout selection always uses ASR")
    p.add_argument("--asr-model", type=Path, help="Required for ASR validation rollouts at every training mix ratio")
    p.add_argument("--device", choices=["cpu"], default="cpu", help="ASR is CPU-only")
    p.add_argument("--asr-precision", choices=["int8", "fp32"], default="int8")
    p.add_argument("--asr-threads", type=int, default=4)
    add_arguments(p)
    p.add_argument("--tokenizer", type=Path, help="Local base model/tokenizer, required for augmentation token-budget checks")
    p.add_argument("--max-length", type=int, default=MAX_CONTEXT_TOKENS)
    args = p.parse_args()
    configure_cuda_device(args)
    validate_options(args)
    validate_training_length(args.max_length)
    if (args.rename_variants or args.distractor_probability) and args.tokenizer is None:
        p.error("--tokenizer is required for app/tool augmentation")
    if args.asr_model is None:
        p.error("--asr-model is required for ASR validation rollouts, even when --asr-transcript-ratio is 0")
    split_dir, env_root, holder = resolve_data_layout(args.data_root, "train")
    source = split_dir / "train.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    catalog_path = split_dir / "train_tools_catalog.json"
    catalog = load_catalog(catalog_path)
    groups = load_groups(args.app_groups_file, catalog)
    partitions, holdout = split_generalization(records, catalog, args, groups)
    train, validation = partitions['train'], partitions['validation']
    print('App-held-out split:', json.dumps({k: v for k, v in holdout.items() if k != 'exposed_apps_by_id'}), flush=True)
    # Round once for the whole source dataset and for the training partition;
    # Regular validation and optional app-held-out share the remainder.
    total_asr = int(len(records) * args.asr_transcript_ratio + 0.5)
    train_asr = int(len(train) * args.asr_transcript_ratio + 0.5)
    val_asr = total_asr - train_asr
    if partitions['app_heldout']:
        val_asr = max(total_asr-train_asr-len(partitions['app_heldout']),
                      min(total_asr-train_asr, int(len(validation)*args.asr_transcript_ratio+.5)))
    asr_counts = {"train": train_asr, "validation": val_asr, "app_heldout": total_asr-train_asr-val_asr}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory; prepared data is never overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer
        from lifemate.llm import configure_tokenizer
        tokenizer = configure_tokenizer(AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True), args.tokenizer)
    augmenter = AppAugmenter(catalog, train, holdout['apps'], groups, seed=args.seed,
        rename_variants=args.rename_variants, distractor_probability=args.distractor_probability,
        tokenizer=tokenizer, max_length=args.max_length)
    asr = None
    asr_settings = None
    if args.asr_model:
        from lifemate.models import LocalASR
        from lifemate.transcripts import asr_identity
        asr_settings = asr_identity(args.asr_model, args.asr_precision, args.asr_threads)
        asr = LocalASR(args.asr_model, precision=args.asr_precision, num_threads=args.asr_threads)
    metadata = dict(source=str(source), source_sha256=file_sha256(source),
                    catalog_sha256=file_sha256(catalog_path), seed=args.seed,
                    prompt_policy=SFT_PROMPT_POLICY, missing_tool_policy=args.missing_tool_policy,
                    text_normalization=TEXT_NORMALIZATION,
                    itn_policy=ITN_POLICY,
                    execution_policy=EXECUTION_POLICY,
                    local_time_policy=LOCAL_TIME_POLICY,
                    local_helper_policy=LOCAL_HELPER_POLICY,
                    specialist_training_policy=SPECIALIST_TRAINING_POLICY,
                    temporal_helper_training_policy=TEMPORAL_TRAINING_POLICY,
                    transcript_mixing={"policy": "seeded-sample-v1", "unit": "sample",
                                       "asr_ratio": args.asr_transcript_ratio, "seed": args.seed,
                                       "rounding": ("nearest-total-train-validation-app-heldout-remainder"
                                                    if partitions['app_heldout'] else
                                                    "nearest-total-and-train-validation-remainder")},
                    asr=asr_settings,
                    app_holdout=holdout, partitions={})
    metadata['batching'] = {'policy': BATCH_POLICY, 'max_read_calls': MAX_READ_BATCH_CALLS,
                           'max_helper_calls': None, 'max_helpers_per_sample': None,
                           'token_budget_checked': tokenizer is not None, 'max_new_tokens': DEFAULT_MAX_NEW_TOKENS}
    rollout_rows = {"validation": [], "app_heldout": []}
    for partition, rows in partitions.items():
        if not rows:
            continue
        sources = assign_transcript_sources(rows, args.asr_transcript_ratio, args.seed,
                                            count=asr_counts[partition])
        requested_counts = {name: list(sources.values()).count(name) for name in ("audio_text", "asr")}
        print(f"{partition} transcript assignment: {requested_counts}", flush=True)
        path = args.output_dir / f"{partition}.jsonl"
        count = 0
        helper_count = 0
        batch_count = 0
        ungrouped_count = 0
        helper_counts = {name: 0 for name in helper_tool_schemas()}
        retained_ids = []
        skipped = []
        # Partial preparations stay marked .partial and are never consumed by the trainer.
        with path.with_suffix(".partial").open("w", encoding="utf-8") as f:
            for i, record in enumerate(rows):
                env = ToolEnv("train", record["id"], root=str(env_root))
                try:
                    prepared = prepare_train_transcript(record, split_dir, asr,
                                                        transcript_source=sources[record["id"]])
                    if partition != "train":
                        asr_prepared, clean_prepared = prepare_validation_transcripts(record, split_dir, asr, prepared)
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
                if partition != "train":
                    gold_hash = env.get_state_hash()
                    if not isinstance(gold_hash, str) or not gold_hash:
                        raise ValueError(f"{record['id']}: ToolEnv returned an invalid gold state hash")
                    supplied_hash = record.get("label", {}).get("state_hash")
                    if supplied_hash is not None and supplied_hash != gold_hash:
                        raise ValueError(f"{record['id']}: replayed gold hash differs from the provided gold hash")
                    rollout_rows[partition].append({
                        "item": {k: v for k, v in record.items() if k not in ("label", "audio_text")},
                        "prepared_transcript": asr_prepared,
                        "clean_prepared_transcript": clean_prepared,
                        "gold_tool_calls": record["label"]["tool_calls"], "gold_state_hash": gold_hash,
                    })
                retained_ids.append(record["id"])
                if partition == 'train':
                    for example in examples:
                        assert_no_heldout(example, holdout['apps'])
                    augmented = augmenter.augment(record, examples)
                else:
                    augmented = []
                # Group after augmentation so aliases, helper omissions and token
                # budgets are evaluated on the actual trajectory being trained.
                trajectories = [examples]
                for row in augmented:
                    if len(trajectories) == 1 or trajectories[-1][-1]['augmentation']['variant'] != row['augmentation']['variant']:
                        trajectories.append([])
                    trajectories[-1].append(row)
                ungrouped_count += sum(map(len, trajectories))
                examples = [row for trajectory in trajectories
                            for row in group_examples(trajectory, tokenizer=tokenizer, max_length=args.max_length)]
                for example in examples:
                    target = json.loads(example["messages"][-1]["content"])
                    if target.get('name') == BATCH_TOOL:
                        batch_count += 1
                        targets = target['arguments']['calls']
                    else:
                        targets = [target]
                    for child in targets:
                        if child.get('name') in helper_counts:
                            helper_count += 1
                            helper_counts[child['name']] += 1
                    example["transcript_source"] = prepared["transcript_source"]
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
                print(f"{partition}: {i+1}/{len(rows)} samples, {count} SFT steps", flush=True)
        if not retained_ids:
            raise ValueError(f"No usable {partition} samples; inspect skipped report and candidate selection")
        path.with_suffix(".partial").replace(path)
        metadata["partitions"][partition] = dict(sample_ids=retained_ids,
            samples=len(retained_ids), source_samples=len(rows), skipped_samples=skipped,
            transcript_sources={sample_id: sources[sample_id] for sample_id in retained_ids},
            transcript_source_counts={name: sum(sources[sample_id] == name for sample_id in retained_ids)
                                      for name in ("audio_text", "asr")},
            requested_transcript_source_counts=requested_counts,
            examples=count, ungrouped_examples=ungrouped_count, batch_examples=batch_count,
            local_helper_examples=helper_count,
            local_helper_examples_by_tool=helper_counts, sha256=file_sha256(path))
        print(f"{partition} summary: kept {len(retained_ids)}/{len(rows)}, skipped {len(skipped)}, {count} steps; "
              f"transcript sources: {metadata['partitions'][partition]['transcript_source_counts']}", flush=True)
    for partition, rows in rollout_rows.items():
        if not rows:
            continue
        if metadata['partitions'][partition]['skipped_samples']:
            raise ValueError('State/tool evaluation requires every held-out sample; do not skip them')
        rollout_path = args.output_dir / f'{partition}_rollouts.jsonl'
        rollout_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        metadata[f'{partition}_rollouts'] = {'path': rollout_path.name, 'sha256': file_sha256(rollout_path),
            'policy': VALIDATION_ROLLOUT_POLICY, 'selection_transcript_source': 'asr',
            'clean_view_samples': sum(r['clean_prepared_transcript'] is not None for r in rows)}
    metadata['training_augmentation'] = {'policy': GENERALIZATION_POLICY, 'rename_variants': args.rename_variants,
        'distractor_probability': args.distractor_probability, 'distractor_range': [1, 3],
        'max_length': args.max_length, 'groups': {k: sorted(v) for k, v in groups.items()}, **augmenter.stats}
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared TRAIN-only SFT data:", args.output_dir)
    if holder is not None:
        holder.cleanup()


if __name__ == "__main__":
    main()
