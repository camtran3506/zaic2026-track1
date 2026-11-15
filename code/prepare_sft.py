#!/usr/bin/env python3
"""Convert all TRAIN gold calls to observed next-action dialogues."""
import argparse
import json
import math
from pathlib import Path
import sys

from main import resolve_data_layout
from lifemate.agent import _train_transcript
from lifemate.catalog import load_catalog
from lifemate.reasoning import (DEFAULT_REASONING_STEPWISE_RATIO, DEFAULT_SHORT_REASONING, REASONING_POLICY,
                                attach_reasoning_to_trajectory, select_reasoning_stepwise_rows)
from lifemate.recovery import (DEFAULT_GUARD_RECOVERY_RATIO, GUARD_RECOVERY_POLICY,
                               build_guard_recovery_examples, build_recovery_trajectory)
from lifemate.sft import (DEFAULT_ASR_TRANSCRIPT_RATIO, SFT_PROMPT_POLICY, SYNTHETIC_SFT_POLICY,
                         VALIDATION_ROLLOUT_POLICY, GoldGuardConflict, UnexposedToolError,
                         asr_transcript_ratio, assign_transcript_sources, collapse_trajectory, file_sha256,
                         is_asr_argument_damaged, is_synthetic_sample, prepare_train_transcript,
                         prepare_validation_transcripts, pretranscribe_sft_partition, replay_sample,
                         replay_synthetic_sample)
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device
from lifemate.asr import DEFAULT_SFT_ASR_MODEL
from lifemate.history import history_entries
from lifemate.text import TEXT_NORMALIZATION
from lifemate.itn import ITN_POLICY
from lifemate.execution_policy import EXECUTION_POLICY
from lifemate.agent import AgentConfig
from lifemate.missing_info_guard import MISSING_INFO_POLICY
from lifemate.local_time import LOCAL_TIME_POLICY
from lifemate.local_helpers import LOCAL_HELPER_POLICY, helper_tool_schemas
from lifemate.specialist_training import SPECIALIST_TRAINING_POLICY
from lifemate.temporal_training import TEMPORAL_TRAINING_POLICY
from lifemate.assist_training import ASSIST_TRAINING_POLICY
from lifemate.local_assist import ASSIST_POLICY
from lifemate.schema_prompt import SCHEMA_PROMPT_POLICY
from lifemate.data_generalization import (POLICY as GENERALIZATION_POLICY, AppAugmenter, add_arguments,
    exposed_apps, load_groups, split_generalization, assert_no_heldout, validate_options)
from lifemate.context import MAX_CONTEXT_TOKENS, DEFAULT_MAX_NEW_TOKENS, validate_training_length
from lifemate.batching import BATCH_POLICY, BATCH_TOOL, MAX_READ_BATCH_CALLS, group_examples


def split_sft_records(records, catalog, args, groups):
    """Only executable originals can enter state/hash rollout partitions."""
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate sample IDs')
    originals = [r for r in records if not is_synthetic_sample(r)]
    synthetic = [r for r in records if is_synthetic_sample(r)]
    if not originals:
        if args.validation_fraction or args.heldout_apps != 'none' or args.heldout_groups:
            raise ValueError('State/hash diagnostic splits require original samples with ToolEnv')
        return split_generalization(records, catalog, args, groups)
    partitions, holdout = split_generalization(originals, catalog, args, groups)
    for record in sorted(synthetic, key=lambda r: str(r['id'])):
        exposure = exposed_apps(record, catalog)
        if exposure & set(holdout['apps']):
            raise ValueError(f"{record['id']}: train-only synthetic sample exposes a held-out app")
        holdout['exposed_apps_by_id'][record['id']] = sorted(exposure)
        partitions['train'].append(record)
    return partitions, holdout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(p)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--validation-fraction", type=float, default=0.0,
                   help="Optional standalone diagnostic split; default uses all samples for training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing-tool-policy", choices=["error", "skip"], default="error",
                   help="Skip/report whole samples whose gold tools are absent from prompts, or stop")
    p.add_argument("--guard-conflict-policy", choices=["error", "skip"], default="skip",
                   help="Skip/report whole samples whose gold contradicts automatic guards, or stop")
    p.add_argument("--asr-transcript-ratio", type=asr_transcript_ratio, default=DEFAULT_ASR_TRANSCRIPT_RATIO,
                   help="Fraction of SFT sample IDs using ASR in mix mode; 0 disables ASR, 1 uses ASR only")
    p.add_argument("--transcript-mode", choices=["dual", "mix"], default="dual",
                   help="dual creates 100%% Gold + 100%% ASR copies when 0 < ratio < 1; mix splits by ratio")
    p.add_argument("--include-date-helpers", action="store_true",
                   help="Demonstrate calendar_reference / date helper calls despite user_context.date_reference")
    p.add_argument("--no-inject-recovery", dest="inject_recovery", action="store_false",
                   help="Disable injecting failed-search + recovery trajectories on eligible TRAIN samples")
    p.add_argument("--no-inject-guard-recovery", dest="inject_guard_recovery", action="store_false",
                   help="Disable injecting xref/validation/redundant-ask self-correction rows on TRAIN samples")
    p.add_argument("--guard-recovery-ratio", type=float, default=DEFAULT_GUARD_RECOVERY_RATIO,
                   help="Deterministic hash ratio of eligible TRAIN samples for guard self-correction rows (0..1)")
    p.add_argument("--short-reasoning", action=argparse.BooleanOptionalAction,
                   default=DEFAULT_SHORT_REASONING,
                   help="Add Short Checked Reasoning inside <think>...</think> (default: disabled)")
    p.add_argument("--reasoning-stepwise-ratio", type=float, default=DEFAULT_REASONING_STEPWISE_RATIO,
                   help="Fraction of intermediate call turns emitted as step-wise <think> rows in full-trajectory mode")
    p.add_argument("--step-wise-sft", action=argparse.BooleanOptionalAction, default=True,
                   help="Write step-wise SFT rows instead of full-trajectory packing (default: enabled)")
    p.add_argument("--asr-model", type=Path, default=DEFAULT_SFT_ASR_MODEL,
                   help="ASR ONNX weights for dual/mix transcripts; default: code/models/asr_ft (no base-ASR fallback)")
    p.add_argument("--device", choices=["cpu"], default="cpu", help="ASR is CPU-only")
    p.add_argument("--asr-precision", choices=["auto", "int8", "fp32"], default="auto")
    p.add_argument("--asr-decoding-method", choices=["modified_beam_search", "greedy_search"],
                   default="modified_beam_search")
    p.add_argument("--asr-beam-size", type=int, default=4)
    p.add_argument("--asr-hotwords", choices=["off", "auto"], default="off")
    p.add_argument("--asr-hotwords-score", type=float, default=1.5)
    p.add_argument("--asr-post-normalize", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--asr-threads", type=int, default=4)
    p.add_argument("--asr-batch-size", type=int, default=16,
                   help="Batch size for offline ASR stream decoding during SFT data prep")
    p.add_argument("--transcript-cache", type=Path, default=None,
                   help="Completed ASR transcript cache JSONL (with manifest) to reuse instead of running live ASR")
    p.add_argument("--filter-damaged-asr", action="store_true",
                   help="Drop contradictory TRAIN ASR copies when spoken exact-match gold arguments are lost in ASR")
    add_arguments(p)
    p.add_argument("--tokenizer", type=Path, help="Local base model/tokenizer, required for augmentation token-budget checks")
    p.add_argument("--max-length", type=int, default=MAX_CONTEXT_TOKENS)
    args = p.parse_args()
    configure_cuda_device(args)
    validate_options(args)
    validate_training_length(args.max_length)
    if args.asr_threads < 1:
        p.error("--asr-threads must be positive")
    if args.asr_beam_size < 1:
        p.error("--asr-beam-size must be positive")
    if not math.isfinite(args.asr_hotwords_score) or args.asr_hotwords_score <= 0:
        p.error("--asr-hotwords-score must be finite and positive")
    if args.asr_batch_size < 1:
        p.error("--asr-batch-size must be positive")
    if not 0.0 <= args.guard_recovery_ratio <= 1.0:
        p.error("--guard-recovery-ratio must be between 0 and 1")
    if not 0.0 <= args.reasoning_stepwise_ratio <= 1.0:
        p.error("--reasoning-stepwise-ratio must be between 0 and 1")
    if (args.rename_variants or args.distractor_probability) and args.tokenizer is None:
        p.error("--tokenizer is required for app/tool augmentation")
    split_dir, env_root, holder = resolve_data_layout(args.data_root, "train")
    source = split_dir / "train.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    catalog_path = split_dir / "train_tools_catalog.json"
    catalog = load_catalog(catalog_path)
    groups = load_groups(args.app_groups_file, catalog)
    partitions, holdout = split_sft_records(records, catalog, args, groups)
    train, validation = partitions['train'], partitions['validation']
    originals = [r for r in records if not is_synthetic_sample(r)]
    original_partitions = {name: [r for r in rows if not is_synthetic_sample(r)]
                           for name, rows in partitions.items()}
    synthetic_ids = {r['id'] for r in records if is_synthetic_sample(r)}
    print('App-held-out split:', json.dumps({k: v for k, v in holdout.items() if k != 'exposed_apps_by_id'}), flush=True)
    # Round once for the whole source dataset and for the training partition;
    # Regular validation and optional app-held-out share the remainder.
    total_asr = int(len(originals) * args.asr_transcript_ratio + 0.5)
    train_asr = int(len(original_partitions['train']) * args.asr_transcript_ratio + 0.5)
    val_asr = total_asr - train_asr
    if partitions['app_heldout']:
        val_asr = max(total_asr-train_asr-len(partitions['app_heldout']),
                      min(total_asr-train_asr, int(len(validation)*args.asr_transcript_ratio+.5)))
    asr_counts = {"train": train_asr, "validation": val_asr, "app_heldout": total_asr-train_asr-val_asr}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory; prepared data is never overwritten")
    asr = None
    asr_settings = None
    cached_transcripts = {}
    if args.transcript_cache is not None:
        from lifemate.asr import ASR_BACKEND, ASR_REPO_ID
        from lifemate.transcripts import dataset_identity, load_transcripts
        if not originals:
            raise ValueError("--transcript-cache requires non-synthetic train samples")
        cache_identity = dataset_identity(originals, split_dir, "train", False)
        cached_rows, cached_manifest = load_transcripts(args.transcript_cache, cache_identity)
        cached_asr = cached_manifest.get("asr", {})
        if cached_asr.get("backend") != ASR_BACKEND or cached_asr.get("repo_id") != ASR_REPO_ID:
            raise ValueError("Transcript ASR cache is not GIPformer 1.5; rebuild ASR cache")
        asr_settings = cached_asr
        cached_transcripts = {row["id"]: row for row in cached_rows}
        print(f"SFT ASR cache: {args.transcript_cache} ({len(cached_transcripts)} samples)", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ToolEnv = None
    if originals:
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
    needs_asr = bool(originals) and (
        args.asr_transcript_ratio > 0 or bool(validation) or bool(partitions['app_heldout'])
        or any(text is None and audio for r in originals
               for _role, text, audio in history_entries(r.get('history', []))))
    if needs_asr and not cached_transcripts:
        from lifemate.asr import asr_extra_options, resolve_asr_precision
        from lifemate.models import LocalASR
        from lifemate.transcripts import asr_identity
        precision = (
            resolve_asr_precision(args.asr_model, args.asr_precision)
            if args.asr_precision == "auto" and Path(args.asr_model).expanduser().is_dir()
            else args.asr_precision
        )
        extra = asr_extra_options(args)
        asr_settings = asr_identity(args.asr_model, precision, args.asr_threads, **extra)
        print(f'SFT ASR model: {args.asr_model} (precision={precision})', flush=True)
        asr = LocalASR(args.asr_model, precision=precision, num_threads=args.asr_threads, **extra)
    metadata = dict(source=str(source), source_sha256=file_sha256(source),
                    catalog_sha256=file_sha256(catalog_path), seed=args.seed,
                    prompt_policy=SFT_PROMPT_POLICY, missing_tool_policy=args.missing_tool_policy,
                    guard_conflict_policy=args.guard_conflict_policy,
                    text_normalization=TEXT_NORMALIZATION,
                    itn_policy=ITN_POLICY,
                    execution_policy=EXECUTION_POLICY,
                    training_guards={'missing_info_policy': MISSING_INFO_POLICY,
                                     'empty_read_limit': AgentConfig().empty_read_limit,
                                     'enforce_observed_xref': AgentConfig().enforce_observed_xref},
                    local_time_policy=LOCAL_TIME_POLICY,
                    local_helper_policy=LOCAL_HELPER_POLICY,
                    assist_policy=ASSIST_POLICY, assist_training_policy=ASSIST_TRAINING_POLICY,
                    schema_prompt_policy=SCHEMA_PROMPT_POLICY,
                    specialist_training_policy=SPECIALIST_TRAINING_POLICY,
                    temporal_helper_training_policy=TEMPORAL_TRAINING_POLICY,
                    reasoning_policy=REASONING_POLICY,
                    guard_recovery_policy=GUARD_RECOVERY_POLICY,
                    synthetic_training={'policy': SYNTHETIC_SFT_POLICY,
                                        'identification': 'id starts with synthetic_ or synthetic=true',
                                        'transcript_source': 'audio_text', 'dual': False,
                                        'tool_env': False, 'env_recovery': False},
                    transcript_mixing={"policy": "seeded-sample-v1", "unit": "sample",
                                       "asr_ratio": args.asr_transcript_ratio, "seed": args.seed,
                                       "rounding": ("nearest-total" if not validation and not partitions['app_heldout'] else
                                                    "nearest-total-train-validation-app-heldout-remainder"
                                                    if partitions['app_heldout'] else
                                                    "nearest-total-and-train-validation-remainder")},
                    asr=asr_settings,
                    app_holdout=holdout, partitions={})
    metadata['batching'] = {'policy': BATCH_POLICY, 'max_read_calls': MAX_READ_BATCH_CALLS,
                           'max_helper_calls': None, 'max_helpers_per_sample': None,
                           'token_budget_checked': tokenizer is not None, 'max_new_tokens': DEFAULT_MAX_NEW_TOKENS}
    full_trajectory_sft = (not args.step_wise_sft) and (getattr(tokenizer, "lifemate_native_fc", False) is True)
    metadata['sft_format'] = {
        'full_trajectory': full_trajectory_sft,
        'transcript_mode': args.transcript_mode,
        'dual_transcript_source_prompt': args.transcript_mode == 'dual' and 0 < args.asr_transcript_ratio < 1,
        'include_date_helpers': args.include_date_helpers,
        'inject_recovery': args.inject_recovery,
        'inject_guard_recovery': args.inject_guard_recovery,
        'guard_recovery_ratio': args.guard_recovery_ratio,
        'short_reasoning': args.short_reasoning,
        'reasoning_stepwise_ratio': args.reasoning_stepwise_ratio,
    }
    if args.transcript_cache is not None:
        metadata['sft_format']['transcript_cache'] = str(args.transcript_cache.expanduser().resolve())
    rollout_rows = {"validation": [], "app_heldout": []}
    for partition, rows in partitions.items():
        if not rows:
            continue
        dual_train = (partition == "train" and args.transcript_mode == "dual" and 0 < args.asr_transcript_ratio < 1)
        sources = assign_transcript_sources(original_partitions[partition], args.asr_transcript_ratio, args.seed,
                                            count=asr_counts[partition])
        sources.update({r['id']: 'audio_text' for r in rows if is_synthetic_sample(r)})
        if dual_train:
            requested_counts = {"audio_text": len(rows), "asr": len(original_partitions[partition])}
            print(f"{partition} transcript assignment: dual for {len(original_partitions[partition])} original samples; "
                  f"{len(rows)-len(original_partitions[partition])} synthetic samples use audio_text only", flush=True)
            if asr is not None:
                pretranscribe_sft_partition(rows, split_dir, asr, partition, {r["id"]: "asr" for r in rows},
                                            batch_size=args.asr_batch_size)
        else:
            requested_counts = {name: list(sources.values()).count(name) for name in ("audio_text", "asr")}
            print(f"{partition} transcript assignment: {requested_counts}", flush=True)
            if asr is not None:
                pretranscribe_sft_partition(rows, split_dir, asr, partition, sources,
                                            batch_size=args.asr_batch_size)
        path = args.output_dir / f"{partition}.jsonl"
        count = 0
        trajectory_count = 0
        recovery_count = 0
        guard_recovery_count = 0
        reasoning_stepwise_count = 0
        helper_count = 0
        batch_count = 0
        ungrouped_count = 0
        asr_damaged_filtered = 0
        written_source_counts = {"audio_text": 0, "asr": 0}
        helper_counts = {name: 0 for name in helper_tool_schemas()}
        retained_ids = []
        skipped = []
        # Partial preparations stay marked .partial and are never consumed by the trainer.
        with path.with_suffix(".partial").open("w", encoding="utf-8") as f:
            for i, record in enumerate(rows):
                synthetic = is_synthetic_sample(record)
                if synthetic:
                    variant_sources = ['audio_text']
                elif dual_train:
                    variant_sources = (["audio_text", "asr"] if _train_transcript(record) is not None else ["asr"])
                else:
                    variant_sources = [sources[record["id"]]]
                variant_runs = []
                sample_skipped = False
                for src in variant_sources:
                    env = None if synthetic else ToolEnv("train", record["id"], root=str(env_root))
                    try:
                        cached = None if synthetic else cached_transcripts.get(record["id"])
                        prepared = prepare_train_transcript(
                            record, split_dir, asr, transcript_source=src, cached_transcript=cached
                        )
                        if dual_train and not synthetic:
                            prepared['prompt_transcript_source'] = prepared['transcript_source']
                        if (partition == "train" and args.filter_damaged_asr and src == "asr"
                                and is_asr_argument_damaged(record, prepared, catalog)):
                            asr_damaged_filtered += 1
                            continue
                        if partition != "train" and not variant_runs:
                            asr_prepared, clean_prepared = prepare_validation_transcripts(
                                record, split_dir, asr, prepared, cached_transcript=cached
                            )
                        if synthetic:
                            examples = replay_synthetic_sample(
                                record, catalog, split_dir, prepared_transcript=prepared, tokenizer=tokenizer,
                                include_date_helpers=args.include_date_helpers,
                            )
                        else:
                            examples = replay_sample(
                                record, catalog, env, split_dir, asr,
                                prepared_transcript=prepared, tokenizer=tokenizer,
                                include_date_helpers=args.include_date_helpers,
                            )
                    except (UnexposedToolError, GoldGuardConflict) as exc:
                        guard_conflict = isinstance(exc, GoldGuardConflict)
                        policy = args.guard_conflict_policy if guard_conflict else args.missing_tool_policy
                        if policy == "error":
                            raise
                        reason = {"sample_id": record["id"], "reason": str(exc),
                                  "category": "automatic_guard_conflict" if guard_conflict else "unexposed_tool"}
                        if guard_conflict:
                            reason['guard_feedback'] = exc.feedback
                        skipped.append(reason)
                        with (args.output_dir / f"{partition}_skipped.jsonl").open("a", encoding="utf-8") as report:
                            report.write(json.dumps(reason, ensure_ascii=False) + "\n")
                        print(f"WARNING skipped entire {partition} sample: {exc}", flush=True)
                        sample_skipped = True
                        break
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
                    variant_runs.append((prepared, examples, env))
                if sample_skipped or not variant_runs:
                    continue
                retained_ids.append(record["id"])
                for prepared, examples, variant_env in variant_runs:
                    if partition == 'train':
                        for example in examples:
                            assert_no_heldout(example, holdout['apps'])
                        augmented = augmenter.augment(record, examples)
                    else:
                        augmented = []
                    # Group after augmentation so aliases, helper omissions and token
                    # budgets are evaluated on the actual trajectory being trained.
                    trajectories = [(examples, False)]
                    if partition == "train" and not synthetic and args.inject_recovery:
                        recovery_res = build_recovery_trajectory(
                            record, prepared, examples, catalog,
                            lambda rid=record["id"]: ToolEnv("train", rid, root=str(env_root)),
                        )
                        if recovery_res is not None:
                            rec_steps, _ = recovery_res
                            trajectories.append((rec_steps, True))
                            recovery_count += 1
                    aug_buckets = []
                    for row in augmented:
                        if not aug_buckets or aug_buckets[-1][-1]['augmentation']['variant'] != row['augmentation']['variant']:
                            aug_buckets.append([])
                        aug_buckets[-1].append(row)
                    for bucket in aug_buckets:
                        trajectories.append((bucket, False))
                    ungrouped_count += sum(len(t) for t, _ in trajectories)
                    for traj, is_recovery in trajectories:
                        grouped = traj if is_recovery else group_examples(traj, tokenizer=tokenizer, max_length=args.max_length)
                        if args.short_reasoning:
                            grouped = attach_reasoning_to_trajectory(grouped)
                        for example in grouped:
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
                        if full_trajectory_sft:
                            rows_to_write = [collapse_trajectory(grouped)]
                            if (partition == "train" and args.short_reasoning
                                    and args.reasoning_stepwise_ratio > 0):
                                extra_stepwise = select_reasoning_stepwise_rows(
                                    grouped, ratio=args.reasoning_stepwise_ratio, seed=args.seed,
                                )
                                rows_to_write.extend(extra_stepwise)
                                reasoning_stepwise_count += len(extra_stepwise)
                        else:
                            rows_to_write = [r for r in grouped if r.get("train", True) is not False]
                        trajectory_count += 1
                        written_source_counts[prepared["transcript_source"]] += 1
                        for row_out in rows_to_write:
                            row_out["transcript_source"] = prepared["transcript_source"]
                            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                            count += 1
                    if (partition == "train" and not synthetic and args.inject_guard_recovery
                            and args.guard_recovery_ratio > 0):
                        guard_rows = build_guard_recovery_examples(
                            record, prepared, examples, catalog,
                            variant_env,
                            ratio=args.guard_recovery_ratio, seed=args.seed,
                        )
                        for g_row in guard_rows:
                            assert_no_heldout(g_row, holdout['apps'])
                        if args.short_reasoning and guard_rows:
                            guard_rows = attach_reasoning_to_trajectory(guard_rows)
                        guard_recovery_count += len(guard_rows)
                        for g_row in guard_rows:
                            g_row["transcript_source"] = prepared["transcript_source"]
                            f.write(json.dumps(g_row, ensure_ascii=False) + "\n")
                            count += 1
                print(f"{partition}: {i+1}/{len(rows)} samples, {count} SFT rows", flush=True)
        if not retained_ids:
            raise ValueError(f"No usable {partition} samples; inspect skipped report and candidate selection")
        path.with_suffix(".partial").replace(path)
        source_summary = (written_source_counts if dual_train
                          else {name: sum(sources[sample_id] == name for sample_id in retained_ids)
                                for name in ("audio_text", "asr")})
        metadata["partitions"][partition] = dict(sample_ids=retained_ids,
            samples=len(retained_ids), source_samples=len(rows), skipped_samples=skipped,
            synthetic_samples=sum(sample_id in synthetic_ids for sample_id in retained_ids),
            trajectory_sources={sample_id: ('synthetic_recorded' if sample_id in synthetic_ids else 'tool_env')
                                for sample_id in retained_ids},
            transcript_sources={sample_id: ("dual" if dual_train and sample_id not in synthetic_ids
                                            else sources[sample_id]) for sample_id in retained_ids},
            transcript_source_counts=source_summary,
            requested_transcript_source_counts=requested_counts,
            examples=count, trajectories=trajectory_count, recovery_trajectories=recovery_count,
            guard_recovery_examples=guard_recovery_count,
            reasoning_stepwise_examples=reasoning_stepwise_count,
            asr_damaged_filtered=asr_damaged_filtered,
            ungrouped_examples=ungrouped_count, batch_examples=batch_count,
            local_helper_examples=helper_count,
            local_helper_examples_by_tool=helper_counts, sha256=file_sha256(path))
        print(f"{partition} summary: kept {len(retained_ids)}/{len(rows)}, skipped {len(skipped)}, {count} rows ({trajectory_count} trajectories, {recovery_count} recovery); "
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
    if asr is not None and hasattr(asr, "close"):
        asr.close()
    if holder is not None:
        holder.cleanup()


if __name__ == "__main__":
    main()
