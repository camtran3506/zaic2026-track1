#!/usr/bin/env python3
"""One command: resolve HF weights -> replay TRAIN -> LoRA/QLoRA -> optional merge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from lifemate.gpu import add_cuda_device_argument, configure_cuda_device
from lifemate.sft import DEFAULT_ASR_TRANSCRIPT_RATIO, asr_transcript_ratio
from lifemate.data_generalization import add_arguments, validate_options
from lifemate.context import MAX_CONTEXT_TOKENS, validate_training_length


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(p)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--base-model", required=True, help="Local directory or Hugging Face model ID")
    p.add_argument("--revision", default="main", help="HF commit or revision; resolved SHA is logged")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--asr-model", type=Path, help="Local ASR weights; required when --asr-transcript-ratio > 0 or history needs ASR")
    p.add_argument("--asr-transcript-ratio", type=asr_transcript_ratio, default=DEFAULT_ASR_TRANSCRIPT_RATIO,
                   help="Fraction of sample IDs using ASR transcripts in mix mode (0 disables ASR)")
    p.add_argument("--transcript-mode", choices=["dual", "mix"], default="dual",
                   help="dual generates 100%% Gold + 100%% ASR copies when 0 < ratio < 1; mix splits by ratio")
    p.add_argument("--include-date-helpers", action="store_true",
                   help="Demonstrate calendar_reference / date helper calls despite user_context.date_reference")
    p.add_argument("--no-inject-recovery", dest="inject_recovery", action="store_false",
                   help="Disable injecting failed-search + recovery trajectories on eligible TRAIN samples")
    p.add_argument("--step-wise-sft", action="store_true",
                   help="Keep separate per-step SFT rows instead of full-trajectory packing")
    p.add_argument("--asr-device", choices=["cpu"], default="cpu")
    p.add_argument("--asr-precision", choices=["auto", "int8", "fp32"], default="auto")
    p.add_argument("--asr-decoding-method", choices=["modified_beam_search", "greedy_search"],
                   default="modified_beam_search")
    p.add_argument("--asr-beam-size", type=int, default=4)
    p.add_argument("--asr-threads", type=int, default=4)
    p.add_argument("--filter-damaged-asr", action="store_true",
                   help="Skip ASR SFT copy if exact-match spoken gold args are missing from ASR transcript")
    p.add_argument("--validation-fraction", type=float, choices=[0.0], default=0.0,
                   help="Training uses the complete train set, without validation")
    p.add_argument("--missing-tool-policy", choices=["error", "skip"], default="error")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-merged", action="store_true", help="Needs RAM/disk for full-precision base + adapter")
    add_arguments(p, enabled=False)
    p.add_argument('--max-length', type=int, default=MAX_CONTEXT_TOKENS)
    # Remaining args are forwarded to train_lora.py, e.g. --max-steps or --quantization.
    args, training_options = p.parse_known_args()
    configure_cuda_device(args)
    validate_options(args)
    validate_training_length(args.max_length)
    if args.heldout_apps != "none" or args.heldout_groups:
        p.error("Training uses all samples; --heldout-apps must be none and --heldout-groups must be empty")
    if args.asr_model is None and args.asr_transcript_ratio > 0:
        p.error("--asr-model is required when --asr-transcript-ratio > 0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty run directory. To resume training use train_lora.py directly.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    app = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    def run(script, options):
        command = [sys.executable, str(app / script), *map(str, options)]
        with (args.output_dir / (Path(script).stem + ".log")).open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, env=env, cwd=app, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, errors="replace") as child:
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if child.wait():
                    raise RuntimeError(f"{script} failed; see {log.name}")
    base = Path(args.base_model).expanduser()
    source = {"local": str(base.resolve())}
    if not base.is_dir():
        from huggingface_hub import HfApi, snapshot_download
        sha = HfApi().model_info(args.base_model, revision=args.revision).sha
        base = Path(snapshot_download(args.base_model, revision=sha))
        source = {"repo": args.base_model, "revision": sha, "local": str(base)}
    (args.output_dir / "base_model.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    # Offline after staging. Training/preparation will not contact model registries.
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    prepare = ["--data-root", args.data_root, "--output-dir", args.output_dir / "sft",
               "--validation-fraction", args.validation_fraction, "--seed", args.seed,
               "--asr-transcript-ratio", args.asr_transcript_ratio,
               "--transcript-mode", args.transcript_mode,
               "--missing-tool-policy", args.missing_tool_policy]
    if args.include_date_helpers:
        prepare.append("--include-date-helpers")
    if not args.inject_recovery:
        prepare.append("--no-inject-recovery")
    if args.step_wise_sft:
        prepare.append("--step-wise-sft")
    prepare.extend(['--tokenizer', base, '--max-length', args.max_length,
        '--rename-variants', args.rename_variants, '--distractor-probability', args.distractor_probability,
        '--heldout-apps', args.heldout_apps, '--heldout-fraction', args.heldout_fraction,
        '--heldout-groups', args.heldout_groups])
    if args.app_groups_file:
        prepare.extend(['--app-groups-file', args.app_groups_file])
    if args.asr_model:
        prepare.extend([
            "--asr-model", args.asr_model, "--device", args.asr_device,
            "--asr-precision", args.asr_precision, "--asr-decoding-method", args.asr_decoding_method,
            "--asr-beam-size", str(args.asr_beam_size), "--asr-threads", str(args.asr_threads),
        ])
    if args.filter_damaged_asr:
        prepare.append("--filter-damaged-asr")
    run("prepare_sft.py", prepare)
    run("train_lora.py", [*training_options, "--base-model", base, "--sft-dir", args.output_dir / "sft",
                          "--output-dir", args.output_dir / "training", "--seed", args.seed, '--max-length', args.max_length])
    if args.export_merged:
        run("merge_lora.py", ["--base-model", base, "--adapter", args.output_dir / "training/adapter",
                              "--output-dir", args.output_dir / "merged"])
    print("Fine-tune completed:", args.output_dir)


if __name__ == "__main__":
    main()
