#!/usr/bin/env python3
"""LifeMate offline baseline: configurable data/model/output paths."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Must be set before loading any model. No network at inference.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from lifemate.catalog import load_catalog
from lifemate.agent import AgentConfig, LifeMateAgent
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device
from lifemate.context import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS, validate_inference_budget
from lifemate.runtime import DEFAULT_LLM_DTYPE

LOG = logging.getLogger("lifemate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline LifeMate ASR -> LLM -> ToolEnv baseline")
    add_cuda_device_argument(p)
    p.add_argument("--data-root", type=Path, required=True,
                   help="Folder holding <split>/ OR the split folder itself (e.g. /data in Docker)")
    p.add_argument("--split", choices=["train", "public_test", "private_test"], default="public_test")
    p.add_argument("--stage", choices=["full", "asr", "llm"], default="full",
                   help="full runs ASR then LLM in separate processes; llm only reads a complete transcript cache")
    p.add_argument("--transcript-cache", type=Path, help="JSONL of all current/history transcripts, plus manifest")
    p.add_argument("--refresh-transcripts", action="store_true", help="Replace completed ASR cache; resume matching partial work")
    p.add_argument("--asr-model", type=Path, help="Local GIPformer ONNX directory; not required for --stage llm")
    p.add_argument("--llm-model", type=Path, help="Local HF Transformers LLM directory; not required for --stage asr")
    p.add_argument("--llm-device-map", choices=["single", "balanced"], default="single")
    p.add_argument("--llm-max-memory-gib", type=float, default=6,
                   help="Weight budget per GPU for balanced sharding (reserve activation/KV memory)")
    p.add_argument("--llm-adapter", type=Path, default=None, help="Optional local fine-tuned LoRA adapter")
    p.add_argument("--output", type=Path, default=Path("result/submission.jsonl"))
    p.add_argument("--log-path", type=Path, default=None, help="Optional diagnostic JSONL, never submitted")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--asr-precision", default="int8", choices=["int8", "fp32"])
    p.add_argument("--asr-threads", type=int, default=4, help="GIPformer CPU threads; independent of LLM device")
    p.add_argument("--llm-dtype", default=DEFAULT_LLM_DTYPE, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-invalid", type=int, default=3)
    p.add_argument("--sample-seconds", type=float, default=float("inf"),
                   help="LLM agent time budget, default inf = no per-sample timeout")
    p.add_argument("--use-train-transcript", action="store_true", help="TRAIN ONLY: ablate ASR with train audio_text")
    p.add_argument("--limit", type=int, default=0, help="0 means run entire split; debugging only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fail-fast", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_data_layout(data_root: Path, split: str):
    """Return (split_dir, ToolEnv-root, temp owner). Symlink solves /data being split itself.

    Only creates a directory link; never opens/reads/copies/parses environment files.
    """
    location = data_root.expanduser().resolve()
    if (location / split / f"{split}.jsonl").is_file():
        return location / split, location, None
    if (location / f"{split}.jsonl").is_file():
        holder = tempfile.TemporaryDirectory(prefix="lifemate_split_root_")
        alias = Path(holder.name) / split
        alias.symlink_to(location, target_is_directory=True)
        return location, Path(holder.name), holder
    raise FileNotFoundError(
        f"Cannot locate {split}.jsonl under {location} or {location / split}. "
        "Point --data-root either to the split or to its parent."
    )


def run_child_stage(args, stage):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--stage", stage]
    for key, value in vars(args).items():
        if key == "stage" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    log_path = (args.transcript_cache.with_suffix(".asr.log") if stage == "asr"
                else args.output.with_suffix(".llm.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace") as child:
            for line in child.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            if child.wait():
                raise RuntimeError(f"{stage.upper()} stage failed; see {log_path}")


def run_asr_stage(args, items, split_dir, identity):
    from lifemate.transcripts import asr_identity, prepare_transcripts
    from lifemate.models import LocalASR
    settings = asr_identity(args.asr_model, args.asr_precision, args.asr_threads)
    def factory():
        return LocalASR(args.asr_model, precision=args.asr_precision, num_threads=args.asr_threads)
    prepare_transcripts(items, split_dir, args.transcript_cache, identity, settings, factory,
                        refresh=args.refresh_transcripts)


def run_llm_stage(args, items, transcript_rows, split_dir, env_root):
    # Import only the zaic_tools version shipped with this exact split.
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    seed_everything(args.seed)
    catalog = load_catalog(split_dir / f"{args.split}_tools_catalog.json")
    from lifemate.models import LocalLLM
    LOG.info("LLM-only stage: reading complete transcript cache; GIPformer is not loaded")
    llm = LocalLLM(args.llm_model, device=args.device, dtype=args.llm_dtype,
                   max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                   adapter_path=args.llm_adapter, device_map=args.llm_device_map,
                   max_memory_gib=args.llm_max_memory_gib, seed=args.seed)
    cfg = AgentConfig(max_calls=args.max_calls, max_invalid=args.max_invalid,
                      sample_seconds=args.sample_seconds,
                      use_train_transcript=args.use_train_transcript)
    if args.use_train_transcript and args.split != "train":
        raise ValueError("--use-train-transcript is forbidden for test splits")
    agent = LifeMateAgent(asr=None, llm=llm, catalog=catalog, cfg=cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
    overall_start = time.monotonic()
    count = 0
    # Write progressively, one line per sample. A sample's env is created exactly once.
    with args.output.open("w", encoding="utf-8") as fout:
        log_file = args.log_path.open("w", encoding="utf-8") if args.log_path else None
        try:
            for item, prepared in zip(items, transcript_rows):
                sample_start = time.monotonic()
                env = ToolEnv(args.split, item["id"], root=str(env_root))
                try:
                    report = agent.run(item, env, split_dir, args.split, prepared_transcript=prepared)
                except Exception as exc:
                    LOG.exception("Sample %s encountered an error", item["id"])
                    report = {"error": repr(exc), "transcript": prepared["transcript"]}
                    if args.fail_fast:
                        raise
                calls = env.get_tool_calls()
                state_hash = env.get_state_hash()
                record = {"id": item["id"], "tool_calls": calls, "state_hash": state_hash}
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                count += 1
                sample_time = time.monotonic() - sample_start
                LOG.info("%d %s: %d calls, %.2fs", count, item["id"], len(calls), sample_time)
                if log_file:
                    log_file.write(json.dumps({"id": item["id"], "sample_seconds": round(sample_time, 3),
                        "asr_seconds": prepared["asr_seconds"],
                        "total_sample_seconds": round(sample_time + prepared["asr_seconds"], 6), **report}, ensure_ascii=False) + "\n")
                    log_file.flush()
        finally:
            if log_file:
                log_file.close()
    LOG.info("Finished %d samples in %.1fs. Submission: %s", count, time.monotonic()-overall_start, args.output)


def main() -> None:
    args = parse_args()
    validate_inference_budget(args.max_input_tokens, args.max_new_tokens)
    configure_cuda_device(args)
    if args.stage == "asr":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.asr_threads < 1:
        raise ValueError("--asr-threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.use_train_transcript and args.split != "train":
        raise ValueError("--use-train-transcript is forbidden for test splits")
    if args.limit < 0 or math.isnan(args.sample_seconds) or args.sample_seconds <= 0:
        raise ValueError("Use nonnegative --limit and positive --sample-seconds (inf = unlimited)")
    if args.stage in ("full", "asr") and args.asr_model is None:
        raise ValueError("--asr-model is required for the ASR stage")
    if args.stage in ("full", "llm") and args.llm_model is None:
        raise ValueError("--llm-model is required for the LLM stage")
    LOG.info("Starting stage=%s pid=%d", args.stage, os.getpid())
    for name in ("asr_model", "llm_model", "llm_adapter"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser().resolve()
            setattr(args, name, path)
            LOG.info("%s=%s", name.upper(), path)
    if args.transcript_cache is None:
        args.transcript_cache = args.output.with_suffix(".transcripts.jsonl")
    if args.log_path is None:
        args.log_path = args.output.with_suffix(".diagnostics.jsonl")
    if len({args.output.resolve(), args.transcript_cache.resolve(), args.log_path.resolve()}) != 3:
        raise ValueError("Submission, transcripts and diagnostics need distinct paths")
    if args.stage == "full":
        # Parent never initializes either model or a CUDA context. Process exit frees
        # all ASR runtime allocations before LLM initialization, including caches.
        run_child_stage(args, "asr")
        LOG.info("ASR CPU process exited; starting LLM stage")
        run_child_stage(args, "llm")
        return
    from lifemate.transcripts import dataset_identity, load_transcripts, read_jsonl
    split_dir, env_root, holder = resolve_data_layout(args.data_root, args.split)
    try:
        protected = {(split_dir / f"{args.split}.jsonl").resolve(),
                     (split_dir / f"{args.split}_tools_catalog.json").resolve()}
        if any(path.resolve() in protected for path in (args.output, args.log_path, args.transcript_cache)):
            raise ValueError("Output/cache/log paths cannot overwrite dataset files")
        items = read_jsonl(split_dir / f"{args.split}.jsonl")
        if args.limit:
            items = items[:args.limit]
        identity = dataset_identity(items, split_dir, args.split, args.use_train_transcript)
        if args.stage == "asr":
            run_asr_stage(args, items, split_dir, identity)
        else:
            # Validate completeness, IDs, source fingerprint and checksum BEFORE loading LLM.
            rows, manifest = load_transcripts(args.transcript_cache, identity)
            from lifemate.asr import ASR_BACKEND, ASR_REPO_ID
            if manifest["asr"].get("backend") != ASR_BACKEND or manifest["asr"].get("repo_id") != ASR_REPO_ID:
                raise ValueError("Transcript ASR source is not GIPformer 1.5; rerun ASR with --refresh-transcripts")
            LOG.info("Transcript provenance: %s", json.dumps(manifest["asr"], ensure_ascii=False))
            run_llm_stage(args, items, rows, split_dir, env_root)
    finally:
        if holder is not None:
            holder.cleanup()


if __name__ == "__main__":
    main()
