#!/usr/bin/env python3
"""LifeMate offline baseline: configurable data/model/output paths."""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
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

LOG = logging.getLogger("lifemate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline LifeMate ASR -> LLM -> ToolEnv baseline")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Folder holding <split>/ OR the split folder itself (e.g. /data in Docker)")
    p.add_argument("--split", choices=["train", "public_test", "private_test"], default="public_test")
    p.add_argument("--asr-model", type=Path, required=True, help="Local CTranslate2 Whisper directory")
    p.add_argument("--llm-model", type=Path, required=True, help="Local HF Transformers LLM directory")
    p.add_argument("--llm-adapter", type=Path, default=None, help="Optional local fine-tuned LoRA adapter")
    p.add_argument("--output", type=Path, default=Path("result/submission.jsonl"))
    p.add_argument("--log-path", type=Path, default=None, help="Optional diagnostic JSONL, never submitted")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--asr-compute-type", default="float16", choices=["float16", "int8_float16", "int8", "float32"])
    p.add_argument("--llm-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--asr-beam", type=int, default=1)
    p.add_argument("--max-input-tokens", type=int, default=6000)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-invalid", type=int, default=3)
    p.add_argument("--max-tool-details", type=int, default=35)
    p.add_argument("--sample-seconds", type=float, default=28.0)
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split_dir, env_root, _temp_holder = resolve_data_layout(args.data_root, args.split)
    # Import only the zaic_tools version shipped with this exact split.
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    seed_everything(args.seed)
    catalog = load_catalog(split_dir / f"{args.split}_tools_catalog.json")
    from lifemate.models import LocalASR, LocalLLM
    asr = LocalASR(args.asr_model, device=args.device, compute_type=args.asr_compute_type, beam_size=args.asr_beam)
    llm = LocalLLM(args.llm_model, device=args.device, dtype=args.llm_dtype,
                   max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
                   adapter_path=args.llm_adapter)
    cfg = AgentConfig(max_calls=args.max_calls, max_invalid=args.max_invalid,
                      max_tool_details=args.max_tool_details, sample_seconds=args.sample_seconds,
                      use_train_transcript=args.use_train_transcript)
    if args.use_train_transcript and args.split != "train":
        raise ValueError("--use-train-transcript is forbidden for test splits")
    agent = LifeMateAgent(asr=asr, llm=llm, catalog=catalog, cfg=cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)
    input_file = split_dir / f"{args.split}.jsonl"
    overall_start = time.monotonic()
    count = 0
    # Write progressively, one line per sample. A sample's env is created exactly once.
    with input_file.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        log_file = args.log_path.open("w", encoding="utf-8") if args.log_path else None
        try:
            for line in fin:
                if not line.strip():
                    continue
                if args.limit and count >= args.limit:
                    break
                item = json.loads(line)
                sample_start = time.monotonic()
                env = ToolEnv(args.split, item["id"], root=str(env_root))
                try:
                    report = agent.run(item, env, split_dir, args.split)
                except Exception as exc:
                    LOG.exception("Sample %s encountered an error", item["id"])
                    report = {"error": repr(exc)}
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
                    log_file.write(json.dumps({"id": item["id"], "sample_seconds": round(sample_time, 3), **report}, ensure_ascii=False) + "\n")
                    log_file.flush()
        finally:
            if log_file:
                log_file.close()
    LOG.info("Finished %d samples in %.1fs. Submission: %s", count, time.monotonic()-overall_start, args.output)


if __name__ == "__main__":
    main()
