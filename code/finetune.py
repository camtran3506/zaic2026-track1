#!/usr/bin/env python3
"""One command: resolve HF weights -> replay TRAIN -> LoRA/QLoRA -> optional merge."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--base-model", required=True, help="Local directory or Hugging Face model ID")
    p.add_argument("--revision", default="main", help="HF commit or revision; resolved SHA is logged")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--asr-model", type=Path, help="Optional fallback for train audio/history without text")
    p.add_argument("--asr-device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--asr-compute-type", default="float16")
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--max-tool-details", type=int, default=35)
    p.add_argument("--missing-tool-policy", choices=["error", "skip"], default="error")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-merged", action="store_true", help="Needs RAM/disk for full-precision base + adapter")
    # Remaining args are forwarded to train_lora.py, e.g. --max-steps or --quantization.
    args, training_options = p.parse_known_args()
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
               "--max-tool-details", args.max_tool_details, "--missing-tool-policy", args.missing_tool_policy]
    if args.asr_model:
        prepare.extend(["--asr-model", args.asr_model, "--device", args.asr_device,
                        "--asr-compute-type", args.asr_compute_type])
    run("prepare_sft.py", prepare)
    run("train_lora.py", [*training_options, "--base-model", base, "--sft-dir", args.output_dir / "sft",
                          "--output-dir", args.output_dir / "training", "--seed", args.seed])
    if args.export_merged:
        run("merge_lora.py", ["--base-model", base, "--adapter", args.output_dir / "training/adapter",
                              "--output-dir", args.output_dir / "merged"])
    print("Fine-tune completed:", args.output_dir)


if __name__ == "__main__":
    main()
