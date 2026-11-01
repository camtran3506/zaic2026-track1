#!/usr/bin/env python3
"""Run once on a connected machine, never during offline inference."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "llm": "Qwen/Qwen3.5-4B",
    "asr": "g-group-ai-lab/gipformer1.5-68M-rnnt",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "models",
                   help="Download root (default: code/models, independent of working directory)")
    p.add_argument("--only", choices=["llm", "asr", "both"], default="both")
    p.add_argument("--llm-revision", default=None, help="Pin a git commit before final submission")
    p.add_argument("--asr-revision", default=None, help="Pin a git commit before final submission")
    p.add_argument("--asr-precision", choices=["int8", "fp32", "both"], default="int8",
                   help="GIPformer ONNX files to stage; inference defaults to int8")
    args = p.parse_args()
    for part, repo in MODELS.items():
        if args.only not in ("both", part):
            continue
        target = args.models_dir / part
        target.mkdir(parents=True, exist_ok=True)
        revision = args.llm_revision if part == "llm" else args.asr_revision
        print(f"Fetching {repo} -> {target.resolve()}")
        options = {}
        if part == "asr":
            suffixes = [".int8.onnx", ".onnx"] if args.asr_precision == "both" else [
                ".int8.onnx" if args.asr_precision == "int8" else ".onnx"]
            files = [part + suffix for suffix in suffixes for part in ("encoder", "decoder", "joiner")]
            options["allow_patterns"] = ["tokens.txt", "README.md", *files]
        snapshot_download(repo_id=repo, local_dir=str(target), revision=revision, **options)
        if part == "asr":
            for filename in ["tokens.txt", *files]:
                path = target / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty GIPformer artifact: {path}")
    print("Models downloaded. Model inference will use local paths only.")


if __name__ == "__main__":
    main()
