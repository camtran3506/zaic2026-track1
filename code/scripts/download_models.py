#!/usr/bin/env python3
"""Run once on a connected machine, never during offline inference."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "llm": "Qwen/Qwen3-4B-Instruct-2507",
    "asr": "vinai/PhoWhisper-large",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "models",
                   help="Download root (default: code/models, independent of working directory)")
    p.add_argument("--only", choices=["llm", "asr", "both"], default="both")
    p.add_argument("--llm-revision", default=None, help="Pin a git commit before final submission")
    p.add_argument("--asr-revision", default=None, help="Pin a git commit before final submission")
    args = p.parse_args()
    for part, repo in MODELS.items():
        if args.only not in ("both", part):
            continue
        target = args.models_dir / part
        target.mkdir(parents=True, exist_ok=True)
        revision = args.llm_revision if part == "llm" else args.asr_revision
        print(f"Fetching {repo} -> {target.resolve()}")
        snapshot_download(repo_id=repo, local_dir=str(target), revision=revision)
    print("Models downloaded. Model inference will use local paths only.")


if __name__ == "__main__":
    main()
