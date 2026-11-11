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
    p.add_argument("--asr-precision", choices=["int8", "fp32", "both", "all"], default="all",
                   help="GIPformer assets to stage into models/asr (all includes fp32/int8 ONNX + bpe.model + model.pt)")
    p.add_argument("--asr-ft-repo", default=None,
                   help="Optional Hugging Face repo containing fine-tuned ONNX checkpoint to stage into models/asr_ft")
    p.add_argument("--asr-ft-subdir", default="asr-gipformer",
                   help="Subfolder inside --asr-ft-repo containing ONNX files when staging models/asr_ft")
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
            if args.asr_precision in {"both", "all"}:
                suffixes = [".int8.onnx", ".onnx"]
            elif args.asr_precision == "int8":
                suffixes = [".int8.onnx"]
            else:
                suffixes = [".onnx"]
            onnx_files = [part_name + suffix for suffix in suffixes for part_name in ("encoder", "decoder", "joiner")]
            patterns = ["tokens.txt", "README.md", *onnx_files]
            required = ["tokens.txt", *onnx_files]
            if args.asr_precision == "all":
                patterns.extend(["bpe.model", "bpe.vocab", "config.json", "model.pt"])
                required.extend(["bpe.model", "model.pt"])
            options["allow_patterns"] = patterns
        snapshot_download(repo_id=repo, local_dir=str(target), revision=revision, **options)
        if part == "asr":
            for filename in required:
                path = target / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(f"Missing or empty GIPformer artifact: {path}")
    if args.asr_ft_repo and args.only in ("both", "asr"):
        ft_target = args.models_dir / "asr_ft"
        ft_target.mkdir(parents=True, exist_ok=True)
        subdir = args.asr_ft_subdir.strip("/")
        patterns = ([f"{subdir}/*.onnx", f"{subdir}/tokens.txt", f"{subdir}/bpe.*", f"{subdir}/*.json"]
                    if subdir else ["*.onnx", "tokens.txt", "bpe.*", "*.json"])
        print(f"Fetching fine-tuned ASR {args.asr_ft_repo} ({subdir or '.'}) -> {ft_target.resolve()}")
        snapshot_download(repo_id=args.asr_ft_repo, local_dir=str(ft_target), allow_patterns=patterns)
        if subdir and (ft_target / subdir).is_dir():
            import shutil
            for item in (ft_target / subdir).iterdir():
                dest = ft_target / item.name
                if dest.exists():
                    continue
                shutil.move(str(item), str(dest))
        if not (ft_target / "tokens.txt").is_file() or not (
            (ft_target / "encoder.onnx").is_file() or (ft_target / "encoder.int8.onnx").is_file()
        ):
            raise FileNotFoundError(f"Missing fine-tuned GIPformer ONNX artifacts in {ft_target}")
    print("Models downloaded. Model inference will use local paths only.")


if __name__ == "__main__":
    main()
