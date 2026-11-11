"""Export a fine-tuned Gipformer PyTorch checkpoint to FP32 ONNX under code/models/asr_ft."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lifemate.agent import audio_path  # noqa: E402
from lifemate.asr import LocalASR  # noqa: E402
from scripts.asr_finetune.prepare_gipformer_sft import _resolve_asr_audio_dir  # noqa: E402
from scripts.asr_finetune.gipformer_pt import (  # noqa: E402
    DEFAULT_FINETUNED_ASR_DIR,
    DEFAULT_ICEFALL_ROOT,
    ORIGINAL_ASR_DIR,
    sha256,
    strict_model,
)


def export_finetuned_onnx(
    *,
    best: Path,
    original_dir: Path = ORIGINAL_ASR_DIR,
    output_dir: Path = DEFAULT_FINETUNED_ASR_DIR,
    icefall_root: Path = DEFAULT_ICEFALL_ROOT,
    data_root: Path = REPO_ROOT / "train+public_test",
) -> dict:
    import torch

    original_dir = Path(original_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == original_dir:
        raise ValueError("Refusing to overwrite original base ASR folder; export to a distinct directory such as models/asr_ft")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty export directory: {output_dir}")

    original, _ = strict_model(original_dir, icefall_root)
    saved = torch.load(best, map_location="cpu", weights_only=True)
    if saved.get("source_sha256") != sha256(original_dir / "model.pt"):
        raise ValueError("Adapted checkpoint source_sha256 differs from base models/asr/model.pt")
    if set(saved["model"]) != set(original.state_dict()):
        raise ValueError("Adapted checkpoint keys differ from strictly loaded Zipformer model")
    original.load_state_dict(saved["model"], strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokens.txt", "bpe.model", "bpe.vocab", "config.json"):
        src = original_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)
    for required_file in ("tokens.txt",):
        if not (output_dir / required_file).is_file():
            raise FileNotFoundError(f"Missing required artifact {output_dir / required_file}")

    torch.save({"model": saved["model"]}, output_dir / "model.pt")
    shutil.copy2(output_dir / "model.pt", output_dir / "epoch-1.pt")

    icefall_root = Path(icefall_root).expanduser().resolve()
    export_script = icefall_root / "egs" / "librispeech" / "ASR" / "zipformer" / "export-onnx.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"Missing icefall ONNX exporter: {export_script}")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([
        str(icefall_root),
        str(icefall_root / "egs" / "librispeech" / "ASR"),
        str(icefall_root / "egs" / "librispeech" / "ASR" / "zipformer"),
        env.get("PYTHONPATH", ""),
    ])
    subprocess.run(
        [
            sys.executable, str(export_script),
            "--tokens", str(output_dir / "tokens.txt"),
            "--exp-dir", str(output_dir),
            "--epoch", "1",
            "--avg", "1",
            "--use-averaged-model", "false",
            "--enable-int8-quantization", "0",
            "--use-ctc", "true",
        ],
        check=True,
        env=env,
    )
    for part in ("encoder", "decoder", "joiner"):
        built = output_dir / f"{part}-epoch-1-avg-1.onnx"
        if not built.is_file() or built.stat().st_size < 1024:
            raise FileNotFoundError(f"ONNX export missing or empty: {built}")
        built.rename(output_dir / f"{part}.onnx")
    (output_dir / "epoch-1.pt").unlink(missing_ok=True)
    (output_dir / "model.pt").unlink(missing_ok=True)

    # Smoke-test exported FP32 ONNX using modified_beam_search on CPU.
    split_dir = _resolve_asr_audio_dir(data_root)
    jsonl_path = (split_dir / "train.jsonl") if (split_dir / "train.jsonl").is_file() else (REPO_ROOT / "train_asr.jsonl")
    sample = json.loads(jsonl_path.open(encoding="utf-8").readline())
    recognizer = LocalASR(output_dir, precision="fp32", decoding_method="modified_beam_search")
    transcript = recognizer.transcribe(audio_path(split_dir, sample["audio"]))

    if not transcript:
        raise RuntimeError("Exported ONNX model produced an empty smoke transcript")
    if any(tok in transcript for tok in ("<sos/eos>", "<unk>", "<blk>")):
        raise RuntimeError(f"Exported ONNX model emitted special token: {transcript!r}")

    manifest = {
        "model": "Gipformer 1.5 LifeMate Fine-Tuned",
        "base_model_dir": str(original_dir),
        "source_sha256": saved["source_sha256"],
        "adapted_checkpoint": str(Path(best).resolve()),
        "adapted_sha256": sha256(Path(best)),
        "onnx_precision": "fp32",
        "onnx_sha256": {part: sha256(output_dir / f"{part}.onnx") for part in ("encoder", "decoder", "joiner")},
        "smoke_sample_id": sample["id"],
        "smoke_transcript": transcript,
    }
    (output_dir / "lifemate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, default=CODE_DIR / "result" / "asr_sft" / "best.pt")
    parser.add_argument("--original-dir", type=Path, default=ORIGINAL_ASR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FINETUNED_ASR_DIR,
                        help="Destination for fine-tuned FP32 ONNX (default: code/models/asr_ft)")
    parser.add_argument("--icefall-root", type=Path, default=DEFAULT_ICEFALL_ROOT)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "train+public_test")
    args = parser.parse_args()
    manifest = export_finetuned_onnx(
        best=args.best,
        original_dir=args.original_dir,
        output_dir=args.output_dir,
        icefall_root=args.icefall_root,
        data_root=args.data_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
