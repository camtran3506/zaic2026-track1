"""Fine-tune Gipformer 1.5 (68M Zipformer-RNNT) on train_asr.jsonl prepared targets.

Training reads base weights from code/models/asr/model.pt (never overwriting it)
and saves adapted checkpoints to --output-dir (default: code/result/asr_sft) for
export_gipformer.py -> code/models/asr_ft.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

CODE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lifemate.agent import audio_path  # noqa: E402
from lifemate.asr import read_audio  # noqa: E402
from lifemate.asr_targets import build_asr_target, encode_bpe_ids, load_vocab_chars, normalized_word_error_rate  # noqa: E402
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device  # noqa: E402
from scripts.asr_finetune.prepare_gipformer_sft import _resolve_asr_audio_dir  # noqa: E402
from scripts.asr_finetune.gipformer_pt import (  # noqa: E402
    DEFAULT_ICEFALL_ROOT,
    ORIGINAL_ASR_DIR,
    sha256,
    strict_model,
)


def select_training_device(cuda_visible_devices: str, torch_module):
    """Select training device cleanly: explicit CPU (-1/''), otherwise cuda:0 across >=1 GPUs."""
    if str(cuda_visible_devices).strip() in ("", "-1"):
        return torch_module.device("cpu")
    if torch_module.cuda.is_available() and torch_module.cuda.device_count() >= 1:
        return torch_module.device("cuda:0")
    raise RuntimeError("No CUDA GPU available for Gipformer training; pass --cuda-visible-devices -1 for CPU")


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def resolve_training_rows(
    rows: list[dict],
    sp,
    vocab_size: int,
    *,
    hypotheses: dict[str, str],
    vocab_chars: set[str] | frozenset[str] | None = None,
    max_history_wer: float | None = None,
    max_main_wer: float | None = None,
    limit: int = 100000,
    is_train: bool = True,
) -> tuple[list[dict], int]:
    """Ensure every row has a valid SentencePiece BPE target, with optional WER filtering."""
    kept: list[dict] = []
    for row in rows:
        is_history = row.get("source") == "history"
        human_verified = bool(row.get("human_verified")) and not is_history
        target = row.get("target_text")
        hyp_text = hypotheses.get(row["audio"], "")
        if target and human_verified:
            ids = encode_bpe_ids(target, sp, vocab_size)
            if ids is not None and not (
                is_train and not is_history and max_main_wer is not None and hyp_text
                and normalized_word_error_rate(target, hyp_text) > max_main_wer
            ):
                kept.append({**row, "target_text": target})
            continue
        if not target:
            target, _ = build_asr_target(
                row["text"],
                hypothesis=(hyp_text if not human_verified else ""),
                is_history=is_history,
                vocab_chars=vocab_chars,
                sp=sp,
                vocab_size=vocab_size,
                max_history_wer=(max_history_wer if is_train else None),
            )
        else:
            ids = encode_bpe_ids(target, sp, vocab_size)
            if ids is None:
                target = None
        if (
            target is not None
            and is_train
            and not is_history
            and max_main_wer is not None
            and hyp_text
            and normalized_word_error_rate(target, hyp_text) > max_main_wer
        ):
            target = None
        if target is not None:
            kept.append({**row, "target_text": target})
    return kept[:limit], len(rows) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(parser)
    parser.add_argument("--data-dir", type=Path, default=CODE_DIR / "result" / "asr_sft_data",
                        help="Directory prepared by prepare_gipformer_sft.py (train.jsonl / dev.jsonl)")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "train+public_test")
    parser.add_argument("--model-dir", type=Path, default=ORIGINAL_ASR_DIR,
                        help="Base Gipformer checkpoint folder containing model.pt, bpe.model, tokens.txt")
    parser.add_argument("--icefall-root", type=Path, default=DEFAULT_ICEFALL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "result" / "asr_sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--max-train-audio", type=int, default=100000)
    parser.add_argument("--max-dev-audio", type=int, default=300)
    parser.add_argument("--encoder-only", action="store_true",
                        help="Freeze decoder/joiner and train encoder only (default trains all parameters)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        help="Disable train-time speed perturbation, additive noise, and SpecAugment")
    parser.add_argument("--augment-scope", choices=["all", "unverified"], default="all",
                        help="Apply acoustic augmentation to all utterances or only unverified history/leftover turns")
    parser.add_argument("--hyp-cache", type=Path, default=None)
    parser.add_argument("--max-history-wer", type=float, default=None)
    parser.add_argument("--max-main-wer", type=float, default=None)
    parser.add_argument("--eval-wer", action="store_true",
                        help="Decode dev split with greedy RNNT search to log dev_wer alongside dev_loss")
    parser.add_argument("--selection-metric", choices=["loss", "wer"], default="loss",
                        help="Metric used for checkpoint selection (best.pt) and early stopping")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--final", action="store_true",
                        help="Keep last checkpoint as best.pt cleanly even when training on all audio")
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run one forward/backward pass and check finite gradients without saving")
    args = parser.parse_args()

    if any(x < 1 for x in (args.epochs, args.accum, args.max_train_audio, args.max_dev_audio, args.eval_every, args.patience)):
        parser.error("epochs, accum, data limits, eval interval, and patience must be positive")
    if not 0 < args.learning_rate <= 1e-4:
        parser.error("Use a learning rate in (0, 1e-4]")

    configure_cuda_device(args)

    import torch
    import torchaudio
    import sentencepiece as spm
    import k2
    import kaldifeat

    device = select_training_device(args.cuda_visible_devices, torch)

    split_dir = _resolve_asr_audio_dir(args.data_root)
    if True:
        train_rows = _read_rows(args.data_dir / "train.jsonl")
        dev_rows = _read_rows(args.data_dir / "dev.jsonl")
        if not train_rows:
            raise ValueError(f"Empty ASR train split under {args.data_dir}")

        model, params = strict_model(args.model_dir, args.icefall_root)
        source_sha = sha256(args.model_dir / "model.pt")
        torch.manual_seed(42)
        random.seed(42)
        model = model.to(device)

        train_all = not args.encoder_only
        for name, parameter in model.named_parameters():
            parameter.requires_grad = train_all or name.startswith(("encoder.", "encoder_embed."))
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise AssertionError("No trainable parameters selected")
        optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)

        sp = spm.SentencePieceProcessor(model_file=str(args.model_dir / "bpe.model"))
        vocab_chars = load_vocab_chars(args.model_dir / "tokens.txt") if (args.model_dir / "tokens.txt").is_file() else None
        hypotheses: dict[str, str] = {}
        if args.hyp_cache and args.hyp_cache.is_file():
            for line in args.hyp_cache.open(encoding="utf-8"):
                if line.strip():
                    item = json.loads(line)
                    hypotheses[item["audio"]] = item.get("raw_transcript") or item.get("transcript") or ""

        train, skipped_train = resolve_training_rows(
            train_rows, sp, params.vocab_size, hypotheses=hypotheses, vocab_chars=vocab_chars,
            max_history_wer=args.max_history_wer, max_main_wer=args.max_main_wer,
            limit=args.max_train_audio, is_train=True,
        )
        dev, skipped_dev = resolve_training_rows(
            dev_rows, sp, params.vocab_size, hypotheses=hypotheses, vocab_chars=vocab_chars,
            max_history_wer=None, max_main_wer=None, limit=args.max_dev_audio, is_train=False,
        )
        if not train:
            raise ValueError("No valid BPE targets remain in ASR train set")
        print(
            f"ASR targets: train={len(train)} (skipped={skipped_train}), "
            f"dev={len(dev)} (skipped={skipped_dev}), train_all={train_all}, "
            f"augment={args.augment} ({args.augment_scope}), device={device}",
            flush=True,
        )

        opts = kaldifeat.FbankOptions()
        opts.device = device
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = 16000
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        fbank = kaldifeat.Fbank(opts)
        aug_rng = random.Random(1234)

        def augment_wave(samples: torch.Tensor) -> torch.Tensor:
            speed = aug_rng.choice([0.9, 1.0, 1.1])
            if speed != 1.0:
                samples = torchaudio.functional.resample(samples, 16000, int(round(16000 / speed)))
            if aug_rng.random() < 0.3:
                snr = aug_rng.uniform(15, 30)
                rms = samples.pow(2).mean().sqrt().clamp_min(1e-5)
                samples = samples + torch.randn_like(samples) * rms / (10 ** (snr / 20))
            return samples

        def spec_augment(feature: torch.Tensor) -> torch.Tensor:
            if aug_rng.random() > 0.8:
                return feature
            feature = feature.clone()
            fill = feature.mean()
            n_frames = feature.shape[0]
            for _ in range(2):
                width = aug_rng.randint(0, 15)
                f0 = aug_rng.randint(0, 80 - width)
                feature[:, f0:f0 + width] = fill
            for _ in range(min(4, n_frames // 50)):
                width = aug_rng.randint(0, max(1, min(20, int(n_frames * 0.05))))
                t0 = aug_rng.randint(0, max(0, n_frames - width))
                feature[t0:t0 + width, :] = fill
            return feature

        def _load_feature(row: dict, apply_aug: bool = False) -> torch.Tensor:
            wav_path = audio_path(split_dir, row["audio"])
            mono, rate = read_audio(wav_path)
            samples = torch.from_numpy(mono).to(device)
            if rate != 16000:
                samples = torchaudio.functional.resample(samples, rate, 16000)
            if apply_aug:
                samples = augment_wave(samples)
            feature = fbank([samples])[0]
            if apply_aug:
                feature = spec_augment(feature)
            if feature.shape[0] < 16:
                raise ValueError(f"Audio too short for Zipformer: {row['audio']}")
            return feature

        def sample_loss(row: dict, apply_aug: bool = False) -> torch.Tensor:
            feature = _load_feature(row, apply_aug=apply_aug)
            tokens = encode_bpe_ids(row["target_text"], sp, params.vocab_size)
            if not tokens:
                raise ValueError(f"Invalid BPE sequence for {row['audio']}")
            y = k2.RaggedTensor([tokens])
            simple, pruned, ctc, _, _ = model(
                feature.unsqueeze(0),
                torch.tensor([feature.shape[0]], device=device),
                y,
                prune_range=params.prune_range,
                am_scale=params.am_scale,
                lm_scale=params.lm_scale,
            )
            loss = 0.5 * simple + pruned + 0.2 * ctc
            return loss / max(1, len(tokens))

        decode_wer = bool(args.eval_wer or args.selection_metric == "wer")

        def evaluate() -> tuple[float | None, float | None]:
            if not dev:
                return None, None
            greedy_fn = None
            if decode_wer:
                from beam_search import greedy_search
                greedy_fn = greedy_search
            model.eval()
            losses = []
            wers = []
            with torch.no_grad():
                for row in dev:
                    feature = _load_feature(row, apply_aug=False)
                    tokens = encode_bpe_ids(row["target_text"], sp, params.vocab_size)
                    if not tokens:
                        raise ValueError(f"Invalid BPE sequence for {row['audio']}")
                    y = k2.RaggedTensor([tokens])
                    feat_lens = torch.tensor([feature.shape[0]], device=device)
                    simple, pruned, ctc, _, _ = model(
                        feature.unsqueeze(0),
                        feat_lens,
                        y,
                        prune_range=params.prune_range,
                        am_scale=params.am_scale,
                        lm_scale=params.lm_scale,
                    )
                    losses.append(float((0.5 * simple + pruned + 0.2 * ctc) / max(1, len(tokens))))
                    if greedy_fn is not None:
                        encoder_out, encoder_out_lens = model.forward_encoder(feature.unsqueeze(0), feat_lens)
                        hyp_ids = greedy_fn(model, encoder_out[:, :encoder_out_lens[0]], max_sym_per_frame=1)
                        wers.append(normalized_word_error_rate(row["target_text"], sp.decode(hyp_ids)))
            model.train()
            avg_loss = sum(losses) / len(losses)
            avg_wer = (sum(wers) / len(wers)) if wers else None
            return avg_loss, avg_wer

        if args.smoke_only:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = sample_loss(train[0], apply_aug=False)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite smoke RNNT loss")
            loss.backward()
            gradients = [p.grad for p in trainable if p.grad is not None]
            if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
                raise FloatingPointError("Missing or nonfinite smoke gradients")
            print(json.dumps({
                "smoke_only": True,
                "train_loss": float(loss.detach()),
                "gradient_tensors": len(gradients),
                "source_sha256": source_sha,
            }))
            return

        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Choose an empty output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "source_checkpoint": str((args.model_dir / "model.pt").resolve()),
            "source_sha256": source_sha,
            "data_dir": str(args.data_dir.resolve()),
            "icefall_root": str(args.icefall_root.resolve()),
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "accum": args.accum,
            "train_all": train_all,
            "augment": args.augment,
            "augment_scope": args.augment_scope,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "train_audio": len(train),
            "dev_audio": len(dev),
            "skipped_train": skipped_train,
            "skipped_dev": skipped_dev,
        }
        (args.output_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

        best = math.inf
        stale = 0
        step = 0
        with (args.output_dir / "training.jsonl").open("w", encoding="utf-8") as log:
            for epoch in range(1, args.epochs + 1):
                random.Random(42 + epoch).shuffle(train)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for index, row in enumerate(train):
                    started = time.monotonic()
                    allow_aug = args.augment and (args.augment_scope == "all" or not row.get("human_verified", False))
                    loss = sample_loss(row, apply_aug=allow_aug)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite RNNT loss at {row['audio']}")
                    (loss / args.accum).backward()
                    if (index + 1) % args.accum == 0 or (index + 1) == len(train):
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    step += 1
                    rec = {
                        "epoch": epoch,
                        "step": step,
                        "train_loss": float(loss.detach()),
                        "audio": row["audio"],
                        "seconds": round(time.monotonic() - started, 4),
                    }
                    log.write(json.dumps(rec) + "\n")
                    if step % 25 == 0:
                        log.flush()
                        print(f"Gipformer SFT epoch={epoch} step={step} loss={rec['train_loss']:.4f}", flush=True)
                    is_last_sample = (epoch == args.epochs and index == len(train) - 1)
                    if step % args.eval_every == 0 or is_last_sample:
                        dev_loss, dev_wer = evaluate()
                        monitored = dev_wer if args.selection_metric == "wer" else dev_loss
                        ckpt = {
                            "model": {key: val.detach().cpu().clone() for key, val in model.state_dict().items()},
                            "step": step,
                            "epoch": epoch,
                            "dev_loss": dev_loss,
                            "dev_wer": dev_wer,
                            "selection_metric": args.selection_metric,
                            "source_sha256": source_sha,
                        }
                        ckpt_path = args.output_dir / f"checkpoint-{step:06d}.pt"
                        torch.save(ckpt, ckpt_path)
                        improved = (monitored is not None and monitored < best - 1e-4) or (args.final and is_last_sample) or monitored is None
                        if improved:
                            if monitored is not None:
                                best = monitored
                            stale = 0
                            torch.save(ckpt, args.output_dir / "best.pt")
                        else:
                            stale += 1
                        log.write(json.dumps({
                            "step": step,
                            "dev_loss": dev_loss,
                            "dev_wer": dev_wer,
                            "selection_metric": args.selection_metric,
                            "checkpoint": str(ckpt_path),
                            "best": improved,
                        }) + "\n")
                        log.flush()
                        if not args.final and monitored is not None and stale >= args.patience:
                            print("Early stopping Gipformer SFT", flush=True)
                            return


if __name__ == "__main__":
    main()
