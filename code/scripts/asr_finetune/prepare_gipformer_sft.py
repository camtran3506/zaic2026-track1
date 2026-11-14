"""Prepare Gipformer fine-tuning dataset from train_asr.jsonl.

Selective normalization & anti-noise policy:
- Main current-turn `audio_text` had numbers converted to text accurately by human annotators.
  All other normalizations (symbol conversion for &, %, +, @, °, time/date delimiters,
  UPPERCASE, punctuation removal, BPE model_style tone alignment, and vocabulary checks)
  are applied identically to history audio. Main turns are never dropped by WER/noise checks.
- Any leftover `audio_text` turns still carrying digits apply `target_v3` spoken-word expansion.
- History user audio turns (unedited) apply `target_v3` spoken-word alignment and optional noisy-history filtering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lifemate.agent import _train_transcript, audio_path  # noqa: E402
from lifemate.asr import DEFAULT_ASR_DECODING_METHOD, DEFAULT_ASR_MAX_ACTIVE_PATHS, resolve_asr_precision  # noqa: E402
from lifemate.asr_targets import (  # noqa: E402
    DEFAULT_TOKENS_CHARS,
    build_asr_target,
    load_vocab_chars,
    needs_target_normalization,
    normalized_word_error_rate,
)


def _resolve_asr_audio_dir(data_root: Path) -> Path:
    location = Path(data_root).expanduser().resolve()
    if (location / "train" / "audio").is_dir() or (location / "train" / "train.jsonl").is_file():
        return location / "train"
    if (location / "audio").is_dir() or (location / "train.jsonl").is_file():
        return location
    if (location / "train").is_dir():
        return location / "train"
    if location.is_dir():
        return location
    raise FileNotFoundError(f"Audio data directory not found: {location}")


def _utterances(record: dict) -> list[tuple[str, str, str]]:
    """Yield (audio, text, source_kind) for current turn and user history turns with audio+text."""
    rows: list[tuple[str, str, str]] = []
    main_text = _train_transcript(record)
    if isinstance(record.get("audio"), str) and isinstance(main_text, str) and main_text.strip():
        rows.append((record["audio"], main_text, "main"))
    for turn in record.get("history") or []:
        if not isinstance(turn, dict):
            continue
        if turn.get("role", turn.get("speaker")) != "user":
            continue
        aud = turn.get("audio", turn.get("audio_path"))
        txt = turn.get("audio_text", turn.get("transcript", turn.get("text")))
        if isinstance(aud, str) and isinstance(txt, str) and txt.strip():
            rows.append((aud, txt, "history"))
    return rows


def _load_hypotheses(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    hyp: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = row.get("audio")
            text = row.get("raw_transcript") or row.get("transcript") or ""
            if isinstance(audio, str) and isinstance(text, str) and text.strip():
                hyp[audio] = text
    return hyp


def _auto_hypotheses(
    rows_needing_hyp: list[dict],
    split_dir: Path,
    model_dir: Path,
    precision: str,
    decoding_method: str,
    beam_size: int,
    threads: int,
    batch_size: int,
) -> dict[str, str]:
    if not rows_needing_hyp:
        return {}
    from lifemate.asr import LocalASR

    resolved = resolve_asr_precision(model_dir, precision)
    engine = LocalASR(
        model_dir,
        precision=resolved,
        num_threads=threads,
        decoding_method=decoding_method,
        max_active_paths=beam_size,
    )
    rel_paths = list(dict.fromkeys(r["audio"] for r in rows_needing_hyp))
    full_paths = [audio_path(split_dir, rel) for rel in rel_paths]
    texts = engine.transcribe_batch(full_paths, batch_size=batch_size)
    return dict(zip(rel_paths, texts))


def prepare_asr_dataset(
    *,
    input_jsonl: Path,
    data_root: Path,
    output_dir: Path,
    model_dir: Path | None = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    hyp_cache: Path | None = None,
    generate_hypotheses: bool = False,
    asr_precision: str = "auto",
    asr_decoding_method: str = DEFAULT_ASR_DECODING_METHOD,
    asr_beam_size: int = DEFAULT_ASR_MAX_ACTIVE_PATHS,
    asr_threads: int = 4,
    asr_batch_size: int = 16,
    max_history_wer: float | None = None,
    verify_main: bool = False,
    max_main_wer: float | None = None,
    final: bool = False,
) -> dict:
    if not input_jsonl.is_file():
        raise FileNotFoundError(f"Missing ASR training JSONL: {input_jsonl}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    split_dir = _resolve_asr_audio_dir(data_root)
    if True:
        records = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            raise ValueError(f"No records in {input_jsonl}")

        vocab_chars = DEFAULT_TOKENS_CHARS
        sp = None
        if model_dir is not None and model_dir.is_dir():
            tokens_file = model_dir / "tokens.txt"
            if tokens_file.is_file():
                vocab_chars = load_vocab_chars(tokens_file)
            bpe_file = model_dir / "bpe.model"
            if bpe_file.is_file():
                import sentencepiece as spm
                sp = spm.SentencePieceProcessor(model_file=str(bpe_file))

        # Connected components over shared audio files avoid leaking recordings between train/dev.
        parents = {r["id"]: r["id"] for r in records}

        def find(x: str) -> str:
            while parents[x] != x:
                parents[x] = parents[parents[x]]
                x = parents[x]
            return x

        def union(a: str, b: str) -> None:
            parents[find(a)] = find(b)

        owner: dict[str, str] = {}
        for rec in records:
            for aud, _, _ in _utterances(rec):
                audio_path(split_dir, aud)
                if aud in owner:
                    union(rec["id"], owner[aud])
                owner[aud] = rec["id"]

        components: dict[str, list[dict]] = {}
        for rec in records:
            components.setdefault(find(rec["id"]), []).append(rec)

        buckets: dict[str, dict[str, dict]] = {"train": {}, "dev": {}}
        dev_mod = round(1.0 / val_fraction) if val_fraction > 0 else 0
        for comp in components.values():
            sig = min(str(r["id"]) for r in comp)
            is_dev = dev_mod > 0 and (int(hashlib.sha256(f"asr:{seed}:{sig}".encode()).hexdigest(), 16) % dev_mod == 0)
            bucket_name = "dev" if is_dev else "train"
            bucket = buckets[bucket_name]
            for rec in comp:
                for aud, text, source_kind in _utterances(rec):
                    prev = bucket.get(aud)
                    is_human_main = source_kind == "main" and not needs_target_normalization(text, vocab_chars)
                    candidate = {
                        "sample_id": rec["id"],
                        "audio": aud,
                        "text": text,
                        "source": source_kind,
                        "human_verified": bool(is_human_main),
                    }
                    if prev is None:
                        bucket[aud] = candidate
                    elif prev["source"] == "history" and source_kind == "main":
                        # Human-edited main audio_text always takes precedence over unedited history.
                        bucket[aud] = candidate
                    elif prev["source"] == source_kind and prev["text"] != text:
                        raise ValueError(f"Conflicting {source_kind} transcripts for {aud}: {prev['text']!r} vs {text!r}")

        # Ensure train and dev never share audio keys (if final=False).
        for aud in list(buckets["dev"]):
            if aud in buckets["train"]:
                del buckets["dev"][aud]

        all_items = list(buckets["train"].values()) + list(buckets["dev"].values())
        hypotheses = _load_hypotheses(hyp_cache)
        check_main = bool(verify_main or max_main_wer is not None)
        if generate_hypotheses and model_dir is not None:
            # Generate base-model hypotheses for unverified rows and optionally main rows when verify_main is set.
            missing_unverified = [
                row for row in all_items
                if (not row["human_verified"] or check_main) and row["audio"] not in hypotheses
            ]
            hypotheses.update(
                _auto_hypotheses(
                    missing_unverified,
                    split_dir=split_dir,
                    model_dir=model_dir,
                    precision=asr_precision,
                    decoding_method=asr_decoding_method,
                    beam_size=asr_beam_size,
                    threads=asr_threads,
                    batch_size=asr_batch_size,
                )
            )

        prepared: dict[str, list[dict]] = {"train": [], "dev": []}
        skipped: list[dict] = []
        stats = {
            "main_human_verified": 0,
            "main_leftover_normalized": 0,
            "history_kept": 0,
            "skipped_main": 0,
            "skipped_history": 0,
        }
        for bucket_name in ("train", "dev"):
            for row in buckets[bucket_name].values():
                need_hyp = (not row["human_verified"]) or (row["source"] == "main" and check_main)
                hyp_text = hypotheses.get(row["audio"], "") if need_hyp else ""
                target, meta = build_asr_target(
                    row["text"],
                    hypothesis=(hyp_text if not row["human_verified"] else ""),
                    is_history=(row["source"] == "history"),
                    vocab_chars=vocab_chars,
                    sp=sp,
                    max_history_wer=(max_history_wer if bucket_name == "train" else None),
                )
                if (
                    target is not None
                    and row["source"] == "main"
                    and bucket_name == "train"
                    and max_main_wer is not None
                    and hyp_text
                ):
                    main_wer = normalized_word_error_rate(target, hyp_text)
                    if main_wer > max_main_wer:
                        target = None
                        meta = {"reason": "high_main_wer", "wer": round(main_wer, 4), "hypothesis": hyp_text}
                if target is None:
                    skipped.append({**row, **meta})
                    if row["source"] == "main":
                        stats["skipped_main"] += 1
                    else:
                        stats["skipped_history"] += 1
                    continue
                if meta["mode"] == "human_verified":
                    stats["main_human_verified"] += 1
                elif meta["mode"] == "leftover_main":
                    stats["main_leftover_normalized"] += 1
                else:
                    stats["history_kept"] += 1
                prepared[bucket_name].append({**row, "target_text": target, "target_meta": meta})

        if final and prepared["dev"]:
            seen_train_audio = {r["audio"] for r in prepared["train"]}
            for row in prepared["dev"]:
                if row["audio"] not in seen_train_audio:
                    prepared["train"].append(row)
                    seen_train_audio.add(row["audio"])

        if not prepared["train"]:
            raise ValueError("Prepared ASR training set is empty")

        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Choose an empty output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        for bucket_name in ("train", "dev"):
            with (output_dir / f"{bucket_name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in prepared[bucket_name]:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if skipped:
            with (output_dir / "skipped.jsonl").open("w", encoding="utf-8") as handle:
                for row in skipped:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest = {
            "source_jsonl": str(input_jsonl.resolve()),
            "data_root": str(split_dir),
            "final": final,
            "val_fraction": val_fraction,
            "seed": seed,
            "max_history_wer": max_history_wer,
            "verify_main": verify_main,
            "max_main_wer": max_main_wer,
            "train_rows": len(prepared["train"]),
            "dev_rows": len(prepared["dev"]),
            "skipped_rows": len(skipped),
            "selective_policy": "normalization-and-denoising-on-history-and-leftover-main-only-v1",
            **stats,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=REPO_ROOT / "train_asr.jsonl")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "train+public_test")
    parser.add_argument("--model-dir", type=Path, default=CODE_DIR / "models" / "asr",
                        help="Base ASR checkpoint (provides tokens.txt, bpe.model, and optional base ONNX hypotheses)")
    parser.add_argument("--output-dir", type=Path, default=CODE_DIR / "result" / "asr_sft_data")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hyp-cache", type=Path, default=None)
    parser.add_argument("--generate-hypotheses", action="store_true",
                        help="Run base LocalASR on unverified rows (plus main rows when --verify-main is set)")
    parser.add_argument("--asr-precision", choices=["auto", "int8", "fp32"], default="auto")
    parser.add_argument("--asr-decoding-method", choices=["modified_beam_search", "greedy_search"],
                        default=DEFAULT_ASR_DECODING_METHOD)
    parser.add_argument("--asr-beam-size", type=int, default=DEFAULT_ASR_MAX_ACTIVE_PATHS)
    parser.add_argument("--asr-threads", type=int, default=4)
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--max-history-wer", type=float, default=None,
                        help="Optional WER threshold for dropping noisy unverified history rows")
    parser.add_argument("--verify-main", action="store_true",
                        help="Also generate/check hypotheses on digit-free main audio_text instead of unconditionally trusting it")
    parser.add_argument("--max-main-wer", type=float, default=None,
                        help="Optional WER threshold for filtering noisy main audio_text rows in train")
    parser.add_argument("--final", action="store_true",
                        help="Include dev rows in final training while keeping dev.jsonl as loss monitor")
    args = parser.parse_args()
    manifest = prepare_asr_dataset(
        input_jsonl=args.input_jsonl,
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        hyp_cache=args.hyp_cache,
        generate_hypotheses=args.generate_hypotheses,
        asr_precision=args.asr_precision,
        asr_decoding_method=args.asr_decoding_method,
        asr_beam_size=args.asr_beam_size,
        asr_threads=args.asr_threads,
        asr_batch_size=args.asr_batch_size,
        max_history_wer=args.max_history_wer,
        verify_main=args.verify_main,
        max_main_wer=args.max_main_wer,
        final=args.final,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
