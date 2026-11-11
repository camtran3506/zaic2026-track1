"""Persist all ASR text before loading the LLM. Never accesses environment state."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import time

from .agent import _train_transcript, audio_path
from .history import history_entries as iter_history_entries, resolve_history
from .text import TEXT_NORMALIZATION, normalize_transcript

LOG = logging.getLogger("lifemate.asr")
# Version 4 records the shared utterance normalization policy.
TRANSCRIPT_CACHE_VERSION = 4


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_paths(path):
    path = Path(path)
    return path.with_suffix(".manifest.json"), path.with_suffix(".partial.jsonl"), path.with_suffix(".partial.manifest.json")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def history_entries(item):
    """Yield all history text/audio without a turn or character limit."""
    yield from iter_history_entries(item.get("history", []))


def dataset_identity(items, split_dir, split, use_train_transcript=False):
    if use_train_transcript and split != "train":
        raise ValueError("Train transcript ablation is forbidden on test splits")
    ids = [item["id"] for item in items]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Selected dataset must have nonempty, unique sample IDs")
    audio_files = set()
    for item in items:
        if not (use_train_transcript and _train_transcript(item) is not None):
            audio_files.add(audio_path(split_dir, item["audio"]))
        for role, text, audio in history_entries(item):
            if text is None and audio:
                audio_files.add(audio_path(split_dir, audio))
    stamps = []
    for path in sorted(audio_files):
        stat = path.stat()
        stamps.append([str(path.relative_to(split_dir.resolve())), stat.st_size, stat.st_mtime_ns])
    serialized = json.dumps({"items": items, "audio_stamps": stamps}, sort_keys=True, ensure_ascii=False)
    return {"split": split, "sample_ids": ids, "use_train_transcript": use_train_transcript,
            "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}


def asr_identity(
    model_path,
    precision="auto",
    num_threads=4,
    decoding_method=None,
    max_active_paths=None,
):
    from .asr import (
        ASR_BACKEND,
        ASR_REPO_ID,
        DEFAULT_ASR_DECODING_METHOD,
        DEFAULT_ASR_MAX_ACTIVE_PATHS,
        asr_files,
        resolve_asr_precision,
    )
    if num_threads < 1:
        raise ValueError("ASR num_threads must be positive")
    decoding_method = decoding_method or DEFAULT_ASR_DECODING_METHOD
    max_active_paths = DEFAULT_ASR_MAX_ACTIVE_PATHS if max_active_paths is None else max_active_paths
    if decoding_method not in {"modified_beam_search", "greedy_search"}:
        raise ValueError("ASR decoding_method must be modified_beam_search or greedy_search")
    if max_active_paths < 1:
        raise ValueError("ASR max_active_paths must be positive")
    model_path = Path(model_path).expanduser().resolve()
    resolved_precision = resolve_asr_precision(model_path, precision)
    files = asr_files(model_path, resolved_precision)
    stamps = [[str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns]
              for p in files.values()]
    return {"model_path": str(model_path), "file_stamps": stamps,
            "repo_id": ASR_REPO_ID, "backend": ASR_BACKEND, "device": "cpu",
            "precision": resolved_precision, "num_threads": num_threads,
            "sample_rate": 16000, "feature_dim": 80, "decoding_method": decoding_method,
            "max_active_paths": (max_active_paths if decoding_method == "modified_beam_search" else 1),
            "sherpa_onnx_version": version("sherpa-onnx"), "soundfile_version": version("soundfile")}


def validate_rows(rows, identity, complete=True):
    expected = identity["sample_ids"]
    if len(rows) > len(expected) or (complete and len(rows) != len(expected)):
        raise ValueError("Transcript cache has missing/extra sample rows")
    if [row.get("id") for row in rows] != expected[:len(rows)]:
        raise ValueError("Transcript cache IDs/order differ from the selected dataset")
    for row in rows:
        if not isinstance(row.get("transcript"), str) or not isinstance(row.get("history"), list):
            raise ValueError(f"Invalid transcript row: {row.get('id')}")
        if any(not isinstance(turn, dict) or not isinstance(turn.get("role"), str)
               or not isinstance(turn.get("text"), str) for turn in row["history"]):
            raise ValueError(f"Invalid cached history: {row.get('id')}")
        if (row["transcript"] != normalize_transcript(row["transcript"])
                or any(turn["text"] != normalize_transcript(turn["text"]) for turn in row["history"])):
            raise ValueError(f"Cached text violates transcript normalization: {row.get('id')}")
        if not isinstance(row.get("asr_seconds"), (float, int)) or row["asr_seconds"] < 0:
            raise ValueError(f"Invalid ASR timing: {row.get('id')}")


def load_transcripts(path, identity, expected_asr=None):
    path = Path(path)
    manifest_path, _, _ = cache_paths(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != TRANSCRIPT_CACHE_VERSION:
        raise ValueError("Transcript cache format changed; rebuild with --refresh-transcripts")
    if manifest.get("text_normalization") != TEXT_NORMALIZATION:
        raise ValueError("Transcript normalization changed; rebuild with --refresh-transcripts")
    if manifest.get("status") != "complete":
        raise ValueError("Transcript cache is incomplete; finish the ASR stage first")
    if manifest.get("dataset") != identity:
        raise ValueError("Transcript dataset/audio/selection changed; use a new cache or --refresh-transcripts")
    if expected_asr is not None and manifest.get("asr") != expected_asr:
        raise ValueError("ASR model/settings changed; use a new cache or --refresh-transcripts")
    if manifest.get("transcripts_sha256") != sha256_file(path):
        raise ValueError("Transcript cache checksum mismatch")
    rows = read_jsonl(path)
    validate_rows(rows, identity)
    return rows, manifest


def prepare_transcripts(items, split_dir, cache_path, identity, asr_settings, asr_factory, refresh=False, batch_size=16):
    """Resume valid prefix on retry using batched offline decoding. An ASR error stops before any LLM is loaded."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    cache_path = Path(cache_path)
    manifest_path, partial, partial_manifest = cache_paths(cache_path)
    if cache_path.exists() and not refresh:
        rows, _ = load_transcripts(cache_path, identity, asr_settings)
        LOG.info("Reusing completed ASR cache: %s (%d samples)", cache_path, len(rows))
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    header = {"version": TRANSCRIPT_CACHE_VERSION, "dataset": identity, "asr": asr_settings,
              "text_normalization": TEXT_NORMALIZATION}
    rows = []
    if partial.exists():
        previous = json.loads(partial_manifest.read_text(encoding="utf-8")) if partial_manifest.exists() else None
        if previous != header:
            if not refresh:
                raise ValueError("Partial ASR cache settings changed; use a new path or --refresh-transcripts")
        else:
            rows = read_jsonl(partial)
            validate_rows(rows, identity, complete=False)
    if not rows:
        partial_manifest.write_text(json.dumps(header, ensure_ascii=False, indent=2), encoding="utf-8")
        partial.write_text("", encoding="utf-8")
    LOG.info("ASR stage: %d/%d samples already cached", len(rows), len(items))
    asr = None

    def get_asr():
        nonlocal asr
        if asr is None:
            asr = asr_factory()
        return asr

    def transcribe(relative):
        return get_asr().transcribe(audio_path(split_dir, relative))

    def pretranscribe_chunk(chunk):
        needed_paths = []
        for item in chunk:
            if not (identity["use_train_transcript"] and _train_transcript(item) is not None):
                needed_paths.append(audio_path(split_dir, item["audio"]))
            for _role, text, audio in history_entries(item):
                if text is None and audio:
                    needed_paths.append(audio_path(split_dir, audio))
        unique_paths = list(dict.fromkeys(needed_paths))
        if not unique_paths:
            return
        engine = get_asr()
        if hasattr(engine, "transcribe_batch"):
            try:
                engine.transcribe_batch(unique_paths, batch_size=batch_size)
            except Exception:
                # Fall back to sequential decoding below so exact failing item ID is logged cleanly.
                pass

    with partial.open("a", encoding="utf-8") as f:
        start_idx = len(rows)
        for chunk_start in range(start_idx, len(items), batch_size):
            chunk = items[chunk_start:chunk_start + batch_size]
            pretranscribe_chunk(chunk)
            for offset, item in enumerate(chunk):
                index = chunk_start + offset
                start = time.monotonic()
                try:
                    transcript = _train_transcript(item) if identity["use_train_transcript"] else None
                    if transcript is None:
                        transcript = transcribe(item["audio"])
                    transcript = normalize_transcript(transcript)
                    history = resolve_history(item.get("history", []), transcribe)
                    row = {"id": item["id"], "transcript": transcript, "history": history,
                           "asr_seconds": round(time.monotonic() - start, 6)}
                    validate_rows([row], {"sample_ids": [item["id"]]})
                except Exception as exc:
                    LOG.exception("ASR failed for %s; partial cache retained; LLM will not start", item["id"])
                    with cache_path.with_suffix(".errors.jsonl").open("a", encoding="utf-8") as errors:
                        errors.write(json.dumps({"id": item["id"], "stage": "asr", "error": repr(exc)}, ensure_ascii=False) + "\n")
                    raise
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                LOG.info("ASR %d/%d %s: %.3fs", index + 1, len(items), item["id"], row["asr_seconds"])
    # Complete marker is written last; LLM refuses incomplete or modified caches.
    partial.replace(cache_path)
    manifest = {**header, "status": "complete", "transcripts_sha256": sha256_file(cache_path)}
    temp_manifest = manifest_path.with_suffix(".tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_manifest.replace(manifest_path)
    LOG.info("All ASR transcripts saved: %s", cache_path)
