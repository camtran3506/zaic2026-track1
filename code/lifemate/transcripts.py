"""Persist all ASR text before loading the LLM. Never accesses environment state."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import time

from .agent import _train_transcript, audio_path

LOG = logging.getLogger("lifemate.asr")


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
    """Keep full history transcripts on disk; agent applies prompt history limits later."""
    history = item.get("history", [])
    if not isinstance(history, list):
        return
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", turn.get("speaker", "user")))
        text = next((turn[k] for k in ("audio_text", "text", "content", "transcript")
                     if isinstance(turn.get(k), str) and turn[k].strip()), "")
        audio = turn.get("audio", turn.get("audio_path"))
        yield role, text, audio if isinstance(audio, str) else None


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
            if not text and audio:
                audio_files.add(audio_path(split_dir, audio))
    stamps = []
    for path in sorted(audio_files):
        stat = path.stat()
        stamps.append([str(path.relative_to(split_dir.resolve())), stat.st_size, stat.st_mtime_ns])
    serialized = json.dumps({"items": items, "audio_stamps": stamps}, sort_keys=True, ensure_ascii=False)
    return {"split": split, "sample_ids": ids, "use_train_transcript": use_train_transcript,
            "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest()}


def asr_identity(model_path, compute_type, beam_size, seed):
    model_path = Path(model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    # Detect normal local model replacement without hashing multi-GB weights every run.
    stamps = [[str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns]
              for p in sorted(model_path.rglob("*")) if p.is_file() and ".cache" not in p.relative_to(model_path).parts]
    return {"model_path": str(model_path), "file_stamps": stamps,
            "compute_type": compute_type, "beam_size": beam_size, "seed": seed,
            "language": "vi", "temperature": 0,
            "faster_whisper_version": version("faster-whisper"),
            "ctranslate2_version": version("ctranslate2"), "av_version": version("av")}


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
        if not isinstance(row.get("asr_seconds"), (float, int)) or row["asr_seconds"] < 0:
            raise ValueError(f"Invalid ASR timing: {row.get('id')}")


def load_transcripts(path, identity, expected_asr=None):
    path = Path(path)
    manifest_path, _, _ = cache_paths(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("status") != "complete":
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


def prepare_transcripts(items, split_dir, cache_path, identity, asr_settings, asr_factory, refresh=False):
    """Resume valid prefix on retry. An ASR error stops before any LLM is loaded."""
    cache_path = Path(cache_path)
    manifest_path, partial, partial_manifest = cache_paths(cache_path)
    if cache_path.exists() and not refresh:
        rows, _ = load_transcripts(cache_path, identity, asr_settings)
        LOG.info("Reusing completed ASR cache: %s (%d samples)", cache_path, len(rows))
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    header = {"version": 1, "dataset": identity, "asr": asr_settings}
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
    def transcribe(relative):
        nonlocal asr
        if asr is None:
            asr = asr_factory()
        return asr.transcribe(audio_path(split_dir, relative))
    with partial.open("a", encoding="utf-8") as f:
        for index in range(len(rows), len(items)):
            item = items[index]
            start = time.monotonic()
            try:
                transcript = _train_transcript(item) if identity["use_train_transcript"] else None
                if transcript is None:
                    transcript = transcribe(item["audio"])
                history = []
                for role, text, audio in history_entries(item):
                    if not text and audio:
                        text = transcribe(audio)
                    if text:
                        history.append({"role": role, "text": text})
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
