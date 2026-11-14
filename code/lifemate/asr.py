"""Offline GIPformer 1.5 ONNX transducer inference on CPU."""
from collections import OrderedDict
from pathlib import Path
from .text import normalize_transcript

import math
import tempfile

from .asr_biasing import (
    DEFAULT_ASR_HOTWORDS_SCORE,
    VALID_HOTWORD_MODES,
    ensure_bpe_vocab,
    load_tokens_vocab_chars,
    post_normalize_asr_text,
)

ASR_REPO_ID = "g-group-ai-lab/gipformer1.5-68M-rnnt"
ASR_BACKEND = "sherpa-onnx"
DEFAULT_SFT_ASR_MODEL = Path(__file__).resolve().parents[1] / 'models' / 'asr_ft'
DEFAULT_ASR_DECODING_METHOD = "modified_beam_search"
DEFAULT_ASR_MAX_ACTIVE_PATHS = 4


def asr_extra_options(args) -> dict:
    """Build optional LocalASR / asr_identity kwargs, omitting defaults for compatibility."""
    extra = {}
    if args.asr_decoding_method != DEFAULT_ASR_DECODING_METHOD:
        extra["decoding_method"] = args.asr_decoding_method
    if args.asr_beam_size != DEFAULT_ASR_MAX_ACTIVE_PATHS:
        extra["max_active_paths"] = args.asr_beam_size
    hotwords = getattr(args, "asr_hotwords", "off")
    if hotwords != "off":
        extra["hotwords_mode"] = hotwords
        score = getattr(args, "asr_hotwords_score", DEFAULT_ASR_HOTWORDS_SCORE)
        if score != DEFAULT_ASR_HOTWORDS_SCORE:
            extra["hotwords_score"] = score
    if getattr(args, "asr_post_normalize", False):
        extra["post_normalize"] = True
    return extra


def resolve_asr_precision(model_path: Path, precision: str = "auto") -> str:
    """Resolve explicit precision strictly, or prefer FP32 ONNX when precision='auto'."""
    if precision in {"int8", "fp32"}:
        return precision
    if precision != "auto":
        raise ValueError("ASR precision must be auto, int8, or fp32")
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for candidate, suffix in (("fp32", ".onnx"), ("int8", ".int8.onnx")):
        if all((root / (part + suffix)).is_file() and (root / (part + suffix)).stat().st_size > 0
               for part in ("encoder", "decoder", "joiner")):
            return candidate
    raise FileNotFoundError(f"Missing GIPformer ONNX model files in {root}")


def asr_files(model_path: Path, precision: str = "int8") -> dict[str, Path]:
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if precision == "auto":
        precision = resolve_asr_precision(model_path, "auto")
    if precision not in {"int8", "fp32"}:
        raise ValueError("ASR precision must be int8 or fp32")
    suffix = ".int8.onnx" if precision == "int8" else ".onnx"
    files = {part: model_path / (part + suffix) for part in ("encoder", "decoder", "joiner")}
    files["tokens"] = model_path / "tokens.txt"
    for path in files.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty GIPformer model file: {path}")
    return files


def read_audio(path: Path):
    """PCM float32 mono; accept_waveform handles resampling from the source rate."""
    import numpy as np
    import soundfile as sf
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if not samples.size or sample_rate <= 0 or not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1:
        raise ValueError(f"Empty or invalid audio: {path}")
    return np.ascontiguousarray(samples.mean(axis=1, dtype=np.float32)), sample_rate


class LocalASR:
    def __init__(
        self,
        model_path: Path,
        precision: str = "auto",
        num_threads: int = 4,
        max_cache: int = 8192,
        decoding_method: str = DEFAULT_ASR_DECODING_METHOD,
        max_active_paths: int = DEFAULT_ASR_MAX_ACTIVE_PATHS,
        hotwords_mode: str = "off",
        hotwords_score: float = DEFAULT_ASR_HOTWORDS_SCORE,
        post_normalize: bool = False,
    ):
        if num_threads < 1:
            raise ValueError("ASR num_threads must be positive")
        if decoding_method not in {"modified_beam_search", "greedy_search"}:
            raise ValueError("ASR decoding_method must be modified_beam_search or greedy_search")
        if max_active_paths < 1:
            raise ValueError("ASR max_active_paths must be positive")
        if hotwords_mode not in VALID_HOTWORD_MODES:
            raise ValueError(f"Unsupported hotwords_mode {hotwords_mode!r}; expected {VALID_HOTWORD_MODES}")
        if not math.isfinite(hotwords_score) or hotwords_score <= 0:
            raise ValueError("hotwords_score must be a finite positive float")
        resolved_precision = resolve_asr_precision(model_path, precision)
        files = asr_files(model_path, resolved_precision)
        import sherpa_onnx
        kwargs = {
            **{name: str(path) for name, path in files.items()},
            "num_threads": num_threads,
            "sample_rate": 16000,
            "feature_dim": 80,
            "decoding_method": decoding_method,
            "provider": "cpu",
        }
        if decoding_method == "modified_beam_search":
            kwargs["max_active_paths"] = max_active_paths
        self._hotwords_tmpdir = None
        supports_hotwords = False
        if hotwords_mode != "off" and decoding_method == "modified_beam_search":
            resolved_dir = Path(model_path).expanduser().resolve()
            direct_vocab = resolved_dir / "bpe.vocab"
            if direct_vocab.is_file():
                vocab_path = direct_vocab
            else:
                self._hotwords_tmpdir = tempfile.TemporaryDirectory(prefix="lifemate_bpe_")
                vocab_path = ensure_bpe_vocab(resolved_dir, Path(self._hotwords_tmpdir.name))
            if vocab_path is not None:
                kwargs["modeling_unit"] = "bpe"
                kwargs["bpe_vocab"] = str(vocab_path)
                kwargs["hotwords_score"] = float(hotwords_score)
                supports_hotwords = True
            elif self._hotwords_tmpdir is not None:
                self._hotwords_tmpdir.cleanup()
                self._hotwords_tmpdir = None
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        self.precision = resolved_precision
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self.hotwords_mode = hotwords_mode
        self.hotwords_score = float(hotwords_score)
        self.post_normalize = bool(post_normalize)
        self.supports_hotwords = supports_hotwords
        self.vocab_chars = load_tokens_vocab_chars(files.get("tokens"))
        self.max_cache = max_cache
        self.cache: OrderedDict[str | tuple[str, str], str] = OrderedDict()

    def close(self) -> None:
        tmpdir = getattr(self, "_hotwords_tmpdir", None)
        if tmpdir is not None:
            tmpdir.cleanup()
            self._hotwords_tmpdir = None

    def __del__(self) -> None:
        self.close()

    def _create_stream(self, hotwords: str = ""):
        hw = hotwords.strip() if (self.supports_hotwords and hotwords) else ""
        if hw:
            return self.model.create_stream(hotwords=hw)
        return self.model.create_stream()

    def _finalize_text(self, raw_text: str) -> str:
        text = post_normalize_asr_text(raw_text) if self.post_normalize else raw_text
        return normalize_transcript(text)

    def _cache_key(self, resolved_path: Path, hotwords: str = "") -> str | tuple[str, str]:
        hw = hotwords.strip() if (self.supports_hotwords and hotwords) else ""
        return (str(resolved_path), hw) if hw else str(resolved_path)

    def _remember(self, key: str | tuple[str, str], text: str) -> None:
        self.cache[key] = text
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)

    def transcribe(self, path: Path, hotwords: str = "") -> str:
        resolved = Path(path).resolve()
        key = self._cache_key(resolved, hotwords)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        samples, sample_rate = read_audio(resolved)
        stream = self._create_stream(hotwords)
        stream.accept_waveform(sample_rate, samples)
        self.model.decode_streams([stream])
        result = self._finalize_text(stream.result.text)
        self._remember(key, result)
        return result

    def transcribe_batch(
        self,
        paths: list[Path],
        batch_size: int = 16,
        hotwords_list: list[str] | None = None,
    ) -> list[str]:
        """Batched offline transcription for SFT preparation and ASR stage caching."""
        if not paths:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if hotwords_list is not None and len(hotwords_list) != len(paths):
            raise ValueError("hotwords_list length must match paths length")
        resolved_paths = [Path(p).resolve() for p in paths]
        hw_items = hotwords_list if hotwords_list is not None else [""] * len(resolved_paths)
        unique_missing: list[tuple[Path, str, str | tuple[str, str]]] = []
        seen_keys: set[str | tuple[str, str]] = set()
        for p, hw in zip(resolved_paths, hw_items):
            key = self._cache_key(p, hw)
            if key not in self.cache and key not in seen_keys:
                seen_keys.add(key)
                unique_missing.append((p, hw, key))

        for start_idx in range(0, len(unique_missing), batch_size):
            chunk = unique_missing[start_idx:start_idx + batch_size]
            streams = []
            for p, hw, _key in chunk:
                samples, sample_rate = read_audio(p)
                stream = self._create_stream(hw)
                stream.accept_waveform(sample_rate, samples)
                streams.append(stream)

            if streams:
                self.model.decode_streams(streams)
                for (_p, _hw, key), stream in zip(chunk, streams):
                    self._remember(key, self._finalize_text(stream.result.text))

        results: list[str] = []
        for p, hw in zip(resolved_paths, hw_items):
            key = self._cache_key(p, hw)
            results.append(self.cache[key])
            self.cache.move_to_end(key)
        return results
