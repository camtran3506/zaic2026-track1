"""Offline GIPformer 1.5 ONNX transducer inference on CPU."""
from collections import OrderedDict
from pathlib import Path
from .text import normalize_transcript

ASR_REPO_ID = "g-group-ai-lab/gipformer1.5-68M-rnnt"
ASR_BACKEND = "sherpa-onnx"
DEFAULT_SFT_ASR_MODEL = Path(__file__).resolve().parents[1] / 'models' / 'asr_ft'
DEFAULT_ASR_DECODING_METHOD = "modified_beam_search"
DEFAULT_ASR_MAX_ACTIVE_PATHS = 4


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
    ):
        if num_threads < 1:
            raise ValueError("ASR num_threads must be positive")
        if decoding_method not in {"modified_beam_search", "greedy_search"}:
            raise ValueError("ASR decoding_method must be modified_beam_search or greedy_search")
        if max_active_paths < 1:
            raise ValueError("ASR max_active_paths must be positive")
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
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)
        self.precision = resolved_precision
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self.max_cache = max_cache
        self.cache: OrderedDict[str, str] = OrderedDict()

    def transcribe(self, path: Path) -> str:
        path = Path(path)
        key = str(path.resolve())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        samples, sample_rate = read_audio(path)
        stream = self.model.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self.model.decode_streams([stream])
        result = normalize_transcript(stream.result.text)
        self.cache[key] = result
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)
        return result

    def transcribe_batch(self, paths: list[Path], batch_size: int = 16) -> list[str]:
        """Batched offline transcription for SFT preparation; inference uses single-stream transcribe()."""
        if not paths:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        resolved_paths = [Path(p).resolve() for p in paths]
        unique_missing: list[Path] = []
        for p in resolved_paths:
            key = str(p)
            if key not in self.cache and p not in unique_missing:
                unique_missing.append(p)

        for start_idx in range(0, len(unique_missing), batch_size):
            chunk = unique_missing[start_idx:start_idx + batch_size]
            streams = []
            for p in chunk:
                samples, sample_rate = read_audio(p)
                stream = self.model.create_stream()
                stream.accept_waveform(sample_rate, samples)
                streams.append(stream)

            if streams:
                self.model.decode_streams(streams)
                for p, stream in zip(chunk, streams):
                    text = normalize_transcript(stream.result.text)
                    key = str(p)
                    self.cache[key] = text
                    if len(self.cache) > self.max_cache:
                        self.cache.popitem(last=False)

        results: list[str] = []
        for p in resolved_paths:
            key = str(p)
            results.append(self.cache[key])
            self.cache.move_to_end(key)
        return results
