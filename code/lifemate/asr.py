"""Offline GIPformer 1.5 ONNX transducer inference on CPU."""
from collections import OrderedDict
from pathlib import Path

from .text import normalize_transcript


ASR_REPO_ID = "g-group-ai-lab/gipformer1.5-68M-rnnt"
ASR_BACKEND = "sherpa-onnx"


def asr_files(model_path: Path, precision: str = "int8") -> dict[str, Path]:
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
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
    def __init__(self, model_path: Path, precision: str = "int8", num_threads: int = 4):
        if num_threads < 1:
            raise ValueError("ASR num_threads must be positive")
        files = asr_files(model_path, precision)
        import sherpa_onnx
        self.model = sherpa_onnx.OfflineRecognizer.from_transducer(
            **{name: str(path) for name, path in files.items()},
            num_threads=num_threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", provider="cpu",
        )
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
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return result
