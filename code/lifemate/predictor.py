"""Shared single-process end-to-end (ASR + LLM -> ToolEnv) inference engine."""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import random
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agent import AgentConfig, LifeMateAgent
from .asr import (
    DEFAULT_ASR_DECODING_METHOD,
    DEFAULT_ASR_MAX_ACTIVE_PATHS,
    asr_extra_options,
    resolve_asr_precision,
)
from .asr_biasing import DEFAULT_ASR_HOTWORDS_SCORE
from .catalog import load_catalog
from .context import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_THINKING_BUDGET,
    validate_inference_budget,
)
from .runtime import DEFAULT_LLM_DTYPE, DEFAULT_SEED
from .schema_cache import DEFAULT_SCHEMA_CACHE_MB
from .transcripts import read_jsonl

LOG = logging.getLogger("lifemate.predictor")


def configure_offline_env() -> None:
    """Enforce offline Hugging Face and deterministic cuBLAS defaults before model load."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    """Set reproducible seeds across Python, NumPy, PyTorch, and CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def detect_split_name(data_root: Path) -> str:
    """Detect the unique split folder under data_root (e.g. /data/private_test)."""
    location = Path(data_root).expanduser().resolve()
    if not location.exists():
        raise FileNotFoundError(f"Data root does not exist: {location}")

    candidates: list[str] = []
    if location.is_dir():
        for entry in sorted(location.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if (entry / f"{name}_tools_catalog.json").is_file() or (entry / f"{name}.jsonl").is_file():
                candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple split directories found under {location}: {candidates}. "
            "Pass --split explicitly."
        )

    # Support pointing data_root directly at a single split directory.
    if (location / f"{location.name}_tools_catalog.json").is_file() or (
        location / f"{location.name}.jsonl"
    ).is_file():
        return location.name

    catalogs = sorted(location.glob("*_tools_catalog.json"))
    if len(catalogs) == 1:
        return catalogs[0].name[: -len("_tools_catalog.json")]

    raise FileNotFoundError(
        f"Cannot auto-detect split directory under {location} "
        "(expected <split>/<split>_tools_catalog.json or <split>/<split>.jsonl)."
    )


def resolve_split_layout(
    data_root: Path, split: str = "auto"
) -> tuple[str, Path, Path, tempfile.TemporaryDirectory | None]:
    """Resolve (split_name, split_dir, ToolEnv_root, optional temp holder)."""
    location = Path(data_root).expanduser().resolve()
    resolved_split = detect_split_name(location) if split == "auto" else split
    if (location / resolved_split / f"{resolved_split}.jsonl").is_file():
        return resolved_split, location / resolved_split, location, None
    if (location / f"{resolved_split}.jsonl").is_file():
        holder = tempfile.TemporaryDirectory(prefix="lifemate_split_root_")
        alias = Path(holder.name) / resolved_split
        alias.symlink_to(location, target_is_directory=True)
        return resolved_split, location, Path(holder.name), holder
    raise FileNotFoundError(
        f"Cannot locate {resolved_split}.jsonl under {location} or {location / resolved_split}. "
        "Point --data-root either to the split or to its parent."
    )


def import_split_tool_env(split_dir: Path):
    """Import ToolEnv strictly from split_dir/zaic_tools (never from a bundled copy)."""
    split_resolved = Path(split_dir).resolve()
    pkg_dir = split_resolved / "zaic_tools"
    mod_file_src = split_resolved / "zaic_tools.py"
    if not (pkg_dir.is_dir() or mod_file_src.is_file()):
        raise FileNotFoundError(f"Missing zaic_tools inside split directory: {split_resolved}")

    existing = sys.modules.get("zaic_tools")
    if existing is not None:
        mod_file = Path(getattr(existing, "__file__", "") or "").resolve()
        if not mod_file.is_relative_to(split_resolved):
            for name in [k for k in sys.modules if k == "zaic_tools" or k.startswith("zaic_tools.")]:
                del sys.modules[name]

    split_str = str(split_resolved)
    if split_str in sys.path:
        sys.path.remove(split_str)
    sys.path.insert(0, split_str)

    mod = importlib.import_module("zaic_tools")
    mod_file = Path(getattr(mod, "__file__", "") or "").resolve()
    if not mod_file.is_relative_to(split_resolved):
        raise ImportError(f"zaic_tools resolved to {mod_file}, expected inside {split_resolved}")
    return mod.ToolEnv


def resolve_default_models(code_dir: Path | None = None) -> tuple[Path, Path, Path | None]:
    """Select default ASR, LLM, and optional LoRA adapter directories cleanly."""
    root = Path(code_dir).resolve() if code_dir is not None else Path(__file__).resolve().parents[1]
    models_dir = root / "saved_models"
    if not models_dir.is_dir():
        if (root / "models").is_dir():
            models_dir = root / "models"
        elif Path("/opt/saved_models").is_dir():
            models_dir = Path("/opt/saved_models")
        elif Path("/opt/models").is_dir():
            models_dir = Path("/opt/models")

    env_asr = os.environ.get("LIFEMATE_ASR_MODEL")
    if env_asr:
        asr_model = Path(env_asr).expanduser()
    elif (models_dir / "asr_ft" / "tokens.txt").is_file() and (models_dir / "asr_ft" / "tokens.txt").stat().st_size > 0:
        asr_model = models_dir / "asr_ft"
    else:
        asr_model = models_dir / "asr"

    env_llm = os.environ.get("LIFEMATE_LLM_MODEL")
    if env_llm:
        llm_model = Path(env_llm).expanduser()
    elif (models_dir / "llm_ft" / "config.json").is_file():
        llm_model = models_dir / "llm_ft"
    else:
        llm_model = models_dir / "llm"

    env_adapter = os.environ.get("LIFEMATE_LLM_ADAPTER")
    if env_adapter:
        llm_adapter: Path | None = Path(env_adapter).expanduser()
    elif llm_model.name != "llm_ft" and (models_dir / "llm_adapter" / "adapter_config.json").is_file():
        llm_adapter = models_dir / "llm_adapter"
    else:
        llm_adapter = None

    return asr_model, llm_model, llm_adapter


@dataclass
class PredictorConfig:
    """Configuration for single-process resident ASR + LLM inference."""

    asr_model: Path
    llm_model: Path
    llm_adapter: Path | None = None
    device: str = "cuda"
    llm_dtype: str = DEFAULT_LLM_DTYPE
    llm_device_map: str = "single"
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    thinking_budget: int = DEFAULT_THINKING_BUDGET
    thinking_reserve_seconds: float = 2.0
    prefix_cache: str = "auto"
    prompt_cache: str = "auto"
    schema_cache: str = "auto"
    schema_cache_mb: int = DEFAULT_SCHEMA_CACHE_MB
    asr_precision: str = "auto"
    asr_decoding_method: str = DEFAULT_ASR_DECODING_METHOD
    asr_beam_size: int = 8
    asr_hotwords: str = "auto"
    asr_hotwords_score: float = DEFAULT_ASR_HOTWORDS_SCORE
    asr_post_normalize: bool = True
    asr_threads: int = 8
    max_calls: int = 8
    max_invalid: int = 3
    sample_seconds: float = float("inf")
    use_train_transcript: bool = False
    seed: int = DEFAULT_SEED
    fail_fast: bool = False

    def validate(self) -> None:
        validate_inference_budget(self.max_input_tokens, self.max_new_tokens)
        if self.asr_threads < 1:
            raise ValueError("--asr-threads must be positive")
        if self.asr_beam_size < 1:
            raise ValueError("--asr-beam-size must be positive")
        if not math.isfinite(self.asr_hotwords_score) or self.asr_hotwords_score <= 0:
            raise ValueError("--asr-hotwords-score must be finite and positive")
        if math.isnan(self.sample_seconds) or self.sample_seconds <= 0:
            raise ValueError("--sample-seconds must be positive (inf = unlimited)")
        if self.thinking_budget < 0:
            raise ValueError("--thinking-budget must be non-negative")
        if math.isnan(self.thinking_reserve_seconds) or self.thinking_reserve_seconds < 0:
            raise ValueError("--thinking-reserve-seconds must be finite and non-negative")


def timing_summary(
    times_ms: list[int],
    *,
    load_seconds: float = 0.0,
    asr_load_seconds: float = 0.0,
    llm_load_seconds: float = 0.0,
    total_predict_seconds: float = 0.0,
    asr_ms_list: list[float] | None = None,
) -> dict[str, Any]:
    """Compute latency percentiles and SLA compliance stats matching BTC guidelines."""
    n = len(times_ms)
    if n == 0:
        return {
            "num_samples": 0,
            "asr_load_seconds": round(asr_load_seconds, 4),
            "llm_load_seconds": round(llm_load_seconds, 4),
            "model_load_seconds": round(load_seconds, 4),
            "total_predict_seconds": round(total_predict_seconds, 4),
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0,
            "max_ms": 0,
            "over_30s_count": 0,
        }
    ordered = sorted(times_ms)
    p95_idx = min(n - 1, math.ceil(0.95 * n) - 1)
    summary: dict[str, Any] = {
        "num_samples": n,
        "asr_load_seconds": round(asr_load_seconds, 4),
        "llm_load_seconds": round(llm_load_seconds, 4),
        "model_load_seconds": round(load_seconds, 4),
        "total_predict_seconds": round(total_predict_seconds, 4),
        "mean_ms": round(sum(times_ms) / n, 2),
        "median_ms": round(float(statistics.median(ordered)), 2),
        "p95_ms": round(float(ordered[p95_idx]), 2),
        "min_ms": int(ordered[0]),
        "max_ms": int(ordered[-1]),
        "over_30s_count": sum(1 for ms in times_ms if ms > 30000),
    }
    if asr_ms_list:
        mean_asr = sum(asr_ms_list) / len(asr_ms_list)
        summary["asr_mean_ms"] = round(mean_asr, 2)
        summary["asr_max_ms"] = round(max(asr_ms_list), 2)
        summary["llm_agent_mean_ms"] = round(summary["mean_ms"] - mean_asr, 2)
    return summary


def compare_submissions(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Compare two JSONL submissions for exact id, tool_calls, and state_hash parity."""
    rows_a = read_jsonl(Path(path_a))
    rows_b = read_jsonl(Path(path_b))
    mismatches: list[dict[str, Any]] = []
    min_len = min(len(rows_a), len(rows_b))
    for idx in range(min_len):
        a, b = rows_a[idx], rows_b[idx]
        if (
            a.get("id") != b.get("id")
            or a.get("tool_calls") != b.get("tool_calls")
            or a.get("state_hash") != b.get("state_hash")
        ):
            mismatches.append(
                {
                    "index": idx,
                    "a_id": a.get("id"),
                    "b_id": b.get("id"),
                    "a_hash": a.get("state_hash"),
                    "b_hash": b.get("state_hash"),
                }
            )
    return {
        "matches": len(rows_a) == len(rows_b) and not mismatches,
        "count_a": len(rows_a),
        "count_b": len(rows_b),
        "mismatch_count": len(mismatches) + abs(len(rows_a) - len(rows_b)),
        "mismatches": mismatches[:10],
    }


class _TimedASRProxy:
    """Measure live ASR wall time per sample without modifying LifeMateAgent."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.seconds = 0.0

    def reset(self) -> None:
        self.seconds = 0.0

    def transcribe(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return self._inner.transcribe(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - t0

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class LifeMatePredictor:
    """Single-process inference runner keeping ASR (CPU) and LLM (GPU) loaded together."""

    def __init__(self, cfg: PredictorConfig) -> None:
        configure_offline_env()
        cfg.validate()
        self.cfg = cfg
        seed_everything(cfg.seed)

        # Import from lifemate.models so unit tests monkeypatching LocalASR/LocalLLM work directly.
        import lifemate.models as runtime_models

        load_start = time.perf_counter()

        asr_dir = Path(cfg.asr_model).expanduser()
        precision = (
            resolve_asr_precision(asr_dir, cfg.asr_precision)
            if cfg.asr_precision == "auto" and asr_dir.is_dir()
            else cfg.asr_precision
        )
        extra = asr_extra_options(cfg)
        asr_t0 = time.perf_counter()
        self.asr = runtime_models.LocalASR(
            asr_dir,
            precision=precision,
            num_threads=cfg.asr_threads,
            **extra,
        )
        self.asr_load_seconds = time.perf_counter() - asr_t0

        llm_t0 = time.perf_counter()
        try:
            self.llm = runtime_models.LocalLLM(
                Path(cfg.llm_model).expanduser(),
                device=cfg.device,
                dtype=cfg.llm_dtype,
                max_input_tokens=cfg.max_input_tokens,
                max_new_tokens=cfg.max_new_tokens,
                thinking_budget=cfg.thinking_budget,
                adapter_path=cfg.llm_adapter,
                device_map=cfg.llm_device_map,
                seed=cfg.seed,
                prefix_cache=cfg.prefix_cache,
                prompt_cache=cfg.prompt_cache,
                schema_cache=cfg.schema_cache,
                schema_cache_mb=cfg.schema_cache_mb,
                schema_cache_eager=False,
            )
        except Exception:
            close_asr = getattr(self.asr, "close", None)
            if callable(close_asr):
                close_asr()
            raise
        self.llm_load_seconds = time.perf_counter() - llm_t0
        self.load_seconds = time.perf_counter() - load_start
        self._closed = False

    def build_agent(self, catalog: dict[str, dict], *, asr_override: Any = None) -> LifeMateAgent:
        agent_cfg = AgentConfig(
            max_calls=self.cfg.max_calls,
            max_invalid=self.cfg.max_invalid,
            sample_seconds=self.cfg.sample_seconds,
            thinking_budget=self.cfg.thinking_budget,
            thinking_reserve_seconds=self.cfg.thinking_reserve_seconds,
            use_train_transcript=self.cfg.use_train_transcript,
        )
        return LifeMateAgent(
            asr=self.asr if asr_override is None else asr_override,
            llm=self.llm,
            catalog=catalog,
            cfg=agent_cfg,
        )

    def predict_sample(
        self,
        item: dict,
        env_cls: Any,
        split_dir: Path,
        split: str,
        env_root: Path,
        agent: LifeMateAgent,
        *,
        asr_proxy: _TimedASRProxy | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run one sample end-to-end with t1 measured before reading audio / instantiating ToolEnv."""
        import torch

        if asr_proxy is not None:
            asr_proxy.reset()

        t1 = time.perf_counter()
        env = env_cls(split, item["id"], root=str(env_root))
        try:
            report = agent.run(item, env, split_dir, split, prepared_transcript=None)
        except torch.OutOfMemoryError:
            LOG.exception("LLM out of memory on sample %s; aborting without CPU/disk fallback", item["id"])
            raise
        except Exception as exc:
            LOG.exception("Sample %s encountered an error", item["id"])
            if self.cfg.fail_fast:
                raise
            report = {"error": repr(exc)}
        calls = env.get_tool_calls()
        state_hash = env.get_state_hash()
        elapsed_seconds = time.perf_counter() - t1
        ms = int(elapsed_seconds * 1000)

        record = {
            "id": item["id"],
            "tool_calls": calls,
            "state_hash": state_hash,
            "time": ms,
        }
        diag = {
            "id": item["id"],
            "time_ms": ms,
            "sample_seconds": round(elapsed_seconds, 4),
            "calls": len(calls),
            **({"asr_seconds": round(asr_proxy.seconds, 4)} if asr_proxy is not None else {}),
            **report,
        }
        return record, diag

    def predict_dataset(
        self,
        data_root: Path,
        output_path: Path,
        *,
        split: str = "auto",
        limit: int = 0,
        log_path: Path | None = None,
        count_asr: bool = False,
        progress_cb: Callable[[int, int, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run sample-by-sample inference in original dataset order and flush lines immediately."""
        if limit < 0:
            raise ValueError("--limit must be non-negative")

        split_name, split_dir, env_root, holder = resolve_split_layout(data_root, split)
        try:
            if self.cfg.use_train_transcript and split_name != "train":
                raise ValueError("--use-train-transcript is forbidden for test splits")

            protected = {
                (split_dir / f"{split_name}.jsonl").resolve(),
                (split_dir / f"{split_name}_tools_catalog.json").resolve(),
            }
            output_resolved = Path(output_path).resolve()
            if output_resolved in protected:
                raise ValueError("Output path cannot overwrite dataset files")
            if log_path is not None:
                log_resolved = Path(log_path).resolve()
                if log_resolved in protected:
                    raise ValueError("Log path cannot overwrite dataset files")
                if log_resolved == output_resolved:
                    raise ValueError("Submission and diagnostics need distinct paths")

            ToolEnv = import_split_tool_env(split_dir)
            catalog = load_catalog(split_dir / f"{split_name}_tools_catalog.json")
            items = read_jsonl(split_dir / f"{split_name}.jsonl")
            if limit:
                items = items[:limit]

            asr_proxy = _TimedASRProxy(self.asr) if count_asr else None
            agent = self.build_agent(catalog, asr_override=asr_proxy)

            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if log_path is not None:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)

            times_ms: list[int] = []
            asr_ms_list: list[float] = []
            total = len(items)
            predict_start = time.perf_counter()

            fout = out_path.open("w", encoding="utf-8")
            flog = Path(log_path).open("w", encoding="utf-8") if log_path is not None else None
            try:
                for idx, item in enumerate(items, 1):
                    record, diag = self.predict_sample(
                        item,
                        ToolEnv,
                        split_dir,
                        split_name,
                        env_root,
                        agent,
                        asr_proxy=asr_proxy,
                    )
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    if flog is not None:
                        flog.write(json.dumps(diag, ensure_ascii=False) + "\n")
                        flog.flush()
                    times_ms.append(record["time"])
                    if asr_proxy is not None:
                        asr_ms_list.append(asr_proxy.seconds * 1000.0)
                    LOG.info(
                        "%d/%d %s: %d calls, %.2fs",
                        idx,
                        total,
                        item["id"],
                        len(record["tool_calls"]),
                        record["time"] / 1000.0,
                    )
                    if progress_cb is not None:
                        progress_cb(idx, total, record, diag)
            finally:
                fout.close()
                if flog is not None:
                    flog.close()

            predict_seconds = time.perf_counter() - predict_start
            stats = timing_summary(
                times_ms,
                load_seconds=self.load_seconds,
                asr_load_seconds=self.asr_load_seconds,
                llm_load_seconds=self.llm_load_seconds,
                total_predict_seconds=predict_seconds,
                asr_ms_list=asr_ms_list if count_asr else None,
            )
            stats["split"] = split_name
            stats["output_path"] = str(out_path)
            return stats
        finally:
            if holder is not None:
                holder.cleanup()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for comp in (getattr(self, "llm", None), getattr(self, "asr", None)):
            close_fn = getattr(comp, "close", None)
            if callable(close_fn):
                close_fn()

    def __enter__(self) -> LifeMatePredictor:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
