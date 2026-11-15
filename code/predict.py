#!/usr/bin/env python3
"""BTC-compliant single-process Docker entrypoint for Zalo AI Challenge 2026 - LifeMate."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from lifemate.predictor import configure_offline_env

configure_offline_env()

from lifemate.asr_biasing import DEFAULT_ASR_HOTWORDS_SCORE
from lifemate.context import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS, DEFAULT_THINKING_BUDGET
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device
from lifemate.predictor import (
    LifeMatePredictor,
    PredictorConfig,
    resolve_default_models,
)
from lifemate.runtime import DEFAULT_LLM_DTYPE, DEFAULT_SEED
from lifemate.schema_cache import DEFAULT_SCHEMA_CACHE_MB, cache_budget_mb

LOG = logging.getLogger("lifemate.predict")
CODE_DIR = Path(__file__).resolve().parent


def build_arg_parser(
    *,
    default_output: Path = Path("/result/submission.jsonl"),
    description: str = "Run single-process ASR + LLM end-to-end prediction (/data -> /result/submission.jsonl)",
) -> argparse.ArgumentParser:
    default_asr, default_llm, default_adapter = resolve_default_models(CODE_DIR)
    p = argparse.ArgumentParser(description=description)
    add_cuda_device_argument(p)
    p.add_argument("--data-root", type=Path, default=Path("/data"))
    p.add_argument("--split", choices=["auto", "train", "public_test", "private_test"], default="auto")
    p.add_argument("--output", type=Path, default=default_output)
    p.add_argument("--log-path", type=Path, default=None)
    p.add_argument("--asr-model", type=Path, default=default_asr)
    p.add_argument("--llm-model", type=Path, default=default_llm)
    p.add_argument("--llm-adapter", type=Path, default=default_adapter)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--llm-dtype", default=DEFAULT_LLM_DTYPE, choices=["float16", "bfloat16", "float32"])
    p.add_argument("--llm-device-map", choices=["single"], default="single")
    p.add_argument("--asr-precision", default="auto", choices=["auto", "int8", "fp32"])
    p.add_argument(
        "--asr-decoding-method",
        default="modified_beam_search",
        choices=["modified_beam_search", "greedy_search"],
    )
    p.add_argument("--asr-beam-size", type=int, default=8)
    p.add_argument("--asr-hotwords", default="auto", choices=["off", "auto"])
    p.add_argument("--asr-hotwords-score", type=float, default=DEFAULT_ASR_HOTWORDS_SCORE)
    p.add_argument("--asr-post-normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--asr-threads", type=int, default=8)
    p.add_argument("--prefix-cache", choices=["auto", "off"], default="auto")
    p.add_argument("--prompt-cache", choices=["auto", "off"], default="auto")
    p.add_argument("--schema-cache", choices=["auto", "off", "shared"], default="auto")
    p.add_argument("--schema-cache-mb", type=cache_budget_mb, default=DEFAULT_SCHEMA_CACHE_MB)
    p.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--thinking-budget", type=int, default=DEFAULT_THINKING_BUDGET)
    p.add_argument("--thinking-reserve-seconds", type=float, default=2.0)
    p.add_argument("--max-calls", type=int, default=8)
    p.add_argument("--max-invalid", type=int, default=3)
    p.add_argument("--sample-seconds", type=float, default=float("inf"))
    p.add_argument("--use-train-transcript", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--fail-fast", action="store_true")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def build_predictor_config(args: argparse.Namespace) -> PredictorConfig:
    return PredictorConfig(
        asr_model=args.asr_model.expanduser().resolve(),
        llm_model=args.llm_model.expanduser().resolve(),
        llm_adapter=args.llm_adapter.expanduser().resolve() if args.llm_adapter else None,
        device=args.device,
        llm_dtype=args.llm_dtype,
        llm_device_map=args.llm_device_map,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=args.thinking_budget,
        thinking_reserve_seconds=args.thinking_reserve_seconds,
        prefix_cache=args.prefix_cache,
        prompt_cache=args.prompt_cache,
        schema_cache=args.schema_cache,
        schema_cache_mb=args.schema_cache_mb,
        asr_precision=args.asr_precision,
        asr_decoding_method=args.asr_decoding_method,
        asr_beam_size=args.asr_beam_size,
        asr_hotwords=args.asr_hotwords,
        asr_hotwords_score=args.asr_hotwords_score,
        asr_post_normalize=args.asr_post_normalize,
        asr_threads=args.asr_threads,
        max_calls=args.max_calls,
        max_invalid=args.max_invalid,
        sample_seconds=args.sample_seconds,
        use_train_transcript=args.use_train_transcript,
        seed=args.seed,
        fail_fast=args.fail_fast,
    )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    configure_cuda_device(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = build_predictor_config(args)
    with LifeMatePredictor(cfg) as predictor:
        print(
            f"[predict] Model load time: {predictor.load_seconds:.2f}s "
            f"(ASR: {predictor.asr_load_seconds:.2f}s, LLM: {predictor.llm_load_seconds:.2f}s)",
            flush=True,
        )
        stats = predictor.predict_dataset(
            args.data_root,
            args.output,
            split=args.split,
            limit=args.limit,
            log_path=args.log_path,
        )

    print(
        f"[predict] Split={stats['split']} | Samples={stats['num_samples']} | "
        f"Total predict={stats['total_predict_seconds']:.2f}s | "
        f"Mean={stats['mean_ms'] / 1000.0:.2f}s | Median={stats['median_ms'] / 1000.0:.2f}s | "
        f"P95={stats['p95_ms'] / 1000.0:.2f}s | Max={stats['max_ms'] / 1000.0:.2f}s | "
        f">30s={stats['over_30s_count']} | Output={stats['output_path']}",
        flush=True,
    )
    return stats


if __name__ == "__main__":
    main()
