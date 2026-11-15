#!/usr/bin/env python3
"""Timing benchmark & submission parity script following BTC Appendix (Section 7)."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from lifemate.predictor import (
    LifeMatePredictor,
    _TimedASRProxy,
    compare_submissions,
    configure_offline_env,
    import_split_tool_env,
    resolve_split_layout,
    seed_everything,
    timing_summary,
)

configure_offline_env()

from lifemate.catalog import load_catalog
from lifemate.gpu import configure_cuda_device
from lifemate.transcripts import read_jsonl
from predict import build_arg_parser, build_predictor_config

LOG = logging.getLogger("lifemate.benchmark")


def parse_args(argv: list[str] | None = None):
    parser = build_arg_parser(
        default_output=Path("/result/jupyter_submission.jsonl"),
        description="Step-by-step latency measurement & parity check following BTC Appendix Section 7",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional reference submission.jsonl to check id/tool_calls/state_hash parity",
    )
    parser.add_argument(
        "--strict-compare",
        action="store_true",
        help="Exit non-zero if --compare detects any mismatch in id, tool_calls, or state_hash",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to save detailed JSON timing & per-sample breakdown report",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    configure_cuda_device(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Step 1: Set Seed (before loading any model)
    seed_everything(args.seed)
    print(f"[Step 1/4] Seed set to {args.seed}", flush=True)

    # Step 2: Load ASR and LLM upfront in one process & print load time
    cfg = build_predictor_config(args)
    predictor = LifeMatePredictor(cfg)
    try:
        print(
            f"[Step 2/4] Loaded ASR + LLM in {predictor.load_seconds:.2f}s "
            f"(ASR={predictor.asr_load_seconds:.2f}s, LLM={predictor.llm_load_seconds:.2f}s)",
            flush=True,
        )

        # Step 3: Load dataset split & strictly import zaic_tools from split directory
        split_name, split_dir, env_root, holder = resolve_split_layout(args.data_root, args.split)
        try:
            if cfg.use_train_transcript and split_name != "train":
                raise ValueError("--use-train-transcript is forbidden for test splits")

            ToolEnv = import_split_tool_env(split_dir)
            catalog = load_catalog(split_dir / f"{split_name}_tools_catalog.json")
            test_cases = read_jsonl(split_dir / f"{split_name}.jsonl")
            if args.limit:
                test_cases = test_cases[: args.limit]
            print(
                f"[Step 3/4] Split={split_name} | Samples={len(test_cases)} | DataDir={split_dir}",
                flush=True,
            )

            # Step 4: Sequential sample-by-sample timing (t1 before audio -> ToolEnv -> ASR -> LLM -> ms)
            asr_proxy = _TimedASRProxy(predictor.asr)
            agent = predictor.build_agent(catalog, asr_override=asr_proxy)

            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = None
            if args.log_path is not None:
                Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
                log_fh = Path(args.log_path).open("w", encoding="utf-8")

            times_ms: list[int] = []
            asr_ms_list: list[float] = []
            per_sample: list[dict[str, Any]] = []
            predict_t0 = time.perf_counter()

            with out_path.open("w", encoding="utf-8") as fout:
                try:
                    for idx, test_case in enumerate(test_cases, 1):
                        record, diag = predictor.predict_sample(
                            test_case,
                            ToolEnv,
                            split_dir,
                            split_name,
                            env_root,
                            agent,
                            asr_proxy=asr_proxy,
                        )
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fout.flush()
                        if log_fh is not None:
                            log_fh.write(json.dumps(diag, ensure_ascii=False) + "\n")
                            log_fh.flush()

                        asr_ms = round(asr_proxy.seconds * 1000.0, 2)
                        llm_agent_ms = round(max(0.0, record["time"] - asr_ms), 2)
                        times_ms.append(record["time"])
                        asr_ms_list.append(asr_ms)
                        per_sample.append(
                            {
                                "index": idx,
                                "id": record["id"],
                                "time_ms": record["time"],
                                "asr_ms": asr_ms,
                                "llm_agent_ms": llm_agent_ms,
                                "calls": len(record["tool_calls"]),
                                "state_hash": record["state_hash"],
                            }
                        )
                finally:
                    if log_fh is not None:
                        log_fh.close()

            total_predict_seconds = time.perf_counter() - predict_t0
        finally:
            if holder is not None:
                holder.cleanup()
    finally:
        predictor.close()

    stats = timing_summary(
        times_ms,
        load_seconds=predictor.load_seconds,
        asr_load_seconds=predictor.asr_load_seconds,
        llm_load_seconds=predictor.llm_load_seconds,
        total_predict_seconds=total_predict_seconds,
        asr_ms_list=asr_ms_list,
    )
    stats["split"] = split_name
    stats["output_path"] = str(out_path)

    print(
        f"[Step 4/4] Completed {stats['num_samples']} samples in {stats['total_predict_seconds']:.2f}s | "
        f"Mean={stats['mean_ms'] / 1000.0:.2f}s (ASR={stats.get('asr_mean_ms', 0.0) / 1000.0:.2f}s, "
        f"LLM+Agent={stats.get('llm_agent_mean_ms', 0.0) / 1000.0:.2f}s) | "
        f"Median={stats['median_ms'] / 1000.0:.2f}s | P95={stats['p95_ms'] / 1000.0:.2f}s | "
        f"Max={stats['max_ms'] / 1000.0:.2f}s | >30s={stats['over_30s_count']}",
        flush=True,
    )
    print(f"[Step 4/4] Saved output to {out_path}", flush=True)

    if args.compare is not None:
        parity = compare_submissions(args.compare, out_path)
        stats["parity"] = parity
        print(
            f"[Parity] matches={parity['matches']} "
            f"(ref={parity['count_a']}, new={parity['count_b']}, mismatches={parity['mismatch_count']})",
            flush=True,
        )
        if args.strict_compare and not parity["matches"]:
            raise SystemExit(
                f"Submission parity mismatch between {args.compare} and {out_path}: {parity['mismatches']}"
            )

    if args.report_json is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": stats, "samples": per_sample}
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Report] Saved timing report to {report_path}", flush=True)

    return stats


if __name__ == "__main__":
    main()
