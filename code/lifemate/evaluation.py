"""Held-out TRAIN agent rollouts; no reads of application environment files."""
import copy
from contextlib import contextmanager
import importlib
import json
import logging
from pathlib import Path
import random
import sys

from .agent import AgentConfig, LifeMateAgent
from .catalog import load_catalog
from .eval_diagnostics import DiagnosticSummary, analyze_rollout
from .eval_trace import EvalTraceWriter, exception_details
from .scoring import call_f1, state_tool_score
from .sft import VALIDATION_ROLLOUT_POLICY, file_sha256, validate_prompt_policy


class StateToolEvaluator:
    def __init__(self, sft_dir, manifest, *, require_clean_transcripts=False, partition='validation'):
        if partition not in {'validation', 'app_heldout'}:
            raise ValueError('Unknown rollout partition')
        self.partition = partition
        validate_prompt_policy(manifest)
        info = manifest.get(f'{partition}_rollouts')
        if not info:
            raise ValueError("Missing validation rollout data; rerun prepare_sft.py with the current code")
        if (info.get("policy") != VALIDATION_ROLLOUT_POLICY
                or info.get("selection_transcript_source") != "asr"):
            raise ValueError("Validation rollouts must use the ASR-only selection policy; rerun prepare_sft.py")
        path = sft_dir / info["path"]
        if file_sha256(path) != info["sha256"]:
            raise ValueError("Validation rollout data changed after preparation")
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [r["item"]["id"] for r in self.rows]
        valid = manifest["partitions"][partition]
        if (not ids or len(set(ids)) != len(ids) or set(ids) != set(valid["sample_ids"])
                or set(ids) & set(manifest["partitions"]["train"]["sample_ids"])
                or valid["skipped_samples"]):
            raise ValueError("Rollouts must cover all held-out validation IDs, with no train IDs or skipped samples")
        for row in self.rows:
            sample_id = row["item"]["id"]
            prepared = row.get("prepared_transcript", {})
            if (prepared.get("id") != sample_id or prepared.get("transcript_source") != "asr"
                    or not isinstance(prepared.get("transcript"), str) or not isinstance(prepared.get("history"), list)):
                raise ValueError("Every held-out selection rollout requires ASR text; rerun prepare_sft.py")
            clean = row.get("clean_prepared_transcript")
            if clean is not None and (clean.get("id") != sample_id
                    or clean.get("transcript_source") != "audio_text"
                    or not isinstance(clean.get("transcript"), str) or clean.get("history") != prepared["history"]):
                raise ValueError("Clean diagnostic views must match the ASR held-out IDs and history; rerun prepare_sft.py")
        if require_clean_transcripts and any(not row.get("clean_prepared_transcript") for row in self.rows):
            raise ValueError("Clean diagnostic evaluation requires audio_text for every held-out sample")
        source = Path(manifest["source"])
        if file_sha256(source) != manifest["source_sha256"]:
            raise ValueError("Source TRAIN data changed after preparation")
        self.split_dir = source.parent
        catalog_path = self.split_dir / "train_tools_catalog.json"
        if file_sha256(catalog_path) != manifest["catalog_sha256"]:
            raise ValueError("TRAIN catalog changed after preparation")
        self.catalog = load_catalog(catalog_path)
        if manifest.get('app_holdout', {}).get('status') == 'enabled':
            from .data_generalization import validate_split_manifest
            validate_split_manifest(manifest, self.catalog)
        self.config = AgentConfig()
        from main import resolve_data_layout
        _, self.env_root, self.holder = resolve_data_layout(self.split_dir, "train")
        sys.path.insert(0, str(self.split_dir))
        self.tool_env = importlib.import_module("zaic_tools").ToolEnv

    def close(self):
        if self.holder is not None:
            self.holder.cleanup()

    def evaluate(self, llm, output_path, *, transcript_source="asr"):
        """Run one independent rollout/view; only ASR metrics select checkpoints."""
        if transcript_source not in {"asr", "audio_text"}:
            raise ValueError("Validation transcript source must be asr or audio_text")
        prepared_key = "prepared_transcript" if transcript_source == "asr" else "clean_prepared_transcript"
        if any(not row.get(prepared_key) for row in self.rows):
            raise ValueError("Clean diagnostic evaluation requires audio_text for every held-out sample")
        output_path = Path(output_path)
        from .schema_cache import schedule_by_schema, restore_original_order
        orig_indices, _, sched_rows = schedule_by_schema([r["item"] for r in self.rows], self.rows)
        sched_scores, sched_f1s, sched_hashes = [], [], []
        sched_records = []
        diagnostics = DiagnosticSummary()
        agent = LifeMateAgent(None, llm, self.catalog, self.config)
        partial = output_path.with_suffix(".partial.jsonl")
        trace_path = output_path.with_suffix(".trace.jsonl")
        trace_partial = output_path.with_suffix(".trace.partial.jsonl")
        logging.info("Validation debug trace (in progress): %s", trace_partial)
        with partial.open("w", encoding="utf-8") as partial_stream, trace_partial.open("w", encoding="utf-8") as trace_stream:
            trace = EvalTraceWriter(trace_stream)
            for row in sched_rows:
                # Golds are used only below for scoring, never passed into the agent.
                item = copy.deepcopy(row["item"])
                item.pop("label", None)
                prepared = row[prepared_key]
                source = prepared["transcript_source"]
                emit = trace.for_sample(item["id"])
                emit({"event": "sample_start", "transcript_source": source,
                      "request_transcript": prepared["transcript"]})
                try:
                    env = self.tool_env("train", item["id"], root=str(self.env_root))
                    stats = agent.run(item, env, self.split_dir, "train",
                        prepared_transcript=copy.deepcopy(prepared), trace=emit)
                    calls, state_hash = env.get_tool_calls(), env.get_state_hash()
                    if not isinstance(calls, list) or not isinstance(state_hash, str) or not state_hash:
                        raise ValueError(f"{item['id']}: invalid ToolEnv call log/state hash")
                    score = state_tool_score(calls, row["gold_tool_calls"], state_hash,
                                             row["gold_state_hash"], self.catalog)
                    f1, _ = call_f1(calls, row["gold_tool_calls"], self.catalog)
                    diagnostic = analyze_rollout(calls, row["gold_tool_calls"], state_hash,
                        row["gold_state_hash"], self.catalog, stats.get("stop_reason", "unknown"))
                    diagnostics.add(diagnostic, source)
                    sched_scores.append(score)
                    sched_f1s.append(f1)
                    sched_hashes.append(state_hash == row["gold_state_hash"])
                    record = {"id": item["id"], "tool_calls": calls,
                        "state_hash": state_hash, "gold_state_hash": row["gold_state_hash"],
                        "gold_tool_calls": row["gold_tool_calls"], "transcript_source": source,
                        "score": score, "tool_call_f1": f1, "agent_stats": stats,
                        "trace_file": trace_path.name, "diagnostics": diagnostic}
                    sched_records.append(record)
                    partial_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    partial_stream.flush()
                    emit({"event": "sample_end", "agent_stats": stats, "tool_calls": calls,
                          "state_hash": state_hash})
                except Exception as exc:
                    emit({"event": "sample_error", "exception": exception_details(exc)})
                    logging.exception("Validation sample %s failed; partial trace: %s", item["id"], trace_partial)
                    raise
                logging.info("Validation rollout %d/%d %s: score %.4f, hash_match=%s",
                             len(sched_scores), len(self.rows), item["id"], score, sched_hashes[-1])
        scores = restore_original_order(orig_indices, sched_scores)
        f1s = restore_original_order(orig_indices, sched_f1s)
        hashes = restore_original_order(orig_indices, sched_hashes)
        records = restore_original_order(orig_indices, sched_records)
        with partial.open("w", encoding="utf-8") as output:
            for rec in records:
                output.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = len(scores)
        metrics = {"eval_state_tool_score": 100 * sum(scores) / n,
                "eval_tool_call_f1": 100 * sum(f1s) / n,
                "eval_state_match_rate": 100 * sum(hashes) / n,
                "eval_rollout_samples": n, **diagnostics.metrics()}
        if transcript_source == "audio_text":
            metrics = {key.replace("eval_", "eval_clean_", 1): value for key, value in metrics.items()}
        if self.partition == 'app_heldout':
            metrics = {key.replace('eval_', 'eval_app_heldout_', 1): value for key, value in metrics.items()}
        summary_path = output_path.with_suffix(".diagnostics.json")
        summary_partial = summary_path.with_suffix(".partial.json")
        summary_partial.write_text(json.dumps({**diagnostics.as_dict(), "metrics": metrics,
                                              "trace_file": trace_path.name},
                                              ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        partial.replace(output_path)
        trace_partial.replace(trace_path)
        summary_partial.replace(summary_path)
        logging.info("Validation debug trace saved: %s", trace_path)
        return metrics


@contextmanager
def _rollout_inference_mode(model, tokenizer, fast_language_model):
    """Restore training/config/RNG state even if rollout or mode switching fails."""
    import numpy as np
    import torch
    was_training = model.training
    configs = [model.config]
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and text_config is not model.config:
        configs.append(text_config)
    cache_settings = [(cfg, cfg.use_cache) for cfg in configs if hasattr(cfg, "use_cache")]
    sides = tokenizer.padding_side, tokenizer.truncation_side
    py_rng, np_rng = random.getstate(), np.random.get_state()
    device = model.get_input_embeddings().weight.device
    devices = [device.index] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            if fast_language_model is not None:
                fast_language_model.for_inference(model)
            model.eval()
            for cfg, _ in cache_settings:
                cfg.use_cache = True
            yield
        finally:
            try:
                if fast_language_model is not None:
                    fast_language_model.for_training(model)
            finally:
                model.train(was_training)
                for cfg, value in cache_settings:
                    cfg.use_cache = value
                tokenizer.padding_side, tokenizer.truncation_side = sides
                random.setstate(py_rng)
                np.random.set_state(np_rng)


def score_callback(evaluator, tokenizer, output_dir, fast_language_model=None, *, eval_clean_transcripts=False,
                   eval_prefix_cache="auto", eval_prompt_cache="auto", app_evaluator=None,
                   eval_schema_cache='auto', eval_schema_cache_mb=256):
    # Called only after Unsloth has been imported/patched by the training entrypoint.
    from transformers import TrainerCallback
    from .models import LocalLLM

    class ScoreCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Always score the last candidate, even when fewer than save_steps were run.
            if args.eval_strategy == "steps" and state.global_step >= state.max_steps:
                control.should_evaluate = control.should_save = True
            return control

        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            # Keep all evaluation attempts (including re-evaluation of the loaded best model).
            folder = output_dir / "validation_rollouts"
            folder.mkdir(exist_ok=True)
            import tempfile
            with tempfile.NamedTemporaryFile(prefix=f"step-{state.global_step}-", suffix=".jsonl",
                                             dir=folder, delete=False) as tmp:
                path = Path(tmp.name)
            path.unlink()  # Only a fully completed evaluation promotes a final JSONL.
            with _rollout_inference_mode(model, tokenizer, fast_language_model):
                # Fresh wrapper after each weight update. No snapshot may cross
                # checkpoints, transcript views, or the return to backward().
                llm = LocalLLM.from_loaded_model(model, tokenizer,
                    prefix_cache=eval_prefix_cache, prompt_cache=eval_prompt_cache,
                    schema_cache=eval_schema_cache, schema_cache_mb=eval_schema_cache_mb,
                    schema_cache_eager=True)
                try:
                    def view(ev, target, source=None):
                        try:
                            return ev.evaluate(llm, target, **({'transcript_source': source} if source else {}))
                        finally:
                            llm.close()
                    result = view(evaluator, path)
                    if eval_clean_transcripts:
                        clean_path = path.with_name(path.stem + ".clean.jsonl")
                        result.update(view(evaluator, clean_path, 'audio_text'))
                    if app_evaluator is not None:
                        app_path = path.with_name(path.stem + '.app-heldout.jsonl')
                        result.update(view(app_evaluator, app_path))
                        if eval_clean_transcripts:
                            app_clean_path = path.with_name(path.stem + '.app-heldout.clean.jsonl')
                            result.update(view(app_evaluator, app_clean_path, 'audio_text'))
                finally:
                    llm.close()
            metrics.update(result)
            if state.log_history and state.log_history[-1].get("step") == state.global_step:
                state.log_history[-1].update(result)
            with (output_dir / "validation_scores.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                    "rollouts": str(path), "diagnostics": str(path.with_suffix(".diagnostics.json")),
                    "trace": str(path.with_suffix(".trace.jsonl")),
                    "selection_transcript_source": "asr",
                    **({"clean_rollouts": str(clean_path),
                        "clean_diagnostics": str(clean_path.with_suffix(".diagnostics.json")),
                        "clean_trace": str(clean_path.with_suffix(".trace.jsonl"))}
                       if eval_clean_transcripts else {}),
                    **({'app_heldout_rollouts': str(app_path),
                        'app_heldout_diagnostics': str(app_path.with_suffix('.diagnostics.json')),
                        'app_heldout_trace': str(app_path.with_suffix('.trace.jsonl'))}
                       if app_evaluator is not None else {}),
                    **({'app_heldout_clean_rollouts': str(app_clean_path),
                        'app_heldout_clean_diagnostics': str(app_clean_path.with_suffix('.diagnostics.json')),
                        'app_heldout_clean_trace': str(app_clean_path.with_suffix('.trace.jsonl'))}
                       if app_evaluator is not None and eval_clean_transcripts else {}),
                    **result}) + "\n")
            logging.info("Validation state/tool score: %.4f/100", result["eval_state_tool_score"])
            return control

    return ScoreCallback()
