"""Held-out TRAIN agent rollouts; no reads of application environment files."""
import copy
import importlib
import json
import logging
from pathlib import Path
import random
import sys

from .agent import AgentConfig, LifeMateAgent
from .catalog import load_catalog
from .scoring import call_f1, state_tool_score
from .sft import file_sha256, validate_prompt_policy


class StateToolEvaluator:
    def __init__(self, sft_dir, manifest):
        validate_prompt_policy(manifest)
        info = manifest.get("validation_rollouts")
        if not info:
            raise ValueError("Missing validation rollout data; rerun prepare_sft.py with the current code")
        path = sft_dir / info["path"]
        if file_sha256(path) != info["sha256"]:
            raise ValueError("Validation rollout data changed after preparation")
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [r["item"]["id"] for r in self.rows]
        valid = manifest["partitions"]["validation"]
        if (not ids or len(set(ids)) != len(ids) or set(ids) != set(valid["sample_ids"])
                or set(ids) & set(manifest["partitions"]["train"]["sample_ids"])
                or valid["skipped_samples"]):
            raise ValueError("Rollouts must cover all held-out validation IDs, with no train IDs or skipped samples")
        source = Path(manifest["source"])
        if file_sha256(source) != manifest["source_sha256"]:
            raise ValueError("Source TRAIN data changed after preparation")
        self.split_dir = source.parent
        catalog_path = self.split_dir / "train_tools_catalog.json"
        if file_sha256(catalog_path) != manifest["catalog_sha256"]:
            raise ValueError("TRAIN catalog changed after preparation")
        self.catalog = load_catalog(catalog_path)
        self.config = AgentConfig()
        from main import resolve_data_layout
        _, self.env_root, self.holder = resolve_data_layout(self.split_dir, "train")
        sys.path.insert(0, str(self.split_dir))
        self.tool_env = importlib.import_module("zaic_tools").ToolEnv

    def close(self):
        if self.holder is not None:
            self.holder.cleanup()

    def evaluate(self, llm, output_path):
        scores, f1s, hashes = [], [], []
        agent = LifeMateAgent(None, llm, self.catalog, self.config)
        partial = output_path.with_suffix(".partial.jsonl")
        with partial.open("w", encoding="utf-8") as output:
            for row in self.rows:
                # Golds are used only below for scoring, never passed into the agent.
                item = copy.deepcopy(row["item"])
                item.pop("label", None)
                env = self.tool_env("train", item["id"], root=str(self.env_root))
                stats = agent.run(item, env, self.split_dir, "train",
                                  prepared_transcript=copy.deepcopy(row["prepared_transcript"]))
                calls, state_hash = env.get_tool_calls(), env.get_state_hash()
                if not isinstance(calls, list) or not isinstance(state_hash, str) or not state_hash:
                    raise ValueError(f"{item['id']}: invalid ToolEnv call log/state hash")
                score = state_tool_score(calls, row["gold_tool_calls"], state_hash,
                                         row["gold_state_hash"], self.catalog)
                f1, _ = call_f1(calls, row["gold_tool_calls"], self.catalog)
                scores.append(score)
                f1s.append(f1)
                hashes.append(state_hash == row["gold_state_hash"])
                output.write(json.dumps({"id": item["id"], "tool_calls": calls,
                    "state_hash": state_hash, "gold_state_hash": row["gold_state_hash"],
                    "score": score, "tool_call_f1": f1, "agent_stats": stats}, ensure_ascii=False) + "\n")
                output.flush()
                logging.info("Validation rollout %d/%d %s: score %.4f, hash_match=%s",
                             len(scores), len(self.rows), item["id"], score, hashes[-1])
        partial.replace(output_path)
        n = len(scores)
        return {"eval_state_tool_score": 100 * sum(scores) / n,
                "eval_tool_call_f1": 100 * sum(f1s) / n,
                "eval_state_match_rate": 100 * sum(hashes) / n,
                "eval_rollout_samples": n}


def score_callback(evaluator, tokenizer, output_dir, fast_language_model=None):
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
            import numpy as np
            import torch
            was_training, use_cache = model.training, model.config.use_cache
            padding_side, truncation_side = tokenizer.padding_side, tokenizer.truncation_side
            py_rng, np_rng = random.getstate(), np.random.get_state()
            device = model.get_input_embeddings().weight.device
            devices = [device.index] if device.type == "cuda" else []
            # Keep all evaluation attempts (including re-evaluation of the loaded best model).
            folder = output_dir / "validation_rollouts"
            folder.mkdir(exist_ok=True)
            import tempfile
            with tempfile.NamedTemporaryFile(prefix=f"step-{state.global_step}-", suffix=".jsonl",
                                             dir=folder, delete=False) as tmp:
                path = Path(tmp.name)
            path.unlink()  # Only a fully completed evaluation promotes a final JSONL.
            try:
                with torch.random.fork_rng(devices=devices):
                    if fast_language_model is not None:
                        fast_language_model.for_inference(model)
                    model.eval()
                    model.config.use_cache = True
                    llm = LocalLLM.from_loaded_model(model, tokenizer)
                    result = evaluator.evaluate(llm, path)
            finally:
                if fast_language_model is not None:
                    fast_language_model.for_training(model)
                model.train(was_training)
                model.config.use_cache = use_cache
                tokenizer.padding_side, tokenizer.truncation_side = padding_side, truncation_side
                random.setstate(py_rng)
                np.random.set_state(np_rng)
            metrics.update(result)
            if state.log_history and state.log_history[-1].get("step") == state.global_step:
                state.log_history[-1].update(result)
            with (output_dir / "validation_scores.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                    "rollouts": str(path), **result}) + "\n")
            logging.info("Validation state/tool score: %.4f/100", result["eval_state_tool_score"])
            return control

    return ScoreCallback()
