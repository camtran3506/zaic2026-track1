#!/usr/bin/env python3
"""Assistant-only Unsloth LoRA/QLoRA; optional Transformers CPU test backend."""
import argparse
import json
import logging
import math
import os
from pathlib import Path

from lifemate.sft import AssistantCollator, SupervisedDataset, file_sha256, validate_prompt_policy
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device, model_device_map
from lifemate.context import MAX_CONTEXT_TOKENS, validate_training_length
from lifemate.llm import is_qwen35, model_loader, lora_targets, configure_tokenizer
from lifemate.reasoning import DEFAULT_REASONING_WEIGHT
from lifemate.runtime import (make_runtime_manifest, runtime_save_callback,
                              save_runtime_manifest, training_dtype)
from lifemate.numerics import NUMERICS_POLICY, guarded_trainer_class


def _output_head(model):
    head = model.get_output_embeddings() if callable(getattr(model, "get_output_embeddings", None)) else None
    if head is not None:
        return head
    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    head = base.get_output_embeddings() if callable(getattr(base, "get_output_embeddings", None)) else None
    return head if head is not None else getattr(base, "lm_head", None)


def make_weighted_trainer_class(base_trainer, backend):
    import torch
    import torch.nn.functional as F

    class WeightedLossTrainer(base_trainer):
        """Compute token-weighted cross entropy on supervised positions only."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels = inputs["labels"]
            loss_weights = inputs.get("loss_weights")
            if labels.shape[0] == 1 and (labels[0] != -100).any():
                tgt_pos = (labels[0] != -100).nonzero(as_tuple=False)[:, 0]
                tgt_pos = tgt_pos[tgt_pos > 0]
                if len(tgt_pos) > 0:
                    pos = tgt_pos - 1
                    forward_inputs = {
                        k: v for k, v in inputs.items()
                        if k not in {"labels", "loss_weights"}
                    }
                    if backend == "unsloth":
                        prev_env = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
                        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
                        try:
                            out = model(**forward_inputs, logits_to_keep=pos)
                        finally:
                            if prev_env is None:
                                os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
                            else:
                                os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prev_env
                        head = _output_head(model)
                        if head is None:
                            raise RuntimeError("Model output head (lm_head) not found for sparse loss computation")
                        h = out.logits[0]
                        if h.shape[0] != pos.shape[0]:
                            raise RuntimeError(
                                f"Expected {pos.shape[0]} hidden states with logits_to_keep, got {tuple(h.shape)}"
                            )
                        logits = head(h.to(head.weight.dtype)).float()
                    else:
                        try:
                            out = model(**forward_inputs, logits_to_keep=pos)
                            logits = out.logits[0]
                            if logits.shape[0] != pos.shape[0]:
                                logits = logits[pos]
                        except TypeError:
                            out = model(**forward_inputs)
                            logits = out.logits[0, pos]
                        logits = logits.float()
                    ce = F.cross_entropy(logits, labels[0, tgt_pos], reduction="none")
                    weights = (
                        loss_weights[0, tgt_pos].float()
                        if loss_weights is not None
                        else torch.ones_like(ce)
                    )
                    if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", True):
                        denom = (
                            num_items_in_batch.to(ce.device)
                            if isinstance(num_items_in_batch, torch.Tensor)
                            else torch.tensor(float(num_items_in_batch), device=ce.device)
                        )
                    else:
                        denom = torch.tensor(float(len(tgt_pos)), device=ce.device)
                    loss = (ce * weights).sum() / denom.clamp(min=1.0)
                    return (loss, out) if return_outputs else loss

            forward_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "loss_weights"}}
            out = model(**forward_inputs)
            shift_logits = out.logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            mask = shift_labels.ne(-100)
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            weights = loss_weights[:, 1:].contiguous().float() if loss_weights is not None else mask.float()
            if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", True):
                denom = (
                    num_items_in_batch.to(ce.device)
                    if isinstance(num_items_in_batch, torch.Tensor)
                    else torch.tensor(float(num_items_in_batch), device=ce.device)
                )
            else:
                denom = mask.sum().float().clamp(min=1.0)
            loss = (ce * weights).sum() / denom
            return (loss, out) if return_outputs else loss

    return WeightedLossTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(p)
    p.add_argument("--base-model", type=Path, required=True, help="Local model directory (stage HF weights first)")
    p.add_argument("--sft-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", choices=["unsloth", "transformers"], default="unsloth")
    p.add_argument("--quantization", choices=["4bit", "none"], default="none")
    p.add_argument("--max-length", type=int, default=MAX_CONTEXT_TOKENS)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--fuzzy-weight", type=float, default=0.3,
                   help="Loss weight on x-match=fuzzy parameter value spans (0..1; 1.0 disables downweighting)")
    p.add_argument("--reasoning-weight", type=float, default=DEFAULT_REASONING_WEIGHT,
                   help="Loss weight on short checked reasoning tokens inside <think>...</think> (0..1; 1.0 disables downweighting)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0, help="0 enables Unsloth optimized LoRA kernels")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-strategy", choices=["epoch", "steps"], default="epoch",
                   help="Save checkpoints each epoch, or every --save-steps; validation is disabled")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", type=Path)
    args = p.parse_args()
    if not math.isfinite(args.fuzzy_weight) or not 0.0 <= args.fuzzy_weight <= 1.0:
        p.error("--fuzzy-weight must be between 0.0 and 1.0")
    if not math.isfinite(args.reasoning_weight) or not 0.0 <= args.reasoning_weight <= 1.0:
        p.error("--reasoning-weight must be between 0.0 and 1.0")
    validate_training_length(args.max_length)
    configure_cuda_device(args)
    # Unsloth must patch Transformers/PEFT BEFORE either is imported.
    FastLanguageModel = None
    if args.backend == "unsloth":
        try:
            from unsloth import FastLanguageModel, FastVisionModel
            if is_qwen35(args.base_model):
                FastLanguageModel = FastVisionModel
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt in a supported CUDA environment for Unsloth") from exc
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoTokenizer, BitsAndBytesConfig,
                              Trainer, TrainingArguments, set_seed)
    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_checkpoint:
        raise FileExistsError("Use an empty training output directory or --resume-from-checkpoint")
    if args.resume_from_checkpoint and not (args.resume_from_checkpoint / "trainer_state.json").is_file():
        raise FileNotFoundError("Resume requires a Trainer checkpoint including optimizer/scheduler state")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(),
        logging.FileHandler(args.output_dir / "training.log", encoding="utf-8")])
    set_seed(args.seed)
    cuda = torch.cuda.is_available()
    if args.backend == "unsloth" and not cuda:
        raise RuntimeError("Unsloth backend requires CUDA; CPU tests must explicitly use --backend transformers --quantization none")
    if args.quantization == "4bit" and not cuda:
        raise RuntimeError("QLoRA requires CUDA; use --quantization none for a CPU smoke test")
    dtype = training_dtype(torch)
    bf16 = cuda and dtype == torch.bfloat16
    placement = model_device_map('cuda' if cuda else 'cpu')
    logging.info('Explicit model device map (no CPU/disk fallback): %s', placement)
    if args.backend == "unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.base_model.resolve()), max_seq_length=args.max_length,
            dtype=dtype, load_in_4bit=args.quantization == "4bit",
            local_files_only=True, trust_remote_code=False,
            fast_inference=False,
            device_map=placement,
            **({"auto_model": model_loader(args.base_model)} if is_qwen35(args.base_model)
               else {"use_exact_model_name": True}),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=False)
    if is_qwen35(args.base_model):
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=False)
    configure_tokenizer(tokenizer, args.base_model)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs an existing pad/eos token")
    manifest = json.loads((args.sft_dir / "manifest.json").read_text())
    validate_prompt_policy(manifest)
    if any(info.get("sample_ids") for name, info in manifest["partitions"].items() if name != "train"):
        raise ValueError("Training requires all samples in train; rerun prepare_sft.py without validation/app holdouts")
    path = args.sft_dir / "train.jsonl"
    if file_sha256(path) != manifest["partitions"]["train"]["sha256"]:
        raise ValueError("Prepared train data changed after preparation")
    examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {e["sample_id"] for e in examples} != set(manifest["partitions"]["train"]["sample_ids"]):
        raise ValueError("train sample IDs disagree with manifest")
    fuzzy_arg = None if args.fuzzy_weight >= 1.0 else args.fuzzy_weight
    reasoning_arg = None if args.reasoning_weight >= 1.0 else args.reasoning_weight
    train_data = SupervisedDataset(examples, tokenizer, args.max_length,
                                   include_debug_metadata=True, source_path=path,
                                   fuzzy_weight=fuzzy_arg,
                                   reasoning_weight=reasoning_arg)
    runtime = make_runtime_manifest(args.base_model, tokenizer, seed=args.seed)
    run_config = {
        **{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "train_steps": len(train_data),
        "validation_steps": 0, "eval_strategy": "no", "data_manifest": manifest,
        "selection_metric": None, "checkpoint_selection": "last",
        "runtime": runtime,
        "training_dtype": str(dtype),
        "numerics_guard": NUMERICS_POLICY,
    }
    if args.resume_from_checkpoint:
        original_path = args.resume_from_checkpoint.parent / "training_config.json"
        if not original_path.is_file():
            raise FileNotFoundError("Resume needs the original training_config.json next to checkpoint directories")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        for key in ("backend", "base_model", "data_manifest", "quantization", "max_length",
                    "batch_size", "gradient_accumulation", "learning_rate", "lora_r",
                    "lora_alpha", "lora_dropout", "seed", "eval_strategy", "save_strategy", "runtime"):
            if original.get(key) != run_config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
    # Save before training so interrupted runs can still be resumed safely.
    config_path = args.output_dir / "training_config.json"
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    targets = lora_targets(args.base_model)
    if args.backend == "unsloth":
        model = FastLanguageModel.get_peft_model(model, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", target_modules=targets,
            use_gradient_checkpointing="unsloth", random_state=args.seed,
            use_rslora=False, loftq_config=None,
            **({"finetune_vision_layers": False} if is_qwen35(args.base_model) else {}))
    else:
        kwargs = dict(local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
                      attn_implementation="sdpa", device_map=placement)
        if args.quantization == "4bit":
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype))
        model = model_loader(args.base_model).from_pretrained(str(args.base_model), **kwargs)
        if args.quantization == "4bit":
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=targets)
        model = get_peft_model(model, config)
    model.config.use_cache = False
    model.lifemate_generation_policy = runtime["generation"]
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(args.output_dir),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_steps=0.03,
        weight_decay=0.01, max_grad_norm=1.0, optim="adamw_torch",
        bf16=bf16, fp16=False, use_cpu=not cuda,
        gradient_checkpointing=args.backend != "unsloth", gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, logging_nan_inf_filter=False, eval_strategy="no",
        save_strategy=args.save_strategy, save_steps=args.save_steps, save_total_limit=2,
        load_best_model_at_end=False,
        prediction_loss_only=True, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, remove_unused_columns=False)
    WeightedTrainer = make_weighted_trainer_class(Trainer, args.backend)
    GuardedTrainer = guarded_trainer_class(WeightedTrainer)
    trainer = GuardedTrainer(model=model, args=training_args, processing_class=tokenizer,
        numerics_context={'backend': args.backend, 'sft_dir': str(args.sft_dir.resolve()),
                          'training_config': str(config_path.resolve())},
        train_dataset=train_data,
        data_collator=AssistantCollator(tokenizer.pad_token_id),
        callbacks=[runtime_save_callback(runtime)])
    train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    save_runtime_manifest(adapter, runtime)
    run_config["final_step"] = trainer.state.global_step
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved final adapter and tokenizer: {adapter}; training metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
