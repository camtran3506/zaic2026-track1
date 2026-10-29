#!/usr/bin/env python3
"""Assistant-only Unsloth LoRA/QLoRA; optional Transformers CPU test backend."""
import argparse
import json
import logging
import os
from pathlib import Path

from lifemate.sft import AssistantCollator, SupervisedDataset, file_sha256
from lifemate.gpu import add_cuda_device_argument, configure_cuda_device
from lifemate.context import MAX_CONTEXT_TOKENS, validate_training_length


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_cuda_device_argument(p)
    p.add_argument("--base-model", type=Path, required=True, help="Local model directory (stage HF weights first)")
    p.add_argument("--sft-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", choices=["unsloth", "transformers"], default="unsloth")
    p.add_argument("--quantization", choices=["4bit", "none"], default="none")
    p.add_argument("--max-length", type=int, default=MAX_CONTEXT_TOKENS)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0, help="0 enables Unsloth optimized LoRA kernels")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-strategy", choices=["epoch", "steps"], default="epoch",
                   help="Run full held-out agent rollouts and save candidates each epoch, or every --save-steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from-checkpoint", type=Path)
    args = p.parse_args()
    validate_training_length(args.max_length)
    configure_cuda_device(args)
    # Unsloth must patch Transformers/PEFT BEFORE either is imported.
    FastLanguageModel = None
    if args.backend == "unsloth":
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt in a supported CUDA environment for Unsloth") from exc
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
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
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    if args.backend == "unsloth":
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(args.base_model.resolve()), max_seq_length=args.max_length,
            dtype=dtype, load_in_4bit=args.quantization == "4bit",
            local_files_only=True, trust_remote_code=False,
            use_exact_model_name=True, fast_inference=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs an existing pad/eos token")
    manifest = json.loads((args.sft_dir / "manifest.json").read_text())
    data = {}
    ids = {}
    for part in ("train", "validation"):
        path = args.sft_dir / f"{part}.jsonl"
        if file_sha256(path) != manifest["partitions"][part]["sha256"]:
            raise ValueError(f"Prepared {part} data changed after preparation")
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids[part] = {e["sample_id"] for e in examples}
        if ids[part] != set(manifest["partitions"][part]["sample_ids"]):
            raise ValueError(f"{part} sample IDs disagree with manifest")
        data[part] = SupervisedDataset(examples, tokenizer, args.max_length)
    if ids["train"] & ids["validation"]:
        raise ValueError("Train/validation sample leakage")
    from lifemate.evaluation import StateToolEvaluator, score_callback
    evaluator = StateToolEvaluator(args.sft_dir, manifest)
    run_config = {
        **{k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "train_steps": len(data["train"]),
        "validation_steps": len(data["validation"]), "data_manifest": manifest,
        "selection_metric": "eval_state_tool_score",
    }
    if args.resume_from_checkpoint:
        original_path = args.resume_from_checkpoint.parent / "training_config.json"
        if not original_path.is_file():
            raise FileNotFoundError("Resume needs the original training_config.json next to checkpoint directories")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        for key in ("backend", "base_model", "data_manifest", "quantization", "max_length",
                    "batch_size", "gradient_accumulation", "learning_rate", "lora_r",
                    "lora_alpha", "lora_dropout", "seed", "eval_strategy"):
            if original.get(key) != run_config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
    # Save before training so interrupted runs can still be resumed safely.
    config_path = args.output_dir / "training_config.json"
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.backend == "unsloth":
        model = FastLanguageModel.get_peft_model(model, r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", target_modules=targets,
            use_gradient_checkpointing="unsloth", random_state=args.seed,
            use_rslora=False, loftq_config=None)
    else:
        kwargs = dict(local_files_only=True, trust_remote_code=False, torch_dtype=dtype,
                      attn_implementation="sdpa")
        if args.quantization == "4bit":
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype), device_map={"": int(os.environ.get("LOCAL_RANK", 0))})
        model = AutoModelForCausalLM.from_pretrained(str(args.base_model), **kwargs)
        if args.quantization == "4bit":
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=targets)
        model = get_peft_model(model, config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(args.output_dir),
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, max_grad_norm=1.0, optim="adamw_torch",
        bf16=bf16, fp16=cuda and not bf16, use_cpu=not cuda,
        gradient_checkpointing=args.backend != "unsloth", gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, eval_strategy=args.eval_strategy, eval_steps=args.save_steps,
        save_strategy=args.eval_strategy, save_steps=args.save_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="state_tool_score", greater_is_better=True,
        prediction_loss_only=True, report_to="none", seed=args.seed, data_seed=args.seed,
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(model=model, args=training_args, processing_class=tokenizer,
        train_dataset=data["train"], eval_dataset=data["validation"],
        data_collator=AssistantCollator(tokenizer.pad_token_id),
        callbacks=[score_callback(evaluator, tokenizer, args.output_dir, FastLanguageModel)])
    try:
        train_result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        trainer.save_metrics("train", train_result.metrics)
        metrics = trainer.evaluate()
        trainer.save_metrics("eval", metrics)
        trainer.save_state()
        adapter = args.output_dir / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(str(adapter))
    finally:
        evaluator.close()
    run_config["best_checkpoint"] = trainer.state.best_model_checkpoint
    run_config["best_state_tool_score"] = trainer.state.best_metric
    run_config["selection_metric"] = "eval_state_tool_score"
    if not trainer.state.best_metric:
        logging.warning("Best validation state/tool score is 0; the exported checkpoint has not demonstrated successful tool rollouts")
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved adapter and tokenizer: {adapter}; validation metrics: {metrics}")


if __name__ == "__main__":
    main()
