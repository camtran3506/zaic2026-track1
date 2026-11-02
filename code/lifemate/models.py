"""Two completely local, deterministic model wrappers."""
from __future__ import annotations

import logging
from pathlib import Path

from .context import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS, validate_inference_budget
from .asr import LocalASR
from .llm import model_loader, configure_tokenizer, render_prompt
from .runtime import (DEFAULT_LLM_DTYPE, DEFAULT_SEED, generation_policy,
                      load_runtime_manifest, select_tokenizer_path, generation_context)


class LocalLLM:
    def __init__(self, model_path: Path, device: str = "cuda", dtype: str = DEFAULT_LLM_DTYPE,
                 max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 adapter_path: Path | None = None, device_map: str = "single",
                 max_memory_gib: float = 6, seed: int = DEFAULT_SEED):
        validate_inference_budget(max_input_tokens, max_new_tokens)
        import torch
        from transformers import AutoTokenizer
        if not model_path.is_dir():
            raise FileNotFoundError(f"LLM model directory not found: {model_path}")
        self.torch = torch
        tokenizer_path = select_tokenizer_path(model_path, adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        configure_tokenizer(self.tokenizer, model_path)
        self.runtime_manifest = load_runtime_manifest(model_path, self.tokenizer, adapter_path)
        self.generation_policy = (dict(self.runtime_manifest["generation"]) if self.runtime_manifest
                                  else generation_policy(self.tokenizer, model_path, seed))
        self.generation_policy.update(seed=seed, max_input_tokens=max_input_tokens,
                                      max_new_tokens=max_new_tokens)
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        sharding = {}
        if device_map == "balanced":
            if device != "cuda" or torch.cuda.device_count() < 2:
                raise RuntimeError("balanced inference requires at least two visible CUDA GPUs")
            if max_memory_gib <= 0:
                raise ValueError("--llm-max-memory-gib must be positive")
            budgets = {}
            for gpu in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(gpu)
                budgets[gpu] = min(int(max_memory_gib * 1024**3), free - 4 * 1024**3)
                if budgets[gpu] <= 0:
                    raise RuntimeError(f"Insufficient free VRAM on GPU {gpu}")
                logging.info("LLM GPU %d %s: free %.2f/%.2f GiB; weight budget %.2f GiB",
                    gpu, torch.cuda.get_device_name(gpu), free/1024**3, total/1024**3, budgets[gpu]/1024**3)
            sharding = {"device_map": "balanced", "max_memory": budgets}
        elif device_map != "single":
            raise ValueError(f"Unknown LLM device map: {device_map}")
        self.model = model_loader(model_path).from_pretrained(
            str(model_path), torch_dtype=torch_dtype,
            local_files_only=True, trust_remote_code=False,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            **sharding,
        )
        if sharding:
            placements = getattr(self.model, "hf_device_map", {})
            gpu_ids = {str(v).removeprefix("cuda:") for v in placements.values()}
            if not gpu_ids <= {str(i) for i in range(torch.cuda.device_count())} or len(gpu_ids) < 2:
                raise RuntimeError(f"Expected LLM weights on multiple GPUs without CPU/disk offload: {placements}")
            logging.info("LLM device map: %s", placements)
        if adapter_path is not None:
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path),
                                                 local_files_only=True, is_trainable=False,
                                                 autocast_adapter_dtype=False)
        if not sharding:
            self.model.to(device)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device if sharding else device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_loaded_model(cls, model, tokenizer, max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
                          max_new_tokens=DEFAULT_MAX_NEW_TOKENS):
        """Run identical agent decoding on current training weights without a second model."""
        validate_inference_budget(max_input_tokens, max_new_tokens)
        import torch
        obj = cls.__new__(cls)
        obj.torch, obj.model, obj.tokenizer = torch, model, tokenizer
        obj.generation_policy = dict(getattr(model, "lifemate_generation_policy", None)
                                     or generation_policy(tokenizer))
        obj.generation_policy.update(max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens)
        obj.device = model.get_input_embeddings().weight.device
        obj.max_input_tokens, obj.max_new_tokens = max_input_tokens, max_new_tokens
        return obj

    def generate(self, system: str, user: str) -> str:
        # Import lazily: Unsloth must be imported before Transformers during training.
        from transformers import StoppingCriteriaList
        from .decoding import DecisionJsonStoppingCriteria, NativeToolStoppingCriteria
        tokenizer = self.tokenizer
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        _, token_ids = render_prompt(tokenizer, messages)
        if len(token_ids) > self.max_input_tokens:
            raise ValueError(f"Prompt has {len(token_ids)} tokens, exceeds "
                             f"--max-input-tokens={self.max_input_tokens}; "
                             "full schemas, history and tool results are never truncated")
        input_ids = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
        encoded = {"input_ids": input_ids, "attention_mask": self.torch.ones_like(input_ids)}
        native = getattr(tokenizer, "lifemate_native_fc", False) is True
        stopping_class = NativeToolStoppingCriteria if native else DecisionJsonStoppingCriteria
        stop = stopping_class(tokenizer, encoded["input_ids"].shape[1])
        with self.torch.inference_mode(), generation_context(
                self.model, self.generation_policy, self.max_new_tokens) as config:
            ids = self.model.generate(
                **encoded, generation_config=config,
                stopping_criteria=StoppingCriteriaList([stop]),
            )
        completed = stop.completed_output if native else stop.completed_json
        if completed is not None:
            return completed
        completion = ids[0, encoded["input_ids"].shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=not native).strip()
