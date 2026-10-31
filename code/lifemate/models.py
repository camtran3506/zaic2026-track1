"""Two completely local, deterministic model wrappers."""
from __future__ import annotations

import logging
from pathlib import Path

from .context import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS, validate_inference_budget
from .asr import LocalASR


class LocalLLM:
    def __init__(self, model_path: Path, device: str = "cuda", dtype: str = "float16",
                 max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 adapter_path: Path | None = None, device_map: str = "single",
                 max_memory_gib: float = 6):
        validate_inference_budget(max_input_tokens, max_new_tokens)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not model_path.is_dir():
            raise FileNotFoundError(f"LLM model directory not found: {model_path}")
        self.torch = torch
        tokenizer_path = adapter_path if adapter_path and (adapter_path / "tokenizer_config.json").is_file() else model_path
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        self.tokenizer.truncation_side = "right"
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
        self.model = AutoModelForCausalLM.from_pretrained(
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
        obj.device = model.get_input_embeddings().weight.device
        obj.max_input_tokens, obj.max_new_tokens = max_input_tokens, max_new_tokens
        return obj

    def generate(self, system: str, user: str) -> str:
        # Import lazily: Unsloth must be imported before Transformers during training.
        from transformers import StoppingCriteriaList
        from .decoding import DecisionJsonStoppingCriteria
        tokenizer = self.tokenizer
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt", truncation=False, return_token_type_ids=False)
        if encoded["input_ids"].shape[1] > self.max_input_tokens:
            raise ValueError(f"Prompt exceeds --max-input-tokens={self.max_input_tokens}; "
                             "increase the input budget instead of silently truncating tool schemas")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        stop = DecisionJsonStoppingCriteria(tokenizer, encoded["input_ids"].shape[1])
        with self.torch.inference_mode():
            ids = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([stop]),
            )
        if stop.completed_json is not None:
            return stop.completed_json
        completion = ids[0, encoded["input_ids"].shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=True).strip()
