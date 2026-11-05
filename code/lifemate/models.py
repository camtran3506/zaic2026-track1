"""Two completely local, deterministic model wrappers."""
from __future__ import annotations

import logging
from pathlib import Path

from .context import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS, validate_inference_budget
from .asr import LocalASR
from .gpu import model_device_map
from .llm import model_loader, configure_tokenizer, render_prompt
from .prefix_cache import PrefixCache, stable_prefix_ids
from .prompt_session import PreparedPrompt
from .runtime import (DEFAULT_LLM_DTYPE, DEFAULT_SEED, generation_policy,
                      load_runtime_manifest, select_tokenizer_path, generation_context)


class LocalLLM:
    def __init__(self, model_path: Path, device: str = "cuda", dtype: str = DEFAULT_LLM_DTYPE,
                 max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 adapter_path: Path | None = None, device_map: str = "single",
                 seed: int = DEFAULT_SEED,
                 prefix_cache: str = "auto", prompt_cache: str = "auto"):
        validate_inference_budget(max_input_tokens, max_new_tokens)
        if device_map != 'single':
            raise ValueError('Only explicit single-device placement is supported; auto/balanced offload is disabled')
        if prompt_cache not in {"auto", "off"}:
            raise ValueError("prompt_cache must be auto or off")
        self.prompt_cache_mode = prompt_cache
        self.prefix_cache = PrefixCache(prefix_cache)
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
        placement = model_device_map(device)
        logging.info('Explicit model device map (no CPU/disk fallback): %s', placement)
        self.model = model_loader(model_path).from_pretrained(
            str(model_path), torch_dtype=torch_dtype,
            local_files_only=True, trust_remote_code=False,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map=placement,
        )
        if adapter_path is not None:
            if not (adapter_path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path),
                                                 local_files_only=True, is_trainable=False,
                                                 autocast_adapter_dtype=False, device_map=placement)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_loaded_model(cls, model, tokenizer, max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
                          max_new_tokens=DEFAULT_MAX_NEW_TOKENS, *, prefix_cache="off", prompt_cache="off"):
        """Run identical agent decoding on current training weights without a second model."""
        validate_inference_budget(max_input_tokens, max_new_tokens)
        if prompt_cache not in {"auto", "off"}:
            raise ValueError("prompt_cache must be auto or off")
        import torch
        obj = cls.__new__(cls)
        obj.torch, obj.model, obj.tokenizer = torch, model, tokenizer
        obj.prompt_cache_mode = prompt_cache
        # Training callbacks may patch forwards/change weights. Leave their
        # existing rollout path intact unless explicitly requested by a caller.
        obj.prefix_cache = PrefixCache(prefix_cache)
        obj.generation_policy = dict(getattr(model, "lifemate_generation_policy", None)
                                     or generation_policy(tokenizer))
        obj.generation_policy.update(max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens)
        obj.device = model.get_input_embeddings().weight.device
        obj.max_input_tokens, obj.max_new_tokens = max_input_tokens, max_new_tokens
        return obj

    def begin_sample(self):
        self.prefix_cache.begin_sample()

    def end_sample(self):
        self.prefix_cache.clear()

    def prefix_cache_stats(self):
        return self.prefix_cache.report()

    def generate(self, system: str, user: str) -> str:
        # Import lazily: Unsloth must be imported before Transformers during training.
        from transformers import StoppingCriteriaList
        from .decoding import DecisionJsonStoppingCriteria, NativeToolStoppingCriteria
        tokenizer = self.tokenizer
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prepared = (user.session.render(tokenizer, system, user)
                    if self.prompt_cache_mode == "auto" and isinstance(user, PreparedPrompt) else None)
        if prepared is None:
            _, token_ids = render_prompt(tokenizer, messages)
            prepared_prefix = None
        else:
            _, token_ids, prepared_prefix = prepared
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
            ids = None
            if self.prefix_cache.eligible(self.model):
                try:
                    prefix = prepared_prefix or stable_prefix_ids(
                        tokenizer, messages, token_ids, memo=self.prefix_cache.prefix_memo)
                    if prefix:
                        with self.prefix_cache.restore(self.model, input_ids, prefix) as cached:
                            ids = self.model.generate(
                                **encoded, **cached, generation_config=config,
                                stopping_criteria=StoppingCriteriaList([stop]))
                    else:
                        self.prefix_cache.clear()
                except Exception as exc:
                    # Nothing has reached ToolEnv. Retry this generation once
                    # with the original full prompt and unpatched HF forwards.
                    self.prefix_cache.stats["fallbacks"] += 1
                    self.prefix_cache.disable(f"{type(exc).__name__}: {exc}")
                    stop = stopping_class(tokenizer, input_ids.shape[1])
                    cached = None
                    if isinstance(exc, self.torch.OutOfMemoryError) and self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
            if ids is None:
                ids = self.model.generate(
                    **encoded, generation_config=config,
                    stopping_criteria=StoppingCriteriaList([stop]))
        completed = stop.completed_output if native else stop.completed_json
        if completed is not None:
            return completed
        completion = ids[0, encoded["input_ids"].shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=not native).strip()
