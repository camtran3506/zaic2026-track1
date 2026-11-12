"""Two completely local, deterministic model wrappers."""
from __future__ import annotations

import logging
from pathlib import Path

from .context import (DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS,
                      adaptive_thinking_budget, validate_inference_budget)
from .asr import LocalASR
from .gpu import model_device_map
from .llm import model_loader, configure_tokenizer, render_prompt
from .prefix_cache import PrefixCache, stable_prefix_ids, committed_prefix_ids
from .schema_cache import SchemaCache, DEFAULT_SCHEMA_CACHE_MB
from .prompt_session import PreparedPrompt
from .runtime import (DEFAULT_LLM_DTYPE, DEFAULT_SEED, generation_policy,
                      load_runtime_manifest, select_tokenizer_path, generation_context)


class LocalLLM:
    def __init__(self, model_path: Path, device: str = "cuda", dtype: str = DEFAULT_LLM_DTYPE,
                 max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 adapter_path: Path | None = None, device_map: str = "single",
                 seed: int = DEFAULT_SEED,
                 prefix_cache: str = "auto", prompt_cache: str = "auto",
                 schema_cache: str = "auto", schema_cache_mb: float = DEFAULT_SCHEMA_CACHE_MB,
                 schema_cache_eager: bool = False, thinking_budget: int = 0):
        validate_inference_budget(max_input_tokens, max_new_tokens)
        if thinking_budget < 0:
            raise ValueError("thinking_budget must be non-negative")
        if device_map != 'single':
            raise ValueError('Only explicit single-device placement is supported; auto/balanced offload is disabled')
        if prompt_cache not in {"auto", "off"}:
            raise ValueError("prompt_cache must be auto or off")
        self.prompt_cache_mode = prompt_cache
        self.thinking_budget = int(thinking_budget)
        self.prefix_cache = PrefixCache(prefix_cache)
        self.schema_cache = SchemaCache(schema_cache, schema_cache_mb, eager=schema_cache_eager)
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
                                      max_new_tokens=max_new_tokens, thinking_budget=self.thinking_budget)
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
                          max_new_tokens=DEFAULT_MAX_NEW_TOKENS, *, prefix_cache="off", prompt_cache="off",
                          schema_cache="off", schema_cache_mb=DEFAULT_SCHEMA_CACHE_MB,
                          schema_cache_eager: bool = False, thinking_budget: int = 0):
        """Run identical agent decoding on current training weights without a second model."""
        validate_inference_budget(max_input_tokens, max_new_tokens)
        if thinking_budget < 0:
            raise ValueError("thinking_budget must be non-negative")
        if prompt_cache not in {"auto", "off"}:
            raise ValueError("prompt_cache must be auto or off")
        import torch
        obj = cls.__new__(cls)
        obj.torch, obj.model, obj.tokenizer = torch, model, tokenizer
        obj.prompt_cache_mode = prompt_cache
        obj.thinking_budget = int(thinking_budget)
        # Training callbacks may patch forwards/change weights. Leave their
        # existing rollout path intact unless explicitly requested by a caller.
        obj.prefix_cache = PrefixCache(prefix_cache)
        obj.schema_cache = SchemaCache(schema_cache, schema_cache_mb, eager=schema_cache_eager)
        obj.generation_policy = dict(getattr(model, "lifemate_generation_policy", None)
                                     or generation_policy(tokenizer))
        obj.generation_policy.update(max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
                                     thinking_budget=obj.thinking_budget)
        obj.device = model.get_input_embeddings().weight.device
        obj.max_input_tokens, obj.max_new_tokens = max_input_tokens, max_new_tokens
        return obj

    def begin_sample(self):
        self.prefix_cache.begin_sample()
        self.schema_cache.begin_sample()

    def end_sample(self):
        self.prefix_cache.clear()

    def clear_schema_cache(self):
        self.end_sample()
        self.schema_cache.clear()

    def close(self):
        self.clear_schema_cache()

    def schema_cache_stats(self):
        return {**self.schema_cache.report(), 'disabled_reason':
                'prefix-cache off' if self.prefix_cache.mode == 'off' else self.prefix_cache.disabled_reason}

    def prefix_cache_stats(self):
        return self.prefix_cache.report()

    def _render_tokens(self, system: str, user: str, *, enable_thinking: bool | None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prepared = (
            user.session.render(self.tokenizer, system, user, enable_thinking=enable_thinking)
            if self.prompt_cache_mode == "auto" and isinstance(user, PreparedPrompt)
            else None
        )
        if prepared is None:
            _, token_ids = render_prompt(self.tokenizer, messages, enable_thinking=enable_thinking)
            prepared_prefix = None
        else:
            _, token_ids, prepared_prefix = prepared
        return messages, prepared, token_ids, prepared_prefix

    def generate(self, system: str, user: str, *, thinking_budget: int | None = None) -> str:
        # Import lazily: Unsloth must be imported before Transformers during training.
        from transformers import LogitsProcessorList, StoppingCriteriaList
        from .decoding import (DecisionJsonStoppingCriteria, NativeToolStoppingCriteria,
                               ThinkingBudgetLogitsProcessor)
        if thinking_budget is not None and int(thinking_budget) < 0:
            raise ValueError("thinking_budget must be non-negative")
        tokenizer = self.tokenizer
        native = getattr(tokenizer, "lifemate_native_fc", False) is True
        requested_budget = self.thinking_budget if thinking_budget is None else int(thinking_budget)
        initial_enable_thinking = (requested_budget > 0) if native else None

        messages, prepared, token_ids, prepared_prefix = self._render_tokens(
            system, user, enable_thinking=initial_enable_thinking
        )
        if len(token_ids) > self.max_input_tokens:
            raise ValueError(f"Prompt has {len(token_ids)} tokens, exceeds "
                             f"--max-input-tokens={self.max_input_tokens}; "
                             "full schemas, history and tool results are never truncated")

        effective_max_new_tokens, effective_thinking_budget = adaptive_thinking_budget(
            len(token_ids),
            self.max_new_tokens,
            requested_budget if native else 0,
        )
        if native and requested_budget > 0 and effective_thinking_budget == 0:
            # Close `<think>` in the prompt so zero output tokens are spent on thinking.
            messages, prepared, token_ids, prepared_prefix = self._render_tokens(
                system, user, enable_thinking=False
            )
            if len(token_ids) > self.max_input_tokens:
                raise ValueError(f"Prompt has {len(token_ids)} tokens, exceeds "
                                 f"--max-input-tokens={self.max_input_tokens}; "
                                 "full schemas, history and tool results are never truncated")
            effective_max_new_tokens, _ = adaptive_thinking_budget(
                len(token_ids), self.max_new_tokens, 0
            )

        input_ids = self.torch.tensor([token_ids], dtype=self.torch.long, device=self.device)
        encoded = {"input_ids": input_ids, "attention_mask": self.torch.ones_like(input_ids)}
        stopping_class = NativeToolStoppingCriteria if native else DecisionJsonStoppingCriteria
        thinking_opened = effective_thinking_budget > 0

        def make_stopping_and_processors(prompt_len: int):
            criteria = StoppingCriteriaList(
                [stopping_class(tokenizer, prompt_len, thinking_opened=thinking_opened)]
            )
            if not thinking_opened:
                return criteria, None
            processors = LogitsProcessorList(
                [ThinkingBudgetLogitsProcessor(tokenizer, prompt_len, max_thinking_tokens=effective_thinking_budget)]
            )
            return criteria, processors

        stopping_criteria, processors = make_stopping_and_processors(encoded["input_ids"].shape[1])
        with self.torch.inference_mode(), generation_context(
                self.model, self.generation_policy, effective_max_new_tokens) as config:
            ids = None
            if self.prefix_cache.mode != 'off' and not self.prefix_cache.disabled_reason:
                try:
                    if self.schema_cache.bind(self.model):
                        self.prefix_cache.clear()
                        self.prefix_cache._layers = None
                except Exception as exc:
                    self.schema_cache.clear()
                    self.prefix_cache.disable(f'Cannot verify model identity: {exc}')
            if self.prefix_cache.eligible(self.model):
                try:
                    prefix = prepared_prefix or stable_prefix_ids(
                        tokenizer, messages, token_ids, memo=self.prefix_cache.prefix_memo)
                    if prefix:
                        commit_ids = ()
                        if self.prefix_cache.snapshot is not None:
                            commit_ids = committed_prefix_ids(tokenizer, messages, token_ids)
                        schema_prefix = ()
                        if self.prefix_cache.snapshot is None or self.prefix_cache.tokens != tuple(prefix):
                            head = user.session.schema_head(tokenizer, system) if isinstance(user, PreparedPrompt) and prepared is not None else None
                            schema_prefix = self.schema_cache.prefix_ids(tokenizer, messages, token_ids, head=head)
                        with self.prefix_cache.restore(self.model, input_ids, prefix,
                                schema_cache=self.schema_cache, schema_prefix=schema_prefix,
                                commit_ids=commit_ids) as cached:
                            gen_kwargs = {"generation_config": config, "stopping_criteria": stopping_criteria}
                            if processors is not None:
                                gen_kwargs["logits_processor"] = processors
                            ids = self.model.generate(**encoded, **cached, **gen_kwargs)
                    else:
                        self.prefix_cache.clear()
                except Exception as exc:
                    # Nothing has reached ToolEnv. Retry this generation once
                    # with the original full prompt and unpatched HF forwards.
                    self.prefix_cache.stats["fallbacks"] += 1
                    self.prefix_cache.disable(f"{type(exc).__name__}: {exc}")
                    self.schema_cache.clear()
                    stopping_criteria, processors = make_stopping_and_processors(input_ids.shape[1])
                    cached = None
                    exc.__traceback__ = None  # Release failed-cache frames before a GPU retry.
                    if isinstance(exc, self.torch.OutOfMemoryError) and self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
            if ids is None:
                gen_kwargs = {"generation_config": config, "stopping_criteria": stopping_criteria}
                if processors is not None:
                    gen_kwargs["logits_processor"] = processors
                ids = self.model.generate(**encoded, **gen_kwargs)
        stop = stopping_criteria[0]
        completed = stop.completed_output if native else stop.completed_json
        if completed is not None:
            if thinking_opened and native and ids is not None:
                raw_text = tokenizer.decode(ids[0, encoded["input_ids"].shape[1]:], skip_special_tokens=False)
                end_idx = raw_text.find("</think>")
                if end_idx >= 0:
                    reasoning = raw_text[:end_idx].strip()
                    if reasoning:
                        return f"<think>\n{reasoning}\n</think>\n\n{completed}"
            return completed
        completion = ids[0, encoded["input_ids"].shape[1]:]
        decoded = tokenizer.decode(completion, skip_special_tokens=not native).strip()
        if thinking_opened and native and not decoded.lstrip().startswith("<think>"):
            return f"<think>\n{decoded}" if decoded else decoded
        return decoded
