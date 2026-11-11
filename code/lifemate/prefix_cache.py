"""Per-sample, exact-token prefix snapshots for the pinned Qwen3.5 runtime.

Only text-only Transformers 5.4.0 is enabled. Other runtimes keep the existing
full-prefill path. Never crop a hybrid cache: its recurrent state cannot be
rolled back by slicing attention tensors.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import logging
import sys
from pathlib import Path
from types import MethodType


def stable_prefix_ids(tokenizer, messages, full_ids, *, memo=None):
    """Find a closed, immutable native-chat prefix, excluding step observations.

    Tokenize the candidate independently and require an exact token prefix. A
    template/BPE boundary that does not match is a miss, never a guessed offset.
    """
    if getattr(tokenizer, "lifemate_native_fc", False) is not True:
        return []
    try:
        payload = json.loads(messages[1]["content"])
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict) or "available_tools" not in payload:
        return []
    from .llm import render_prompt
    payload["previous_actions_and_results"] = []
    content = json.dumps(payload, ensure_ascii=False)
    source = (messages[0]["content"], content)
    if memo is not None and memo.get("source") == source:
        candidate = memo["ids"]
    else:
        base = [messages[0], {"role": "user", "content": content}]
        _, candidate = render_prompt(tokenizer, base, add_generation_prompt=False)
        if memo is not None:
            memo.update(source=source, ids=candidate)
    if not candidate or len(candidate) >= len(full_ids) or full_ids[:len(candidate)] != candidate:
        return []
    return candidate


def executed_observation_run(observations):
    """Leading observations that are real, executed tool results.

    Rejected proposals (`agent_feedback`) have no `result` and disappear from the
    prompt as soon as the next call executes, so they must never be committed
    into an incremental snapshot. The run stops at the first such entry because
    token order cannot skip a message.
    """
    run = []
    for observation in observations or []:
        if isinstance(observation, dict) and all(key in observation for key in ("name", "arguments", "result")):
            run.append(observation)
        else:
            break
    return run


def committed_prefix_ids(tokenizer, messages, full_ids):
    """Exact tokens of the closed prompt through the last leading executed observation.

    Unlike `stable_prefix_ids` this includes executed tool calls/results (but not
    the generation prompt), so the next turn can continue from this boundary
    instead of re-prefilling every earlier observation. The candidate is rendered
    independently and must be an exact token prefix of `full_ids`; a template
    boundary that does not match is a miss, never a guessed offset. The caller
    additionally revalidates the prefix against the NEXT prompt before reuse,
    because Qwen's template re-renders earlier assistant turns when a later user
    message arrives.
    """
    if getattr(tokenizer, "lifemate_native_fc", False) is not True:
        return []
    try:
        payload = json.loads(messages[1]["content"])
    except (ValueError, TypeError):
        return []
    if not isinstance(payload, dict) or "available_tools" not in payload:
        return []
    run = executed_observation_run(payload.get("previous_actions_and_results"))
    if not run:
        return []
    from .llm import render_prompt
    payload["previous_actions_and_results"] = run
    content = json.dumps(payload, ensure_ascii=False)
    _, candidate = render_prompt(tokenizer, [messages[0], {"role": "user", "content": content}],
                                 add_generation_prompt=False)
    if not candidate or len(candidate) >= len(full_ids) or full_ids[:len(candidate)] != candidate:
        return []
    return candidate


def _continue_qwen54_chunk(layer, hidden_states, cache):
    """Extend BOTH linear-attention states for a multi-token cached suffix.

    HF 5.4.0's Qwen3_5GatedDeltaNet.forward only continues an existing state for
    seq_len == 1. This mirrors its projections/gating, but carries convolution
    history and passes the previous recurrent state into the existing chunk
    kernel. Single-token decoding and fresh prefill use HF's original forward.
    """
    import torch
    import torch.nn.functional as F

    batch, length, _ = hidden_states.shape
    index = layer.layer_idx
    mixed = layer.in_proj_qkv(hidden_states).transpose(1, 2)
    convolution_input = torch.cat((cache.conv_states[index], mixed), dim=-1)
    cache.conv_states[index] = convolution_input[..., -layer.conv_kernel_size:].clone()
    if layer.causal_conv1d_fn is not None and convolution_input.is_cuda:
        mixed = layer.causal_conv1d_fn(
            x=convolution_input, weight=layer.conv1d.weight.squeeze(1),
            bias=layer.conv1d.bias, activation=layer.activation, seq_idx=None)
    else:
        mixed = F.silu(F.conv1d(convolution_input, layer.conv1d.weight,
                              layer.conv1d.bias, groups=layer.conv1d.groups))
    mixed = mixed[..., -length:].transpose(1, 2)
    query, key, value = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, layer.head_k_dim)
    key = key.reshape(batch, length, -1, layer.head_k_dim)
    value = value.reshape(batch, length, -1, layer.head_v_dim)
    ratio = layer.num_v_heads // layer.num_k_heads
    if ratio > 1:
        query = query.repeat_interleave(ratio, dim=2)
        key = key.repeat_interleave(ratio, dim=2)
    z = layer.in_proj_z(hidden_states).reshape(batch, length, -1, layer.head_v_dim)
    beta = layer.in_proj_b(hidden_states).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(layer.in_proj_a(hidden_states).float() + layer.dt_bias)
    output, state = layer.chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=cache.recurrent_states[index], output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    cache.recurrent_states[index] = state
    output = layer.norm(output.reshape(-1, layer.head_v_dim), z.reshape(-1, layer.head_v_dim))
    return layer.out_proj(output.reshape(batch, length, -1))


@contextmanager
def qwen54_continuation(layers):
    """Instance-local, temporary compatibility shim; never patch HF globally."""
    previous = []
    try:
        for layer in layers:
            original = layer.forward
            previous.append((layer, "forward" in vars(layer), original))

            def forward(self, hidden_states, cache_params=None, attention_mask=None, _original=original):
                if (cache_params is not None and cache_params.has_previous_state
                        and hidden_states.shape[1] > 1):
                    return _continue_qwen54_chunk(self, hidden_states, cache_params)
                return _original(hidden_states, cache_params=cache_params, attention_mask=attention_mask)

            layer.forward = MethodType(forward, layer)
        yield
    finally:
        for layer, owned, original in reversed(previous):
            if owned:
                layer.forward = original
            else:
                del layer.forward


class PrefixCache:
    """One immutable GPU snapshot; clone before use and release between samples.

    `snapshot` is the sample-static prefix (system/tools/history/request) and is
    never written after it is built. `chain` is an additional, privately owned
    snapshot that advances in place through the leading executed observations of
    consecutive turns, so turn k only prefills the observations added since turn
    k-1 instead of every observation again. Hybrid recurrent state cannot be
    cropped: the chain is reused only when its tokens are an exact prefix of the
    new prompt and is otherwise rebuilt from the static snapshot.
    """
    def __init__(self, mode="auto"):
        if mode not in {"auto", "off"}:
            raise ValueError("prefix_cache must be auto or off")
        self.mode = mode
        self.disabled_reason = None
        self._layers = None
        self.begin_sample()

    def clear(self):
        self.tokens = ()
        self.snapshot = None
        self.chain = None
        self.chain_tokens = ()
        self.prefix_memo = {}

    def begin_sample(self):
        self.clear()
        self.stats = {"mode": self.mode, "hits": 0, "misses": 0,
                      "reused_tokens": 0, "snapshot_tokens": 0, "fallbacks": 0,
                      "chain_hits": 0, "chain_resets": 0, "chain_reused_tokens": 0,
                      "chain_advanced_tokens": 0, "commit_errors": 0}

    def eligible(self, model):
        if self.mode == "off" or self.disabled_reason:
            return False
        if getattr(model, "training", True):
            self.clear()
            return False
        if self._layers is None:
            import transformers
            if transformers.__version__ != "5.4.0":
                self.disable("only Transformers 5.4.0 Qwen3.5 continuation is verified")
                return False
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5GatedDeltaNet, Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration)
            base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
            if type(base) not in {Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM}:
                self.disable("model is not a supported Qwen3.5 model")
                return False
            text_config = getattr(base.config, 'text_config', base.config)
            rope = getattr(text_config, 'rope_parameters', None) or {}
            if rope.get('rope_type', 'default') != 'default':
                self.disable('only default, sequence-length-independent RoPE is verified')
                return False
            layers = [m for m in base.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
            if not layers or any(getattr(m.forward, "__func__", None) is not Qwen3_5GatedDeltaNet.forward for m in layers):
                self.disable("linear-attention forward is absent or patched by another runtime")
                return False
            # Unsloth may replace a class method globally (including wrapped
            # text/decoder forwards) while keeping the same class identity.
            # Check every wrapper in each Qwen forward, not just __wrapped__'s
            # final function; external patches require their own parity tests.
            trusted_root = Path(transformers.__file__).resolve().parent
            import torch
            context_decorator = Path(torch.__file__).resolve().parent / "utils/_contextlib.py"
            unsloth = sys.modules.get("unsloth")
            unsloth_root = Path(unsloth.__file__).resolve().parent if unsloth is not None and getattr(unsloth, "__file__", None) else None
            for module in base.modules():
                if not type(module).__module__.startswith("transformers.models.qwen3_5."):
                    continue
                fn = getattr(module.forward, "__func__", None)
                if fn is None:
                    self.disable("Qwen forward is patched by an unverified runtime")
                    return False
                while fn is not None:
                    code = getattr(fn, "__code__", None)
                    source = Path(code.co_filename) if code is not None else None
                    torch_context = source is not None and source.resolve() == context_decorator and hasattr(fn, "__wrapped__")
                    unsloth_ok = (unsloth_root is not None and source is not None and
                                  (str(source).startswith("<unsloth") or source.name.startswith("<unsloth") or
                                   (source.is_absolute() and source.resolve().is_relative_to(unsloth_root))) and
                                  not getattr(model, "training", True))
                    if source is None or not ((source.is_absolute() and source.resolve().is_relative_to(trusted_root)) or torch_context or unsloth_ok):
                        self.disable("Qwen forward is patched by an unverified runtime")
                        return False
                    fn = getattr(fn, "__wrapped__", None)
                if getattr(module, "_compiled_call_impl", None) is not None:
                    self.disable("compiled Qwen forward is not verified for prefix continuation")
                    return False
            self._layers = layers
        return True

    def disable(self, reason):
        self.clear()
        if self.disabled_reason is None:
            logging.warning("Prefix cache disabled; using full prefill: %s", reason)
        self.disabled_reason = reason

    def report(self):
        return {**self.stats, "disabled_reason": self.disabled_reason}

    @staticmethod
    def _extend(model, cache, input_ids, start, end):
        """Advance a privately owned hybrid cache over input_ids[:, start:end] in place."""
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
        # Explicit 4-channel text positions, as everywhere else in this module.
        positions = torch.arange(start, end, device=input_ids.device).view(1, 1, -1).expand(4, 1, -1)
        output = model(input_ids=input_ids[:, start:end], attention_mask=torch.ones_like(input_ids[:, :end]),
                       position_ids=positions, past_key_values=cache, use_cache=True,
                       logits_to_keep=1, return_dict=True)
        result = output.past_key_values
        if type(result) is not Qwen3_5DynamicCache or result.get_seq_length() != end:
            raise ValueError('Unexpected hybrid cache type or committed prefix length')
        return result

    def _committed_snapshot(self, model, input_ids, commit_ids):
        """Cache holding the longest verified committed prefix; the static snapshot if none.

        Never reads a token the NEXT prompt does not contain: the committed ids must
        extend the static prefix and be an exact prefix of `input_ids`, and the chain
        is kept only while its own tokens remain an exact prefix of the new commit.
        """
        commit = tuple(commit_ids)
        count, base = len(commit), len(self.tokens)
        if (count <= base or commit[:base] != self.tokens
                or input_ids[0, :count].tolist() != list(commit)):
            return self.snapshot
        chain = len(self.chain_tokens)
        if self.chain is not None and chain <= count and commit[:chain] == self.chain_tokens:
            self.stats["chain_hits"] += 1
            self.stats["chain_reused_tokens"] += chain
            start = chain
        else:
            self.chain, self.chain_tokens = deepcopy(self.snapshot), ()
            self.stats["chain_resets"] += 1
            start = base
        if start < count:
            try:
                self.chain = self._extend(model, self.chain, input_ids, start, count)
            except BaseException:
                # A half-advanced recurrent state is unusable; drop it and let the
                # caller's existing failure path fall back to full prefill.
                self.chain, self.chain_tokens = None, ()
                raise
            self.chain_tokens = commit
            self.stats["chain_advanced_tokens"] += count - start
        return self.chain

    @contextmanager
    def restore(self, model, input_ids, prefix_ids, *, schema_cache=None, schema_prefix=(), commit_ids=()):
        """Build/restore at an exact boundary, then shield snapshot from writes.

        `commit_ids` optionally names a longer exact prefix (through the leading
        executed observations); generation then starts from the incrementally
        advanced chain instead of the static prefix.
        """
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
        tokens = tuple(prefix_ids)
        with qwen54_continuation(self._layers):
            if self.snapshot is None or self.tokens != tokens:
                self.snapshot, self.tokens = None, ()
                self.chain, self.chain_tokens = None, ()
                self.stats["misses"] += 1
                prefix = input_ids[:, :len(tokens)]
                if tuple(input_ids[0, :len(tokens)].tolist()) != tokens:
                    raise ValueError('Prefix token mismatch')

                def prefill(ids, start=0, initial=None):
                    positions = torch.arange(start, ids.shape[1], device=ids.device).view(1, -1)
                    if initial is not None:
                        positions = positions.unsqueeze(0).expand(4, -1, -1)
                    output = model(input_ids=ids[:, start:], attention_mask=torch.ones_like(ids),
                                   position_ids=positions, past_key_values=initial,
                                   use_cache=True, logits_to_keep=1, return_dict=True)
                    result = output.past_key_values
                    if type(result) is not Qwen3_5DynamicCache or result.get_seq_length() != ids.shape[1]:
                        raise ValueError('Unexpected hybrid cache type or prefix length')
                    return result

                shared = None
                if schema_cache is not None and schema_prefix:
                    if len(schema_prefix) > len(tokens) or tokens[:len(schema_prefix)] != tuple(schema_prefix):
                        raise ValueError('Schema token prefix mismatch')
                    shared = schema_cache.snapshot_for(schema_prefix,
                        lambda: prefill(prefix[:, :len(schema_prefix)]))
                if shared is None:
                    snapshot = prefill(prefix)
                elif len(schema_prefix) == len(tokens):
                    snapshot = deepcopy(shared)
                else:
                    snapshot = prefill(prefix, len(schema_prefix), deepcopy(shared))
                self.snapshot, self.tokens = snapshot, tokens
                self.stats["snapshot_tokens"] = len(tokens)
            else:
                self.stats["hits"] += 1
                self.stats["reused_tokens"] += len(tokens)
            source = self._committed_snapshot(model, input_ids, commit_ids) if commit_ids else self.snapshot
            # Includes key/value, conv_states and recurrent_states on their own devices.
            working = deepcopy(source)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
            # Explicit text positions avoid dependence on model-global mRoPE deltas.
            positions = positions.unsqueeze(0).expand(4, -1, -1)
            yield {"past_key_values": working, "position_ids": positions}
