"""Per-sample, exact-token prefix snapshots for the pinned Qwen3.5 runtime.

Only text-only Transformers 5.4.0 is enabled. Other runtimes keep the existing
full-prefill path. Never crop a hybrid cache: its recurrent state cannot be
rolled back by slicing attention tensors.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import logging
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
    if layer.causal_conv1d_fn is not None:
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
    """One immutable GPU snapshot; clone before use and release between samples."""
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
        self.prefix_memo = {}

    def begin_sample(self):
        self.clear()
        self.stats = {"mode": self.mode, "hits": 0, "misses": 0,
                      "reused_tokens": 0, "snapshot_tokens": 0, "fallbacks": 0}

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
            layers = [m for m in base.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
            if not layers or any(getattr(m.forward, "__func__", None) is not Qwen3_5GatedDeltaNet.forward for m in layers):
                self.disable("linear-attention forward is absent or patched by another runtime")
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

    @contextmanager
    def restore(self, model, input_ids, prefix_ids):
        """Build/restore at an exact boundary, then shield snapshot from writes."""
        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
        tokens = tuple(prefix_ids)
        with qwen54_continuation(self._layers):
            if self.snapshot is None or self.tokens != tokens:
                self.snapshot, self.tokens = None, ()
                self.stats["misses"] += 1
                prefix = input_ids[:, :len(tokens)]
                positions = torch.arange(len(tokens), device=input_ids.device).view(1, -1)
                output = model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                               position_ids=positions, use_cache=True, logits_to_keep=1,
                               return_dict=True)
                snapshot = output.past_key_values
                if type(snapshot) is not Qwen3_5DynamicCache or snapshot.get_seq_length() != len(tokens):
                    raise ValueError("Unexpected hybrid cache type or prefix length")
                self.snapshot, self.tokens = snapshot, tokens
                self.stats["snapshot_tokens"] = len(tokens)
                del output
            else:
                self.stats["hits"] += 1
                self.stats["reused_tokens"] += len(tokens)
            # Includes key/value, conv_states and recurrent_states on their own devices.
            working = deepcopy(self.snapshot)
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1)
            # Explicit text positions avoid dependence on model-global mRoPE deltas.
            positions = positions.unsqueeze(0).expand(4, -1, -1)
            yield {"past_key_values": working, "position_ids": positions}
