"""CPU-only, per-sample prompt preparation; observations are never append-cached."""
import hashlib
import json

from .catalog import prompt_catalog
from .execution_policy import EXECUTION_POLICY
from .function_calling import native_chat, native_observation_messages
from .history import prompt_history
from .itn import ITN_POLICY, detect_candidates
from .local_helpers import with_local_tools
from .text import normalize_transcript


# Qwen3.5 template body exercised by tests/fixtures/qwen35_chat_template.jinja.
# Other templates still benefit from prepared JSON/native messages, but are
# rendered/tokenized in full. Never infer arbitrary Jinja semantics from a probe.
_QWEN35_TEMPLATE_SHA256 = "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
_ANCHOR = {"role": "user", "content": "LifeMate observation boundary"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def fixed_prompt_payload(transcript, context, history, enabled):
    """The same ordered payload used by the reference SFT/inference builder."""
    tools = prompt_catalog(with_local_tools(enabled))
    transcript = normalize_transcript(transcript)
    history = prompt_history(history)
    return tools, {
        "request_transcript": transcript,
        "itn_policy": ITN_POLICY,
        "automatic_execution_policy": EXECUTION_POLICY,
        "itn_candidates": detect_candidates(transcript),
        "itn_history_candidates": [
            {"history_index": i, "candidates": candidates}
            for i, turn in enumerate(history)
            if (candidates := detect_candidates(turn["text"]))],
        "user_context": {k: v for k, v in context.items() if k != "installed_apps"},
        "conversation_history": history,
        "available_tools": tools,
    }


class PreparedPrompt(str):
    """A portable JSON string with trusted, local preparation metadata.

    Existing planners/traces still see the identical string. Metadata is never
    serialized into a prompt, passed to ToolEnv, or loaded from dataset text.
    """
    def __new__(cls, content, session, observations):
        obj = super().__new__(cls, content)
        obj.session = session
        obj.observations = observations
        obj.fixed_json = session._fixed_json
        return obj

    def __copy__(self):
        return str(self)

    def __deepcopy__(self, memo):
        return str(self)


def _template_digest(template):
    body = template.strip()
    if body.startswith("{#"):
        body = body.split("#}", 1)[-1].strip()
    return hashlib.sha256(body.encode()).hexdigest()


def _safe_tokenizer(tokenizer):
    """A special-token boundary isolates byte BPE pieces without normalization."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or getattr(tokenizer, "split_special_tokens", False):
        return False
    if type(backend.model).__name__ != "BPE" or backend.model.dropout is not None:
        return False
    if backend.normalizer is not None:
        return False
    def safe_post(state):
        if state["type"] == "ByteLevel":
            return True  # Changes offsets, not token IDs.
        if state["type"] == "Sequence":
            return all(safe_post(part) for part in state["processors"])
        if state["type"] == "TemplateProcessing":
            # add_special_tokens=False omits SpecialToken entries; HF may
            # install this identity processor even on a byte-BPE tokenizer.
            sequences = [entry for entry in state["single"] if "Sequence" in entry]
            return sequences == [{"Sequence": {"id": "A", "type_id": 0}}]
        return False
    if backend.post_processor is not None and not safe_post(json.loads(backend.post_processor.__getstate__())):
        return False
    pretokenizer = backend.pre_tokenizer
    if pretokenizer is None:
        return False
    def safe_pre(state):
        if state["type"] == "Sequence":
            return all(safe_pre(part) for part in state["pretokenizers"])
        return state["type"] == "Split" or (state["type"] == "ByteLevel" and not state["add_prefix_space"])
    if not safe_pre(json.loads(pretokenizer.__getstate__())):
        return False
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    token = tokenizer.added_tokens_decoder.get(end_id)
    return bool(token and str(token) == "<|im_end|>" and token.special
                and not any((token.lstrip, token.rstrip, token.single_word, token.normalized)))


def _boundary_key(tokenizer):
    """Cheap settings only; do not serialize the full vocabulary every step."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    token = tokenizer.added_tokens_decoder.get(end_id)
    return (tuple(component.__getstate__() if component is not None else None
                  for component in (backend.normalizer, backend.pre_tokenizer, backend.post_processor)),
            type(backend.model).__name__, getattr(backend.model, "dropout", None),
            getattr(tokenizer, "split_special_tokens", False), end_id, repr(token))


class InferenceSession:
    """Own the immutable sample payload and one tokenizer's prepared CPU prefix."""
    def __init__(self):
        self.stats = {"fixed_builds": 0, "renders": 0, "suffix_renders": 0,
                      "full_renders": 0, "reused_tokens": 0, "fallback_reason": None}
        self.close()

    def prepare(self, transcript, context, history, enabled):
        self.close()
        self.tools, payload = fixed_prompt_payload(transcript, context, history, enabled)
        # Freeze nested context/schema/history, not just the outer mapping.
        self._fixed_json = _json(payload)[:-1] + ',"previous_actions_and_results":'
        self.stats["fixed_builds"] += 1

    def prompt(self, observations):
        if self._fixed_json is None:
            raise ValueError("InferenceSession is not prepared or has been closed")
        serialized = _json(observations)
        # Match the old full-payload JSON round trip, including tuples and
        # objects serialized by default=str. Parse only this mutable suffix.
        return PreparedPrompt(self._fixed_json + serialized + "}", self, json.loads(serialized))

    def close(self):
        self._fixed_json = None
        self.tools = None
        self._renderer_key = None
        self._tokenizer = None
        self._base_chat = self._tools = self._kwargs = None
        self._prefix_text = self._prefix_ids = self._anchor_text = None
        self._newline_ids = self._gpu_prefix_ids = None

    def report(self):
        return dict(self.stats)

    def schema_head(self, tokenizer, system):
        """Sample-free native system/tools material after render() prepared it."""
        if (self._base_chat and self._tokenizer is tokenizer
                and self._renderer_key is not None and self._renderer_key[-1] == system):
            return self._base_chat[:1], {**self._kwargs, 'tools': self._tools}
        return None

    @staticmethod
    def _tokens(tokenizer, text):
        return tokenizer(text, add_special_tokens=False, truncation=False,
                         return_token_type_ids=False)["input_ids"]

    def _render_full(self, tokenizer, observations):
        chat = self._base_chat + observations
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True,
                                             tools=self._tools, **self._kwargs)
        self.stats["full_renders"] += 1
        return text, self._tokens(tokenizer, text), []

    def _render_suffix(self, tokenizer, observations):
        # Re-render ALL observations: later user feedback can change how Qwen
        # renders earlier assistant reasoning, and rejected feedback can vanish.
        rendered = tokenizer.apply_chat_template([_ANCHOR, *observations], tokenize=False,
            add_generation_prompt=True, **self._kwargs)
        if not rendered.startswith(self._anchor_text):
            raise ValueError("Observation anchor is not an exact rendered prefix")
        suffix = "\n" + rendered[len(self._anchor_text):]
        ids = self._prefix_ids + self._tokens(tokenizer, suffix)
        # Keep the GPU snapshot boundary identical to stable_prefix_ids from
        # before CPU caching (the closed request INCLUDING its newline).
        end = len(self._gpu_prefix_ids)
        gpu_prefix = (self._gpu_prefix_ids if end < len(ids)
                      and ids[len(self._prefix_ids):end] == self._newline_ids else [])
        return self._prefix_text + suffix, ids, gpu_prefix

    def render(self, tokenizer, system, prompt):
        """Return exactly the reference text/IDs, or use full rendering on a miss."""
        if (self._fixed_json is None or prompt.session is not self
                or prompt.fixed_json is not self._fixed_json):
            return None
        if getattr(tokenizer, "lifemate_native_fc", False) is not True:
            return None
        template = tokenizer.get_chat_template()
        kwargs = dict(getattr(tokenizer, "lifemate_template_kwargs", {}))
        kwargs.pop("tools", None)  # template_inputs always overrides this with the real catalog.
        key = (id(tokenizer), template, _json(kwargs), len(tokenizer), _boundary_key(tokenizer), system)
        observations = native_observation_messages(prompt.observations)
        self.stats["renders"] += 1
        if key != self._renderer_key:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": self._fixed_json + "[]}"}]
            self._base_chat, self._tools = native_chat(messages)
            if "tool_calls" not in template or "tools" not in template:
                raise ValueError("Qwen3.5 checkpoint chat template must support tools and tool_calls")
            self._kwargs, self._renderer_key = kwargs, key
            self._tokenizer = tokenizer  # Keep identity alive until close/rebuild.
            self._prefix_text = self._prefix_ids = self._anchor_text = None
            self._newline_ids = self._gpu_prefix_ids = None
            self.stats["fallback_reason"] = None
            if (not isinstance(tokenizer.chat_template, str) or "chat_template" in kwargs
                    or _template_digest(template) != _QWEN35_TEMPLATE_SHA256 or not _safe_tokenizer(tokenizer)):
                self.stats["fallback_reason"] = "unverified template or tokenizer boundary"
            else:
                fixed = tokenizer.apply_chat_template(self._base_chat, tokenize=False,
                    add_generation_prompt=False, tools=self._tools, **kwargs)
                if fixed.endswith("<|im_end|>\n"):
                    # Split immediately AFTER the special token, moving the
                    # trailing newline to the independently tokenized suffix.
                    self._prefix_text = fixed[:-1]
                    self._prefix_ids = self._tokens(tokenizer, self._prefix_text)
                    self._newline_ids = self._tokens(tokenizer, "\n")
                    self._gpu_prefix_ids = self._prefix_ids + self._newline_ids
                    self._anchor_text = tokenizer.apply_chat_template([_ANCHOR], tokenize=False,
                        add_generation_prompt=False, **kwargs)
                    full = self._render_full(tokenizer, observations)
                    joined = self._render_suffix(tokenizer, observations)
                    if joined[:2] != full[:2] or self._prefix_ids[-1] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
                        self._prefix_text = self._prefix_ids = None
                        self.stats["fallback_reason"] = "initial full-prompt parity check failed"
                        return full
                    return joined
                self.stats["fallback_reason"] = "no closed special-token prefix boundary"
        if self._prefix_ids is not None:
            try:
                result = self._render_suffix(tokenizer, observations)
            except ValueError:
                self._prefix_text = self._prefix_ids = None
                self.stats["fallback_reason"] = "observation rendering boundary changed"
            else:
                self.stats["suffix_renders"] += 1
                self.stats["reused_tokens"] += len(self._prefix_ids)
                return result
        return self._render_full(tokenizer, observations)
