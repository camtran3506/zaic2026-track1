"""Portable inference settings and artifact identity; no engine is imported here."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path

from .context import DEFAULT_MAX_INPUT_TOKENS, DEFAULT_MAX_NEW_TOKENS, MAX_CONTEXT_TOKENS

from .function_calling import JSON_PROTOCOL, NATIVE_PROTOCOL
from .local_time import LOCAL_TIME_POLICY
from .local_helpers import LOCAL_HELPER_POLICY
from .local_assist import ASSIST_POLICY
from .schema_prompt import SCHEMA_PROMPT_POLICY
from .text import TEXT_NORMALIZATION
from .itn import ITN_POLICY
from .execution_policy import EXECUTION_POLICY

DEFAULT_LLM_DTYPE = "bfloat16"
DEFAULT_SEED = 42
RUNTIME_MANIFEST = "lifemate_runtime.json"


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def model_fingerprint(model_path):
    """Hash local config and weight files once per load/export, independent of path."""
    root = Path(model_path)
    weights = sorted({*root.glob("*.safetensors"), *root.glob("pytorch_model*.bin")})
    if not weights or not (root / "config.json").is_file():
        raise FileNotFoundError(f"Model config/weights missing: {root}")
    files = [root / "config.json", *weights, *sorted(root.glob("*.index.json"))]
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    return hashes


def tokenizer_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Padding/truncation are transient per-call state, not vocabulary or rules.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    return _digest({"backend": backend, "chat_template": tokenizer.chat_template,
                    "special_tokens": tokenizer.special_tokens_map})


def generation_policy(tokenizer, model_path=None, seed=DEFAULT_SEED):
    source = Path(model_path) / "generation_config.json" if model_path else None
    config = json.loads(source.read_text()) if source and source.is_file() else {}
    eos = config.get("eos_token_id", tokenizer.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    eos = [token for token in eos if type(token) is int]
    if type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id not in eos:
        eos.append(tokenizer.eos_token_id)
    pad = tokenizer.pad_token_id
    pad = pad if type(pad) is int else (eos[0] if eos else None)
    return {"decoding": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": -1,
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "seed": seed, "eos_token_ids": eos, "pad_token_id": pad,
            "max_input_tokens": DEFAULT_MAX_INPUT_TOKENS,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS, "max_context_tokens": MAX_CONTEXT_TOKENS}


def transformers_generation_config(policy):
    """A fresh config prevents model-repository sampling defaults leaking in."""
    from transformers import GenerationConfig
    if (policy["decoding"] != "greedy" or policy["temperature"] != 0.0
            or policy["top_p"] != 1.0 or policy["top_k"] != -1
            or policy["presence_penalty"] != 0.0 or policy["frequency_penalty"] != 0.0):
        raise ValueError("Unsupported generation policy; expected neutral greedy decoding")
    # HF implements greedy with do_sample=False; temperature=0 is a portable
    # setting for other engines, but is not valid in a saved HF GenerationConfig.
    return GenerationConfig(do_sample=False, num_beams=1,
        repetition_penalty=policy["repetition_penalty"],
        eos_token_id=policy["eos_token_ids"] or None, pad_token_id=policy["pad_token_id"],
        bos_token_id=None, use_cache=True)


@contextmanager
def generation_context(model, policy, max_new_tokens):
    """Stop HF from filling None fields from repository generation defaults.

    GenerationMixin consults model.generation_config even when a fresh config
    is passed. PEFT delegates generation to its base model, so cover both objects
    and restore them after this single-request decode (also during validation).
    """
    config = transformers_generation_config(policy)
    config.max_new_tokens = max_new_tokens
    targets = [model]
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        base = get_base()
        if base is not model:
            targets.append(base)
    previous = [(target, "generation_config" in vars(target),
                 getattr(target, "generation_config", None)) for target in targets]
    try:
        for target in targets:
            target.generation_config = config
        yield config
    finally:
        for target, own_attribute, original in reversed(previous):
            if own_attribute:
                target.generation_config = original
            else:
                del target.generation_config


def select_tokenizer_path(model_path, adapter_path=None):
    path = Path(adapter_path) if adapter_path is not None else Path(model_path)
    if adapter_path is not None and not (path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Adapter tokenizer missing: {path}; restore the tokenizer saved during training")
    return path


def make_runtime_manifest(model_path, tokenizer, seed=DEFAULT_SEED, policy=None):
    return {"version": 1, "inference_dtype": DEFAULT_LLM_DTYPE,
            "text_normalization": TEXT_NORMALIZATION,
            "itn_policy": ITN_POLICY,
            "execution_policy": EXECUTION_POLICY,
            "local_time_policy": LOCAL_TIME_POLICY,
            "local_helper_policy": LOCAL_HELPER_POLICY,
            "assist_policy": ASSIST_POLICY,
            "schema_prompt_policy": SCHEMA_PROMPT_POLICY,
            "model_fingerprint": model_fingerprint(model_path),
            "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
            "chat_template_kwargs": dict(getattr(tokenizer, "lifemate_template_kwargs", {})),
            "prompt": {"add_generation_prompt": True, "add_special_tokens": False},
            "generation": deepcopy(policy) if policy else generation_policy(tokenizer, model_path, seed),
            "output_protocol": NATIVE_PROTOCOL if getattr(tokenizer, "lifemate_native_fc", False) is True else JSON_PROTOCOL}


def save_runtime_manifest(output_dir, manifest):
    generation = transformers_generation_config(manifest["generation"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / RUNTIME_MANIFEST
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    generation.save_pretrained(output)


def load_runtime_manifest(model_path, tokenizer, artifact_path=None):
    path = Path(artifact_path if artifact_path is not None else model_path) / RUNTIME_MANIFEST
    if not path.is_file():
        if artifact_path is not None:
            logging.warning("Legacy adapter has no %s; base/tokenizer identity cannot be verified; "
                            "local time tool training is unverified; Python specialist training is unverified", RUNTIME_MANIFEST)
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("Unsupported runtime manifest version")
    if manifest["model_fingerprint"] != model_fingerprint(model_path):
        raise ValueError("Base/model checkpoint differs from the recorded runtime artifact")
    if manifest["tokenizer_fingerprint"] != tokenizer_fingerprint(tokenizer):
        raise ValueError("Tokenizer vocabulary, special tokens or chat template differ from training/export")
    if manifest["chat_template_kwargs"] != getattr(tokenizer, "lifemate_template_kwargs", {}):
        raise ValueError("Chat template kwargs differ from training/export")
    if manifest["prompt"] != {"add_generation_prompt": True, "add_special_tokens": False}:
        raise ValueError("Prompt tokenization policy differs from training/export")
    if manifest.get("text_normalization") != TEXT_NORMALIZATION:
        raise ValueError("Text normalization differs from training/export; regenerate SFT and retrain the adapter")
    if manifest.get("local_time_policy") != LOCAL_TIME_POLICY:
        raise ValueError("Local time tool policy differs from training/export; regenerate SFT and retrain the adapter")
    if manifest.get("local_helper_policy") != LOCAL_HELPER_POLICY:
        raise ValueError("Python specialist helper policy differs from training/export; regenerate SFT and retrain the adapter")
    # Existing primitive-only adapters remain loadable; new exports pin workflows.
    if manifest.get("assist_policy", ASSIST_POLICY) != ASSIST_POLICY:
        raise ValueError("Python task workflow policy differs from training/export; regenerate SFT and retrain the adapter")
    if manifest.get("schema_prompt_policy", SCHEMA_PROMPT_POLICY) != SCHEMA_PROMPT_POLICY:
        raise ValueError("Schema prompt policy differs from training/export; regenerate SFT and retrain the adapter")
    if manifest.get("itn_policy") != ITN_POLICY:
        raise ValueError("ITN policy differs from training/export; regenerate SFT and retrain the adapter")
    if manifest.get("execution_policy") != EXECUTION_POLICY:
        raise ValueError("Automatic execution policy differs from training/export; regenerate SFT and retrain the adapter")
    expected_protocol = NATIVE_PROTOCOL if getattr(tokenizer, "lifemate_native_fc", False) is True else JSON_PROTOCOL
    if manifest["output_protocol"] != expected_protocol:
        raise ValueError("Output protocol differs from the LifeMate agent; regenerate SFT and retrain the adapter")
    transformers_generation_config(manifest["generation"])
    return manifest


def runtime_save_callback(manifest):
    """Give every Trainer checkpoint the same portable settings as the final adapter."""
    from transformers import TrainerCallback

    class RuntimeSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            save_runtime_manifest(Path(args.output_dir) / f"checkpoint-{state.global_step}", manifest)
            return control

    return RuntimeSaveCallback()


def training_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32  # Explicit CPU smoke-test backend only.
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 training requires a CUDA device with BF16 support; no automatic FP16 fallback")
    return torch.bfloat16
