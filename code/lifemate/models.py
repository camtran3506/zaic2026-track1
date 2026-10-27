"""Two completely local, deterministic model wrappers."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


class LocalASR:
    def __init__(self, model_path: Path, device: str = "cuda", compute_type: str = "float16", beam_size: int = 1):
        from faster_whisper import WhisperModel
        if not model_path.is_dir():
            raise FileNotFoundError(f"ASR model directory not found: {model_path}")
        self.model = WhisperModel(
            str(model_path), device=device, compute_type=compute_type,
            local_files_only=True, cpu_threads=4,
        )
        self.beam_size = beam_size
        self.cache: OrderedDict[str, str] = OrderedDict()

    def transcribe(self, path: Path) -> str:
        key = str(path.resolve())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        segments, _ = self.model.transcribe(
            str(path), language="vi", task="transcribe", beam_size=self.beam_size,
            temperature=0, condition_on_previous_text=False, vad_filter=False,
        )
        result = " ".join(seg.text.strip() for seg in segments).strip()
        self.cache[key] = result
        if len(self.cache) > 512:
            self.cache.popitem(last=False)
        return result


class LocalLLM:
    def __init__(self, model_path: Path, device: str = "cuda", dtype: str = "float16",
                 max_input_tokens: int = 6000, max_new_tokens: int = 192):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not model_path.is_dir():
            raise FileNotFoundError(f"LLM model directory not found: {model_path}")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
        self.tokenizer.truncation_side = "right"
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch_dtype,
            local_files_only=True, trust_remote_code=False,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_input_tokens = min(max_input_tokens, 16000)
        self.max_new_tokens = min(max_new_tokens, 384)

    def generate(self, system: str, user: str) -> str:
        tokenizer = self.tokenizer
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            ids = self.model.generate(
                **encoded, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        completion = ids[0, encoded["input_ids"].shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=True).strip()
