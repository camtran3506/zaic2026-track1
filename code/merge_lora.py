#!/usr/bin/env python3
"""Export a local LoRA adapter as a standalone offline model (no bitsandbytes)."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Merged output directory must be empty")
    if args.output_dir.resolve() in (args.base_model.resolve(), args.adapter.resolve()):
        raise ValueError("Output cannot overwrite base model or adapter")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(str(args.base_model),
        torch_dtype=getattr(torch, args.dtype), device_map="cpu", local_files_only=True,
        trust_remote_code=False, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer_path = args.adapter if (args.adapter / "tokenizer_config.json").is_file() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(str(args.output_dir))
    print("Merged offline model:", args.output_dir)


if __name__ == "__main__":
    main()
