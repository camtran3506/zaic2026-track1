#!/usr/bin/env python3
"""Export a local LoRA adapter as a standalone offline model (no bitsandbytes)."""
import argparse
from pathlib import Path
from lifemate.llm import model_loader, configure_tokenizer
from lifemate.runtime import (DEFAULT_LLM_DTYPE, load_runtime_manifest, make_runtime_manifest,
                              save_runtime_manifest, select_tokenizer_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=DEFAULT_LLM_DTYPE)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Merged output directory must be empty")
    if args.output_dir.resolve() in (args.base_model.resolve(), args.adapter.resolve()):
        raise ValueError("Output cannot overwrite base model or adapter")
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    tokenizer_path = select_tokenizer_path(args.base_model, args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    configure_tokenizer(tokenizer, args.base_model)
    source_runtime = load_runtime_manifest(args.base_model, tokenizer, args.adapter)
    model = model_loader(args.base_model).from_pretrained(str(args.base_model),
        torch_dtype=getattr(torch, args.dtype), device_map="cpu", local_files_only=True,
        trust_remote_code=False, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True,
                                     autocast_adapter_dtype=False, device_map={'': 'cpu'})
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(args.output_dir))
    runtime = make_runtime_manifest(args.output_dir, tokenizer,
                                   policy=source_runtime["generation"] if source_runtime else None)
    runtime["inference_dtype"] = args.dtype
    save_runtime_manifest(args.output_dir, runtime)
    print("Merged offline model:", args.output_dir)


if __name__ == "__main__":
    main()
