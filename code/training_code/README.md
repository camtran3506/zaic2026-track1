# Training Code & Reproducibility Guide (BTC Section 6.2)

This folder reproduces both inference artifacts shipped under `../saved_models/`:
1. **`saved_models/asr_ft/`** — Domain-adapted **GIPformer 1.5** (`68M` Zipformer-RNNT, FP32 ONNX)
2. **`saved_models/llm_ft/`** — Fine-tuned & merged **Qwen3.5-4B** (`r=16, alpha=32` BF16 LoRA)

---

## 1. Default Training Pipeline Architecture

### Stage A — ASR Domain Adaptation (`training_code/asr_finetune/`)
- **Data preparation (`prepare_gipformer_sft.py`)**: Reads `../data/train_asr.jsonl`, keeps human-verified `audio_text` on `1,536` main utterances untouched, and applies selective text normalization on `9` leftover main + `3,491` user history turns -> **`4,496` train / `540` dev rows** (`val_fraction=0.1`, `seed=42`).
- **PyTorch RNNT SFT (`train_gipformer.py`)**: Full-model fine-tuning (encoder + decoder + joiner) from `saved_models/asr/model.pt` with acoustic augmentation (`augment_scope=all`: speed perturbation, additive noise, SpecAugment), AdamW (`lr=3e-5`, `accum=4`, `epochs=3` = 13,488 steps), evaluating every `1000` steps (`max_dev_audio=300`, `patience=5`). Selects `best.pt` by lowest `dev_loss` (**step `13,000`**, `dev_loss=1.1992`) and exports **FP32 ONNX** via `export_gipformer.py` to `saved_models/asr_ft/`.

### Stage B — LLM Tool-Calling LoRA & Merge (`training_code/finetune.py` -> `merge_lora.py`)
- **Base & LoRA Architecture**: Base `Qwen/Qwen3.5-4B` (`Qwen3_5ForConditionalGeneration`, BF16, `16,384` max sequence length) patched with Unsloth 16-bit LoRA (`FastVisionModel`, vision tower frozen). LoRA (`r=16`, `lora_alpha=32`, `lora_dropout=0.0`) targets all language attention (`q/k/v/o_proj`), GatedDeltaNet linear attention (`in_proj_qkv/in_proj_z/out_proj`), and MLP projections (`gate/up/down_proj`) -> **`30,474,240` trainable params** (`0.67%` of `4,569,739,776` total params).
- **SFT Replay & Augmentation (`prepare_sft.py`, profile `run_sft_step`)**: Replays the full train split through `ToolEnv` (`--validation-fraction 0.0`, `1,544/1,545` kept) using `sft_run_step_compat.py` defaults:
  - **Dual-transcript + filtering (`dual`, ratio `0.5`)**: Generates paired `audio_text` and live `saved_models/asr_ft/` (`modified_beam_search`, `beam=4`, `hotwords=off`, `no-asr-post-normalize`) trajectories, dropping `28` ASR copies whose exact-match spoken gold arguments are corrupted (`--filter-damaged-asr`).
  - **Recovery & helper demonstrations**: Injects `61` failed-search recovery trajectories (`inject_recovery=True`), `382` guard-recovery self-correction examples (`guard_recovery_ratio=0.18`), deterministic `local_batch.run` grouping (up to `4` read/helper calls), and local helper / ITN demonstrations -> **`10,724` step-wise rows** (`3,121` trajectories; `sha256: a32459e8a200626f846a519b17fffa33427905838fcd220a076f54b182ab0996`).
- **Weighted LoRA Training & Merge (`train_lora.py`, `merge_lora.py`)**: `WeightedLossTrainer` computes assistant-only cross-entropy (`--no-short-reasoning`, `fuzzy_weight=0.3` on `x-match=fuzzy` spans, `1.0` on exact args/tool calls) for `epochs=3` (`4,023` steps, `batch=1 x accum=8`, cosine `lr=1e-4`, `seed=42`), then merges the adapter offline into BF16 standalone weights (`saved_models/llm_ft/`) with `lifemate_runtime.json`.

---

## 2. Datasets, Base Weights & External Repositories

- **Base LLM**: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0, pinned commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`)
- **Base ASR**: [`g-group-ai-lab/gipformer1.5-68M-rnnt`](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt) (pinned commit `dd9227dcd8705c13f33bdbe59728d546ab94480f`)
- **Hugging Face Dataset**: [`camtran3506/zaic2026-lifemate-data`](https://huggingface.co/datasets/camtran3506/zaic2026-lifemate-data) (mirrors `../data/`: `train/`, `train_asr.jsonl`)
  ```bash
  hf download camtran3506/zaic2026-lifemate-data --repo-type dataset --local-dir ../data
  ```
- **Hugging Face Models Archive**: [`camtran3506/zaic2026-lifemate-models`](https://huggingface.co/camtran3506/zaic2026-lifemate-models) (`asr-gipformer/` -> `saved_models/asr_ft/`, `llm-adapter/` -> `run_sft_step` LoRA adapter, `llm-merged/` -> merged BF16 `saved_models/llm_ft/`)
- **Seed Policy**: `seed = 42` fixed across `PYTHONHASHSEED`, `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic`, and Transformers `set_seed`.

---

## 3. Clean Reproduction Steps (Run from `code/`)

Install pinned training dependencies on top of runtime `requirements.txt` (requires GPU + CUDA 12.8 for Unsloth & `k2`):
```bash
pip install -r training_code/requirements-train.txt
# Optional for Step 2 (GIPformer PyTorch training): install CUDA 12.8 / torch 2.11 k2 + kaldifeat
# pip install k2==1.24.4.dev20260625+cuda12.8.torch2.11.0 -f https://k2-fsa.github.io/k2/cuda.html
# pip install kaldifeat==1.24 -f https://csukuangfj.github.io/kaldifeat/cuda.html
```

### Step 1: Stage Pinned Base Models (`saved_models/asr/` & `saved_models/llm/`)
```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 training_code/download_models.py
# Or also fetch the pre-exported fine-tuned ASR ONNX directly:
# HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 training_code/download_models.py \
#   --only asr --asr-ft-repo camtran3506/zaic2026-lifemate-models --asr-ft-subdir asr-gipformer
```

### Step 2: Fine-Tune GIPformer ASR & Export FP32 ONNX (`saved_models/asr_ft/`)
*(All default hyperparameters match the shipped `asr_ft` run.)*
```bash
bash training_code/asr_finetune/setup_icefall.sh
python3 training_code/asr_finetune/prepare_gipformer_sft.py
python3 training_code/asr_finetune/train_gipformer.py --cuda-visible-devices 0
python3 training_code/asr_finetune/export_gipformer.py --best result/asr_sft/best.pt
```

### Step 3: Train Qwen3.5-4B LoRA (`run_sft_step` Defaults) & Merge (`saved_models/llm_ft/`)
*(Automatically uses `saved_models/llm` or `models/llm`, `saved_models/asr_ft`, and all `run_sft_step` replay/LoRA defaults; `merge_lora.py` exports standalone BF16 weights to `saved_models/llm_ft/`.)*
```bash
python3 training_code/finetune.py --cuda-visible-devices 0 --output-dir result/run_sft_step_repro
python3 training_code/merge_lora.py --adapter result/run_sft_step_repro/training/adapter
```

> **Docker Packaging Note**: `code/.dockerignore` excludes base/intermediate artifacts (`models/`, `saved_models/asr/`, `saved_models/.research/`, `result/`, `unsloth_compiled_cache/`) so the submission container only carries `saved_models/asr_ft/` and `saved_models/llm_ft/`.

---

## 4. Core Training Scripts

| File | Purpose |
| --- | --- |
| `requirements-train.txt` | Pinned training dependencies (`unsloth`, `trl`, `datasets`, `lhotse`, `k2`/`kaldifeat` notes) |
| `download_models.py` | Downloads pinned HF base commits and optional `asr-gipformer` ONNX snapshot |
| `asr_finetune/` | `setup_icefall.sh`, `prepare_gipformer_sft.py`, `train_gipformer.py`, `export_gipformer.py` |
| `finetune.py` | End-to-end LLM SFT orchestrator (`prepare_sft.py` -> `train_lora.py`) |
| `prepare_sft.py` | `ToolEnv` replay, dual-transcript pairing, recovery/guard injection, batching |
| `sft_run_step_compat.py` | Scoped replay profile (`--sft-profile run_sft_step`, default) preserving exact SFT parity |
| `train_lora.py` | Unsloth BF16 LoRA trainer with assistant-only `WeightedLossTrainer` |
| `merge_lora.py` | Offline BF16 LoRA merge exporting `saved_models/llm_ft/` + `lifemate_runtime.json` |
| `score_train.py` | Offline tool-call F1 / state-hash scorer |
