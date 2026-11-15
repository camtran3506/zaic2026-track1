# Training Code & Reproducibility Guide (BTC Section 6.2)

This directory contains the complete training pipeline to reproduce both the domain-adapted **GIPformer 1.5** ASR model (`saved_models/asr_ft/`) and the fine-tuned **Qwen3.5-4B** voice-to-tools LLM (`saved_models/llm_ft/` or LoRA adapter).

---

## 1. Datasets & Base Model URLs

- **Base LLM**: [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache-2.0)
- **Base ASR**: [`g-group-ai-lab/gipformer1.5-68M-rnnt`](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt)
- **Consolidated Training Data** (local relative layout `../data/` when executed from `code/`, or `../../data/` when executed from `code/training_code/`):
  - `../data/train/` (`train.jsonl`, `train_tools_catalog.json`, `audio/`, `train_environment/`, `zaic_tools/`)
  - `../data/public_test/`
  - `../data/train_asr.jsonl` (1,536 ASR training examples with human-verified `audio_text` + history turns)
- **External Artifact URLs (for BTC clean reproduction)**:
  - Hugging Face dataset / checkpoint repository URLs (update before final submission freeze):
    - Training dataset mirror (if hosted externally): `<HUGGINGFACE_DATASET_URL>` (mirrors `../data/`)
    - Stageable base weights are fetched automatically via `training_code/download_models.py` from the official HF IDs above.

---

## 2. Reproducibility & Fixed Seed Policy

All dataset splits, dual-transcript assignments, synthetic guard-recovery sampling, data collation, and optimizer initialization fix **`seed = 42`** (`random`, `PYTHONHASHSEED=42`, `numpy`, `torch`, `torch.cuda`, and Transformers `set_seed(42)`).

---

## 3. Step-by-Step Reproduction (Relative Commands from `code/`)

### Step 1: Stage Base Models into `saved_models/`
```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python3 training_code/download_models.py \
  --models-dir saved_models \
  --only both \
  --asr-precision all
```

### Step 2: Fine-Tune Gipformer 1.5 ASR & Export FP32 ONNX (`saved_models/asr_ft/`)
```bash
# 2a. Clone icefall architecture helpers into saved_models/.research/icefall
bash training_code/asr_finetune/setup_icefall.sh

# 2b. Prepare selective normalized ASR SFT dataset from ../data/train_asr.jsonl
python3 training_code/asr_finetune/prepare_gipformer_sft.py \
  --input-jsonl ../data/train_asr.jsonl \
  --data-root ../data \
  --model-dir saved_models/asr \
  --output-dir result/asr_sft_data \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --max-history-wer 0.4 \
  --seed 42 \
  --final

# 2c. Train Gipformer PyTorch checkpoint
python3 training_code/asr_finetune/train_gipformer.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --data-dir result/asr_sft_data \
  --model-dir saved_models/asr \
  --output-dir result/asr_sft \
  --epochs 5 \
  --learning-rate 5e-5 \
  --accum 2 \
  --augment-scope all \
  --max-history-wer 0.4 \
  --eval-wer \
  --selection-metric wer \
  --final

# 2d. Export best checkpoint to FP32 ONNX (never overwriting base saved_models/asr/)
python3 training_code/asr_finetune/export_gipformer.py \
  --best result/asr_sft/best.pt \
  --original-dir saved_models/asr \
  --output-dir saved_models/asr_ft \
  --data-root ../data
```

### Step 3: Train Qwen3.5-4B Tool-Calling LoRA
```bash
python3 training_code/finetune.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --base-model saved_models/llm \
  --output-dir result/run_sft_trajj \
  --asr-model saved_models/asr_ft \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --asr-precision auto \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 8 \
  --asr-hotwords auto \
  --asr-hotwords-score 1.5 \
  --asr-post-normalize \
  --asr-batch-size 16 \
  --filter-damaged-asr \
  --fuzzy-weight 0.3 \
  --guard-recovery-ratio 0.18 \
  --no-step-wise-sft \
  --no-short-reasoning \
  --epochs 3 \
  --batch-size 1 \
  --gradient-accumulation 8 \
  --learning-rate 1e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.0 \
  --save-strategy epoch \
  --seed 42
```

### Step 4: Merge LoRA Adapter into Standalone Inference Weights (`saved_models/llm_ft/`)
```bash
python3 training_code/merge_lora.py \
  --base-model saved_models/llm \
  --adapter result/run_sft_trajj/training/adapter \
  --output-dir saved_models/llm_ft
```

> **Note for Docker Submission Packaging (BTC Section 6.2)**: Keep only fine-tuned inference assets (`saved_models/asr_ft/` and merged `saved_models/llm_ft/`) in the final inference Docker container, and avoid bundling unused training-only base checkpoints (`models/`, `saved_models/.research/`, `saved_models/asr/model.pt`).

---

## 4. Utility Scripts in `training_code/`
- `finetune.py`: End-to-end orchestrator (`prepare_sft.py` -> `train_lora.py` -> optional `merge_lora.py`).
- `prepare_sft.py`: Replays `train` samples through `ToolEnv` to build SFT JSONL + manifest.
- `train_lora.py`: Assistant-only weighted cross-entropy Unsloth / Transformers LoRA trainer.
- `merge_lora.py`: Offline BF16 LoRA merge exporting `lifemate_runtime.json`.
- `audit_initial_tokens.py`: Measures prompt token distributions against the 16,000-token budget.
- `score_train.py`: Offline train proxy tool-call F1 scorer.
- `download_models.py`: Downloads pinned HF base models into `saved_models/`.
- `check_lifemate_env.py`: Verifies CUDA/BF16, `sherpa-onnx`, Transformers/PEFT, Unsloth, and `zaic_tools`.
