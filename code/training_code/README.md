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
Both `prepare_gipformer_sft.py` and `train_gipformer.py` default to the exact settings that produced the shipped `saved_models/asr_ft/` model (`4,496` train / `540` dev rows, `epochs=3`, `learning_rate=3e-5`, `accum=4`, `eval_every=1000`, `patience=5`, `selection_metric=loss`; optional `d76a2c8` flags such as `--verify-main`, `--max-main-wer`, `--eval-wer`, `--selection-metric wer` remain off unless explicitly passed):

```bash
# 2a. Clone icefall architecture helpers into saved_models/.research/icefall
bash training_code/asr_finetune/setup_icefall.sh

# 2b. Prepare selective normalized ASR SFT dataset from ../data/train_asr.jsonl
python3 training_code/asr_finetune/prepare_gipformer_sft.py \
  --input-jsonl ../data/train_asr.jsonl \
  --data-root ../data \
  --model-dir saved_models/asr \
  --output-dir result/asr_sft_data \
  --val-fraction 0.1 \
  --seed 42

# 2c. Train Gipformer PyTorch checkpoint
python3 training_code/asr_finetune/train_gipformer.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --data-dir result/asr_sft_data \
  --model-dir saved_models/asr \
  --output-dir result/asr_sft \
  --epochs 3 \
  --learning-rate 3e-5 \
  --accum 4 \
  --max-dev-audio 300 \
  --augment-scope all \
  --selection-metric loss \
  --eval-every 1000 \
  --patience 5

# 2d. Export best checkpoint to FP32 ONNX (never overwriting base saved_models/asr/)
python3 training_code/asr_finetune/export_gipformer.py \
  --best result/asr_sft/best.pt \
  --original-dir saved_models/asr \
  --output-dir saved_models/asr_ft \
  --data-root ../data
```

### Step 3: Train Qwen3.5-4B Tool-Calling LoRA (`run_sft_step`)
`finetune.py` defaults to the **`run_sft_step`** hyperparameters, ASR settings (`--step-wise-sft`, `--no-short-reasoning`, `--asr-beam-size 4`, `--asr-hotwords off`, `--no-asr-post-normalize`, `--asr-threads 4`, `--filter-damaged-asr`), and replay profile (`--sft-profile run_sft_step`), and automatically uses `saved_models/llm` if staged via Step 1 or falls back to `models/llm` when unmerged base weights are stored there. *(Both static inference prompts `SYSTEM_PROMPT`/`date_reference` and the default `--sft-profile run_sft_step` compatibility hooks in `sft_run_step_compat.py` match the Nov-13 `79fdce3` `run_sft_step` SFT dataset byte-for-byte (`10,724` rows, `sha256: a32459e8a200626f846a519b17fffa33427905838fcd220a076f54b182ab0996`), while keeping Hybrid ASR and runtime guards active for inference. Pass `--sft-profile current` if you want SFT replay to use the newer runtime guard behavior instead.)*

```bash
python3 training_code/finetune.py \
  --cuda-visible-devices 0 \
  --data-root ../data \
  --base-model saved_models/llm \
  --output-dir result/run_sft_step \
  --asr-model saved_models/asr_ft \
  --transcript-mode dual \
  --asr-transcript-ratio 0.5 \
  --asr-precision fp32 \
  --asr-decoding-method modified_beam_search \
  --asr-beam-size 4 \
  --asr-hotwords off \
  --asr-hotwords-score 1.5 \
  --no-asr-post-normalize \
  --asr-threads 4 \
  --asr-batch-size 16 \
  --filter-damaged-asr \
  --sft-profile run_sft_step \
  --fuzzy-weight 0.3 \
  --reasoning-weight 0.25 \
  --guard-recovery-ratio 0.18 \
  --step-wise-sft \
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
*(Uses `saved_models/llm` by default, or automatically falls back to `models/llm` if omitted; pass `--base-model models/llm` explicitly when merging directly from `models/llm`.)*
```bash
python3 training_code/merge_lora.py \
  --base-model saved_models/llm \
  --adapter result/run_sft_step/training/adapter \
  --output-dir saved_models/llm_ft
```

> **Note for Docker Submission Packaging (BTC Section 6.2)**: Keep only fine-tuned inference assets (`saved_models/asr_ft/` and merged `saved_models/llm_ft/`) in the final inference Docker container, and avoid bundling unused training-only base checkpoints (`models/`, `saved_models/.research/`, `saved_models/asr/model.pt`).

---

## 4. Utility Scripts in `training_code/`
- `finetune.py`: End-to-end orchestrator (`prepare_sft.py` -> `train_lora.py` -> optional `merge_lora.py`).
- `prepare_sft.py`: Replays `train` samples through `ToolEnv` to build SFT JSONL + manifest.
- `sft_run_step_compat.py`: Replay-scoped compatibility hooks (`--sft-profile run_sft_step`, default) so `prepare_sft.py` reproduces `result/run_sft_step/sft/train.jsonl` byte-for-byte without modifying runtime inference guards.
- `train_lora.py`: Assistant-only weighted cross-entropy Unsloth / Transformers LoRA trainer.
- `merge_lora.py`: Offline BF16 LoRA merge exporting `lifemate_runtime.json`.
- `audit_initial_tokens.py`: Measures prompt token distributions against the 16,000-token budget.
- `score_train.py`: Offline train proxy tool-call F1 scorer.
- `download_models.py`: Downloads pinned HF base models into `saved_models/`.
- `check_lifemate_env.py`: Verifies CUDA/BF16, `sherpa-onnx`, Transformers/PEFT, Unsloth, and `zaic_tools`.
