# LifeMate — Offline Vietnamese Voice-to-Tools Agent (Zalo AI Challenge 2026)

Offline end-to-end voice agent for Zalo AI Challenge 2026 (LifeMate track). All models, tokenizers, and runtime dependencies run **100% offline** inside Docker without external network access.

---

## 1. Solution Overview (BTC Section 6.1)

### 1.1 ASR Model
- **Model**: `g-group-ai-lab/gipformer1.5-68M-rnnt` (**~68M parameters**), Zipformer-RNNT architecture, exported to **FP32 ONNX** (`saved_models/asr_ft/`) for offline CPU decoding via `sherpa-onnx`.
- **Domain Adaptation**: Fine-tuned on 1,536 human-verified Vietnamese main utterances + 9 leftover digit utterances + 3,491 user history turns from `../data/train_asr.jsonl` (`4,496` train / `540` dev rows) using selective normalization (`training_code/asr_finetune/`, `epochs=3`, `lr=3e-5`, `accum=4`, `eval_every=1000`, `best.pt` at step 13,000).
- **Inference Decoding (Hybrid ASR)**: 16 kHz mono audio, CPU `8` threads, `modified_beam_search` (`max_active_paths=8`), dynamic per-sample Hybrid contextual BPE hotwords extracted from installed apps/contacts/entities/tool schemas (`hotwords_score=1.5`), plus deterministic Hybrid post-ASR normalization (`--asr-post-normalize` in [lifemate/asr_biasing.py](lifemate/asr_biasing.py)).

### 1.2 LLM Model
- **Model**: `Qwen/Qwen3.5-4B` (**~4.0B parameters**, **BF16**, loaded via `AutoModelForImageTextToText` in text-only mode with native Qwen3.5 function calling, fine-tuned with **LoRA** `r=16, alpha=32` targeting language attention, GatedDeltaNet `linear_attn`, and MLP projections — vision layers frozen/unmodified).
- **Total Inference Parameters**: ~4.07B (< 5B limit).
- **Context Length**: **16,384 tokens** shared strictly across SFT training and inference (`max_input_tokens=16,000`, `max_new_tokens=384`, `thinking_budget=0` by default with `DirectActionLogitsProcessor` and greedy deterministic decoding).

### 1.3 Tool Calling & Agent Loop
- **Single-Process E2E Pipeline (`lifemate/predictor.py`, `predict.py`)**: Loads both ASR (CPU) and LLM (CUDA GPU) upfront once at startup with fixed seed (`seed=42`). Then iterates through test cases one-by-one in original dataset order:
  1. Start sample timer `t1 = time.perf_counter()` **before reading audio** and create one `ToolEnv(split, sample["id"], root=...)` imported strictly from `<split>/zaic_tools`.
  2. Transcribe current utterance WAV and any audio history turns on demand with sample-specific contextual BPE hotwords.
  3. Run multi-turn agent loop: render prompt with exact tool schemas (reusing KV prefix cache & cross-sample GPU schema snapshot cache), execute native tool calls via `ToolEnv.validate` + `ToolEnv.call` or stateless local helpers (`local_time.*`, `local_entity.*`, `local_context.*`, `local_checks.*`, `local_assist.solve`) and batched independent reads (`local_batch.run`).
  4. Record `tool_calls`, `state_hash`, and `time` (ms), flushing each line immediately to `/result/submission.jsonl` (or `/result/jupyter_submission.jsonl`).
- **Safety & Reliability Guards**: Automatic precheck/missing-info guards (`missing_info_guard.py`), deterministic recovery retries (`max_calls=8`, `max_invalid=3`), and full schema fidelity.

### 1.4 Training Data & Synthetic / Augmentation Data
- **Official Dataset (`../data/`)**: Consolidated dataset (`../data/train/`, `../data/public_test/`, `../data/train_asr.jsonl`). No validation split is held out during final training (`--validation-fraction 0.0`).
- **Dual-Transcript SFT Replay (`training_code/prepare_sft.py`)**: Replays every `train` sample against the real `ToolEnv`, generating both clean gold `audio_text` transcripts and live fine-tuned `asr_ft` transcripts (`--transcript-mode dual`), filtered with `--filter-damaged-asr` to drop contradictory ASR transcripts where exact-match spoken arguments are corrupted.
- **Synthetic & Recovery Augmentation**:
  - Deterministic injection of failed-search self-correction and missing-info/xref guard recovery trajectories (`lifemate/recovery.py`, `--guard-recovery-ratio 0.18`).
  - Local Python specialist & Vietnamese ITN/time helper demonstrations (`local_time`, `local_entity`, `local_checks`, `local_assist`).
  - Weighted cross-entropy loss (`--fuzzy-weight 0.3` on `x-match=fuzzy` string spans, full `1.0` weight on exact args and tool names).

---

## 2. Repository Structure (BTC Section 5 Layout)

```text
lifemate/
├── data/                              # Consolidated training/eval data & train_asr.jsonl
│   ├── train/
│   ├── public_test/
│   └── train_asr.jsonl
└── code/                              # Docker /code root
    ├── predict.py                     # BTC entrypoint (/data -> /result/submission.jsonl)
    ├── predict.sh                     # BTC batch runner (/code/predict.sh)
    ├── start_jupyter.sh               # BTC JupyterLab launcher (port 9777, pw/token zac2026)
    ├── predict_notebook.ipynb         # 4-cell BTC timing notebook (/result/jupyter_submission.jsonl)
    ├── benchmark_timing.py            # CLI timing & parity verification script
    ├── main.py                        # Flexible inference CLI (--stage online/full/asr/llm)
    ├── run_server.sh                  # Local inference runner (relative paths -> ../data, saved_models/)
    ├── requirements.txt               # Pinned Python 3.12 + CUDA 12.8 runtime dependencies
    ├── Dockerfile                     # Offline CUDA 12.8 Docker build recipe
    ├── .dockerignore
    ├── README.md                      # This solution description (BTC Section 6.1)
    ├── lifemate/                      # Core ASR, LLM, agent, guards, helpers, predictor package
    ├── saved_models/                  # Inference model weights
    │   ├── asr_ft/                    # Fine-tuned GIPformer 1.5 FP32 ONNX
    │   ├── asr/                       # Base GIPformer checkpoint & ONNX assets
    │   └── llm_ft/ (or llm/)          # Merged BF16 Qwen3.5-4B model (or base + adapter)
    ├── training_code/                 # All training scripts & reproducibility guide (BTC Section 6.2)
    │   ├── README.md                  # Reproduction instructions, seed=42, model URLs
    │   ├── finetune.py                # End-to-end LLM SFT + LoRA training orchestrator
    │   ├── prepare_sft.py             # ToolEnv SFT replay & dual-transcript preparation
    │   ├── train_lora.py              # Unsloth / Transformers WeightedLossTrainer LoRA training
    │   ├── merge_lora.py              # Merge LoRA adapter -> saved_models/llm_ft
    │   ├── download_models.py         # Offline weight staging from Hugging Face
    │   ├── audit_initial_tokens.py    # Context budget audit utility
    │   ├── score_train.py             # Train proxy F1 scorer
    │   ├── check_lifemate_env.py      # Environment verification helper
    │   └── asr_finetune/              # GIPformer 1.5 PyTorch SFT + ONNX export scripts
    └── tests/                         # Automated unit & integration tests
```

---

## 3. Running Inference & Timing Verification

### Local Inference (relative paths from `code/`)
```bash
bash predict.sh --data-root ../data --split public_test --output result/submission.jsonl
# or via run_server.sh (defaults to ../data and saved_models/)
bash run_server.sh --split public_test
```

### Container Inference (BTC Standard)
```bash
# 1. Generate /result/submission.jsonl from mounted /data:
bash /code/predict.sh

# 2. Launch JupyterLab for per-sample latency notebook (Section 7):
bash /code/start_jupyter.sh
# Open port 9777 with password/token `zac2026` and run all cells of predict_notebook.ipynb

# 3. Verify timing & exact parity with benchmark_timing.py:
python3 benchmark_timing.py \
  --data-root ../data --split public_test \
  --output result/jupyter_submission.jsonl \
  --compare result/submission.jsonl --strict-compare
```

All default hyperparameters (`seed=42`, `asr-beam-size=8`, `asr-hotwords=auto`, `asr-post-normalize`, `asr-threads=8`, `max-input-tokens=16000`, `max-new-tokens=384`, `thinking-budget=0`) are fixed directly inside `predict.py` and `predict_notebook.ipynb`. For full training reproduction steps, see [training_code/README.md](training_code/README.md).
