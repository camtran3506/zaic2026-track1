# LifeMate — Offline Vietnamese Voice-to-Tools Agent (Zalo AI Challenge 2026)

Offline single-process voice-to-tools agent for Zalo AI Challenge 2026 (LifeMate track). All weights, tokenizers, and runtime components execute **100% offline** inside Docker (`--network none`).

---

## 1. Default End-to-End Inference Architecture

At runtime, [`predict.py`](predict.py) / [`predict_notebook.ipynb`](predict_notebook.ipynb) launch [`LifeMatePredictor`](lifemate/predictor.py) with `seed=42` and offline flags (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`):

1. **Startup (once)**: Loads the fine-tuned **ASR (CPU)** from `saved_models/asr_ft/` and the merged **LLM (CUDA GPU, BF16, single-device)** from `saved_models/llm_ft/`.
2. **Per-sample execution (dataset order)**: Starts `t1 = time.perf_counter()` **before reading audio**, creates one `ToolEnv(split, id, root=...)` imported strictly from `/data/<split>/zaic_tools`, runs `Hybrid ASR -> Prompt/ITN -> Agent Loop -> ToolEnv`, and flushes `{"id", "tool_calls", "state_hash", "time"}` line-by-line to `/result/submission.jsonl` (or `/result/jupyter_submission.jsonl`).

### 1.1 ASR Model (CPU, Hybrid ONNX)
- **Model**: `g-group-ai-lab/gipformer1.5-68M-rnnt` (**~68M params**, Zipformer-RNNT), domain-adapted and exported to **FP32 ONNX** (`saved_models/asr_ft/`) for CPU execution via `sherpa-onnx`.
- **Default Hybrid Decoding (`lifemate/asr.py`, `lifemate/asr_biasing.py`)**: 16 kHz mono WAV, `8` CPU threads, `modified_beam_search` (`max_active_paths=8`), **`hotwords="auto"` (`hotwords_score=1.5`)** extracting non-leaky per-sample/per-turn BPE phrases (from `user_profile`, structured `context` entities, prior assistant turns/`tool_results`, and domain triggers), and **`post_normalize=True`** (deterministic spoken-code normalization and sample-scoped OOV entity repair) + `normalize_transcript`.

### 1.2 LLM Backbone & Deterministic Decoding (GPU, BF16 Merged)
- **Model**: `Qwen/Qwen3.5-4B` (**`Qwen3_5ForConditionalGeneration`**, loaded via `AutoModelForImageTextToText` in BF16 text-only mode with `sdpa` attention; **~4.57B total params** including ~4.0B language backbone with 32 hybrid layers — 24 GatedDeltaNet `linear_attention` + 8 `full_attention` — and frozen vision tower; **~4.64B combined with ASR**, within the 5B limit).
- **Standalone Weights**: LoRA-adapted (`r=16, alpha=32`) and merged offline into standalone BF16 weights (`saved_models/llm_ft/`) with [`lifemate_runtime.json`](saved_models/llm_ft/lifemate_runtime.json).
- **Context & Generation**: **16,384 tokens** (`max_input_tokens=16000`, `max_new_tokens=384`, never truncating schemas/history/results), native `<tool_call>` protocol, `thinking_budget=0` (`enable_thinking=False` enforced by `DirectActionLogitsProcessor` + `NativeToolStoppingCriteria`), deterministic greedy decoding (`do_sample=False`, `seed=42`).

### 1.3 Prompt, 3-Tier Caching, Agent Loop & Guards
- **Prompt & Deterministic ITN (`lifemate/prompt_session.py`, `lifemate/itn.py`, `lifemate/local_time.py`)**: Builds an untruncated JSON user payload containing `request_transcript`, deterministic Vietnamese `itn_candidates` / `itn_history_candidates`, `user_context` (with `date_reference` & `apps`), `conversation_history`, `available_tools`, and `previous_actions_and_results`.
- **3-Tier Prefix Caching (`lifemate/models.py`, `schema_cache.py`, `prefix_cache.py`)**: `PromptCache` (`auto`) reuses tokenized fixed prefixes within a sample; `SchemaCache` (`auto`) caches GPU KV states across samples sharing the same tool catalog; `PrefixCache` (`auto`) reuses intra-sample KV prefixes across agent turns, falling back cleanly to uncached generation if needed.
- **Agent Loop (`lifemate/agent.py`, `local_helpers.py`, `batching.py`)**: Budgets `max_calls=8` (`ToolEnv`), `max_invalid=3`, `max_total_steps=32`, `max_llm_steps=64`. Supports single calls, `local_batch.run` for independent reads/helpers, and stateless Python helpers (`local_time.*`, `local_context.lookup`, `local_entity.resolve`, `local_text.normalize`, `local_checks.evaluate`, `local_schema.inspect`, `local_assist.solve`).
- **Execution Guards (`lifemate/tool_execution.py`, `execution_policy.py`, `missing_info_guard.py`)**: `prepare_call_arguments` and `precheck_call` normalize empty filters / unmentioned default profile cities / guaranteed-empty retries, enforce catalog/schema validity, prevent duplicate or stagnant reads (`LoopMonitor`, `empty_read_limit=0`), block premature `sys.ask_missing_info` when values are observable, and verify `x-ref` provenance (`enforce_observed_xref=True`).

### 1.4 Training Summary
All training uses `../data/` without holding out validation on the LLM split (`--validation-fraction 0.0`, `seed=42`). GIPformer ASR is fine-tuned on `4,496` train / `540` dev utterances (`../data/train_asr.jsonl`) and exported to FP32 ONNX; Qwen3.5-4B LoRA is trained on `10,724` step-wise rows from `ToolEnv` dual-transcript replay (`audio_text` + `asr_ft`), recovery trajectories, and weighted assistant-only cross-entropy, then merged offline into `saved_models/llm_ft/`. See [training_code/README.md](training_code/README.md) for full details and reproduction commands.

---

## 2. Repository Layout (BTC Section 5)

```text
lifemate/
├── data/                              # Consolidated dataset & train_asr.jsonl
│   ├── train/
│   ├── public_test/
│   └── train_asr.jsonl
└── code/                              # Docker /code root
    ├── predict.py                     # Default BTC entrypoint (/data -> /result/submission.jsonl)
    ├── predict.sh                     # Container batch runner (/code/predict.sh)
    ├── start_jupyter.sh               # JupyterLab launcher (port 9777, --IdentityProvider.token=zac2026)
    ├── predict_notebook.ipynb         # 4-cell BTC timing notebook (/result/jupyter_submission.jsonl)
    ├── benchmark_timing.py            # CLI latency & strict submission parity verifier
    ├── run_server.sh                  # Local inference wrapper (relative ../data & saved_models/)
    ├── main.py                        # Multi-stage CLI (--stage online/full/asr/llm)
    ├── requirements.txt               # Pinned runtime + CUDA 12.8 inference dependencies
    ├── Dockerfile                     # Offline CUDA 12.8 Docker recipe
    ├── .dockerignore                  # Excludes base checkpoints (models/, saved_models/asr/, caches)
    ├── README.md                      # Solution overview (BTC Section 6.1)
    ├── lifemate/                      # Core ASR, LLM, caching, agent, helpers, guards, predictor
    ├── saved_models/                  # Inference-only weights packaged into Docker
    │   ├── asr_ft/                    # Fine-tuned GIPformer 1.5 FP32 ONNX + tokens.txt + manifest
    │   └── llm_ft/                    # Merged BF16 Qwen3.5-4B weights + lifemate_runtime.json
    ├── training_code/                 # Clean reproduction pipeline (see training_code/README.md)
    └── tests/                         # Automated test suite
```

---

## 3. Running Inference & Timing Verification

### Container Standard (BTC)
```bash
# 1. Batch inference (/data -> /result/submission.jsonl):
bash /code/predict.sh

# 2. Launch JupyterLab on port 9777 (token: zac2026, override via JUPYTER_TOKEN) & run predict_notebook.ipynb:
bash /code/start_jupyter.sh

# 3. Verify exact parity between predict.py and notebook outputs:
python3 /code/benchmark_timing.py \
  --data-root /data --split public_test \
  --output /result/jupyter_submission.jsonl \
  --compare /result/submission.jsonl --strict-compare
```

### Local Host Verification (run from `code/`)
```bash
bash predict.sh --data-root ../data --split public_test --output result/submission.jsonl
```
