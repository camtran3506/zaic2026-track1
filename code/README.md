# LifeMate baseline — fully offline Vietnamese voice-to-tools agent

A runnable baseline for the challenge format supplied in the problem statement. It is an **engineering baseline, not a validated competition score**: the real `zaic_tools`, audio and schemas were not provided for integration testing.

## Architecture

`16-kHz Vietnamese WAV -> local faster-whisper ASR -> Qwen3-4B-Instruct-2507 -> catalog-aware next-call JSON -> env.validate -> env.call -> result -> LLM (loop) -> env.get_tool_calls + env.get_state_hash`

Defaults: `Qwen/Qwen3-4B-Instruct-2507` (~4.0B, Apache-2.0) and `Systran/faster-whisper-small` (~244M, MIT). Total model parameters ~4.24B; LLM <5B. LLM context input is capped at 6,000 tokens by default (<16,384). Check all model/dependency licenses and freeze revisions before submission. Neither model is fetched at runtime.

## 1. Install dependencies on the server

Tested only with Python syntax/unit tests; use Linux + CUDA 12.x + GPU for actual inference. The GPU container image supplies PyTorch. For local runs, install a CUDA-enabled PyTorch compatible with your GPU first.

```bash
cd lifemate_baseline
python -m pip install -r requirements.txt
```

## 2. Download both pretrained models during development (network allowed)

```bash
python scripts/download_models.py --models-dir ./models
```

This creates:

```text
lifemate_baseline/
├── main.py
├── lifemate/
├── scripts/
├── models/
│   ├── asr/  # Whisper converted to CTranslate2 (model.bin etc.)
│   └── llm/  # Qwen Transformers weights and tokenizer
├── Dockerfile
└── requirements.txt
```

For final scoring, keep **both** model folders inside the Docker image. The download script is NOT invoked during inference.

## 3. Customize paths and run public_test

**Simplest way to configure server paths:** edit the first variables in `run_server.sh`, or use environment variables:

```bash
DATA_ROOT=/mnt/datasets/LifeMate SPLIT=public_test \
ASR_MODEL=/mnt/weights/whisper-small LLM_MODEL=/mnt/weights/qwen3-4b \
OUTPUT=/mnt/results/submission.jsonl bash run_server.sh
```

All paths can also be supplied via CLI as shown below.

### Dataset root is the parent of split folders

```bash
mkdir -p result
python main.py \
    --data-root /your/storage/lifemate_data \
    --split public_test \
    --asr-model /your/weights/faster-whisper-small \
    --llm-model /your/weights/Qwen3-4B-Instruct-2507 \
    --output /your/outputs/submission.jsonl \
    --log-path /your/outputs/diagnostics.jsonl \
    --device cuda \
    --sample-seconds 28
```

`/your/storage/lifemate_data` contains `train/`, `public_test/`, etc.

### Dataset root IS the split folder (e.g. Docker's `/data`)

```bash
python main.py \
    --data-root /data \
    --split private_test \
    --asr-model /opt/models/asr \
    --llm-model /opt/models/llm \
    --output result/submission.jsonl
```

The program creates a temporary *symlink* `<temporary-root>/<split> -> /data` so that the split-specific `ToolEnv(split, id, root=...)` can locate its data. The solution never opens, reads, copies or parses files inside `<split>_environment/`; ONLY the organizer's `ToolEnv` accesses them.

### Smoke test on a few samples

```bash
python main.py --data-root /your/storage/lifemate_data --split public_test \
    --asr-model ./models/asr --llm-model ./models/llm \
    --output result/smoke.jsonl --limit 5
```

Remove `--limit` for official submission, which requires exactly one JSONL line per sample.

### Train evaluation (ASR ablation, NO test labels used)

```bash
python main.py --data-root /your/storage/lifemate_data --split train \
    --asr-model ./models/asr --llm-model ./models/llm \
    --use-train-transcript --output result/train_pred.jsonl

python score_train.py \
    --train-jsonl /your/storage/lifemate_data/train/train.jsonl \
    --catalog /your/storage/lifemate_data/train/train_tools_catalog.json \
    --submission result/train_pred.jsonl
```

`score_train.py` calculates a **proxy tool-call F1 only**. It cannot test the mandatory state-hash gate because gold state hashes are not in the given train record. The precise organizer fuzzy-tokenization rules may differ. Never use `--use-train-transcript` outside `train`.

## 4. Docker

Before building, stage the fully downloaded model folders into this project as `models/asr` and `models/llm`. If your models live elsewhere, copy them into the Docker build context; DO NOT download them in the container at inference.

```bash
# Build phase (network may be available for dependencies)
docker build -t lifemate:baseline .

# Container's input /data is the mounted split folder, not its parent.
mkdir -p ./result
docker run --rm --gpus '"device=0"' --network none \
    -v /your/storage/lifemate_data/public_test:/data:ro \
    -v "$(pwd)/result:/app/result" \
    lifemate:baseline \
    --data-root /data --split public_test \
    --asr-model /opt/models/asr --llm-model /opt/models/llm \
    --output /app/result/submission.jsonl
```

By default `docker run ... lifemate:baseline` targets **private_test** at `/data` and writes `/app/result/submission.jsonl`. The organizers' exact invocation/entrypoint rules supersede this example.

`--network none` verifies the inference path is offline. Model inference uses `local_files_only=True` and `HF_HUB_OFFLINE=1`.

## 5. Useful parameters

| Flag | Meaning | Default |
|---|---|---|
| `--data-root` | split folder OR its parent | mandatory |
| `--split` | train / public_test / private_test | public_test |
| `--asr-model`, `--llm-model` | local checkpoint paths | mandatory |
| `--output` | submission JSONL output | result/submission.jsonl |
| `--device` | cuda / cpu | cuda |
| `--asr-compute-type` | faster-whisper precision | float16 |
| `--llm-dtype` | LLM weight dtype | bfloat16 |
| `--asr-beam` | deterministic beam width | 1 |
| `--max-input-tokens` | maximum prompt tokens | 6000 |
| `--max-new-tokens` | per-agent-step generated tokens | 192 |
| `--max-calls` | maximum executed tools/sample | 8 |
| `--max-invalid` | maximum invalid model plans | 3 |
| `--max-tool-details` | number of detailed tools shown per step | 35 |
| `--sample-seconds` | cooperative budget (not hard interrupt) | 28 |
| `--log-path` | diagnostics, separate from submission | unset |
| `--limit` | debug: process only first N | 0 (all) |

## Important limitations and compliance notes

- **30-second rule**: this is a *soft*, per-sample cooperative deadline; CUDA generation/ASR may overrun one call. Benchmark on A30 with real audio, reduce model size, output length or loop counts if needed. This code does NOT guarantee sub-30s inference.
- `env.validate` catches schema-invalid arguments before they become logged calls; valid but semantically wrong IDs may still be rejected by `env.call`.
- The agent prompts the LLM to perform read calls before x-ref-dependent writes, but does not implement a full symbolic planner. Long/complex tasks will be imperfect.
- The model's response is constrained by a JSON-only prompt and strict parser, not constrained decoder. Malformed generations consume retries.
- No external API or online requests at inference. `state.json` is NEVER read directly.
- Fine-tuning reads only `train/train.jsonl` labels. Public/private test labels are never used. See the training workflow below.
- Random seeds and greedy decoding are set; deterministic GPU kernels may still depend on exact GPU, driver and library versions. Freeze versions and run repeatability checks before final Docker submission.
- The `zaic_tools` API is integrated strictly according to the prompt-provided interface; if your package exposes different structures, adapt the catalog parsing only after inspecting its *public* API/documentation (never environment files).
- For stronger models/ASR, replace local checkpoint paths but keep combined parameters <6B and LLM <5B.

## Fine-tune the LLM for next-tool calling with Unsloth (LoRA / QLoRA)

The training schema is the same as inference: `audio_text`, `context`, `history`, and
`label.tool_calls` (ordered `{name, arguments}` objects). Empty call lists produce a
`done` target; missing labels are errors. Only TRAIN data is accepted by the preparation
entrypoint. Keep the organizer's train-specific `zaic_tools` next to `train.jsonl`.

The default training backend is **Unsloth** (`FastLanguageModel`), imported before
Transformers/PEFT in the training process. It uses Unsloth gradient checkpointing,
QLoRA 4-bit and optimized LoRA kernels (dropout=0). Install a CUDA-enabled PyTorch
suitable for your GPU first, then:

```bash
cd code
python -m pip install -r requirements-train.txt

# One command: stage base weights, prepare dialogues, train/evaluate, save adapter.
# A local model path works fully offline. A HF ID downloads the resolved revision first.
python finetune.py \
  --data-root /path/to/lifemate_data \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir /path/to/runs/toolcalling_v1 \
  --backend unsloth --epochs 3 --quantization 4bit --max-length 8192 \
  --batch-size 1 --gradient-accumulation 8 --learning-rate 2e-4
```

QLoRA uses NF4 double quantization, LoRA on attention/MLP projections, SDPA,
gradient checkpointing, and BF16 on supporting GPUs (FP16 on T4). Start with a
16-GB+ CUDA GPU; required VRAM depends on sequence length. Reduce `--max-length`
only if examples fit: preparation/training rejects oversized examples instead of
silently truncating JSON targets. CPU LoRA (`--quantization none`) is supported for
tiny smoke tests only with `--backend transformers --quantization none`; Unsloth
requires a supported CUDA GPU and never silently falls back to another backend.
Training and inference are single
GPU by default; two T4 cards do not automatically combine VRAM.

`prepare_sft.py` splits source sample IDs deterministically before expanding steps,
preventing a sample's trajectory from leaking into validation. It replays each gold
call through the real train `ToolEnv.validate/call`; the next prompt contains only
previous real tool results. It never opens environment state files. Gold lists must
be in execution order. It uses the **same candidate ranking and prompt builder** as
inference; unavailable/schema-invalid/non-exposed targets fail with the sample ID.
Use `--max-tool-details` consistently in preparation and inference if your gold tools
need a larger candidate set. Training transcripts are required; pass `--asr-model`
for records/history that need ASR. No synthetic tool results or gold arguments are
inserted into earlier prompts. Loss masks system/user/padding tokens and trains only
the assistant JSON and its end-of-turn tokens.

If a valid gold tool is missing from the final prompt (candidate ranking or prompt
size pruning), CLI preparation stops by default. Add `--missing-tool-policy skip`
to exclude the **entire sample**, including all its steps, with a warning and an
entry in `sft/train_skipped.jsonl` or `sft/validation_skipped.jsonl`. The Kaggle notebook
defaults to `TRAIN_MISSING_TOOL_POLICY="skip"`; set it to `"error"` for strict mode.
The manifest records retained IDs, source counts and skipped reasons separately
for each partition. Validation loss then describes only the retained validation
samples; this filtering does not fix candidate recall at inference time. No gold
tool is injected into the prompt, and invalid labels still cause preparation to fail.

Outputs: `sft/{train,validation}.jsonl`, split/hash manifest, `training/adapter/`,
checkpoints (optimizer/scheduler/RNG), `training/{train,eval}_results.json`,
`training/trainer_state.json`, training config, resolved base-model SHA (for HF IDs),
and logs for every stage. Held-out teacher-forced `eval_loss` is **not tool-call F1**
or an official competition score. Judge quality with real end-to-end runs on held-out
train IDs, comparing the same IDs/base model/settings; a completed training run alone
does not establish improved accuracy.

### Run the main task with the fine-tuned adapter

Keep the **exact base model** used for training. The adapter folder includes its tokenizer:

```bash
mkdir -p result
python main.py \
  --data-root /path/to/lifemate_data --split public_test \
  --asr-model /path/to/asr \
  --llm-model /path/to/exact-base-model \
  --llm-adapter /path/to/runs/toolcalling_v1/training/adapter \
  --llm-dtype float16 --device cuda \
  --output result/submission.jsonl --log-path result/diagnostics.jsonl \
  > result/inference.log 2>&1
```

Adapter loading is offline and requires PEFT. Inference loads full-precision base + LoRA, so QLoRA's
low training VRAM is not a promise about inference VRAM.

### Export a standalone model and resume training

```bash
# Can also request --export-merged in finetune.py. CPU merge needs RAM/disk for
# the full base model; run it separately if training GPU memory is tight.
python merge_lora.py --base-model /path/to/exact-base-model \
  --adapter /path/to/runs/toolcalling_v1/training/adapter \
  --output-dir /path/to/runs/toolcalling_v1/merged --dtype float16

# Standalone inference: use merged as --llm-model and omit --llm-adapter.
python train_lora.py --base-model /path/to/exact-base-model \
  --sft-dir /path/to/runs/toolcalling_v1/sft \
  --output-dir /path/to/runs/toolcalling_v1/training \
  --resume-from-checkpoint /path/to/runs/toolcalling_v1/training/checkpoint-100 \
  --backend unsloth --epochs 3 --quantization 4bit --max-length 8192
```

Resume with the same model, data, optimizer/batch/LoRA settings as the original run;
change the total epoch/step budget as needed. New pipeline runs require an empty
directory to avoid overwriting previous artifacts. Generated training datasets contain
transcripts and tool results: keep them private. Merge does not require PEFT during
later inference; keep both merged LLM and ASR inside the final offline image.

The standalone Kaggle notebook in `../notebooks/LifeMate_Kaggle.ipynb` includes
these training scripts and can train first, then pass the saved adapter into inference.
`TRAIN_LLM=True` and `TRAIN_BACKEND="unsloth"` are the notebook defaults. Dependencies
pin Unsloth 2025.12.10, unsloth-zoo 2025.12.8 and compatible Transformers 4.57.3;
the notebook constrains Kaggle's existing Torch/Torchvision/Triton stack during pip
resolution. If that stack is incompatible, installation stops with the resolver's
error rather than replacing CUDA packages silently. Full Unsloth GPU training still
needs validation on the actual Kaggle GPU; CPU tests exercise the shared SFT pipeline
and the explicit Transformers backend.

## Unit tests (no competition data required)

```bash
python -m pytest -q
python -m compileall -q main.py lifemate score_train.py
```

With training dependencies installed, tests also run a tiny randomly initialized
Qwen through actual LoRA updates, checkpoint resume, offline adapter generation,
and merged-model equivalence. They do not download models or establish real dataset accuracy.

## Submission file

Each output line is written ONLY from the real split-specific `ToolEnv` instance:

```json
{"id":"public_0001","tool_calls":[{"name":"smart_home.get_status","arguments":{}}],"state_hash":"returned-by-real-ToolEnv"}
```
