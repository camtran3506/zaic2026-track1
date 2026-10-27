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
- Train labels are used only by the separate train evaluation command (or train audio_text in an explicit ablation). No public/private labeling or tuning is implemented.
- Random seeds and greedy decoding are set; deterministic GPU kernels may still depend on exact GPU, driver and library versions. Freeze versions and run repeatability checks before final Docker submission.
- The `zaic_tools` API is integrated strictly according to the prompt-provided interface; if your package exposes different structures, adapt the catalog parsing only after inspecting its *public* API/documentation (never environment files).
- For stronger models/ASR, replace local checkpoint paths but keep combined parameters <6B and LLM <5B.

## Unit tests (no competition data required)

```bash
python -m pytest -q
python -m compileall -q main.py lifemate score_train.py
```

## Submission file

Each output line is written ONLY from the real split-specific `ToolEnv` instance:

```json
{"id":"public_0001","tool_calls":[{"name":"smart_home.get_status","arguments":{}}],"state_hash":"returned-by-real-ToolEnv"}
```
