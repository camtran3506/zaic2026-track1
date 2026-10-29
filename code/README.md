# LifeMate baseline — fully offline Vietnamese voice-to-tools agent

A runnable baseline for the challenge format supplied in the problem statement. It is an **engineering baseline, not a validated competition score**: the real `zaic_tools`, audio and schemas were not provided for integration testing.

## Architecture

`All WAVs + history audio -> ASR process -> transcripts.jsonl + manifest -> ASR process exits -> LLM process reads text -> catalog-aware next-call JSON -> env.validate -> env.call -> result -> LLM (loop) -> env.get_tool_calls + env.get_state_hash`

ASR and LLM weights are never resident together during the main pipeline. `main.py`
defaults to `--stage full`: it waits for the ASR subprocess to complete and exit
before launching the LLM subprocess. The notebook runs these two stages in separate
cells/subprocesses. Per-sample time is unlimited by default (`--sample-seconds inf`);
`max_calls`, `max_invalid`, schema-inspection and token/context limits still apply.

Defaults: `Qwen/Qwen3-4B-Instruct-2507` (~4.0B, Apache-2.0) and `Systran/faster-whisper-medium` (~769M, MIT). Total model parameters ~4.77B; LLM <5B. Training and inference share a 16,384-token context limit. Inference defaults to 16,000 prompt tokens + up to 384 generated tokens per decision. Check all model/dependency licenses and freeze revisions before submission. Neither model is fetched at runtime.

## 1. Install dependencies on the server

`requirements.txt` is the shared dependency list for inference and fine-tuning.
It includes Unsloth/PEFT and pins Transformers 4.57.3 for the Unsloth stack;
PyAV 16.1.0 preserves the audio decoding API used by faster-whisper.
Install a compatible PyTorch/CUDA stack first; Unsloth training requires a supported GPU.

Tested only with Python syntax/unit tests; use Linux + CUDA 12.x + GPU for actual inference. The GPU container image supplies PyTorch. For local runs, install a CUDA-enabled PyTorch compatible with your GPU first.

```bash
cd code
python -m pip install -r requirements.txt
```

## 2. Download both pretrained models during development (network allowed)

```bash
python scripts/download_models.py --models-dir ./models
```

By default the script writes to `code/models/`, even when invoked from the repository root
with `python code/scripts/download_models.py`. Use `--models-dir` to override it.
`run_server.sh` uses `models/asr` and `models/llm` beside itself by default.
The directory is tracked through `.gitignore`; downloaded weights stay out of Git.

This creates:

```text
code/
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
ASR_MODEL=/mnt/weights/whisper-medium LLM_MODEL=/mnt/weights/qwen3-4b \
OUTPUT=/mnt/results/submission.jsonl bash run_server.sh
```

All paths can also be supplied via CLI as shown below.

### Dataset root is the parent of split folders

```bash
mkdir -p result
python main.py \
    --data-root /your/storage/lifemate_data \
    --split public_test \
    --asr-model /your/weights/faster-whisper-medium \
    --llm-model /your/weights/Qwen3-4B-Instruct-2507 \
    --output /your/outputs/submission.jsonl \
    --log-path /your/outputs/diagnostics.jsonl \
    --device cuda \
    --sample-seconds inf
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

### Run ASR once, then evaluate LLMs from saved transcripts

```bash
# Stage 1: no LLM checkpoint is required or loaded.
python main.py --stage asr --data-root /path/to/data --split public_test \
    --asr-model /path/to/whisper-medium-ct2 --device cuda \
    --transcript-cache result/medium_transcripts.jsonl

# Stage 2: no Whisper checkpoint is required or loaded.
python main.py --stage llm --data-root /path/to/data --split public_test \
    --transcript-cache result/medium_transcripts.jsonl \
    --llm-model /path/to/qwen --output result/submission.jsonl \
    --log-path result/diagnostics.jsonl --device cuda --llm-dtype float16
```

The cache has one row per selected sample: `id`, full current `transcript`, full
text `history`, and `asr_seconds`. All history audio is processed in advance;
the agent retains the existing last-12-turn/1700-character prompt history limits.
`--use-train-transcript` still requires `split=train`; test request audio is always
transcribed rather than read from a possible `audio_text` field.

The adjacent `.manifest.json` stores dataset/audio file stamps, IDs/order, ASR
model file stamps and absolute path, precision, beam, library versions and seed.
A checksum detects edited transcript files. Partial results stay in
`.partial.jsonl`; failures are reported in `.errors.jsonl`. The LLM refuses an
incomplete/mismatched cache. Matching partial ASR work resumes on retry. A completed
cache is reused without reloading ASR; use a different cache per ASR variant, or
`--refresh-transcripts` when deliberately replacing it. Model/audio file stamps
detect ordinary replacement; they are not cryptographic hashes of all source audio
and model weights. Keep the original data and frozen model revisions for reproduction.

With `--stage full`, the default cache is `<output-stem>.transcripts.jsonl`, and
stdout/stderr for both stages is saved to `<cache-stem>.asr.log` and
`<output-stem>.llm.log`. Diagnostics default to `<output-stem>.diagnostics.jsonl`.
An ASR failure stops the pipeline before LLM execution; it is never substituted
with empty predictions. LLM sample errors remain recorded in diagnostics.
`sample_seconds` measures the LLM stage, `asr_seconds` the cached ASR work, and
`total_sample_seconds` their sum. Cache reuse does not rerun/timestamp ASR work.

### Select one GPU for inference and fine-tuning

All CUDA entrypoints (`main.py`, `finetune.py`, `train_lora.py`, `prepare_sft.py`)
accept `--cuda-visible-devices` with **one** GPU index or GPU/MIG UUID. The mask is
set before importing Unsloth, PyTorch, or CTranslate2 and inherited by child
processes. Multiple GPU IDs such as `0,1` are rejected. Without an explicit flag,
the first entry in the inherited `CUDA_VISIBLE_DEVICES` is used, otherwise GPU 0.
An empty inherited mask or `--cuda-visible-devices -1` disables CUDA for CPU runs;
select the CPU device/backend separately as appropriate.

```bash
# From code/, select GPU 2 for both ASR and LLM.
bash run_server.sh --cuda-visible-devices 2

# Select GPU 2 for preparation and Unsloth training.
python finetune.py --cuda-visible-devices 2 \
    --data-root /path/to/data --base-model "$PWD/models/llm" \
    --asr-model "$PWD/models/asr" --output-dir "$PWD/result/finetune_gpu2"
```

The chosen physical GPU is remapped to local CUDA device 0 in each process. Do
not change model placement to `cuda:2` after selecting GPU 2. Selection does not
reserve the GPU; choose one with sufficient free VRAM. `--llm-device-map single`
is the inference default. The legacy `balanced` option is limited to the one
selected GPU through these entrypoints. ASR exits before the LLM loads.
The Kaggle notebook retains its `GPU_INDEX` setting and passes that selection
through the environment to all subprocesses.

A complete background run, using the existing training defaults:

```bash
mkdir -p logs
nohup bash -c '
  set -e
  GPU=2
  RUN_DIR="$PWD/result/finetune_$(date +%Y%m%d_%H%M%S)"
  python scripts/download_models.py
  python finetune.py --cuda-visible-devices "$GPU" \
    --data-root /path/to/data --base-model "$PWD/models/llm" \
    --asr-model "$PWD/models/asr" --output-dir "$RUN_DIR"
  bash run_server.sh --cuda-visible-devices "$GPU" \
    --data-root /path/to/data --llm-adapter "$RUN_DIR/training/adapter"
' > logs/finetune_pipeline.log 2>&1 < /dev/null &
```

Use a new/empty fine-tuning output directory for each fresh run. The shared GPU
flag is also available on `train_lora.py` when resuming an existing checkpoint.

Set `RUN_ASR=False` and `TRANSCRIPT_CACHE` to a completed
cache to repeat only LLM inference; keep the manifest beside that cache. To run
without fine-tuning set `TRAIN_LLM=False` and choose base weights or `LOCAL_ADAPTER`.

Greedy inference now stops immediately after the first complete decision JSON
(`call`, `inspect`, `done`, or the supported `tool_calls` wrapper). The stop check
only examines generated tokens, respects nested objects/arrays and escaped strings,
and ignores JSON inside thinking blocks. The generation returns the complete
decision rather than trailing explanations. Tool availability and argument schema
validation still happen in the agent; this is not constrained decoding.
`--max-new-tokens` remains a safety limit for incomplete/invalid output. No model
weights, precision, prompt or tool-call limits are changed by early stopping.

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
| `--asr-model`, `--llm-model` | local checkpoint paths | required only by their stage |
| `--stage` | full (separate subprocesses) / asr / llm | full |
| `--transcript-cache` | transcript JSONL + adjacent manifest | output stem + .transcripts.jsonl |
| `--output` | submission JSONL output | result/submission.jsonl |
| `--device` | cuda / cpu | cuda |
| `--asr-compute-type` | faster-whisper precision | float16 |
| `--llm-dtype` | LLM weight dtype | bfloat16 |
| `--asr-beam` | deterministic beam width | 1 |
| `--max-input-tokens` | maximum prompt tokens | 16000 |
| `--max-new-tokens` | per-agent-step generated tokens | 384 |
| `--max-calls` | maximum executed tools/sample | 8 |
| `--max-invalid` | maximum invalid model plans | 3 |
| `--max-tool-details` | number of detailed tools shown per step | 35 |
| `--sample-seconds` | optional LLM cooperative budget; inf disables it | inf |
| `--llm-device-map` | single / balanced across visible GPUs | single |
| `--llm-max-memory-gib` | weight budget per GPU for balanced | 6 |
| `--log-path` | diagnostics, separate from submission | output stem + .diagnostics.jsonl |
| `--limit` | debug: process only first N | 0 (all) |

## Important limitations and compliance notes

- **Timing**: there is no per-sample timeout by default. A finite `--sample-seconds` applies only to the LLM stage and is cooperative, not a hard interrupt. Offline ASR-precomputation experiments do NOT establish compliance with a competition's end-to-end 30-second rule.
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
unquantized LoRA and optimized LoRA kernels (dropout=0). The defaults are
`--quantization none --max-length 16384 --batch-size 32`.
Gradient accumulation remains 8, giving an effective batch size of 256 on one GPU.
Install a CUDA-enabled PyTorch
suitable for your GPU first, then:

```bash
cd code
python -m pip install -r requirements.txt

# One command: stage base weights, prepare dialogues, train/evaluate, save adapter.
# A local model path works fully offline. A HF ID downloads the resolved revision first.
python finetune.py \
  --data-root /path/to/lifemate_data \
  --base-model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir /path/to/runs/toolcalling_v1 \
  --backend unsloth --epochs 3 --quantization none --max-length 16384 \
  --batch-size 32 --gradient-accumulation 8 --learning-rate 2e-4
```

LoRA updates attention/MLP projections, with gradient checkpointing and BF16 on
supporting GPUs (FP16 otherwise on CUDA). No 4-bit/8-bit weight quantization is
enabled by default. `--quantization 4bit` remains an explicit QLoRA option.
An unquantized batch of 32 at up to 16,384 tokens requires substantial VRAM; this
configuration has not been validated to fit an RTX A5000/T4 or this CPU environment.
Padding is to the longest example in each batch. The shared context limit is defined in `lifemate/context.py`: train defaults to
16,384 total tokens (prompt + target), and inference to `--max-input-tokens 16000`
plus `--max-new-tokens 384`. Chat template/special tokens count toward these budgets.
For longer JSON outputs, an explicit alternative is 15,360 input + 1,024 output;
the CLI and model wrapper reject nonpositive budgets or a sum above 16,384 before
loading models. Budgets are never silently clamped, and prompts are never silently
truncated. Training examples need not use a fixed prompt/target split, but their
combined token count must fit the same context limit. Reduce `--max-length` only if examples fit: preparation/training rejects oversized examples instead of
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
inference. `tool_index` retains all allowed tools; `--max-tool-details` only controls
the initial detailed schemas. If a gold tool is absent from that initial selection,
SFT adds an assistant `{"action":"inspect","name":"app.tool"}` decision before the
call. Inference supports the same action: it reveals the requested allowed schema
without calling ToolEnv or changing application state. Size fitting protects that
schema's complete parameters/required/enum definitions; actual calls are validated
against the schemas in the final prompt. Unavailable/schema-invalid gold labels
still fail with the sample ID. Training transcripts are required; pass `--asr-model`
for records/history that need ASR. No synthetic tool results or gold arguments are
inserted into earlier prompts. Loss masks system/user/padding tokens and trains only
the assistant JSON and its end-of-turn tokens.

The notebook defaults to `TRAIN_MISSING_TOOL_POLICY="error"`: an out-of-ranking
gold tool is recovered through a supervised inspection step, not filtered out.
Gold arguments/schema are absent from the first prompt; the schema is provided
only after the assistant requests it, using the same mechanism as inference.
The old `--missing-tool-policy skip` remains an explicit diagnostic fallback for
an unrecoverable missing-schema error, with skipped IDs/reasons in the manifest.
Oversized inference prompts now fail explicitly instead of silently truncating
schemas: raise `--max-input-tokens` within the model's supported context and available
VRAM if necessary. The SFT `--max-length` budget includes prompt and target tokens.

Outputs: `sft/{train,validation}.jsonl`, split/hash manifest, `training/adapter/`,
checkpoints (optimizer/scheduler/RNG), `training/{train,eval}_results.json`,
`training/trainer_state.json`, training config, resolved base-model SHA (for HF IDs),
and logs for every stage.

### Best-checkpoint selection: state match + tool-call F1

The default selection metric is **`eval_state_tool_score` (0–100), higher is better**,
not teacher-forced `eval_loss`. After each epoch, the current model runs the same
agent/JSON decoding used for inference on every held-out TRAIN sample ID (10% by
default, split before SFT expansion). ASR is not loaded during these rollouts:
validation uses the train transcript/history resolved during preparation, so this
measures the tool agent rather than end-to-end audio accuracy.

Preparation replays gold calls through the train-specific `ToolEnv` and saves the
result of `env.get_state_hash()`. If a gold hash is supplied in labels, it must
agree with that replay. Each candidate prediction starts with a new `ToolEnv` for
the same sample; only its `get_tool_calls()` and `get_state_hash()` are scored.
No environment state files are opened directly. Gold calls/hashes are never
passed to the model, and no public/private test labels are used.

For a nonempty gold list, a hash mismatch or zero matched calls scores 0;
otherwise the sample scores `2*M / (len(submitted) + len(gold))`. Matching is
maximum one-to-one and ignores call order, while replay still preserves order.
Unmatched submitted calls, including schema-invalid entries in the environment's
call log, are penalized. Internal invalid proposals that are not submitted are
not added to that log. `sys.ask_missing_info` is a normal scored call. For empty
gold, score 1 only for an empty submitted list. The selection metric is **100 ×
the mean over original validation samples**, not over SFT steps or tool calls.

Fuzzy arguments use token-F1 **strictly greater than 0.6**, lowercase + Unicode
NFC, with whitespace tokenization. Non-fuzzy arguments are compared exactly.
The organisers' fuzzy tokenizer and authoritative replay may differ; this is a
local validation metric following the supplied rules, not an official test score.

Both evaluation and checkpoint saving default to `--eval-strategy epoch`.
Use `--eval-strategy steps --save-steps 100` to evaluate/save every 100 optimizer
steps instead; the final step is also evaluated. Full rollouts can add substantial
time. At most two checkpoints are retained under Trainer's best/latest handling;
ties retain the first best checkpoint. The best model is reloaded before saving
`training/adapter/`. No model is selected by loss as a fallback.

Audit files: `sft/validation_rollouts.jsonl` (inputs, replayed gold hashes/calls),
`training/validation_rollouts/*.jsonl` (per-sample predictions/hashes/scores),
`training/validation_scores.jsonl`, and `training/training_config.json`
(`selection_metric`, `best_state_tool_score`, `best_checkpoint`). These contain
private train text/tool results. All held-out IDs must be evaluated: skipped
validation samples are rejected, and unexpected runtime/evaluation failures stop
the run rather than silently reducing the denominator. Old SFT folders without
rollout metadata must be regenerated; old loss-selected checkpoints cannot be
resumed with the new selection settings.

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

Adapter loading is offline and requires PEFT. Inference loads the base model + LoRA
in the selected inference dtype; training batch size and sequence length do not
change the inference settings.

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
  --backend unsloth --epochs 3 --quantization none --max-length 16384
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
