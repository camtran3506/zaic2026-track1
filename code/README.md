# LifeMate baseline — fully offline Vietnamese voice-to-tools agent

A runnable baseline for the challenge format supplied in the problem statement. It is an **engineering baseline, not a validated competition score**: the real `zaic_tools`, audio and schemas were not provided for integration testing.

## Architecture

`All WAVs + history audio -> ASR process -> transcripts.jsonl + manifest -> ASR process exits -> LLM process reads text -> catalog-aware next-call JSON -> env.validate -> env.call -> result -> LLM (loop) -> env.get_tool_calls + env.get_state_hash`

ASR and LLM weights are never resident together during the main pipeline. `main.py`
defaults to `--stage full`: it waits for the ASR subprocess to complete and exit
before launching the LLM subprocess. The notebook runs these two stages in separate
cells/subprocesses. Per-sample time is unlimited by default (`--sample-seconds inf`);
`max_calls`, `max_invalid` and token/context limits still apply.

Defaults: `Qwen/Qwen3.5-4B` (~4.0B, Apache-2.0) and `g-group-ai-lab/gipformer1.5-68M-rnnt` (~68M). Total model parameters ~4.07B; LLM <5B. Training and inference share a 16,384-token context limit. Inference defaults to 16,000 prompt tokens + up to 384 generated tokens per decision. Check all model/dependency licenses and freeze revisions before submission. Neither model is fetched at runtime.

Qwen3.5 uses its native conditional-generation checkpoint through
`AutoModelForImageTextToText`, with text-only prompts. The same loader is used for
inference, Transformers fine-tuning and adapter merge. The chat template uses
`enable_thinking=False` consistently in inference and SFT for direct native tool calls
within the existing token budget. Unsloth uses
`FastVisionModel`; LoRA targets only language attention/MLP projections. Stage
Qwen3.5 weights in a fresh directory and train a new adapter; Qwen3 adapters
are incompatible. Other inference and training budgets are unchanged.

### Shared inference settings and artifact identity

Inference and merge default to **BF16** (`bfloat16`), including `--export-merged`.
GPU training and the notebook require BF16-capable hardware; unsupported GPUs
raise an explicit error. CPU smoke tests use FP32. Explicit inference/merge dtype
overrides remain available for controlled comparisons.

`lifemate.llm` supplies the shared chat-template kwargs, SFT tokenization and
portable `render_prompt(tokenizer, messages)` (rendered text plus exact input IDs).
`lifemate.runtime` defines neutral greedy decoding: temperature 0, repetition
penalty 1, zero presence/frequency penalties, explicit EOS/pad IDs, seed and the
existing 16000/384/16384 token budgets. Transformers uses a fresh GenerationConfig
with `do_sample=False`; sampling defaults from the model repository are not
inherited. Request budget/seed overrides remain supported.

New adapters, Trainer checkpoints and merged models include
`lifemate_runtime.json` plus a neutral Transformers `generation_config.json`.
The JSON sidecar records BF16 defaults, `enable_thinking=False` for Qwen3.5,
the local date/time tool policy, prompt settings, decoding policy and SHA-256 fingerprints of the checkpoint
(adapter base or exported merged model), tokenizer rules/vocabulary, special
tokens and chat template. Inference
and merge check these identities before loading model weights. Paths may change
without invalidating the identity. Verification reads weight files once at
startup/export; it does not run per token.

An adapter must include its saved tokenizer: missing tokenizer files no longer
fall back to the base tokenizer. Legacy adapters without the sidecar can still
run, but emit a warning that identity and local time tool training are unverified.
An existing sidecar without the current local tool policy is rejected: regenerate
SFT and retrain rather than silently mixing prompt/tool capabilities.
Old checkpoints without the
recorded runtime identity cannot be resumed as verified runs; start a new run.

The runtime remains Transformers/PyTorch. A future engine adapter must explicitly
consume this sidecar, matching prompt IDs and sampling settings; placing the
file beside a model does not automatically configure another engine. Decision
stopping callbacks, LoRA registration and native tool-call protocol mapping
remain work for that migration.

## Native function calling for Qwen3.5-4B

For [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), inference and SFT
use the **checkpoint tokenizer's own chat template** with `tools=...` and
`enable_thinking=False`. Each tool is passed as
`{"type":"function","function": <complete catalog schema>}`; all schema fields
are retained. Prior calls become assistant `tool_calls`, and their real results
become `role="tool"` messages. The native template renders `<tools>`,
`<tool_call>`, `<function=...>`, `<parameter=...>` and `<tool_response>`.
This matches the `qwen3_coder` format used for Qwen3.5 tool serving; see also
[Qwen-Agent's Qwen3.5 example](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/assistant_qwen3.5.py).

SFT JSONL still stores the portable LifeMate payload and action target. At
**tokenization**, the shared adapter turns call targets into structured assistant
`tool_calls`; it does not teach the model the custom action-call JSON format.
Inference uses the same conversion, including every history turn and result.
Normalized historical text is wrapped in JSON to preserve its content and literal
reasoning markers through the native template. Only assistant target tokens
(including native call delimiters and end-of-turn) receive training loss.

Native output is parsed using the declared parameter types. Strings stay strings,
including numeric-looking IDs; JSON objects/arrays and numeric/bool/null values
are decoded without argument repair. `ToolEnv.validate` remains authoritative.
Generation waits for the entire `</tool_call>` before stopping, so JSON inside
parameters cannot prematurely stop a call. One tool is executed per decision;
completion remains the explicit `{"action":"done"}` control marker.

Other model families keep the portable JSON-decision protocol. Runtime manifests
record `qwen35-native-fc-v1` for Qwen3.5; an old Qwen3.5 JSON-protocol adapter is
rejected instead of silently mixing protocols. Regenerate SFT in a fresh directory
and retrain that adapter. The current preparation policy is
`full-context-python-automatic-guards-v9`. Production never replaces the checkpoint template;
tests use a pinned upstream Qwen3.5 template fixture without downloading weights.

## Internal date and time tools

The agent exposes three additional pure Python tools alongside the tools allowed
by each sample's `installed_apps`:

| Internal tool | Inputs | Result |
| --- | --- | --- |
| `local_time.calendar_reference` | No arguments; uses the sample's `context.current_time` and optional `context.timezone` | Today/yesterday/tomorrow/day-after-tomorrow and Monday–Sunday dates for the previous, current and next weeks |
| `local_time.departure_time` | Offset-aware ISO `arrival`, integer `travel_minutes` and `early_minutes` | Offset-aware `departure` timestamp |
| `local_time.duration_between` | Offset-aware ISO `start` and `end` | Nonnegative whole-minute `minutes` |

For a 09:00 appointment, 15 minutes of travel and arrival 15 minutes early,
`departure_time` returns 08:30. `duration_between` returns 40 minutes between
19:30 and 20:10. Time arithmetic uses elapsed time through UTC and converts the
departure back to the supplied sample timezone when present. The calendar uses
local calendar dates. It never reads the machine clock, guesses a missing date
or timezone, or silently interprets a naive timestamp. Invalid input is reported
to the agent; durations/offsets are bounded to seven days.

These names belong to a separate local registry. Local calls are validated and
executed in Python, not sent to `ToolEnv.validate` or `ToolEnv.call`. Their
results return to the model as ordinary observations/native tool responses.
Application calls retain their original sample whitelist, schema validation and
ToolEnv execution. Submission and evaluation still use the unmodified
`env.get_tool_calls()` and `env.get_state_hash()`; local calls do not appear in
the official call list or alter application state. Local results do not provide
application IDs or replace successful application reads for `x-ref` parameters.

One model output calls one tool. A helper result therefore requires another
generation before an application call can use it. Helpers have a separate
default budget of eight calls per sample, with a total limit of 32 agent steps;
the existing application-call and invalid-proposal limits still apply. Internal
calls are recorded in diagnostic traces/stats so their latency and failure rates
can be inspected even though they are absent from submission calls.

SFT preparation can insert local call/result steps before a gold application
action when temporal inputs can be extracted conservatively from the request,
context, history and earlier actual tool observations. Helper inputs are not
copied from future gold arguments/results. The original gold application calls
and replayed final state remain unchanged. Unrecognized or ambiguous language
does not produce a guessed arithmetic training target. The helper examples use
the same assistant-only loss mask and native tool serialization as inference.
Each partition's manifest reports `local_helper_examples` and
`local_helper_examples_by_tool`; inspect these counts after preparation to check
coverage for the chosen clean/ASR mixture. Helper demonstrations are conditional,
so adding schemas does not guarantee that every input supplies all three targets.

Regenerate prepared SFT in a fresh directory: the manifest must record
`prompt_policy=full-context-python-automatic-guards-v9` and
`local_time_policy=local-time-tools-v1` and
`local_helper_policy=python-specialists-itn-v2` and
`itn_policy=vi-explicit-spans-v1` and
`execution_policy=automatic-tool-guards-v1`. Existing runtime artifacts must carry
these same local policies; adding tool schemas alone does not establish that an old
adapter learned to use them. Unit and fixture tests cover these paths; real
Qwen3.5 GPU training/inference and accuracy improvements still require validation.

## Python specialist helpers

Before each model request, shared SFT/inference preprocessing supplies rule-based
ITN candidates for explicit phone/email/VND expressions, including text history.
The normalized transcript stays intact; annotations retain exact source spans.
The model can use them without a helper turn, or call `local_text.normalize`
(`email`, `phone`, `money`) for an unmarked span. Detection is cached across steps.
Rebuild prepared SFT and retrain for the new policy; valid ASR caches can be reused.

The same Qwen model remains the planner. All specialists are bounded standard-library
Python functions; no additional model, framework package or remote API is required.
See [HELPERS.md](HELPERS.md) for inputs, outputs, evidence rules, SFT coverage and
links to the framework source code that informed this design.

| Role | Implementation |
| --- | --- |
| Context/history retrieval | `local_context.lookup` |
| Entity resolution with provenance | `local_entity.resolve` |
| Calendar and time arithmetic | The three `local_time.*` tools above |
| Spoken email/phone/money normalization | `local_text.normalize` |
| Advisory tool/schema routing | `local_schema.inspect` and exposed-catalog checks |
| Numeric conditions and result verification | `local_checks.evaluate` and automatic `verify_result` |
| Retry/recovery feedback | Automatic `LoopMonitor` before/after application calls |

Entity resolution only indexes context and historical successful read results.
It cannot search ToolEnv or read hidden state. An unmatched name remains unresolved;
ambiguous candidates keep their original names/IDs and evidence. Current-turn
application read results remain visible to the planner through ordinary observations.
No helper rewrites application arguments or deletes official calls.

SFT adds conservative demonstrations from visible inputs and prior actual results,
with the same local execution and assistant-only loss path used at inference.
Schema-repair demonstrations require a prior rejected proposal; result verification
and retry monitoring also run automatically. Rebuild prepared SFT and retrain an
adapter for policy `full-context-python-automatic-guards-v9`. Manifests with a different
helper policy are rejected; legacy adapters without a manifest emit an explicit
warning and have no verified helper training. Accuracy and added
LLM-turn latency must be measured on held-out Qwen GPU evaluation.

## 1. Install dependencies on the server

`requirements.txt` is the shared dependency list for inference and fine-tuning.
It includes Unsloth/PEFT and pins Transformers 5.4.0 for the Unsloth stack;
GIPformer uses `sherpa-onnx` (ONNX Runtime CPU) and `soundfile` for audio decoding.
Install a compatible PyTorch/CUDA stack for the LLM first; Unsloth training requires a supported GPU. ASR itself runs on CPU and does not import PyTorch.

The ASR runtime is CPU-only; use Linux + CUDA 12.x + GPU for the standard LLM workflow. The GPU container image supplies PyTorch. For local runs, install a CUDA-enabled PyTorch compatible with your GPU first.

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
`run_server.sh` loads the local folders configured by its `ASR_MODEL` and
`LLM_MODEL` assignments (`models/asr` and `models/llm` beside itself).
The directory is tracked through `.gitignore`; downloaded weights stay out of Git.

This creates:

```text
code/
├── main.py
├── lifemate/
├── scripts/
├── models/
│   ├── asr/  # GIPformer encoder/decoder/joiner.int8.onnx + tokens.txt
│   └── llm/  # Qwen Transformers weights and tokenizer
├── Dockerfile
└── requirements.txt
```

For final scoring, keep **both** model folders inside the Docker image. The download script is NOT invoked during inference.

GIPformer uses the local ONNX transducer artifacts from
[g-group-ai-lab/gipformer1.5-68M-rnnt](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt):
`encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt` by default.
Use `python scripts/download_models.py --only asr --asr-precision fp32` to stage
`encoder.onnx`, `decoder.onnx`, `joiner.onnx`, then pass `--asr-precision fp32`
to inference/preparation (or download `--asr-precision both` to stage both).
No Whisper/CTranslate2 conversion is used. Missing files for the requested
precision are errors; the code never silently selects another precision/model.

The CPU recognizer uses `OfflineRecognizer.from_transducer`, 16 kHz features,
80 feature dimensions and greedy decoding. SoundFile decodes PCM to float32,
channels are averaged to mono, and `accept_waveform` receives the original
sample rate so sherpa-onnx resamples internally if needed. Empty, nonfinite and
out-of-range audio is rejected. Recognizer output uses the shared utterance
normalization below before caching and model input. `--asr-threads 4` controls CPU concurrency independently
of `--device` / CUDA GPU selection for the LLM.

Example ASR-only run (no LLM weights or Torch/CUDA needed for this stage):

```bash
python main.py --stage asr --data-root /path/to/data --split train \
    --asr-model ./models/asr --asr-precision int8 --asr-threads 4 \
    --transcript-cache /path/to/new/train.transcripts.jsonl
```

`--asr-compute-type` and `--asr-beam` from the Whisper backend are removed.
Rebuild old Whisper/PhoWhisper caches with `--refresh-transcripts` in ASR/full
mode or choose a new cache path. LLM-only mode checks the ASR source before
loading the LLM; it does not require the ASR weight directory to remain present.
The Dockerfile checks the default int8 artifacts; include that set when building.


## 3. Customize paths and run public_test

**Model folders for the server launcher are configured only in `run_server.sh`:**

```bash
ASR_MODEL="${SCRIPT_DIR}/models/asr"
LLM_MODEL="${SCRIPT_DIR}/models/llm"
```

Edit these assignments to change folders. The launcher ignores inherited
`ASR_MODEL`/`LLM_MODEL` values and appends these two model-path arguments last, so
even a legacy `--asr-model`/`--llm-model` option cannot select another folder.
It logs both configured paths; Python logs the resolved paths, stage and PID.
Changing the repo in `scripts/download_models.py` changes which weights are
downloaded, not the inference path. To download ASR without touching the base LLM:

```bash
python scripts/download_models.py --only asr
```

Both scripts use `code/models/{asr,llm}` regardless of the caller's working
directory. If you change the configured folders, stage the matching checkpoints
there. ASR must contain a complete GIPformer ONNX transducer export; the launcher does not search
old folders or download a fallback model. The adapter remains an explicit
`--llm-adapter /path/to/training/adapter` argument; keep the correct base LLM.

Dataset, output and device settings still accept environment variables or CLI:

```bash
DATA_ROOT=/mnt/datasets/LifeMate SPLIT=public_test \
OUTPUT=/mnt/results/submission.jsonl bash run_server.sh
```

For per-command model-path selection, invoke `python main.py` directly as shown
below. The CLI, Docker and notebook keep their own explicit local model paths;
this launcher policy does not change fine-tuning or adapter files.

### Dataset root is the parent of split folders

```bash
mkdir -p result
python main.py \
    --data-root /your/storage/lifemate_data \
    --split public_test \
    --asr-model /your/weights/gipformer1.5-68M-rnnt \
    --llm-model /your/weights/Qwen3.5-4B \
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
    --asr-model /path/to/gipformer-onnx --device cuda \
    --transcript-cache result/medium_transcripts.jsonl

# Stage 2: no GIPformer checkpoint is required or loaded.
python main.py --stage llm --data-root /path/to/data --split public_test \
    --transcript-cache result/medium_transcripts.jsonl \
    --llm-model /path/to/qwen --output result/submission.jsonl \
    --log-path result/diagnostics.jsonl --device cuda --llm-dtype bfloat16
```

The cache has one row per selected sample: `id`, full current `transcript`, full
text `history`, and `asr_seconds`. All history audio is processed in advance.
SFT and inference share the same history resolver: audio-only turns are
transcribed for every role, empty turns are filtered, and **all usable turns and
full text** reach the prompt. There is no turn window or character limit. Paired
assistant `tool_calls`/`tool_results` are preserved as structured history,
including tool-only turns. Tool results are not clipped. The on-disk transcript
cache retains this same full history, so compatible caches can be reused.
Regenerate older SFT datasets in a new output directory: shortened data cannot
be recovered by the trainer. Training and rollout evaluation reject manifests
without the current `prompt_policy: full-context-python-automatic-guards-v9` marker.
`--use-train-transcript` still requires `split=train`; test request audio is always
transcribed rather than read from a possible `audio_text` field.

The adjacent `.manifest.json` stores dataset/audio file stamps, IDs/order, ASR
model file stamps and absolute path, ONNX precision, CPU threads, sample rate, decoder and runtime library versions.
A checksum detects edited transcript files. Partial results stay in
`.partial.jsonl`; failures are reported in `.errors.jsonl`. The LLM refuses an
incomplete/mismatched cache. Matching partial ASR work resumes on retry. A completed
cache is reused without reloading ASR; use a different cache per ASR variant, or
`--refresh-transcripts` when deliberately replacing it. Model/audio file stamps
detect ordinary replacement; they are not cryptographic hashes of all source audio
and model weights. Keep the original data and frozen model revisions for reproduction.

### Shared transcript normalization

ASR output, TRAIN `audio_text` and the text of every history turn use the same
`lifemate.text.normalize_transcript` function in finetuning and inference:

1. Normalize Unicode to NFC.
2. Lowercase the whole turn; collapse Unicode whitespace to single spaces and
   strip leading/trailing whitespace.
3. Capitalize only the first alphabetic character of the turn, including after
   leading quotes/digits. Later sentences remain lowercase.
4. Append a period if terminal punctuation is absent. Existing `.`, `?`, `!` or
   `…` endings (also inside closing quotes/brackets) are kept without an extra dot.
   Empty/whitespace-only text stays empty.

For example, `  BẬT\tĐÈN PHÒNG KHÁCH  ` becomes `Bật đèn phòng khách.` and
`TẮT ĐÈN. ĐẶT BÁO THỨC.` becomes `Tắt đèn. đặt báo thức.`. Existing internal
punctuation, diacritics and numeric separators are preserved; no turn or content
is truncated. The function does not infer sentence boundaries, split joined words, correct
ASR errors or convert spoken numbers. Case changes also apply to names/emails
inside utterance text. Structured tool schemas, arguments, results, context and
SFT target calls remain unchanged. No extra model or tool call is needed.

Normalization is idempotent and applies at ASR, cache and prompt boundaries, so
both sources in the existing 50:50 mixture and direct/cached inference use the
same style. Original dataset files are not modified. After rendering the chat
template, inference still tokenizes with `add_special_tokens=False`, matching SFT.

Transcript caches now use manifest **version 4**, recording
`text_normalization: nfc-whitespace-lower-initial-terminal-v1`. Rebuild old caches
with `--refresh-transcripts` in ASR/full mode or choose a new path. LLM-only mode
rejects stale caches, and incompatible partial prefixes cannot be resumed into
new ones. Regenerate existing SFT in a fresh directory and retrain adapters;
prepared-data and runtime manifests verify the normalization policy rather than
silently mixing old and new input styles.

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
set before importing Unsloth or PyTorch and inherited by child
processes. Multiple GPU IDs such as `0,1` are rejected. Without an explicit flag,
the first entry in the inherited `CUDA_VISIBLE_DEVICES` is used, otherwise GPU 0.
An empty inherited mask or `--cuda-visible-devices -1` disables CUDA for CPU runs;
select the CPU device/backend separately as appropriate.

```bash
# From code/, select GPU 2 for both the LLM; GIPformer ASR remains on CPU.
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
(`call`, `done`, or the supported `tool_calls` wrapper). The stop check
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
| `--device` | LLM cuda / cpu; ASR always uses CPU | cuda |
| `--asr-precision` | GIPformer ONNX int8 / fp32 | int8 |
| `--llm-dtype` | LLM weight dtype | bfloat16 |
| `--asr-threads` | GIPformer CPU threads; greedy decoding | 4 |
| `--max-input-tokens` | maximum prompt tokens | 16000 |
| `--max-new-tokens` | per-agent-step generated tokens | 384 |
| `--max-calls` | maximum executed tools/sample | 8 |
| `--max-invalid` | maximum invalid model plans | 3 |
| `--sample-seconds` | optional LLM cooperative budget; inf disables it | inf |
| `--llm-device-map` | single / balanced across visible GPUs | single |
| `--llm-max-memory-gib` | weight budget per GPU for balanced | 6 |
| `--log-path` | diagnostics, separate from submission | output stem + .diagnostics.jsonl |
| `--limit` | debug: process only first N | 0 (all) |

## Important limitations and compliance notes

- **Timing**: there is no per-sample timeout by default. A finite `--sample-seconds` applies only to the LLM stage and is cooperative, not a hard interrupt. Offline ASR-precomputation experiments do NOT establish compliance with a competition's end-to-end 30-second rule.
- `env.validate` catches schema-invalid arguments before execution. Under the organizer contract, nonexistent IDs and insufficient resources are business no-ops (`ok:true, done:false`), not schema refusals. Every executed call remains in the official log.
- The model's response is constrained by a JSON-only prompt and strict parser, not constrained decoder. Malformed generations consume retries.
- No external API or online requests at inference. `state.json` is NEVER read directly.
- Fine-tuning reads only `train/train.jsonl` labels. Public/private test labels are never used. See the training workflow below.
- Random seeds and greedy decoding are set; deterministic GPU kernels may still depend on exact GPU, driver and library versions. Freeze versions and run repeatability checks before final Docker submission.
- The `zaic_tools` API is integrated strictly according to the prompt-provided interface; if your package exposes different structures, adapt the catalog parsing only after inspecting its *public* API/documentation (never environment files).
- For stronger models/ASR, replace local checkpoint paths but keep combined parameters <6B and LLM <5B.

### ToolEnv argument validation

Before `env.call`, the agent checks tool availability and argument-object shape,
then passes the original arguments directly to `env.validate`. Only a dictionary
response with `ok` set to true permits execution. Invalid arguments or validation
exceptions are returned to the model as observations so it can correct its proposal.
Each rejected proposal preserves its rejection stage, attempted tool name and
arguments, and the complete validation response or exception details. Parse
failures preserve the original model output and ask for the format required by
the system/chat template, including native FC for Qwen3.5. Under native FC,
rejections are `agent_feedback` user messages; they are never represented as an
executed tool call or tool result. Original values remain unmodified.
Pending parse/precheck/validation feedback stays visible while the model retries.
After validation succeeds and `env.call` is attempted and counted, all pending
rejection feedback is removed from subsequent planner prompts. All executed
calls/results remain, including `ok: false` returns and call exceptions. The
cumulative invalid-proposal count and the full debug trace retain earlier errors.
The automatic Python guard first checks tool names/sample permissions and then
blocks exact schema-rejected proposals for the entire sample. It reuses successful
reads from the current ToolEnv (including empty results) and suppresses repeated
no-op writes until a possible state change. Successful writes remain repeatable.
Read/helper calls and known no-op writes never clear a block. Successful or
uncertain writes invalidate transient read/no-op caches across apps, while schema
rejections stay permanent. The program does not infer changes from state hashes
or inspect hidden app state. See [HELPERS.md](HELPERS.md) for the complete rules.

Both inference and gold SFT replay use the same `tool_execution` boundary and
`automatic_execution_feedback`. These guards run automatically; the model does
not call them as tools. Blocked reads report the original cached result/source
without creating a fake executed call. A conflicting gold proposal fails SFT
preparation rather than silently changing labels. Technical exceptions are
separate from schema refusals and receive at most one retry per exact proposal.
The existing call/invalid/step limits remain in force. No local type coercion,
temporal repair, or automatic name/ID substitution is applied; organizer
validation remains authoritative.

SFT gold replay uses the same `env.validate` acceptance rule and preserves gold
arguments verbatim. Gold rejected by ToolEnv stops preparation with a schema error.
Structured historical tool results remain available to the planner in SFT and
inference; historical calls are not re-executed. No environment state files are read.

## Fine-tune the LLM for next-tool calling with Unsloth (LoRA / QLoRA)

The training schema is the same as inference: `audio_text`, `context`, `history`, and
`label.tool_calls` (ordered `{name, arguments}` objects). Empty call lists produce a
`done` target; missing labels are errors. Only TRAIN data is accepted by the preparation
entrypoint. Keep the organizer's train-specific `zaic_tools` next to `train.jsonl`.

Fine-tuning now defaults to **50% provided `audio_text` + 50% ASR transcripts**
(`--asr-transcript-ratio 0.5`). Train/validation are split by original sample ID
first; each partition then gets a deterministic seeded selection of transcript
sources. The ASR count is rounded to the nearest integer for the full source
dataset and for the training partition (ties round up); validation receives the
remainder. Thus 50:50 stays exact for an even total, and an odd partition differs
by at most one sample between sources. The ratio is by sample,
not by expanded tool-call steps. Each SFT sample keeps the same selected transcript
throughout its trajectory; samples are not duplicated across partitions. History
uses the existing full text/audio resolver unchanged. Checkpoint-selection
rollouts use ASR for **every** held-out sample, independently of its SFT source.

Pass local GIPformer weights via `--asr-model`, including when all train records
already contain `audio_text`. ASR-selected samples are transcribed once during
preparation and **ignore** the supplied current `audio_text`. A missing selected
source or ASR failure stops preparation; it never silently substitutes clean text.
`--asr-transcript-ratio 0` is an explicit clean-text SFT ablation; `1` selects ASR
for all SFT samples. **Both still require `--asr-model`**, because checkpoint
selection always evaluates ASR input. Notebook defaults match the CLI through
`TRAIN_ASR_TRANSCRIPT_RATIO`.

Each SFT row records `transcript_source`. The manifest saves the ratio, seed,
ASR model/settings, per-sample assignments and requested/retained source counts.
Preparation also saves ASR transcripts for the entire held-out set, reusing an
already transcribed current request where possible. Validation rollouts reuse
these ASR transcripts without loading ASR alongside the LLM. If the
explicit diagnostic skip policy removes samples, retained counts may differ from
the requested ratio; both counts are reported. Create a new SFT/run directory to
apply the mix; older prepared data is rejected by the current preparation policy.

The default training backend is **Unsloth** (`FastVisionModel` for Qwen3.5 with
vision layers frozen and text-only LoRA), imported before
Transformers/PEFT in the training process. It uses Unsloth gradient checkpointing,
unquantized LoRA and optimized LoRA kernels (dropout=0). The defaults are
`--quantization none --max-length 16384 --batch-size 1`.
Gradient accumulation remains 8, giving an effective batch size of 8 on one GPU.
Install a CUDA-enabled PyTorch
suitable for your GPU first, then:

```bash
cd code
python -m pip install -r requirements.txt

# One command: stage base weights, prepare dialogues, train/evaluate, save adapter.
# A local model path works fully offline. A HF ID downloads the resolved revision first.
python finetune.py \
  --data-root /path/to/lifemate_data \
  --base-model Qwen/Qwen3.5-4B \
  --asr-model ./models/asr --asr-transcript-ratio 0.5 \
  --output-dir /path/to/runs/toolcalling_v1 \
  --backend unsloth --epochs 3 --quantization none --max-length 16384 \
  --batch-size 1 --gradient-accumulation 8 --learning-rate 2e-4
```

LoRA updates attention/MLP projections, with gradient checkpointing and BF16 on
supporting GPUs. GPU training requires BF16 support and does not silently fall
back to FP16; the explicit CPU test backend uses FP32. No 4-bit/8-bit weight quantization is
enabled by default. `--quantization 4bit` remains an explicit QLoRA option.
An unquantized batch of 1 at up to 16,384 tokens requires substantial VRAM; this
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
be in execution order. It uses the **same complete prompt builder** as inference.
Every allowed tool is present in `tool_schemas` from the first step, with all
catalog fields intact: full descriptions, structured returns, nested parameters,
and extension metadata. `tool_index` also retains full descriptions. Tools not
exposed by `installed_apps` remain unavailable.

There is no schema ranking/subset, character-based size fitting, history window,
text clipping, tool-result clipping, or limit on the number of past observations
included in a prompt. Each step sees all earlier real calls, arguments, results
and any rejection feedback since the latest executed call. JSON serialization still uses compact whitespace; it does
not shorten field values or remove fields. SFT emits call/done examples directly;
all schemas are already visible. The internal `inspect` action and its counters
have been removed; legacy outputs requesting it are invalid proposals and consume
`max_invalid`, never executing a tool. `--max-tool-details` and the notebook's
`MAX_TOOL_DETAILS` setting have also been removed.

Unavailable/schema-invalid gold labels still fail with the sample ID. Training
transcripts and ASR weights for held-out evaluation are required at every mix ratio. No
synthetic tool results or gold arguments are inserted into earlier prompts. Loss
masks system/user/padding tokens and trains only the assistant JSON and its
end-of-turn tokens. The notebook defaults to `TRAIN_MISSING_TOOL_POLICY="error"`;
`--missing-tool-policy skip` remains an explicit diagnostic fallback for an
unrecoverable missing-schema error, with skipped IDs/reasons in the manifest.

Complete prompts can exceed the unchanged 16,384-token context limit. Inference
checks the actual chat-template token count against `--max-input-tokens`; SFT
checks prompt plus target against `--max-length`. Oversized inputs **fail
explicitly**, with no automatic data reduction or token truncation. When needed,
reallocate the input/output budgets within the supported context limit or handle
the oversized sample explicitly. Larger prompts also require more memory.

Outputs: `sft/{train,validation}.jsonl`, split/hash manifest, `training/adapter/`,
checkpoints (optimizer/scheduler/RNG), `training/{train,eval}_results.json`,
`training/trainer_state.json`, training config, resolved base-model SHA (for HF IDs),
and logs for every stage.

### Best-checkpoint selection: state match + tool-call F1

The default selection metric is **`eval_state_tool_score` (0–100), higher is better**,
not teacher-forced `eval_loss`. After each epoch, the current model runs the same
agent decoding used for inference on every held-out TRAIN sample ID (10% by
default, split before SFT expansion), using **ASR transcripts for all IDs**.
The training mixture remains 50:50 by default; teacher-forced validation loss
still uses the assigned SFT sources. ASR runs during preparation and is not
loaded during LLM rollouts, so selection measures the agent on cached ASR input
without including ASR latency.

Pass `--eval-clean-transcripts` to `finetune.py` or `train_lora.py` to additionally
evaluate clean `audio_text` on the same held-out IDs. This optional comparison
runs separate rollouts, each with exactly one fresh ToolEnv per sample, and
approximately doubles the rollout work. Its metrics are named `eval_clean_*`
and its reports are separate; **only ASR `eval_state_tool_score` selects the
checkpoint**. Clean text must be available for every held-out sample when this
option is enabled. In the notebook set `TRAIN_EVAL_CLEAN_TRANSCRIPTS=True`.

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
ASR-only rollout-policy metadata must be regenerated in a new directory;
older mixed-source rollout data is rejected. Old loss-selected checkpoints
cannot be resumed with the new selection settings.

### Validation error diagnostics

Each held-out TRAIN evaluation automatically writes reports under
`training/validation_rollouts/`: `<run>.jsonl` contains per-sample predictions,
gold calls/hashes, `agent_stats.stop_reason`, transcript source and `diagnostics`;
`<run>.diagnostics.json` contains aggregate counts, rates and the same numeric
metrics returned to Trainer. `validation_scores.jsonl` links both files. The
notebook exports completed reports and partial debug traces to Output alongside
other training artifacts.

Analysis happens **after** the rollout. It does not add model generations, tokens
or tool calls, does not put gold/diagnostics into prompts, and does not access
application environment files. The checkpoint selection metric remains
`eval_state_tool_score`; diagnostic counters are additional observations.

Error categories can overlap:

| Category | Definition |
| --- | --- |
| `state_hash_mismatch` | Final hash differs from the saved gold replay hash. |
| `missing_calls` | At least one gold call has no matching prediction. |
| `extra_calls` | At least one submitted call has no matching gold call. |
| `wrong_arguments` | After full-call matching, an unmatched prediction and gold call share a tool name but have differing arguments. |
| `early_stop` | The agent explicitly returned `done`, the hash is wrong, and a gold call remains unattempted after full-call and same-name argument pairing. This is a conservative diagnostic indicator, not proof of the root cause. |

Matching uses the scorer's same maximum one-to-one assignment, duplicate-call
counting and exact/fuzzy argument rules, independently of call order. A wrong-arg
call still contributes one missing gold and one extra prediction. Rejected
proposals absent from ToolEnv's submitted-call log are tracked by
`agent_stats.invalid_proposals`, not counted as extra executed calls. For repeated
tool names, remaining same-name calls are paired in their original order for
argument diagnosis only; this advisory pairing never changes the score. Reports
include zero-based call indices and top-level argument differences (missing,
unexpected, type mismatch or value mismatch, with complete expected/predicted
values). Nested values are preserved, not shortened.

Stop reasons are `done`, `ask_missing_info`, `max_calls`, `max_invalid`,
`time_limit`, `no_available_tools` or `unknown`. Reaching a limit is recorded
separately from explicit early completion. Missing a gold read with the correct
state hash is not labeled `early_stop`; an attempted tool with wrong arguments
alone is not sufficient either. Hash mismatch remains observable even in the
empty-gold case, where the existing scoring rule may still award full credit.

The summary separates **samples with each error** (`error_samples`, rates from
0 to 1) from **individual calls** (`call_counts`), and groups results by
`audio_text`, `asr` or `unknown` transcript source. Trainer receives
`eval_<category>_samples`, `eval_missing_calls`, `eval_extra_calls`,
`eval_wrong_arguments_calls` and `eval_stop_<reason>_samples` in addition to its
existing metrics. These counters are also saved in `eval_results.json`.

Unexpected evaluation failures still abort the run and leave `.partial` output;
completed JSONL/summary files are promoted only after every held-out sample has
been evaluated. This module classifies discrepancies against gold, not their
cause: ASR mistakes or the first reasoning error require inspection of the sample.

### Automatic per-step eval debug trace

Every validation evaluation also writes `<run>.trace.jsonl` beside its rollout
report under `training/validation_rollouts/`. No flag is required. The rollout
rows and diagnostics summary contain `trace_file`; `validation_scores.jsonl`
links the full `trace` path for each evaluation attempt/epoch/checkpoint.

Each JSONL event has `version`, a run-wide one-based `event_index`, `sample_id`
and elapsed seconds since the evaluation started. Agent events also have a
one-based `step` identifying the LLM turn (zero when no generation occurred).
Execution events include a zero-based `call_index` for the execution attempt.

| Event | Saved data |
| --- | --- |
| `sample_start` | Sample ID, selected transcript and transcript source. |
| `llm_start` | Complete system/user request passed to `LocalLLM.generate`. For native FC this is the portable input, before checkpoint-template rendering. |
| `llm_output` | Exact output returned by the model wrapper, including native XML call delimiters. |
| `decision_parsed` | Parsed action, tool name and arguments. |
| `decision_rejected` | Parse/precheck/validation rejection stage and error feedback. |
| `validate_start` / `validate_result` | Proposed tool/arguments, original validator return value and whether the agent accepted it. |
| `call_start` / `call_result` | Executed tool/arguments and complete original return value, including `ok: false` returns. |
| `llm_error`, `validate_error`, `call_error` | Exception type, message and repr. A call exception also includes `agent_result`, the synthesized error sent back to the planner. |
| `stop` / `sample_end` | Stop reason, counters and final official calls/state hash. |
| `sample_error` | Unexpected sample failure; evaluation still aborts instead of treating it as a completed sample. |

Events are written and flushed immediately. During evaluation the filename is
`<run>.trace.partial.jsonl`; a successful full evaluation promotes it to
`<run>.trace.jsonl`. If a model/runtime failure occurs after tool execution, the
partial trace retains those results and the failing stage. The notebook also
exports these explicitly named partial traces; console logs report their path.

Tracing is enabled only by the evaluator. Normal inference has no trace sink by
default. It does not repeat `validate` or `call`, create a second ToolEnv, read
application state files, change prompts/retry budgets or add model generations.
Logged arguments/results are not text-normalized or truncated. The trace stores
full prompts/results and can be large; it is local debug output, not a substitute
for the official ToolEnv call log or part of the submission. No cached SFT/model
format changes are needed for this logging feature.

### Run the main task with the fine-tuned adapter

Keep the **exact base model** used for training. The adapter folder includes its tokenizer:

```bash
mkdir -p result
python main.py \
  --data-root /path/to/lifemate_data --split public_test \
  --asr-model /path/to/asr \
  --llm-model /path/to/exact-base-model \
  --llm-adapter /path/to/runs/toolcalling_v1/training/adapter \
  --llm-dtype bfloat16 --device cuda \
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
  --output-dir /path/to/runs/toolcalling_v1/merged --dtype bfloat16

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
pin Unsloth 2026.10.11, unsloth-zoo 2026.10.11 and compatible Transformers 5.4.0;
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
