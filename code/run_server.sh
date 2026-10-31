#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset root: thư mục chứa train/ và public_test/
DATA_ROOT="${DATA_ROOT:-/mnt/mmlab2024nas/trantran/lifemate/train+public_test}"

# Tên split, KHÔNG phải đường dẫn
SPLIT="${SPLIT:-public_test}"

# Model directories are configured HERE. Inherited ASR_MODEL/LLM_MODEL variables
# and model-path CLI options must not select another checkpoint for this launcher.
# scripts/download_models.py stages the selected ASR repo into models/asr.
# ASR must be a local CTranslate2 Whisper checkpoint (model.bin, config.json, ...).
ASR_MODEL="${SCRIPT_DIR}/models/phowhisper-large-ct2"

# LLM - Qwen3-4B-Instruct-2507
LLM_MODEL="${SCRIPT_DIR}/models/llm"

# Output
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/result/submission.jsonl}"
TRANSCRIPT_CACHE="${TRANSCRIPT_CACHE:-${SCRIPT_DIR}/result/transcripts.jsonl}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-inf}"

# GPU
DEVICE="${DEVICE:-cuda}"
# main.py accepts --cuda-visible-devices N through "$@" and defaults to one GPU.

# Log the authoritative folders before starting Python, including in nohup logs.
printf '[run_server] ASR_MODEL=%s\n[run_server] LLM_MODEL=%s\n' \
  "${ASR_MODEL}" "${LLM_MODEL}" >&2

# Keep the two model flags AFTER "$@": argparse uses the last value.
exec python -u "${SCRIPT_DIR}/main.py" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output "${OUTPUT}" \
  --transcript-cache "${TRANSCRIPT_CACHE}" \
  --sample-seconds "${SAMPLE_SECONDS}" \
  --device "${DEVICE}" \
  "$@" \
  --asr-model "${ASR_MODEL}" \
  --llm-model "${LLM_MODEL}"
