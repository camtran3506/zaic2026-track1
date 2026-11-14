#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset root: thư mục chứa train/ và public_test/
DATA_ROOT="${DATA_ROOT:-/mnt/mmlab2024nas/trantran/lifemate/train+public_test}"

# Tên split, KHÔNG phải đường dẫn
SPLIT="${SPLIT:-public_test}"

# Model directories are configured HERE. Inherited ASR_MODEL/LLM_MODEL variables
# and model-path CLI options must not select another checkpoint for this launcher.
# scripts/download_models.py stages the original ASR repo into models/asr.
# If an adapted checkpoint is staged in models/asr_ft, prefer it while preserving models/asr.
# ASR must be a local GIPformer ONNX transducer (encoder/decoder/joiner + tokens.txt).
ASR_MODEL="${SCRIPT_DIR}/models/asr"
if [ "${ASR_MODEL}" = "${SCRIPT_DIR}/models/asr" ] && [ -s "${SCRIPT_DIR}/models/asr_ft/tokens.txt" ]; then
  ASR_MODEL="${SCRIPT_DIR}/models/asr_ft"
fi

# LLM - Qwen3.5-4B
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
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
ASR_BEAM_SIZE="${ASR_BEAM_SIZE:-8}"
ASR_HOTWORDS="${ASR_HOTWORDS:-auto}"
ASR_HOTWORDS_SCORE="${ASR_HOTWORDS_SCORE:-1.5}"
ASR_POST_NORMALIZE_FLAG="--asr-post-normalize"
if [[ "${ASR_POST_NORMALIZE:-1}" == "0" ]]; then
  ASR_POST_NORMALIZE_FLAG="--no-asr-post-normalize"
fi
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/main.py" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output "${OUTPUT}" \
  --transcript-cache "${TRANSCRIPT_CACHE}" \
  --sample-seconds "${SAMPLE_SECONDS}" \
  --asr-beam-size "${ASR_BEAM_SIZE}" \
  --asr-hotwords "${ASR_HOTWORDS}" \
  --asr-hotwords-score "${ASR_HOTWORDS_SCORE}" \
  "${ASR_POST_NORMALIZE_FLAG}" \
  --device "${DEVICE}" \
  "$@" \
  --asr-model "${ASR_MODEL}" \
  --llm-model "${LLM_MODEL}"
