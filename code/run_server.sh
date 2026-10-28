#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset root: thư mục chứa train/ và public_test/
DATA_ROOT="${DATA_ROOT:-/mnt/mmlab2024nas/trantran/lifemate/train+public_test}"

# Tên split, KHÔNG phải đường dẫn
SPLIT="${SPLIT:-public_test}"

# Models đã được tải trong Hugging Face cache
# ASR - Whisper Small
ASR_MODEL="${ASR_MODEL:-${SCRIPT_DIR}/models/phowhisper-small-ct2}"

# LLM - Qwen3-4B-Instruct-2507
LLM_MODEL="${LLM_MODEL:-/mnt/mmlab2024nas/trantran/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}"

# Output
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/result/submission.jsonl}"
TRANSCRIPT_CACHE="${TRANSCRIPT_CACHE:-${SCRIPT_DIR}/result/transcripts.jsonl}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-inf}"

# GPU
DEVICE="${DEVICE:-cuda}"

# Chạy inference
exec python "${SCRIPT_DIR}/main.py" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --asr-model "${ASR_MODEL}" \
  --llm-model "${LLM_MODEL}" \
  --output "${OUTPUT}" \
  --transcript-cache "${TRANSCRIPT_CACHE}" \
  --sample-seconds "${SAMPLE_SECONDS}" \
  --device "${DEVICE}" \
  "$@"
