#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

DATA_ROOT="${DATA_ROOT:-/data}"
SPLIT="${SPLIT:-auto}"
OUTPUT="${OUTPUT:-/result/submission.jsonl}"

mkdir -p "$(dirname "${OUTPUT}")"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/predict.py" \
  --data-root "${DATA_ROOT}" \
  --split "${SPLIT}" \
  --output "${OUTPUT}" \
  "$@"
