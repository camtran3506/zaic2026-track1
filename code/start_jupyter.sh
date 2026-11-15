#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p /result

JUPYTER_TOKEN="${JUPYTER_TOKEN:-zac2026}"

exec jupyter lab \
  --port "${JUPYTER_PORT:-9777}" \
  --ip 0.0.0.0 \
  --IdentityProvider.token="${JUPYTER_TOKEN}" \
  --allow-root \
  --no-browser \
  --notebook-dir="${SCRIPT_DIR}"
