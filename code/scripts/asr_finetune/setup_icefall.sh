#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ICEFALL_DIR="${LIFEMATE_ICEFALL_ROOT:-${CODE_DIR}/models/.research/icefall}"

mkdir -p "$(dirname "${ICEFALL_DIR}")"
if [ ! -d "${ICEFALL_DIR}/.git" ] && [ ! -f "${ICEFALL_DIR}/icefall/__init__.py" ]; then
  git clone --depth 1 https://github.com/k2-fsa/icefall.git "${ICEFALL_DIR}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
PYTHONPATH="${PYTHONPATH:-}" "${PYTHON_BIN}" -c "import torch, k2, kaldifeat, lhotse, sentencepiece, soundfile; print('ASR training deps ok:', torch.__version__, 'cuda=', torch.cuda.is_available())"
echo "Icefall root: ${ICEFALL_DIR}"
