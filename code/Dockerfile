# Build the image with locally staged inference weights in ./saved_models/.
# Matched to .venv3 runtime (Python 3.12, torch 2.11.0+cu128, transformers 5.4.0,
# sherpa_onnx 1.13.8, causal_conv1d 1.6.1, flash-linear-attention 0.6.0).
# CUDA version <= 13.2 per ZAC 2026 LifeMate Docker Guide; final container runs offline.
# Note: ensure ./saved_models/llm (or ./saved_models/llm_ft) is a real directory (not a symlink outside saved_models/) before building.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/code \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    build-essential \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /code

COPY requirements.txt /code/requirements.txt
# flash-linear-attention==0.6.0 declares transformers>=5.6.2 in metadata while .venv3 runs 5.4.0;
# fla-core==0.6.0 and einops==0.8.2 are pinned in requirements.txt, so --no-deps preserves transformers==5.4.0 & torch==2.11.0+cu128.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt && \
    pip install --no-cache-dir --no-deps \
      "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
      "flash-linear-attention==0.6.0" && \
    python -c "import torch, torchaudio, transformers, peft, accelerate, bitsandbytes, sherpa_onnx, onnxruntime, soundfile, soxr, scipy, numpy, einops, causal_conv1d, fla, jupyterlab; print('Runtime OK:', torch.__version__, transformers.__version__)"

COPY main.py predict.py benchmark_timing.py predict_notebook.ipynb \
     predict.sh start_jupyter.sh run_server.sh README.md /code/
COPY lifemate/ /code/lifemate/
COPY training_code/ /code/training_code/
COPY saved_models/ /code/saved_models/

RUN chmod +x /code/predict.sh /code/start_jupyter.sh /code/run_server.sh && \
    mkdir -p /result && \
    test -f /code/predict.py && \
    test -f /code/predict.sh && \
    test -f /code/start_jupyter.sh && \
    test -f /code/predict_notebook.ipynb && \
    test -f /code/README.md && \
    test -f /code/training_code/README.md && \
    (test -f /code/saved_models/llm_ft/config.json || test -f /code/saved_models/llm/config.json) && \
    ((test -s /code/saved_models/asr_ft/tokens.txt && \
      (test -s /code/saved_models/asr_ft/encoder.onnx || test -s /code/saved_models/asr_ft/encoder.int8.onnx) && \
      (test -s /code/saved_models/asr_ft/decoder.onnx || test -s /code/saved_models/asr_ft/decoder.int8.onnx) && \
      (test -s /code/saved_models/asr_ft/joiner.onnx || test -s /code/saved_models/asr_ft/joiner.int8.onnx)) || \
     (test -s /code/saved_models/asr/tokens.txt && \
      (test -s /code/saved_models/asr/encoder.onnx || test -s /code/saved_models/asr/encoder.int8.onnx) && \
      (test -s /code/saved_models/asr/decoder.onnx || test -s /code/saved_models/asr/decoder.int8.onnx) && \
      (test -s /code/saved_models/asr/joiner.onnx || test -s /code/saved_models/asr/joiner.int8.onnx)))

EXPOSE 9777
CMD ["/bin/bash", "/code/predict.sh"]
