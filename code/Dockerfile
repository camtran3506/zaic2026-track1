# Build the image with locally staged model weights in ./models/.
# Build may use internet for pip packages; the final container runs fully offline.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    CUBLAS_WORKSPACE_CONFIG=:4096:8
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py score_train.py /app/
COPY lifemate/ /app/lifemate/
COPY models/ /opt/models/
RUN test -f /opt/models/llm/config.json && \
    test -s /opt/models/asr/encoder.int8.onnx && \
    test -s /opt/models/asr/decoder.int8.onnx && \
    test -s /opt/models/asr/joiner.int8.onnx && \
    test -s /opt/models/asr/tokens.txt && \
    mkdir -p /app/result

ENTRYPOINT ["python", "/app/main.py"]
CMD ["--data-root", "/data", \
     "--split", "private_test", \
     "--asr-model", "/opt/models/asr", \
     "--llm-model", "/opt/models/llm", \
     "--output", "/app/result/submission.jsonl"]
