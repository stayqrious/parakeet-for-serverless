# Parakeet TDT 0.6B v2 serverless worker
# Base = the exact torch/CUDA combination validated on the A40 pod:
# torch 2.6.0 + cu124 (satisfies NeMo's torch>=2.6 floor, so pip will NOT
# upgrade torch during the NeMo install -- this avoids the version cascade
# that cost an hour of pod debugging).

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "nemo_toolkit[asr]" runpod requests && \
    python -c "import torch; assert torch.__version__.startswith('2.6.'), \
        f'NeMo install moved torch to {torch.__version__} -- pin broken'"

# Bake the model (~2.5GB): no network volume, no first-request bootstrap,
# and extraction headroom is paid at build time, not on a live worker disk.
RUN python -c "import nemo.collections.asr as a; \
    a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')"

WORKDIR /app
COPY parakeet_engine.py handler.py ./

CMD ["python", "-u", "handler.py"]
