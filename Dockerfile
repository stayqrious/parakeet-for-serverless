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
        f'NeMo install moved torch to {torch.__version__} -- pin broken'" && \
    echo "--- installed size (image size drives pod pull time) ---" && \
    du -sh /opt/conda/lib/python3*/site-packages | tail -1 && \
    du -sh /opt/conda/lib/python3*/site-packages/* 2>/dev/null \
      | sort -h | tail -15

# Bake the model (~2.5GB): nothing to download at boot,
# and extraction headroom is paid at build time, not on a live worker disk.
# The second load asserts the weights resolve with the hub switched off, which
# is what makes HF_HUB_OFFLINE below safe: if a future NeMo needs the network
# to locate a cached model, this build fails instead of shipping an image that
# boots slowly (or not at all) when huggingface.co is unreachable.
RUN python -c "import nemo.collections.asr as a; \
    a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2')" \
    && HF_HUB_OFFLINE=1 python -c "import nemo.collections.asr as a; \
    a.models.ASRModel.from_pretrained(model_name='nvidia/parakeet-tdt-0.6b-v2'); \
    print('[build] offline model resolve OK')"

# Startup: the weights are baked, so never round-trip to huggingface.co for a
# revision check on boot. Saves a network call on every cold start (serverless
# and pod alike) and removes an external dependency from the boot path.
ENV HF_HUB_OFFLINE=1

WORKDIR /app
# server.py is the pod-mode entry point; it is inert on serverless (handler.py
# is still the image CMD). Pods select it with RunPod's dockerStartCmd.
COPY parakeet_engine.py handler.py server.py ./

CMD ["python", "-u", "handler.py"]
