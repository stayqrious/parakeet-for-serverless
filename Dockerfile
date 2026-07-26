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

# Deps, then two startup-cost measures, all in ONE layer on purpose:
#  - Prune packaged test suites. Image bytes are billed GPU-seconds, because a
#    pod pulls its image after the GPU is allocated. The deletion must happen
#    in the same layer as the install: an rm in a later layer leaves the bytes
#    in this one and shrinks nothing. Only "tests" is removed -- "testing" is a
#    real imported subpackage in numpy and sympy. The model load in the next
#    layer re-validates the pruned tree, so a bad prune fails the build.
#  - Precompile bytecode so no boot pays to write .pyc.
# The freeze and du output are read from the build log to decide the next round
# of slimming (and to pin a future split-layer build).
RUN pip install "nemo_toolkit[asr]" runpod requests && \
    python -c "import torch; assert torch.__version__.startswith('2.6.'), \
        f'NeMo install moved torch to {torch.__version__} -- pin broken'" && \
    SP="$(python -c 'import site; print(site.getsitepackages()[0])')" && \
    du -sh "$SP" | sed 's/^/[build] before prune: /' && \
    find "$SP" -type d -name tests -prune -exec rm -rf {} + && \
    (python -m compileall -q -j 0 "$SP" >/dev/null 2>&1 || true) && \
    echo "[build] --- pip freeze ---" && \
    pip freeze && \
    echo "[build] --- largest site-packages entries ---" && \
    du -sh "$SP" | sed 's/^/[build] after prune: /' && \
    du -sh "$SP"/* 2>/dev/null | sort -h | tail -20

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
