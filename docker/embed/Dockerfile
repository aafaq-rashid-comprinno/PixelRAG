# docker/embed/Dockerfile
# Embeds screenshot tiles into vectors using Qwen3-VL-Embedding-2B.
# Supports GPU (CUDA) and CPU modes.
#
# Build:
#   docker build -f docker/embed/Dockerfile -t pixelrag-embed .
#
# Run (GPU):
#   docker run --rm --gpus all \
#       -v ./tiles:/data/tiles:ro \
#       -v ./embeddings:/data/embeddings \
#       -v hf-cache:/models \
#       pixelrag-embed embed \
#       --shard-dir /data/tiles --output-dir /data/embeddings --gpu-ids 0
#
# Run (CPU/MPS — local dev):
#   docker run --rm \
#       -v ./tiles:/data/tiles:ro \
#       -v ./embeddings:/data/embeddings \
#       -v hf-cache:/models \
#       pixelrag-embed embed_cpu \
#       --shard-dir /data/tiles --output-dir /data/embeddings --device cpu

FROM --platform=linux/amd64 nvidia/cuda:12.9.0-runtime-ubuntu24.04 AS base

LABEL org.opencontainers.image.source="https://github.com/StarTrail-org/PixelRAG"
LABEL org.opencontainers.image.description="PixelRAG embed — tiles to FAISS vectors (GPU)"

# Python 3.12 + essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgomp1 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package sources
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY embed/ ./embed/
COPY index/ ./index/

# Install with embed extra (torch, transformers, faiss-cpu, numpy)
# torch with CUDA comes from the pytorch-cu129 index defined in pyproject.toml
RUN echo 'cef-capi-py ; sys_platform == "never"' > /tmp/overrides.txt && \
    uv pip install --system --no-cache -e ".[embed]" --override /tmp/overrides.txt && \
    rm /tmp/overrides.txt

# HuggingFace model cache — mount a named volume to persist across runs
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

VOLUME ["/models", "/data/tiles", "/data/embeddings"]

ENTRYPOINT ["pixelrag"]
CMD ["embed", "--help"]
