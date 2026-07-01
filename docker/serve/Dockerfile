# docker/serve/Dockerfile
# FAISS-backed visual search API (FastAPI + Qwen3-VL for query embedding).
#
# Build:
#   docker build -f docker/serve/Dockerfile -t pixelrag-serve .
#
# Run:
#   docker run -d --name pixelrag-serve \
#       -p 30001:30001 \
#       -v ./index:/data/index:ro \
#       -v ./tiles:/data/tiles:ro \
#       -v ./articles.json:/data/articles.json:ro \
#       -v hf-cache:/models \
#       pixelrag-serve
#
# The Wikipedia index is ~216GB — ensure the host has enough RAM.
# The Qwen3-VL model (~4GB) is loaded for real-time query embedding.

FROM --platform=linux/amd64 python:3.12-slim AS base

LABEL org.opencontainers.image.source="https://github.com/StarTrail-org/PixelRAG"
LABEL org.opencontainers.image.description="PixelRAG serve — FAISS visual search API"

# Runtime libs for FAISS (OpenBLAS) and torch (libgomp for OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package sources
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY serve/ ./serve/
COPY embed/ ./embed/

# Install serve extra + torchvision (needed by Qwen3-VL processor)
RUN echo 'cef-capi-py ; sys_platform == "never"' > /tmp/overrides.txt && \
    uv pip install --system --no-cache -e ".[serve]" torchvision --override /tmp/overrides.txt && \
    rm /tmp/overrides.txt

# HuggingFace model cache
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models

VOLUME ["/models", "/data/index", "/data/tiles"]

EXPOSE 30001

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:30001/health || exit 1

ENTRYPOINT ["pixelrag", "serve"]
CMD ["--index-dir", "/data/index", \
     "--tiles-dir", "/data/tiles", \
     "--articles-json", "/data/articles.json", \
     "--device", "cpu", \
     "--port", "30001"]
