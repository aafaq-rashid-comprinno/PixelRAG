# docker/render/Dockerfile
# Renders documents (URLs, PDFs, images) to screenshot tiles via Chrome CDP.
#
# Build:
#   docker build --platform linux/amd64 -f docker/render/Dockerfile -t pixelrag-render .
#
# Run:
#   docker run --rm --shm-size=2g -v ./output:/data/tiles pixelrag-render \
#       https://en.wikipedia.org/wiki/Python --output /data/tiles
#
# The turbo Chrome path needs /dev/shm. Docker default is 64MB; override with --shm-size=2g.

FROM --platform=linux/amd64 python:3.12-slim AS base

LABEL org.opencontainers.image.source="https://github.com/StarTrail-org/PixelRAG"
LABEL org.opencontainers.image.description="PixelRAG render — document to screenshot tiles"

# Chrome headless shell runtime deps + build deps for pyturbojpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    fonts-liberation \
    fonts-noto-cjk \
    libasound2t64 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libatspi2.0-0 \
    libcups2t64 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libgtk-3-0t64 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    xdg-utils \
    zstd \
    libturbojpeg0 \
    libturbojpeg0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only what's needed for install (cache-friendly layer ordering)
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY render/ ./render/

# Install core package, excluding cef-capi-py (unused, 118MB, no linux-arm64 wheel)
# The render container only needs: pillow, websockets, pymupdf, pyturbojpeg
RUN echo 'cef-capi-py ; sys_platform == "never"' > /tmp/overrides.txt && \
    uv pip install --system --no-cache -e "." --override /tmp/overrides.txt && \
    rm /tmp/overrides.txt

# Pre-install the patched headless Chrome (linux-x64 turbo path)
RUN pixelshot install-chrome

# Render outputs go here
VOLUME ["/data/tiles"]

ENTRYPOINT ["pixelshot"]
CMD ["--help"]
