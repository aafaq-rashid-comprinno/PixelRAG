# Docker

Containerized deployment of the PixelRAG stack.

## Images

| Image | Size | Purpose |
|-------|------|---------|
| `pixelrag-render` | ~420MB | Document → screenshot tiles (Chrome CDP) |
| `pixelrag-embed` | ~8GB | Tiles → FAISS vectors (CUDA GPU) |
| `pixelrag-serve` | ~4GB | Search API (FastAPI + FAISS + Qwen3-VL) |
| `pixelrag-agent` | ~250MB | Claude chat backend (Node.js SSE) |
| `pixelrag-nginx` | ~76MB | Reverse proxy + rate limiting |

## Quick Start

```bash
# 1. Configure
cp .env.example .env
# Edit .env — set INDEX_DIR, TILES_DIR, ARTICLES_JSON paths

# 2. Start the runtime stack
docker compose up -d

# 3. Verify
curl http://localhost/health
curl http://localhost:30001/status
```

## Usage

### Runtime Stack (always-on services)

```bash
# Start all services (serve + agent + nginx)
docker compose up -d

# Check logs
docker compose logs -f serve
docker compose logs -f agent

# Stop
docker compose down
```

### Development Mode

```bash
# Hot-reload, relaxed rate limits, source-mounted
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Frontend accessible at :8080 (avoids port 80 conflicts)
```

### Pipeline (one-shot indexing jobs)

```bash
# Render a URL to tiles
docker compose -f docker-compose.pipeline.yml run --rm render \
    https://en.wikipedia.org/wiki/Python --output /data/tiles

# Render a PDF
docker compose -f docker-compose.pipeline.yml run --rm render \
    /data/input/paper.pdf --output /data/tiles --dpi 200

# Chunk tiles
docker compose -f docker-compose.pipeline.yml run --rm embed \
    chunk --shard-dir /data/tiles --workers 8

# Embed (requires GPU)
docker compose -f docker-compose.pipeline.yml run --rm embed \
    embed --shard-dir /data/tiles --output-dir /data/embeddings --gpu-ids 0

# Build FAISS index
docker compose -f docker-compose.pipeline.yml run --rm embed \
    build-index --embeddings-dir /data/embeddings --output-dir /data/index

# Full pipeline from pixelrag.yaml config
docker compose -f docker-compose.pipeline.yml run --rm index-build
```

### Build Images Locally

```bash
# All images (linux/amd64 — required for Chrome + torch)
docker compose build

# Individual images
docker build --platform linux/amd64 -f docker/render/Dockerfile -t pixelrag-render .
docker build --platform linux/amd64 -f docker/serve/Dockerfile -t pixelrag-serve .
docker build --platform linux/amd64 -f docker/embed/Dockerfile -t pixelrag-embed .
docker build -f docker/agent/Dockerfile -t pixelrag-agent .
docker build -f docker/nginx/Dockerfile -t pixelrag-nginx docker/nginx/
```

## Blue-Green Deploy

The nginx config supports blue-green for zero-downtime search API updates:

```bash
# 1. Start the new slot (green) with updated index
docker run -d --name pixelrag-serve-green \
    -p 30002:30001 \
    -v /path/to/new-index:/data/index:ro \
    -v /path/to/tiles:/data/tiles:ro \
    pixelrag-serve

# 2. Wait for health
until curl -sf http://localhost:30002/health; do sleep 5; done

# 3. Update nginx upstream to point to green, reload
docker exec pixelrag-nginx nginx -s reload

# 4. Stop the old (blue) slot
docker stop pixelrag-serve && docker rm pixelrag-serve
```

## Volumes & Data

| Volume | Contents | Typical Size |
|--------|----------|-------------|
| `model-cache` | HuggingFace model weights (Qwen3-VL-2B) | ~4GB |
| Tiles dir | Rendered screenshot JPEGs | 1-500GB |
| Index dir | FAISS index + metadata | 1-216GB |
| Embeddings dir | Intermediate numpy shards | 1-100GB |

**Tip:** Mount `model-cache` as a named Docker volume shared between `serve` and `embed`
to avoid downloading the 4GB model twice.

## Platform Notes

- **All Python images target `linux/amd64`** — the patched Chrome binary and several
  wheel packages (cef-capi-py, torch CUDA) only publish x86_64 wheels.
- **GPU (embed)** — requires `nvidia-docker` runtime and `--gpus all`.
- **Shared memory (render)** — Chrome needs `/dev/shm` > 64MB. Use `--shm-size=2g` or
  the compose `shm_size` directive.
- **Memory (serve)** — the full Wikipedia index is ~216GB. Set `SERVE_MEMORY_LIMIT`
  appropriately. A small custom index works fine with 4-8GB.

## CI/CD

The `.github/workflows/docker.yml` workflow:
- Builds all 5 images in parallel on push to main
- Pushes to GitHub Container Registry (GHCR)
- Validates compose file syntax
- Uses Docker layer caching (GHA cache) for fast rebuilds
