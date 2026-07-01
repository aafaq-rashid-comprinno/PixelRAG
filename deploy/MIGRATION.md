# Migration Guide: systemd → Docker

How to transition the PixelRAG production deployment from systemd services to Docker Compose.

## Current State (systemd)

```
┌─────────────────────────────────────────────────┐
│  Deploy Host                                     │
│                                                  │
│  pixelrag-api.service      → :30001 (blue)       │
│  pixelrag-api-green.service → :30002 (green)     │
│  pixelrag-agent.service    → :30010              │
│  nginx                     → :80/:443            │
│                                                  │
│  Data: /home/yichuan/visrag-data/                │
│  Venv: /home/yichuan/visrag/.venv/               │
│  Logs: journal + /home/yichuan/visrag/logs/      │
└─────────────────────────────────────────────────┘
```

**Services:**
- `pixelrag-api.service` — search API (blue slot)
- `pixelrag-api-green.service` — search API (green slot)
- `pixelrag-agent.service` — chat agent
- nginx — manually configured, upstream switched by `deploy/api-switch.sh`

**CD:** GitHub self-hosted runner → `deploy/deploy.sh` → selective restart

## Target State (Docker Compose)

```
┌─────────────────────────────────────────────────┐
│  Deploy Host                                     │
│                                                  │
│  docker compose up -d                            │
│  ├── pixelrag-serve     → :30001                 │
│  ├── pixelrag-agent     → :30010                 │
│  └── pixelrag-nginx     → :80                    │
│                                                  │
│  Volumes:                                        │
│  ├── /data/index (ro)                            │
│  ├── /data/tiles (ro)                            │
│  └── pixelrag-models (shared)                    │
│                                                  │
│  Logs: docker compose logs -f                    │
└─────────────────────────────────────────────────┘
```

## Migration Steps

### Phase 0: Preparation (no downtime)

```bash
# 1. Install Docker on the deploy host
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# 2. Clone or update the repo
cd ~/visrag && git pull origin main

# 3. Create .env
cp .env.example .env
```

Edit `.env` to point at your existing data:
```bash
INDEX_DIR=/home/yichuan/visrag-data/search_index
TILES_DIR=/home/yichuan/visrag-data/tiles
ARTICLES_JSON=/home/yichuan/visrag-data/articles.json
SERVE_MEMORY_LIMIT=250g
ALLOWED_ORIGIN=https://pixelrag.ai
```

### Phase 1: Build Images (no downtime)

```bash
# Build on the host (or pull from GHCR once CI publishes them)
docker compose build serve agent
docker build -f docker/nginx/Dockerfile -t pixelrag-nginx docker/nginx/
```

This takes 10-20 minutes (downloads torch, etc.) but doesn't affect running services.

### Phase 2: Smoke Test (no downtime)

```bash
# Start Docker serve on a different port
docker run -d --name pixelrag-serve-test \
    -p 30099:30001 \
    -v /home/yichuan/visrag-data/search_index:/data/index:ro \
    -v /home/yichuan/visrag-data/tiles:/data/tiles:ro \
    -v /home/yichuan/visrag-data/articles.json:/data/articles.json:ro \
    -v pixelrag-models:/models \
    pixelrag-serve

# Wait for model load + index load (2-5 minutes for 216G)
watch curl -sf http://localhost:30099/health

# Test search
curl -X POST http://localhost:30099/search \
    -H "Content-Type: application/json" \
    -d '{"queries": [{"text": "Nikola Tesla"}], "n_docs": 3}'

# Verify results match the systemd instance
# Then tear down the test container
docker rm -f pixelrag-serve-test
```

### Phase 3: Cutover (1-5 minutes downtime)

```bash
# 1. Stop systemd services
sudo systemctl stop pixelrag-api pixelrag-api-green pixelrag-agent

# 2. Start Docker stack
docker compose up -d

# 3. Wait for health (serve takes longest — index load)
docker compose logs -f serve  # watch for "Uvicorn running on..."

# 4. Verify nginx is proxying
curl http://localhost/health
curl http://localhost:30001/status

# 5. Disable systemd services (don't start on reboot)
sudo systemctl disable pixelrag-api pixelrag-api-green pixelrag-agent
```

### Phase 4: Update CD (after cutover)

Update `.github/workflows/deploy.yml` to use Docker instead of systemd:

```yaml
# Old: deploy/deploy.sh (selective systemd restart)
# New: docker compose pull && docker compose up -d
```

Or keep the self-hosted runner and update `deploy/deploy.sh`:

```bash
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

git pull origin main

# Rebuild only changed images
docker compose build

# Rolling restart (agent is instant, serve takes time)
docker compose up -d --no-deps agent nginx

# Serve: only restart if serve/ code changed
if git diff --name-only HEAD~1 HEAD | grep -qE '^serve/'; then
    echo "serve/ changed — use docker/scripts/api-switch.sh for zero-downtime"
fi
```

## Rollback

If anything goes wrong during cutover:

```bash
# 1. Stop Docker
docker compose down

# 2. Restart systemd
sudo systemctl start pixelrag-api pixelrag-agent

# 3. Verify
curl http://localhost:30001/health
```

## Comparison

| Aspect | systemd | Docker |
|--------|---------|--------|
| **Start** | `systemctl start` | `docker compose up -d` |
| **Logs** | `journalctl -u pixelrag-api` | `docker compose logs serve` |
| **Restart** | `systemctl restart` | `docker compose restart serve` |
| **Blue-green** | `deploy/api-switch.sh <port>` | `docker/scripts/api-switch.sh` |
| **Dependencies** | System Python, uv sync | Docker image (self-contained) |
| **Isolation** | Shared system libs | Fully isolated |
| **Portability** | Host-specific | Any Docker host |
| **Memory limit** | None (or cgroup manual) | `deploy.resources.limits.memory` |
| **Health checks** | None (manual curl) | Built-in `HEALTHCHECK` |
| **Auto-restart** | `Restart=always` | `restart: unless-stopped` |

## What to Keep from systemd

- **deploy/deploy.sh** — still useful for git pull + selective rebuild
- **deploy/api-switch.sh** — replaced by `docker/scripts/api-switch.sh`
- **Service files** — keep for rollback reference, delete after 2 weeks stable
- **nginx config** — now managed inside the container, remove from /etc/nginx

## Files to Remove After Migration

```bash
# After confirming Docker is stable for 2+ weeks:
sudo rm /etc/systemd/system/pixelrag-api.service
sudo rm /etc/systemd/system/pixelrag-api-green.service
sudo rm /etc/systemd/system/pixelrag-agent.service
sudo rm /etc/nginx/conf.d/pixelrag-api-upstream.conf
sudo systemctl daemon-reload
```
