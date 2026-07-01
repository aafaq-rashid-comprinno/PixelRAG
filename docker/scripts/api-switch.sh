#!/usr/bin/env bash
# docker/scripts/api-switch.sh — Blue-green switch for the search API
#
# Brings up a new serve container, health-checks it, updates nginx upstream,
# and tears down the old container. Zero-downtime deployment.
#
# Usage:
#   ./docker/scripts/api-switch.sh [new-index-dir]
#
# Examples:
#   # Switch to a new index (different data)
#   ./docker/scripts/api-switch.sh /data/new-index
#
#   # Restart with same index (e.g. after code update)
#   ./docker/scripts/api-switch.sh

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"  # 5 minutes for index load
HEALTH_INTERVAL=5
NEW_INDEX_DIR="${1:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[switch]${NC} $*"; }
warn() { echo -e "${YELLOW}[switch]${NC} $*"; }
err() { echo -e "${RED}[switch]${NC} $*" >&2; }

# Determine which slot is active and which is idle
ACTIVE_CONTAINER=$(docker ps --filter "name=pixelrag-serve" --filter "status=running" --format "{{.Names}}" | head -1)

if [[ "$ACTIVE_CONTAINER" == *"green"* ]]; then
    NEW_SLOT="blue"
    NEW_NAME="pixelrag-serve-blue"
    NEW_PORT=30001
    OLD_NAME="pixelrag-serve-green"
else
    NEW_SLOT="green"
    NEW_NAME="pixelrag-serve-green"
    NEW_PORT=30002
    OLD_NAME="pixelrag-serve-blue"
fi

log "Active: ${ACTIVE_CONTAINER:-none}, deploying to: $NEW_SLOT ($NEW_NAME on :$NEW_PORT)"

# Build serve image if needed
log "Building serve image..."
docker compose -f "$COMPOSE_FILE" build serve 2>&1 | tail -5

# Start the new slot
log "Starting $NEW_NAME..."
INDEX_MOUNT="${NEW_INDEX_DIR:-${INDEX_DIR:-./data/index}}"

docker run -d \
    --name "$NEW_NAME" \
    --network "$(docker network ls --filter 'name=pixelrag' --format '{{.Name}}' | head -1)" \
    -p "$NEW_PORT:30001" \
    -v "$INDEX_MOUNT:/data/index:ro" \
    -v "${TILES_DIR:-./data/tiles}:/data/tiles:ro" \
    -v "${ARTICLES_JSON:-./data/articles.json}:/data/articles.json:ro" \
    -v "pixelrag-models:/models" \
    -e "HF_HOME=/models" \
    pixelrag-serve

# Health check the new slot
log "Waiting for $NEW_NAME to become healthy (timeout: ${HEALTH_TIMEOUT}s)..."
elapsed=0
while [[ $elapsed -lt $HEALTH_TIMEOUT ]]; do
    if curl -sf "http://localhost:$NEW_PORT/health" > /dev/null 2>&1; then
        log "$NEW_NAME is healthy after ${elapsed}s"
        break
    fi
    sleep $HEALTH_INTERVAL
    elapsed=$((elapsed + HEALTH_INTERVAL))
    if (( elapsed % 30 == 0 )); then
        warn "  Still waiting... (${elapsed}s / ${HEALTH_TIMEOUT}s)"
    fi
done

if ! curl -sf "http://localhost:$NEW_PORT/health" > /dev/null 2>&1; then
    err "$NEW_NAME failed health check after ${HEALTH_TIMEOUT}s"
    err "Logs:"
    docker logs --tail 20 "$NEW_NAME"
    err "Rolling back — removing failed container"
    docker rm -f "$NEW_NAME"
    exit 1
fi

# Smoke test — run a search query
log "Smoke test: running search query..."
SMOKE_RESULT=$(curl -sf -X POST "http://localhost:$NEW_PORT/search" \
    -H "Content-Type: application/json" \
    -d '{"queries": [{"text": "test"}], "n_docs": 1}' 2>&1) || true

if echo "$SMOKE_RESULT" | grep -q '"results"'; then
    log "Smoke test passed"
else
    warn "Smoke test returned unexpected response (may be ok for empty index)"
fi

# Switch nginx upstream
log "Switching nginx upstream to $NEW_NAME:30001..."
NGINX_CONTAINER=$(docker ps --filter "name=pixelrag-nginx" --format "{{.Names}}" | head -1)

if [[ -n "$NGINX_CONTAINER" ]]; then
    # Update the upstream to point to the new container
    docker exec "$NGINX_CONTAINER" sh -c "
        sed -i 's/server serve:[0-9]*/server $NEW_NAME:30001/' /etc/nginx/nginx.conf
        nginx -s reload
    "
    log "Nginx reloaded, traffic flowing to $NEW_NAME"
else
    warn "No nginx container found — skipping upstream switch"
fi

# Drain and stop the old container
if [[ -n "$ACTIVE_CONTAINER" ]] && docker ps -q --filter "name=$ACTIVE_CONTAINER" | grep -q .; then
    log "Stopping old container: $ACTIVE_CONTAINER"
    sleep 5  # Grace period for in-flight requests
    docker stop "$ACTIVE_CONTAINER" --time 30
    docker rm "$ACTIVE_CONTAINER"
    log "Old container removed"
fi

log "✓ Switch complete: $NEW_SLOT is now active on :$NEW_PORT"
log "  Rollback: docker stop $NEW_NAME && docker start $OLD_NAME"
