# docker/agent/Dockerfile
# PixelRAG chat agent backend — Claude Agent SDK over SSE.
#
# Build:
#   docker build -f docker/agent/Dockerfile -t pixelrag-agent .
#
# Run:
#   docker run -d --name pixelrag-agent \
#       -p 30010:30010 \
#       -e PIXELRAG_SEARCH_URL=http://serve:30001 \
#       -e ALLOWED_ORIGIN=https://pixelrag.ai \
#       pixelrag-agent
#
# Note: The Claude Agent SDK includes native binaries (~2.5GB total image).
# This is irreducible — the SDK IS the payload.

FROM node:22-slim

LABEL org.opencontainers.image.source="https://github.com/StarTrail-org/PixelRAG"
LABEL org.opencontainers.image.description="PixelRAG agent — Claude chat backend"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies only
COPY web/package.json web/package-lock.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy the agent server
COPY web/agent-server.mjs ./

# Non-root user for security
RUN useradd -r -s /bin/false agent
USER agent

ENV NODE_ENV=production
ENV AGENT_PORT=30010
ENV PIXELRAG_SEARCH_URL=http://serve:30001
ENV CHAT_MAX_BUDGET_USD=2.00
ENV CHAT_THINKING_TOKENS=2000
ENV ALLOWED_ORIGIN=*
ENV RL_PER_IP=12
ENV RL_GLOBAL_DAILY=500
ENV RL_MAX_CONCURRENT=3

EXPOSE 30010

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:30010/health || exit 1

CMD ["node", "agent-server.mjs"]
