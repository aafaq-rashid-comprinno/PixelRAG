# docker/nginx/Dockerfile
# Reverse proxy for the PixelRAG API with blue-green upstream support.
#
# Build:
#   docker build -f docker/nginx/Dockerfile -t pixelrag-nginx docker/nginx/

FROM nginx:1.27-alpine

LABEL org.opencontainers.image.source="https://github.com/StarTrail-org/PixelRAG"
LABEL org.opencontainers.image.description="PixelRAG nginx — reverse proxy"

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD curl -sf http://localhost/nginx-health || exit 1
