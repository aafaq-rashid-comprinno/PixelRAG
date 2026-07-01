# PixelRAG Architecture

## Overview

PixelRAG is a Visual Retrieval-Augmented Generation system. Instead of parsing documents
to text (lossy), it renders them as screenshots and retrieves over the images directly.
Visual structure — tables, charts, layout, infographics — stays intact.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PixelRAG System                                │
│                                                                      │
│  ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌──────────────────┐   │
│  │ Source  │──▶│ Render  │──▶│  Embed   │──▶│  FAISS Index     │   │
│  │(URL/PDF)│   │(pixelshot)│  │(Qwen3-VL)│  │  (search-ready)  │   │
│  └─────────┘   └─────────┘   └──────────┘   └────────┬─────────┘   │
│                                                        │             │
│                                              ┌────────▼─────────┐   │
│                                              │  Serve (FastAPI)  │   │
│                                              │  /search endpoint │   │
│                                              └────────┬─────────┘   │
│                                                        │             │
│  ┌────────────────┐                          ┌────────▼─────────┐   │
│  │ Web Frontend   │◀─────────────────────────│  Agent (Claude)  │   │
│  │ (Next.js)      │         SSE stream       │  /chat endpoint  │   │
│  └────────────────┘                          └──────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### Stage 0: Render (`pixelshot`)

Converts documents into tiled JPEG screenshots.

```
Input (URL/PDF/HTML/image)
    │
    ▼
Chrome CDP (headless_shell)
    │  viewport: 875×1080px
    │  scroll + capture per tile
    ▼
tiles/
├── article_0.png.tiles/
│   ├── tile_0000.jpg    (875×8192px max)
│   ├── tile_0001.jpg
│   └── tiles.json       (metadata: dimensions, count)
└── article_1.png.tiles/
    └── ...
```

**Key design decisions:**
- Tile height: 8192px max (fits in GPU memory for embedding)
- JPEG quality: 85 (good balance of fidelity vs size)
- Viewport: 875px (matches Wikipedia's content width)
- Backend: Custom CDP over raw websocket (no Playwright dependency)
- Turbo path: Patched Chrome with `rawFilePath` writes directly to /dev/shm

**Package:** `render/src/pixelrag_render/`  
**CLI:** `pixelshot <url> --output ./tiles`

### Stage 1: Chunk

Splits large tiles into model-sized pieces (≤1024px tall).

```
tile_0000.jpg (875×8192)
    │
    ▼
chunk_0000_00.png (875×1024)
chunk_0000_01.png (875×1024)
chunk_0000_02.png (875×1024)
...
chunks.json (manifest with offsets)
```

**Package:** `embed/src/pixelrag_embed/chunk.py`  
**CLI:** `pixelrag chunk --shard-dir ./tiles`

### Stage 2: Embed

Encodes chunk images into dense vectors using Qwen3-VL-Embedding-2B.

```
chunk_0000_00.png
    │
    ▼
Qwen3-VL-Embedding-2B (+ optional LoRA adapter)
    │  instruction: "Retrieve relevant content"
    │  output: float32[2048]
    ▼
embeddings/shard_000.npz
    embeddings: float32[N, 2048]
    metadata: article_id, tile_idx, chunk_idx, y_offset
```

**Two backends:**
- `embed.py`: GPU-heavy, multi-GPU, vLLM/sglang inference server
- `embed_cpu.py`: Local CPU/MPS, single-process, for small indexes

**Package:** `embed/src/pixelrag_embed/`  
**CLI:** `pixelrag embed --shard-dir ./tiles --output-dir ./embeddings`

### Stage 3: Build Index

Creates a FAISS IVF index from embedding shards.

```
embeddings/shard_*.npz
    │
    ▼
FAISS IndexIVFFlat (cosine similarity)
    │  nlist: auto-scaled (total_vectors / 40, max 4096)
    │  L2 normalized → inner product = cosine
    ▼
index/
├── index.faiss       (IVF index, ~1 byte/vector overhead)
└── metadata.npz      (article_ids, tile_indices, chunk_indices, y_offsets)
```

**Package:** `embed/src/pixelrag_embed/index.py`  
**CLI:** `pixelrag build-index --embeddings-dir ./embeddings --output-dir ./index`

### Orchestrator

`pixelrag index build` runs all stages end-to-end from a YAML config:

```yaml
# pixelrag.yaml
source:
  type: local          # local | web | pdf | kiwix
  path: ./my_docs

embed:
  model: Qwen/Qwen3-VL-Embedding-2B
  device: auto         # cuda | mps | cpu

output: ./my_index
```

**Package:** `index/src/pixelrag_index/`  
**CLI:** `pixelrag index build --config pixelrag.yaml`

## Runtime Services

### Search API (`pixelrag serve`)

FastAPI server that loads the FAISS index and Qwen3-VL model for real-time query embedding.

```
POST /search
{
  "queries": [{"text": "What is the capital of France?"}],
  "n_docs": 5
}

Response:
{
  "results": [{
    "hits": [
      {"score": 0.82, "url": "https://en.wikipedia.org/wiki/France", "tile_index": 0, ...}
    ]
  }]
}
```

**Endpoints:**
- `POST /search` — text/image query → ranked results
- `GET /health` — liveness check
- `GET /status` — index stats (vector count, dimension, nlist)
- `GET /tile/{article_id}/{tile_index}/{chunk_index}` — serve tile images

**Memory footprint:** Index size + model (~4GB). Wikipedia = ~216GB total.

**Package:** `serve/src/pixelrag_serve/`  
**CLI:** `pixelrag serve --index-dir ./index --tiles-dir ./tiles --port 30001`

### Agent (`agent-server.mjs`)

Claude-powered chat backend that uses the search API as a tool.

```
User: "Who invented the telephone?"
    │
    ▼
Claude Agent SDK (subscription auth)
    │  tools: pixelrag_search, pixelrag_tile
    │
    ├── pixelrag_search("telephone invention")
    │       → hits: [{url: ".../Alexander_Graham_Bell", tile: 0, chunk: 2}]
    │
    ├── pixelrag_tile(article=..., tile=0, chunk=2)
    │       → [screenshot of the Wikipedia article section]
    │
    └── Claude reads the screenshot tile, answers with citation
    ▼
SSE stream → frontend
```

**Package:** `web/agent-server.mjs`  
**Port:** 30010

### Web Frontend

Next.js 16 app with search bar, tile viewer, and chat interface.

- **Search page** (`/`): direct search against the API, tile lightbox
- **Chat page** (`/chat`): conversational agent with visual search
- **Docs page** (`/docs`): API reference

**Package:** `web/`  
**Deploy:** Vercel (automatic from `main`)

## Data Flow

```
                    OFFLINE (pipeline)                    ONLINE (runtime)
                    ─────────────────                    ────────────────

URLs/PDFs ──▶ pixelshot ──▶ tiles/ ──▶ chunk ──▶ embed ──▶ index.faiss
                              │                                  │
                              │              ┌──────────────────┘
                              ▼              ▼
                         tiles on disk    FAISS in RAM
                              │              │
                              └──────┬───────┘
                                     ▼
                              pixelrag serve (:30001)
                                     │
                         ┌───────────┼───────────┐
                         ▼           ▼           ▼
                    /search      /tile/...    /health
                         │
                         ▼
                  agent-server (:30010)
                         │
                         ▼
                   Next.js frontend
                   (pixelrag.ai)
```

## Source Plugins

The index orchestrator supports pluggable document sources:

| Source | Class | Use Case |
|--------|-------|----------|
| `local` | `LocalSource` | Directory of PDFs, HTML, images, markdown |
| `web` | `WebSource` | URL list (one per line) |
| `pdf` | `PDFSource` | Single PDF file |
| `kiwix` | `KiwixSource` | Wikipedia ZIM archives (8.28M articles) |

## Distributed Indexing

For large-scale indexing (millions of articles), the system supports S3-based coordination:

```
Machine 1 ──┐
Machine 2 ──┤──▶ S3 claim files ──▶ Each machine processes claimed shards
Machine 3 ──┘         │
                      ▼
              manifest.json (shard definitions)
              claims/000.json (in_progress by machine-1)
              claims/001.json (completed by machine-2)
              output/shard_000/ (tiles + embeddings)
```

No fixed assignment — machines claim work dynamically, heartbeat to prevent stale claims.

**Package:** `index/src/pixelrag_index/distributed.py`

## Deployment Topology

```
┌─────────────────────────────────────────────────────┐
│  Vercel                                              │
│  └── pixelrag.ai (Next.js SSR/CDN)                  │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────────────┐
│  Deploy Host (api.pixelrag.ai)                       │
│                                                      │
│  nginx (:80/:443)                                    │
│  ├── /search, /tile, /health → serve (:30001)        │
│  └── /chat                   → agent (:30010)        │
│                                                      │
│  serve (blue/green slots)                            │
│  ├── FAISS index (216GB in RAM)                      │
│  ├── Qwen3-VL model (4GB)                           │
│  └── Tiles on disk (500GB)                           │
│                                                      │
│  agent                                               │
│  └── Claude Agent SDK (subscription auth)            │
└──────────────────────────────────────────────────────┘
```

## Model Architecture

**Embedding model:** `Qwen/Qwen3-VL-Embedding-2B`  
**LoRA adapter:** `Chrisyichuan/wiki-screenshot-embedding-lora` (ckpt200)

- Base: Qwen3-VL-2B (vision-language model)
- Fine-tuned with contrastive learning (InfoNCE loss) on screenshot data
- Training data: LLM-generated queries paired with screenshot tiles
- Hard negative mining: both visual (similar screenshots) and textual (BM25)
- Output: 2048-dim normalized embeddings (cosine similarity)

## Key Files

```
pixelrag/
├── src/pixelrag/cli.py              # umbrella CLI dispatcher
├── render/src/pixelrag_render/
│   ├── render.py                    # public API (render_url, render_pdf)
│   ├── chrome.py                    # Chrome binary management
│   ├── backends/cdp.py              # CDP rendering backend (28K lines)
│   └── strategies/                  # capture optimization strategies
├── embed/src/pixelrag_embed/
│   ├── chunk.py                     # tile → 1024px chunks
│   ├── embed.py                     # GPU embedding (multi-GPU, 100K lines)
│   ├── embed_cpu.py                 # CPU/MPS embedding (local dev)
│   └── index.py                     # FAISS index builder
├── index/src/pixelrag_index/
│   ├── pipelines.py                 # end-to-end orchestrator
│   ├── config.py                    # YAML config parser
│   ├── distributed.py               # S3-based multi-machine coordination
│   └── sources/                     # document source plugins
├── serve/src/pixelrag_serve/
│   └── api.py                       # FastAPI search server
├── web/
│   ├── agent-server.mjs             # Claude chat backend
│   └── app/                         # Next.js frontend
└── train/                           # separate project (LoRA fine-tuning)
    ├── src/training/                # training loop
    ├── src/data_pipeline/           # synthetic data generation
    └── src/evaluation/              # checkpoint evaluation
```
