#!/usr/bin/env python3
"""Generate a minimal FAISS index for Docker smoke tests.

Creates a tiny but valid index (5 articles, random embeddings) that lets
the serve container start and respond to /health and /search without needing
the real 216GB Wikipedia index or a GPU.

Usage:
    python docker/test-fixtures/generate_test_index.py [output_dir]

Output:
    output_dir/
    ├── index.faiss          (~1KB, 5 vectors, dimension 2048)
    ├── metadata.npz         (article_ids, tile_indices, chunk_indices, y_offsets, tile_heights)
    ├── articles.json        (5 test articles with titles and URLs)
    └── tiles/
        └── 0.png.tiles/
            ├── tiles.json
            ├── chunks.json
            └── chunk_0000_00.png  (1x1 placeholder)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

OUTPUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docker/test-fixtures/data")

# Dimensions match Qwen3-VL-Embedding-2B output
DIM = 2048
N_ARTICLES = 5
CHUNKS_PER_ARTICLE = 2
N_VECTORS = N_ARTICLES * CHUNKS_PER_ARTICLE

TEST_ARTICLES = [
    {"title": "Python (programming language)", "url": "https://en.wikipedia.org/wiki/Python_(programming_language)"},
    {"title": "Retrieval-augmented generation", "url": "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"},
    {"title": "FAISS", "url": "https://en.wikipedia.org/wiki/Faiss"},
    {"title": "Screenshot", "url": "https://en.wikipedia.org/wiki/Screenshot"},
    {"title": "Visual search", "url": "https://en.wikipedia.org/wiki/Visual_search"},
]


def main():
    print(f"Generating test index in {OUTPUT}/ ...")

    index_dir = OUTPUT / "index"
    tiles_dir = OUTPUT / "tiles"
    index_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # Generate random normalized embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((N_VECTORS, DIM)).astype(np.float32)
    # L2 normalize (cosine similarity index)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Build a flat (no IVF) FAISS index — works without training
    try:
        import faiss
    except ImportError:
        # Fallback: write raw index file that faiss can read
        # Use IndexFlatIP (inner product = cosine on normalized vectors)
        print("faiss not installed, writing index manually...")
        _write_flat_index(index_dir / "index.faiss", embeddings)
    else:
        index = faiss.IndexFlatIP(DIM)
        index.add(embeddings)
        faiss.write_index(index, str(index_dir / "index.faiss"))
        print(f"  FAISS index: {N_VECTORS} vectors, dim={DIM}")

    # Metadata
    article_ids = np.repeat(np.arange(N_ARTICLES), CHUNKS_PER_ARTICLE).astype(np.int64)
    tile_indices = np.tile(np.array([0, 0]), N_ARTICLES).astype(np.int32)
    chunk_indices = np.tile(np.arange(CHUNKS_PER_ARTICLE), N_ARTICLES).astype(np.int32)
    y_offsets = np.tile(np.array([0, 1024]), N_ARTICLES).astype(np.int32)
    tile_heights = np.full(N_VECTORS, 1024, dtype=np.int32)

    np.savez(
        index_dir / "metadata.npz",
        article_ids=article_ids,
        tile_indices=tile_indices,
        chunk_indices=chunk_indices,
        y_offsets=y_offsets,
        tile_heights=tile_heights,
    )
    print(f"  Metadata: {N_VECTORS} entries")

    # Articles JSON
    articles_path = OUTPUT / "articles.json"
    with open(articles_path, "w") as f:
        json.dump(TEST_ARTICLES, f, indent=2)
    print(f"  Articles: {len(TEST_ARTICLES)} entries")

    # Minimal tile structure (so /tile/ endpoint doesn't 404)
    for i in range(min(2, N_ARTICLES)):
        tile_dir = tiles_dir / f"{i}.png.tiles"
        tile_dir.mkdir(exist_ok=True)

        # tiles.json
        with open(tile_dir / "tiles.json", "w") as f:
            json.dump({"tiles": [{"file": "tile_0000.png", "width": 875, "height": 2048}]}, f)

        # chunks.json
        chunks = [
            {"file": f"chunk_0000_{j:02d}.png", "tile_index": 0, "chunk_index": j,
             "x_offset": 0, "y_offset": j * 1024, "width": 875, "height": 1024}
            for j in range(CHUNKS_PER_ARTICLE)
        ]
        with open(tile_dir / "chunks.json", "w") as f:
            json.dump({"chunks": chunks}, f)

        # Placeholder 1x1 PNG for each chunk
        _write_1x1_png(tile_dir / "chunk_0000_00.png")
        _write_1x1_png(tile_dir / "chunk_0000_01.png")

    print("  Tiles: placeholder structure created")
    print(f"\nDone. Use with:\n  INDEX_DIR={index_dir} TILES_DIR={tiles_dir} ARTICLES_JSON={articles_path} docker compose up")


def _write_1x1_png(path: Path):
    """Write a minimal valid 1x1 white PNG (67 bytes)."""
    import struct
    import zlib

    def chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\xff\xff\xff")
    idat = chunk(b"IDAT", raw)
    iend = chunk(b"IEND", b"")
    path.write_bytes(sig + ihdr + idat + iend)


def _write_flat_index(path: Path, embeddings: np.ndarray):
    """Write a FAISS IndexFlatIP file without importing faiss.

    FAISS flat index binary format (little-endian):
    - 4 bytes magic: "IxFI" (IndexFlatIP)
    - Header with dimension and metric info
    This is complex — just skip if faiss unavailable.
    """
    raise ImportError("Install faiss-cpu to generate the test index: pip install faiss-cpu")


if __name__ == "__main__":
    main()
