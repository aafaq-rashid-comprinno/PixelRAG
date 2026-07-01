"""Retrieval strategies for evaluation benchmarks.

Split into focused modules by retrieval approach:
- base: RetrievalResult, BaseRetriever ABC, shared utilities
- screenshot: pre-captured screenshot retrievers
- text: text-fetching retrievers (Jina, Wikipedia API)
- vector: FAISS/ColQwen vector similarity retrievers
- api: external search API retrievers (PixelRAG serve, Qwen3-VL)
- wrappers: composable wrappers (OCR, rendered text, hybrid)

All classes are re-exported here for backwards compatibility:
    from eval.lib.retrieval import BaseRetriever, LocalAPIRetriever, ...
"""

from .base import (
    BaseRetriever,
    RetrievalResult,
)
from .screenshot import (
    NaiveRetriever,
    EVQANoRetrievalRetriever,
    WorldVQANoRetrievalRetriever,
    ScreenshotRetriever,
    TiledScreenshotRetriever,
    LocalWikiTiledScreenshotRetriever,
)
from .text import (
    TextRetriever,
    JinaReaderRetriever,
    WikipediaAPIRetriever,
)
from .vector import (
    VectorRetriever,
    ColQwenVectorRetriever,
    TiledVectorRetriever,
    TiledColQwenVectorRetriever,
    TextVectorRetriever,
)
from .api import (
    DsServeRetriever,
    LocalAPIRetriever,
    TiledQwen3VLEmbeddingRetriever,
    TextAPIRetriever,
)
from .wrappers import (
    OCRWrappedRetriever,
    RenderedTextWrapper,
    HybridRetriever,
    HTMLDOMLookupRetriever,
)

__all__ = [
    "BaseRetriever",
    "RetrievalResult",
    "NaiveRetriever",
    "EVQANoRetrievalRetriever",
    "WorldVQANoRetrievalRetriever",
    "ScreenshotRetriever",
    "TiledScreenshotRetriever",
    "LocalWikiTiledScreenshotRetriever",
    "TextRetriever",
    "JinaReaderRetriever",
    "WikipediaAPIRetriever",
    "VectorRetriever",
    "ColQwenVectorRetriever",
    "TiledVectorRetriever",
    "TiledColQwenVectorRetriever",
    "TextVectorRetriever",
    "DsServeRetriever",
    "LocalAPIRetriever",
    "TiledQwen3VLEmbeddingRetriever",
    "TextAPIRetriever",
    "OCRWrappedRetriever",
    "RenderedTextWrapper",
    "HybridRetriever",
    "HTMLDOMLookupRetriever",
]
