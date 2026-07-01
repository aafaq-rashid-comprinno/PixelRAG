"""Screenshot-based retrievers.

Retrievers that use pre-captured or on-demand screenshots as context:
- NaiveRetriever: no retrieval (baseline)
- EVQANoRetrievalRetriever: no retrieval for EVQA tasks
- WorldVQANoRetrievalRetriever: no retrieval for WorldVQA
- ScreenshotRetriever: single full-page screenshot
- TiledScreenshotRetriever: tiled screenshot capture
- LocalWikiTiledScreenshotRetriever: pre-cached Wikipedia tiles
"""

import base64
import io
import logging
import os

from .base import BaseRetriever, RetrievalResult, _save_task_query_image, _save_worldvqa_query_image, _worldvqa_image_to_base64

logger = logging.getLogger(__name__)



class NaiveRetriever(BaseRetriever):
    """No retrieval - returns empty result, LLM answers from its own knowledge."""

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        return RetrievalResult(retrieval_type="naive")


class EVQANoRetrievalRetriever(BaseRetriever):
    """EVQA without retrieval: query + iNaturalist image only, no Wikipedia tiles.

    Used to test VLM's ability to answer from the species image alone.
    """

    def __init__(self, tiles_dir: str = "tiles/evqa"):
        self.tiles_dir = tiles_dir

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        inat_image_path = _get_query_image_path_for_example(example, self.tiles_dir)
        return RetrievalResult(
            images=[],
            retrieval_type="evqa_no_retrieval_multimodal",
            pixel_query_path=inat_image_path,
            query_image_path=inat_image_path,
        )


def _save_task_query_image(
    example: dict, task_name: str, base_dir: str = "tiles"
) -> str | None:
    """Save query image from any task to disk. Returns path or None.
    Images saved to {base_dir}/{task_name}_images/{example_id}.png
    Works with PIL images, base64 strings, or dict with 'bytes' key.
    """
    img = example.get("image")
    if img is None:
        return None
    example_id = example.get("id", "unknown")
    save_dir = os.path.join(base_dir, f"{task_name}_images")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{example_id}.png")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    try:
        if hasattr(img, "save"):
            img.save(out_path, format="PNG")
            return out_path
        if isinstance(img, str):
            raw = (
                img.split(",", 1)[1] if img.startswith("data:") and "," in img else img
            )
            data = base64.b64decode(raw)
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        if isinstance(img, dict) and "bytes" in img:
            b = img["bytes"]
            if b:
                with open(out_path, "wb") as f:
                    f.write(b)
                return out_path
    except Exception as e:
        logger.warning(f"Failed to save {task_name} image for {example_id}: {e}")
    return None


def _save_worldvqa_query_image(example: dict, base_dir: str = "tiles") -> str | None:
    """Save WorldVQA query image to disk. Returns path or None.
    Images saved to {base_dir}/worldvqa_images/{example_id}.png
    """
    img = example.get("image")
    if img is None:
        return None
    example_id = example.get("id", "unknown")
    save_dir = os.path.join(base_dir, "worldvqa_images")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{example_id}.png")

    try:
        if hasattr(img, "save"):
            img.save(out_path, format="PNG")
            return out_path
        if isinstance(img, str):
            raw = (
                img.split(",", 1)[1] if img.startswith("data:") and "," in img else img
            )
            data = base64.b64decode(raw)
            ext = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            out_path = os.path.join(save_dir, f"{example_id}{ext}")
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        if isinstance(img, dict) and "bytes" in img:
            b = img["bytes"]
            if b:
                ext = ".png" if b[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
                out_path = os.path.join(save_dir, f"{example_id}{ext}")
                with open(out_path, "wb") as f:
                    f.write(b)
                return out_path
    except Exception as e:
        logger.warning(f"Failed to save WorldVQA image for {example_id}: {e}")
    return None


def _worldvqa_image_to_base64(img) -> str | None:
    """Convert WorldVQA image (PIL, base64 str, or dict) to base64 string."""
    if img is None:
        return None
    if isinstance(img, str):
        if img.startswith("data:"):
            if "," in img:
                return img.split(",", 1)[1]
        return img
    if hasattr(img, "save"):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    if isinstance(img, dict) and "bytes" in img:
        b = img["bytes"]
        return base64.b64encode(b).decode() if b else None
    return None


class WorldVQANoRetrievalRetriever(BaseRetriever):
    """WorldVQA without retrieval: query + image from dataset only.

    WorldVQA images are embedded in the HuggingFace dataset (PIL or base64).
    """

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        img = example.get("image")
        base64_img = _worldvqa_image_to_base64(img)
        return RetrievalResult(
            base64_image=base64_img,
            retrieval_type="worldvqa_no_retrieval",
        )


class ScreenshotRetriever(BaseRetriever):
    """Use screenshot that was prepared in data layer.

    Expects screenshot to be captured beforehand. This retriever just
    loads and encodes the existing screenshot.

    For ground truth evaluation, uses encode_screenshot_for_vlm_async which
    does NOT apply max_height limit. You can control max_pixels to study
    the effect of resize on VLM performance.

    Args:
        screenshot_dir: Directory containing screenshots.
        max_pixels: Maximum pixels before resize. If None, no resize (89M limit).
                    Common values:
                    - None: No resize (let VLM handle it)
                    - 16_777_216 (16M): Qwen3-VL default, ~16K tokens
                    - 4_000_000 (4M): ~4K tokens
                    - 1_000_000 (1M): ~1K tokens
    """

    def __init__(
        self, screenshot_dir: str = "screenshots", max_pixels: int | None = None
    ):
        self.screenshot_dir = screenshot_dir
        self.max_pixels = max_pixels

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import (
            capture_screenshot_async,
            encode_screenshot_for_vlm_async,
            extract_url_from_metadata,
        )

        # Get or capture screenshot
        screenshot_path = await capture_screenshot_async(example, self.screenshot_dir)

        if not screenshot_path:
            return RetrievalResult(
                retrieval_type="screenshot",
                source_url=extract_url_from_metadata(example),
            )

        # Encode to base64 with configurable max_pixels
        base64_image = await encode_screenshot_for_vlm_async(
            screenshot_path, max_pixels=self.max_pixels
        )

        return RetrievalResult(
            base64_image=base64_image,
            source_url=extract_url_from_metadata(example),
            retrieval_type="screenshot",
        )


class TiledScreenshotRetriever(BaseRetriever):
    """Use tiled screenshot from ground truth URL.

    Captures screenshot for the example's URL, splits it into tiles,
    and returns tiles. This is ground truth (not vector search).

    Args:
        max_tiles: Maximum number of tiles to return. If None, returns all tiles.
                   For context-aware limiting, calculate based on model context length.
                   Rough estimate: max_tiles = (context_length - 2000) / tokens_per_tile
                   where tokens_per_tile ≈ 1500-2000 for most VLMs.
    """

    def __init__(
        self,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int = 512,
        overlap: int = 0,
        max_tiles: int | None = None,
    ):
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.max_tiles = max_tiles
        os.makedirs(tiles_dir, exist_ok=True)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import (
            capture_screenshot_async,
            encode_screenshot_async,
            extract_url_from_metadata,
            split_image_to_tiles,
        )

        # Get or capture screenshot
        screenshot_path = await capture_screenshot_async(example, self.screenshot_dir)

        if not screenshot_path:
            return RetrievalResult(
                retrieval_type="tiled_screenshot",
                source_url=extract_url_from_metadata(example),
            )

        # Split into tiles
        example_id = example.get("id", "unknown")
        example_tiles_dir = os.path.join(self.tiles_dir, example_id)
        tile_paths = split_image_to_tiles(
            screenshot_path,
            example_tiles_dir,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

        if not tile_paths:
            # Fall back to full screenshot
            base64_image = await encode_screenshot_async(screenshot_path)
            return RetrievalResult(
                base64_image=base64_image,
                source_url=extract_url_from_metadata(example),
                retrieval_type="tiled_screenshot",
            )

        # Limit tiles if max_tiles is set
        if self.max_tiles is not None and len(tile_paths) > self.max_tiles:
            logger.info(f"Limiting tiles from {len(tile_paths)} to {self.max_tiles}")
            tile_paths = tile_paths[: self.max_tiles]

        # Return tiles as images list (path, score=1.0 for ground truth)
        images = [(path, 1.0) for path in tile_paths]

        return RetrievalResult(
            images=images,
            source_url=extract_url_from_metadata(example),
            retrieval_type="tiled_screenshot",
        )


class LocalWikiTiledScreenshotRetriever(BaseRetriever):
    """Ground-truth tiled retriever using pre-rendered Wikipedia tiles from local kiwix.

    For each example, looks up the Wikipedia URL in the local kiwix tile store,
    copies raw tiles to a local cache, cuts into tile_height strips, and passes
    all tiles to the VLM as context. No Selenium, no SSH.

    Args:
        tiles_dir: Directory for cut tile strips (output).
        wiki_cache_dir: Directory for raw kiwix tile copies.
        tile_height: Height of each strip in pixels (default 1024).
        max_tiles: Maximum tiles to pass to VLM (None = all).
    """

    def __init__(
        self,
        tiles_dir: str = "tiles-local-wiki",
        wiki_cache_dir: str = "screenshots-localwiki",
        tile_height: int = 1024,
        max_tiles: int | None = None,
    ):
        self.tiles_dir = tiles_dir
        self.wiki_cache_dir = wiki_cache_dir
        self.tile_height = tile_height
        self.max_tiles = max_tiles
        os.makedirs(tiles_dir, exist_ok=True)
        os.makedirs(wiki_cache_dir, exist_ok=True)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import extract_url_from_metadata

        ex_id = example.get("id", "unknown")
        url = extract_url_from_metadata(example) or ""

        loop = asyncio.get_event_loop()
        try:
            tile_paths = await loop.run_in_executor(
                None,
                lambda: _lookup_and_copy_local_wiki_tiles(
                    ex_id, url, self.tiles_dir, self.wiki_cache_dir, self.tile_height
                ),
            )
        except RuntimeError as e:
            logger.error(f"local-wiki [{ex_id}]: {e}")
            return RetrievalResult(retrieval_type="local_wiki_tiled", source_url=url)

        if self.max_tiles is not None and len(tile_paths) > self.max_tiles:
            tile_paths = tile_paths[: self.max_tiles]

        images = [(path, 1.0) for path in tile_paths]
        return RetrievalResult(
            images=images,
            source_url=url,
            retrieval_type="local_wiki_tiled",
        )


class TextRetriever(BaseRetriever):
    """Use text content fetched from URL.

    Can use pre-cached text or fetch on demand.
    """

    def __init__(
        self,
        max_chars: int = 50000,
        text_cache: dict | None = None,
        cache_path: str | None = None,
    ):
        self.max_chars = max_chars
        self.text_cache = text_cache
        self.cache_path = cache_path
        self._cache_lock = asyncio.Lock()

    async def _save_to_cache(self, example_id: str, text: str, url: str):
        """Append result to cache file."""
        if not self.cache_path:
            return
        try:
            import json

            async with self._cache_lock:
                with open(self.cache_path, "a") as f:
                    cache_entry = {"id": example_id, "text": text, "url": url}
                    f.write(json.dumps(cache_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        from .simpleqa_data import fetch_text_async

        example_id = example.get("id", "")
        was_cached = self.text_cache and example_id in self.text_cache

        text, source_url = await fetch_text_async(
            example, self.max_chars, self.text_cache
        )

        # Save to cache if not already cached
        if not was_cached and text and source_url:
            await self._save_to_cache(example_id, text, source_url)

        return RetrievalResult(
            text=text, source_url=source_url, retrieval_type="text_rag"
        )


