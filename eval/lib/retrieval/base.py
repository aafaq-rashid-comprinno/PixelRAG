"""Base classes and utilities for retrieval strategies.

Defines RetrievalResult, BaseRetriever ABC, and shared helper functions
for tile lookup, image loading, and query image handling.
"""

import base64
import io
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    # Text content (for text-based retrieval)
    text: str | None = None

    # Image paths with scores (for vector retrieval)
    images: list[tuple[str, float]] = field(default_factory=list)

    # Per-image source URLs, aligned with ``images`` when provided.
    image_urls: list[str | None] = field(default_factory=list)

    # Base64 encoded image (for screenshot)
    base64_image: str | None = None

    # Source URL
    source_url: str | None = None

    # Which retrieval type was used
    retrieval_type: str = "naive"

    # Path to pixel query image used for retrieval embedding (rendered card or raw photo)
    pixel_query_path: str | None = None

    # Path to raw species/landmark photo for generation (always the original photo,
    # never the rendered card). If None, falls back to pixel_query_path in build_messages.
    query_image_path: str | None = None

    @property
    def has_content(self) -> bool:
        """Check if retrieval found any content."""
        return bool(self.text or self.images or self.base64_image)


class BaseRetriever(ABC):
    """Base class for retrieval strategies."""

    @abstractmethod
    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        """Retrieve relevant content for the query.

        Args:
            query: The question/query text.
            example: The full example dict (may contain metadata, prepared data, etc.).

        Returns:
            RetrievalResult with retrieved content.
        """
        raise NotImplementedError


# EVQA query image data dirs (iNaturalist 2021, Google Landmarks v2)
_INAT2021_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "inat2021",
)
_LANDMARK_V2_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "landmark_v2",
)

# Local kiwix tile store (pre-rendered Wikipedia pages)
_WIKI_SCREENSHOT_DIR = "/path/to/project"
_KIWIX_OUTPUT_DIR = "/path/to/data"
_KIWIX_ARTICLES_JSON = "/path/to/data"
_KIWIX_REDIRECTS_JSON = "/path/to/data"


def _lookup_and_copy_local_wiki_tiles(
    ex_id: str,
    url: str,
    tiles_dir: str,
    wiki_cache_dir: str,
    cut_height: int,
) -> list[str]:
    """Look up a Wikipedia URL in the local kiwix tile store, copy raw tiles, cut into strips.

    Args:
        ex_id: Example ID (used for output tile naming).
        url: Wikipedia URL.
        tiles_dir: Directory where cut tile strips are written ({ex_id}_tile_*.png).
        wiki_cache_dir: Directory where raw kiwix tile pages are cached ({ex_id}/).
        cut_height: Height of each output strip in pixels.

    Returns:
        Sorted list of cut tile paths.

    Raises:
        RuntimeError: If kiwix index unavailable, URL not found, or no tiles produced.
    """
    import glob as _glob
    import shutil
    import sys as _sys
    from PIL import Image

    # Return cached tiles if already cut
    existing = sorted(_glob.glob(os.path.join(tiles_dir, f"{ex_id}_tile_*.png")))
    if existing:
        return existing

    if not url or "wikipedia.org" not in url:
        raise RuntimeError(f"Not a Wikipedia URL: {url!r}")

    if not os.path.isdir(_KIWIX_OUTPUT_DIR) or not os.path.isfile(_KIWIX_ARTICLES_JSON):
        raise RuntimeError(f"kiwix tiles unavailable at {_KIWIX_OUTPUT_DIR}")

    if _WIKI_SCREENSHOT_DIR not in _sys.path:
        _sys.path.insert(0, _WIKI_SCREENSHOT_DIR)
    from scripts.build_index import batch_query_by_url as _batch_query

    redirects = _KIWIX_REDIRECTS_JSON if os.path.isfile(_KIWIX_REDIRECTS_JSON) else None
    results = _batch_query(
        _KIWIX_OUTPUT_DIR, [url], _KIWIX_ARTICLES_JSON, redirects_json=redirects
    )
    result = results.get(url)
    if result is None:
        raise RuntimeError(f"URL not found in local kiwix: {url}")

    # Copy raw kiwix tiles to wiki_cache_dir/{ex_id}/
    src_dir = os.path.join(_KIWIX_OUTPUT_DIR, result["tiles_dir"])
    article_cache = os.path.join(wiki_cache_dir, str(ex_id))
    if not os.path.exists(article_cache):
        if not os.path.isdir(src_dir):
            raise RuntimeError(f"kiwix tiles dir not on disk: {src_dir}")
        shutil.copytree(src_dir, article_cache)

    # Cut raw tiles into height=cut_height strips
    os.makedirs(tiles_dir, exist_ok=True)
    raw_tiles = sorted(
        f
        for f in os.listdir(article_cache)
        if f.endswith(".png") and f.startswith("tile_")
    )
    if not raw_tiles:
        raise RuntimeError(f"No tile PNGs found in {article_cache}")

    global_row = 0
    for raw_name in raw_tiles:
        raw_path = os.path.join(article_cache, raw_name)
        if os.path.getsize(raw_path) == 0:
            continue
        img = Image.open(raw_path)
        img.load()
        w, h = img.size
        y = 0
        while y < h:
            y2 = min(y + cut_height, h)
            strip = img.crop((0, y, w, y2))
            strip.save(os.path.join(tiles_dir, f"{ex_id}_tile_{global_row}_0.png"))
            strip.close()
            global_row += 1
            y += cut_height
        img.close()

    tile_paths = sorted(_glob.glob(os.path.join(tiles_dir, f"{ex_id}_tile_*.png")))
    if not tile_paths:
        raise RuntimeError(f"No strips cut for {ex_id} (source: {article_cache})")
    return tile_paths


def _get_inat_image_path_for_example(example: dict, tiles_dir: str) -> str | None:
    """Get iNaturalist 2021 query image path. dataset_name must be 'inaturalist'."""
    inat_ids = example.get("inat_image_ids", [])
    if not inat_ids:
        return None
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "inat_images")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    local_path = os.path.join(cache_dir, f"{example_id}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    import shutil

    id_map = TiledQwen3VLEmbeddingRetriever._load_inat2021_mapping()
    for str_id in inat_ids:
        try:
            img_id = int(str_id)
        except ValueError:
            continue
        file_name = id_map.get(img_id)
        if not file_name:
            continue
        src_path = os.path.join(_INAT2021_DATA_DIR, file_name)
        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            shutil.copy2(src_path, local_path)
            return local_path
    logger.warning(f"Failed to find iNaturalist image for {example_id}")
    return None


def _get_landmark_image_path_for_example(
    example: dict, tiles_dir: str, quiet: bool = False
) -> str | None:
    """Get Google Landmarks v2 query image path. dataset_name must be 'landmarks'.

    GLDv2 stores images as {split}/{a}/{b}/{c}/{id}.jpg (a,b,c = first 3 chars of id).
    Searches train, index, test in order.
    """
    ids = example.get("dataset_image_ids_parsed", [])
    if not ids:
        return None
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "landmark_images")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    local_path = os.path.join(cache_dir, f"{example_id}.jpg")
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    import shutil

    data_dir = _LANDMARK_V2_DATA_DIR
    for img_id in ids:
        if len(img_id) < 3:
            continue
        # GLDv2 path: {split}/{a}/{b}/{c}/{id}.jpg
        subpath = f"{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
        for split in ("train", "index", "test"):
            src_path = os.path.join(data_dir, split, subpath)
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                shutil.copy2(src_path, local_path)
                return local_path
    # Fallback: download from train.csv URL (requires data/landmark_v2/train.csv)
    # Try each img_id in order; first URL may be 404, others might work
    for img_id in ids:
        if _try_download_landmark_from_url(example_id, img_id, local_path):
            return local_path
    if not quiet:
        logger.warning(
            f"Failed to find Landmark image for {example_id} (data in {data_dir}?)"
        )
    return None


def _try_download_landmark_from_url(
    example_id: str, img_id: str, local_path: str
) -> bool:
    """Try to download landmark image from train.csv URL. Used when GLDv2 TARs unavailable.

    Returns True if download succeeded and file is valid, False otherwise.
    """
    import urllib.request

    train_csv = os.path.join(_LANDMARK_V2_DATA_DIR, "train.csv")
    if not os.path.exists(train_csv):
        return False
    import csv

    with open(train_csv) as f:
        for row in csv.DictReader(f):
            if row.get("id") == img_id:
                url = row.get("url", "")
                if url:
                    try:
                        req = urllib.request.Request(
                            url, headers={"User-Agent": "PixelRAG-Bot/1.0"}
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = resp.read()
                        if len(data) >= 1000:
                            with open(local_path, "wb") as out:
                                out.write(data)
                            return True
                    except Exception as e:
                        logger.debug(
                            f"URL download failed for {example_id} (img_id={img_id}): {e}"
                        )
                return False
    return False


def _get_query_image_path_for_example(
    example: dict, tiles_dir: str, quiet: bool = False
) -> str | None:
    """Get EVQA query image path. Dispatches by dataset_name: inaturalist | landmarks."""
    ds = (example.get("dataset_name") or "").lower()
    if ds == "inaturalist":
        return _get_inat_image_path_for_example(example, tiles_dir)
    if ds == "landmarks":
        return _get_landmark_image_path_for_example(example, tiles_dir, quiet=quiet)
    # Fallback: try inaturalist (backward compat when dataset_name missing)
    return _get_inat_image_path_for_example(example, tiles_dir)


def _get_all_inat_image_paths(example: dict, tiles_dir: str) -> list[str]:
    """Get ALL iNaturalist query image paths for an example (not just the first)."""
    inat_ids = example.get("inat_image_ids", [])
    if not inat_ids:
        return []
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "inat_images_multi")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    import shutil

    id_map = TiledQwen3VLEmbeddingRetriever._load_inat2021_mapping()
    paths = []
    for i, str_id in enumerate(inat_ids):
        local_path = os.path.join(cache_dir, f"{example_id}_{i}.jpg")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            paths.append(local_path)
            continue
        try:
            img_id = int(str_id)
        except ValueError:
            continue
        file_name = id_map.get(img_id)
        if not file_name:
            continue
        src_path = os.path.join(_INAT2021_DATA_DIR, file_name)
        if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
            shutil.copy2(src_path, local_path)
            paths.append(local_path)
    return paths


_landmark_url_map_cache: dict[str, str] | None = None


def _load_landmark_url_map() -> dict[str, str]:
    """Load GLDv2 train.csv: img_id -> url. Cached after first call."""
    global _landmark_url_map_cache
    if _landmark_url_map_cache is not None:
        return _landmark_url_map_cache
    import csv

    train_csv = os.path.join(_LANDMARK_V2_DATA_DIR, "train.csv")
    if not os.path.exists(train_csv):
        _landmark_url_map_cache = {}
        return _landmark_url_map_cache
    url_map = {}
    with open(train_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id = row.get("id", "").strip()
            url = row.get("url", "").strip()
            if img_id and url:
                url_map[img_id] = url
    _landmark_url_map_cache = url_map
    logger.info(f"Loaded landmark URL map: {len(url_map)} entries")
    return url_map


def _download_landmark_image_by_id(img_id: str, local_path: str) -> bool:
    """Download a landmark image by its GLDv2 ID. Returns True on success."""
    import urllib.request

    url_map = _load_landmark_url_map()
    url = url_map.get(img_id)
    if not url:
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PixelRAG-Bot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) >= 1000:
            with open(local_path, "wb") as out:
                out.write(data)
            return True
    except Exception as e:
        logger.debug(f"Download failed for landmark {img_id}: {e}")
    return False


def _get_all_landmark_image_paths(example: dict, tiles_dir: str) -> list[str]:
    """Get ALL Google Landmarks query image paths for an example (not just the first)."""
    ids = example.get("dataset_image_ids_parsed", [])
    if not ids:
        return []
    cache_dir = os.path.join(os.path.dirname(tiles_dir), "landmark_images_multi")
    os.makedirs(cache_dir, exist_ok=True)
    example_id = example.get("id", "unknown")
    import shutil

    data_dir = _LANDMARK_V2_DATA_DIR
    paths = []
    for i, img_id in enumerate(ids):
        local_path = os.path.join(cache_dir, f"{example_id}_{i}.jpg")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            paths.append(local_path)
            continue
        if len(img_id) < 3:
            continue
        subpath = f"{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.jpg"
        found = False
        for split in ("train", "index", "test"):
            src_path = os.path.join(data_dir, split, subpath)
            if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
                shutil.copy2(src_path, local_path)
                paths.append(local_path)
                found = True
                break
        if not found:
            if _download_landmark_image_by_id(img_id, local_path):
                paths.append(local_path)
    return paths


def _get_all_query_image_paths(example: dict, tiles_dir: str) -> list[str]:
    """Get ALL query image paths for an EVQA example (all available images, not just the first).

    Falls back to the single ``query_image_path`` / ``_get_query_image_path_for_example``
    when the multi-image helpers return nothing (e.g. ``dataset_image_ids_parsed`` lives
    inside ``original_data`` rather than at top level).
    """
    ds = (example.get("dataset_name") or "").lower()
    if ds not in ("inaturalist", "landmarks"):
        od = example.get("original_data", {})
        if isinstance(od, str):
            import ast

            try:
                od = ast.literal_eval(od)
            except Exception:
                od = {}
        ds = (od.get("dataset_name") or "").lower()
    if ds == "inaturalist":
        paths = _get_all_inat_image_paths(example, tiles_dir)
    elif ds == "landmarks":
        paths = _get_all_landmark_image_paths(example, tiles_dir)
    else:
        paths = _get_all_inat_image_paths(example, tiles_dir)
    if not paths:
        single = example.get("query_image_path") or _get_query_image_path_for_example(
            example, tiles_dir, quiet=True
        )
        if single and os.path.exists(single):
            paths = [single]
    return paths


