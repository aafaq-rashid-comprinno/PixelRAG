"""Vector similarity retrievers.

Retrievers that search pre-built embedding indexes:
- VectorRetriever: basic FAISS vector search
- ColQwenVectorRetriever: ColQwen-based retrieval
- TiledVectorRetriever: search over tiled screenshots
- TiledColQwenVectorRetriever: ColQwen over tiles
- TextVectorRetriever: text embedding search
"""

import base64
import io
import logging
import os

import numpy as np

from .base import BaseRetriever, RetrievalResult, _filter_tiles_by_aspect_ratio

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Retrieve similar screenshots using vector similarity search.

    Uses Jina API for embedding and retrieval across dataset screenshots only.
    """

    def __init__(
        self,
        api_key: str,
        screenshot_dir: str = "screenshots",
        cache_path: str | None = None,
        use_multivector: bool = True,
        top_k: int = 3,
        examples: list[dict] | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)

        # Prepare missing screenshots and get file paths
        screenshot_paths = self._prepare_screenshots()

        # Import retrieval system
        try:
            from scripts.jina_retrieval import JinaAPIRetrievalSystem
        except ImportError:
            try:
                from jina_retrieval import JinaAPIRetrievalSystem
            except ImportError:
                raise ImportError("JinaAPIRetrievalSystem not available")

        vector_type = "single vector" if not use_multivector else "multivector"
        logger.info(f"Initializing VectorRetriever with {vector_type} mode")

        self.retrieval_system = JinaAPIRetrievalSystem(
            api_key=api_key,
            use_multivector=use_multivector,
            device="cpu",  # Use CPU to avoid OOM when VLM is on GPU
        )
        # Only embed screenshots for current dataset
        self.retrieval_system.embed_images(
            file_paths=screenshot_paths, cache_path=cache_path
        )
        logger.info(
            f"VectorRetriever ready with {len(self.retrieval_system.image_paths)} images"
        )

    def _prepare_screenshots(self) -> list[str]:
        """Prepare screenshots for dataset and return list of paths."""
        from .simpleqa_data import capture_screenshot_for_example

        screenshot_paths = []
        missing = []

        for ex in self.examples:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        if missing:
            logger.info(
                f"Found {len(missing)} missing screenshots out of {len(self.examples)} total examples"
            )
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            # Use a more robust approach: continue even if some screenshots fail
            success_count = 0
            for ex in missing:
                try:
                    capture_screenshot_for_example(ex, self.screenshot_dir)
                    success_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to capture screenshot for {ex.get('id', 'unknown')}: {e}"
                    )
                    # Continue with next screenshot instead of failing completely
            logger.info(
                f"Screenshots prepared: {success_count}/{len(missing)} successful"
            )
        else:
            logger.info(
                f"All {len(self.examples)} screenshots already exist, skipping preparation"
            )

        # Return only existing screenshots
        return [
            p for p in screenshot_paths if os.path.exists(p) and os.path.getsize(p) > 0
        ]

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                return RetrievalResult(images=results, retrieval_type="vector")
        except Exception as e:
            logger.warning(f"Vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="vector")


class ColQwenVectorRetriever(BaseRetriever):
    """Retrieve similar screenshots using ColQwen2 LEANN multi-vector retrieval."""

    def __init__(
        self,
        index_path: str,
        screenshot_dir: str = "screenshots",
        model_name: str = "colqwen2",
        search_method: str = "ann",
        first_stage_k: int = 500,
        rebuild_index: bool = False,
        recursive: bool = False,
        top_k: int = 3,
        examples: list[dict] | None = None,
        prepare_screenshots: bool = False,  # ColQwen2 doesn't need to prepare specific screenshots
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)

        # Build list of image paths for the specific examples (only Wikipedia samples)
        image_paths = self._get_example_image_paths()

        if image_paths:
            logger.info(
                f"ColQwen2 will retrieve from {len(image_paths)} images for {len(self.examples)} examples"
            )
        else:
            logger.warning(
                f"No images found for examples, falling back to all images in: {screenshot_dir}"
            )

        # Import ColQwen2 retrieval system
        import sys
        from pathlib import Path

        # Add scripts directory to path for import
        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
        except ImportError:
            try:
                from scripts.colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
            except ImportError:
                raise ImportError(
                    "ColQwenLEANNRetrievalSystem not available. Make sure colqwen_leann_retrieval.py is in the scripts directory."
                )

        logger.info("Initializing ColQwen2 LEANN retrieval system...")
        logger.info(f"Search method: {search_method}")

        # Use filtered image paths if available, otherwise fall back to directory scanning
        if image_paths:
            self.retrieval_system = ColQwenLEANNRetrievalSystem(
                index_path=index_path,
                model_name=model_name,
                search_method=search_method,
                first_stage_k=first_stage_k,
                rebuild_index=rebuild_index,
                custom_image_paths=image_paths,  # Pass specific image paths
            )
        else:
            self.retrieval_system = ColQwenLEANNRetrievalSystem(
                index_path=index_path,
                model_name=model_name,
                search_method=search_method,
                first_stage_k=first_stage_k,
                rebuild_index=rebuild_index,
                custom_folder_path=screenshot_dir,
                custom_folder_recursive=recursive,
            )
        logger.info("ColQwen2 LEANN retrieval system ready")

    def _get_example_image_paths(self) -> list[str]:
        """Get image paths for the specific examples."""
        image_paths = []
        for ex in self.examples:
            example_id = ex.get("id", "")
            if not example_id:
                continue
            path = os.path.join(self.screenshot_dir, f"{example_id}_fullhd.png")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                image_paths.append(path)
        return image_paths

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                return RetrievalResult(images=results, retrieval_type="colqwen_vector")
        except Exception as e:
            logger.warning(f"ColQwen2 vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="colqwen_vector")


def _filter_tiles_by_aspect_ratio(
    tile_paths: list[str], max_aspect_ratio: float = 100.0
) -> list[str]:
    """Filter out tiles with extreme aspect ratios.

    Args:
        tile_paths: List of tile image paths.
        max_aspect_ratio: Maximum allowed aspect ratio (default 100, ColQwen requires < 200).

    Returns:
        Filtered list of tile paths.
    """
    from PIL import Image

    filtered = []
    for tile_path in tile_paths:
        try:
            with Image.open(tile_path) as img:
                w, h = img.size
                if w > 0 and h > 0:
                    aspect_ratio = max(w / h, h / w)
                    if aspect_ratio <= max_aspect_ratio:
                        filtered.append(tile_path)
                    else:
                        logger.warning(
                            f"Skipping tile with extreme aspect ratio {aspect_ratio:.2f}: {tile_path}"
                        )
        except Exception as e:
            logger.warning(f"Failed to check tile {tile_path}: {e}")

    return filtered


class TiledVectorRetriever(BaseRetriever):
    """Retrieve similar image tiles using vector similarity search.

    Splits dataset screenshots into fixed-size tiles, embeds each tile,
    and retrieves the most relevant tiles for a query.
    """

    def __init__(
        self,
        api_key: str,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int = 512,
        overlap: int = 0,
        cache_path: str | None = None,
        use_multivector: bool = True,
        top_k: int = 3,
        examples: list[dict] | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        # Build example_id -> URL mapping (prioritize Wikipedia URLs)
        from .simpleqa_data import extract_url_from_metadata

        self.id_to_url = {}
        for ex in self.examples:
            ex_id = ex.get("id", "")
            url = extract_url_from_metadata(ex)  # Uses Wikipedia-first priority
            if url:
                self.id_to_url[ex_id] = url

        # Prepare screenshots and get tile paths
        tile_paths = self._prepare_screenshots_and_tiles()

        # Import retrieval system
        try:
            from scripts.jina_retrieval import JinaAPIRetrievalSystem
        except ImportError:
            try:
                from jina_retrieval import JinaAPIRetrievalSystem
            except ImportError:
                raise ImportError("JinaAPIRetrievalSystem not available")

        vector_type = "single vector" if not use_multivector else "multivector"
        logger.info(f"Initializing TiledVectorRetriever with {vector_type} mode")

        self.retrieval_system = JinaAPIRetrievalSystem(
            api_key=api_key,
            use_multivector=use_multivector,
            device="cpu",  # Use CPU to avoid OOM when VLM is on GPU
        )
        # Only embed tiles for current dataset
        self.retrieval_system.embed_images(file_paths=tile_paths, cache_path=cache_path)
        logger.info(
            f"TiledVectorRetriever ready with {len(self.retrieval_system.image_paths)} tiles"
        )

    def _prepare_screenshots_and_tiles(self) -> list[str]:
        """Prepare screenshots and tiles for dataset, return tile paths."""
        from .simpleqa_data import capture_screenshot_for_example, split_image_to_tiles
        from tqdm import tqdm

        screenshot_paths = []
        missing = []

        # Collect screenshot paths and identify missing
        for ex in self.examples:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        # Capture missing screenshots
        if missing:
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            for ex in tqdm(missing, desc="Capturing screenshots"):
                capture_screenshot_for_example(ex, self.screenshot_dir)
            logger.info("Screenshots prepared.")

        # Split each screenshot into tiles
        all_tile_paths = []
        logger.info(
            f"Splitting {len(screenshot_paths)} screenshots into tiles (output: {self.tiles_dir})..."
        )
        for screenshot_path in tqdm(screenshot_paths, desc="Splitting tiles"):
            if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                tile_paths = split_image_to_tiles(
                    screenshot_path, self.tiles_dir, self.tile_size, self.overlap
                )
                all_tile_paths.extend(tile_paths)

        # Filter out tiles with extreme aspect ratios
        filtered_tile_paths = _filter_tiles_by_aspect_ratio(all_tile_paths)
        logger.info(
            f"Prepared {len(filtered_tile_paths)} tiles from {len(screenshot_paths)} screenshots (filtered {len(all_tile_paths) - len(filtered_tile_paths)} extreme aspect ratio tiles)"
        )
        return filtered_tile_paths

    def _extract_urls_from_results(self, results: list) -> str:
        """Extract source URLs from tile paths in results, preserving retrieval order."""
        urls = []
        seen = set()
        for item in results:
            # item is (path, score) tuple
            path = item[0] if isinstance(item, tuple) else item
            # Extract example_id from tile path: {example_id}_fullhd_tile_{x}_{y}.png
            filename = os.path.basename(path)
            # Split by _fullhd_ or just get the first part before _tile_
            if "_tile_" in filename:
                example_id = filename.split("_tile_")[0]
                # Remove _fullhd suffix if present
                if example_id.endswith("_fullhd"):
                    example_id = example_id[:-7]
                if example_id in self.id_to_url:
                    url = self.id_to_url[example_id]
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        return ", ".join(urls)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        del example  # Not used - retrieval is from pre-built index
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                source_url = self._extract_urls_from_results(results)
                return RetrievalResult(
                    images=results, source_url=source_url, retrieval_type="tiled_vector"
                )
        except Exception as e:
            logger.warning(f"Tiled vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="tiled_vector")


class TiledColQwenVectorRetriever(BaseRetriever):
    """Retrieve similar image tiles using ColQwen2 LEANN multi-vector retrieval.

    Splits dataset screenshots into fixed-size tiles, embeds each tile with ColQwen2,
    and retrieves the most relevant tiles for a query using LEANN.
    """

    def __init__(
        self,
        index_path: str,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int = 512,
        overlap: int = 0,
        model_name: str = "colqwen2",
        search_method: str = "ann",
        first_stage_k: int = 500,
        rebuild_index: bool = False,
        top_k: int = 3,
        examples: list[dict] | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.examples = examples or []
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        # Build example_id -> URL mapping (prioritize Wikipedia URLs)
        from .simpleqa_data import extract_url_from_metadata

        self.id_to_url = {}
        for ex in self.examples:
            ex_id = ex.get("id", "")
            url = extract_url_from_metadata(ex)  # Uses Wikipedia-first priority
            if url:
                self.id_to_url[ex_id] = url

        # Prepare screenshots and get tile paths
        tile_paths = self._prepare_screenshots_and_tiles()

        # Import ColQwen2 retrieval system
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
        except ImportError:
            try:
                from scripts.colqwen_leann_retrieval import ColQwenLEANNRetrievalSystem
            except ImportError:
                raise ImportError("ColQwenLEANNRetrievalSystem not available.")

        logger.info("Initializing TiledColQwen2 LEANN retrieval system...")
        logger.info(f"Search method: {search_method}, tiles: {len(tile_paths)}")

        self.retrieval_system = ColQwenLEANNRetrievalSystem(
            index_path=index_path,
            custom_image_paths=tile_paths,
            model_name=model_name,
            search_method=search_method,
            first_stage_k=first_stage_k,
            rebuild_index=rebuild_index,
        )

        logger.info(
            f"TiledColQwen2 LEANN retrieval system ready with {len(tile_paths)} tiles"
        )

    def _prepare_screenshots_and_tiles(self) -> list[str]:
        """Prepare screenshots and tiles for dataset, return tile paths."""
        from .simpleqa_data import capture_screenshot_for_example, split_image_to_tiles
        from tqdm import tqdm

        screenshot_paths = []
        missing = []

        # Collect screenshot paths and identify missing
        for ex in self.examples:
            screenshot_path = os.path.join(
                self.screenshot_dir, f"{ex['id']}_fullhd.png"
            )
            screenshot_paths.append(screenshot_path)
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                missing.append(ex)

        # Capture missing screenshots
        if missing:
            logger.info(f"Preparing {len(missing)} missing screenshots...")
            for ex in tqdm(missing, desc="Capturing screenshots"):
                capture_screenshot_for_example(ex, self.screenshot_dir)
            logger.info("Screenshots prepared.")

        # Split each screenshot into tiles
        all_tile_paths = []
        logger.info(
            f"Splitting {len(screenshot_paths)} screenshots into tiles (output: {self.tiles_dir})..."
        )
        for screenshot_path in tqdm(screenshot_paths, desc="Splitting tiles"):
            if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 0:
                tile_paths = split_image_to_tiles(
                    screenshot_path, self.tiles_dir, self.tile_size, self.overlap
                )
                all_tile_paths.extend(tile_paths)

        # Filter out tiles with extreme aspect ratios
        filtered_tile_paths = _filter_tiles_by_aspect_ratio(all_tile_paths)
        logger.info(
            f"Prepared {len(filtered_tile_paths)} tiles from {len(screenshot_paths)} screenshots (filtered {len(all_tile_paths) - len(filtered_tile_paths)} extreme aspect ratio tiles)"
        )
        return filtered_tile_paths

    def _extract_urls_from_results(self, results: list) -> str:
        """Extract source URLs from tile paths in results, preserving retrieval order."""
        urls = []
        seen = set()
        for item in results:
            # item is (path, score) tuple
            path = item[0] if isinstance(item, tuple) else item
            # Extract example_id from tile path: {example_id}_fullhd_tile_{x}_{y}.png
            filename = os.path.basename(path)
            if "_tile_" in filename:
                example_id = filename.split("_tile_")[0]
                if example_id.endswith("_fullhd"):
                    example_id = example_id[:-7]
                if example_id in self.id_to_url:
                    url = self.id_to_url[example_id]
                    if url not in seen:
                        seen.add(url)
                        urls.append(url)
        return ", ".join(urls)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        del example  # Not used - retrieval is from pre-built index
        loop = asyncio.get_event_loop()

        try:
            results = await loop.run_in_executor(
                None, self.retrieval_system.retrieve, query, self.top_k
            )

            if results:
                source_url = self._extract_urls_from_results(results)
                return RetrievalResult(
                    images=results,
                    source_url=source_url,
                    retrieval_type="tiled_colqwen_vector",
                )
        except Exception as e:
            logger.warning(f"TiledColQwen2 vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="tiled_colqwen_vector")


class TextVectorRetriever(BaseRetriever):
    """Retrieve text using LEANN vector search.

    Uses LEANN's integrated embedding + indexing system for text retrieval.
    Supports various embedding models (Qwen3, nomic-embed-text, OpenAI, etc.)
    """

    def __init__(
        self,
        text_cache: dict,
        index_path: str,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        embedding_mode: str = "sentence-transformers",
        embedding_options: dict | None = None,
        top_k: int = 3,
        rebuild_index: bool = False,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """Initialize TextVectorRetriever.

        Args:
            text_cache: Dict of {id: {"text": ..., "url": ...}}
            index_path: Path to LEANN index
            embedding_model: Embedding model name (default: Qwen/Qwen3-Embedding-0.6B)
            embedding_mode: Embedding mode (sentence-transformers, openai, gemini, ollama)
            embedding_options: Additional options for embedding (e.g., base_url, api_key for OpenAI-compatible APIs)
            top_k: Number of results to retrieve
            rebuild_index: Force rebuild index even if exists
            chunk_size: Max tokens per chunk (default: 512)
            chunk_overlap: Overlap tokens between chunks (default: 128)
        """
        import sys
        from pathlib import Path as PathLib

        # Add LEANN to path
        leann_path = (
            PathLib(__file__).parent.parent.parent
            / "LEANN"
            / "packages"
            / "leann-core"
            / "src"
        )
        if str(leann_path) not in sys.path:
            sys.path.insert(0, str(leann_path))

        from leann.api import LeannBuilder, LeannSearcher

        self.text_cache = text_cache
        self.top_k = top_k
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.embedding_mode = embedding_mode
        self.embedding_options = embedding_options or {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Check if index exists
        meta_path = f"{index_path}.meta.json"
        index_exists = os.path.exists(meta_path)

        if rebuild_index or not index_exists:
            logger.info(f"Building LEANN text index at {index_path}...")
            self._build_index(LeannBuilder)
            logger.info(f"LEANN text index built with {len(text_cache)} documents")
        else:
            logger.info(f"Loading existing LEANN text index from {index_path}")

        # Load searcher
        self.searcher = LeannSearcher(index_path)
        logger.info(
            f"TextVectorRetriever ready with {len(text_cache)} documents, top_k={top_k}"
        )

    def _build_index(self, LeannBuilder):
        """Build LEANN index from text_cache with chunking for long texts."""
        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model=self.embedding_model,
            embedding_mode=self.embedding_mode,
            embedding_options=self.embedding_options,
            is_recompute=False,  # Store embeddings to avoid recomputing at search time
        )

        # Chunking parameters (from CLI or defaults)
        max_tokens = self.chunk_size
        overlap_tokens = self.chunk_overlap

        # Import tiktoken for accurate chunking
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            enc = None
            logger.warning("tiktoken not available, using character-based chunking")

        chunk_count = 0
        for example_id, data in self.text_cache.items():
            text = data.get("text", "")
            url = data.get("url", "")
            if not text:
                continue

            if enc:
                # Token-based chunking
                tokens = enc.encode(text)
                if len(tokens) <= max_tokens:
                    # Short text, add as single passage
                    builder.add_text(text, metadata={"id": example_id, "url": url})
                    chunk_count += 1
                else:
                    # Long text, chunk it with overlap
                    start = 0
                    chunk_idx = 0
                    while start < len(tokens):
                        end = min(start + max_tokens, len(tokens))
                        chunk_tokens = tokens[start:end]
                        chunk_text = enc.decode(chunk_tokens)

                        chunk_id = f"{example_id}_chunk_{chunk_idx}"
                        builder.add_text(
                            chunk_text,
                            metadata={
                                "id": chunk_id,
                                "original_id": example_id,
                                "url": url,
                                "chunk_idx": chunk_idx,
                            },
                        )
                        chunk_count += 1
                        chunk_idx += 1

                        if end >= len(tokens):
                            break
                        start = end - overlap_tokens  # Overlap
            else:
                # Fallback: character-based chunking (~4 chars per token)
                max_chars = max_tokens * 4
                overlap_chars = overlap_tokens * 4

                if len(text) <= max_chars:
                    builder.add_text(text, metadata={"id": example_id, "url": url})
                    chunk_count += 1
                else:
                    start = 0
                    chunk_idx = 0
                    while start < len(text):
                        end = min(start + max_chars, len(text))
                        chunk_text = text[start:end]

                        chunk_id = f"{example_id}_chunk_{chunk_idx}"
                        builder.add_text(
                            chunk_text,
                            metadata={
                                "id": chunk_id,
                                "original_id": example_id,
                                "url": url,
                                "chunk_idx": chunk_idx,
                            },
                        )
                        chunk_count += 1
                        chunk_idx += 1

                        if end >= len(text):
                            break
                        start = end - overlap_chars

        logger.info(
            f"Created {chunk_count} chunks from {len(self.text_cache)} documents"
        )

        # Build index
        builder.build_index(self.index_path)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        """Retrieve relevant texts using LEANN vector search."""
        del example  # Not used - retrieval is from pre-built index
        loop = asyncio.get_event_loop()

        try:
            # Run search in executor (LEANN search is sync)
            results = await loop.run_in_executor(
                None,
                lambda: self.searcher.search(
                    query, top_k=self.top_k, recompute_embeddings=False
                ),
            )

            if results:
                # Combine retrieved texts
                texts = []
                urls = []
                for r in results:
                    texts.append(r.text)
                    url = r.metadata.get("url", "") if r.metadata else ""
                    urls.append(url)

                combined_text = "\n\n---\n\n".join(texts)
                combined_urls = ", ".join(u for u in urls if u)

                return RetrievalResult(
                    text=combined_text,
                    source_url=combined_urls,
                    retrieval_type="text_vector",
                )
        except Exception as e:
            logger.warning(f"Text vector retrieval failed: {e}")

        return RetrievalResult(retrieval_type="text_vector")


