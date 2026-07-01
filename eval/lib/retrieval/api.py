"""API-based retrievers.

Retrievers that call external search APIs:
- DsServeRetriever: direct serve API client
- LocalAPIRetriever: local PixelRAG search API
- TiledQwen3VLEmbeddingRetriever: Qwen3-VL embedding via API
- TextAPIRetriever: text-based API search
"""

import asyncio
import base64
import io
import json
import logging
import os
import time

import numpy as np

from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class DsServeRetriever(BaseRetriever):
    """Use ds-serve API for external text augmentation.

    Calls ds-serve search API to retrieve relevant text passages for the query.
    """

    def __init__(
        self, api_url: str = "http://api.ds-serve.org:30888/search", top_k: int = 3
    ):
        self.api_url = api_url
        self.top_k = top_k

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        import aiohttp
        import asyncio

        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                payload = {"query": query}

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status == 200:
                            result = await response.json()

                            # Extract passages from response
                            passages = []
                            if "results" in result and "passages" in result["results"]:
                                # passages is a list of lists, get the first list
                                passage_list = (
                                    result["results"]["passages"][0]
                                    if result["results"]["passages"]
                                    else []
                                )

                                # Take top_k passages
                                for i, passage_data in enumerate(
                                    passage_list[: self.top_k]
                                ):
                                    if isinstance(passage_data, dict):
                                        text = passage_data.get(
                                            "text", ""
                                        ) or passage_data.get("center_text", "")
                                        if text:
                                            passages.append(text)

                            # Combine passages into context text
                            if passages:
                                combined_text = "\n\n".join(
                                    [
                                        f"[Passage {i + 1}]\n{text}"
                                        for i, text in enumerate(passages)
                                    ]
                                )

                                return RetrievalResult(
                                    text=combined_text,
                                    source_url=f"ds-serve:{self.api_url}",
                                    retrieval_type="ds_serve",
                                )
                            else:
                                return RetrievalResult(
                                    text="No passages found from ds-serve.",
                                    source_url=f"ds-serve:{self.api_url}",
                                    retrieval_type="ds_serve",
                                )
                        elif response.status == 429:
                            if attempt < max_retries - 1:
                                wait_time = min(2**attempt * 2, 10)
                                logger.warning(
                                    f"Rate limited (429), waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                error_msg = f"ds-serve API rate limited after {max_retries} retries"
                                logger.error(error_msg)
                                return RetrievalResult(
                                    text=error_msg, retrieval_type="ds_serve"
                                )
                        else:
                            error_text = await response.text()
                            error_msg = f"ds-serve API error: {response.status} - {error_text[:200]}"
                            logger.error(error_msg)
                            return RetrievalResult(
                                text=error_msg, retrieval_type="ds_serve"
                            )
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 5)
                    logger.warning(
                        f"Timeout, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"ds-serve API timeout after {max_retries} retries"
                    logger.error(error_msg)
                    return RetrievalResult(text=error_msg, retrieval_type="ds_serve")
            except Exception as e:
                error_msg = f"ds-serve API call failed: {e}"
                logger.error(error_msg)
                return RetrievalResult(text=error_msg, retrieval_type="ds_serve")

        return RetrievalResult(
            text="ds-serve API call failed after all retries", retrieval_type="ds_serve"
        )


class LocalAPIRetriever(BaseRetriever):
    """Retrieve tiles from a local search API (e.g. localhost:30888/search).

    The API accepts batch queries:
        {"queries": [{"text": "..."}, ...], "n_docs": N}
    and returns:
        {"results": [{"hits": [{"path": ..., "url": ..., "score": ...}, ...]}, ...]}

    Call prefetch(examples) before the main loop to batch all queries in one API
    call. Individual retrieve() calls then return cached results instantly.

    When query_rewrite is enabled, uses an LLM to rewrite questions into
    keyword-rich search queries before retrieval.
    """

    REWRITE_PROMPT = (
        "You are a search query optimizer. Given a trivia/factual question, "
        "rewrite it as a Wikipedia search query that would find the article "
        "containing the answer. Output ONLY the search query, nothing else.\n\n"
        "Rules:\n"
        "- Focus on the key entity or topic the question is about\n"
        "- Include all specific names, dates, awards, events, or other details mentioned\n"
        "- Remove filler words like 'what is', 'who was', 'in which year'\n"
        "- Preserve all proper nouns and technical terms exactly as written\n\n"
        "Question: {question}\n"
        "Search query:"
    )

    def __init__(
        self,
        api_url: str = "http://localhost:30888/search",
        top_k: int = 5,
        batch_size: int = 32,
        query_rewrite: bool = False,
        rewrite_model: str | None = None,
        rewrite_api_base: str | None = None,
        rewrite_api_key: str = "dummy",
        nprobe: int | None = None,
        reranker=None,
        rerank_top_k: int = 3,
        query_image_fn=None,
        multi_image_query: bool = False,
        tiles_dir: str = "tiles/evqa",
        lookup_reference_url: bool = False,
        query_instruction: str | None = None,
    ):
        self.api_url = api_url
        self.top_k = top_k
        self.batch_size = batch_size
        self.query_rewrite = query_rewrite
        self.rewrite_model = rewrite_model
        self.rewrite_api_base = rewrite_api_base
        self.rewrite_api_key = rewrite_api_key
        self.nprobe = nprobe
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k
        self.query_image_fn = query_image_fn  # callable(example) -> image_path or None
        self.multi_image_query = multi_image_query
        self.tiles_dir = tiles_dir
        self.lookup_reference_url = lookup_reference_url
        self.query_instruction = query_instruction
        self._cache: dict[str, list[dict]] = {}  # example_id -> hits
        self._rewritten_queries: dict[str, str] = {}  # example_id -> rewritten query

    async def _rewrite_queries(self, examples: list[dict]) -> dict[str, str]:
        """Batch-rewrite questions into search queries using an LLM."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.rewrite_api_key,
            base_url=self.rewrite_api_base,
            timeout=60.0,
        )

        rewritten = {}
        sem = asyncio.Semaphore(20)

        async def rewrite_one(ex):
            eid = ex.get("id", "unknown")
            prompt = self.REWRITE_PROMPT.format(question=ex["problem"])
            async with sem:
                try:
                    resp = await client.chat.completions.create(
                        model=self.rewrite_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200,
                    )
                    rewritten[eid] = resp.choices[0].message.content.strip()
                except Exception as e:
                    logger.warning(f"Query rewrite failed for {eid}: {e}")
                    rewritten[eid] = ex["problem"]  # fallback to original

        await asyncio.gather(*[rewrite_one(ex) for ex in examples])
        return rewritten

    def _lookup_reference_tiles(self, examples: list[dict]) -> dict[str, list[dict]]:
        """Look up reference URL tiles from kiwix for each example.

        Returns dict: example_id -> list of hit dicts with path/score/url/is_reference.
        """
        import sys as _sys
        from .simpleqa_data import extract_url_from_metadata

        if not os.path.isdir(_KIWIX_OUTPUT_DIR) or not os.path.isfile(
            _KIWIX_ARTICLES_JSON
        ):
            logger.error(
                f"lookup_reference_url: kiwix tiles unavailable at {_KIWIX_OUTPUT_DIR}"
            )
            return {}

        if _WIKI_SCREENSHOT_DIR not in _sys.path:
            _sys.path.insert(0, _WIKI_SCREENSHOT_DIR)
        from scripts.build_index import batch_query_by_url as _batch_query

        # Collect URLs, group by URL to avoid duplicate lookups
        url_to_eids: dict[str, list[str]] = {}
        for ex in examples:
            eid = ex.get("id", "unknown")
            url = extract_url_from_metadata(ex)
            if url and "wikipedia.org" in url:
                url_to_eids.setdefault(url, []).append(eid)

        if not url_to_eids:
            return {}

        redirects = (
            _KIWIX_REDIRECTS_JSON if os.path.isfile(_KIWIX_REDIRECTS_JSON) else None
        )
        results = _batch_query(
            _KIWIX_OUTPUT_DIR,
            list(url_to_eids.keys()),
            _KIWIX_ARTICLES_JSON,
            redirects_json=redirects,
        )

        ref_tiles: dict[str, list[dict]] = {}
        found, missing = 0, 0
        for url, eids in url_to_eids.items():
            result = results.get(url)
            if result is None:
                missing += 1
                logger.warning(f"lookup_reference_url: URL not found in kiwix: {url}")
                continue
            tiles_dir_abs = os.path.join(_KIWIX_OUTPUT_DIR, result["tiles_dir"])
            if not os.path.isdir(tiles_dir_abs):
                missing += 1
                logger.warning(
                    f"lookup_reference_url: tiles dir missing: {tiles_dir_abs}"
                )
                continue
            chunks = sorted(
                f
                for f in os.listdir(tiles_dir_abs)
                if f.startswith("chunk_") and f.endswith(".png")
            )
            if not chunks:
                missing += 1
                logger.warning(
                    f"lookup_reference_url: no chunk files in {tiles_dir_abs}"
                )
                continue
            found += 1
            hits = [
                {
                    "path": os.path.join(tiles_dir_abs, c),
                    "score": 0.0,
                    "url": url,
                    "is_reference": True,
                }
                for c in chunks
            ]
            for eid in eids:
                ref_tiles[eid] = hits

        logger.info(
            f"lookup_reference_url: batch lookup {found} found, {missing} missing "
            f"out of {len(url_to_eids)} unique URLs"
        )
        return ref_tiles

    async def prefetch(self, examples: list[dict]):
        """Batch-fetch retrieval results for all examples via the API."""
        import aiohttp

        # Step 1: Query rewriting (if enabled)
        if self.query_rewrite and self.rewrite_model:
            to_rewrite = [
                ex
                for ex in examples
                if ex.get("id", "unknown") not in self._rewritten_queries
            ]
            if to_rewrite:
                logger.info(
                    f"LocalAPIRetriever: rewriting {len(to_rewrite)} queries..."
                )
                self._rewritten_queries.update(await self._rewrite_queries(to_rewrite))
                # Log some examples
                for ex in to_rewrite[:3]:
                    eid = ex.get("id", "unknown")
                    orig = ex["problem"][:60]
                    rewr = self._rewritten_queries.get(eid, "")[:60]
                    logger.info(f"  Rewrite: '{orig}...' -> '{rewr}'")

        # Step 2: Build query list
        queries = []
        example_ids = []

        if self.multi_image_query:
            # Multi-image: send one query per image, track which example each belongs to
            # We'll aggregate after receiving results
            multi_image_groups: dict[
                str, list[int]
            ] = {}  # eid -> list of indices in queries[]
            for ex in examples:
                eid = ex.get("id", "unknown")
                if eid in self._cache:
                    continue
                if self.query_rewrite and eid in self._rewritten_queries:
                    query_text = self._rewritten_queries[eid]
                else:
                    query_text = ex["problem"]

                all_paths = _get_all_query_image_paths(ex, self.tiles_dir)
                if len(all_paths) <= 1:
                    # Single or no image: just use the standard path
                    query_dict = {"text": query_text}
                    if all_paths:
                        import base64

                        with open(all_paths[0], "rb") as f:
                            query_dict["image"] = base64.b64encode(f.read()).decode()
                    elif self.query_image_fn:
                        img_path = self.query_image_fn(ex)
                        if img_path and os.path.exists(img_path):
                            import base64

                            with open(img_path, "rb") as f:
                                query_dict["image"] = base64.b64encode(
                                    f.read()
                                ).decode()
                    multi_image_groups[eid] = [len(queries)]
                    queries.append(query_dict)
                    example_ids.append(eid)
                else:
                    # Multiple images: one query per image
                    group_indices = []
                    import base64

                    for img_path in all_paths:
                        query_dict = {"text": query_text}
                        with open(img_path, "rb") as f:
                            query_dict["image"] = base64.b64encode(f.read()).decode()
                        group_indices.append(len(queries))
                        queries.append(query_dict)
                        example_ids.append(eid)
                    multi_image_groups[eid] = group_indices
                    logger.info(
                        f"Multi-image query for {eid[:8]}: {len(all_paths)} images"
                    )
        else:
            for ex in examples:
                eid = ex.get("id", "unknown")
                if eid in self._cache:
                    continue
                if self.query_rewrite and eid in self._rewritten_queries:
                    query_text = self._rewritten_queries[eid]
                else:
                    query_text = ex["problem"]
                query_dict = {"text": query_text}
                if self.query_image_fn:
                    img_path = self.query_image_fn(ex)
                    if img_path and os.path.exists(img_path):
                        import base64

                        with open(img_path, "rb") as f:
                            query_dict["image"] = base64.b64encode(f.read()).decode()
                queries.append(query_dict)
                example_ids.append(eid)

        if not queries:
            logger.info("LocalAPIRetriever: all examples already cached")
            return

        # Use smaller batches when queries contain images (GPU memory)
        has_images = any("image" in q for q in queries)
        batch_size = min(self.batch_size, 16) if has_images else self.batch_size
        logger.info(
            f"LocalAPIRetriever: prefetching {len(queries)} queries in batches of {batch_size}"
            f"{' (multimodal)' if has_images else ''}"
        )

        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start : batch_start + batch_size]
            batch_ids = example_ids[batch_start : batch_start + batch_size]

            n_docs = self.top_k * 2 if self.multi_image_query else self.top_k
            payload = {
                "queries": batch_queries,
                "n_docs": n_docs,
                "include_images": True,
            }
            if self.nprobe is not None:
                payload["nprobe"] = self.nprobe
            if self.query_instruction is not None:
                payload["instruction"] = self.query_instruction
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=600),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Local API batch error {response.status}: {error_text[:200]}"
                            )
                            for eid in batch_ids:
                                self._cache[eid] = []
                            continue
                        result = await response.json()
            except Exception as e:
                logger.error(f"Local API batch call failed: {e}")
                for eid in batch_ids:
                    self._cache[eid] = []
                continue

            results_list = result.get("results", [])
            for i, eid in enumerate(batch_ids):
                if i < len(results_list):
                    hits = results_list[i].get("hits", [])
                else:
                    hits = []
                if eid not in self._cache:
                    self._cache[eid] = hits
                else:
                    # Multi-image: accumulate hits from all images for this example
                    self._cache[eid].extend(hits)

            logger.info(
                f"  Batch {batch_start // batch_size + 1}/{(len(queries) + batch_size - 1) // batch_size}: "
                f"{len(batch_queries)} queries done"
            )

        # Multi-image aggregation: deduplicate and keep max score per tile path
        if self.multi_image_query:
            for eid in list(self._cache.keys()):
                hits = self._cache[eid]
                if not hits:
                    continue
                # Aggregate by path: keep hit with max score
                best_by_path: dict[str, dict] = {}
                for hit in hits:
                    path = hit.get("path", "")
                    score = hit.get("score", 0.0)
                    if path not in best_by_path or score > best_by_path[path].get(
                        "score", 0.0
                    ):
                        best_by_path[path] = hit
                # Sort by score descending, take top_k
                sorted_hits = sorted(
                    best_by_path.values(),
                    key=lambda h: h.get("score", 0.0),
                    reverse=True,
                )
                self._cache[eid] = sorted_hits[: self.top_k]

        logger.info(f"LocalAPIRetriever: prefetch complete, {len(self._cache)} cached")

        # Step 2.5: Merge reference URL tiles (if enabled) — chunk-level dedup
        if self.lookup_reference_url:
            ref_tiles = self._lookup_reference_tiles(examples)
            total_added, total_skipped = 0, 0
            for eid, ref_hits in ref_tiles.items():
                existing = self._cache.get(eid, [])
                existing_paths = {hit.get("path", "") for hit in existing}
                new_chunks = [rh for rh in ref_hits if rh["path"] not in existing_paths]
                skipped = len(ref_hits) - len(new_chunks)
                if new_chunks:
                    logger.info(
                        f"  [{eid[:8]}]: adding {len(new_chunks)} reference URL chunks "
                        f"({skipped} already in API results)"
                    )
                    self._cache[eid] = existing + new_chunks
                    total_added += len(new_chunks)
                total_skipped += skipped
            logger.info(
                f"lookup_reference_url: added {total_added} chunks, "
                f"skipped {total_skipped} duplicates"
            )

        # Step 3: Rerank (if reranker provided)
        if self.reranker is not None:
            # Build batch of (query, candidates) for all examples
            batch_inputs = []
            batch_eids = []
            for ex in examples:
                eid = ex.get("id", "unknown")
                hits = self._cache.get(eid, [])
                if not hits:
                    continue
                candidates = []
                for hit in hits:
                    path = hit.get("path", "")
                    score = hit.get("score", 0.0)
                    if path and os.path.exists(path):
                        candidates.append((path, score))
                if not candidates:
                    continue
                batch_inputs.append((ex["problem"], candidates))
                batch_eids.append(eid)

            if batch_inputs:
                all_reranked = self.reranker.rerank_batch(
                    batch_inputs,
                    top_k=self.rerank_top_k,
                )
                # Update cache with reranked results
                for eid, reranked_results in zip(batch_eids, all_reranked):
                    hits = self._cache[eid]
                    path_to_hit = {hit["path"]: hit for hit in hits if "path" in hit}
                    new_hits = []
                    for path, rerank_score in reranked_results:
                        orig_hit = path_to_hit.get(path, {})
                        new_hits.append(
                            {**orig_hit, "path": path, "score": rerank_score}
                        )
                    self._cache[eid] = new_hits
                logger.info(
                    f"LocalAPIRetriever: reranking complete ({len(batch_inputs)} examples)"
                )

    @staticmethod
    @staticmethod
    def _resolve_tile_path(hit: dict, tiles_dir: str | None = None) -> str | None:
        """Resolve tile path from hit, searching local shard dirs if needed."""
        path = hit.get("path", "")
        if path and os.path.exists(path):
            return path
        if not tiles_dir:
            return path if path else None
        article_id = hit.get("article_id")
        tile_index = hit.get("tile_index", 0)
        chunk_index = hit.get("chunk_index", 0)
        if article_id is None:
            return path if path else None
        tiles_dirname = f"{article_id}.png.tiles"
        chunk_name = f"chunk_{tile_index:04d}_{chunk_index:02d}.png"
        shard_size = 8284
        top_shard = article_id // shard_size
        top_shard_dir = os.path.join(tiles_dir, f"shard_{top_shard:03d}")
        if os.path.isdir(top_shard_dir):
            for sub in sorted(os.listdir(top_shard_dir)):
                sub_path = os.path.join(top_shard_dir, sub, tiles_dirname)
                if os.path.isdir(sub_path):
                    full = os.path.join(sub_path, chunk_name)
                    if os.path.exists(full):
                        return full
        flat = os.path.join(tiles_dir, tiles_dirname, chunk_name)
        if os.path.exists(flat):
            return flat
        return path if path else None

    @staticmethod
    def _hits_to_result(
        hits: list[dict], tiles_dir: str | None = None
    ) -> RetrievalResult:
        """Convert API hits to RetrievalResult."""
        if not hits:
            return RetrievalResult(retrieval_type="local_api")

        images = []
        image_urls = []
        urls = []
        seen_urls = set()
        for hit in hits:
            score = hit.get("score", 0.0)
            url = hit.get("url", "")
            path = LocalAPIRetriever._resolve_tile_path(hit, tiles_dir)
            if path and os.path.exists(path):
                images.append((path, score))
                image_urls.append(url or None)
            elif hit.get("image_base64"):
                images.append((hit["image_base64"], score))
                image_urls.append(url or None)
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        return RetrievalResult(
            images=images,
            image_urls=image_urls,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="local_api",
        )

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        eid = example.get("id", "unknown")

        # Return cached result if available (from prefetch)
        if eid in self._cache:
            return self._hits_to_result(self._cache[eid], tiles_dir=self.tiles_dir)

        # Fallback: single query (if prefetch wasn't called)
        import aiohttp

        query_dict = {"text": query}
        if self.query_image_fn:
            img_path = self.query_image_fn(example)
            if img_path and os.path.exists(img_path):
                import base64

                with open(img_path, "rb") as f:
                    query_dict["image"] = base64.b64encode(f.read()).decode()
        payload = {"queries": [query_dict], "n_docs": self.top_k}
        if self.nprobe is not None:
            payload["nprobe"] = self.nprobe
        if self.query_instruction is not None:
            payload["instruction"] = self.query_instruction
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status != 200:
                        return RetrievalResult(retrieval_type="local_api")
                    result = await response.json()
        except Exception as e:
            logger.error(f"Local API call failed: {e}")
            return RetrievalResult(retrieval_type="local_api")

        hits = result.get("results", [{}])[0].get("hits", [])
        self._cache[eid] = hits
        return self._hits_to_result(hits, tiles_dir=self.tiles_dir)

    async def get_hits(self, query: str, example: dict) -> list[dict]:
        """Return raw per-hit dicts (path/url/score/...) for this example.

        Used by wrappers that need per-hit granularity (e.g. HybridRetriever).
        Uses the same cache as retrieve().
        """
        await self.retrieve(query, example)
        return self._cache.get(example.get("id", "unknown"), [])


class TiledQwen3VLEmbeddingRetriever(BaseRetriever):
    """Retrieves context by searching through image tiles using Qwen3-VL-Embedding.

    Uses single vector embeddings (2048 dim) with cosine similarity for retrieval.

    When *pixel_query_map* is provided the retriever embeds the rendered query
    image (pixel query) instead of the raw text, so retrieval happens entirely
    in pixel space.
    """

    def __init__(
        self,
        screenshot_dir: str = "screenshots",
        tiles_dir: str = "tiles",
        tile_size: int | tuple[int, int] = 512,
        overlap: int = 0,
        cache_path: str | None = None,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        top_k: int = 3,
        examples: list[dict] | None = None,
        gpu_ids: list[int] | None = None,
        tensor_parallel_size: int = 1,
        pixel_query_map: dict[str, str] | None = None,
        multimodal_query_text_only: bool = False,
        multimodal_query_image_only: bool = False,
        local_wiki: bool = False,
        local_wiki_screenshot_dir: str | None = None,
        multi_image_query: bool = False,
        prebuilt_tiles_dir: str | None = None,
        embedding_backend: str = "vllm",  # "vllm", "hf", or "biqwen3"
        peft_adapter: str | None = None,
    ):
        self.top_k = top_k
        self.screenshot_dir = screenshot_dir
        self.tiles_dir = tiles_dir
        self.tile_size = tile_size
        self.overlap = overlap
        self.examples = examples or []
        self.pixel_query_map = pixel_query_map  # example_id -> pixel query image path
        self.multimodal_query_text_only = multimodal_query_text_only
        self.multimodal_query_image_only = multimodal_query_image_only
        self.local_wiki = local_wiki
        self.local_wiki_screenshot_dir = local_wiki_screenshot_dir
        self.multi_image_query = multi_image_query
        self.prebuilt_tiles_dir = prebuilt_tiles_dir
        self.embedding_backend = embedding_backend
        self.peft_adapter = peft_adapter
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(tiles_dir, exist_ok=True)

        # Build example_id -> URL mapping and deduplicate by URL
        from .simpleqa_data import extract_url_from_metadata

        self.id_to_url = {}
        seen_urls: dict[str, str] = {}  # url -> first example_id that uses it
        self.url_to_representative_id: dict[
            str, str
        ] = {}  # url -> representative example_id
        dedup_examples = []
        for ex in self.examples:
            ex_id = ex.get("id", "")
            url = extract_url_from_metadata(ex)
            if url:
                self.id_to_url[ex_id] = url
                if url not in seen_urls:
                    seen_urls[url] = ex_id
                    self.url_to_representative_id[url] = ex_id
                    dedup_examples.append(ex)

        logger.info(
            f"Deduplicated {len(self.examples)} examples -> {len(dedup_examples)} unique URLs "
            f"(removed {len(self.examples) - len(dedup_examples)} duplicate pages)"
        )
        self._dedup_examples = dedup_examples

        # Prepare tile paths: prebuilt dir (hard mini-datastore), local-wiki, or Selenium
        if self.prebuilt_tiles_dir:
            tile_paths = self._load_prebuilt_tiles()
        elif self.local_wiki:
            tile_paths = self._prepare_local_wiki_tiles()
        else:
            tile_paths = self._prepare_screenshots_and_tiles()

        # Import Qwen3-VL-Embedding retrieval system
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).parent.parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from qwen3vl_embedding_retrieval import Qwen3VLEmbeddingSystem
        except ImportError:
            try:
                from scripts.qwen3vl_embedding_retrieval import Qwen3VLEmbeddingSystem
            except ImportError:
                raise ImportError("Qwen3VLEmbeddingSystem not available.")

        logger.info("Initializing Qwen3-VL-Embedding retrieval system...")
        logger.info(f"Model: {model_name}, tiles: {len(tile_paths)}, GPUs: {gpu_ids}")
        if self.pixel_query_map:
            logger.info(
                f"Pixel query mode ENABLED ({len(self.pixel_query_map)} queries)"
            )

        self.retrieval_system = Qwen3VLEmbeddingSystem(
            model_name=model_name,
            cache_path=cache_path,
            gpu_ids=gpu_ids,
            tensor_parallel_size=tensor_parallel_size,
            backend=self.embedding_backend,
            peft_adapter=self.peft_adapter,
        )

        # Embed all tiles (batch_size=8 for HF backend to avoid OOM on shared GPUs)
        embed_bs = 8 if self.embedding_backend == "hf" else 32
        self.retrieval_system.embed_images(
            file_paths=tile_paths,
            cache_path=cache_path,
            batch_size=embed_bs,
        )
        logger.info(
            f"Qwen3-VL-Embedding retrieval ready with {len(self.retrieval_system.image_paths)} tiles"
        )

    def _load_prebuilt_tiles(self) -> list[str]:
        """Load ALL .png tiles from a prebuilt tile directory (e.g. hard mini-datastore).

        Unlike _prepare_local_wiki_tiles which only loads golden tiles matching
        example IDs, this loads every tile in the directory — including distractors.
        """
        import glob as _glob

        all_tiles = sorted(_glob.glob(os.path.join(self.prebuilt_tiles_dir, "*.png")))
        filtered = _filter_tiles_by_aspect_ratio(all_tiles)
        logger.info(
            f"prebuilt-tiles: loaded {len(filtered)} tiles from {self.prebuilt_tiles_dir} "
            f"(filtered {len(all_tiles) - len(filtered)} extreme aspect ratio tiles)"
        )
        return filtered

    def _prepare_local_wiki_tiles(self) -> list[str]:
        """Prepare tiles from local kiwix tile store for all examples in the batch.

        Does a single batch URL lookup (fast), then copies+cuts tiles per example.
        Reports an error (no fallback) if a URL is not found in kiwix.

        Returns the list of all cut tile paths ready for embedding.
        """
        import glob as _glob
        import shutil
        import sys as _sys
        from PIL import Image
        from .simpleqa_data import extract_url_from_metadata
        from tqdm import tqdm

        cut_height = (
            self.tile_size[1] if isinstance(self.tile_size, tuple) else self.tile_size
        )
        wiki_cache = self.local_wiki_screenshot_dir or os.path.join(
            self.screenshot_dir, "local-wiki"
        )
        os.makedirs(wiki_cache, exist_ok=True)
        os.makedirs(self.tiles_dir, exist_ok=True)

        # Separate already-cached examples from ones that need processing
        need: list[tuple[str, str]] = []  # (ex_id, url)
        for ex in self._dedup_examples:
            ex_id = ex["id"]
            if not _glob.glob(os.path.join(self.tiles_dir, f"{ex_id}_tile_*.png")):
                url = extract_url_from_metadata(ex) or ""
                need.append((ex_id, url))

        logger.info(
            f"local-wiki: {len(self._dedup_examples) - len(need)} cached, {len(need)} need processing"
        )

        if need:
            # Single batch lookup for all URLs at once (loads articles.json once)
            if not os.path.isdir(_KIWIX_OUTPUT_DIR) or not os.path.isfile(
                _KIWIX_ARTICLES_JSON
            ):
                logger.error(
                    f"local-wiki: kiwix tiles unavailable at {_KIWIX_OUTPUT_DIR}"
                )
            else:
                if _WIKI_SCREENSHOT_DIR not in _sys.path:
                    _sys.path.insert(0, _WIKI_SCREENSHOT_DIR)
                from scripts.build_index import batch_query_by_url as _batch_query

                redirects = (
                    _KIWIX_REDIRECTS_JSON
                    if os.path.isfile(_KIWIX_REDIRECTS_JSON)
                    else None
                )
                urls_to_lookup = [u for _, u in need if u and "wikipedia.org" in u]
                results = _batch_query(
                    _KIWIX_OUTPUT_DIR,
                    urls_to_lookup,
                    _KIWIX_ARTICLES_JSON,
                    redirects_json=redirects,
                )
                found = sum(1 for r in results.values() if r is not None)
                logger.info(
                    f"local-wiki: batch lookup found {found}/{len(urls_to_lookup)} URLs"
                )

                # Copy + cut per example
                ok, failed = 0, 0
                for ex_id, url in tqdm(need, desc="local-wiki: copying+cutting tiles"):
                    # Check cache again (may have been done by a parallel run)
                    if _glob.glob(os.path.join(self.tiles_dir, f"{ex_id}_tile_*.png")):
                        ok += 1
                        continue
                    result = results.get(url)
                    if result is None:
                        logger.error(
                            f"local-wiki [{ex_id}]: URL not found in kiwix: {url}"
                        )
                        failed += 1
                        continue
                    src_dir = os.path.join(_KIWIX_OUTPUT_DIR, result["tiles_dir"])
                    article_cache = os.path.join(wiki_cache, str(ex_id))
                    if not os.path.exists(article_cache):
                        if not os.path.isdir(src_dir):
                            logger.error(
                                f"local-wiki [{ex_id}]: tiles dir not on disk: {src_dir}"
                            )
                            failed += 1
                            continue
                        shutil.copytree(src_dir, article_cache)
                    # Cut into strips
                    raw_tiles = sorted(
                        f
                        for f in os.listdir(article_cache)
                        if f.endswith(".png") and f.startswith("tile_")
                    )
                    if not raw_tiles:
                        logger.error(
                            f"local-wiki [{ex_id}]: no tile PNGs in {article_cache}"
                        )
                        failed += 1
                        continue
                    global_row = 0
                    for raw_name in raw_tiles:
                        raw_path = os.path.join(article_cache, raw_name)
                        if os.path.getsize(raw_path) == 0:
                            continue
                        try:
                            img = Image.open(raw_path)
                            img.load()
                        except Exception as e:
                            logger.warning(
                                f"local-wiki [{ex_id}]: corrupt tile {raw_path}: {e}"
                            )
                            continue
                        w, h = img.size
                        y = 0
                        while y < h:
                            y2 = min(y + cut_height, h)
                            img.crop((0, y, w, y2)).save(
                                os.path.join(
                                    self.tiles_dir, f"{ex_id}_tile_{global_row}_0.png"
                                )
                            )
                            global_row += 1
                            y += cut_height
                        img.close()
                    ok += 1
                logger.info(
                    f"local-wiki: {ok} articles prepared, {failed} not found/failed"
                )

        all_tile_paths = []
        for ex in self._dedup_examples:
            ex_id = ex["id"]
            tiles = sorted(
                _glob.glob(os.path.join(self.tiles_dir, f"{ex_id}_tile_*.png"))
            )
            all_tile_paths.extend(tiles)

        filtered = _filter_tiles_by_aspect_ratio(all_tile_paths)
        logger.info(
            f"local-wiki: {len(filtered)} tiles ready for embedding "
            f"(filtered {len(all_tile_paths) - len(filtered)} extreme aspect ratio tiles)"
        )
        return filtered

    def _prepare_screenshots_and_tiles(self) -> list[str]:
        """Prepare screenshots and tiles for dataset, return tile paths.

        Uses deduplicated examples (one per unique URL) to avoid
        duplicate tiles inflating the retrieval index.
        """
        from .simpleqa_data import capture_screenshot_for_example, split_image_to_tiles
        from tqdm import tqdm

        examples_to_process = self._dedup_examples
        screenshot_paths = []
        missing = []

        # Collect screenshot paths and identify missing (deduplicated)
        for ex in examples_to_process:
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
            f"Splitting {len(screenshot_paths)} unique screenshots into tiles (output: {self.tiles_dir})..."
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
            f"Prepared {len(filtered_tile_paths)} tiles from {len(screenshot_paths)} unique screenshots "
            f"(filtered {len(all_tile_paths) - len(filtered_tile_paths)} extreme aspect ratio tiles)"
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

    # Class-level cache for iNat 2021 image_id -> file_name mapping
    _inat2021_id_map: dict[int, str] | None = None
    INAT2021_DATA_DIR = _INAT2021_DATA_DIR

    @classmethod
    def _load_inat2021_mapping(cls) -> dict[int, str]:
        """Load iNaturalist 2021 competition image_id -> file_name mapping.

        Downloads val.json from the competition S3 bucket if not cached locally.
        """
        if cls._inat2021_id_map is not None:
            return cls._inat2021_id_map

        import json
        import tarfile
        import urllib.request
        from pathlib import Path

        data_dir = Path(cls.INAT2021_DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        val_json = data_dir / "val.json"

        if not val_json.exists():
            tar_path = data_dir / "val.json.tar.gz"
            if not tar_path.exists():
                logger.info("Downloading iNaturalist 2021 val annotations...")
                urllib.request.urlretrieve(
                    "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
                    str(tar_path),
                )
            with tarfile.open(str(tar_path), "r:gz") as tf:
                tf.extractall(path=str(data_dir))
            logger.info(f"Extracted iNat 2021 val.json to {val_json}")

        with open(val_json) as f:
            data = json.load(f)

        cls._inat2021_id_map = {img["id"]: img["file_name"] for img in data["images"]}
        logger.info(f"Loaded iNat 2021 mapping: {len(cls._inat2021_id_map)} images")
        return cls._inat2021_id_map

    def _get_inat_image_path(self, example: dict) -> str | None:
        """Get EVQA query image (iNaturalist or Landmarks). Delegates to _get_query_image_path_for_example."""
        return _get_query_image_path_for_example(example, self.tiles_dir)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        # Dispatch to multi-image retrieval if enabled
        if self.multi_image_query:
            return await self.retrieve_multi_image(query, example)
        return await self._retrieve_single(query, example)

    async def _retrieve_single(self, query: str, example: dict) -> RetrievalResult:
        example_id = example.get("id", "")
        loop = asyncio.get_event_loop()

        # Priority: pixel_query_map > iNaturalist image > text-only
        pixel_query_path = None
        if self.pixel_query_map and example_id in self.pixel_query_map:
            pixel_query_path = self.pixel_query_map[example_id]

        # Check for iNaturalist query image (multimodal text+image query)
        inat_image_path = self._get_inat_image_path(example)

        try:
            # Determine query modality
            query_image = None
            if pixel_query_path and os.path.exists(pixel_query_path):
                # Pixel query: image-only (rendered text as image)
                query_image = pixel_query_path
                query_text = None
                retrieval_type = "tiled_qwen3vl_embedding_pixel_query"
            elif self.multimodal_query_text_only:
                # Ablation: text-only (no image)
                query_image = None
                query_text = query
                retrieval_type = "tiled_qwen3vl_embedding_multimodal_textonly"
            elif self.multimodal_query_image_only and inat_image_path:
                # Ablation: image-only (no text)
                query_image = inat_image_path
                query_text = None
                retrieval_type = "tiled_qwen3vl_embedding_multimodal_imageonly"
            elif inat_image_path:
                # Multimodal: text + image
                query_image = inat_image_path
                query_text = query
                retrieval_type = "tiled_qwen3vl_embedding_multimodal"
            else:
                # Text-only (no query image available)
                query_text = query
                retrieval_type = "tiled_qwen3vl_embedding"

            results = await loop.run_in_executor(
                None,
                lambda: self.retrieval_system.search(
                    text=query_text, image=query_image, top_k=self.top_k
                ),
            )

            if results:
                source_url = self._extract_urls_from_results(results)
                return RetrievalResult(
                    images=results,
                    source_url=source_url,
                    retrieval_type=retrieval_type,
                    pixel_query_path=pixel_query_path or inat_image_path,
                    query_image_path=inat_image_path,
                )
            else:
                return RetrievalResult(
                    text="No relevant tiles found via Qwen3-VL-Embedding search",
                    retrieval_type=retrieval_type,
                    pixel_query_path=pixel_query_path or inat_image_path,
                    query_image_path=inat_image_path,
                )
        except Exception as e:
            logger.error(f"Qwen3-VL-Embedding search failed: {e}")
            return RetrievalResult(
                text=f"Qwen3-VL-Embedding retrieval error: {e}",
                retrieval_type="tiled_qwen3vl_embedding",
                pixel_query_path=pixel_query_path or inat_image_path,
                query_image_path=inat_image_path,
            )

    async def retrieve_multi_image(self, query: str, example: dict) -> RetrievalResult:
        """Multi-image retrieval: search with ALL query images, aggregate scores, return top-K.

        For each query image, does a multimodal search (text + image), then combines
        scores across all images using max-score aggregation per tile.
        Falls back to single-image retrieve() if only 0-1 images available.
        """
        all_image_paths = _get_all_query_image_paths(example, self.tiles_dir)
        # Get single image for generation (first available, used in RetrievalResult)
        single_image_path = self._get_inat_image_path(example)

        if len(all_image_paths) <= 1:
            return await self._retrieve_single(query, example)

        example_id = example.get("id", "")
        loop = asyncio.get_event_loop()
        logger.info(
            f"Multi-image retrieval for {example_id}: {len(all_image_paths)} query images"
        )

        try:
            # Score aggregation: for each tile, keep the max score across all query images
            tile_best_score: dict[str, float] = {}

            for img_path in all_image_paths:
                results = await loop.run_in_executor(
                    None,
                    lambda p=img_path: self.retrieval_system.search(
                        text=query, image=p, top_k=self.top_k * 2
                    ),
                )
                for tile_path, score in results:
                    if (
                        tile_path not in tile_best_score
                        or score > tile_best_score[tile_path]
                    ):
                        tile_best_score[tile_path] = score

            # Sort by score descending, take top_k
            sorted_tiles = sorted(
                tile_best_score.items(), key=lambda x: x[1], reverse=True
            )
            top_results = sorted_tiles[: self.top_k]

            retrieval_type = (
                f"tiled_qwen3vl_embedding_multiimage_{len(all_image_paths)}imgs"
            )

            if top_results:
                source_url = self._extract_urls_from_results(top_results)
                return RetrievalResult(
                    images=top_results,
                    source_url=source_url,
                    retrieval_type=retrieval_type,
                    pixel_query_path=single_image_path,
                    query_image_path=single_image_path,
                )
            else:
                return RetrievalResult(
                    text="No relevant tiles found via multi-image search",
                    retrieval_type=retrieval_type,
                    pixel_query_path=single_image_path,
                    query_image_path=single_image_path,
                )
        except Exception as e:
            logger.error(f"Multi-image retrieval failed: {e}")
            return await self._retrieve_single(query, example)


class TextAPIRetriever(BaseRetriever):
    """Retrieve text chunks from a text search API (wiki-screenshot text_search_api.py).

    The API accepts:
        POST /search
        {"queries": [{"text": "..."}], "n_docs": N}
    and returns:
        {"results": [{"hits": [{"text": ..., "title": ..., "url": ..., "score": ...}, ...]}]}

    Supports batch prefetch for efficient evaluation.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:30889/search",
        top_k: int = 3,
        batch_size: int = 32,
        nprobe: int | None = None,
        query_instruction: str | None = None,
        reader_top_k: int | None = None,
        query_image_fn=None,
    ):
        self.api_url = api_url
        self.top_k = top_k
        # If reader_top_k is set and < top_k, only the first reader_top_k hits are
        # passed to the reader. Mirrors the image-side reader_top_k slicing in
        # run_naive_simpleqa.py so text + image cells are comparable at fixed k.
        self.reader_top_k = reader_top_k
        self.batch_size = batch_size
        self.nprobe = nprobe
        self.query_instruction = query_instruction
        self.query_image_fn = query_image_fn
        self._cache: dict[str, list[dict]] = {}

    async def prefetch(self, examples: list[dict]):
        """Batch-fetch retrieval results for all examples."""
        import aiohttp

        queries = []
        example_ids = []
        for ex in examples:
            eid = ex.get("id", "unknown")
            if eid in self._cache:
                continue
            query_dict = {"text": ex["problem"]}
            if self.query_image_fn:
                img_path = self.query_image_fn(ex)
                if img_path and os.path.exists(img_path):
                    import base64

                    with open(img_path, "rb") as f:
                        query_dict["image"] = base64.b64encode(f.read()).decode()
            queries.append(query_dict)
            example_ids.append(eid)

        if not queries:
            logger.info("TextAPIRetriever: all examples already cached")
            return

        has_images = any("image" in q for q in queries)
        batch_size = min(self.batch_size, 16) if has_images else self.batch_size
        logger.info(
            f"TextAPIRetriever: prefetching {len(queries)} queries in batches of {batch_size}"
            f"{' (multimodal)' if has_images else ''}"
        )

        for batch_start in range(0, len(queries), batch_size):
            batch_queries = queries[batch_start : batch_start + batch_size]
            batch_ids = example_ids[batch_start : batch_start + batch_size]

            payload = {"queries": batch_queries, "n_docs": self.top_k}
            if self.nprobe is not None:
                payload["nprobe"] = self.nprobe
            if self.query_instruction is not None:
                payload["instruction"] = self.query_instruction
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=600),
                    ) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"TextAPI batch error {response.status}: {error_text[:200]}"
                            )
                            for eid in batch_ids:
                                self._cache[eid] = []
                            continue
                        result = await response.json()
            except Exception as e:
                logger.error(f"TextAPI batch call failed: {e}")
                for eid in batch_ids:
                    self._cache[eid] = []
                continue

            results_list = result.get("results", [])
            for i, eid in enumerate(batch_ids):
                if i < len(results_list):
                    self._cache[eid] = results_list[i].get("hits", [])
                else:
                    self._cache[eid] = []

            logger.info(
                f"  Batch {batch_start // self.batch_size + 1}/"
                f"{(len(queries) + self.batch_size - 1) // self.batch_size}: "
                f"{len(batch_queries)} queries done"
            )

        logger.info(f"TextAPIRetriever: prefetch complete, {len(self._cache)} cached")

    @staticmethod
    def _hits_to_result(
        hits: list[dict], max_passages: int | None = None
    ) -> RetrievalResult:
        """Convert text API hits to RetrievalResult.

        If max_passages is set, only the first max_passages hits are joined into
        the reader prompt. The cache itself is not truncated, so the same cached
        hits can serve multiple reader_top_k values.
        """
        if not hits:
            return RetrievalResult(retrieval_type="text_api")

        if max_passages is not None and max_passages < len(hits):
            hits = hits[:max_passages]

        passages = []
        urls = []
        seen_urls = set()
        for hit in hits:
            text = hit.get("text", "")
            url = hit.get("url", "")
            # Option 1 (2026-04-29): no `[title]` prefix on chunks. Title is leaked
            # metadata for entity-answering tasks (often contains the answer outright).
            # Reader sees only the chunk content. URL lives in retrieval_result.source_url
            # for logging/grading but is not injected into the prompt by build_messages.
            if text:
                passages.append(text)
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        combined_text = "\n\n".join(passages) if passages else None
        return RetrievalResult(
            text=combined_text,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="text_api",
        )

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        eid = example.get("id", "unknown")

        if eid in self._cache:
            return self._hits_to_result(
                self._cache[eid], max_passages=self.reader_top_k
            )

        # Fallback: single query
        import aiohttp

        payload = {"queries": [{"text": query}], "n_docs": self.top_k}
        if self.nprobe is not None:
            payload["nprobe"] = self.nprobe
        if self.query_instruction is not None:
            payload["instruction"] = self.query_instruction
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status != 200:
                        return RetrievalResult(retrieval_type="text_api")
                    result = await response.json()
        except Exception as e:
            logger.error(f"TextAPI call failed: {e}")
            return RetrievalResult(retrieval_type="text_api")

        hits = result.get("results", [{}])[0].get("hits", [])
        self._cache[eid] = hits
        return self._hits_to_result(hits, max_passages=self.reader_top_k)

    async def get_hits(self, query: str, example: dict) -> list[dict]:
        """Return raw per-hit dicts (title/text/url/score/...) for this example.

        Used by wrappers that need per-chunk granularity (e.g. RenderedTextWrapper).
        Uses the same cache as retrieve().
        """
        await self.retrieve(query, example)
        return self._cache.get(example.get("id", "unknown"), [])


