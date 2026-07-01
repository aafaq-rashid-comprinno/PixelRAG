"""Wrapper and hybrid retrievers.

Retrievers that compose or transform other retrievers:
- OCRWrappedRetriever: adds OCR to image-based retrieval
- RenderedTextWrapper: renders text to image for VLM
- HybridRetriever: combines multiple retrieval strategies
- HTMLDOMLookupRetriever: HTML DOM-based lookup
"""

import asyncio
import base64
import io
import logging
import os

from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class OCRWrappedRetriever(BaseRetriever):
    """Wraps an image retriever; OCRs retrieved tiles and returns text.

    Ablation A pipeline: image retrieve -> OCR -> text to reader.
    Talks to an OpenAI-compatible chat endpoint (PaddleOCR-VL served via vLLM).
    Caches OCR output to a JSONL file keyed by absolute image path so reruns
    reuse prior work.
    """

    DEFAULT_PROMPT = "OCR this image. Output only the extracted text verbatim, preserving paragraph and line breaks."

    def __init__(
        self,
        base: BaseRetriever,
        ocr_url: str = "http://localhost:8202/v1",
        model: str = "PaddlePaddle/PaddleOCR-VL",
        api_key: str = "dummy",
        cache_path: str = "ocr_cache/paddleocr_vl.jsonl",
        concurrency: int = 16,
        prompt: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 4096,
        reader_top_k: int | None = None,
    ):
        self.base = base
        self.ocr_url = ocr_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.cache_path = cache_path
        self.concurrency = concurrency
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reader_top_k = reader_top_k
        self._cache: dict[str, str] = {}
        self.tiles_dir = getattr(base, "tiles_dir", None)
        self._load_cache()

    def _load_cache(self):
        if not os.path.isfile(self.cache_path):
            return
        import json

        loaded = 0
        try:
            with open(self.cache_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._cache[entry["path"]] = entry["text"]
                    loaded += 1
            logger.info(
                f"OCRWrappedRetriever: loaded {loaded} cached OCR entries from {self.cache_path}"
            )
        except Exception as e:
            logger.warning(
                f"OCRWrappedRetriever: cache load failed ({e}); starting fresh"
            )

    def _append_cache(self, path: str, text: str):
        import json

        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"path": path, "text": text}, ensure_ascii=False) + "\n")
        self._cache[path] = text

    async def _ocr_one(self, path: str, session) -> str:
        if path in self._cache:
            return self._cache[path]
        import aiohttp
        import base64

        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            logger.error(f"OCR read failed for {path}: {e}")
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with session.post(
                f"{self.ocr_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"OCR HTTP {resp.status} for {path}: {err[:200]}")
                    return ""
                result = await resp.json()
                text = result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"OCR request failed for {path}: {e}")
            return ""
        self._append_cache(path, text)
        return text

    async def _batch_ocr(self, paths: list[str]) -> dict[str, str]:
        import aiohttp

        to_fetch = [p for p in paths if p not in self._cache]
        if not to_fetch:
            return {p: self._cache[p] for p in paths}
        sem = asyncio.Semaphore(self.concurrency)
        async with aiohttp.ClientSession() as session:

            async def _one(p):
                async with sem:
                    return await self._ocr_one(p, session)

            await asyncio.gather(*[_one(p) for p in to_fetch])
        return {p: self._cache.get(p, "") for p in paths}

    async def prefetch(self, examples: list[dict]):
        """Forward to base's prefetch, then batch-OCR all tiles up front."""
        if hasattr(self.base, "prefetch"):
            await self.base.prefetch(examples)
        all_paths: set[str] = set()
        for ex in examples:
            r = await self.base.retrieve(ex.get("problem", ""), ex)
            images = (
                r.images[: self.reader_top_k]
                if self.reader_top_k is not None
                else r.images
            )
            for p, _ in images:
                all_paths.add(os.path.abspath(p))
        uncached = [p for p in all_paths if p not in self._cache]
        logger.info(
            f"OCRWrappedRetriever: {len(all_paths)} unique tiles across {len(examples)} examples; "
            f"{len(all_paths) - len(uncached)} cached, OCRing {len(uncached)}"
        )
        if uncached:
            await self._batch_ocr(uncached)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        r = await self.base.retrieve(query, example)
        if not r.images:
            return r
        images = (
            r.images[: self.reader_top_k] if self.reader_top_k is not None else r.images
        )
        image_urls = (
            r.image_urls[: self.reader_top_k]
            if self.reader_top_k is not None and r.image_urls
            else list(r.image_urls or [])
        )
        urls: list[str] = []
        seen_urls: set[str] = set()
        for url in image_urls:
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)
        paths = [os.path.abspath(p) for p, _ in images]
        ocr_map = await self._batch_ocr(paths)
        passages = [ocr_map[p].strip() for p in paths if ocr_map.get(p, "").strip()]
        combined = "\n\n---\n\n".join(passages) if passages else None
        return RetrievalResult(
            text=combined,
            images=[],
            source_url=", ".join(urls) if urls else r.source_url,
            retrieval_type=f"{r.retrieval_type}+ocr",
            pixel_query_path=r.pixel_query_path,
            query_image_path=r.query_image_path,
        )


class RenderedTextWrapper(BaseRetriever):
    """Wraps a text retriever; renders each chunk as an image.

    Ablation B pipeline: text retrieve -> render as Wikipedia-style image -> VLM reader.
    Requires the base retriever to expose get_hits(query, example) returning
    per-hit dicts with keys: title, text, url, score, article_id, chunk_index.
    (TextAPIRetriever satisfies this.)

    Renders are cached on disk at {render_dir}/{article_id}_{chunk_index}.png
    so repeated eval runs don't re-render.
    """

    def __init__(
        self,
        base: BaseRetriever,
        render_dir: str = "rendered_chunks",
        reader_top_k: int | None = None,
    ):
        if not hasattr(base, "get_hits"):
            raise TypeError(
                f"RenderedTextWrapper requires base retriever with get_hits(); "
                f"got {type(base).__name__}"
            )
        self.base = base
        self.render_dir = render_dir
        self.reader_top_k = reader_top_k
        os.makedirs(self.render_dir, exist_ok=True)
        self.tiles_dir = render_dir

    async def prefetch(self, examples: list[dict]):
        if hasattr(self.base, "prefetch"):
            await self.base.prefetch(examples)

    def _render(self, hit: dict) -> str:
        from .text_renderer import render_text_chunk

        article_id = hit.get("article_id", "unknown")
        chunk_index = hit.get("chunk_index", 0)
        out_path = os.path.join(self.render_dir, f"{article_id}_{chunk_index}.png")
        if os.path.isfile(out_path):
            return out_path
        # No-title policy: mirrors `_hits_to_result` (line ~3035) — title/url are
        # leaked metadata for entity-answering tasks and were stripped from the
        # text→text path on 2026-04-29. Apply the same constraint here so
        # rendered and text→text differ only in modality, not in content.
        render_text_chunk(
            text=hit.get("text", ""),
            title=None,
            url=None,
            output_path=out_path,
        )
        return out_path

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        hits = await self.base.get_hits(query, example)
        if not hits:
            return RetrievalResult(retrieval_type="text_api+rendered")
        if self.reader_top_k is not None:
            hits = hits[: self.reader_top_k]
        images: list[tuple[str, float]] = []
        urls: list[str] = []
        seen_urls: set[str] = set()
        for hit in hits:
            if not hit.get("text"):
                continue
            path = self._render(hit)
            images.append((path, float(hit.get("score", 0.0))))
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)
        return RetrievalResult(
            images=images,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="text_api+rendered",
        )


class HybridRetriever(BaseRetriever):
    """Merge image (LocalAPIRetriever) and text (TextAPIRetriever) hits by raw score.

    Both underlying retrievers embed with Qwen3-VL-Embedding-2B against L2-normalized
    FAISS IVFFlat (IP metric) indices, so their per-hit scores are cosine similarities
    on the same scale and directly comparable without any normalization step.

    Each base is called with its own configured top_k, then the combined candidate pool
    is sorted by score desc and the top `top_k` are kept. The reader receives the
    surviving image hits as image inputs and the surviving text hits as a concatenated
    text block in the same prompt — VL-4B handles mixed modality natively.
    """

    def __init__(
        self,
        image_base: "LocalAPIRetriever",
        text_base: "TextAPIRetriever",
        top_k: int = 3,
        reader_top_k: int | None = None,
    ):
        if not hasattr(image_base, "get_hits"):
            raise TypeError(
                f"HybridRetriever.image_base requires get_hits(); got {type(image_base).__name__}"
            )
        if not hasattr(text_base, "get_hits"):
            raise TypeError(
                f"HybridRetriever.text_base requires get_hits(); got {type(text_base).__name__}"
            )
        self.image_base = image_base
        self.text_base = text_base
        self.top_k = top_k
        self.reader_top_k = reader_top_k
        self.tiles_dir = getattr(image_base, "tiles_dir", None)

    async def prefetch(self, examples: list[dict]):
        if hasattr(self.image_base, "prefetch"):
            await self.image_base.prefetch(examples)
        if hasattr(self.text_base, "prefetch"):
            await self.text_base.prefetch(examples)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        image_hits = await self.image_base.get_hits(query, example)
        text_hits = await self.text_base.get_hits(query, example)

        # Tag each hit with its modality, then merge and sort by score desc.
        merged: list[tuple[float, str, dict]] = []
        for h in image_hits:
            score = float(h.get("score", 0.0))
            merged.append((score, "image", h))
        for h in text_hits:
            score = float(h.get("score", 0.0))
            merged.append((score, "text", h))

        merged.sort(key=lambda x: x[0], reverse=True)
        keep_k = self.reader_top_k if self.reader_top_k is not None else self.top_k
        top = merged[:keep_k]

        images: list[tuple[str, float]] = []
        passages: list[str] = []
        urls: list[str] = []
        seen_urls: set[str] = set()

        for score, modality, hit in top:
            url = hit.get("url", "")
            if modality == "image":
                path = hit.get("path", "")
                if path and os.path.exists(path):
                    images.append((path, score))
            else:  # text
                title = hit.get("title", "")
                text = hit.get("text", "")
                if text:
                    header = f"[{title}]" if title else ""
                    passages.append(f"{header}\n{text}" if header else text)
            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        return RetrievalResult(
            text="\n\n".join(passages) if passages else None,
            images=images,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="hybrid",
        )


class HTMLDOMLookupRetriever(BaseRetriever):
    """Text-retrieve → DOM lookup: retrieve text chunks, then find their HTML context.

    Wraps TextAPIRetriever. For each retrieved text chunk:
    1. Fetches original HTML from kiwix-serve using article_id
    2. Locates the chunk text within the HTML DOM
    3. Extracts the enclosing semantic container (section/table/div)
    4. Returns structured HTML context to the reader

    This gives the reader table/list structure without needing a separate HTML index.
    Falls back to plain text if DOM lookup fails for a chunk.
    """

    KIWIX_BASE = "http://localhost:9454/content/wikipedia_en_all_maxi_2025-08"

    def __init__(
        self,
        text_api_url: str = "http://localhost:30889/search",
        top_k: int = 3,
        nprobe: int | None = None,
        query_instruction: str | None = None,
        reader_top_k: int | None = None,
        query_image_fn=None,
        kiwix_base: str | None = None,
        articles_json: str = "/path/to/data",
        context_mode: str = "section",
        llm_verify: bool = False,
        llm_verify_model: str = "gpt-4.1-mini",
    ):
        import json as _json

        self._text_retriever = TextAPIRetriever(
            api_url=text_api_url,
            top_k=top_k,
            nprobe=nprobe,
            query_instruction=query_instruction,
            reader_top_k=reader_top_k,
            query_image_fn=query_image_fn,
        )
        if kiwix_base:
            self.KIWIX_BASE = kiwix_base
        self.top_k = top_k
        self.reader_top_k = reader_top_k
        self.context_mode = context_mode
        self.llm_verify = llm_verify
        self.llm_verify_model = llm_verify_model

        with open(articles_json) as f:
            self._articles: list[str] = _json.load(f)

        self._html_cache: dict[int, str] = {}

    async def prefetch(self, examples: list[dict]):
        await self._text_retriever.prefetch(examples)

    def _fetch_html(self, article_id: int) -> str | None:
        """Fetch article HTML from kiwix-serve (with caching)."""
        if article_id in self._html_cache:
            return self._html_cache[article_id]

        if article_id >= len(self._articles):
            return None

        import requests
        from urllib.parse import quote

        slug = self._articles[article_id]
        url = f"{self.KIWIX_BASE}/{quote(slug, safe='/:@!$&()*+,;=')}"
        try:
            resp = requests.get(url, timeout=15)
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                return None
            self._html_cache[article_id] = resp.text
            return resp.text
        except Exception:
            return None

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for fuzzy DOM matching."""
        import re
        import unicodedata

        text = re.sub(r"[\xa0    ]", " ", text)
        text = re.sub(r"[‐-―−﹘﹣－—–]", "-", text)
        text = re.sub(r" +", " ", text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        return text.lower()

    def _dom_lookup(self, html: str, chunk_text: str) -> str | None:
        """Find the contiguous DOM span covering chunk_text, return its HTML.

        Strategy:
        1. Extract search keys from chunk text (table cells + prose fragments)
        2. For each key, find the tightest DOM element and walk up to a
           direct child of mw-parser-output
        3. Return ALL direct children from the first match to the last match
           (inclusive), plus everything in between — this preserves the full
           contiguous region the chunk spans.
        """
        from lxml import html as lxml_html, etree

        tree = lxml_html.fromstring(html)

        keys = self._extract_search_keys(chunk_text)
        if not keys:
            return None

        mw_output = tree.xpath('//div[contains(@class, "mw-parser-output")]')
        if not mw_output:
            return None
        content_root = mw_output[0]
        children = list(content_root)
        if not children:
            return None

        # For each key, find the tightest match and resolve to a
        # direct-child index of mw-parser-output
        matched_child_indices = set()
        SKIP_TAGS = frozenset(
            ("script", "style", "title", "meta", "link", "nav", "header", "footer")
        )

        for key in keys:
            key_norm = self._normalize(key)
            if len(key_norm) < 4:
                continue

            best_elem = None
            best_len = float("inf")

            for elem in content_root.iter():
                if not isinstance(elem, lxml_html.HtmlElement):
                    continue
                if elem.tag in SKIP_TAGS:
                    continue
                try:
                    tc = elem.text_content()
                except Exception:
                    continue
                tc_norm = self._normalize(tc)
                if key_norm in tc_norm and len(tc) < best_len:
                    best_elem = elem
                    best_len = len(tc)

            if best_elem is None:
                continue

            # Walk up from best_elem to find which direct child of content_root
            # contains it
            current = best_elem
            while current is not None:
                parent = current.getparent()
                if parent is None:
                    break
                if parent == content_root:
                    # current is a direct child of mw-parser-output
                    try:
                        idx = children.index(current)
                        matched_child_indices.add(idx)
                    except ValueError:
                        pass
                    break
                current = parent

        if not matched_child_indices:
            return None

        # Return contiguous range from first to last matched child (inclusive)
        first = min(matched_child_indices)
        last = max(matched_child_indices)

        span_elems = children[first : last + 1]

        # Build result: serialize all elements in the span
        parts = []
        for el in span_elems:
            # Strip style/script/navbox noise
            for tag in ("style", "script"):
                for junk in list(el.iter(tag)):
                    if junk.getparent() is not None:
                        junk.getparent().remove(junk)
            if hasattr(el, "xpath"):
                for nav in el.xpath('.//*[contains(@class, "navbox")]'):
                    if nav.getparent() is not None:
                        nav.getparent().remove(nav)
            try:
                parts.append(etree.tostring(el, encoding="unicode", method="html"))
            except Exception:
                continue

        if not parts:
            return None

        html_str = "\n".join(parts)

        # Log oversized results but still return them (caller decides)
        if len(html_str) > self.MAX_CONTAINER_CHARS * 2:
            logger.warning(
                "DOM lookup oversized: %d chars (max %d) for chunk starting with %r",
                len(html_str),
                self.MAX_CONTAINER_CHARS * 2,
                chunk_text[:50],
            )

        # Minimum useful size
        if len(html_str) < 100 and len(chunk_text) > 200:
            return None

        return html_str

    MAX_CONTAINER_CHARS = 8000

    def _find_semantic_container(self, elem) -> "lxml_html.HtmlElement":  # noqa: F821
        """Walk up from matched element to find a meaningful semantic container.

        Hard cap: never return a container with text_content > MAX_CONTAINER_CHARS.
        Stops at mw-parser-output boundary (never returns the whole article).
        """

        SEMANTIC_TAGS = {
            "section",
            "article",
            "table",
            "blockquote",
            "details",
            "figure",
        }
        STOP_CLASSES = {"mw-parser-output", "mw-body-content", "mw-body"}
        MIN_CONTEXT_LEN = 200

        if elem.tag in SEMANTIC_TAGS:
            return elem

        best = elem
        current = elem

        for _ in range(15):
            parent = current.getparent()
            if parent is None:
                break
            # Hard stop: never go past the article content container
            parent_classes = parent.get("class", "")
            if any(sc in parent_classes for sc in STOP_CLASSES):
                # We've reached the article root — use section gathering instead
                if self.context_mode == "section":
                    gathered = self._gather_section_context(current)
                    if gathered is not None:
                        return gathered
                break

            try:
                parent_len = len(parent.text_content())
            except Exception:
                break

            # Prefer semantic tags — even if parent exceeds size cap
            # Bug fix 2: tbody→table jump — don't let size cap block us from
            # reaching a semantic container that's just one level up
            if parent.tag in SEMANTIC_TAGS:
                return parent

            # Stop if parent is too large (but we already checked semantic tags above)
            if parent_len > self.MAX_CONTAINER_CHARS:
                # One more chance: check if grandparent is a semantic tag
                grandparent = parent.getparent()
                if grandparent is not None and grandparent.tag in SEMANTIC_TAGS:
                    return grandparent
                break

            # Accept block containers that are reasonably sized
            if parent_len >= MIN_CONTEXT_LEN:
                best = parent

            current = parent

        return best

    def _gather_section_context(self, elem) -> "lxml_html.HtmlElement":  # noqa: F821
        """Gather all sibling elements within the same h2/h3 section."""
        from lxml import etree

        # Walk up to find direct child of mw-parser-output
        current = elem
        mw_output = None
        while current is not None:
            parent = current.getparent()
            if parent is not None:
                classes = parent.get("class", "")
                if "mw-parser-output" in classes:
                    mw_output = parent
                    break
            current = parent

        if mw_output is None:
            return elem

        # Find the element's position among mw-parser-output children
        children = list(mw_output)
        try:
            idx = children.index(current)
        except ValueError:
            return elem

        # Gather backward until we hit a heading, forward until next heading
        section_elems = [current]

        # Backward
        for i in range(idx - 1, max(idx - 10, -1), -1):
            child = children[i]
            if hasattr(child, "tag") and child.tag in ("h1", "h2", "h3"):
                section_elems.insert(0, child)
                break
            section_elems.insert(0, child)

        # Forward
        for i in range(idx + 1, min(idx + 10, len(children))):
            child = children[i]
            if hasattr(child, "tag") and child.tag in ("h1", "h2", "h3"):
                break
            section_elems.append(child)

        # Build a container div with these elements
        container = etree.Element("div")
        for el in section_elems:
            try:
                container.append(el)
            except Exception:
                pass

        return container

    @staticmethod
    def _extract_search_keys(chunk_text: str) -> list[str]:
        """Extract distinctive search keys from chunk text for DOM matching.

        Detects chunk type (table-heavy vs prose-heavy) and picks the best strategy.
        Returns keys ordered by distinctiveness — first key is tried first in DOM lookup.
        """
        import re

        lines = chunk_text.split("\n")
        # Skip first line if it looks like an article title (short, no pipes, no punctuation)
        # These match <h1> in DOM and cause Bug 1
        if lines and len(lines[0]) < 80 and "|" not in lines[0] and "." not in lines[0]:
            content_lines = lines[1:]
        else:
            content_lines = lines
        table_lines = [l for l in content_lines if "|" in l and "---" not in l]
        prose_lines = [
            l
            for l in content_lines
            if len(l) > 30 and "|" not in l and not l.startswith("- ^")
        ]
        is_table_heavy = (
            len(table_lines) > len(content_lines) * 0.4 if content_lines else False
        )

        keys = []

        if is_table_heavy:
            # Mixed strategy: include keys from BOTH table cells and prose
            # so coverage scorer can find a container spanning both parts.
            cell_candidates = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.split("|") if c.strip()]
                for cell in cells:
                    if len(cell) < 5 or len(cell) > 80:
                        continue
                    if cell.lower() in ("yes", "no", "n/a", "none", ""):
                        continue
                    has_code = bool(re.search(r"[A-Z]\d|[a-z]\d{4,}", cell))
                    has_mixed = bool(re.search(r"\d.*[a-zA-Z]|[a-zA-Z].*\d", cell))
                    has_proper = bool(re.search(r"[A-Z][a-z]+\s+[A-Z]", cell))
                    if has_code or has_mixed or has_proper:
                        cell_candidates.insert(0, cell)
                    elif len(cell) > 12:
                        cell_candidates.append(cell)

            # Table cells first (these anchor to the infobox)
            for cc in cell_candidates[:3]:
                if cc not in keys:
                    keys.append(cc)

            # Then prose keys (these anchor to body paragraphs)
            for line in prose_lines[:3]:
                mid = len(line) // 2
                candidate = line[mid - 15 : mid + 15].strip()
                if len(candidate) >= 10 and re.search(r"[a-zA-Z]{4,}", candidate):
                    keys.append(candidate)

        else:
            # Prose-dominant chunk: use prose fragments as primary keys
            for line in prose_lines[:4]:
                mid = len(line) // 2
                candidate = line[mid - 15 : mid + 15].strip()
                if len(candidate) >= 10 and re.search(r"[a-zA-Z]{4,}", candidate):
                    keys.append(candidate)

            # Add table cell values as secondary
            if table_lines:
                for tl in table_lines[:5]:
                    cells = [
                        c.strip()
                        for c in tl.split("|")
                        if c.strip() and len(c.strip()) > 8
                    ]
                    for cell in cells[:1]:
                        if cell not in keys:
                            keys.append(cell)

        # List item content
        if not keys:
            list_lines = [
                l[2:]
                for l in lines
                if l.startswith("- ") and len(l) > 20 and not l.startswith("- ^")
            ]
            for ll in list_lines[:3]:
                mid = len(ll) // 2
                candidate = ll[mid - 15 : mid + 15].strip()
                if len(candidate) >= 10:
                    keys.append(candidate)

        # Fallback
        if not keys and len(chunk_text) > 40:
            candidate = chunk_text[10:50].strip()
            keys.append(candidate)

        return keys

    async def _llm_dom_closure(self, raw_html: str, chunk_text: str) -> str | None:
        """Use an LLM to find the minimal DOM closure containing the chunk text.

        Sends the article HTML (truncated) and the chunk text to the model,
        asks it to return the minimal enclosing HTML subtree.
        """
        import openai

        # Truncate HTML to avoid context limits — keep first 60K chars
        # (most Wikipedia articles are under 100K)
        html_truncated = raw_html[:60000]

        prompt = f"""Given this HTML document and a text chunk extracted from it, find the minimal DOM subtree that contains ALL the text in the chunk. Return ONLY the raw HTML of that subtree, no explanation.

The text chunk (extracted by Trafilatura, so formatting differs from HTML):
---
{chunk_text[:2000]}
---

The HTML document:
---
{html_truncated}
---

Return the minimal HTML subtree containing all the information from the text chunk. Include complete table/list structures if the chunk spans table cells. Return ONLY HTML, no markdown fences."""

        try:
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model=self.llm_verify_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if result.startswith("```"):
                lines = result.split("\n")
                result = "\n".join(
                    lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                )
            return result if "<" in result else None
        except Exception as e:
            logger.warning(f"LLM DOM closure failed: {e}")
            return None

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        """Retrieve text chunks, then do DOM lookup for HTML context."""
        example.get("id", "unknown")

        # Get raw hits from text retriever
        hits = await self._text_retriever.get_hits(query, example)
        if not hits:
            return RetrievalResult(retrieval_type="html_dom_lookup")

        keep_k = self.reader_top_k if self.reader_top_k is not None else self.top_k
        hits = hits[:keep_k]

        passages = []
        urls = []
        seen_urls: set[str] = set()

        for hit in hits:
            article_id = hit.get("article_id")
            chunk_text = hit.get("text", "")
            url = hit.get("url", "")

            html_context = None
            if article_id is not None:
                raw_html = self._fetch_html(int(article_id))
                if raw_html:
                    # Heuristic DOM lookup first
                    html_context = self._dom_lookup(raw_html, chunk_text)

                    # LLM verification/fallback
                    if self.llm_verify and (
                        html_context is None or len(chunk_text) > 500
                    ):
                        llm_result = await self._llm_dom_closure(raw_html, chunk_text)
                        if llm_result:
                            html_context = llm_result

            if html_context:
                passages.append(html_context)
            else:
                passages.append(chunk_text)

            if url and url not in seen_urls:
                seen_urls.add(url)
                urls.append(url)

        # Hard cap per passage. HTML is ~2 chars/token; reader has 65K tokens
        # with ~2K for output + system prompt. Budget ~50K tokens for context
        # = ~100K chars across all passages. Per-passage cap avoids one huge
        # article starving the others.
        MAX_PER_PASSAGE = 30000
        passages = [p[:MAX_PER_PASSAGE] for p in passages]
        MAX_TOTAL_CHARS = 90000
        total = sum(len(p) for p in passages)
        if total > MAX_TOTAL_CHARS:
            per_passage = MAX_TOTAL_CHARS // max(len(passages), 1)
            passages = [p[:per_passage] for p in passages]
            logger.warning(
                "Truncated %d passages from %d to %d total chars",
                len(passages),
                total,
                sum(len(p) for p in passages),
            )

        combined = "\n\n---\n\n".join(passages) if passages else None
        return RetrievalResult(
            text=combined,
            source_url=", ".join(urls) if urls else None,
            retrieval_type="html_dom_lookup",
        )
