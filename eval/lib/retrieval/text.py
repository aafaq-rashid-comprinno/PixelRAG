"""Text-based retrievers.

Retrievers that fetch and provide text content as context:
- TextRetriever: use pre-fetched/cached text
- JinaReaderRetriever: fetch via Jina Reader API
- WikipediaAPIRetriever: fetch via Wikipedia API
"""

import asyncio
import base64
import io
import json
import logging
import os
import re

from .base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


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


class JinaReaderRetriever(BaseRetriever):
    """Use Jina Reader API to fetch clean markdown text from URL.

    Jina Reader (r.jina.ai) converts any URL to LLM-friendly markdown text.
    """

    def __init__(
        self,
        max_chars: int = 50000,
        api_key: str | None = None,
        text_cache: dict | None = None,
        cache_path: str | None = None,
    ):
        self.max_chars = max_chars
        self.api_key = api_key
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
        import aiohttp
        import asyncio
        from .simpleqa_data import extract_url_from_metadata

        # Check cache first
        example_id = example.get("id", "")
        if self.text_cache and example_id in self.text_cache:
            cached = self.text_cache[example_id]
            text = cached.get("text", "")
            source_url = cached.get("url", "")
            if text:
                if len(text) > self.max_chars:
                    text = text[: self.max_chars] + "\n\n[Content truncated...]"
                return RetrievalResult(
                    text=text, source_url=source_url, retrieval_type="jina_reader"
                )

        target_url = extract_url_from_metadata(example)
        if not target_url:
            return RetrievalResult(
                text="No URL found in metadata.", retrieval_type="jina_reader"
            )

        # Use Jina Reader API with retry logic
        reader_url = f"https://r.jina.ai/{target_url}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        reader_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as response:
                        # Handle rate limiting (429) with exponential backoff
                        if response.status == 429:
                            if attempt < max_retries - 1:
                                wait_time = min(2**attempt * 2, 30)  # Max 30 seconds
                                logger.warning(
                                    f"Rate limited (429) for {target_url}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                error_msg = f"Jina Reader API rate limited (429) after {max_retries} retries"
                                logger.error(f"{error_msg} for {target_url}")
                                return RetrievalResult(
                                    text=error_msg,
                                    source_url=target_url,
                                    retrieval_type="jina_reader",
                                )

                        # Handle server errors (5xx) with retry
                        if 500 <= response.status < 600:
                            if attempt < max_retries - 1:
                                wait_time = min(2**attempt, 10)  # Max 10 seconds
                                logger.warning(
                                    f"Server error ({response.status}) for {target_url}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                error_msg = (
                                    f"Jina Reader API server error: {response.status}"
                                )
                                logger.error(f"{error_msg} for {target_url}")
                                return RetrievalResult(
                                    text=error_msg,
                                    source_url=target_url,
                                    retrieval_type="jina_reader",
                                )

                        # Handle client errors (4xx) - don't retry for most
                        if response.status == 200:
                            text = await response.text()
                            # Save to cache before truncation
                            await self._save_to_cache(example_id, text, target_url)
                            # Truncate if too long
                            if len(text) > self.max_chars:
                                text = (
                                    text[: self.max_chars]
                                    + "\n\n[Content truncated...]"
                                )
                            return RetrievalResult(
                                text=text,
                                source_url=target_url,
                                retrieval_type="jina_reader",
                            )
                        else:
                            # Other 4xx errors (403, 404, etc.) - don't retry
                            error_msg = f"Jina Reader API error: {response.status}"
                            logger.warning(f"{error_msg} for {target_url}")
                            return RetrievalResult(
                                text=error_msg,
                                source_url=target_url,
                                retrieval_type="jina_reader",
                            )
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 10)  # Max 10 seconds
                    logger.warning(
                        f"Timeout for {target_url}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Jina Reader fetch timeout after {max_retries} retries"
                    logger.error(f"{error_msg} for {target_url}")
                    return RetrievalResult(
                        text=error_msg,
                        source_url=target_url,
                        retrieval_type="jina_reader",
                    )
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    wait_time = min(2**attempt, 10)  # Max 10 seconds
                    logger.warning(
                        f"Client error for {target_url}: {e}, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Jina Reader fetch failed: {e}"
                    logger.error(f"{error_msg} for {target_url}")
                    return RetrievalResult(
                        text=error_msg,
                        source_url=target_url,
                        retrieval_type="jina_reader",
                    )
            except Exception as e:
                error_msg = f"Jina Reader fetch failed: {e}"
                logger.error(f"{error_msg} for {target_url}")
                return RetrievalResult(
                    text=error_msg, source_url=target_url, retrieval_type="jina_reader"
                )

        # Should not reach here, but just in case
        error_msg = f"Jina Reader fetch failed after {max_retries} retries"
        return RetrievalResult(
            text=error_msg, source_url=target_url, retrieval_type="jina_reader"
        )


class WikipediaAPIRetriever(BaseRetriever):
    """Use Wikipedia API to fetch clean article text.

    Extracts Wikipedia page title from URL and fetches content via API.
    Much cleaner and faster than web scraping.
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

    def _extract_wiki_title(self, url: str) -> str | None:
        """Extract Wikipedia page title from URL."""
        import re
        from urllib.parse import unquote

        # Match patterns like:
        # https://en.wikipedia.org/wiki/Python_(programming_language)
        # https://zh.wikipedia.org/wiki/Artificial_intelligence
        pattern = r"https?://[a-z]{2,3}\.wikipedia\.org/wiki/(.+?)(?:#.*)?$"
        match = re.match(pattern, url)
        if match:
            title = unquote(match.group(1))
            # Replace underscores with spaces
            title = title.replace("_", " ")
            return title
        return None

    def _get_wiki_lang(self, url: str) -> str:
        """Extract Wikipedia language code from URL."""
        import re

        match = re.match(r"https?://([a-z]{2,3})\.wikipedia\.org", url)
        return match.group(1) if match else "en"

    def _html_to_text(self, html: str) -> str:
        """Convert Wikipedia HTML to plain text, preserving table content."""
        import re
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove unwanted elements
        for tag in soup.find_all(["script", "style", "link", "meta"]):
            tag.decompose()

        # Remove edit section links
        for tag in soup.find_all("span", class_="mw-editsection"):
            tag.decompose()

        # Remove reference numbers [1], [2], etc.
        for tag in soup.find_all("sup", class_="reference"):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    def _parse_infobox(self, wikitext: str) -> str:
        """Parse infobox from wikitext and convert to plain text."""
        import re

        # Find infobox start
        start = wikitext.find("{{Infobox")
        if start == -1:
            start = wikitext.find("{{infobox")
        if start == -1:
            return ""

        # Count braces to find matching end
        depth = 0
        end = start
        for i in range(start, len(wikitext)):
            if wikitext[i : i + 2] == "{{":
                depth += 1
            elif wikitext[i : i + 2] == "}}":
                depth -= 1
                if depth == 0:
                    end = i + 2
                    break

        infobox_raw = wikitext[start:end]

        # Parse fields
        lines = []
        for match in re.finditer(
            r"\|\s*([^=|]+?)\s*=\s*([^|]*?)(?=\n\s*\||\}\})", infobox_raw, re.DOTALL
        ):
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Skip image-related fields
            if key.lower() in (
                "image",
                "caption",
                "alt",
                "width",
                "height",
                "image_size",
                "imagesize",
            ):
                continue

            # Clean up wikitext markup
            value = re.sub(
                r"\{\{[^}|]*\|([^}]*)\}\}", r"\1", value
            )  # {{template|value}} -> value
            value = re.sub(
                r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", value
            )  # [[link|text]] -> text
            value = re.sub(r"'''?", "", value)  # bold/italic
            value = re.sub(r"<[^>]+>", "", value)  # HTML tags
            value = re.sub(r"\{\{[^}]*\}\}", "", value)  # remaining templates
            value = " ".join(value.split())  # normalize whitespace

            if value:
                lines.append(f"{key}: {value}")

        return "\n".join(lines)

    async def retrieve(self, query: str, example: dict) -> RetrievalResult:
        import aiohttp
        from .simpleqa_data import extract_url_from_metadata

        # Check cache first
        example_id = example.get("id", "")
        if self.text_cache and example_id in self.text_cache:
            cached = self.text_cache[example_id]
            text = cached.get("text", "")
            source_url = cached.get("url", "")
            if text:
                if len(text) > self.max_chars:
                    text = text[: self.max_chars] + "\n\n[Content truncated...]"
                return RetrievalResult(
                    text=text, source_url=source_url, retrieval_type="wikipedia_api"
                )

        target_url = extract_url_from_metadata(example)
        if not target_url:
            return RetrievalResult(
                text="No URL found in metadata.", retrieval_type="wikipedia_api"
            )

        # Check if it's a Wikipedia URL
        if "wikipedia.org" not in target_url.lower():
            return RetrievalResult(
                text=f"URL is not a Wikipedia page: {target_url}",
                source_url=target_url,
                retrieval_type="wikipedia_api",
            )

        title = self._extract_wiki_title(target_url)
        if not title:
            return RetrievalResult(
                text=f"Could not extract Wikipedia title from: {target_url}",
                source_url=target_url,
                retrieval_type="wikipedia_api",
            )

        lang = self._get_wiki_lang(target_url)
        api_url = f"https://{lang}.wikipedia.org/w/api.php"

        headers = {
            "User-Agent": "SimpleQA-Evaluation/1.0 (https://github.com/example; contact@example.com)"
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Use action=parse to get full HTML (includes tables)
                parse_params = {
                    "action": "parse",
                    "page": title,
                    "prop": "text",
                    "format": "json",
                    "redirects": "1",
                }

                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(
                    api_url, params=parse_params, headers=headers, timeout=timeout
                ) as resp:
                    if resp.status != 200:
                        error_msg = f"Wikipedia API error: {resp.status}"
                        logger.warning(f"{error_msg} for {target_url}")
                        return RetrievalResult(
                            text=error_msg,
                            source_url=target_url,
                            retrieval_type="wikipedia_api",
                        )

                    data = await resp.json()

                    # Check for error
                    if "error" in data:
                        error_msg = data["error"].get("info", "Unknown error")
                        return RetrievalResult(
                            text=f"Wikipedia page not found: {title}",
                            source_url=target_url,
                            retrieval_type="wikipedia_api",
                        )

                    html = data.get("parse", {}).get("text", {}).get("*", "")
                    if not html:
                        return RetrievalResult(
                            text=f"No content found for Wikipedia page: {title}",
                            source_url=target_url,
                            retrieval_type="wikipedia_api",
                        )

                    # Parse HTML to text (includes tables)
                    text = self._html_to_text(html)

                    # Save to cache before truncation
                    await self._save_to_cache(example_id, text, target_url)

                    # Truncate if too long
                    if len(text) > self.max_chars:
                        text = text[: self.max_chars] + "\n\n[Content truncated...]"

                    return RetrievalResult(
                        text=text, source_url=target_url, retrieval_type="wikipedia_api"
                    )
        except Exception as e:
            error_msg = f"Wikipedia API fetch failed: {e}"
            logger.warning(f"{error_msg} for {target_url}")
            return RetrievalResult(
                text=error_msg, source_url=target_url, retrieval_type="wikipedia_api"
            )


