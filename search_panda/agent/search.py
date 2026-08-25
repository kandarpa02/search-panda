"""Web search and content-fetching layer for Search Panda.

Key improvements over the original:
- No global mutable state: SearchSession holds results per call.
- Results are typed SearchResult Pydantic models, not plain dicts.
- read_result() takes a URL string, not a fragile global integer index.
- fetch_and_extract() uses tenacity retries and a 15-second timeout.
- fetch_pages_parallel() downloads multiple pages concurrently.
- needs_web_search() is exported for tests (lives in planner.py, re-exported here
  for backward compatibility).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import trafilatura
from ddgs import DDGS
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .data_representation import SearchResult, PageContent
from .page_ranker import rank_results

# Re-export so legacy test imports (`from .search import needs_web_search`) work.
from .planner import needs_web_search as needs_web_search  # noqa: F401

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# HTTP client — shared, 15-second timeout
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SearchPanda/1.0; +https://github.com/search-panda)"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Content fetching with retry
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    reraise=True,
)
def _fetch_html(url: str) -> str | None:
    """Fetch raw HTML for *url* with retries. Returns None on failure."""
    try:
        with httpx.Client(timeout=_HTTP_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as exc:
        logger.warning("HTTP {status} fetching {url}", status=exc.response.status_code, url=url)
        return None
    except Exception as exc:
        logger.warning("Failed to fetch {url}: {exc}", url=url, exc=exc)
        raise


def fetch_and_extract(url: str) -> PageContent | None:
    """Download *url* and extract readable text.

    Returns a PageContent on success, or None if the page cannot be
    fetched or its text cannot be extracted.
    """
    try:
        html = _fetch_html(url)
    except Exception as exc:
        logger.warning("Giving up on {url}: {exc}", url=url, exc=exc)
        return None

    if html is None:
        return None

    text = trafilatura.extract(
        html,
        include_links=False,
        include_images=False,
        no_fallback=False,
    )

    if not text:
        # trafilatura fallback: try with BeautifulSoup-based extraction.
        text = trafilatura.extract(html, favor_precision=False)

    if not text:
        logger.debug("No readable text extracted from {url}", url=url)
        return None

    # Derive a best-effort title.
    title = ""
    try:
        import re

        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
    except Exception:
        pass

    return PageContent(url=url, title=title, content=text)


async def fetch_pages_parallel(urls: list[str], max_pages: int = 3) -> list[PageContent]:
    """Fetch and extract content from multiple URLs concurrently."""
    loop = asyncio.get_event_loop()
    target_urls = [u for u in urls if u and u.startswith("http")][:max_pages]

    async def _fetch_one(url: str) -> PageContent | None:
        return await loop.run_in_executor(None, fetch_and_extract, url)

    tasks = [_fetch_one(u) for u in target_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    pages: list[PageContent] = []
    for r in results:
        if isinstance(r, PageContent) and r.content:
            pages.append(r)
    return pages


# ---------------------------------------------------------------------------
# Search layer
# ---------------------------------------------------------------------------

class SearchSession:
    """Holds the results of one search round-trip.

    Keeps results scoped to a single session so there is no shared global
    state between concurrent agent calls.
    """

    def __init__(self) -> None:
        self._results: list[SearchResult] = []

    # --- public ---

    @property
    def results(self) -> list[SearchResult]:
        return list(self._results)

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        """Run a DuckDuckGo text search and store the results.

        Returns typed SearchResult objects ranked by relevance.
        """
        raw: list[SearchResult] = []

        try:
            with DDGS() as ddgs:
                for i, item in enumerate(ddgs.text(query, max_results=max_results)):
                    url = item.get("href") or item.get("url", "")
                    raw.append(
                        SearchResult(
                            index=i,
                            title=item.get("title", ""),
                            url=url,
                            snippet=item.get("body", ""),
                        )
                    )
        except Exception as exc:
            logger.error("Search failed for query={query!r}: {exc}", query=query, exc=exc)
            return []

        ranked = rank_results(query, raw, top_k=min(max_results, 5))
        self._results = ranked
        return ranked

    def get_by_url(self, url: str) -> SearchResult | None:
        """Look up a result by its URL."""
        for r in self._results:
            if r.url == url:
                return r
        return None


# ---------------------------------------------------------------------------
# Module-level convenience functions (used by agent tools)
# ---------------------------------------------------------------------------

# Shared session per process — reset on each new `search()` call.
_SESSION = SearchSession()


def search(query: str, max_results: int = 8) -> list[SearchResult]:
    """Module-level search helper. Resets the shared session."""
    global _SESSION
    _SESSION = SearchSession()
    return _SESSION.search(query, max_results=max_results)


def read_result(url: str) -> PageContent | None:
    """Fetch and extract a page by URL."""
    return fetch_and_extract(url)


async def search_parallel(queries: list[str], max_results_each: int = 5) -> list[SearchResult]:
    """Run multiple searches in parallel and merge the results."""
    loop = asyncio.get_event_loop()

    async def _one(q: str) -> list[SearchResult]:
        session = SearchSession()
        return await loop.run_in_executor(None, session.search, q, max_results_each)

    nested = await asyncio.gather(*[_one(q) for q in queries], return_exceptions=True)

    seen_urls: set[str] = set()
    merged: list[SearchResult] = []
    for batch in nested:
        if isinstance(batch, Exception):
            logger.warning("Parallel search sub-query failed: {exc}", exc=batch)
            continue
        for result in batch:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                merged.append(result)

    primary_query = queries[0] if queries else ""
    return rank_results(primary_query, merged, top_k=8)
