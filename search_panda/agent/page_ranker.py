"""Result ranking for Search Panda.

Scores and orders search results using:
1. BM25-style keyword overlap between the query and title/snippet.
2. Domain authority signals (a curated allowlist of trusted domains).
3. Deduplication by normalised domain so diverse sources are returned.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

from .data_representation import SearchResult

# ---------------------------------------------------------------------------
# Domain authority tiers
# (higher score = more trusted / relevant for general use)
# ---------------------------------------------------------------------------

_DOMAIN_SCORES: dict[str, float] = {
    # encyclopaedic / reference
    "wikipedia.org": 0.9,
    "britannica.com": 0.85,
    "scholarpedia.org": 0.8,
    # news / journalism
    "reuters.com": 0.9,
    "apnews.com": 0.9,
    "bbc.com": 0.85,
    "bbc.co.uk": 0.85,
    "theguardian.com": 0.8,
    "nytimes.com": 0.8,
    "bloomberg.com": 0.8,
    "ft.com": 0.8,
    "economist.com": 0.8,
    "techcrunch.com": 0.75,
    "theverge.com": 0.75,
    "wired.com": 0.75,
    "arstechnica.com": 0.75,
    # developer / technical
    "stackoverflow.com": 0.85,
    "github.com": 0.85,
    "docs.python.org": 0.9,
    "developer.mozilla.org": 0.9,
    "realpython.com": 0.8,
    "python.org": 0.85,
    "pypi.org": 0.8,
    "npmjs.com": 0.8,
    # science / academic
    "arxiv.org": 0.9,
    "pubmed.ncbi.nlm.nih.gov": 0.9,
    "scholar.google.com": 0.8,
    "nature.com": 0.85,
    "science.org": 0.85,
    # general tech docs
    "readthedocs.io": 0.75,
    "docs.microsoft.com": 0.8,
    "learn.microsoft.com": 0.8,
    "cloud.google.com": 0.8,
    "aws.amazon.com": 0.75,
    # AI / ML
    "openai.com": 0.85,
    "huggingface.co": 0.85,
    "pytorch.org": 0.85,
    "tensorflow.org": 0.85,
    "deepmind.google": 0.85,
}

_DEFAULT_DOMAIN_SCORE: float = 0.5


def _domain_authority(domain: str) -> float:
    """Return a [0, 1] authority score for a given domain."""
    d = domain.lower()
    if d.startswith("www."):
        d = d[4:]
    # Exact match
    if d in _DOMAIN_SCORES:
        return _DOMAIN_SCORES[d]
    # Subdomain match — e.g. "en.wikipedia.org" matches "wikipedia.org"
    for known, score in _DOMAIN_SCORES.items():
        if d.endswith("." + known) or d == known:
            return score
    return _DEFAULT_DOMAIN_SCORE


# ---------------------------------------------------------------------------
# BM25-style keyword overlap
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_score(
    query_tokens: list[str],
    field_tokens: list[str],
    k1: float = 1.5,
    b: float = 0.75,
    avg_field_len: float = 20.0,
) -> float:
    """Compute a simple BM25-style relevance score."""
    if not query_tokens or not field_tokens:
        return 0.0

    tf_map = Counter(field_tokens)
    field_len = len(field_tokens)
    score = 0.0

    for term in query_tokens:
        tf = tf_map.get(term, 0)
        if tf == 0:
            continue
        # IDF approximation: treat each missing term as DF=1 out of a corpus of 1000.
        idf = math.log((1001) / (tf + 1)) + 1
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * field_len / avg_field_len)
        score += idf * numerator / denominator

    return score


def _relevance_score(query: str, result: SearchResult) -> float:
    """Combine keyword relevance and domain authority into a single score."""
    query_tokens = _tokenise(query)

    title_tokens = _tokenise(result.title)
    snippet_tokens = _tokenise(result.snippet)

    # Title matches are worth more than snippet matches.
    title_score = _bm25_score(query_tokens, title_tokens) * 1.5
    snippet_score = _bm25_score(query_tokens, snippet_tokens)

    keyword_score = title_score + snippet_score

    authority = _domain_authority(result.domain)

    # Blend: 60% keyword relevance (normalised to ~1), 40% domain authority.
    normalised_keyword = min(keyword_score / 10.0, 1.0)
    combined = 0.60 * normalised_keyword + 0.40 * authority

    return round(combined, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_results(
    query: str,
    results: list[SearchResult],
    top_k: int = 5,
    deduplicate: bool = True,
) -> list[SearchResult]:
    """Score, sort, optionally deduplicate, and return the top-k results.

    Args:
        query:       The original (or rewritten) search query.
        results:     Unranked list of SearchResult objects.
        top_k:       Maximum number of results to return.
        deduplicate: When True, only one result per root domain is kept
                     (the highest-scoring one), ensuring source diversity.

    Returns:
        Ranked list of SearchResult with the ``score`` field populated.
    """
    scored: list[SearchResult] = []
    for result in results:
        result.score = _relevance_score(query, result)
        scored.append(result)

    # Sort descending by score.
    scored.sort(key=lambda r: r.score, reverse=True)

    if not deduplicate:
        return scored[:top_k]

    # Keep only the top result per root domain.
    seen_domains: set[str] = set()
    deduped: list[SearchResult] = []
    for result in scored:
        root = result.domain.lstrip("www.")
        if root not in seen_domains:
            seen_domains.add(root)
            deduped.append(result)
        if len(deduped) >= top_k:
            break

    return deduped
