"""Shared data models for Search Panda."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single result returned from a web search."""

    index: int
    title: str
    url: str
    snippet: str
    domain: str = ""
    score: float = 0.0

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        """Derive domain from URL when not supplied."""
        if not self.domain and self.url:
            try:
                from urllib.parse import urlparse

                netloc = urlparse(self.url).netloc.lower()
                self.domain = netloc[4:] if netloc.startswith("www.") else netloc
            except Exception:
                self.domain = ""


class PageContent(BaseModel):
    """Extracted text content from a fetched web page."""

    url: str
    title: str
    content: str
    word_count: int = Field(default=0)

    def model_post_init(self, __context: object) -> None:
        if not self.word_count:
            self.word_count = len(self.content.split())


class QueryPlan(BaseModel):
    """The planner's decision for how to handle a user query."""

    intent: Literal["factual", "news", "code", "math", "opinion", "conversational"]
    needs_search: bool
    sub_queries: list[str]
    site_hints: list[str] = Field(default_factory=list)
    time_sensitive: bool = False