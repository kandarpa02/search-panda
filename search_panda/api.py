"""Search Panda Python API & Web App Integration SDK.

Use this module to interact with Search Panda directly from Python applications,
scripts, background workers, FastAPI/Flask/Django services, or Streamlit apps.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from .agent.agents import agent_completion_structured
from .agent.data_representation import QueryPlan, SearchResponse, SearchResult, Source
from .agent.planner import plan as query_planner
from .agent.search import search as search_engine, search_parallel
from .config import Setup, ConfigManager, DEFAULT_MODEL, DEFAULT_OLLAMA_BASE_URL, BASE_URL


class SearchPanda:
    """High-level Python SDK client for Search Panda.

    Example:
    ```python
    from search_panda import SearchPanda

    # Initialize client (uses Ollama by default or any OpenAI-compatible provider)
    client = SearchPanda(model="llama3.2:1b")
    response = client.ask("who won fifa 2026?")

    print(response.answer)
    for source in response.sources:
        print(f"- [{source.title}]({source.url})")
    ```
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        provider: str = "ollama",
        api_key: str | None = None,
        base_url: str | None = None,
        web_mode: str = "on",
        reasoning_level: str | None = None,
        load_config: bool = False,
    ) -> None:
        if load_config:
            manager = ConfigManager()
            saved = manager.load()
            self.setup = Setup(
                model=model if model != DEFAULT_MODEL else saved.model,
                provider=provider if provider != "ollama" else saved.provider,
                api_key=api_key or saved.api_key,
                base_url=base_url or saved.base_url,
                web_mode=web_mode if web_mode != "on" else saved.web_mode,
                reasoning_level=reasoning_level or saved.reasoning_level,
            ).resolve()
        else:
            self.setup = Setup(
                model=model,
                provider=provider,
                api_key=api_key or ("ollama" if provider == "ollama" else None),
                base_url=base_url or BASE_URL.get(provider, DEFAULT_OLLAMA_BASE_URL),
                web_mode=web_mode,
                reasoning_level=reasoning_level,
            ).resolve()

    # -----------------------------------------------------------------------
    # Core Queries
    # -----------------------------------------------------------------------

    def ask(self, query: str | Sequence[str]) -> SearchResponse:
        """Run an AI search query synchronously and return a SearchResponse."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an active async event loop (e.g. Jupyter or nested async)
            # import nest_asyncio  # noqa: F401 (optional)
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.ask_async(query)).result()
        else:
            return asyncio.run(self.ask_async(query))

    async def ask_async(self, query: str | Sequence[str]) -> SearchResponse:
        """Run an AI search query asynchronously and return a SearchResponse."""
        return await agent_completion_structured(config=self.setup, message=query)

    # -----------------------------------------------------------------------
    # Lower-level helpers
    # -----------------------------------------------------------------------

    def plan(self, query: str) -> QueryPlan:
        """Analyze query intent and build a query execution plan."""
        return query_planner(query, web_mode=self.setup.web_mode)

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        """Perform a web search using Search Panda's ranking engine."""
        return search_engine(query, max_results=max_results)

    async def search_async(self, queries: list[str], max_results_each: int = 5) -> list[SearchResult]:
        """Run parallel searches across multiple sub-queries."""
        return await search_parallel(queries, max_results_each=max_results_each)

    # -----------------------------------------------------------------------
    # Configuration updates
    # -----------------------------------------------------------------------

    def set_model(self, model: str) -> SearchPanda:
        """Switch active LLM model."""
        self.setup.model = model.strip()
        self.setup.resolve()
        return self

    def set_provider(self, provider: str, api_key: str | None = None, base_url: str | None = None) -> SearchPanda:
        """Switch active LLM provider (e.g. 'openai', 'groq', 'deepseek', 'openrouter')."""
        self.setup.provider = provider.strip().lower()
        if api_key:
            self.setup.api_key = api_key
        if base_url:
            self.setup.base_url = base_url
        self.setup.resolve()
        return self

    def set_web_mode(self, mode: str) -> SearchPanda:
        """Switch web search mode ('on', 'off', 'auto')."""
        self.setup.web_mode = mode.strip().lower()
        return self

    def set_reasoning_level(self, level: str | None) -> SearchPanda:
        """Set reasoning effort ('low', 'medium', 'high' or None)."""
        self.setup.reasoning_level = level
        return self


# ---------------------------------------------------------------------------
# Global Convenience Functions
# ---------------------------------------------------------------------------

def ask(
    query: str,
    model: str = DEFAULT_MODEL,
    provider: str = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
    web_mode: str = "on",
    reasoning_level: str | None = None,
) -> SearchResponse:
    """Synchronous shortcut to query Search Panda."""
    client = SearchPanda(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        web_mode=web_mode,
        reasoning_level=reasoning_level,
    )
    return client.ask(query)


async def ask_async(
    query: str,
    model: str = DEFAULT_MODEL,
    provider: str = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
    web_mode: str = "on",
    reasoning_level: str | None = None,
) -> SearchResponse:
    """Asynchronous shortcut to query Search Panda."""
    client = SearchPanda(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        web_mode=web_mode,
        reasoning_level=reasoning_level,
    )
    return await client.ask_async(query)


# Alias
async_ask = ask_async


# ---------------------------------------------------------------------------
# FastAPI Web App Factory
# ---------------------------------------------------------------------------
def create_api_app(
    default_model: str = DEFAULT_MODEL,
    default_provider: str = "ollama",
    default_api_key: str | None = None,
    default_base_url: str | None = None,
    default_web_mode: str = "on",
    default_reasoning_level: str | None = None,
) -> Any:
    """Create and return a provider-agnostic FastAPI application.

    Supports:
    - Ollama
    - OpenAI
    - Groq
    - OpenRouter
    - DeepSeek
    - Any provider supported by Search Panda's Setup configuration
    - Custom OpenAI-compatible APIs via base_url

    API key priority:
        request api_key
        -> app default_api_key
        -> saved configuration api_key
    """

    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required for create_api_app(). "
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Search Panda API",
        description="Provider-agnostic agentic AI search API",
        version="0.2.0",
    )

    # Load saved config so existing Search Panda configuration
    # continues to work as a fallback.
    config_client = SearchPanda(
        model=default_model,
        provider=default_provider,
        api_key=default_api_key,
        base_url=default_base_url,
        web_mode=default_web_mode,
        reasoning_level=default_reasoning_level,
        load_config=True,
    )

    class SearchRequest(BaseModel):
        query: str = Field(
            ...,
            min_length=1,
            description="The user question or search prompt",
        )

        model: str | None = Field(
            default=None,
            description="Model name to use for this request",
        )

        provider: str | None = Field(
            default=None,
            description=(
                "LLM provider. Examples: ollama, openai, "
                "groq, openrouter, deepseek"
            ),
        )

        api_key: str | None = Field(
            default=None,
            description=(
                "API key for this request. "
                "Not required for Ollama."
            ),
        )

        base_url: str | None = Field(
            default=None,
            description=(
                "Optional custom API base URL. "
                "Useful for OpenAI-compatible providers."
            ),
        )

        web_mode: str | None = Field(
            default=None,
            description="Web search mode: on, off, or auto",
        )

        reasoning_level: str | None = Field(
            default=None,
            description="Reasoning effort: low, medium, or high",
        )

    class PlanRequest(BaseModel):
        query: str = Field(
            ...,
            min_length=1,
            description="Query to analyze and plan",
        )

        web_mode: str | None = Field(
            default=None,
            description="Web search mode: on, off, or auto",
        )

    def resolve_request_setup(req: SearchRequest) -> Setup:
        """Create an isolated provider configuration per request."""

        provider = (
            req.provider
            or default_provider
            or config_client.setup.provider
        ).strip().lower()

        model = (
            req.model
            or default_model
            or config_client.setup.model
        ).strip()

        # Priority:
        # request -> app default -> saved config
        api_key = (
            req.api_key
            or default_api_key
            or config_client.setup.api_key
        )

        # Priority:
        # request -> app default -> saved config
        base_url = (
            req.base_url
            or default_base_url
            or config_client.setup.base_url
        )

        web_mode = (
            req.web_mode
            or default_web_mode
            or config_client.setup.web_mode
        )

        reasoning_level = (
            req.reasoning_level
            or default_reasoning_level
            or config_client.setup.reasoning_level
        )

        # Ollama does not require a user-supplied API key.
        if provider == "ollama" and not api_key:
            api_key = "ollama"

        # Non-Ollama providers require credentials.
        elif provider != "ollama" and not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"An API key is required for provider "
                    f"'{provider}'."
                ),
            )

        try:
            return Setup(
                model=model,
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                web_mode=web_mode,
                reasoning_level=reasoning_level,
            ).resolve()

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid provider configuration: {exc}",
            )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Return API status and default configuration.

        API keys are intentionally never exposed.
        """

        return {
            "status": "healthy",
            "default_model": config_client.setup.model,
            "default_provider": config_client.setup.provider,
            "default_base_url": config_client.setup.base_url,
            "default_web_mode": config_client.setup.web_mode,
            "default_reasoning_level": (
                config_client.setup.reasoning_level
            ),
        }

    @app.post(
        "/api/search",
        response_model=SearchResponse,
    )
    async def api_search(
        req: SearchRequest,
    ) -> SearchResponse:

        if not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        setup = resolve_request_setup(req)

        try:
            return await agent_completion_structured(
                config=setup,
                message=req.query,
            )

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Search execution failed: {str(exc)}",
            )

    @app.post(
        "/api/plan",
        response_model=QueryPlan,
    )
    
    def api_plan(
        req: PlanRequest,
    ) -> QueryPlan:

        if not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        web_mode = (
            req.web_mode
            or default_web_mode
            or config_client.setup.web_mode
        )

        try:
            return query_planner(
                req.query,
                web_mode=web_mode,
            )

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Planning failed: {str(exc)}",
            )

    return app