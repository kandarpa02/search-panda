"""Search Panda Python API & Web App Integration SDK.

Use this module to interact with Search Panda directly from Python applications,
scripts, background workers, FastAPI/Flask/Django services, or Streamlit apps.

It also exposes a FastAPI application with:

- Standard JSON search endpoint
- Query planning endpoint
- SSE streaming endpoint for web UIs
- Agent execution lifecycle events
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator, Sequence

from fastapi import HTTPException, Request, FastAPI
from fastapi.responses import JSONResponse
import time
import uuid

from openai import AsyncOpenAI

from search_panda.cloud_deployment import app

from .agent.agents import agent_completion
from .agent.data_representation import (
    QueryPlan,
    SearchResponse,
    SearchResult,
)
from .agent.planner import plan as query_planner
from .agent.search import search as search_engine, search_parallel
from .config import (
    Setup,
    ConfigManager,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    BASE_URL,
)


# ============================================================================
# Python SDK
# ============================================================================


class SearchPanda:
    """High-level Python SDK client for Search Panda."""

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
                web_mode=(
                    web_mode
                    if web_mode != "on"
                    else saved.web_mode
                ),
                reasoning_level=reasoning_level or saved.reasoning_level,
            )

        else:
            self.setup = Setup(
                model=model,
                provider=provider,
                api_key=api_key or (
                    "ollama"
                    if provider == "ollama"
                    else None
                ),
                base_url=(
                    base_url
                    or BASE_URL.get(
                        provider,
                        DEFAULT_OLLAMA_BASE_URL,
                    )
                ),
                web_mode=web_mode,
                reasoning_level=reasoning_level,
            )

    # ------------------------------------------------------------------------
    # Core Queries
    # ------------------------------------------------------------------------

    def ask(
        self,
        query: str | Sequence[str],
    ) -> SearchResponse:
        """Run an AI search query synchronously."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            ) as pool:
                return pool.submit(
                    asyncio.run,
                    self.ask_async(query),
                ).result()

        return asyncio.run(
            self.ask_async(query)
        )

    async def ask_async(
        self,
        query: str | Sequence[str],
    ) -> SearchResponse:
        """Run an AI search query asynchronously."""

        return await agent_completion(
            config=self.setup,
            message=query,
        )

    # ------------------------------------------------------------------------
    # Lower-level helpers
    # ------------------------------------------------------------------------

    def plan(
        self,
        query: str,
    ) -> QueryPlan:
        """Analyze query intent and build an execution plan."""

        return query_planner(
            query,
            web_mode=self.setup.web_mode,
        )

    def search(
        self,
        query: str,
        max_results: int = 8,
    ) -> list[SearchResult]:
        """Perform a web search using Search Panda."""

        return search_engine(
            query,
            max_results=max_results,
        )

    async def search_async(
        self,
        queries: list[str],
        max_results_each: int = 5,
    ) -> list[SearchResult]:
        """Run parallel searches."""

        return await search_parallel(
            queries,
            max_results_each=max_results_each,
        )

    # ------------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------------

    def set_model(
        self,
        model: str,
    ) -> SearchPanda:
        self.setup.model = model.strip()
        self.setup
        return self

    def set_provider(
        self,
        provider: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> SearchPanda:

        self.setup.provider = provider.strip().lower()

        if api_key:
            self.setup.api_key = api_key

        if base_url:
            self.setup.base_url = base_url

        self.setup

        return self

    def set_web_mode(
        self,
        mode: str,
    ) -> SearchPanda:

        self.setup.web_mode = mode.strip().lower()

        return self

    def set_reasoning_level(
        self,
        level: str | None,
    ) -> SearchPanda:

        self.setup.reasoning_level = level

        return self


# ============================================================================
# Convenience Functions
# ============================================================================


def ask(
    query: str,
    model: str = DEFAULT_MODEL,
    provider: str = "ollama",
    api_key: str | None = None,
    base_url: str | None = None,
    web_mode: str = "on",
    reasoning_level: str | None = None,
) -> SearchResponse:

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

    client = SearchPanda(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        web_mode=web_mode,
        reasoning_level=reasoning_level,
    )

    return await client.ask_async(query)


async_ask = ask_async


# ============================================================================
# FastAPI Application
# ============================================================================


def create_api_app(
    default_model: str = DEFAULT_MODEL,
    default_provider: str = "ollama",
    default_api_key: str | None = None,
    default_base_url: str | None = None,
    default_web_mode: str = "on",
    default_reasoning_level: str | None = None,
    model_prefix: str = "search-panda/",
) -> Any:
    """Create a Search Panda FastAPI application.

    Endpoints:

        GET  /api/health
        POST /api/search
        POST /api/search/stream
        POST /api/plan

    The streaming endpoint uses Server-Sent Events.

    Event flow:

        connected
        planning
        plan
        searching
        synthesizing
        answer
        done

    Error event:

        error
    """

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field

    except ImportError:
        raise ImportError(
            "FastAPI is required for create_api_app(). "
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Search Panda API",
        description=(
            "Provider-agnostic agentic AI search API "
            "with real-time execution streaming."
        ),
        version="0.2.0",
    )

    # ------------------------------------------------------------------------
    # Default configuration
    # ------------------------------------------------------------------------

    config_client = SearchPanda(
        model=default_model,
        provider=default_provider,
        api_key=default_api_key,
        base_url=default_base_url,
        web_mode=default_web_mode,
        reasoning_level=default_reasoning_level,
        load_config=True,
    )

    # ------------------------------------------------------------------------
    # Request Models
    # ------------------------------------------------------------------------

    class SearchRequest(BaseModel):

        query: str = Field(
            ...,
            min_length=1,
            description="The user question or search prompt",
        )

        model: str | None = Field(
            default=None,
            description="Model name to use",
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
                "Optional OpenAI-compatible API base URL."
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

    # ------------------------------------------------------------------------
    # Setup Resolver
    # ------------------------------------------------------------------------

    def resolve_request_setup(
        req: SearchRequest,
    ) -> Setup:
        """Create an isolated provider configuration."""

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

        api_key = (
            req.api_key
            or default_api_key
            or config_client.setup.api_key
        )

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

        if provider == "ollama" and not api_key:
            api_key = "ollama"

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
            )

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid provider configuration: {exc}"
                ),
            )

    # ------------------------------------------------------------------------
    # SSE Helpers
    # ------------------------------------------------------------------------

    def serialize(
        value: Any,
    ) -> Any:
        """Convert Pydantic objects to JSON-safe data."""

        if hasattr(value, "model_dump"):
            return value.model_dump()

        if hasattr(value, "dict"):
            return value.dict()

        return value

    def sse_event(
        event: str,
        data: Any,
    ) -> str:
        """Format a Server-Sent Event."""

        payload = serialize(data)

        return (
            f"event: {event}\n"
            f"data: {json.dumps(payload, default=str)}\n\n"
        )

    # ------------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:

        return {
            "status": "healthy",
            "default_model": config_client.setup.model,
            "default_provider": config_client.setup.provider,
            "default_base_url": config_client.setup.base_url,
            "default_web_mode": config_client.setup.web_mode,
            "default_reasoning_level": (
                config_client.setup.reasoning_level
            ),
            "streaming": True,
            "stream_endpoint": "/api/search/stream",
        }

    # ------------------------------------------------------------------------
    # Standard Search
    # ------------------------------------------------------------------------

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
            return await agent_completion(
                config=setup,
                message=req.query,
            )

        except HTTPException:
            raise

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Search execution failed: {str(exc)}"
                ),
            )

    # ------------------------------------------------------------------------
    # Streaming Search
    # ------------------------------------------------------------------------

    @app.post("/api/search/stream")
    async def api_search_stream(
        req: SearchRequest,
    ) -> StreamingResponse:
        """Run a Search Panda query with live execution events."""

        if not req.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        setup = resolve_request_setup(req)

        async def event_generator() -> AsyncGenerator[str, None]:

            try:

                # ------------------------------------------------------------
                # Connection established
                # ------------------------------------------------------------

                yield sse_event(
                    "connected",
                    {
                        "message": "Connected to Search Panda",
                        "query": req.query,
                    },
                )

                # ------------------------------------------------------------
                # Planning
                # ------------------------------------------------------------

                yield sse_event(
                    "status",
                    {
                        "stage": "planning",
                        "state": "running",
                        "message": "Analyzing query",
                    },
                )

                await asyncio.sleep(0)

                try:
                    query_plan = query_planner(
                        req.query,
                        web_mode=setup.web_mode,
                    )

                    yield sse_event(
                        "plan",
                        {
                            "stage": "planning",
                            "state": "completed",
                            "intent": query_plan.intent,
                            "needs_search": query_plan.needs_search,
                            "time_sensitive": (
                                query_plan.time_sensitive
                            ),
                            "sub_queries": (
                                query_plan.sub_queries
                            ),
                            "site_hints": (
                                query_plan.site_hints
                            ),
                        },
                    )

                except Exception as plan_error:

                    yield sse_event(
                        "status",
                        {
                            "stage": "planning",
                            "state": "completed",
                            "message": (
                                "Planning completed with "
                                "limited metadata"
                            ),
                            "warning": str(plan_error),
                        },
                    )

                # ------------------------------------------------------------
                # Search stage
                # ------------------------------------------------------------

                if setup.web_mode != "off":

                    yield sse_event(
                        "status",
                        {
                            "stage": "searching",
                            "state": "running",
                            "message": (
                                "Searching and collecting sources"
                            ),
                        },
                    )

                else:

                    yield sse_event(
                        "status",
                        {
                            "stage": "searching",
                            "state": "skipped",
                            "message": (
                                "Web search is disabled"
                            ),
                        },
                    )

                await asyncio.sleep(0)

                # ------------------------------------------------------------
                # Agent execution
                # ------------------------------------------------------------

                task = asyncio.create_task(
                    agent_completion(
                        config=setup,
                        message=req.query,
                    )
                )

                yielded_synthesizing = False

                while not task.done():

                    if not yielded_synthesizing:

                        yield sse_event(
                            "status",
                            {
                                "stage": "processing",
                                "state": "running",
                                "message": (
                                    "Search Panda is researching "
                                    "and analyzing evidence"
                                ),
                            },
                        )

                        yielded_synthesizing = True

                    # Keep SSE connection alive while the agent works.
                    yield ": keep-alive\n\n"

                    await asyncio.sleep(2)

                response = await task

                # ------------------------------------------------------------
                # Sources
                # ------------------------------------------------------------

                sources = getattr(
                    response,
                    "sources",
                    [],
                ) or []

                if sources:

                    yield sse_event(
                        "status",
                        {
                            "stage": "sources",
                            "state": "completed",
                            "message": (
                                f"Collected {len(sources)} sources"
                            ),
                            "count": len(sources),
                        },
                    )

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):

                        yield sse_event(
                            "source",
                            {
                                "index": index,
                                "source": serialize(source),
                            },
                        )

                        await asyncio.sleep(0)

                # ------------------------------------------------------------
                # Synthesis
                # ------------------------------------------------------------

                yield sse_event(
                    "status",
                    {
                        "stage": "synthesizing",
                        "state": "completed",
                        "message": "Answer generated",
                    },
                )

                # ------------------------------------------------------------
                # Final answer
                # ------------------------------------------------------------

                yield sse_event(
                    "answer",
                    serialize(response),
                )

                yield sse_event(
                    "done",
                    {
                        "success": True,
                        "message": (
                            "Search Panda execution completed"
                        ),
                    },
                )

            except asyncio.CancelledError:

                yield sse_event(
                    "done",
                    {
                        "success": False,
                        "message": (
                            "Search request cancelled"
                        ),
                    },
                )

                raise

            except Exception as exc:

                yield sse_event(
                    "error",
                    {
                        "message": str(exc),
                    },
                )

                yield sse_event(
                    "done",
                    {
                        "success": False,
                    },
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------------
    # Query Planner
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # OpenAI-compatible API for Open WebUI
    # ------------------------------------------------------------------------

    def public_model_id(model_id: str) -> str:
        """Namespace provider models for Open WebUI."""
        if not model_prefix:
            return model_id
        return model_id if model_id.startswith(model_prefix) else f"{model_prefix}{model_id}"

    def provider_model_id(model_id: str) -> str:
        """Convert an Open WebUI model id back to the provider's model id."""
        if model_prefix and model_id.startswith(model_prefix):
            return model_id[len(model_prefix):]
        return model_id

    model_cache: dict[str, Any] = {
        "expires_at": 0.0,
        "data": [],
    }

    @app.get("/v1/models")
    async def openai_models():
        """Expose every model advertised by the configured OpenAI-compatible backend."""

        now = time.time()
        if now < model_cache["expires_at"] and model_cache["data"]:
            return {"object": "list", "data": model_cache["data"]}

        setup = config_client.setup
        api_key = setup.api_key or (
            "ollama" if setup.provider == "ollama" else None
        )

        if not setup.base_url:
            raise HTTPException(
                status_code=500,
                detail="No OpenAI-compatible base URL is configured.",
            )

        try:
            client = AsyncOpenAI(
                api_key=api_key or "none",
                base_url=setup.base_url,
            )
            provider_models = await client.models.list()

            data = []
            for model in provider_models.data:
                model_id = getattr(model, "id", None)
                if not model_id:
                    continue

                data.append({
                    "id": public_model_id(model_id),
                    "object": "model",
                    "created": getattr(model, "created", int(now)),
                    "owned_by": "search-panda",
                })

            # Some compatible servers implement chat but not /models.
            if not data:
                data.append({
                    "id": public_model_id(setup.model),
                    "object": "model",
                    "created": int(now),
                    "owned_by": "search-panda",
                })

            model_cache["data"] = data
            model_cache["expires_at"] = now + 30

            return {"object": "list", "data": data}

        except Exception as exc:
            # Keep Open WebUI usable even if model discovery is unavailable.
            fallback = {
                "id": public_model_id(setup.model),
                "object": "model",
                "created": int(now),
                "owned_by": "search-panda",
            }
            model_cache["data"] = [fallback]
            model_cache["expires_at"] = now + 10

            return {
                "object": "list",
                "data": [fallback],
            }

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request):
        body = await request.json()

        messages = body.get("messages", [])
        requested_model = (
            body.get("model") or config_client.setup.model
        ).strip()
        provider_model = provider_model_id(requested_model)

        user_message = next(
            (
                msg.get("content", "")
                for msg in reversed(messages)
                if msg.get("role") == "user"
            ),
            "",
        )

        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No user message provided.",
            )

        # Per-request config: selecting a model in Open WebUI only affects
        # this request and does not mutate the global/default configuration.
        try:
            setup = config_client.setup.model_copy(deep=True)
        except AttributeError:
            # Fallback for non-Pydantic Setup implementations.
            import copy
            setup = copy.deepcopy(config_client.setup)

        setup.model = provider_model

        response = await agent_completion(
            config=setup,
            message=user_message,
        )

        content = getattr(response, "answer", None)
        if content is None:
            content = str(response)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

    return app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_api_app(),
        host="127.0.0.1",
        port=8387
    )