"""Tests for multi-provider support, reasoning level defaults, and public Python API."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from search_panda import (
    SearchPanda,
    SearchResponse,
    Source,
    Setup,
    ConfigManager,
    ask,
    is_reasoning_model,
    create_api_app,
)
from search_panda.config import BASE_URL
import search_panda.agent.agents as agent_module


def test_is_reasoning_model():
    assert is_reasoning_model("deepseek-reasoner") is True
    assert is_reasoning_model("deepseek-r1") is True
    assert is_reasoning_model("o1") is True
    assert is_reasoning_model("o1-mini") is True
    assert is_reasoning_model("o3-mini") is True
    assert is_reasoning_model("gpt-5") is True
    assert is_reasoning_model("qwq-32b") is True
    assert is_reasoning_model("llama3.2:1b") is False
    assert is_reasoning_model("gpt-4o") is False


def test_setup_auto_defaults_reasoning_to_medium():
    setup = Setup(model="deepseek-reasoner", provider="deepseek").resolve()
    assert setup.reasoning_level == "medium"
    assert setup.base_url == BASE_URL["deepseek"]

    setup_non_reasoning = Setup(model="llama3.1:8b", provider="ollama").resolve()
    assert setup_non_reasoning.reasoning_level is None


def test_setup_resolves_provider_and_env_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-12345")
    setup = Setup(model="llama-3.3-70b", provider="groq").resolve()
    assert setup.base_url == BASE_URL["groq"]
    assert setup.api_key == "test-groq-key-12345"


def test_setup_preserves_explicit_key_and_url():
    setup = Setup(
        model="custom-model",
        provider="custom",
        api_key="my-secret-key",
        base_url="https://custom.endpoint/v1",
        reasoning_level="high",
    ).resolve()
    assert setup.api_key == "my-secret-key"
    assert setup.base_url == "https://custom.endpoint/v1"
    assert setup.reasoning_level == "high"


def test_search_panda_client_initialization():
    client = SearchPanda(model="llama3.2:1b", provider="ollama", web_mode="on")
    assert client.setup.model == "llama3.2:1b"
    assert client.setup.provider == "ollama"
    assert client.setup.web_mode == "on"

    client.set_model("qwen2.5:7b").set_web_mode("auto")
    assert client.setup.model == "qwen2.5:7b"
    assert client.setup.web_mode == "auto"


def test_search_response_model():
    resp = SearchResponse(
        query="test query",
        answer="This is the answer.",
        sources=[
            Source(title="Wikipedia", url="https://en.wikipedia.org", domain="wikipedia.org", snippet="An encyclopedia")
        ],
        intent="factual",
        sub_queries=["test query"],
        time_sensitive=False,
        model="llama3.2:1b",
        provider="ollama",
    )
    assert resp.query == "test query"
    assert resp.answer == "This is the answer."
    assert str(resp) == "This is the answer."
    assert len(resp.sources) == 1
    assert resp.sources[0].domain == "wikipedia.org"


def test_search_panda_ask_sync_and_async(monkeypatch):
    class DummyClient:
        pass

    class DummyModel:
        def __init__(self, model, openai_client):
            self.model = model
            self.openai_client = openai_client

    class DummyAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    async def fake_run(agent, prompt):
        return type("Result", (), {"final_output": "AI search completed successfully."})()

    monkeypatch.setattr(agent_module, "AsyncOpenAI", lambda **kwargs: DummyClient())
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", DummyModel)
    monkeypatch.setattr(agent_module, "Agent", DummyAgent)
    monkeypatch.setattr(agent_module.Runner, "run", fake_run)

    client = SearchPanda(model="llama3.2:1b", web_mode="off")
    resp = client.ask("what is recursion?")

    assert isinstance(resp, SearchResponse)
    assert resp.answer == "AI search completed successfully."
    assert resp.query == "what is recursion?"


def test_create_fastapi_app():
    app = create_api_app()
    assert app.title == "Search Panda API"

    # Test route registrations
    routes = [route.path for route in app.routes]
    assert "/api/health" in routes
    assert "/api/search" in routes
    assert "/api/plan" in routes

