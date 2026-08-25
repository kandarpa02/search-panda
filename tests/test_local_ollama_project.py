"""Tests for Search Panda.

Covers:
- Config defaults and round-trip persistence
- Query planner: intent classification, needs_web_search (modes: on, off, auto), rewrite_query, clean_query_text
- Page ranker: BM25 scoring, domain authority, deduplication
- CLI argument parser & slash commands
- Agent completion (monkeypatched runner)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import search_panda.agent.agents as agent_module
from search_panda.config import ConfigManager, Setup
from search_panda.agent.planner import (
    classify_intent,
    needs_web_search,
    rewrite_query,
    clean_query_text,
    plan,
    is_time_sensitive,
)
from search_panda.agent.page_ranker import rank_results, _domain_authority
from search_panda.agent.data_representation import SearchResult, QueryPlan
from search_panda.run import build_parser, handle_slash_command


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_setup_defaults_to_ollama_localhost():
    setup = Setup(model="llama3.1:8b")
    assert setup.provider == "ollama"
    assert setup.base_url == "http://localhost:11434/v1"
    assert setup.model == "llama3.1:8b"
    assert setup.web_mode == "on"


def test_config_manager_round_trip(tmp_path: Path):
    manager = ConfigManager(config_dir=tmp_path / ".search-panda")
    original = Setup(model="qwen2.5:7b", provider="ollama", web_mode="auto")
    manager.save(original)
    loaded = manager.load()
    assert loaded.model == original.model
    assert loaded.provider == original.provider
    assert loaded.base_url == "http://localhost:11434/v1"
    assert loaded.web_mode == "auto"


# ---------------------------------------------------------------------------
# Planner — needs_web_search & web modes
# ---------------------------------------------------------------------------

def test_needs_web_search_for_recent_or_unknown_facts():
    assert needs_web_search("What happened in the AI world today?") is True
    assert needs_web_search("Summarize the latest OpenAI release notes") is True
    assert needs_web_search("tell me who won fifa 2026?") is True


def test_needs_web_search_modes():
    # web_mode="off" always returns False
    assert needs_web_search("Who won FIFA 2026?", web_mode="off") is False

    # web_mode="on" searches real questions
    assert needs_web_search("Explain recursion in Python", web_mode="on") is True

    # web_mode="auto" skips known programming concepts
    assert needs_web_search("Explain recursion", web_mode="auto") is False

    # pure greetings never search in any mode
    assert needs_web_search("hello", web_mode="on") is False
    assert needs_web_search("what is 12 * 7?", web_mode="on") is False


def test_clean_query_text():
    assert clean_query_text("search and tell me who won fifa 2026?") == "who won fifa 2026?"
    assert clean_query_text("google latest tech news") == "latest tech news"
    assert clean_query_text("can you tell me about quantum computing") == "quantum computing"


# ---------------------------------------------------------------------------
# Planner — classify_intent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query, expected_intent", [
    ("Write a Python function to reverse a list", "code"),
    ("How do I install numpy?", "code"),
    ("What is 42 + 58?", "math"),
    ("Calculate the derivative of x^2", "math"),
    ("What happened in the Ukraine war today?", "news"),
    ("Who won the 2026 election?", "news"),
    ("Who won fifa 2026?", "news"),
    ("What is the capital of France?", "factual"),
    ("Should I use React or Vue?", "opinion"),
])
def test_classify_intent(query: str, expected_intent: str):
    assert classify_intent(query) == expected_intent


# ---------------------------------------------------------------------------
# Planner — rewrite_query
# ---------------------------------------------------------------------------

def test_rewrite_query_news_adds_year_if_missing():
    import datetime

    year = str(datetime.date.today().year)
    result = rewrite_query("OpenAI latest news", "news", time_sensitive=True)
    assert year in result


def test_rewrite_query_preserves_explicit_future_year():
    result = rewrite_query("who won fifa 2026?", "news", time_sensitive=True)
    assert "2026" in result


def test_rewrite_query_code_adds_how_to():
    result = rewrite_query("reverse a list in Python", "code", time_sensitive=False)
    assert result.lower().startswith("how to")


def test_rewrite_query_factual_unchanged():
    query = "capital of France"
    result = rewrite_query(query, "factual", time_sensitive=False)
    assert query in result


# ---------------------------------------------------------------------------
# Planner — time sensitivity
# ---------------------------------------------------------------------------

def test_is_time_sensitive_detects_sports_and_events():
    assert is_time_sensitive("What is happening today?") is True
    assert is_time_sensitive("Latest AI news 2025") is True
    assert is_time_sensitive("who won fifa 2026?") is True
    assert is_time_sensitive("super bowl winner") is True
    assert is_time_sensitive("Explain recursion") is False


# ---------------------------------------------------------------------------
# Planner — plan() end-to-end
# ---------------------------------------------------------------------------

def test_plan_math_no_search():
    p = plan("What is 100 * 3?")
    assert p.needs_search is False
    assert p.intent == "math"


def test_plan_news_needs_search():
    p = plan("What happened in the AI world today?")
    assert p.needs_search is True
    assert p.time_sensitive is True


def test_plan_code_has_site_hints():
    p = plan("How do I use asyncio in Python?")
    assert p.intent == "code"
    assert len(p.site_hints) > 0
    assert any("stackoverflow" in h for h in p.site_hints)


# ---------------------------------------------------------------------------
# Page ranker
# ---------------------------------------------------------------------------

def _make_result(index: int, title: str, url: str, snippet: str) -> SearchResult:
    return SearchResult(index=index, title=title, url=url, snippet=snippet)


def test_rank_results_orders_by_relevance():
    results = [
        _make_result(0, "Random blog post", "https://randomblog.com/post1", "Some random content"),
        _make_result(1, "Python asyncio tutorial", "https://realpython.com/asyncio-python", "asyncio is a Python library for concurrent programming"),
        _make_result(2, "Wikipedia: Asyncio", "https://en.wikipedia.org/wiki/Asyncio", "asyncio is the Python asynchronous I/O framework"),
    ]
    ranked = rank_results("Python asyncio", results, top_k=3, deduplicate=False)
    assert ranked[0].domain in ("realpython.com", "wikipedia.org", "en.wikipedia.org")
    for r in ranked:
        assert r.score >= 0.0


def test_rank_results_deduplicates_by_domain():
    results = [
        _make_result(0, "SO question 1", "https://stackoverflow.com/q/1", "asyncio event loop"),
        _make_result(1, "SO question 2", "https://stackoverflow.com/q/2", "asyncio gather usage"),
        _make_result(2, "Python docs", "https://docs.python.org/asyncio", "asyncio official docs"),
    ]
    ranked = rank_results("asyncio", results, top_k=5, deduplicate=True)
    domains = [r.domain for r in ranked]
    assert domains.count("stackoverflow.com") <= 1


def test_domain_authority_known_domains():
    assert _domain_authority("wikipedia.org") > 0.8
    assert _domain_authority("stackoverflow.com") > 0.8
    assert _domain_authority("stackoverflow.com") > _domain_authority("unknownblog.xyz")


def test_domain_authority_subdomain_match():
    assert _domain_authority("en.wikipedia.org") > 0.8


# ---------------------------------------------------------------------------
# SearchResult model
# ---------------------------------------------------------------------------

def test_search_result_derives_domain():
    r = SearchResult(index=0, title="Test", url="https://www.example.com/page", snippet="x")
    assert r.domain == "example.com"


# ---------------------------------------------------------------------------
# CLI parser & Slash commands
# ---------------------------------------------------------------------------

def test_build_parser_supports_shell_commands():
    parser = build_parser()

    set_args = parser.parse_args(["set", "qwen2.5:7b"])
    assert set_args.command == "set"
    assert set_args.model_name == "qwen2.5:7b"

    run_args = parser.parse_args(["--model", "llama3.1:8b", "--web", "off"])
    assert run_args.model == "llama3.1:8b"
    assert run_args.web_mode == "off"


def test_build_parser_verbose_flag():
    parser = build_parser()
    args = parser.parse_args(["--verbose"])
    assert args.verbose is True


def test_slash_command_web_toggle(tmp_path):
    setup = Setup(model="llama3.1:8b", web_mode="on")
    setup, verbose, should_continue = handle_slash_command("/web off", setup, False)
    assert setup.web_mode == "off"
    assert should_continue is True


def test_slash_command_model_switch():
    setup = Setup(model="llama3.1:8b")
    setup, verbose, should_continue = handle_slash_command("/model llama3.2:1b", setup, False)
    assert setup.model == "llama3.2:1b"
    assert should_continue is True


def test_slash_command_exit():
    setup = Setup(model="llama3.1:8b")
    setup, verbose, should_continue = handle_slash_command("/exit", setup, False)
    assert should_continue is False


# ---------------------------------------------------------------------------
# Agent completion (monkeypatched)
# ---------------------------------------------------------------------------

def test_agent_completion_passes_scalar_prompt_to_runner(monkeypatch):
    captured = {}

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
        captured["prompt"] = prompt
        return type("Result", (), {"final_output": "ok"})()

    monkeypatch.setattr(agent_module, "AsyncOpenAI", lambda **kwargs: DummyClient())
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", DummyModel)
    monkeypatch.setattr(agent_module, "Agent", DummyAgent)
    monkeypatch.setattr(agent_module.Runner, "run", fake_run)

    result = asyncio.run(
        agent_module.agent_completion(Setup(model="llama3.1:8b"), "who won fifa 2026?")
    )

    assert result == "ok"
    assert captured["prompt"] == "who won fifa 2026?"
    assert isinstance(captured["prompt"], str)
