import asyncio

import search_panda.agent.agents as agent_module
from search_panda.config import ConfigManager, Setup
from search_panda.agent.search import needs_web_search
from search_panda.run import build_parser


def test_setup_defaults_to_ollama_localhost():
    setup = Setup(model="llama3.1:8b")

    assert setup.provider == "ollama"
    assert setup.base_url == "http://localhost:11434/v1"
    assert setup.model == "llama3.1:8b"


def test_needs_web_search_for_recent_or_unknown_facts():
    assert needs_web_search("What happened in the AI world today?") is True
    assert needs_web_search("Summarize the latest OpenAI release notes") is True
    assert needs_web_search("What is 12 * 7?") is False
    assert needs_web_search("Explain the concept of recursion in Python") is False


def test_config_manager_round_trip(tmp_path):
    manager = ConfigManager(config_dir=tmp_path / ".search-panda")
    original = Setup(model="qwen2.5:7b", provider="ollama")

    manager.save(original)
    loaded = manager.load()

    assert loaded.model == original.model
    assert loaded.provider == original.provider
    assert loaded.base_url == "http://localhost:11434/v1"


def test_build_parser_supports_shell_commands():
    parser = build_parser()

    set_args = parser.parse_args(["set", "qwen2.5:7b"])
    assert set_args.command == "set"
    assert set_args.model_name == "qwen2.5:7b"

    run_args = parser.parse_args(["--model", "llama3.1:8b"])
    assert run_args.model == "llama3.1:8b"


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

    result = asyncio.run(agent_module.agent_completion(Setup(model="llama3.1:8b"), "who won fifa 2026?"))

    assert result == "ok"
    assert captured["prompt"] == "who won fifa 2026?"
    assert isinstance(captured["prompt"], str)
