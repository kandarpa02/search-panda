from collections.abc import Sequence

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)

from ..config import Setup, DEFAULT_OLLAMA_BASE_URL
from .search import search, read_result


SEARCH_PANDA_INSTRUCTIONS = """You are Search Panda, an autonomous AI search agent.

Your job is to answer the user's question accurately and concisely.

You have access to a web search tool. Use it when the question requires
current, recent, changing, or externally verifiable information.

USE SEARCH when the question asks about:
- Recent events, people, outcomes, or developments
- Current prices, weather, news, markets, or other changing information
- Specific dates or time references such as 2024, 2025, 2026, today, this week, etc.
- Verification of claims or events
- Current status of people, organizations, products, or events
- Information where your built-in knowledge may be outdated

DO NOT SEARCH when:
- The question asks for general timeless knowledge
- The question asks for an explanation of an established concept
- The question is philosophical or opinion-based
- The answer does not require current external information

When you search:
1. Construct a focused search query.
2. Search the web.
3. Inspect the available results.
4. Retrieve the most relevant result when useful.
5. Answer the user's original question using the retrieved information.
6. Do not mention internal tool mechanics unless relevant.

When you do not search, answer directly.

Never fabricate search results or claim that you searched when you did not.
"""


@function_tool
def web_search(query: str) -> str:
    """Search the web for current or externally verifiable information.

    Args:
        query: A focused search query.
    """
    try:
        results = search(query, max_results=3)
    except Exception as exc:
        return f"Search failed: {exc}"

    if not results:
        return "No search results were found."

    if all("error" in result for result in results):
        errors = [
            str(result.get("error"))
            for result in results
            if result.get("error")
        ]
        return f"Search failed: {'; '.join(errors)}"

    output: list[str] = []

    for index, result in enumerate(results):
        if "error" in result:
            continue

        output.append(
            f"RESULT {index + 1}\n"
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"Snippet: {result.get('snippet', '')}"
        )

    return "\n\n".join(output) or "No usable search results were found."


@function_tool
def read_search_result(result_index: int) -> str:
    """Read the full content of a previously returned search result.

    Args:
        result_index: Zero-based index of the search result.
    """
    try:
        result = read_result(result_index)
    except Exception as exc:
        return f"Unable to read search result: {exc}"

    if "error" in result:
        return f"Unable to read search result: {result['error']}"

    content = result.get("content", "")

    if not content:
        return "The search result contained no readable content."

    return content[:8000]


def _build_agent(config: Setup) -> Agent:
    """Build a Search Panda agent using the configured model."""

    model_name = str(config.model or "llama3.1:8b")
    base_url = str(
        getattr(config, "base_url", None)
        or DEFAULT_OLLAMA_BASE_URL
    ).rstrip("/")

    # Ollama exposes an OpenAI-compatible Chat Completions endpoint.
    client = AsyncOpenAI(
        api_key="ollama",
        base_url=base_url,
    )

    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )

    return Agent(
        name="Search Panda",
        instructions=SEARCH_PANDA_INSTRUCTIONS,
        model=model,
        tools=[
            web_search,
            read_search_result,
        ],
    )


async def agent_completion(
    config: Setup,
    message: str | Sequence[str],
) -> str:
    """Run Search Panda and return its final answer."""

    if isinstance(message, (list, tuple)):
        prompt = " ".join(str(part) for part in message)
    else:
        prompt = str(message)

    # Ollama is not an OpenAI-hosted model, so there is no reason to
    # send OpenAI tracing data unless the application explicitly wants it.
    set_tracing_disabled(True)

    agent = _build_agent(config)

    result = await Runner.run(
        agent,
        prompt,
    )

    return result.final_output