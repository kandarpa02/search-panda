"""Search Panda agent — wires the planner, search, and LLM together.

Optimized for local models (including 1B/3B/8B/70B):
- plan_and_search: fetches search results AND automatically reads top page
  contents in parallel, delivering complete grounded evidence immediately.
- read_page: tool for targeted manual page extraction if needed.
- refine_search: tool for follow-up search angles.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from loguru import logger
from rich.console import Console

from ..config import Setup, DEFAULT_OLLAMA_BASE_URL
from .planner import plan
from .search import (
    search_parallel,
    fetch_and_extract,
    fetch_pages_parallel,
    search as search_single,
)

# Suppress verbose default logger output
logger.remove()

console = Console()

# Track last retrieved sources for clean UI display
LAST_SOURCES: list[dict[str, str]] = []


# ---------------------------------------------------------------------------
# Agent system prompt
# ---------------------------------------------------------------------------

SEARCH_PANDA_INSTRUCTIONS = """You are Search Panda, a fast, rigorous, intellectually independent AI search assistant.

1. Use `plan_and_search` for current or external information and ground factual claims in retrieved evidence.
2. Never invent facts, sources, quotes, or URLs.
3. Analyze issues materially and structurally, considering power, institutions, class, labor, ownership, incentives, inequality, and historical context where relevant.
4. Be evidence-driven and unapologetically direct. Do not manufacture "both sides" when evidence clearly favors one conclusion.
5. Apply the same critical standard to governments, corporations, parties, ideologies, institutions, and individuals.
6. Clearly separate facts, evidence, interpretation, assumptions, and opinion.
7. Reject propaganda, loaded framing, false equivalence, and political spin regardless of source.
8. Speak naturally and casually when appropriate. Humor, sarcasm, blunt language, profanity, and irreverent phrasing are allowed when they genuinely fit the context.
9. Do not use identity-based slurs or degrading language toward protected groups. Do not use offensive language merely for shock value.
10. Prioritize truth and accuracy over politeness, consensus, political sensitivity, or ideological convenience.
11. State evidence-supported conclusions plainly and acknowledge genuine uncertainty.
12. Answer directly, clearly, concisely, and without unnecessary disclaimers."""

# ---------------------------------------------------------------------------
# Tool: plan_and_search
# ---------------------------------------------------------------------------

@function_tool
async def plan_and_search(query: str) -> str:
    """Search the web for information relevant to the user query.

    Automatically plans the search, retrieves relevant pages, and extracts
    their content to provide comprehensive evidence.

    Args:
        query: The user's question or search topic.
    """
    global LAST_SOURCES
    LAST_SOURCES = []

    query_plan = plan(query, web_mode="on")
    sub_queries = query_plan.sub_queries or [query]

    # Light, clean visual indicators
    console.print(f"[dim]• Planning: [cyan]{query_plan.intent}[/cyan] ({'time-sensitive' if query_plan.time_sensitive else 'general'})[/dim]")
    for sq in sub_queries:
        console.print(f"[dim]• Searching: [cyan]{sq}[/cyan][/dim]")

    results = await search_parallel(sub_queries, max_results_each=5)

    if not results:
        return "No search results found on the web for this query. Answer based on available context."

    # Store for clean UI presentation
    for r in results[:5]:
        LAST_SOURCES.append({
            "title": r.title,
            "url": r.url,
            "domain": r.domain or "",
            "snippet": r.snippet,
        })

    # Automatically fetch top 2 URLs in parallel for instant deep context
    top_urls = [r.url for r in results if r.url][:2]
    if top_urls:
        console.print(f"[dim]• Reading {len(top_urls)} top sources...[/dim]")

    extracted_pages = await fetch_pages_parallel(top_urls, max_pages=2)
    page_map = {p.url: p for p in extracted_pages}

    output_blocks: list[str] = [
        f"WEB SEARCH EVIDENCE (Intent: {query_plan.intent}):\n"
    ]

    for i, r in enumerate(results[:5]):
        output_blocks.append(f"[Source {i + 1}] {r.title}")
        output_blocks.append(f"URL: {r.url}")
        output_blocks.append(f"Snippet: {r.snippet}")

        if r.url in page_map:
            page = page_map[r.url]
            excerpt = page.content[:1500].replace("\n\n", "\n")
            output_blocks.append(f"Key Excerpt from page:\n{excerpt}\n")
        else:
            output_blocks.append("")

    output_blocks.append(
        "INSTRUCTIONS FOR FINAL ANSWER:\n"
        "Synthesize a clear, direct answer to the user's question using the evidence above. "
        "If the event is in the future or not yet decided, state that clearly."
    )

    return "\n".join(output_blocks)


# ---------------------------------------------------------------------------
# Tool: read_page
# ---------------------------------------------------------------------------

@function_tool
def read_page(url: str) -> str:
    """Fetch and read the full text content of a specific web page.

    Args:
        url: Full URL of the page to fetch and read.
    """
    console.print(f"[dim]• Fetching page: [cyan]{url}[/cyan][/dim]")
    page = fetch_and_extract(url)

    if page is None:
        return f"Could not fetch or extract content from: {url}"

    content = page.content
    if len(content) > 8000:
        cutoff = content.rfind(". ", 0, 8000)
        content = content[: cutoff + 1] if cutoff > 0 else content[:8000]
        content += "\n\n[Content truncated]"

    return f"PAGE: {page.title or url}\nURL: {url}\n\n{content}"


# ---------------------------------------------------------------------------
# Tool: refine_search
# ---------------------------------------------------------------------------

@function_tool
def refine_search(query: str, context: str) -> str:
    """Run a focused follow-up search when initial results need refinement.

    Args:
        query: A refined search query.
        context: Context of what is still missing.
    """
    console.print(f"[dim]• Refining search: [cyan]{query}[/cyan][/dim]")
    results = search_single(query, max_results=5)

    if not results:
        return "Refined search returned no results."

    lines: list[str] = [
        f"Refined search results for: {query!r}\n",
    ]
    for i, r in enumerate(results):
        lines.append(
            f"[{i + 1}] {r.title}\n"
            f"URL: {r.url}\n"
            f"Snippet: {r.snippet[:300]}\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(config: Setup) -> Agent:
    """Construct the Search Panda agent for the given config."""
    model_name = str(config.model or "llama3.1:8b")
    base_url = str(
        getattr(config, "base_url", None) or DEFAULT_OLLAMA_BASE_URL
    ).rstrip("/")

    client = AsyncOpenAI(
        api_key=str(config.api_key or "ollama"),
        base_url=base_url,
    )

    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )

    tools = [plan_and_search, read_page, refine_search]

    return Agent(
        name="Search Panda",
        instructions=SEARCH_PANDA_INSTRUCTIONS,
        model=model,
        tools=tools,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def agent_completion(
    config: Setup,
    message: str | Sequence[str],
) -> str:
    """Run Search Panda and return its final answer."""
    global LAST_SOURCES
    LAST_SOURCES = []

    if isinstance(message, (list, tuple)):
        prompt = " ".join(str(part) for part in message)
    else:
        prompt = str(message)

    set_tracing_disabled(True)

    web_mode = getattr(config, "web_mode", "on")

    # If web_mode is explicitly 'off', run without tools for instant speed
    if web_mode == "off":
        agent = _build_agent(config)
        agent.tools = []
        result = await Runner.run(agent, prompt)
        return result.final_output

    agent = _build_agent(config)
    result = await Runner.run(agent, prompt)
    return result.final_output