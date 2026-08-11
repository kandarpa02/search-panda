import json
from collections.abc import Sequence
from ..core import chat_completion
from ..tools import TOOLS
from .search import search, read_result
from ..helpers import APIError, URLError

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    WebSearchTool,
    set_tracing_disabled
)

from ..config import Setup, BASE_URL

self_knowledge = """
You are Search Panda, an open-source AI search assistant.

You have access ONLY to the tools provided in the request.

Rules:
- Never invent tool names.
- Never output tool calls as plain text.
- Use only the provided tools.
- If a question can be answered without tools, answer directly.
- If no suitable tool exists, answer normally.
- If asked about yourself, identify as Search Panda.
"""

# async def agent_completion(config:Setup, message:Sequence[str]):
#     MODEL = config.model
#     API_KEY = config.api_key
#     PROVIDER = config.provider
#     URL = BASE_URL.get(config.provider, None)

#     if URL is None:
#         raise URLError(f"'provider'={PROVIDER} has no valid url")
    
#     if API_KEY is None:
#         raise APIError("'api_key' can not be None")


#     client = AsyncOpenAI(
#         api_key=API_KEY,
#         base_url=URL,
#     )

#     model = OpenAIChatCompletionsModel(
#         model=MODEL,
#         openai_client=client,
#     )

#     agent = Agent(
#         name="SearchPanda",
#         instructions=self_knowledge,
#         model=model,
#         tools=[search, read_result],
#     )

#     set_tracing_disabled(True)

#     result = await Runner.run(
#         agent,
#         message
#     )

#     return result.final_output


async def agent_completion(config:Setup, message:Sequence[str]):
    MODEL = str(config.model)
    URL = 'http://localhost:1890/v1'

    client = AsyncOpenAI(
        api_key="charliekirk",
        base_url=URL,
    )

    model = OpenAIChatCompletionsModel(
        model=MODEL,
        openai_client=client,
    )

    agent = Agent(
        name="SearchPanda",
        instructions=self_knowledge,
        model=model,
        tools=[search, read_result],
    )

    set_tracing_disabled(True)

    result = await Runner.run(
        agent,
        message
    )

    return result.final_output