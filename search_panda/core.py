from .config import BASE_URL, REASONING_MODELS, DEFAULT_OLLAMA_BASE_URL
from openai import OpenAI
from .helpers import APIError, URLError
import json

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
- For current, recent, or external factual questions, decide whether web search is necessary before answering.
- If the query requires fresh information, use the web search tools; otherwise answer from the model's general knowledge.
"""


def chat_completion(
    config,
    messages,
    tools=None,
    tool_choice=None,
    stream=False
):
    model_name = config.model or "llama3.1:8b"
    api_key = config.api_key or "ollama"
    provider = (config.provider or "ollama").lower()
    base_url = getattr(config, "base_url", None) or BASE_URL.get(provider, DEFAULT_OLLAMA_BASE_URL)

    if base_url is None:
        raise URLError(f"'provider'={provider} has no valid url")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    kwargs = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": self_knowledge
            },
            *messages
        ],
        "stream": stream
    }

    if tools:
        kwargs["tools"] = tools

    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    if model_name in REASONING_MODELS:
        effort = config.reasoning_level

        if effort is None:
            kwargs["reasoning_effort"] = REASONING_MODELS[model_name][0]

        elif effort not in REASONING_MODELS[model_name]:
            print(
                f"reasoning_level='{effort}' "
                f"does not exist for model='{model_name}'"
            )

        else:
            kwargs["reasoning_effort"] = effort

    # print(json.dumps(kwargs, indent=2, default=str))
    return client.chat.completions.create(**kwargs)
