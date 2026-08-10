from .config import BASE_URL, REASONING_MODELS
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
"""
def chat_completion(
    config,
    messages,
    tools=None,
    tool_choice=None,
    stream=False
):
    model_name = config.model
    api_key = config.api_key
    provider = config.provider
    base_url = BASE_URL.get(provider, None)

    if base_url is None:
        raise URLError(f"'provider'={provider} has no valid url")

    if api_key is None:
        raise APIError("'api_key' can not be None")

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
