from .config import BASE_URL, REASONING_MODELS
from openai import OpenAI
from .helpers import APIError, URLError

self_knowledge = (
"You are Search Panda, an open-source AI CLI search assistant. "
"You combine web search with LLM reasoning to provide fast, accurate answers. "
"If asked about yourself, identify as Search Panda and briefly describe your purpose."
)


def chat_completion(config, messages, tools=None, stream=False):

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

    return client.chat.completions.create(**kwargs)