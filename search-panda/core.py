from .config import BASE_URL, REASONING_MODELS
from openai import OpenAI
from .helpers import APIError, URLError

# config = Setup(
#     None,
#     "llama-3.1-8b-instant",
#     "groq",
# )
self_knowledge = "You are Search Panda, an open-source AI CLI search assistant. " \
"You combine web search with LLM reasoning to provide fast, accurate answers. " \
"If asked about yourself, identify as Search Panda and briefly describe your purpose."

def chat_completion(config, context, prompt):
    global self_knowledge
    model_name = config.model
    api_key = config.api_key
    provider = config.provider
    base_url = BASE_URL.get(provider, None)

    if base_url is None:
        raise URLError(f"'provider'={provider} has no valid url")
    
    if api_key is None:
        raise APIError(f"'api_key' can not be 'None")

    client = OpenAI(
        api_key=config.api_key,
        base_url=base_url)

    kwargs = {
    "model": model_name,
    "messages": [
        {"role": "system", "content": self_knowledge},
        {"role": "system", "content": f"Search context:\n{context}"},
        {"role": "user", "content": prompt},
        ],
    "max_completion_tokens": 2048,
    "stream":True
    }

    if model_name in REASONING_MODELS:
        effort = config.reasoning_level
        if effort is None: 
            kwargs["reasoning_effort"] = REASONING_MODELS['model_name'][0]

        elif effort not in REASONING_MODELS['model_name']:
            print(f"reasoning_level='{effort}', does not exist with current provider='{config.provider}'")

        else:
            kwargs["reasoning_effort"] = effort

    
    response = client.chat.completions.create(**kwargs)
    return response

