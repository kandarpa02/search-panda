import json

from ..core import chat_completion
from ..tools import TOOLS
from .search import search, read_result


def agent_completion(config, prompt):

    use_web = "--web" in prompt

    if use_web:
        prompt = prompt.replace("--web", "").strip()

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    tools = TOOLS if use_web else None
    tool_choice = "required" if use_web else None

    while True:

        response = chat_completion(
            config=config,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        messages.append(message)

        for tool_call in message.tool_calls:

            name = tool_call.function.name

            try:
                args = json.loads(
                    tool_call.function.arguments
                )
            except Exception:
                args = {}

            if name == "search":

                result = search(
                    args.get("query", "")
                )

            elif name == "read_result":

                result = read_result(
                    args.get("index", -1)
                )

            else:

                result = {
                    "error": f"Unknown tool '{name}'."
                }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

            # After the first tool call, allow the model
            # to decide whether another tool call is needed.
            tool_choice = "auto"
