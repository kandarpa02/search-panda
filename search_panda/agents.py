import json

from .core import chat_completion
from .tools import TOOLS
from .search import search, read_result


def agent_completion(config, prompt):

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    while True:

        response = chat_completion(
            config=config,
            messages=messages,
            tools=TOOLS,
            stream=False
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content

        messages.append(message)

        for tool_call in message.tool_calls:

            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name == "search":

                result = search(
                    args["query"]
                )

            elif name == "read_result":

                result = read_result(
                    args["index"]
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