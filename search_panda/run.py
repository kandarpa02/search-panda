from .cli_tools import default, model
from .agent.agents import agent_completion
from .config import ConfigManager, CONFIG_DIR, CONFIG_FILE, REASONING_MODELS
from .helpers import UnknownCommand, CLIError
from .ollama.run import api, LlamaAPI
from .ollama.models import CACHE_DIR, ALLOWED_MODELS
import uvicorn
import os
from pathlib import Path
import asyncio

from rich.console import Console
from rich.markdown import Markdown

config = ConfigManager()
console = Console()

async def chat(config, message):
    response = await agent_completion(
        config=config,
        message=message,
    )

    console.print(
        Markdown(response)
    )

async def main():
    global default

    console.print(
        "[bold cyan]Welcome to Search Panda 🐼[/bold cyan]"
    )

    if not config.exists():

        console.print("Setup the engine...")

        model()

        config.save(default)

        console.print("Environment is ready.")
        console.print("Enjoy chatting! 🐼🎍")

    else:
        default = config.load()

    # Load the local LLM + start API in background
    # llama_api = LlamaAPI(default.model)

    # server_task = asyncio.create_task(
    #     llama_api.serve()
    # )
    llama_api = LlamaAPI(default.model)

    server_task = asyncio.create_task(
        llama_api.serve()
    )

    while True:
        try:

            question = input("> ").strip()

            if question.lower() in {"exit", "quit"}:
                console.print("Goodbye!")
                break

            if question.startswith("panda set:"):
                next_part = question[len("panda set:"):-1]

                if next_part.startswith("model="):
                    default.model = next_part[len("model="):]
                    print(f"model is set to {default.model}")
                    break
                else:
                    raise CLIError(
                        "command does not exist. '--help' command will be added soon"
                    )

            if question.lower() == "remove":

                console.print(
                    "Removed current config. Run Search Panda again to setup."
                )

                os.remove(CONFIG_FILE)

                try:
                    os.rmdir(CONFIG_DIR)
                except OSError:
                    pass

                break

            if not question:
                continue

            await chat(
                default,
                question
            )

            console.print()

        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break

        except Exception as e:
            console.print(
                f"[red]{e}[/red]"
            )

    # Stop Uvicorn when CLI exits
    server_task.cancel()

    try:
        await server_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())