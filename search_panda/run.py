from .cli_tools import default, api_key, model, provider
from .agent.agents import agent_completion
from .config import ConfigManager, CONFIG_DIR, CONFIG_FILE

import os

from rich.console import Console
from rich.markdown import Markdown


config = ConfigManager()
console = Console()


def temp(config, message):

    response = agent_completion(
        config=config,
        prompt=message,
    )

    console.print(
        Markdown(response)
    )

def main():

    global default

    console.print(
        "[bold cyan]Welcome to Search Panda 🐼[/bold cyan]"
    )

    if not config.exists():

        console.print("Setup the engine...")

        api_key()
        model()
        provider()

        config.save(default)

        console.print("Environment is ready.")
        console.print("Enjoy chatting! 🐼🎍")

    else:

        default = config.load()

    while True:

        try:

            question = input("> ").strip()

            if question.lower() in {"exit", "quit"}:

                console.print("Goodbye!")
                break

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

            temp(
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


if __name__ == "__main__":

    main()