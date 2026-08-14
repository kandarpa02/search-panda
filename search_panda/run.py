import argparse
import asyncio

from rich.console import Console
from rich.markdown import Markdown

from .agent.agents import agent_completion
from .config import ConfigManager, Setup, DEFAULT_OLLAMA_BASE_URL

console = Console()
config_manager = ConfigManager()


def ensure_setup() -> Setup:
    if config_manager.exists():
        setup = config_manager.load()
        setup.provider = "ollama"
        setup.base_url = DEFAULT_OLLAMA_BASE_URL
        setup.api_key = setup.api_key or "ollama"
        return setup

    setup = Setup()
    config_manager.save(setup)
    return setup


def set_model(model_name: str) -> Setup:
    setup = ensure_setup()
    setup.model = model_name.strip() or setup.model
    setup.provider = "ollama"
    setup.base_url = DEFAULT_OLLAMA_BASE_URL
    setup.api_key = "ollama"
    config_manager.save(setup)
    return setup


async def chat(message: str, config: Setup | None = None):
    current = config or ensure_setup()
    response = await agent_completion(
        config=current,
        message=message,
    )
    console.print(Markdown(response))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-panda",
        description="Local Ollama search agent",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Optional model to save before starting the chat session.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["set"],
        help="Optional command. Use: search-panda set llama3.1:8b",
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Model name to save when using the set command.",
    )
    return parser


async def interactive_loop():
    setup = ensure_setup()

    if not setup.model:
        console.print("[bold yellow]No model is configured.[/bold yellow]")
        console.print("Set one with: python -m search_panda.run set llama3.1:8b")
        return

    console.print("[bold cyan]Search Panda 🐼[/bold cyan]")
    console.print(f"Using local Ollama at {setup.base_url} with model {setup.model}")

    while True:
        try:
            question = input("search-panda> ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                console.print("Goodbye!")
                break

            await chat(question, setup)
            console.print()

        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break

        except Exception as exc:
            console.print(f"[red]{exc}[/red]")


async def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model:
        setup = set_model(args.model)
        console.print(f"Model set to {setup.model}")
        await interactive_loop()
        return

    if args.command == "set":
        if not args.model_name:
            parser.error("Use: search-panda set llama3.1:8b")

        setup = set_model(args.model_name)
        console.print(f"Model set to {setup.model}")
        await interactive_loop()
        return

    await interactive_loop()


if __name__ == "__main__":
    asyncio.run(main())