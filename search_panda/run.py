"""Search Panda CLI — Ollama-style interactive terminal interface."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from .agent.agents import agent_completion, LAST_SOURCES
import search_panda.agent.agents as agents_module
from .agent.planner import plan as query_plan
from .config import ConfigManager, Setup, DEFAULT_OLLAMA_BASE_URL

console = Console()
config_manager = ConfigManager()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def ensure_setup() -> Setup:
    if config_manager.exists():
        setup = config_manager.load()
        setup.provider = "ollama"
        setup.base_url = setup.base_url or DEFAULT_OLLAMA_BASE_URL
        setup.api_key = setup.api_key or "ollama"
        setup.web_mode = getattr(setup, "web_mode", "on")
        return setup

    setup = Setup()
    config_manager.save(setup)
    return setup


def update_setup(
    model: str | None = None,
    base_url: str | None = None,
    web_mode: str | None = None,
) -> Setup:
    setup = ensure_setup()
    if model:
        setup.model = model.strip()
    if base_url:
        setup.base_url = base_url.strip()
    if web_mode:
        setup.web_mode = web_mode.strip().lower()
    config_manager.save(setup)
    return setup


# ---------------------------------------------------------------------------
# UI Helpers & Slash Commands
# ---------------------------------------------------------------------------

def print_banner(setup: Setup, verbose: bool = False) -> None:
    web_color = "green" if setup.web_mode == "on" else ("yellow" if setup.web_mode == "auto" else "red")
    console.print(
        Panel(
            f"[bold cyan]Search Panda 🐼[/bold cyan] — [dim]AI Search Engine[/dim]\n\n"
            f"  [bold]Model:[/bold]     [green]{setup.model}[/green]\n"
            f"  [bold]Endpoint:[/bold]  [dim]{setup.base_url}[/dim]\n"
            f"  [bold]Web Mode:[/bold]  [{web_color}]{setup.web_mode.upper()}[/{web_color}]\n"
            f"  [bold]Verbose:[/bold]   {'[green]ON[/green]' if verbose else '[dim]OFF[/dim]'}\n\n"
            f"[dim]Type [bold cyan]/? [/bold cyan] or [bold cyan]/help[/bold cyan] for commands, [bold cyan]/web on|off[/bold cyan] to toggle web, [bold cyan]/bye[/bold cyan] to exit.[/dim]",
            border_style="cyan",
        )
    )


def print_help() -> None:
    table = Table(title="Search Panda Commands", border_style="dim")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    table.add_row("/help, /?", "Show this list of commands")
    table.add_row("/show, /info", "Display current model and configuration")
    table.add_row("/web [on|off|auto]", "Switch web search mode (on=always search, off=no web, auto=smart)")
    table.add_row("/model <name>", "Switch active LLM model (e.g. /model llama3.2:1b)")
    table.add_row("/set <key> <val>", "Set config parameter (e.g. /set model llama3.1:8b, /set web on)")
    table.add_row("/verbose [on|off]", "Toggle query plan & evidence inspection")
    table.add_row("/clear", "Clear terminal screen")
    table.add_row("/exit, /bye, /quit", "Exit Search Panda")
    console.print(table)


def print_status(setup: Setup, verbose: bool) -> None:
    table = Table(title="Current Configuration", border_style="cyan")
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Model", setup.model)
    table.add_row("Endpoint", setup.base_url)
    table.add_row("Provider", setup.provider)
    table.add_row("Web Search Mode", setup.web_mode.upper())
    table.add_row("Verbose Plan Inspection", "ON" if verbose else "OFF")
    console.print(table)


def handle_slash_command(cmd_line: str, setup: Setup, verbose: bool) -> tuple[Setup, bool, bool]:
    """Handle slash command. Returns (setup, verbose, should_continue_loop)."""
    parts = cmd_line.strip().split(maxsplit=2)
    cmd = parts[0].lower()
    arg1 = parts[1].lower() if len(parts) > 1 else ""
    arg2 = parts[2] if len(parts) > 2 else ""

    if cmd in {"/exit", "/quit", "/bye", "/q"}:
        console.print("[dim]Goodbye! 🐼[/dim]")
        return setup, verbose, False

    if cmd in {"/help", "/?"}:
        print_help()
        return setup, verbose, True

    if cmd in {"/clear", "/cls"}:
        os.system("cls" if os.name == "nt" else "clear")
        print_banner(setup, verbose)
        return setup, verbose, True

    if cmd in {"/show", "/info", "/status"}:
        print_status(setup, verbose)
        return setup, verbose, True

    if cmd in {"/verbose"}:
        if arg1 in {"on", "1", "true"}:
            verbose = True
        elif arg1 in {"off", "0", "false"}:
            verbose = False
        else:
            verbose = not verbose
        console.print(f"Verbose mode is now [bold]{'ON' if verbose else 'OFF'}[/bold]")
        return setup, verbose, True

    if cmd in {"/web", "/search"}:
        if arg1 in {"on", "off", "auto"}:
            setup = update_setup(web_mode=arg1)
            console.print(f"Web mode set to [bold green]{setup.web_mode.upper()}[/bold green]")
        else:
            console.print("[yellow]Usage: /web on | /web off | /web auto[/yellow]")
        return setup, verbose, True

    if cmd in {"/model"}:
        if arg1:
            raw_model = parts[1]
            setup = update_setup(model=raw_model)
            console.print(f"Active model switched to [bold green]{setup.model}[/bold green]")
        else:
            console.print("[yellow]Usage: /model <model_name> (e.g. /model llama3.2:1b)[/yellow]")
        return setup, verbose, True

    if cmd in {"/set"}:
        if arg1 == "model" and arg2:
            setup = update_setup(model=arg2)
            console.print(f"Model set to [bold green]{setup.model}[/bold green]")
        elif arg1 in {"web", "web_mode"} and arg2:
            if arg2.lower() in {"on", "off", "auto"}:
                setup = update_setup(web_mode=arg2)
                console.print(f"Web mode set to [bold green]{setup.web_mode.upper()}[/bold green]")
            else:
                console.print("[yellow]Valid web modes: on, off, auto[/yellow]")
        elif arg1 in {"endpoint", "base_url"} and arg2:
            setup = update_setup(base_url=arg2)
            console.print(f"Endpoint set to [bold green]{setup.base_url}[/bold green]")
        else:
            console.print("[yellow]Usage: /set model <name> | /set web <on|off|auto> | /set endpoint <url>[/yellow]")
        return setup, verbose, True

    console.print(f"[red]Unknown command: {cmd}[/red]. Type [cyan]/help[/cyan] for available commands.")
    return setup, verbose, True


# ---------------------------------------------------------------------------
# Chat Execution
# ---------------------------------------------------------------------------

async def chat(message: str, config: Setup | None = None, verbose: bool = False) -> None:
    current = config or ensure_setup()

    # Show query plan in verbose mode
    if verbose:
        qplan = query_plan(message, web_mode=current.web_mode)
        console.print(
            Panel(
                f"[bold]Intent:[/bold] {qplan.intent}\n"
                f"[bold]Web Search Required:[/bold] {qplan.needs_search} (mode={current.web_mode})\n"
                f"[bold]Time-Sensitive / Live:[/bold] {qplan.time_sensitive}\n"
                f"[bold]Sub-Queries:[/bold] {qplan.sub_queries or ['(direct answer)']}\n"
                f"[bold]Site Hints:[/bold] {qplan.site_hints or ['(none)']}",
                title="[cyan]Query Plan Inspector[/cyan]",
                border_style="dim",
            )
        )

    status_msg = "[cyan]Searching & synthesizing...[/cyan]" if current.web_mode != "off" else "[cyan]Thinking...[/cyan]"
    with Status(status_msg, spinner="dots", console=console):
        response = await agent_completion(config=current, message=message)

    console.print()
    console.print(Markdown(response))

    # Display clean source links if any were retrieved during search
    retrieved = getattr(agents_module, "LAST_SOURCES", [])
    if retrieved:
        console.print()
        console.print("[bold dim]Sources:[/bold dim]")
        for i, src in enumerate(retrieved[:4], 1):
            domain_label = f" [dim]({src['domain']})[/dim]" if src.get("domain") else ""
            console.print(f" [cyan]{i}.[/cyan] {src['title']}{domain_label}\n    [dim]{src['url']}[/dim]")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-panda",
        description="Local Ollama AI Search Engine",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Model to use (e.g. --model llama3.2:1b).",
    )
    parser.add_argument(
        "--web",
        dest="web_mode",
        choices=["on", "off", "auto"],
        default=None,
        help="Web search mode: on (default), off, auto.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show query plan and evidence inspection.",
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
        help="Model name to save when using 'set'.",
    )
    return parser


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def interactive_loop(verbose: bool = False) -> None:
    setup = ensure_setup()

    if not setup.model:
        console.print("[bold yellow]No model configured.[/bold yellow]")
        console.print("Set one: search-panda --model llama3.2:1b or /model llama3.2:1b")
        return

    print_banner(setup, verbose)

    while True:
        try:
            prompt_str = f"search-panda ({setup.model})> "
            try:
                question = input(prompt_str).strip()
            except EOFError:
                console.print("\n[dim]Goodbye! 🐼[/dim]")
                break

            if not question:
                continue

            if question.startswith("/") or question.lower() in {"exit", "quit", "q"}:
                cmd = question if question.startswith("/") else f"/{question}"
                setup, verbose, should_continue = handle_slash_command(cmd, setup, verbose)
                if not should_continue:
                    break
                continue

            await chat(question, setup, verbose=verbose)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye! 🐼[/dim]")
            break

        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

async def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = getattr(args, "verbose", False)

    setup = ensure_setup()

    if args.web_mode:
        setup = update_setup(web_mode=args.web_mode)

    if args.model:
        setup = update_setup(model=args.model)
        console.print(f"Model set to [green]{setup.model}[/green]")
        await interactive_loop(verbose=verbose)
        return

    if args.command == "set":
        if not args.model_name:
            parser.error("Usage: search-panda set llama3.1:8b")
        setup = update_setup(model=args.model_name)
        console.print(f"Model set to [green]{setup.model}[/green]")
        await interactive_loop(verbose=verbose)
        return

    await interactive_loop(verbose=verbose)


if __name__ == "__main__":
    asyncio.run(main())