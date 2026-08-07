from .cli_tools import default, api_key, model, provider
from .core import chat_completion
from .config import ConfigManager, CONFIG_DIR, CONFIG_FILE
import os
from rich.console import Console
from rich.markdown import Markdown

config = ConfigManager()
console = Console()

def temp(config, message):
    response = chat_completion(config, "You are a helpful chatbot", message)
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            console.print(delta, end="")

def main():
    global config
    global default
    print("Welcome to search-panda 🐼")
    if not config.exists():
        print(f"Setup the engine...")

        api_key()
        model()
        provider()

        config.save(default)

        print("Environment is ready.")
        print("Enjoy chatting! 🐼🎍")
    else:
        default = config.load()


    while True:

        question = input("> ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye")
            break

        if question.lower() in {"remove"}:
            print("Removed current config, you need to setup a new one later.")

            os.remove(CONFIG_FILE)
            os.removedirs(CONFIG_DIR)
            break

        if not question:
            continue
        
        temp(default, question)
        print("\n")
        # console.print(Markdown(answer))

if __name__ == "__main__":
    main()