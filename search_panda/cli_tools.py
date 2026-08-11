import typer
import getpass
from .config import Setup
from pathlib import Path
import os

default = Setup()

app = typer.Typer()

# @app.command()
# def api_key():
#     global default
#     key = getpass.getpass("Put a valid api_key: ")
#     default.api_key = key

@app.command()
def model():
    from .ollama.models import download_model, CACHE_DIR, ALLOWED_MODELS

    global default

    model_name = input("Put a valid model name: ").strip()

    if model_name not in ALLOWED_MODELS:
        raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Available models: {', '.join(ALLOWED_MODELS.keys())}"
        )

    filename = ALLOWED_MODELS[model_name]["filename"]
    model_path = CACHE_DIR / filename

    if not model_path.exists():
        model_path = download_model(model_name)

    default.model = model_path

# @app.command()
# def provider():
#     global default
#     prov = input("Choose your LLM provider: ")
#     default.provider = prov


# @app.command()
# def reasoning_level():
#     global default
#     reasoning_lvl = input("Choose reasoning level (e.g. 'low', 'medium', 'high', keep in mind reasoning level only works with reasoning models, incompatible model wont work.): ")
#     default.reasoning_level = reasoning_level

