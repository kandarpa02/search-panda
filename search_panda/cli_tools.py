import typer
import getpass
from .config import Setup

default = Setup()

app = typer.Typer()

@app.command()
def api_key():
    global default
    key = getpass.getpass("Put a valid api_key: ")
    default.api_key = key

@app.command()
def model():
    global default
    model = input("Put a valid model name: ")
    default.model = model

@app.command()
def provider():
    global default
    prov = input("Choose your LLM provider: ")
    default.provider = prov


@app.command()
def reasoning_level():
    global default
    reasoning_lvl = input("Choose reasoning level (e.g. 'low', 'medium', 'high', keep in mind reasoning level only works with reasoning models, incompatible model wont work.): ")
    default.reasoning_level = reasoning_level

