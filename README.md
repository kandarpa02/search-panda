# Search Panda 🐼

A local, privacy-first AI search assistant built around Ollama and the OpenAI Agents SDK.

Search Panda is a secure, agentic search engine for people who want AI-powered answers without sending their questions to a third-party cloud service by default. It runs locally, searches the web when needed, reads the best result, and then uses a local model to answer with context.

```text
      ___..---..___
   .-""  _   _  ""-.
  /   .' \/ \/ '.   \
  |   /  .  .  .  \
   \  \  .  .  .  / /
    `-`-.___.__.-'`
        /  _  \
       |  ( )  |
        \  `-' /
         `-.__.-'

             Search Panda
```

## What makes this project special

This project is more than a simple chatbot. It behaves like a lightweight autonomous agent:

- it decides when a question needs live web information
- it runs a web search using DuckDuckGo
- it fetches and reads the most relevant page
- it passes that content to the local model
- it returns an answer grounded in the retrieved information

Because the default setup uses local Ollama, the system is secure-by-default and friendly to privacy-conscious users.

## Why it is secure and local-first

The app is configured around a local Ollama endpoint by default:

- provider: ollama
- base URL: http://localhost:11434/v1
- API key: "ollama"

That means the default experience is designed to stay on your machine instead of depending on a cloud LLM provider. If you intentionally change the provider or endpoint, you can use remote models, but the built-in setup is local-first.

## Features

- local LLM support through Ollama
- OpenAI-compatible API integration
- web search using DuckDuckGo
- page fetching and content extraction
- grounded answers using live search results
- interactive terminal chat interface
- configurable model selection
- beginner-friendly CLI workflow

## How it works

Search Panda follows this flow:

1. user asks a question
2. the agent decides whether live search is needed
3. it searches the web for relevant results
4. it reads the best matching page
5. it extracts readable text from that page
6. it asks the local Ollama model to answer using that context
7. it returns the final answer in the terminal

## Project structure

```text
search-panda/
├── search_panda/
│   ├── __init__.py
│   ├── config.py
│   ├── core.py
│   ├── helpers.py
│   ├── run.py
│   ├── tools.py
│   └── agent/
│       ├── __init__.py
│       ├── agents.py
│       └── search.py
├── tests/
│   └── test_local_ollama_project.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── search_panda.egg-info/
```

## Requirements

Before running Search Panda, make sure you have:

- Python 3.10 or newer
- Ollama installed and running locally
- a model pulled in Ollama, such as llama3.1:8b or qwen2.5:7b

## Install Ollama

Install Ollama for your operating system, then start it:

```bash
ollama serve
```

Pull a model:

```bash
ollama pull llama3.1:8b
```

## Install Search Panda

Clone the repository:

```bash
git clone <your-repo-url>
cd search-panda
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

## Run the app

Set a model once:

```bash
search-panda set llama3.1:8b
```

Or start directly with a model name:

```bash
python -m search_panda.run --model llama3.1:8b
```

Once it starts, you will see a prompt like this:

```text
search-panda>
```

Type your question and press Enter.

### Example questions

```text
search-panda> What happened in AI this week?
search-panda> What is the latest status of OpenAI?
search-panda> Explain recursion in Python
```

## Default configuration

The app stores its local config in:

```text
~/.search-panda/config.json
```

The default model and local URL are set for Ollama:

```python
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1:8b"
```

## Important note about security and privacy

This project is intentionally designed for local-first usage. In its default state:

- prompts do not need to go to a cloud AI provider
- the model runs on your local machine
- the app can pull current web information without losing the local-first design
- you keep more control over your data and your model choices

That is why Search Panda feels like a secure, privacy-conscious, agentic AI search tool.

## Troubleshooting

### Ollama is not running

Start it with:

```bash
ollama serve
```

### Model not found

Download the model:

```bash
ollama pull llama3.1:8b
```

### App cannot reach local Ollama

Check that the local API is available at:

```text
http://localhost:11434/v1
```

### Search fails

The app uses DuckDuckGo search and fetches page content with an extraction library. If a site is unavailable or blocked, the tool will return a clean error instead of crashing.

## Development status

This project is small, practical, and easy to understand. It is a good example of:

- local-first AI application design
- agentic workflows using tools and search
- secure-by-default architecture
- simple Python CLI development

## License

This project is released under the MIT license.

---

Built with privacy in mind, powered locally by Ollama, and guided by an agentic search workflow. 🐼