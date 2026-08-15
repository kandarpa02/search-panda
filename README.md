# Search Panda 🐼

Private, local-first AI search for the modern web.

Search Panda is a secure, agentic search assistant that combines local Ollama models with web search and source-grounded answers. It helps you find fresh information, inspect the most relevant web result, and answer questions using local AI while keeping the default experience on your machine.

## Why teams and developers use Search Panda

Search Panda is built for users who want the speed and intelligence of AI search without depending on a cloud-only workflow.

- Private by default: runs with local Ollama instead of a remote model by default
- Agentic search: decides when web search is useful and uses retrieved information
- Source-aware answers: reads the most relevant page before answering
- Flexible model choice: switch between local Ollama models easily
- Easy to run: simple terminal-based workflow for fast experimentation and local use

## How it works

Search Panda follows a simple, reliable workflow:

1. You ask a question.
2. The agent decides whether live web information is needed.
3. It performs a targeted search using DuckDuckGo.
4. It reads the most relevant result and extracts usable content.
5. It sends that grounded context to a local Ollama model.
6. It returns a concise answer based on the retrieved information.

## Product value

Search Panda is designed for privacy-conscious users, developers, and teams who want:

- secure local AI interactions
- a better answer quality than plain search alone
- the ability to use fresh web information with local model reasoning
- a lightweight, self-hosted alternative to cloud AI search tools

## Security and privacy

By default, Search Panda is configured for a local Ollama setup:

- provider: ollama
- base URL: http://localhost:11434/v1
- API key: "ollama"

This makes the default experience local-first and self-contained. You can still change the model or provider intentionally, but the built-in design favors privacy, control, and local execution.

## Core features

- Local LLM inference with Ollama
- OpenAI-compatible integration for agent tooling
- Search-driven answers with live web context
- Result reading and content extraction from fetched pages
- Interactive CLI experience
- Model selection via simple command-line configuration
- Built for developers and privacy-first workflows

## Quick start

### 1. Install Ollama

Install Ollama for your operating system, then start the local server:

```bash
ollama serve
```

Pull a model:

```bash
ollama pull llama3.1:8b
```

### 2. Install Search Panda

```bash
git clone <your-repo-url>
cd search-panda
python -m venv .venv
```

Activate the environment:

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

### 3. Run the app

Set a model:

```bash
search-panda set llama3.1:8b
```

Or run directly:

```bash
python -m search_panda.run --model llama3.1:8b
```

Then start asking questions:

```text
search-panda> What happened in AI this week?
search-panda> What is the latest status of OpenAI?
search-panda> Explain recursion in Python
```

## Architecture

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

## Configuration

The app stores its local config here:

```text
~/.search-panda/config.json
```

Default settings are designed around local Ollama:

```python
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1:8b"
```

## Use cases

Search Panda is useful for:

- current-events questions that require live web information
- research-style prompts grounded in relevant sources
- local AI workflows where privacy matters
- developer exploration and experimentation with agent-based search

## Troubleshooting

### Ollama is not running

```bash
ollama serve
```

### Model is missing

```bash
ollama pull llama3.1:8b
```

### App cannot connect to Ollama

Ensure the local endpoint is available:

```text
http://localhost:11434/v1
```

### Search returns no results

Search Panda uses DuckDuckGo. If a result is unavailable or blocked, the application will surface a clean error rather than failing silently.

## License

This project is released under the MIT license.

---

Built for safe, local AI search with a product mindset. 🐼