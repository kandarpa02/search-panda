# Search Panda 🐼

<p align="center">
  <strong>Private, agentic, AI-first search engine powered by local Ollama models.</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#how-it-works">Architecture</a> •
  <a href="#slash-commands">CLI & Commands</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#testing">Testing</a>
</p>

---

## Overview

**Search Panda** is an autonomous, privacy-first AI search engine designed to run entirely on your local machine. It combines local LLMs (via Ollama) with real-time web retrieval, intelligent query planning, BM25 relevance ranking, and deep page reading to produce accurate, fact-checked, and source-grounded answers.

Unlike cloud-dependent search assistants, Search Panda keeps your search history, prompts, and local AI reasoning completely on your machine by default.

---

## Key Features

- 🧠 **Zero-Latency Query Planner**: Automatically classifies query intent (`news`, `code`, `factual`, `math`, `opinion`, `conversational`), detects time-sensitivity, and decomposes complex queries into focused sub-queries without extra LLM latency.
- ⚡ **Optimized for Small & Large Models**: Instant parallel page extraction delivers deep, grounded evidence directly to models ranging from lightweight `1B`/`3B`/`8B` local models to `70B+` architectures without fragile multi-turn tool loops.
- 📊 **BM25 & Domain Authority Ranking**: Ranks search results using term-frequency keyword matching blended with domain credibility signals (e.g., Wikipedia, Reuters, GitHub, StackOverflow, ArXiv) and root-domain deduplication for diverse sources.
- 💻 **Ollama-Style Interactive Terminal**: Clean REPL with interactive slash commands (`/web`, `/model`, `/set`, `/verbose`, `/show`, `/clear`, `/help`).
- 🌐 **Web Search Modes**: Flexible search controls (`/web on` for search-first engine behavior, `/web auto` for heuristic activation, `/web off` for pure offline reasoning).
- 🔗 **Clean Source Link Presentation**: Automatically formats and presents verified source links, domains, and references at the conclusion of every answer.
- 🔒 **100% Local-First & Private**: Direct OpenAI-compatible integration with local Ollama instances (`http://localhost:11434/v1`).

---

## How It Works

```
                        User Query
                            │
                            ▼
              ┌───────────────────────────┐
              │    Query Planner Engine   │
              │  • Intent Classification  │
              │  • Query Normalization    │
              │  • Sub-query Splitting    │
              └─────────────┬─────────────┘
                            │
            ┌───────────────┴───────────────┐
       (Web Mode: ON / Auto)           (Web Mode: OFF)
            │                               │
            ▼                               ▼
  ┌───────────────────┐             ┌────────────────┐
  │ Parallel DDG      │             │ Direct Local   │
  │ Web Search        │             │ Model Response │
  └─────────┬─────────┘             └────────────────┘
            │
            ▼
  ┌───────────────────┐
  │ BM25 & Authority  │
  │ Ranking + Dedup   │
  └─────────┬─────────┘
            │
            ▼
  ┌───────────────────┐
  │ Concurrent Full-  │
  │ Page Extraction   │
  └─────────┬─────────┘
            │
            ▼
  ┌───────────────────────────┐
  │ Grounded Synthesis Agent  │
  │ • Evidence Verification   │
  │ • Formatted Answer        │
  │ • Source Citations        │
  └───────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

Make sure you have [Ollama](https://ollama.com/) installed and running:

```bash
# Start Ollama
ollama serve

# Pull your preferred model (e.g., llama3.2:1b, llama3.1:8b, qwen2.5:7b)
ollama pull llama3.2:1b
```

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/kandarpa02/search-panda.git
cd search-panda

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Optionally install in editable development mode:

```bash
pip install -e .
```

---

## Usage

### Interactive Shell

Launch the interactive search engine terminal:

```bash
python -m search_panda.run --model llama3.2:1b
```

You will enter the Search Panda terminal:

```text
╭─────────────────────────────────────────────────────────────────────────────────────────╮
│ Search Panda 🐼 — AI Search Engine                                                      │
│                                                                                         │
│   Model:     llama3.2:1b                                                                │
│   Endpoint:  http://localhost:11434/v1                                                  │
│   Web Mode:  ON                                                                         │
│   Verbose:   OFF                                                                        │
│                                                                                         │
│ Type /? or /help for commands, /web on|off to toggle web, /bye to exit.                 │
╰─────────────────────────────────────────────────────────────────────────────────────────╯
search-panda (llama3.2:1b)> who won fifa 2026?

• Planning: news (time-sensitive)
• Searching: who won fifa 2026
• Reading 2 top sources...

The 2026 FIFA World Cup has not taken place yet. It is scheduled to be held from June 11 to
July 19, 2026, jointly hosted by 16 cities across three North American countries: Canada,
Mexico, and the United States. Consequently, a winner has not been crowned yet.

Sources:
 1. 2026 FIFA World Cup - Wikipedia (en.wikipedia.org)
    https://en.wikipedia.org/wiki/2026_FIFA_World_Cup
 2. FIFA World Cup 2026 Schedule & Final (fifa.com)
    https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/final
```

---

## Interactive Slash Commands

Search Panda supports in-session slash commands for seamless control:

| Command | Arguments | Description |
|---|---|---|
| `/help`, `/?` | — | Display the list of available commands |
| `/show`, `/info` | — | View current active model, endpoint, and search mode |
| `/web`, `/search` | `on` \| `off` \| `auto` | Switch web search mode dynamically |
| `/model` | `<name>` | Hot-swap active LLM (e.g. `/model llama3.1:8b`) |
| `/set` | `<key> <val>` | Configure settings (`/set model ...`, `/set web ...`, `/set endpoint ...`) |
| `/verbose` | `on` \| `off` | Toggle query plan and source inspection details |
| `/clear`, `/cls` | — | Clear the terminal screen |
| `/exit`, `/bye`, `/quit` | — | Exit Search Panda |

---

## Command-Line Arguments

You can configure Search Panda directly when launching from the CLI:

```bash
# Launch with a specific model
python -m search_panda.run --model llama3.1:8b

# Launch with web mode set to auto or off
python -m search_panda.run --web auto

# Launch with verbose query plan inspection enabled
python -m search_panda.run --verbose

# Save a default model for future sessions
python -m search_panda.run set llama3.1:8b
```

---

## Configuration

Search Panda stores its local user configuration at:

- **Linux / macOS**: `~/.search-panda/config.json`
- **Windows**: `C:\Users\<User>\.search-panda\config.json`

Example configuration:

```json
{
    "api_key": "ollama",
    "model": "llama3.2:1b",
    "provider": "ollama",
    "base_url": "http://localhost:11434/v1",
    "reasoning_level": null,
    "web_mode": "on"
}
```

---

## Architecture & Codebase Structure

```text
search-panda/
├── search_panda/
│   ├── __init__.py
│   ├── config.py                 # Configuration manager & defaults
│   ├── core.py                   # Direct chat completion helpers
│   ├── helpers.py                # Exception definitions
│   ├── run.py                    # Production interactive CLI REPL
│   ├── tools.py                  # OpenAI-compatible tool definitions
│   └── agent/
│       ├── __init__.py
│       ├── agents.py             # Agent runtime, tools & synthesis loop
│       ├── data_representation.py# Pydantic data models (SearchResult, PageContent, QueryPlan)
│       ├── page_ranker.py        # BM25 scoring & domain authority ranker
│       ├── planner.py            # Heuristic query planner & intent router
│       └── search.py             # SearchSession, parallel DDGS & web scrapers
├── tests/
│   └── test_local_ollama_project.py # Comprehensive pytest test suite (33 tests)
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Testing

Search Panda includes a test suite covering intent classification, query decomposition, BM25 ranking, domain scoring, CLI commands, and agent execution:

```bash
# Run test suite
pytest tests/ -v
```

```text
============================= test session starts =============================
collected 33 items

tests/test_local_ollama_project.py::test_setup_defaults_to_ollama_localhost PASSED
tests/test_local_ollama_project.py::test_config_manager_round_trip PASSED
tests/test_local_ollama_project.py::test_needs_web_search_for_recent_or_unknown_facts PASSED
tests/test_local_ollama_project.py::test_needs_web_search_modes PASSED
tests/test_local_ollama_project.py::test_clean_query_text PASSED
tests/test_local_ollama_project.py::test_classify_intent PASSED
tests/test_local_ollama_project.py::test_rewrite_query_news_adds_year_if_missing PASSED
tests/test_local_ollama_project.py::test_rank_results_orders_by_relevance PASSED
tests/test_local_ollama_project.py::test_rank_results_deduplicates_by_domain PASSED
tests/test_local_ollama_project.py::test_domain_authority_known_domains PASSED
tests/test_local_ollama_project.py::test_slash_command_web_toggle PASSED
tests/test_local_ollama_project.py::test_slash_command_model_switch PASSED
...
============================= 33 passed in 9.83s ==============================
```

---

## License

This project is licensed under the [MIT License](LICENSE).