from dataclasses import dataclass, asdict
from typing import Any, Optional
from pathlib import Path
import json

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1:8b"

BASE_URL = {
    "ollama": DEFAULT_OLLAMA_BASE_URL,
    "groq": "https://api.groq.com/openai/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
}

REASONING_MODELS = {
    "gpt-5": ["minimal", "low", "medium", "high"],
    "gpt-5-mini": ["minimal", "low", "medium", "high"],
    "gpt-5-nano": ["minimal", "low", "medium", "high"],
    "deepseek-reasoner": ["low", "medium", "high"],
    "openai/gpt-oss-20b": ["low", "medium", "high"],
    "openai/gpt-oss-120b": ["low", "medium", "high"],
}


@dataclass
class Setup:
    api_key: Any = "ollama"
    model: str = DEFAULT_MODEL
    provider: str = "ollama"
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    reasoning_level: Optional[str] = None


CONFIG_DIR = Path.home() / ".search-panda"
CONFIG_FILE = CONFIG_DIR / "config.json"


class ConfigManager:
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
        self.config_file = self.config_dir / "config.json"

    def exists(self) -> bool:
        return self.config_file.exists()

    def save(self, setup: Setup) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(asdict(setup), f, indent=4, default=str)

    def load(self) -> Setup:
        if not self.config_file.exists():
            return Setup()

        with open(self.config_file, encoding="utf-8") as f:
            data = json.load(f)

        data.setdefault("api_key", "ollama")
        data.setdefault("model", DEFAULT_MODEL)
        data.setdefault("provider", "ollama")
        data.setdefault("base_url", DEFAULT_OLLAMA_BASE_URL)

        return Setup(**data)


import os

# os.remove(CONFIG_FILE)
# os.removedirs(CONFIG_DIR)