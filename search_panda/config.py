from dataclasses import dataclass, asdict
from typing import Any
from pathlib import Path
import json

BASE_URL = {
    'groq':"https://api.groq.com/openai/v1",
    'deepseek':"https://api.deepseek.com/v1",
    'openai':"https://api.openai.com/v1"
}

REASONING_MODELS = {
    # OpenAI
    "gpt-5": ["minimal", "low", "medium", "high"],
    "gpt-5-mini": ["minimal", "low", "medium", "high"],
    "gpt-5-nano": ["minimal", "low", "medium", "high"],

    # DeepSeek
    "deepseek-reasoner": ["low", "medium", "high"],

    # Groq
    "openai/gpt-oss-20b": ["low", "medium", "high"],
    "openai/gpt-oss-120b": ["low", "medium", "high"],
}

@dataclass
class Setup:
    api_key:Any=None
    model:str=None
    provider:str=None
    reasoning_level:str=None


CONFIG_DIR = Path.home() / ".search-panda"
CONFIG_FILE = CONFIG_DIR / "config.json"

class ConfigManager:

    def exists(self) -> bool:
        return CONFIG_FILE.exists()

    def save(self, setup: Setup) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(setup), f, indent=4, default=str)

    def load(self) -> Setup:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)

        data["model"] = Path(data["model"])
        return Setup(**data)


import os

# os.remove(CONFIG_FILE)
# os.removedirs(CONFIG_DIR)