from pathlib import Path
from huggingface_hub import hf_hub_download


CACHE_DIR = Path(".cache/models")


ALLOWED_MODELS = {
    "lfm2.5-350m": {
        "repo_id": "LiquidAI/LFM2.5-350M-GGUF",
        "filename": "LFM2.5-350M-Q4_K_M.gguf",
    },
    "qwen2.5-0.5b": {
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    },
}


def download_model(name: str) -> Path:
    if name not in ALLOWED_MODELS:
        raise ValueError(f"Model '{name}' is not allowed")

    model = ALLOWED_MODELS[name]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=model["repo_id"],
        filename=model["filename"],
        local_dir=CACHE_DIR,
    )

    return Path(path)