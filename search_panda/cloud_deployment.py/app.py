"""Search Panda cloud deployment entrypoint.

This module creates the production FastAPI application using
environment variables. It is designed for platforms such as Render.
"""

import os
from dotenv import load_dotenv
from search_panda.api import create_api_app

# Load local .env during development.
# On Render, environment variables are injected directly by the platform.
load_dotenv()


def get_env(
    name: str,
    default: str | None = None,
) -> str | None:
    """Read and normalize an environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()

    return value or default

# ---------------------------------------------------------------------------
# Search Panda cloud configuration
# ---------------------------------------------------------------------------

MODEL = get_env(
    "MODEL",
    "gemini-2.5-flash",
)

PROVIDER = get_env(
    "PROVIDER",
    "gemini",
)

API_KEY = get_env("GEMINI_API_KEY")

BASE_URL = get_env(
    "BASE_URL",
)

WEB_MODE = get_env(
    "WEB_MODE",
    "on",
)

REASONING_LEVEL = get_env(
    "REASONING_LEVEL",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured. "
        "Set it in your .env file locally or as an environment "
        "variable on your cloud deployment."
    )


app = create_api_app(
    default_model=MODEL,
    default_provider=PROVIDER,
    default_api_key=API_KEY,
    default_base_url=BASE_URL,
    default_web_mode=WEB_MODE,
    default_reasoning_level=REASONING_LEVEL,
)
