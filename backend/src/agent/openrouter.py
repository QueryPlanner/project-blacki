"""OpenRouter API client for listing available models.

Provides a cached fetch of OpenRouter models for the model picker UI.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from dotenv import load_dotenv

_load_dotenv = Path(__file__).resolve().parent.parent.parent
load_dotenv(_load_dotenv / ".env")

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_TTL_SECONDS = 60

_cache: dict[str, Any] | None = None
_cache_updated_at: float = 0.0
_cache_lock = threading.Lock()


def _refresh_cache() -> None:
    global _cache, _cache_updated_at
    models = fetch_models()
    ids = {m.get("id") for m in models if isinstance(m.get("id"), str)}
    simplified = [
        {
            "id": m.get("id"),
            "name": m.get("name", m.get("id", "unknown")),
            "context_length": m.get("context_length"),
        }
        for m in models
        if isinstance(m.get("id"), str)
    ]
    _cache = {"ids": ids, "models": simplified}
    _cache_updated_at = time.monotonic()


def _get_api_key() -> str | None:
    import os

    return os.environ.get("OPENROUTER_API_KEY")


def fetch_models() -> list[dict[str, Any]]:
    """Fetch models from OpenRouter API.

    Returns:
        List of model objects with id, name, context_length, etc.
        Empty list on error.
    """
    api_key = _get_api_key()
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set; cannot fetch models")
        return []

    req = Request(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning("Failed to fetch OpenRouter models: %s: %s", type(e).__name__, e)
        return []

    raw_models = data.get("data")
    if not isinstance(raw_models, list):
        return []

    return raw_models


def get_cached_model_ids() -> set[str]:
    """Return set of allowed model IDs from cache.

    Used by callbacks to validate selectedModel from state.
    """
    with _cache_lock:
        if _cache is None or (time.monotonic() - _cache_updated_at) >= CACHE_TTL_SECONDS:
            _refresh_cache()
        return set(_cache["ids"])  # type: ignore[index]


def get_models_for_ui() -> list[dict[str, Any]]:
    """Return models list for frontend picker, with short TTL cache.

    Returns:
        List of {id, name, context_length} for each model.
    """
    with _cache_lock:
        if _cache is None or (time.monotonic() - _cache_updated_at) >= CACHE_TTL_SECONDS:
            _refresh_cache()
        return list(_cache["models"])  # type: ignore[index]
