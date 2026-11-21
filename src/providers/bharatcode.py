"""BharatCode specifics: model discovery via GET /models and the provider wiring.

Model selection is always driven by the live /models record (status, context window,
max output tokens, capabilities), never hard-coded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from .base import OpenAICompatibleProvider, provider_from_env

DEFAULT_BASE_URL = "https://bharatcode.ai/api/model/v1"


def _key() -> str:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/.config/bharatcode/.env"))
    load_dotenv()
    return os.environ.get("BHARATCODE_API_KEY", "")


def list_models(base_url: str | None = None) -> dict:
    base = (base_url or os.environ.get("BHARATCODE_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
    r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {_key()}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def select_model(records: dict, prefer: list[str] | None = None) -> dict:
    """Pick a live text model, preferring `prefer` order, then largest context."""
    prefer = prefer or []
    live = [m for m in records.get("data", []) if m.get("status") == "live" and not m.get("deprecated")]
    for pid in prefer:
        for m in live:
            if m.get("id") == pid:
                return m
    if not live:
        raise RuntimeError("no live, non-deprecated models returned by /models")
    return max(live, key=lambda m: (m.get("context_window") or 0))


class BharatCodeProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, **kwargs):
        kwargs.setdefault("name", "bharatcode")
        super().__init__(
            model=model,
            base_url=os.environ.get("BHARATCODE_BASE_URL", DEFAULT_BASE_URL),
            api_key=_key(),
            **kwargs,
        )


def record_models(out_path: str | Path | None = None) -> dict:
    records = list_models()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(records, indent=2) + "\n")
    return records
