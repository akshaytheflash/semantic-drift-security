"""Token accounting.

Two estimates are recorded per history and compared with the API's usage block:
  * cl100k_base (tiktoken) - a stable, vendor-neutral estimate for sizing;
  * a per-model correction factor measured from API calls (`calibration_*.jsonl`).

The API's `prompt_tokens` remains the ground truth for scoring gates; tiktoken is used for
planning and for the calibrated pre-check, which avoids spending a full 150k+ call merely to
size a corpus.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("cl100k_base")


def count_text(s: str) -> int:
    return len(_enc().encode(s or ""))


def count_messages(messages: list[dict]) -> int:
    # +4 tokens per message for role/format overhead (a common approximation)
    return sum(count_text(m.get("content", "")) + 4 for m in messages)


def calibration_path(model: str) -> Path:
    return ROOT / "experiments" / "processed" / f"token_ratio_{model}.json"


def measure_ratio(model: str) -> dict | None:
    """Derive API/cl100k ratio from recorded calibration calls (needs the same text)."""
    path = calibration_path(model)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_ratio(model: str, ratio: float, n: int, source: str) -> None:
    path = calibration_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "api_over_cl100k": round(ratio, 4),
                                "n": n, "source": source}, indent=2) + "\n")


def estimate_tokens(model: str, messages: list[dict]) -> int:
    base = count_messages(messages)
    r = measure_ratio(model)
    if r:
        return int(base * r["api_over_cl100k"])
    return base
