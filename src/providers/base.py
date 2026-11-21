"""Provider abstraction.

Any OpenAI-compatible chat-completions endpoint can be used. The experiment code only
sees `ChatProvider.complete(messages) -> Completion`.

Design notes (informed by measured BharatCode behaviour, see docs/harness_adr.md):
* Requests are *streamed* so that long server-side queue waits produce keep-alive
  chunks instead of an idle connection. If streaming fails, we fall back to the
  non-streaming call once.
* Timeouts are per-read, not wall-clock: BharatCode may queue a request for many minutes
  before the first token. `max_wait_s` is a safety valve.
* `finish_reason == "length"` marks the response incomplete. Incomplete responses are
  returned, flagged, and recorded - never silently retried with the same budget.
* 429 / 5xx / network errors use exponential backoff with jitter.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import requests


@dataclass
class Completion:
    model: str
    text: str
    finish_reason: str | None
    incomplete: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    request_id: str | None
    latency_s: float
    attempts: int
    error: str | None = None
    streamed: bool = False
    raw_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ProviderError(RuntimeError):
    pass


class ChatProvider:
    name = "base"

    def complete(self, messages: list[dict], **kwargs) -> Completion:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatibleProvider(ChatProvider):
    """Streaming chat-completions client with backoff and incomplete detection."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int = 400,
        temperature: float = 0.0,
        reasoning_effort: str | None = "none",
        connect_timeout_s: float = 30.0,
        read_timeout_s: float = 300.0,
        max_wait_s: float = 3600.0,
        max_retries: int = 4,
        extra_body: dict | None = None,
        name: str = "openai_compatible",
    ):
        if not api_key:
            raise ProviderError("API key is empty")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.max_wait_s = max_wait_s
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.name = name

    # ------------------------------------------------------------------
    def _payload(self, messages, stream: bool) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        body.update(self.extra_body)
        return body

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" ,
        }

    # ------------------------------------------------------------------
    def complete(self, messages: list[dict], **kwargs) -> Completion:
        for attempt in range(1, self.max_retries + 1):
            started = time.time()
            try:
                comp = self._stream_once(messages, attempt, started)
                if comp is not None:
                    return comp
            except _Retryable as exc:
                if attempt == self.max_retries:
                    return Completion(self.model, "", None, True, None, None, exc.request_id,
                                      time.time() - started, attempt,
                                      error=f"retryable_failure: {exc}")
                time.sleep(min(60, (2 ** (attempt - 1)) * 5) + random.uniform(0, 2))
            except _Fatal as exc:
                return Completion(self.model, "", None, True, None, None, exc.request_id,
                                  time.time() - started, attempt, error=f"fatal: {exc}")
        return Completion(self.model, "", None, True, None, None, None, 0.0,
                          self.max_retries, error="exhausted")

    # ------------------------------------------------------------------
    def _stream_once(self, messages, attempt, started) -> Completion | None:
        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(
                url,
                headers=self._headers(),
                data=json.dumps(self._payload(messages, stream=True)),
                stream=True,
                timeout=(self.connect_timeout_s, self.read_timeout_s),
            )
        except requests.RequestException as exc:
            raise _Retryable(str(exc))
        rid = resp.headers.get("x-request-id")
        if resp.status_code in (429, 500, 502, 503, 504) or resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}", rid)
        if resp.status_code == 410:
            raise _Fatal(f"HTTP 410 retired_model: {resp.text[:200]}", rid)
        if resp.status_code >= 400:
            body = resp.text[:400]
            # request-shape errors are deterministic: do not retry
            if resp.status_code in (400, 401, 403, 404, 422):
                raise _Fatal(f"HTTP {resp.status_code}: {body}", rid)
            raise _Retryable(f"HTTP {resp.status_code}: {body}", rid)

        text_parts: list[str] = []
        finish, usage = None, {}
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if time.time() - started > self.max_wait_s:
                    raise _Retryable("client max_wait_s exceeded", rid)
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith(":") or not line.startswith("data:"):
                    continue  # keep-alive / queue comments
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("error"):
                    raise _Retryable(f"stream error: {ev['error']}", rid)
                usage = ev.get("usage") or usage
                for ch in ev.get("choices", []):
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        text_parts.append(delta["content"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        except requests.RequestException as exc:
            raise _Retryable(f"stream read failed: {exc}", rid)
        finally:
            resp.close()

        text = "".join(text_parts)
        incomplete = finish == "length" or not finish
        return Completion(
            model=self.model, text=text, finish_reason=finish, incomplete=incomplete,
            prompt_tokens=(usage.get("prompt_tokens") if usage else None),
            completion_tokens=(usage.get("completion_tokens") if usage else None),
            request_id=rid, latency_s=time.time() - started, attempts=attempt,
            streamed=True, raw_usage=usage or {},
        )


class _Retryable(Exception):
    def __init__(self, msg, request_id=None):
        super().__init__(msg)
        self.request_id = request_id


class _Fatal(Exception):
    def __init__(self, msg, request_id=None):
        super().__init__(msg)
        self.request_id = request_id


def provider_from_env(model: str | None = None, **overrides) -> OpenAICompatibleProvider:
    from dotenv import load_dotenv

    load_dotenv(os.path.expanduser("~/.config/bharatcode/.env"))
    load_dotenv()
    key = os.environ.get("BHARATCODE_API_KEY", "")
    mdl = model or os.environ.get("BHARATCODE_MODEL", "glm-5.3-flash")
    return OpenAICompatibleProvider(
        model=mdl,
        base_url=os.environ.get("BHARATCODE_BASE_URL", "https://bharatcode.ai/api/model/v1"),
        api_key=key,
        name="bharatcode",
        **overrides,
    )
