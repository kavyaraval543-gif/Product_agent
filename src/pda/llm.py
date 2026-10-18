"""Thin client for a local Ollama server.

Uses `/api/chat` with a JSON-Schema `format`, so the model's output is grammar-constrained to
the Pydantic model. We still validate afterwards: constrained decoding guarantees shape, not truth.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TypeVar

import requests
from pydantic import BaseModel, ValidationError

from pda.errors import LLMModelMissing, LLMOutputError, LLMUnavailable

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMMeta:
    latency_s: float
    eval_tokens: int | None
    prompt_tokens: int | None
    attempts: int


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        *,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        timeout_s: float = 180.0,
        max_retries: int = 2,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        # Local traffic must never be routed through a corporate/system proxy.
        self._session = session or requests.Session()
        if session is None:
            self._session.trust_env = False

    # ------------------------------------------------------------------ health

    def version(self) -> str:
        try:
            resp = self._session.get(f"{self.base_url}/api/version", timeout=5)
            resp.raise_for_status()
            return str(resp.json().get("version", "unknown"))
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve` (or open the Ollama app)."
            ) from exc
        except (requests.HTTPError, ValueError) as exc:
            raise LLMUnavailable(f"Unexpected response from Ollama at {self.base_url}: {exc}") from exc

    def installed_models(self) -> list[str]:
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise LLMUnavailable(f"Could not list Ollama models: {exc}") from exc

    def ensure_ready(self) -> str:
        """Raise a precise, actionable error if Ollama or the model is missing. Returns Ollama's version."""
        version = self.version()
        names = self.installed_models()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise LLMModelMissing(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}")
        return version

    # ------------------------------------------------------------------ embeddings

    def embed(self, texts: list[str], *, model: str, prefix: str = "clustering: ", batch_size: int = 32) -> list[list[float]]:
        """Embed texts with a (small) local embedding model. `prefix` is the task prefix nomic-embed-text expects."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + t for t in texts[i : i + batch_size]]
            try:
                resp = self._session.post(
                    f"{self.base_url}/api/embed", json={"model": model, "input": batch, "keep_alive": "10m"}, timeout=self.timeout_s
                )
            except requests.ConnectionError as exc:
                raise LLMUnavailable(f"Cannot reach Ollama at {self.base_url} for embeddings") from exc
            except requests.Timeout as exc:
                raise LLMOutputError(f"Embedding request timed out after {self.timeout_s:.0f}s") from exc
            if resp.status_code == 404:
                raise LLMModelMissing(f"Embedding model '{model}' is not installed. Run: ollama pull {model}")
            if resp.status_code >= 400:
                raise LLMOutputError(f"Embedding request failed (HTTP {resp.status_code}): {resp.text[:200]}")
            try:
                got = resp.json()["embeddings"]
            except (ValueError, KeyError) as exc:
                raise LLMOutputError("Unexpected embedding response shape") from exc
            if len(got) != len(batch):
                raise LLMOutputError(f"Asked for {len(batch)} embeddings, received {len(got)}")
            vectors.extend(got)
        return vectors

    # ------------------------------------------------------------------ inference

    def chat_structured(self, system: str, user: str, schema: type[T]) -> tuple[T, LLMMeta]:
        """One schema-constrained completion, validated, with bounded retries."""
        started = time.monotonic()
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            # Temperature 0 is deterministic, so an identical retry would fail identically.
            temperature = min(1.0, self.temperature + 0.3 * attempt)
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": schema.model_json_schema(),
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": temperature, "num_ctx": self.num_ctx},
            }
            try:
                resp = self._session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_s)
            except requests.ConnectionError as exc:
                raise LLMUnavailable(f"Lost connection to Ollama at {self.base_url}") from exc
            except requests.Timeout:
                last_error = f"timed out after {self.timeout_s:.0f}s (a larger timeout_s or smaller model may help)"
                log.warning("Ollama attempt %d/%d: %s", attempt + 1, self.max_retries + 1, last_error)
                continue

            if resp.status_code == 404:
                raise LLMModelMissing(f"Ollama does not know model '{self.model}'. Run: ollama pull {self.model}")
            if resp.status_code >= 500:
                last_error = f"Ollama HTTP {resp.status_code}: {resp.text[:200]}"
                log.warning("Ollama attempt %d/%d: %s", attempt + 1, self.max_retries + 1, last_error)
                continue
            if resp.status_code >= 400:
                raise LLMOutputError(f"Ollama rejected the request (HTTP {resp.status_code}): {resp.text[:200]}")

            try:
                body = resp.json()
                content = body["message"]["content"]
                parsed = schema.model_validate(json.loads(content))
            except (ValueError, KeyError, TypeError, ValidationError) as exc:
                last_error = f"invalid structured output: {str(exc)[:200]}"
                log.warning("Ollama attempt %d/%d: %s", attempt + 1, self.max_retries + 1, last_error)
                continue

            return parsed, LLMMeta(
                latency_s=time.monotonic() - started,
                eval_tokens=body.get("eval_count"),
                prompt_tokens=body.get("prompt_eval_count"),
                attempts=attempt + 1,
            )

        raise LLMOutputError(f"No valid output after {self.max_retries + 1} attempts: {last_error}")
