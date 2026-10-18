"""Shared test doubles. No network: HTTP behaviour is tested against a local scripted server."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import requests

from pda.llm import LLMMeta, OllamaClient
from pda.models import RawReview, ReviewAnalysisOut
from pda.sources.base import SourcePage, SourceSnapshot


class ScriptedServer:
    """Serves queued responses per path. Last response for a path repeats forever."""

    def __init__(self) -> None:
        self.routes: dict[str, list[tuple[int, dict[str, str], bytes]]] = {}
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                outer.requests.append(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": raw}
                )
                script = outer.routes.get(self.path.split("?")[0])
                if not script:
                    status, headers, body = 404, {}, b"not found"
                else:
                    status, headers, body = script.pop(0) if len(script) > 1 else script[0]
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _serve
            do_POST = _serve

            def log_message(self, *args: Any) -> None:  # silence
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def add(self, path: str, status: int = 200, body: bytes | str | dict = b"", headers: dict[str, str] | None = None) -> None:
        if isinstance(body, dict):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        self.routes.setdefault(path, []).append((status, headers or {}, body))

    def paths(self) -> list[str]:
        return [r["path"].split("?")[0] for r in self.requests]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def plain_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # never go through a proxy in tests
    return s


class FakeClock:
    """Deterministic clock + sleep, so retry/throttle logic is tested without waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


T0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


def make_review(i: int, *, body: str | None = None, rating: int | None = 2, days_ago: float | None = None, version: str | None = "1.0") -> RawReview:
    """Review #i. Higher i = newer (so a newest-first list is sorted by descending i)."""
    when = T0 + timedelta(days=i) if days_ago is None else T0 - timedelta(days=days_ago)
    return RawReview(
        source="fake",
        source_review_id=str(i),
        rating=rating,
        body=body if body is not None else f"Autofill stopped working on several sites after the latest update number {i}",
        review_date=when,
        app_version=version,
        url=f"https://example.test/reviews/{i}",
    )


@dataclass
class FakeSource:
    """A source over a fixed newest-first list of reviews, `page_size` per page."""

    reviews: list[RawReview]
    page_size: int = 5
    name: str = "fake"
    ordered: bool = True
    fail_on_page: int | None = None
    fail_with: Exception | None = None
    snapshot: SourceSnapshot | None = field(default_factory=lambda: SourceSnapshot(100, {"1": 10, "2": 5, "3": 5, "4": 10, "5": 70}))
    pages_served: int = 0

    def iter_pages(self, max_pages: int) -> Iterator[SourcePage]:
        total_pages = max(1, -(-len(self.reviews) // self.page_size))
        for n in range(1, min(max_pages, total_pages) + 1):
            if self.fail_on_page == n and self.fail_with is not None:
                raise self.fail_with
            chunk = self.reviews[(n - 1) * self.page_size : n * self.page_size]
            self.pages_served += 1
            yield SourcePage(reviews=chunk, page_number=n, has_next=n < total_pages, ordered_newest_first=self.ordered)


class FakeLLM(OllamaClient):
    """OllamaClient whose model is a Python function. Records every call."""

    def __init__(self, responder=None, model: str = "fake-model:1b") -> None:
        super().__init__(model=model)
        self.responder = responder or default_responder
        self.calls: list[str] = []
        self.ready_error: Exception | None = None
        self.embedder = None  # callable(text) -> list[float]; None = embeddings unavailable
        self.embed_calls: list[list[str]] = []

    def ensure_ready(self) -> str:
        if self.ready_error:
            raise self.ready_error
        return "test"

    def embed(self, texts, *, model, prefix="clustering: ", batch_size=32):  # type: ignore[override]
        if self.embedder is None:
            from pda.errors import LLMModelMissing

            raise LLMModelMissing(f"Embedding model '{model}' is not installed. Run: ollama pull {model}")
        self.embed_calls.append(list(texts))
        return [self.embedder(t) for t in texts]

    def chat_structured(self, system, user, schema):  # type: ignore[override]
        self.calls.append(user)
        out = self.responder(user)
        return out, LLMMeta(latency_s=0.5, eval_tokens=50, prompt_tokens=200, attempts=1)


def default_responder(user: str) -> ReviewAnalysisOut:
    body = user.split("<<<REVIEW\n", 1)[1].rsplit("\nREVIEW>>>", 1)[0]
    return ReviewAnalysisOut(
        relevance="genuine_feedback",
        sentiment="negative",
        reports_problem=True,
        theme="bugs_reliability",
        severity="medium",
        issue_label="autofill stops working",
        problem_statement="Autofill stopped working.",
        feature_request="",
        supporting_quote=body[:40],  # verbatim prefix -> verifies
    )
