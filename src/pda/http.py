"""A deliberately polite HTTP client.

Every outbound request to a data source goes through here so the rules live in ONE place:
  * robots.txt is fetched once per origin and checked before every request (not optional).
  * A minimum interval between requests (rate limiting).
  * Timeouts, bounded retries with exponential backoff, Retry-After honoured.
  * An identifying User-Agent.
  * A response size cap and JSON validation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable
from urllib.parse import urljoin, urlsplit

import requests

from pda.errors import HttpError, RateLimited, RobotsDisallowed, SourceFormatError
from pda.robots import RobotsRules, parse_robots

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
REDIRECT_STATUS = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 3
PRODUCT_TOKEN = "ProductDiscoveryAgent"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class PoliteClient:
    def __init__(
        self,
        *,
        user_agent: str,
        min_interval_s: float = 1.0,
        timeout_s: float = 20.0,
        max_retries: int = 3,
        max_backoff_s: float = 30.0,
        max_retry_after_s: float = 60.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_backoff_s = max_backoff_s
        self.max_retry_after_s = max_retry_after_s
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None
        self._robots: dict[str, RobotsRules] = {}
        self.request_count = 0  # exposed for run statistics

    # ------------------------------------------------------------------ robots.txt

    def _rules_for(self, origin: str) -> RobotsRules:
        if origin in self._robots:
            return self._robots[origin]
        robots_url = f"{origin}/robots.txt"
        try:
            resp = self._send(robots_url, None, check_robots=False)
        except HttpError as exc:
            if exc.status is not None and 400 <= exc.status < 500:
                rules = RobotsRules.permissive()  # RFC 9309: robots.txt "unavailable" (4xx) -> no restrictions
                log.info("robots.txt not found at %s (HTTP %s): treating as no restrictions", origin, exc.status)
            else:
                rules = RobotsRules.forbidding()  # 5xx / network failure -> assume disallow, fail closed
                log.warning("robots.txt unreachable at %s (%s): assuming everything is disallowed", origin, exc)
        else:
            rules = parse_robots(resp.text, PRODUCT_TOKEN)
        self._robots[origin] = rules
        return rules

    def check_allowed(self, url: str) -> None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        path = parts.path or "/"
        if parts.query:
            path += "?" + parts.query
        allowed, rule = self._rules_for(origin).decide(path)
        if not allowed:
            raise RobotsDisallowed(url, rule.pattern if rule else None)

    # ------------------------------------------------------------------ requests

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            wait = self.min_interval_s - (self._clock() - self._last_request_at)
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._clock()

    def _backoff(self, attempt: int) -> float:
        return min(self.max_backoff_s, 2.0**attempt)

    def _send(self, url: str, params: dict[str, Any] | None, *, check_robots: bool = True) -> requests.Response:
        """GET with retries. Redirects are followed MANUALLY so every hop is robots-checked (requests would
        silently follow a redirect to another host and skip that host's robots.txt)."""
        redirects = 0
        attempt = 0
        while True:
            self._throttle()
            self.request_count += 1
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    headers={"User-Agent": self.user_agent, "Accept": "application/json, text/plain;q=0.8"},
                    timeout=self.timeout_s,
                    allow_redirects=False,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise HttpError(f"Network error after {attempt + 1} attempts for {url}: {exc}") from exc
                delay = self._backoff(attempt)
                log.warning("Network error (%s); retry %d/%d in %.1fs", exc, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
                attempt += 1
                continue

            if 200 <= resp.status_code < 300:
                if len(resp.content) > MAX_RESPONSE_BYTES:
                    raise HttpError(f"Response from {url} exceeds {MAX_RESPONSE_BYTES} bytes", resp.status_code)
                return resp

            if resp.status_code in REDIRECT_STATUS and resp.headers.get("Location"):
                redirects += 1
                if redirects > MAX_REDIRECTS:
                    raise HttpError(f"Too many redirects (> {MAX_REDIRECTS}) starting from {url}", resp.status_code)
                url = urljoin(resp.url or url, resp.headers["Location"])
                params = None  # the Location already carries the final query string
                if check_robots:
                    self.check_allowed(url)  # a redirect to another path/host needs its own robots.txt decision
                log.info("Following redirect to %s", url)
                continue

            if resp.status_code in RETRYABLE_STATUS:
                retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                if retry_after is not None and retry_after > self.max_retry_after_s:
                    raise RateLimited(
                        f"{url} asked us to wait {retry_after:.0f}s (> {self.max_retry_after_s:.0f}s limit)",
                        resp.status_code,
                    )
                if attempt >= self.max_retries:
                    cls = RateLimited if resp.status_code == 429 else HttpError
                    raise cls(f"HTTP {resp.status_code} from {url} after {attempt + 1} attempts", resp.status_code)
                delay = retry_after if retry_after is not None else self._backoff(attempt)
                log.warning("HTTP %d from %s; retry %d/%d in %.1fs", resp.status_code, url, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
                attempt += 1
                continue

            raise HttpError(f"HTTP {resp.status_code} from {url}", resp.status_code)

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET `url` as JSON after a robots.txt check. Raises on any problem."""
        prepared = requests.Request("GET", url, params=params).prepare()
        self.check_allowed(prepared.url or url)
        resp = self._send(url, params)
        try:
            return resp.json()
        except ValueError as exc:
            raise SourceFormatError(f"{url} did not return valid JSON") from exc


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After may be seconds or an HTTP date; only the seconds form is honoured."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
