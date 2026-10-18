"""Exception hierarchy. Each class maps to a distinct "what should the agent do?" answer."""


class PdaError(Exception):
    """Base class for all expected, handled errors."""


class RobotsDisallowed(PdaError):
    """robots.txt forbids fetching this URL. The agent must NOT retry or work around it."""

    def __init__(self, url: str, rule: str | None = None):
        self.url = url
        self.rule = rule
        detail = f" (rule: {rule})" if rule else ""
        super().__init__(f"robots.txt disallows fetching {url}{detail}")


class HttpError(PdaError):
    """A request failed permanently or exhausted its retries."""

    def __init__(self, message: str, status: int | None = None):
        self.status = status
        super().__init__(message)


class RateLimited(HttpError):
    """The server asked us to slow down for longer than we are willing to wait."""


class SourceFormatError(PdaError):
    """The source returned something that does not match the documented/observed shape."""


class LLMUnavailable(PdaError):
    """Ollama is not reachable. The agent cannot proceed with analysis."""


class LLMModelMissing(PdaError):
    """Ollama is running but the configured model has not been pulled."""


class LLMOutputError(PdaError):
    """The model answered but not with valid structured output (after retries)."""
