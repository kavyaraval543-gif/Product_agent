"""A small RFC 9309 robots.txt implementation.

Why not urllib.robotparser? Measured on 2026-10-17: on Python 3.11 and 3.12 it ignores the `*` and `$`
wildcards that real sites use (Apple's robots.txt relies on `/*/rss/*`) and applies the FIRST matching
rule instead of the longest, so it wrongly reports Apple's review feed as "allowed". Python 3.13 fixed
this. This project supports 3.11+, so it ships its own small parser to behave identically everywhere.

Semantics implemented (RFC 9309):
  * Pick the group whose User-agent token matches ours; otherwise the `*` group.
  * Among matching Allow/Disallow rules the LONGEST pattern wins; on a tie, Allow wins.
  * `*` matches any characters, a trailing `$` anchors the end of the path.
  * Empty `Disallow:` means "nothing is disallowed".
Availability rules (applied by the HTTP client, see http.py):
  * 2xx -> parse.   4xx -> no restrictions.   5xx / network failure -> assume everything disallowed.
Not implemented: percent-encoding normalisation (rare; documented limitation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    allow: bool
    pattern: str

    def matches(self, path: str) -> bool:
        return _compile(self.pattern).match(path) is not None


_cache: dict[str, re.Pattern[str]] = {}


def _compile(pattern: str) -> re.Pattern[str]:
    cached = _cache.get(pattern)
    if cached is None:
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        regex = ".*".join(re.escape(part) for part in body.split("*"))
        cached = re.compile(regex + ("$" if anchored else ""))
        _cache[pattern] = cached
    return cached


class RobotsRules:
    def __init__(self, rules: list[Rule], allow_all: bool = False, disallow_all: bool = False):
        self.rules = rules
        self.allow_all = allow_all
        self.disallow_all = disallow_all

    @classmethod
    def permissive(cls) -> "RobotsRules":
        return cls([], allow_all=True)

    @classmethod
    def forbidding(cls) -> "RobotsRules":
        return cls([], disallow_all=True)

    def decide(self, path: str) -> tuple[bool, Rule | None]:
        """Return (allowed, deciding_rule). `path` includes the query string."""
        if self.disallow_all:
            return False, Rule(False, "/ (robots.txt unavailable: assuming disallow)")
        if self.allow_all:
            return True, None
        best: Rule | None = None
        for rule in self.rules:
            if not rule.matches(path):
                continue
            if best is None or len(rule.pattern) > len(best.pattern) or (
                len(rule.pattern) == len(best.pattern) and rule.allow and not best.allow
            ):
                best = rule
        if best is None:
            return True, None
        return best.allow, best

    def is_allowed(self, path: str) -> bool:
        return self.decide(path)[0]


def parse_robots(text: str, product_token: str) -> RobotsRules:
    """Parse robots.txt text into the rules that apply to `product_token`."""
    groups: list[tuple[list[str], list[Rule]]] = []
    agents: list[str] = []
    rules: list[Rule] = []
    in_rules = False

    def flush() -> None:
        nonlocal agents, rules, in_rules
        if agents:
            groups.append((agents, rules))
        agents, rules, in_rules = [], [], False

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, _, value = line.partition(":")
        field, value = field.strip().lower(), value.strip()
        if field == "user-agent":
            if in_rules:
                flush()
            agents.append(value.lower())
        elif field in ("allow", "disallow"):
            in_rules = True
            if value:  # empty Disallow == allow everything; empty Allow is a no-op
                rules.append(Rule(allow=(field == "allow"), pattern=value))
    flush()

    token = product_token.lower()
    specific = [r for names, r in groups if any(n != "*" and n in token for n in names)]
    chosen = specific or [r for names, r in groups if "*" in names]
    merged = [rule for group in chosen for rule in group]
    return RobotsRules(merged)
