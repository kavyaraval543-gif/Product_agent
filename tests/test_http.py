import pytest

from helpers import FakeClock, ScriptedServer, plain_session
from pda.errors import HttpError, RateLimited, RobotsDisallowed, SourceFormatError
from pda.http import PoliteClient


@pytest.fixture
def server():
    s = ScriptedServer()
    yield s
    s.close()


def make_client(clock: FakeClock | None = None, **kw) -> tuple[PoliteClient, FakeClock]:
    clock = clock or FakeClock()
    client = PoliteClient(
        user_agent="TestAgent/1.0 (+test)", session=plain_session(), sleep=clock.sleep, clock=clock.time,
        min_interval_s=kw.pop("min_interval_s", 1.0), max_retries=kw.pop("max_retries", 3), **kw,
    )
    return client, clock


def test_fetches_json_sends_user_agent_and_checks_robots_first(server):
    server.add("/robots.txt", body="User-agent: *\nAllow: /\n")
    server.add("/api", body={"ok": True})
    client, _ = make_client()
    assert client.get_json(server.url + "/api", {"a": 1}) == {"ok": True}
    assert server.paths() == ["/robots.txt", "/api"]
    assert all(r["headers"]["User-Agent"] == "TestAgent/1.0 (+test)" for r in server.requests)


def test_robots_disallow_blocks_request_and_never_touches_the_endpoint(server):
    server.add("/robots.txt", body="User-agent: *\nDisallow: /api\n")
    server.add("/api", body={"secret": 1})
    client, _ = make_client()
    with pytest.raises(RobotsDisallowed) as exc:
        client.get_json(server.url + "/api")
    assert exc.value.rule == "/api"
    assert "/api" not in server.paths()


def test_robots_cached_per_origin(server):
    server.add("/robots.txt", body="User-agent: *\nAllow: /\n")
    server.add("/api", body={})
    client, _ = make_client()
    client.get_json(server.url + "/api")
    client.get_json(server.url + "/api")
    assert server.paths().count("/robots.txt") == 1


def test_missing_robots_means_no_restrictions(server):
    server.add("/api", body={"ok": 1})  # /robots.txt -> 404
    client, _ = make_client()
    assert client.get_json(server.url + "/api") == {"ok": 1}


def test_robots_server_error_fails_closed(server):
    server.add("/robots.txt", status=503)
    server.add("/api", body={"ok": 1})
    client, _ = make_client(max_retries=1)
    with pytest.raises(RobotsDisallowed):
        client.get_json(server.url + "/api")
    assert "/api" not in server.paths()


def test_429_with_retry_after_waits_then_succeeds(server):
    server.add("/robots.txt", body="")
    server.add("/api", status=429, headers={"Retry-After": "7"})
    server.add("/api", body={"ok": 1})
    client, clock = make_client()
    assert client.get_json(server.url + "/api") == {"ok": 1}
    assert 7.0 in clock.sleeps


def test_retry_after_longer_than_limit_raises_rate_limited_without_waiting(server):
    server.add("/robots.txt", body="")
    server.add("/api", status=429, headers={"Retry-After": "3600"})
    client, clock = make_client()
    with pytest.raises(RateLimited):
        client.get_json(server.url + "/api")
    assert 3600 not in clock.sleeps


def test_server_errors_retry_with_exponential_backoff_then_give_up(server):
    server.add("/robots.txt", body="")
    server.add("/api", status=503)
    client, clock = make_client(max_retries=2)
    with pytest.raises(HttpError) as exc:
        client.get_json(server.url + "/api")
    assert exc.value.status == 503
    assert server.paths().count("/api") == 3  # 1 try + 2 retries
    backoffs = [s for s in clock.sleeps if s >= 1.0]
    assert 1.0 in backoffs and 2.0 in backoffs


def test_non_retryable_4xx_fails_immediately(server):
    server.add("/robots.txt", body="")
    server.add("/api", status=404)
    client, _ = make_client()
    with pytest.raises(HttpError) as exc:
        client.get_json(server.url + "/api")
    assert exc.value.status == 404
    assert server.paths().count("/api") == 1


def test_invalid_json_raises_source_format_error(server):
    server.add("/robots.txt", body="")
    server.add("/api", body="<html>not json</html>")
    client, _ = make_client()
    with pytest.raises(SourceFormatError):
        client.get_json(server.url + "/api")


def test_min_interval_is_enforced_between_requests(server):
    server.add("/robots.txt", body="")
    server.add("/api", body={})
    client, clock = make_client(min_interval_s=2.0)
    client.get_json(server.url + "/api")
    client.get_json(server.url + "/api")
    # robots.txt + 2 API calls = 3 requests -> at least two throttle waits of ~2s each
    assert sum(1 for s in clock.sleeps if s > 1.5) >= 2


def test_connection_error_retries_then_raises():
    client, clock = make_client(max_retries=2)
    with pytest.raises(RobotsDisallowed):
        # Nothing listens on this port: robots.txt is unreachable -> fail closed (never fetch the data)
        client.get_json("http://127.0.0.1:9/api")
    assert len(clock.sleeps) >= 2


# ----------------------------------------------------------------------------- redirects must not bypass robots.txt


def test_same_origin_redirect_is_followed(server):
    server.add("/robots.txt", body="User-agent: *\nAllow: /\n")
    server.add("/old", status=302, headers={"Location": "/new?x=1"})
    server.add("/new", body={"moved": True})
    client, _ = make_client()
    assert client.get_json(server.url + "/old") == {"moved": True}
    assert server.paths() == ["/robots.txt", "/old", "/new"]


def test_redirect_into_a_disallowed_path_is_blocked_before_fetching_it(server):
    server.add("/robots.txt", body="User-agent: *\nDisallow: /private\n")
    server.add("/api", status=301, headers={"Location": "/private/data"})
    server.add("/private/data", body={"secret": 1})
    client, _ = make_client()
    with pytest.raises(RobotsDisallowed):
        client.get_json(server.url + "/api")
    assert "/private/data" not in server.paths()


def test_redirect_to_another_host_uses_that_hosts_own_robots_txt():
    allowed_host, blocked_host = ScriptedServer(), ScriptedServer()
    try:
        allowed_host.add("/robots.txt", body="User-agent: *\nAllow: /\n")
        allowed_host.add("/api", status=302, headers={"Location": blocked_host.url + "/data"})
        blocked_host.add("/robots.txt", body="User-agent: *\nDisallow: /\n")
        blocked_host.add("/data", body={"should": "never be fetched"})
        client, _ = make_client()
        with pytest.raises(RobotsDisallowed):
            client.get_json(allowed_host.url + "/api")
        assert "/robots.txt" in blocked_host.paths()  # we asked the second host for ITS rules...
        assert "/data" not in blocked_host.paths()  # ...and obeyed them
    finally:
        allowed_host.close()
        blocked_host.close()


def test_redirect_loop_is_cut_off(server):
    server.add("/robots.txt", body="")
    server.add("/a", status=302, headers={"Location": "/b"})
    server.add("/b", status=302, headers={"Location": "/a"})
    client, _ = make_client()
    with pytest.raises(HttpError, match="Too many redirects"):
        client.get_json(server.url + "/a")
