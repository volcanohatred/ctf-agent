"""web_fetch egress guard tests.

The previous implementation matched the hostname against string prefixes, so
`http://2130706433:2375/` (decimal IP), `http://0x7f.1/`, raw IPv6 literals and
any DNS name pointing at the host all walked straight past it. Since the agent
container holds /var/run/docker.sock, that was a path from a solver to the
Docker API. These tests pin the resolution-based rule down.

Every case uses a numeric literal or a name that resolves locally, so the suite
needs no external DNS.
"""

from __future__ import annotations

import pytest

from backend.tools.core import _ip_is_forbidden, _is_internal_url, resolve_target

PRIVATE_TARGETS = [
    "http://127.0.0.1:8000/api",
    "http://127.0.0.1/",
    "http://10.0.0.5/",
    "http://192.168.1.1/admin",
    "http://172.17.0.1:2375/containers/json",   # docker bridge gateway
    "http://172.16.0.1/",
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://0.0.0.0/",
    "http://[::1]/",
    "http://[fe80::1]/",
    "http://localhost:8000/",
]


@pytest.mark.parametrize("url", PRIVATE_TARGETS)
def test_private_addresses_are_blocked(url: str) -> None:
    allowed, reason, _ips = resolve_target(url)
    assert not allowed, f"{url} was allowed"
    assert reason


@pytest.mark.parametrize("url", [
    "http://2130706433/",        # 127.0.0.1 as a decimal integer
    "http://2130706433:2375/",   # ...aimed at the Docker API
    "http://0x7f000001/",        # hex form
    "http://0177.0.0.1/",        # octal form
])
def test_nonstandard_ip_literal_forms_are_blocked(url: str) -> None:
    """A prefix blocklist cannot see these; resolution can.

    If the platform refuses to resolve the form at all, that is also a block —
    either outcome is acceptable, silently allowing it is not.
    """
    allowed, _reason, _ips = resolve_target(url)
    assert not allowed, f"{url} was allowed"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://internal/", "gopher://x/", ""])
def test_non_http_schemes_are_blocked(url: str) -> None:
    allowed, reason, _ = resolve_target(url)
    assert not allowed
    assert "scheme" in reason or "host" in reason


def test_public_address_is_allowed() -> None:
    allowed, reason, ips = resolve_target("http://8.8.8.8/dns")
    assert allowed, reason
    assert ips == ["8.8.8.8"]


def test_allowlist_is_hostname_based_and_narrow() -> None:
    """host.docker.internal is how challenge services are reached from the agent
    container; it resolves privately, so it needs an explicit exception."""
    allowed, reason, _ = resolve_target("http://127.0.0.1:31337/", allow_hosts="127.0.0.1")
    assert allowed, reason

    # The exception must not widen to other private targets.
    blocked, _, _ = resolve_target("http://10.0.0.5/", allow_hosts="127.0.0.1")
    assert not blocked


def test_allowlist_accepts_a_comma_separated_string() -> None:
    allowed, reason, _ = resolve_target(
        "http://127.0.0.1:31337/", allow_hosts="example.invalid, 127.0.0.1"
    )
    assert allowed, reason


def test_allowlist_is_case_and_dot_insensitive() -> None:
    allowed, reason, _ = resolve_target("http://LOCALHOST./", allow_hosts="LocalHost")
    assert allowed, reason


def test_ip_forbidden_predicate() -> None:
    import ipaddress

    for addr in ("127.0.0.1", "10.1.2.3", "192.168.0.1", "172.16.5.5",
                 "169.254.1.1", "0.0.0.0", "::1", "fe80::1", "224.0.0.1"):
        assert _ip_is_forbidden(ipaddress.ip_address(addr)), addr
    for addr in ("8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"):
        assert not _ip_is_forbidden(ipaddress.ip_address(addr)), addr


def test_legacy_predicate_still_works() -> None:
    assert _is_internal_url("http://127.0.0.1/") is True
    assert _is_internal_url("http://8.8.8.8/") is False


def test_unresolvable_host_is_blocked() -> None:
    allowed, reason, _ = resolve_target("http://this-host-does-not-exist.invalid/")
    assert not allowed
    assert "DNS" in reason or "resolve" in reason


@pytest.mark.asyncio
async def test_do_web_fetch_reports_the_reason(monkeypatch) -> None:
    from backend.tools import core

    async def fail(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("a blocked URL must never be requested")

    monkeypatch.setattr(core.httpx, "AsyncClient", fail)
    out = await core.do_web_fetch("http://172.17.0.1:2375/_ping")
    assert out.startswith("Fetch error: blocked")
    assert "non-public" in out


@pytest.mark.asyncio
async def test_do_web_fetch_allows_listed_host(monkeypatch) -> None:
    """The guard must not break fetching a challenge service on the host."""
    from backend.tools import core

    calls = []

    class FakeResponse:
        status_code = 200
        reason_phrase = "OK"
        text = "flag{ok}"
        is_redirect = False
        headers: dict = {}

    class FakeClient:
        def __init__(self, *a, **k) -> None:
            assert k.get("follow_redirects") is False, "redirects must stay disabled"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, method, url, **kwargs):
            calls.append((method, url))
            return FakeResponse()

    monkeypatch.setattr(core.httpx, "AsyncClient", FakeClient)
    out = await core.do_web_fetch("http://127.0.0.1:31337/", allow_hosts="127.0.0.1")
    assert "flag{ok}" in out
    assert calls == [("GET", "http://127.0.0.1:31337/")]
