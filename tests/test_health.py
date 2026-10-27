"""Startup health-check tests — no network, httpx transport is faked.

The point of the probe is to turn "hours of silently empty rounds" into a
two-second error message, so what matters is the classification: which failures
are fatal, which are warnings, and which mean "not configured".
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from backend.health import (
    HealthCheckError,
    HealthReport,
    ProbeResult,
    check_health,
    classify_status,
    classify_transport_error,
    ensure_healthy,
    probe_anthropic,
    probe_openai,
)


def _settings(**over) -> SimpleNamespace:
    base = dict(
        anthropic_base_url="http://gw.invalid/anthropic",
        anthropic_auth_token="tok",
        gateway_base_url="http://gw.invalid/v1",
        gateway_api_key="key",
        summarizer_model="deepseek-v4-flash",
        health_check_timeout_s=1.0,
        startup_health_check=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    """Route httpx.AsyncClient through a MockTransport.

    The replacement must stay a CLASS, not a factory function: pydantic-ai does
    `httpx.AsyncClient | httpx2.AsyncClient` at import time, and a function on
    the left of `|` raises TypeError. Subclassing also keeps isinstance checks
    and the async-context-manager protocol intact.
    """

    def _apply(handler):
        transport = _mock_transport(handler)
        real = httpx.AsyncClient

        class PatchedClient(real):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)

    return _apply


# ---------------------------------------------------------------------------
# Transport classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "Connection refused",
    "[Errno 111] Connection refused",
    "getaddrinfo ENOTFOUND gateway.internal",
    "Name or service not known",
    "Connection reset by peer",
    "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed",
])
def test_transport_errors_are_unreachable(message: str) -> None:
    kind, detail = classify_transport_error(httpx.ConnectError(message))
    assert kind == "unreachable"
    assert detail


def test_transport_timeout_is_its_own_kind() -> None:
    kind, _ = classify_transport_error(httpx.ConnectTimeout("timed out"))
    assert kind == "timeout"


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected_ok,expected_kind", [
    (200, True, "ok"),
    (401, False, "auth"),
    (403, False, "auth"),
    (404, False, "bad_response"),   # reachable host, unexpected path — not fatal
    (429, False, "rate_limited"),
    (500, False, "server_error"),
    (502, False, "server_error"),
    (529, False, "rate_limited"),
])
def test_classify_status(status: int, expected_ok: bool, expected_kind: str) -> None:
    ok, kind, _detail = classify_status(status, "body")
    assert ok is expected_ok
    assert kind == expected_kind


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_anthropic_probe_ok(patch_client) -> None:
    patch_client(lambda req: httpx.Response(200, json={"content": []}))
    result = await probe_anthropic("http://gw/anthropic", "tok", "kimi-k3")
    assert result.ok and result.kind == "ok"
    assert result.url.endswith("/v1/messages")


@pytest.mark.asyncio
async def test_anthropic_probe_sends_both_auth_styles(patch_client) -> None:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(dict(req.headers))
        return httpx.Response(200, json={})

    patch_client(handler)
    await probe_anthropic("http://gw/anthropic", "sekret", "kimi-k3")
    assert seen.get("authorization") == "Bearer sekret"
    assert seen.get("x-api-key") == "sekret"
    assert seen.get("anthropic-version")


@pytest.mark.asyncio
async def test_anthropic_probe_connection_refused_is_fatal(patch_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    patch_client(handler)
    result = await probe_anthropic("http://gw/anthropic", "tok", "kimi-k3")
    assert not result.ok and result.kind == "unreachable"


@pytest.mark.asyncio
async def test_bad_credentials_are_fatal(patch_client) -> None:
    patch_client(lambda req: httpx.Response(401, text="invalid api key"))
    result = await probe_anthropic("http://gw/anthropic", "wrong", "kimi-k3")
    assert not result.ok and result.kind == "auth"


@pytest.mark.asyncio
async def test_rate_limit_is_a_warning_not_fatal(patch_client) -> None:
    patch_client(lambda req: httpx.Response(429, text="slow down"))
    result = await probe_anthropic("http://gw/anthropic", "tok", "kimi-k3")
    assert not result.ok and result.kind == "rate_limited"
    report = HealthReport(probes=[result])
    assert report.ok, "a rate limit must not block startup"
    assert report.warnings and not report.fatal


@pytest.mark.asyncio
async def test_unconfigured_endpoints_are_not_failures() -> None:
    a = await probe_anthropic("", "", "kimi-k3")
    g = await probe_openai("", "", "kimi-k3")
    assert a.ok and a.kind == "not_configured"
    assert g.ok and g.kind == "not_configured"


@pytest.mark.asyncio
async def test_openai_probe_hits_chat_completions(patch_client) -> None:
    urls = []

    def handler(req: httpx.Request) -> httpx.Response:
        urls.append(str(req.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    patch_client(handler)
    result = await probe_openai("http://gw/v1/", "key", "deepseek-v4-flash")
    assert result.ok
    assert urls == ["http://gw/v1/chat/completions"]


# ---------------------------------------------------------------------------
# Aggregation / fail-fast
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ensure_healthy_raises_on_unreachable_gateway(patch_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    patch_client(handler)
    with pytest.raises(HealthCheckError) as ei:
        await ensure_healthy(_settings())
    text = str(ei.value)
    assert "unreachable" in text
    assert ei.value.report.fatal


@pytest.mark.asyncio
async def test_ensure_healthy_passes_when_the_gateway_answers(patch_client) -> None:
    patch_client(lambda req: httpx.Response(200, json={"content": []}))
    report = await ensure_healthy(_settings())
    assert report.ok
    assert all(p.ok for p in report.probes)


@pytest.mark.asyncio
async def test_ensure_healthy_can_be_disabled(patch_client) -> None:
    """STARTUP_HEALTH_CHECK=false must skip the probe entirely."""
    def handler(req: httpx.Request) -> httpx.Response:
        raise AssertionError("probe must not run when disabled")

    patch_client(handler)
    report = await ensure_healthy(_settings(startup_health_check=False))
    assert report.probes == []


@pytest.mark.asyncio
async def test_check_health_never_raises(patch_client) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    patch_client(handler)
    report = await check_health(_settings())
    assert not report.ok
    assert len(report.probes) == 2


@pytest.mark.asyncio
async def test_transient_timeout_is_retried_and_recovers(patch_client, monkeypatch) -> None:
    """A busy local gateway must not fail the whole run on one slow probe.

    Found by real end-to-end testing: the same endpoint timed out on one attempt
    and answered fine minutes later, so a single-shot probe blocked a run that
    would otherwise have worked.
    """
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if not str(req.url).endswith("/v1/messages"):
            return httpx.Response(200, json={})   # the gateway probe
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("ReadTimeout")
        return httpx.Response(200, json={"content": []})

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=3))
    anthropic = report.probes[0]
    assert anthropic.ok, f"gave up after {calls['n']} attempts: {anthropic}"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retries_exhausted_reports_the_attempt_count(patch_client, monkeypatch) -> None:
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if not str(req.url).endswith("/v1/messages"):
            return httpx.Response(200, json={})
        calls["n"] += 1
        raise httpx.ReadTimeout("ReadTimeout")

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=3))
    anthropic = report.probes[0]
    assert not anthropic.ok
    assert anthropic.kind == "timeout"
    assert "after 3 attempts" in anthropic.detail, anthropic.detail
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_timeout_is_a_warning_not_a_blocker(patch_client, monkeypatch) -> None:
    """A hung gateway must not veto the whole run.

    Observed live: the local gateway hangs for minutes at a time, then answers in
    ~1.2s. Treating a slow probe as fatal blocked runs that would have worked.
    The swarm's own MAX_INFRA_FAILURES guard bounds the damage instead, and it
    records a reason — so proceeding is safe and strictly more useful.
    """
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/v1/messages"):
            raise httpx.ReadTimeout("ReadTimeout")
        return httpx.Response(200, json={})

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=3))

    assert report.probes[0].kind == "timeout"
    assert report.ok, "a timeout must not be fatal"
    assert not report.fatal
    assert report.warnings


@pytest.mark.asyncio
async def test_connection_refused_stays_fatal(patch_client, monkeypatch) -> None:
    """The original incident: nothing listening at all. Must still hard-stop."""
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=2))

    assert report.probes[0].kind == "unreachable"
    assert report.fatal and not report.ok


@pytest.mark.asyncio
async def test_ensure_healthy_proceeds_with_a_warning_on_timeout(patch_client, monkeypatch) -> None:
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/v1/messages"):
            raise httpx.ReadTimeout("ReadTimeout")
        return httpx.Response(200, json={})

    patch_client(handler)
    report = await ensure_healthy(_settings(health_check_retries=2))
    assert report.ok
    assert report.warnings, "the timeout must still be surfaced loudly"


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried(patch_client, monkeypatch) -> None:
    """Bad credentials do not improve on their own; retrying just wastes time."""
    slept: list[float] = []

    async def spy(seconds):
        slept.append(seconds)

    monkeypatch.setattr("backend.health.asyncio.sleep", spy)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if not str(req.url).endswith("/v1/messages"):
            return httpx.Response(200, json={})
        calls["n"] += 1
        return httpx.Response(401, text="invalid api key")

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=5))
    assert report.probes[0].kind == "auth"
    assert calls["n"] == 1, "an auth failure was retried"
    assert slept == []


@pytest.mark.asyncio
async def test_retries_can_be_disabled(patch_client, monkeypatch) -> None:
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if not str(req.url).endswith("/v1/messages"):
            return httpx.Response(200, json={})
        calls["n"] += 1
        raise httpx.ConnectError("Connection refused")

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=1))
    assert not report.ok
    assert calls["n"] == 1


async def _instant_sleep(_seconds):
    """Keep retry-backoff tests fast without touching the real clock."""


@pytest.mark.asyncio
async def test_auxiliary_gateway_failure_does_not_block_a_runnable_session(patch_client, monkeypatch) -> None:
    """The two endpoints are not equally important.

    The Anthropic-protocol endpoint drives every default claude-sdk solver; the
    OpenAI-compatible one is only the quota fallback and the summarizer backend.
    Observed live: the primary probed `ok` while the auxiliary failed DNS inside
    the container, and the whole run was vetoed even though it would have solved
    challenges normally — losing only the fallback and the distilled journal.
    """
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/v1/messages"):
            return httpx.Response(200, json={"content": []})     # primary: healthy
        raise httpx.ConnectError("No address associated with hostname")  # aux: DNS

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=2))

    primary, aux = report.probes
    assert primary.ok and primary.critical
    assert not aux.ok and aux.critical is False
    assert report.ok, "an auxiliary endpoint vetoed a runnable session"
    assert report.warnings and not report.fatal


@pytest.mark.asyncio
async def test_primary_failure_is_still_fatal_even_when_auxiliary_is_fine(patch_client, monkeypatch) -> None:
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/v1/messages"):
            raise httpx.ConnectError("Connection refused")        # primary: dead
        return httpx.Response(200, json={"choices": []})          # aux: healthy

    patch_client(handler)
    report = await check_health(_settings(health_check_retries=2))

    assert not report.ok
    assert [p.name for p in report.fatal] == ["anthropic"]


@pytest.mark.asyncio
async def test_ensure_healthy_proceeds_when_only_the_auxiliary_endpoint_is_down(patch_client, monkeypatch) -> None:
    monkeypatch.setattr("backend.health.asyncio.sleep", _instant_sleep)

    def handler(req: httpx.Request) -> httpx.Response:
        if str(req.url).endswith("/v1/messages"):
            return httpx.Response(200, json={"content": []})
        raise httpx.ConnectError("No address associated with hostname")

    patch_client(handler)
    report = await ensure_healthy(_settings(health_check_retries=2))
    assert report.ok
    assert report.warnings, "the degraded fallback must still be surfaced loudly"


# ---------------------------------------------------------------------------
# The hint must match the actual failure
# ---------------------------------------------------------------------------

def _fatal_report(url: str, kind: str, detail: str, status: int | None = None) -> HealthReport:
    return HealthReport(probes=[ProbeResult(
        name="anthropic", url=url, ok=False, kind=kind, detail=detail, status=status,
    )])


def test_hint_for_loopback_addressing() -> None:
    text = str(HealthCheckError(_fatal_report(
        "http://127.0.0.1:8090/anthropic/v1/messages", "unreachable", "Connection refused")))
    assert "host.docker.internal" in text
    assert "DNS" not in text, "a loopback problem was diagnosed as DNS"


def test_hint_for_container_dns_failure() -> None:
    """A fixed hint told operators to fix loopback addressing when the real
    problem was container DNS for an external hostname — sending them to edit the
    wrong thing."""
    text = str(HealthCheckError(_fatal_report(
        "https://gw.example.com/v1/chat/completions", "unreachable",
        "[Errno -5] No address associated with hostname")))
    assert "DNS" in text
    assert "host.docker.internal instead of 127.0.0.1" not in text


def test_hint_for_bad_credentials() -> None:
    text = str(HealthCheckError(_fatal_report(
        "http://host.docker.internal:8090/anthropic/v1/messages", "auth",
        "invalid api key", status=401)))
    assert "credentials" in text.lower()
    assert "ANTHROPIC_AUTH_TOKEN" in text
    assert "host.docker.internal instead of 127.0.0.1" not in text


def test_hint_for_nothing_listening() -> None:
    text = str(HealthCheckError(_fatal_report(
        "http://host.docker.internal:8090/anthropic/v1/messages", "unreachable",
        "Connection refused")))
    assert "nothing answered" in text


def test_hint_always_mentions_the_bypass() -> None:
    text = str(HealthCheckError(_fatal_report("http://x/v1/messages", "unreachable", "refused")))
    assert "STARTUP_HEALTH_CHECK=false" in text


def test_hint_covers_every_fatal_probe() -> None:
    report = HealthReport(probes=[
        ProbeResult(name="anthropic", url="http://127.0.0.1:8090/v1/messages",
                    ok=False, kind="unreachable", detail="Connection refused"),
        ProbeResult(name="anthropic2", url="https://gw.example.com/v1/messages",
                    ok=False, kind="auth", detail="bad key", status=401),
    ])
    text = str(HealthCheckError(report))
    assert "anthropic:" in text and "anthropic2:" in text


def test_report_render_is_readable() -> None:
    report = HealthReport(probes=[
        ProbeResult(name="anthropic", url="http://gw/v1/messages", ok=False,
                    kind="unreachable", detail="Connection refused"),
        ProbeResult(name="gateway", url="", ok=True, kind="not_configured"),
    ])
    text = report.render()
    assert "anthropic" in text and "Connection refused" in text
    assert "[FATAL]" in text and "[ok]" in text


def test_report_render_is_ascii_safe() -> None:
    """The message is printed on startup, so it must survive a CJK console.

    A Windows agent on cp936 raised UnicodeEncodeError on the ✓/✗ markers —
    turning a clear "your gateway is down" report into a crash in the very code
    path that was supposed to explain the failure.
    """
    report = HealthReport(probes=[
        ProbeResult(name="anthropic", url="http://gw/v1/messages", ok=False,
                    kind="unreachable", detail="Connection refused"),
        ProbeResult(name="gateway", url="http://gw/v1", ok=False, kind="rate_limited"),
    ])
    text = report.render()
    text.encode("ascii")  # must not raise
    # and it must still encode on a cp936 console
    text.encode("gbk")


# ---------------------------------------------------------------------------
# Startup ordering — the probe must not be skippable by an earlier branch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preflight_runs_the_health_check_when_cleanup_is_disabled(monkeypatch):
    """Regression: CLEANUP_ORPHANS_ON_START=false used to `return` early and
    silently skip the gateway probe — disabling one safeguard disabled the other.
    """
    from backend import cli

    probed: list[bool] = []

    async def fake_ensure_healthy(settings, probe_model=""):
        probed.append(True)
        return HealthReport()

    async def fake_cleanup_containers(*_a, **_k):
        return 0

    monkeypatch.setattr(cli, "_preflight", cli._preflight)  # keep the real one
    monkeypatch.setattr("backend.health.ensure_healthy", fake_ensure_healthy)
    monkeypatch.setattr("backend.sandbox.cleanup_orphan_containers", fake_cleanup_containers)
    monkeypatch.setattr("backend.sandbox.cleanup_orphan_workspaces", lambda *a, **k: 0)
    monkeypatch.setattr("backend.tracing.prune_old_traces", lambda *a, **k: 0)

    settings = _settings(startup_health_check=True)
    settings.cleanup_orphans_on_start = False
    settings.max_concurrent_challenges = 1

    await cli._preflight(settings, ["claude-sdk/kimi-k3"])
    assert probed, "health check was skipped when orphan cleanup was disabled"


@pytest.mark.asyncio
async def test_preflight_still_cleans_up_when_enabled(monkeypatch) -> None:
    from backend import cli

    calls: list[str] = []

    async def fake_ensure_healthy(settings, probe_model=""):
        calls.append("health")
        return HealthReport()

    async def fake_cleanup_containers(*_a, **_k):
        calls.append("containers")
        return 0

    monkeypatch.setattr("backend.health.ensure_healthy", fake_ensure_healthy)
    monkeypatch.setattr("backend.sandbox.cleanup_orphan_containers", fake_cleanup_containers)
    monkeypatch.setattr("backend.sandbox.cleanup_orphan_workspaces",
                        lambda *a, **k: calls.append("workspaces") or 0)
    monkeypatch.setattr("backend.tracing.prune_old_traces",
                        lambda *a, **k: calls.append("prune") or 0)

    settings = _settings(startup_health_check=True)
    settings.cleanup_orphans_on_start = True
    settings.max_concurrent_challenges = 1

    await cli._preflight(settings, ["claude-sdk/kimi-k3"])
    assert "containers" in calls and "workspaces" in calls
    # health check must run LAST, so cleanup can never short-circuit it
    assert calls[-1] == "health", calls


@pytest.mark.asyncio
async def test_preflight_exits_nonzero_on_a_dead_gateway(monkeypatch) -> None:
    from backend import cli

    async def boom(settings, probe_model=""):
        raise HealthCheckError(HealthReport(probes=[
            ProbeResult(name="anthropic", url="http://gw/v1/messages", ok=False,
                        kind="unreachable", detail="Connection refused"),
        ]))

    monkeypatch.setattr("backend.health.ensure_healthy", boom)
    monkeypatch.setattr("backend.sandbox.cleanup_orphan_containers",
                        lambda *a, **k: _never())
    monkeypatch.setattr("backend.sandbox.cleanup_orphan_workspaces", lambda *a, **k: 0)
    monkeypatch.setattr("backend.tracing.prune_old_traces", lambda *a, **k: 0)

    settings = _settings(startup_health_check=True)
    settings.cleanup_orphans_on_start = False
    settings.max_concurrent_challenges = 1

    with pytest.raises(SystemExit) as ei:
        await cli._preflight(settings, ["claude-sdk/kimi-k3"])
    assert ei.value.code == 2


async def _never():  # pragma: no cover - must not be called
    raise AssertionError("cleanup must not run when disabled")
