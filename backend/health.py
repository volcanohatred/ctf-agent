"""Startup health check — fail fast when the model gateway is unreachable.

The single most expensive failure mode this project has hit: the gateway is
down, the Claude CLI reports it as an ordinary assistant text block
("API Error: Connection refused ..."), every solver turn ends in a bogus
GAVE_UP, and the swarm happily burns hours of wall clock bumping and
escalating skill hints at a networking problem. See logs/bench-20261026-*.json
for a 0/10 run whose real cause was ECONNREFUSED.

Probing the endpoint once at startup turns that into a 2-second error message.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

# HTTP statuses that mean "reachable, but you will not get answers right now".
TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
# HTTP statuses that mean "reachable, but the credentials/permission are wrong"
# — retrying will never help.
AUTH_STATUSES = frozenset({401, 403})


@dataclass
class ProbeResult:
    """Outcome of probing one endpoint."""

    name: str
    url: str
    ok: bool = False
    # "ok" | "unreachable" | "auth" | "rate_limited" | "server_error" |
    # "bad_response" | "not_configured" | "timeout"
    kind: str = "ok"
    status: int | None = None
    detail: str = ""

    def __str__(self) -> str:
        # ASCII only — see render(): this string reaches the startup console.
        where = f"{self.name} ({self.url})" if self.url else self.name
        return f"{where}: {self.kind}" + (f" - {self.detail}" if self.detail else "")


@dataclass
class HealthReport:
    """Aggregated startup health."""

    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def fatal(self) -> list[ProbeResult]:
        """Problems that must stop the run.

        Only "nothing is there" (unreachable: refused / DNS failure / TLS) and
        "your credentials are wrong" (auth) are fatal — both are deterministic
        and will not improve on their own.

        A timeout is deliberately NOT fatal. Observed live: the local gateway
        hangs for a few minutes at a time and then answers in ~1s, so treating a
        slow probe as fatal blocked runs that would have worked. Proceeding is
        safe because the swarm has its own guard — MAX_INFRA_FAILURES consecutive
        infrastructure failures retire it within a few rounds with a recorded
        reason, and ctfd-bench marks the report `usable_run: false`. That bounds
        the damage without betting the whole run on one probe.
        """
        return [p for p in self.probes
                if not p.ok and p.kind in ("unreachable", "auth")]

    @property
    def warnings(self) -> list[ProbeResult]:
        return [p for p in self.probes if not p.ok and p not in self.fatal]

    @property
    def ok(self) -> bool:
        return not self.fatal

    def render(self) -> str:
        # ASCII markers on purpose: this text is printed on startup, and a
        # Windows console on a CJK codepage (cp936) raises UnicodeEncodeError on
        # non-ASCII glyphs — which would turn a clear "your gateway is down"
        # message into a crash in the error-reporting path itself.
        fatal = self.fatal
        lines: list[str] = []
        for p in self.probes:
            if p.ok:
                mark = "[ok]"
            elif p in fatal:
                mark = "[FATAL]"
            else:
                mark = "[warn]"
            lines.append(f"  {mark} {p}")
        return "\n".join(lines)


class HealthCheckError(RuntimeError):
    """Raised when the startup probe finds a fatal problem."""

    def __init__(self, report: HealthReport) -> None:
        self.report = report
        super().__init__("Model gateway health check failed:\n" + report.render())


def _anthropic_headers(token: str) -> dict[str, str]:
    h = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if token:
        # The Claude CLI sends ANTHROPIC_AUTH_TOKEN as a bearer token; gateways
        # commonly also accept x-api-key. Send both so either style works.
        h["Authorization"] = f"Bearer {token}"
        h["x-api-key"] = token
    return h


def classify_transport_error(exc: Exception) -> tuple[str, str]:
    """Map an httpx transport error onto (kind, detail)."""
    text = str(exc) or exc.__class__.__name__
    low = text.lower()
    if "refused" in low or "connect" in low or "getaddrinfo" in low or "name or service" in low:
        return "unreachable", text
    if "timed out" in low or "timeout" in low:
        return "timeout", text
    if "certificate" in low or "ssl" in low or "tls" in low:
        return "unreachable", text
    return "unreachable", text


def classify_status(status: int, body: str) -> tuple[bool, str, str]:
    """Map an HTTP status onto (ok, kind, detail)."""
    snippet = (body or "").strip().replace("\n", " ")[:200]
    if 200 <= status < 300:
        return True, "ok", ""
    if status in AUTH_STATUSES:
        return False, "auth", f"HTTP {status} {snippet}"
    if status == 404:
        # Reachable host, wrong path — the gateway is up but this route is not
        # what we expect. Not fatal: some gateways omit /v1/messages.
        return False, "bad_response", f"HTTP 404 at this path {snippet}"
    if status in TRANSIENT_STATUSES:
        return False, "rate_limited" if status in (429, 529) else "server_error", (
            f"HTTP {status} {snippet}"
        )
    return False, "bad_response", f"HTTP {status} {snippet}"


async def _probe_with_retries(
    probe,
    *args,
    attempts: int = 3,
    backoff_s: float = 2.0,
    **kwargs,
) -> ProbeResult:
    """Retry transient probe failures before declaring the endpoint dead.

    A single probe is too fragile: a local gateway that is merely busy answers a
    later attempt fine, and failing the whole run on one ReadTimeout is worse
    than the problem the probe exists to catch. Auth failures are NOT retried —
    bad credentials do not improve on their own.
    """
    attempts = max(1, attempts)
    result = ProbeResult(name="?", url="")
    for attempt in range(1, attempts + 1):
        result = await probe(*args, **kwargs)
        if result.ok or result.kind in ("auth", "not_configured", "bad_response"):
            return result
        if attempt < attempts:
            logger.warning(
                "Health probe %s failed (%s) — retry %d/%d in %.0fs",
                result.name, result.kind, attempt + 1, attempts, backoff_s,
            )
            await asyncio.sleep(backoff_s * attempt)
    if attempts > 1 and not result.ok:
        result.detail = f"{result.detail} (after {attempts} attempts)".strip()
    return result


async def probe_anthropic(
    base_url: str,
    token: str,
    model: str,
    timeout_s: float = 20.0,
) -> ProbeResult:
    """Send a minimal Anthropic-protocol request to verify the primary path."""
    if not base_url:
        return ProbeResult(name="anthropic", url="", ok=True, kind="not_configured",
                           detail="ANTHROPIC_BASE_URL empty — using the official API")
    url = base_url.rstrip("/") + "/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    try:
        # verify=False: self-hosted gateways routinely use self-signed certs,
        # and this probe must not fail for a reason unrelated to reachability.
        async with httpx.AsyncClient(timeout=timeout_s, verify=False) as client:
            resp = await client.post(url, json=payload, headers=_anthropic_headers(token))
    except Exception as e:  # noqa: BLE001 — any transport failure is "unreachable"
        kind, detail = classify_transport_error(e)
        return ProbeResult(name="anthropic", url=url, ok=False, kind=kind, detail=detail)

    ok, kind, detail = classify_status(resp.status_code, resp.text)
    return ProbeResult(name="anthropic", url=url, ok=ok, kind=kind,
                       status=resp.status_code, detail=detail)


async def probe_openai(
    base_url: str,
    api_key: str,
    model: str,
    timeout_s: float = 20.0,
) -> ProbeResult:
    """Probe the OpenAI-compatible endpoint (quota fallback + summarizer)."""
    if not base_url:
        return ProbeResult(name="gateway", url="", ok=True, kind="not_configured",
                           detail="GATEWAY_BASE_URL empty — fallback/summarizer disabled")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = {"content-type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with httpx.AsyncClient(timeout=timeout_s, verify=False) as client:
            resp = await client.post(url, json=payload, headers=headers)
    except Exception as e:  # noqa: BLE001
        kind, detail = classify_transport_error(e)
        return ProbeResult(name="gateway", url=url, ok=False, kind=kind, detail=detail)

    ok, kind, detail = classify_status(resp.status_code, resp.text)
    return ProbeResult(name="gateway", url=url, ok=ok, kind=kind,
                       status=resp.status_code, detail=detail)


async def check_health(settings: object, probe_model: str = "") -> HealthReport:
    """Probe every configured model endpoint. Never raises.

    Transient failures are retried — see `_probe_with_retries`.
    """
    from backend.models import DEFAULT_MODELS, model_id_from_spec

    timeout_s = float(getattr(settings, "health_check_timeout_s", 20.0) or 20.0)
    attempts = int(getattr(settings, "health_check_retries", 3) or 1)
    specs = list(DEFAULT_MODELS)
    primary = probe_model or model_id_from_spec(specs[0])
    summarizer = getattr(settings, "summarizer_model", "") or ""

    report = HealthReport()
    report.probes.append(
        await _probe_with_retries(
            probe_anthropic,
            getattr(settings, "anthropic_base_url", "") or "",
            getattr(settings, "anthropic_auth_token", "")
            or getattr(settings, "gateway_api_key", "")
            or "",
            primary,
            timeout_s,
            attempts=attempts,
        )
    )
    report.probes.append(
        await _probe_with_retries(
            probe_openai,
            getattr(settings, "gateway_base_url", "") or "",
            getattr(settings, "gateway_api_key", "") or "",
            summarizer or primary,
            timeout_s,
            attempts=attempts,
        )
    )
    return report


async def ensure_healthy(settings: object, probe_model: str = "") -> HealthReport:
    """Run the startup probe; raise HealthCheckError on a fatal result.

    Warnings are logged but tolerated. A rate-limited, slow or 404'ing gateway
    may still serve traffic, and the swarm's own infrastructure accounting
    retires it with a recorded reason if it does not — which is a better outcome
    than refusing to start on one ambiguous probe.
    """
    if not getattr(settings, "startup_health_check", True):
        return HealthReport()

    report = await check_health(settings, probe_model)
    for w in report.warnings:
        logger.warning(
            "Health check warning — %s. Proceeding anyway; if it persists the "
            "swarm will retire itself after %s consecutive infrastructure "
            "failures and record the reason.",
            w, getattr(settings, "max_infra_failures", 3),
        )
    if not report.ok:
        raise HealthCheckError(report)
    logger.info("Gateway health check passed:\n%s", report.render())
    return report
