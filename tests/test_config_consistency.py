"""Guard against configuration drift between the sources of truth.

docker-compose.yml used to re-declare tuning defaults with its own
`${VAR:-default}` fallbacks. That duplicated backend/config.py, and the two
drifted: the code default for MAX_ATTEMPTS_PER_CHALLENGE was raised to 12 while
compose still pinned 6, so every compose run silently got a cap that allowed only
one distil->merge cycle. Nothing failed — validate_tuning() happening to warn at
startup was the only reason it surfaced.

The rule these tests enforce: compose may only set what genuinely has to DIFFER
inside the container. Everything else comes from `env_file: .env` and the
defaults in config.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from backend.config import Settings

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"

# Keys that legitimately must differ inside the container, or must be present
# for Docker's own interpolation/validation.
ALLOWED_IN_COMPOSE = {
    "HOST_PROJECT_DIR",   # required by the `:?` guard; DooD bind paths
    "SKILLS_DIR",         # /skills inside the container, a host path outside
    "SKILLS_HOST_DIR",    # the host-side counterpart
    "MSG_HOST",           # must be 0.0.0.0 in-container or the port cannot forward
    "DASHBOARD_HOST",     # ditto
}


@pytest.fixture(scope="module")
def compose() -> dict:
    if not COMPOSE.exists():
        pytest.skip("no docker-compose.yml in this checkout")
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _agent_env(compose: dict) -> dict:
    return (compose.get("services", {}).get("agent", {}) or {}).get("environment", {}) or {}


def test_compose_does_not_redeclare_tuning_defaults(compose) -> None:
    """Any Settings field re-declared in compose is a drift opportunity."""
    env = _agent_env(compose)
    fields = {name.upper() for name in Settings.model_fields}
    offenders = sorted(k for k in env if k.upper() in fields and k.upper() not in ALLOWED_IN_COMPOSE)
    assert not offenders, (
        "docker-compose.yml re-declares defaults that belong to backend/config.py; "
        "they will drift. Remove them and let `env_file: .env` pass the value "
        f"through: {offenders}"
    )


def test_compose_only_sets_allowed_overrides(compose) -> None:
    env = _agent_env(compose)
    unexpected = sorted(k for k in env if k.upper() not in ALLOWED_IN_COMPOSE)
    assert not unexpected, f"unexpected keys in compose environment: {unexpected}"


def test_compose_publishes_both_ports_loopback_only(compose) -> None:
    """The dashboard can inject into solvers and export flags; the msg endpoint
    accepts operator commands. Neither may be reachable from the LAN or from a
    sandbox container via host.docker.internal."""
    ports = (compose.get("services", {}).get("agent", {}) or {}).get("ports", []) or []
    published = [str(p) for p in ports]
    assert any(p.startswith("127.0.0.1:") and p.endswith(":8080") for p in published), published
    assert any(p.startswith("127.0.0.1:") and p.endswith(":9400") for p in published), published
    assert not any(p.startswith("0.0.0.0:") for p in published), published


def test_compose_in_container_hosts_are_wildcard(compose) -> None:
    """...but the listeners themselves must bind 0.0.0.0 or Docker cannot forward."""
    env = _agent_env(compose)
    assert "0.0.0.0" in str(env.get("DASHBOARD_HOST", ""))
    assert "0.0.0.0" in str(env.get("MSG_HOST", ""))


def test_compose_mounts_the_runtime_dirs(compose) -> None:
    vols = " ".join(str(v) for v in (compose.get("services", {}).get("agent", {}) or {}).get("volumes", []))
    for needed in ("/var/run/docker.sock", "/app/challenges", "/app/logs", "/app/.workspaces"):
        assert needed in vols, f"missing mount {needed}"


# ---------------------------------------------------------------------------
# .env.example must document every knob, and must not teach unsafe syntax
# ---------------------------------------------------------------------------

def test_env_example_documents_every_setting() -> None:
    """An undocumented knob is one nobody sets correctly.

    A commented-out `# CTFD_USER=admin` counts as documentation: that is how the
    template shows optional credentials and session-auth fallbacks.
    """
    if not ENV_EXAMPLE.exists():
        pytest.skip("no .env.example in this checkout")
    documented: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "=" not in stripped:
            continue
        key = stripped.lstrip("#").split("=", 1)[0].strip().upper()
        if key and key.replace("_", "").isalnum():
            documented.add(key)
    missing = sorted(
        name.upper() for name in Settings.model_fields
        if name.upper() not in documented
    )
    assert not missing, f".env.example is missing: {missing}"


def test_env_example_has_no_inline_comments() -> None:
    """Docker's --env-file and compose's env_file: keep everything after '='
    verbatim, so `KEY=1500  # note` becomes the literal string and crashes int
    validation. Comments belong on their own line."""
    if not ENV_EXAMPLE.exists():
        pytest.skip("no .env.example in this checkout")
    offenders = []
    for lineno, raw in enumerate(ENV_EXAMPLE.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        value = line.split("=", 1)[1].strip()
        if value[:1] in ("'", '"'):
            continue
        if " #" in value or "\t#" in value:
            offenders.append(f"{lineno}: {line.split('=')[0]}")
    assert not offenders, ".env.example teaches inline comments:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# Shipped defaults must be mutually consistent
# ---------------------------------------------------------------------------

def test_shipped_defaults_allow_two_merge_cycles() -> None:
    """The distil->merge cycle is the point of fresh_restart_rounds; the defaults
    must let it complete more than once."""
    s = Settings(_env_file=None)
    assert s.max_attempts_per_challenge >= 2 * s.fresh_restart_rounds + 1, (
        f"cap {s.max_attempts_per_challenge} vs cycle {s.fresh_restart_rounds}"
    )


def test_shipped_defaults_produce_no_tuning_warnings() -> None:
    """Apart from the gateway not being configured in a bare test environment."""
    from backend.config import validate_tuning

    s = Settings(_env_file=None, gateway_base_url="http://gw/v1", gateway_api_key="k")
    warns = validate_tuning(s)
    assert warns == [], warns


def test_conclude_budget_is_independent_of_the_round_budget() -> None:
    """Cairn invariant: phase 2 needs its own budget, or an exhausted phase 1
    leaves no room to harvest."""
    s = Settings(_env_file=None)
    assert s.conclude_timeout_s > 0
    assert s.conclude_timeout_s < s.solver_time_budget_s
