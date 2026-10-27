"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

import re

from pydantic import model_validator
from pydantic_settings import BaseSettings

# An unquoted value followed by whitespace + '#' is a trailing comment. This is
# python-dotenv's rule, and it has to be applied here too: Docker's own
# `--env-file` / compose `env_file:` parsers take everything after the first '='
# verbatim, so `SOLVER_TIME_BUDGET_S=1500   # per-attempt wall clock` arrives as
# the literal string "1500   # per-attempt wall clock" and fails int validation.
# Under `docker compose` this was masked only because the environment: block
# re-declares those two keys with clean values; any other key with a trailing
# comment broke the same way.
_INLINE_COMMENT_RE = re.compile(r"\s+#.*$")


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing ` # comment` from an unquoted value.

    Quoted values are returned untouched, and a '#' with no preceding whitespace
    is left alone, so passwords and flag-ish strings survive.
    """
    if not value:
        return value
    stripped = value.strip()
    if stripped[:1] in ("'", '"'):
        return value
    return _INLINE_COMMENT_RE.sub("", value).rstrip()


class Settings(BaseSettings):
    # CTFd
    ctfd_url: str = "http://localhost:8000"
    ctfd_user: str = "admin"
    ctfd_pass: str = "admin"
    ctfd_token: str = ""

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Claude Code CLI routing (claude-sdk backend).
    # Set anthropic_base_url to route ALL claude-cli traffic through your
    # unified gateway (Anthropic-protocol endpoint). Leave empty for the
    # official Anthropic API. anthropic_auth_token defaults to the gateway key.
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""

    # Provider-specific (optional, for Bedrock/Azure/Zen fallback)
    aws_region: str = "us-east-1"
    aws_bearer_token: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    opencode_zen_api_key: str = ""

    # Gateway provider (OpenAI-compatible unified gateway, e.g. deepseek/kimi/glm/qwen)
    gateway_base_url: str = ""
    gateway_api_key: str = ""

    # Skill knowledge base (see backend/skills.py)
    # skills_dir: path readable by the AGENT process (triage/index rendering)
    # skills_host_dir: same directory as seen by the DOCKER HOST (bind-mounted
    #   into sandboxes at /skills:ro). Only differs when the agent itself runs
    #   in a container (docker-outside-of-docker). Empty = use skills_dir.
    skills_dir: str = ""
    skills_host_dir: str = ""

    # Docker-outside-of-docker path translation. When the agent runs inside a
    # container with /var/run/docker.sock mounted, sandbox bind mounts must use
    # HOST paths. Set to the absolute project root on the host (e.g. D:/code/ctf).
    # Empty = agent runs directly on the host (no translation).
    host_project_dir: str = ""
    # Base directory for per-sandbox workspaces. Empty = system temp (local
    # mode) or <project>/.workspaces (when host_project_dir is set).
    workspace_root: str = ""

    # Infra
    sandbox_image: str = "ctf-sandbox"
    max_concurrent_challenges: int = 10
    # Hard cap on bump/retry rounds per solver. A solver that keeps coming back
    # GAVE_UP is retired after this many rounds instead of spinning forever.
    # 0 = unlimited (only sane in coordinator mode against a live CTFd).
    #
    # This interacts with fresh_restart_rounds: the distil->merge cycle needs
    # fresh_restart_rounds bumps to complete ONE merge, so the cap must be at
    # least 2*fresh_restart_rounds + 1 for the merged state to drive more than a
    # single round. The default 12 allows two full merge cycles at the default
    # fresh_restart_rounds of 5. A warning is logged when they are misaligned.
    max_attempts_per_challenge: int = 12
    container_memory_limit: str = "16g"
    # CPU quota per sandbox container (NanoCpus = limit * 1e9).
    container_cpu_limit: float = 2.0
    # Per-command wall clock inside the sandbox (seconds). Enforced by wrapping
    # every command in coreutils `timeout`, so a killed solver cannot leave an
    # orphaned process burning CPU in its container.
    sandbox_command_timeout_s: int = 900
    # Timeout for the Claude Code CLI's own HTTP calls to the gateway (ms).
    # 0 = CLI default (which can stall indefinitely on a half-open connection).
    cli_api_timeout_ms: int = 600_000
    # Sweep leftover sandbox containers/workspaces from a crashed previous run at
    # startup. Turn this OFF on one of them if you run two agents concurrently
    # (e.g. a coordinator plus a bench): the sweep matches by label, not by
    # owner, so it would delete the other agent's LIVE sandboxes.
    cleanup_orphans_on_start: bool = True
    # Optional HTTP(S) proxy injected into sandbox containers (e.g.
    # "http://host.docker.internal:7890" so solvers can reach github etc.).
    sandbox_proxy_url: str = ""
    # Comma-separated hostnames the AGENT process may fetch with web_fetch even
    # though they resolve to a non-public address. Everything else private is
    # blocked by resolved IP (not by hostname prefix), because the agent holds
    # /var/run/docker.sock. Keep this list as short as possible.
    web_fetch_allow_hosts: str = "host.docker.internal"

    # Startup health check — probe the gateway before launching anything so a
    # misconfigured/unreachable endpoint fails in seconds instead of grinding
    # through hours of "API Error: Connection refused" rounds.
    startup_health_check: bool = True
    health_check_timeout_s: float = 20.0
    # Probe attempts before declaring an endpoint dead. A local gateway that is
    # merely busy answers a later attempt, and failing the whole run on one
    # ReadTimeout is worse than the outage the probe exists to catch. Auth
    # failures are never retried.
    health_check_retries: int = 3

    # Solver behavior
    # Wall-clock budget per solver attempt (seconds). On expiry the turn is
    # INTERRUPTED (not killed), a Cairn-style conclude phase harvests the
    # session, and the solver is bumped with fresh direction. 0 = no budget.
    solver_time_budget_s: int = 1500
    # Independent budget for the phase-2 conclude harvest (seconds). Kept
    # separate from solver_time_budget_s so harvesting always has room to run.
    conclude_timeout_s: int = 90
    # Consecutive infrastructure failures (connection refused / 5xx / auth)
    # across a swarm before the whole swarm is retired. Stops a dead gateway
    # from burning the entire time limit.
    max_infra_failures: int = 3
    # Grace period for siblings to wind down after one solver wins. They watch
    # cancel_event and exit their stream loop within a couple of seconds, which
    # lets each finish its own container/workspace teardown. Only stragglers are
    # hard-cancelled after this.
    solver_teardown_grace_s: int = 60
    # Backoff between bump rounds: wait base + step*round, capped at max.
    # Configurable (rather than a hardcoded formula) so the retry cadence can be
    # tightened for benchmarks and loosened for a live competition.
    bump_cooldown_base_s: int = 5
    bump_cooldown_step_s: int = 10
    bump_cooldown_max_s: int = 120
    # Backoff after a quota/rate-limit error with no fallback endpoint available.
    quota_backoff_base_s: int = 60
    quota_backoff_max_s: int = 300
    # Start injecting skill-library hints after this many failed attempts (bumps).
    # 1 = hints join from the first retry. Matches README / .env.example /
    # docker-compose.
    skill_after_bumps: int = 1
    # Coordinator model — use your STRONGEST model here (it coaches weaker
    # solvers). Empty = CLI --coordinator-model, else built-in default.
    coordinator_model: str = ""
    # Wall-clock cap for a single coordinator LLM turn (seconds). Without it a
    # hung gateway freezes the whole event loop (poller, patrol, operator UI).
    coordinator_turn_timeout_s: int = 300
    # Proactive coordinator patrol: scan active solvers every N seconds for
    # stuck patterns (quiet too long / looping), alert + advise them.
    patrol_interval_s: int = 120
    # A solver with no new trace activity for this many seconds counts as stuck.
    patrol_stuck_quiet_s: int = 240
    # After this many failed bump attempts across a swarm with no winner,
    # spawn a stronger-model solver (escalation ladder). 0 disables.
    escalate_after_bumps: int = 8
    # Fresh-restart retry: after a GAVE_UP, distill the round into a journal
    # entry, reset the session, and restart with a clean context carrying only
    # the distilled state. Every N rounds the journal is meta-summarized (and
    # de-duplicated) so the carried state never grows unboundedly.
    # 0/1 disables (falls back to the continue-session bump).
    fresh_restart_rounds: int = 5
    # Long findings posted to the blackboard get an LLM key-info summary for
    # sibling injection (full text is always kept; nothing is truncated).
    summarize_threshold_chars: int = 900
    # Minimum distinct solver reports on the blackboard before spending an LLM
    # call to distil them into the shared team state. Below this the raw (bounded
    # and de-duplicated) form is used instead — one solver's own findings do not
    # need merging with anything.
    sibling_digest_min_reports: int = 2
    # The team digest is recomputed at most once per bump generation and shared
    # by every solver restarting in it, so N solvers cost one call, not N.
    sibling_digest_refresh_bumps: int = 1
    # Model used for blackboard summarization (fast/cheap gateway model).
    summarizer_model: str = "deepseek-v4-flash"
    # Operator web dashboard (live swarm view + guidance mailbox UI).
    # 0 disables. Default 8080.
    dashboard_port: int = 8080
    # Loopback by default: the dashboard exposes unauthenticated-write-grade
    # control (advise / bump / broadcast / spawn) plus flag export, and solver
    # sandboxes can reach the host via host.docker.internal. Bind explicitly to
    # 0.0.0.0 only behind a reverse proxy that adds auth.
    dashboard_host: str = "127.0.0.1"
    # Bearer token required on every dashboard request. Empty = generate a
    # random token at startup and print it once (never silently unauthenticated).
    dashboard_token: str = ""
    # Max characters of blackboard content returned by /api/state per finding
    # (the export endpoint still returns everything).
    dashboard_feed_chars: int = 4000
    # Operator message endpoint (ctf-msg). 0 = auto-pick a free port, which
    # breaks `ctf-msg`'s default of 9400 — set it explicitly in .env/compose.
    msg_port: int = 9400
    msg_host: str = "127.0.0.1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def _normalize_inline_comments(cls, data):
        """Strip trailing ` # comments` from env-sourced string values.

        Runs before field validation so an int field never sees the comment text.
        Only plain dicts are touched; already-constructed inputs pass through.
        """
        if not isinstance(data, dict):
            return data
        return {
            k: _strip_inline_comment(v) if isinstance(v, str) else v
            for k, v in data.items()
        }


def validate_tuning(settings: Settings) -> list[str]:
    """Return human-readable warnings about tuning combinations that defeat a
    mechanism silently.

    These are warnings, not errors: every combination is runnable, but each one
    quietly disables something the operator probably intended to use. Surfacing
    them at startup is far cheaper than debugging why a feature "did nothing".
    """
    warnings: list[str] = []

    # getattr with defaults so this stays usable against a partial settings
    # object (tests, and callers that build Settings-like namespaces).
    cap = int(getattr(settings, "max_attempts_per_challenge", 0) or 0)
    fresh = int(getattr(settings, "fresh_restart_rounds", 0) or 0)
    budget = int(getattr(settings, "solver_time_budget_s", 0) or 0)
    if cap > 0 and fresh > 0:
        # One distil->merge cycle costs `fresh` bumps; the merged state then
        # needs at least one further round to be worth computing.
        needed = 2 * fresh + 1
        if cap < fresh + 1:
            warnings.append(
                f"MAX_ATTEMPTS_PER_CHALLENGE={cap} is below FRESH_RESTART_ROUNDS={fresh}, "
                "so the journal is never merged — the distilled state grows as a plain "
                f"list of up to {cap} entries. Raise the cap to at least {fresh + 1}."
            )
        elif cap < needed:
            cycles = (cap - 1) // fresh
            warnings.append(
                f"MAX_ATTEMPTS_PER_CHALLENGE={cap} with FRESH_RESTART_ROUNDS={fresh} allows "
                f"only {cycles} merge cycle(s), and the merged state drives just "
                f"{cap - cycles * fresh} round(s) before the solver retires. "
                f"For {cycles + 1} cycles set the cap to at least {needed}."
            )

    if budget > 0 and cap == 0:
        warnings.append(
            "MAX_ATTEMPTS_PER_CHALLENGE=0 means unlimited rounds; the only bound is an "
            "external time limit. Fine for coordinator mode against a live CTFd, "
            "dangerous for ctf-solve --challenge."
        )

    if (fresh > 0
            and not (getattr(settings, "gateway_base_url", "")
                     and getattr(settings, "gateway_api_key", ""))):
        warnings.append(
            "GATEWAY_BASE_URL/GATEWAY_API_KEY are unset, so round distillation and the "
            "team digest cannot call a summarizer. The journal falls back to raw "
            "(bounded) entries and sibling context to clipped raw findings — the "
            "mechanism still works, just without merging or de-duplication by model."
        )

    return warnings
