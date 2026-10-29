"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

import re
from typing import Any

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
    #
    # `anthropic_api_key` is not read by this codebase — it is read by the Claude
    # Code CLI itself, out of the process environment, when ANTHROPIC_BASE_URL is
    # left empty (i.e. talking to the official API rather than a gateway). It is
    # declared here so it stays documented and validated alongside everything
    # else. Do not "clean it up" as unused.
    #
    # There is deliberately no `openai_api_key`: this fork removed the Codex
    # backend, and `resolve_model` has no bare `openai/` provider. OpenAI-shaped
    # endpoints are reached through `gateway_*`, `azure_openai_*` or
    # `opencode_zen_api_key`. A vestigial key here would imply a code path that
    # does not exist.
    anthropic_api_key: str = ""
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
    # How many times a challenge may be auto-spawned. The coordinator re-checks
    # for unsolved challenges with no active swarm every status interval, which
    # is what makes it self-heal after a swarm retires (round cap / quota /
    # infrastructure) and what lets it notice a CTFd solve-history reset — CTFd
    # emits no event for a challenge becoming unsolved. The cap stops a genuinely
    # hopeless challenge from being re-queued forever.
    max_spawns_per_challenge: int = 3
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
    # 1 = hints join from the first retry, escalating L1 -> L2 -> L3 on each
    # subsequent bump. **0 DISABLES injection entirely**, matching the sibling knob
    # ESCALATE_AFTER_BUMPS. It used to mean the opposite — the gate
    # `bump_count < 0` could never fire and the level formula started at L2, so 0
    # injected the heaviest tier from the very first retry, which is the one value
    # nobody wants and made 0 unable to express the most common intent.
    #
    # This gates hint TEXT only. Approach rotation has its own switch
    # (SKILL_ROTATE_APPROACHES), so 0 + rotation is a real configuration: assign a
    # direction, inject no handbook excerpt.
    skill_after_bumps: int = 1
    # Master switch for the agent-side skill machinery: triage, escalating hint
    # injection, and the technique-rotation queue. False leaves the library MOUNTED
    # at /skills (that is SKILLS_HOST_DIR's job) so solvers can still read it on
    # their own initiative — it only stops the swarm from pushing it at them.
    #
    # This exists because SKILLS_DIR cannot serve as the switch: docker-compose pins
    # it to the in-container mount point and compose `environment:` overrides
    # `env_file:`, so under Docker there was no way to turn injection off at all.
    # That made "does the skill library help this model lineup?" unanswerable in the
    # mode people actually deploy.
    skill_hints: bool = True
    # Rotate the assigned skill handbook on each retry. This is a SEPARATE mechanism
    # from the hint injection above, and it used to have no switch of its own —
    # SKILLS_DIR="" turned off both at once, so there was no way to A/B them and no
    # way to tell which one was hurting.
    #
    # Rotation is the more suspicious of the two: it assigns handbooks in
    # alphabetical order within a pack, unrelated to what the solver is actually
    # stuck on, and the bump text is imperative ("本轮专攻新方向…先读再做"). A strong
    # model ignores an irrelevant instruction; a weaker or quantized one tends to
    # obey it and spends the round reading. Set false to keep the library mounted
    # and the hints injected while letting each solver keep its own line of attack.
    skill_rotate_approaches: bool = True
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
    #
    # This is the MERGE INTERVAL, not a restart interval: ANY value > 0 restarts on
    # every bump, and 0 is the only way to get the continue-session bump. The gate
    # is `> 0`, so the previously documented "0/1 disables" was wrong for 1 — and 1
    # is the most expensive setting there is, because `len(journal) >= 1` is true
    # after every single round, so each bump pays a meta-summarize call and the
    # journal never holds more than one entry. validate_tuning warns about it.
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
    # Summarizer-class call budget. Measured on a real run: the 900-token default
    # `_call_gateway` used was hit repeatedly and every truncated reply is
    # DISCARDED, so those rounds published no summary at all and the swarm fell
    # back to raw sibling text. 2500 leaves room for a real digest; the timeout
    # rises with it because a longer completion on the same model takes longer,
    # and a budget that only the fastest rounds fit is a budget that is usually off.
    summarizer_max_tokens: int = 2500
    summarizer_timeout_s: float = 60.0
    # Fact trust gate. A finding one solver asserts is posted as an UNVERIFIED
    # CLAIM; it is promoted to a corroborated FACT only when a DIFFERENT solver
    # independently reports the same thing. This is what stops one model's
    # hallucination from becoming the swarm's ground truth — every restart payload
    # is built from the board, and siblings previously had no way to tell a
    # three-model consensus from a single model's say-so. Turn off to restore the
    # old "everything a solver reports is a fact" behaviour.
    corroborate_facts: bool = True
    # Active verification. Corroboration can only WITHHOLD trust — it can never
    # refute a false premise (which stays on the board, still readable, still
    # tempting) and it can never confirm a true one that is hard to rediscover
    # (which stays ⚠ forever and gets under-weighted). Both holes close the same
    # way: somebody actually runs a test. The Verifier owns one sandbox per swarm,
    # reused across claims, and runs concurrently with the solvers so it costs no
    # wall-clock. REFUTED claims move into the dead-end ledger and broadcast a
    # prune notice that reaches running solvers within a few tool calls.
    verify_claims: bool = True
    # Model spec for the Verifier. Empty = the first solver in the lineup.
    # Verification is judgement-heavy: a weak model returns INCONCLUSIVE for
    # everything, which is safe but useless.
    verifier_model: str = ""
    # Wall-clock budget for ONE probe. Kept well below SOLVER_TIME_BUDGET_S — a
    # verification that needs 25 minutes is not a minimal test.
    verifier_budget_s: int = 300
    # Pre-filter each claim with one cheap call: is it testable by a probe, and
    # would its falsity invalidate other agents' work? Claims that are neither get
    # skipped. This is not a cost control, it is a wall-clock control: a probe that
    # cannot falsify anything still occupies the verifier for a whole round.
    verifier_triage: bool = True
    # Triage call budget. The hardcoded 1500 tokens / 25s it replaced was observed
    # truncating on a live run: the reply was discarded, triage returned no answer,
    # and the verifier fell back to probing blind. That is the safe direction but it
    # spends a decisive probe on claims that were never worth testing, which is the
    # one resource `verifier_max_probes` exists to protect.
    verifier_triage_max_tokens: int = 4000
    verifier_triage_timeout_s: float = 60.0
    # Hard cap on decisive probes per challenge. 0 = unlimited (the claim queue is
    # bounded regardless). Raise it if you want an exhaustive audit and do not care
    # how long it takes.
    verifier_max_probes: int = 0
    # Consult the Strategist every N swarm-wide bumps. It sees ONLY the challenge
    # metadata and the dead-end ledger — never the tactical findings that drown
    # the signal — and must answer CONTINUE or PIVOT with a new attack surface.
    # The verdict is broadcast as a strategist message, kept visually distinct
    # from a corroborated fact. 0 disables.
    strategist_every_bumps: int = 4
    # Model for the Strategist. Empty = COORDINATOR_MODEL, else a strong built-in
    # default. Deliberately NOT SUMMARIZER_MODEL: this is deep semantic reasoning
    # over the challenge text, and a flash model just rubber-stamps CONTINUE.
    strategist_model: str = ""
    # Model for judgement calls generally (Strategist + contradiction audit) —
    # reasoning, not summarization. Both fall back to COORDINATOR_MODEL then a
    # built-in default. SUMMARIZER_MODEL is never used for these: measured against a
    # live gateway, the flash summarizer answered "no contradiction" to an
    # unambiguous conflicting pair (finish_reason=stop, so not truncation — it
    # genuinely could not see it), and both mechanisms fail silently, so a weak model
    # here does not degrade them, it deletes them.
    reasoning_model: str = ""
    # Wall clock for ONE strategist consultation. Measured 100-150s on a reasoning
    # model, which spends ~3000 tokens thinking before it answers. The original 60s
    # default failed on every call and was logged at debug level, so the mechanism
    # was silently dead in production while looking configured and healthy.
    strategist_timeout_s: float = 300.0
    # Completion budget for the strategist. At 1200 a reasoning model came back
    # finish_reason="length" having used 6743 tokens, so the verdict line was never
    # emitted and the truncated tail read as CONTINUE — the opposite of what the
    # model had concluded.
    strategist_max_tokens: int = 8000
    # Minimum seconds between ANY two strategist consultations, including the ones
    # the stale-round breaker forces. The periodic path is already rate-limited by
    # generation, but a forced call bypassed that entirely, so four stuck solvers
    # each tripping the breaker produced four strong-model calls per breaker cycle
    # with no ceiling — on an unlimited-rounds coordinator run, unbounded spend on
    # the mechanism meant to save the run. 0 disables the interval.
    strategist_min_interval_s: float = 120.0
    # Contradiction audit overrides. Empty/0 = use REASONING_MODEL and the
    # module defaults. The numeric defaults are duplicated in backend/audit.py
    # (which imports this module, so it cannot be the other way round);
    # test_audit.py asserts the two agree.
    audit_model: str = ""
    audit_timeout_s: float = 300.0
    audit_max_tokens: int = 8000
    # Command-centre chat ("what is the progress on X?"). Summarises a challenge's
    # entire live state — blackboard, per-solver traces, dead ends, audit counters —
    # for an operator who cannot read four hours of logs. Also a reasoning task: it
    # must not upgrade a ⚠ claim into a fact while summarising, or the operator
    # steers the swarm on a hallucination.
    briefing_model: str = ""
    briefing_timeout_s: float = 300.0
    briefing_max_tokens: int = 8000
    # Stale-round circuit breaker. After N consecutive rounds in which a solver
    # added NOTHING new to the team (every dead end it reported was already
    # recorded, every fact already asserted by someone), force a strategist
    # consult and hand that solver a hard redirect forbidding further micro-tweaks
    # of the same payload. Catches the failure mode a per-command loop detector
    # cannot see, because each tweak is a genuinely different command. 0 disables.
    stale_round_limit: int = 3
    # Contradiction audit. Each round's new statements are compared against the
    # whole board, and a pair that cannot both be true is broadcast plus pushed to
    # the front of the verification queue. The Verifier tests a claim against the
    # TARGET; this tests claims against EACH OTHER, which catches the case where
    # two agents hold mutually exclusive beliefs and both look productive. Costs one
    # summarizer-class call per published round.
    audit_contradictions: bool = True
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


def _default_for(name: str) -> Any:
    """The declared default of a Settings field, read from the model itself.

    Call sites used to hardcode a fallback literal — `getattr(settings, "x", 0)` —
    because tests pass a SimpleNamespace that lacks fields. Those literals drifted:
    an audit found 11 where the fallback disagreed with the real default, so a
    partial settings object silently ran with unlimited rounds, no wall-clock
    budget, no CLI timeout, and the strategist and circuit breaker disabled. Reading
    the default from `Settings.model_fields` makes that drift structurally
    impossible rather than merely unlikely.
    """
    field = Settings.model_fields.get(name)
    if field is None:
        raise KeyError(f"{name!r} is not a Settings field")
    return field.default


def int_setting(settings: object, name: str) -> int:
    """Read an int knob. Only a MISSING or None value falls back to the default.

    Never `getattr(s, name, X) or X`: that idiom rewrites an explicit 0 into X, so
    every knob whose 0 means "off", "unlimited" or "no cap" became inexpressible.
    Two shipped knobs documented an off-state that the code could not reach —
    PATROL_INTERVAL_S=0 was rewritten to 120 by its own `or`, and
    MAX_SPAWNS_PER_CHALLENGE=0 to 3.
    """
    value = getattr(settings, name, None)
    if value is None or value == "":
        return int(_default_for(name))
    return int(value)


def float_setting(settings: object, name: str) -> float:
    """Read a float knob. Same None-only fallback rule as `int_setting`."""
    value = getattr(settings, name, None)
    if value is None or value == "":
        return float(_default_for(name))
    return float(value)


def bool_setting(settings: object, name: str) -> bool:
    """Read a bool knob, accepting the strings an env file can deliver."""
    value = getattr(settings, name, None)
    if value is None or value == "":
        return bool(_default_for(name))
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


def str_setting(settings: object, name: str) -> str:
    """Read a string knob. Missing/None falls back; an explicit "" is honoured,
    because for several knobs empty is the meaningful value."""
    value = getattr(settings, name, None)
    if value is None:
        return str(_default_for(name) or "")
    return str(value)


def validate_tuning(settings: Settings) -> list[str]:
    """Return human-readable warnings about tuning combinations that defeat a
    mechanism silently.

    These are warnings, not errors: every combination is runnable, but each one
    quietly disables something the operator probably intended to use. Surfacing
    them at startup is far cheaper than debugging why a feature "did nothing".
    """
    warnings: list[str] = []

    # getattr with defaults so this stays usable against a partial settings
    # object (tests, and callers that build Settings-like namespaces). The
    # fallbacks here are deliberately 0 rather than the shipped defaults: for a
    # caller that supplies nothing, "no opinion" must not invent a misalignment
    # warning out of defaults the caller never asked for.
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

    if fresh == 1:
        warnings.append(
            "FRESH_RESTART_ROUNDS=1 restarts the session every bump AND meta-summarizes "
            "the journal every bump, so each round pays a summarizer call and the journal "
            "never holds more than one entry — the distil->merge cycle this knob exists "
            "for cannot happen. This is the most expensive setting available. Use 0 for "
            "the continue-session bump, or 3-5 for a real merge cycle."
        )

    # Deliberately a bare getattr with a 0 fallback, not int_setting: this function
    # must stay usable against a partial settings object, and for such a caller
    # "no opinion" must not invent a warning out of defaults they never asked for.
    # That exemption is enforced by name in tests/test_settings_access.py.
    strategist = int(getattr(settings, "strategist_every_bumps", 0) or 0)
    if (strategist > 0
            and not (getattr(settings, "gateway_base_url", "")
                     and getattr(settings, "gateway_api_key", ""))):
        warnings.append(
            f"STRATEGIST_EVERY_BUMPS={strategist} is set but GATEWAY_BASE_URL/"
            "GATEWAY_API_KEY are unset, so the Strategist can never be consulted. "
            "The stale-round circuit breaker still fires and still hands solvers a "
            "hard redirect, but no PIVOT verdict will ever be produced."
        )

    return warnings
