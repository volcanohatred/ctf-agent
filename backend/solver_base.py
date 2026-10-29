"""Solver result type, status constants, and solver protocol — shared across all backends."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

# Status constants
FLAG_FOUND = "flag_found"
GAVE_UP = "gave_up"
CANCELLED = "cancelled"
ERROR = "error"
QUOTA_ERROR = "quota_error"
# Infrastructure failure: the model endpoint was unreachable / rejected us.
# Kept separate from ERROR so the swarm can back off and abort the whole swarm
# instead of treating it as "the model gave up on this challenge".
INFRA_ERROR = "infra_error"

# Flag confirmation markers from CTFd
CORRECT_MARKERS = ("CORRECT", "ALREADY SOLVED")

# The Claude Code CLI surfaces API failures as an ordinary assistant text block
# rather than raising. Without this classification a dead gateway looks exactly
# like a solver that tried hard and gave up, and the swarm will bump/escalate
# for hours. Ordered: quota-ish signals first.
_QUOTA_TEXT_RE = re.compile(
    r"\b(rate[\s_-]?limit|too many requests|quota|429|529|overloaded|"
    r"insufficient[_ ](?:balance|quota)|billing|credit)\b",
    re.IGNORECASE,
)
_API_ERROR_RE = re.compile(
    r"^\s*(?:api\s+error|api\s+request\s+failed|connection\s+error|"
    r"upstream\s+error|request\s+failed)\b",
    re.IGNORECASE,
)
_TRANSPORT_RE = re.compile(
    r"\b(ECONNREFUSED|ECONNRESET|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|EPIPE|"
    r"connection refused|connection reset|getaddrinfo|name or service not known|"
    r"socket hang up|network is unreachable|no route to host|"
    r"tls|ssl certificate|certificate verify failed|bad gateway|"
    r"gateway time-?out|service unavailable|internal server error)\b",
    re.IGNORECASE,
)
# HTTP statuses the CLI sometimes prints inline.
_STATUS_RE = re.compile(r"\bHTTP\s*(\d{3})\b|\bstatus(?:\s*code)?[:=]\s*(\d{3})\b", re.IGNORECASE)

# A bare transport signal (no "API Error:" prefix, no HTTP status) is only
# trusted when the message is essentially nothing but that signal.
BARE_SIGNAL_MAX_CHARS = 120
BARE_SIGNAL_MAX_OFFSET = 60


def classify_api_error(text: str) -> str | None:
    """Return QUOTA_ERROR / INFRA_ERROR when `text` is an API failure message.

    Returns None when the text looks like genuine model output. Deliberately
    conservative: only matches texts that START like an API error, or that
    contain an unmistakable transport/status signal, so a solver discussing
    "connection refused" as part of a challenge is not misclassified.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    looks_like_api_error = bool(_API_ERROR_RE.match(stripped))
    status = None
    m = _STATUS_RE.search(stripped[:400])
    # An HTTP status counts as an infrastructure signal ONLY when the message is
    # also shaped like a CLI/gateway error. On its own it proves nothing about the
    # gateway, because solvers report the TARGET's statuses constantly — and both
    # of the interesting cases are findings, not outages:
    #   "[fact] the endpoint returns HTTP 422 for blocked classes" is the central
    #     observation of an entire challenge family;
    #   "[fact] that input makes it return HTTP 500" is an exploit primitive.
    # Classifying those as INFRA_ERROR routed them through `publishable()`, which
    # silently deleted them from the blackboard — the team lost its own best
    # evidence and the round was counted as a dead gateway. Success statuses fell
    # through every branch below and hit the trailing `return INFRA_ERROR` too, so
    # "HTTP 200" was deleted exactly like "HTTP 503".
    if m and looks_like_api_error:
        status = int(m.group(1) or m.group(2))

    # No explicit API-error prefix and no HTTP status: accept a bare transport
    # signal only when the whole message IS essentially that signal — short,
    # single-line, keyword near the front. A looser rule produced false
    # positives on real findings such as "[fact] the service returns 'connection
    # refused' when the port is closed", and discarding genuine work costs far
    # more than missing one error report (ResultMessage.is_error and
    # api_error_status still catch those).
    if not looks_like_api_error and status is None:
        match = _TRANSPORT_RE.search(stripped)
        if (len(stripped) > BARE_SIGNAL_MAX_CHARS
                or "\n" in stripped
                or not match
                or match.start() > BARE_SIGNAL_MAX_OFFSET):
            return None

    if status is not None:
        if status in (429, 529):
            return QUOTA_ERROR
        if status in (401, 403):
            return INFRA_ERROR
        if status >= 500 or status in (408, 425):
            return INFRA_ERROR

    if _QUOTA_TEXT_RE.search(stripped[:600]):
        return QUOTA_ERROR
    return INFRA_ERROR


def classify_api_status(status: int | None) -> str | None:
    """Map the SDK's ResultMessage.api_error_status onto a solver status."""
    if not status:
        return None
    if status in (429, 529):
        return QUOTA_ERROR
    if status >= 400:
        return INFRA_ERROR
    return None


# Exception-class classification, matched by NAME across the MRO so it keeps
# working across pydantic-ai / httpx / openai versions without importing any of
# them here (this module is deliberately dependency-free).
_QUOTA_EXC_NAMES = frozenset({
    "ModelRateLimitError", "ModelQuotaExceededError", "ModelOverloadedError",
    "RateLimitError", "APIRateLimitError", "QuotaExceededError",
})
_INFRA_EXC_NAMES = frozenset({
    "ModelConnectionError", "ModelHTTPError", "ModelUnavailableError",
    "ModelTimeoutError", "ModelRequestTimeout", "ModelServerError",
    "ContentFilterError", "ConnectError", "ConnectTimeout", "ReadTimeout",
    "WriteTimeout", "PoolTimeout", "RemoteProtocolError", "APIConnectionError",
    "APITimeoutError", "InternalServerError", "APIStatusError",
    "AuthenticationError", "PermissionDeniedError", "NotFoundError",
})
# Context overflow is recoverable, but only by dropping history — the solver
# handles it locally rather than bubbling a distinct status.
_CONTEXT_EXC_NAMES = frozenset({
    "ModelContextWindowExceededError", "ContextWindowExceededError",
})


def classify_exception(exc: BaseException) -> str | None:
    """Map an exception onto QUOTA_ERROR / INFRA_ERROR, or None if it is neither."""
    names = {c.__name__ for c in type(exc).__mro__}
    if names & _QUOTA_EXC_NAMES:
        return QUOTA_ERROR
    if names & _INFRA_EXC_NAMES:
        return INFRA_ERROR
    return classify_api_error(str(exc))


def is_context_overflow_text(text: str) -> bool:
    """True when an error MESSAGE means "the conversation no longer fits".

    Split out from `is_context_overflow` because the two backends learn this
    differently: the Pydantic backend gets an exception, while the Claude CLI reports
    API failures as an ordinary assistant text block inside a ResultMessage. Only the
    exception form was handled, so on the DEFAULT backend a context overflow was
    classified as INFRA_ERROR — it incremented `_infra_failures` (three retire the
    whole swarm, blaming the gateway) and `consecutive_errors` (five retire the
    solver), while `context_overflow` stayed False and the overflow-specific
    coaching and forced clean restart never ran.
    """
    if not text:
        return False
    low = text.lower()
    return any(needle in low for needle in (
        "context window", "maximum context", "too many tokens",
        "prompt is too long", "context length exceeded", "exceeds the model",
        "reduce the length of your input",
    ))


def is_context_overflow(exc: BaseException) -> bool:
    """True when the failure means "the conversation no longer fits"."""
    names = {c.__name__ for c in type(exc).__mro__}
    if names & _CONTEXT_EXC_NAMES:
        return True
    return is_context_overflow_text(str(exc))


@dataclass
class SolverResult:
    flag: str | None
    status: str
    findings_summary: str
    step_count: int
    cost_usd: float
    log_path: str
    # Tool calls that were ACTUALLY dispatched this round (denied/blocked calls
    # excluded). A round with 0 of these is a broken or unreachable solver,
    # never a legitimate "gave up".
    tool_calls: int = 0
    # Set when the round ended because of an API/transport failure.
    api_error: str = ""
    # Set when the wall-clock budget interrupted the turn (Cairn phase 1).
    interrupted: bool = False
    # Text harvested by the phase-2 conclude prompt ("" when harvesting failed,
    # in which case nothing is written to the blackboard).
    concluded: str = ""
    # Round-level token accounting (from ResultMessage.model_usage).
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    # Set when the round failed because the conversation no longer fits the
    # context window — the solver has already dropped history by then, so the
    # next round can proceed.
    context_overflow: bool = False

    @property
    def did_work(self) -> bool:
        """True when the round shows real activity (a tool actually ran)."""
        return self.tool_calls > 0

    @property
    def is_infra_failure(self) -> bool:
        return self.status in (INFRA_ERROR, QUOTA_ERROR) or bool(self.api_error)


class SolverProtocol(Protocol):
    """Common interface for all solver backends (Pydantic AI, Claude SDK).

    `sandbox` is declared read-only on purpose: a mutable protocol attribute has
    to match invariantly, which made the two backends' concrete sandbox types
    (`DockerSandbox | None` vs `DockerSandbox`) unassignable to the protocol.
    Handing a sandbox over is done through `detach_sandbox()`.
    """

    model_spec: str
    agent_name: str

    @property
    def sandbox(self) -> object: ...

    async def start(self) -> None: ...

    async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult: ...

    async def conclude(self, budget_s: int) -> str:
        """Cairn-style phase 2: resume the SAME session with a conclude prompt
        under an independent budget and return the distilled state.

        Returns "" when the backend cannot conclude (no session continuity) or
        when the conclude phase itself fails — callers must then write NOTHING
        to the blackboard rather than storing a partial guess.
        """
        ...

    def bump(self, insights: str) -> None: ...

    def detach_sandbox(self) -> object:
        """Hand ownership of the sandbox back to the caller.

        After this, `stop()` must leave the container alone. Used by the quota
        fallback, which moves a live sandbox from the claude-cli solver onto the
        Pydantic solver instead of paying to rebuild it.
        """
        ...

    async def stop(self) -> None: ...
