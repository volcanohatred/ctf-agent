"""Infrastructure-failure classification — the fix for the ECONNREFUSED run.

The 2026-10-26 benchmark reported 0/10 with "timeout after 600s" on every
challenge. The real cause was in the traces: every turn returned
"API Error: Connection refused". Because the Claude CLI reports API failures as
an ordinary assistant text block rather than raising, that was classified as
GAVE_UP — so the swarm bumped, rotated techniques and escalated skill hints at a
networking problem for hours.

These tests pin the classification down from both directions: real failures must
be caught, and genuine model prose that merely mentions a socket error must not.
"""

from __future__ import annotations

from backend.solver_base import (
    ERROR,
    INFRA_ERROR,
    QUOTA_ERROR,
    classify_api_error,
    classify_api_status,
    classify_exception,
    is_context_overflow,
)

# ---------------------------------------------------------------------------
# Text the CLI actually emits
# ---------------------------------------------------------------------------

REAL_API_ERRORS = [
    "API Error: Connection refused — a firewall or proxy may be blocking it (ECONNREFUSED)",
    "API Error: Connection error.",
    "API Error: 502 Bad Gateway",
    "API Error: Request failed with status code 401",
    "API Request failed: upstream connect error",
    "Connection error — socket hang up",
    "API Error: getaddrinfo ENOTFOUND gateway.internal",
    "API Error: certificate verify failed",
]


def test_real_api_errors_are_classified_as_infra() -> None:
    for text in REAL_API_ERRORS:
        assert classify_api_error(text) == INFRA_ERROR, text


QUOTA_TEXTS = [
    "API Error: 429 Too Many Requests",
    "API Error: rate limit exceeded, retry later",
    "API Error: quota exhausted for this key",
    "API Error: 529 overloaded — the model is at capacity",
    "API Error: insufficient_balance",
]


def test_rate_limit_texts_are_classified_as_quota() -> None:
    """Quota gets its own status so the swarm can fall back to the OpenAI
    endpoint instead of retiring the solver."""
    for text in QUOTA_TEXTS:
        assert classify_api_error(text) == QUOTA_ERROR, text


def test_http_status_in_text_wins_over_keyword_matching() -> None:
    assert classify_api_error("API Error: HTTP 429") == QUOTA_ERROR
    assert classify_api_error("API Error: HTTP 503") == INFRA_ERROR
    assert classify_api_error("API Error: status_code: 403") == INFRA_ERROR


# ---------------------------------------------------------------------------
# False positives — model prose must NOT be swallowed
# ---------------------------------------------------------------------------

MODEL_PROSE = [
    "[fact] The service returns 'connection refused' when the port is closed, "
    "so I probed 1000-2000 and found 31337 open. [excluded] UDP. [next] try pwntools.",
    "I ran `nc host.docker.internal 31337` and it worked. The banner says the "
    "server is a Flask app with a /debug endpoint that leaks the SECRET_KEY. "
    "Next I will forge an admin session cookie.",
    "The binary has a stack buffer overflow in read_input. Canary is disabled, "
    "PIE is on, so I need a leak first. The puts@plt call at 0x4012 gives me one.",
    "Rate limiting on the login form is 5 attempts per minute per IP, so a "
    "brute force needs a proxy rotation. I have 3 proxies working.",
]


def test_model_prose_mentioning_errors_is_not_misclassified() -> None:
    for text in MODEL_PROSE:
        assert classify_api_error(text) is None, text[:70]


def test_empty_and_none_are_safe() -> None:
    assert classify_api_error("") is None
    assert classify_api_error("   ") is None
    assert classify_api_status(None) is None
    assert classify_api_status(0) is None


def test_short_bare_transport_signal_is_caught() -> None:
    """A terse transport error with no 'API Error' prefix still counts."""
    assert classify_api_error("ECONNREFUSED") == INFRA_ERROR
    assert classify_api_error("connection refused") == INFRA_ERROR


def test_long_text_with_a_transport_word_is_treated_as_prose() -> None:
    """Length gates the bare-signal rule, so a long writeup that happens to
    quote a socket error is not thrown away."""
    long_prose = ("The exploit worked. " * 60) + "connection refused appeared once."
    assert classify_api_error(long_prose) is None


# ---------------------------------------------------------------------------
# The TARGET's status codes are findings, not outages
# ---------------------------------------------------------------------------

TARGET_STATUS_FINDINGS = [
    # The central observation of an entire challenge family. Classifying this as
    # an infrastructure failure sent it through publishable(), which deleted it
    # from the blackboard — the team lost its own best evidence.
    "[fact] the endpoint returns HTTP 422 for every blocked class name",
    "HTTP 422 with body 'legacy extension not permitted'",
    "[fact] that input makes the migrator return HTTP 500",
    "status: 403 on /admin but 200 on /migrate — the filter is path-scoped",
    "[fact] the upload succeeded, response was HTTP 201 with a draft id",
    "probed 40 variants: 38 gave HTTP 422, 2 gave HTTP 200 and took 3s longer",
    "[fact] the service answered HTTP 200 and set a session cookie",
]


def test_the_targets_status_codes_are_not_infrastructure_failures() -> None:
    """A solver reporting what the CHALLENGE returned must never be mistaken for a
    report that the GATEWAY failed. These two are indistinguishable to a regex that
    only looks for a status code, and getting it wrong deletes real findings."""
    for text in TARGET_STATUS_FINDINGS:
        assert classify_api_error(text) is None, text[:70]


def test_those_findings_survive_the_blackboard_gate() -> None:
    """The consequence, asserted where it actually bites: publishable() is what
    decides whether a round's harvest reaches the rest of the swarm."""
    from backend.agents.swarm import publishable

    for text in TARGET_STATUS_FINDINGS:
        assert publishable(text), text[:70]


def test_a_gateway_error_still_wins_when_it_is_prefixed() -> None:
    """The fix must not blind the classifier to real outages — the ECONNREFUSED run
    that motivated this whole module still has to be caught."""
    assert classify_api_error("API Error: HTTP 503 from the gateway") == INFRA_ERROR
    assert classify_api_error("API Error: HTTP 429") == QUOTA_ERROR
    assert classify_api_error("Request failed with status code 500") == INFRA_ERROR


def test_a_short_bare_gateway_status_is_still_caught() -> None:
    """No prefix, but the shape is unmistakably a transport error rather than a
    finding about a target."""
    assert classify_api_error("502 Bad Gateway") == INFRA_ERROR
    assert classify_api_error("500 Internal Server Error") == INFRA_ERROR


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------

def test_classify_api_status() -> None:
    assert classify_api_status(429) == QUOTA_ERROR
    assert classify_api_status(529) == QUOTA_ERROR
    assert classify_api_status(401) == INFRA_ERROR
    assert classify_api_status(403) == INFRA_ERROR
    assert classify_api_status(500) == INFRA_ERROR
    assert classify_api_status(502) == INFRA_ERROR
    assert classify_api_status(200) is None


# ---------------------------------------------------------------------------
# Exception classes (pydantic-ai / httpx path)
# ---------------------------------------------------------------------------

class _FakeModelConnectionError(Exception):
    pass


_FakeModelConnectionError.__name__ = "ModelConnectionError"


class _FakeRateLimit(Exception):
    pass


_FakeRateLimit.__name__ = "ModelRateLimitError"


class _FakeContextOverflow(Exception):
    pass


_FakeContextOverflow.__name__ = "ModelContextWindowExceededError"


def test_exception_names_are_matched_across_the_mro() -> None:
    """Matching by class NAME (not import) keeps this working across
    pydantic-ai versions without importing them in a dependency-free module."""
    assert classify_exception(_FakeModelConnectionError("boom")) == INFRA_ERROR
    assert classify_exception(_FakeRateLimit("slow down")) == QUOTA_ERROR
    assert classify_exception(_FakeContextOverflow("too big")) is None


def test_subclassed_transport_errors_are_matched() -> None:
    class Subclassed(_FakeModelConnectionError):
        pass

    assert classify_exception(Subclassed("boom")) == INFRA_ERROR


def test_plain_exception_falls_back_to_text_classification() -> None:
    assert classify_exception(RuntimeError("nothing to see here")) is None
    assert classify_exception(RuntimeError("API Error: Connection refused")) == INFRA_ERROR


def test_context_overflow_detection() -> None:
    assert is_context_overflow(_FakeContextOverflow("x"))
    assert is_context_overflow(RuntimeError("prompt exceeds the context window"))
    assert is_context_overflow(RuntimeError("maximum context length reached"))
    assert not is_context_overflow(RuntimeError("connection refused"))


def test_unknown_exception_is_a_plain_error_not_infra() -> None:
    """A genuine bug must stay ERROR so the consecutive-error guard, not the
    infrastructure guard, handles it."""
    assert classify_exception(ValueError("bad literal")) is None
    assert (classify_exception(ValueError("bad literal")) or ERROR) == ERROR
