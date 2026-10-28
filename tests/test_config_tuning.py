"""Tuning-combination validation.

Every one of these combinations is runnable — that is exactly the problem. Each
silently disables a mechanism the operator meant to use, and the symptom is
"the feature did nothing" with no error anywhere. Surfacing them at startup is
far cheaper than debugging that.

The canonical case: MAX_ATTEMPTS_PER_CHALLENGE=6 with FRESH_RESTART_ROUNDS=5 let
the distil->merge cycle complete exactly once, and the merged state drove a
single round before the solver retired — so the merge was computed and barely
used, with nothing in the logs to say so.
"""

from __future__ import annotations

import pytest

from backend.config import Settings, validate_tuning


def _settings(**over) -> Settings:
    base = dict(
        max_attempts_per_challenge=12,
        fresh_restart_rounds=5,
        solver_time_budget_s=1500,
        gateway_base_url="http://gw/v1",
        gateway_api_key="k",
        _env_file=None,
    )
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Aligned configurations produce no noise
# ---------------------------------------------------------------------------

def test_default_settings_produce_no_warnings() -> None:
    assert validate_tuning(_settings()) == []


def test_defaults_allow_two_merge_cycles() -> None:
    """The shipped defaults must let the distil->merge cycle run more than once."""
    s = Settings(_env_file=None)
    assert s.max_attempts_per_challenge == 12
    assert s.fresh_restart_rounds == 5
    # bumps 1..5 -> merge; 6..10 -> merge; 11,12 driven by merged state
    assert s.max_attempts_per_challenge >= 2 * s.fresh_restart_rounds + 1
    # Only the cap/cycle relationship is under test here; a bare Settings has no
    # gateway configured, so the summarizer warning is expected and unrelated.
    assert not [w for w in validate_tuning(s) if "MAX_ATTEMPTS_PER_CHALLENGE" in w]


@pytest.mark.parametrize("cap,fresh", [(11, 5), (12, 5), (16, 5), (21, 5), (6, 2), (3, 1)])
def test_aligned_cap_and_cycle_stay_quiet(cap: int, fresh: int) -> None:
    assert validate_tuning(_settings(max_attempts_per_challenge=cap,
                                     fresh_restart_rounds=fresh)) == []


# ---------------------------------------------------------------------------
# Misaligned combinations are reported
# ---------------------------------------------------------------------------

def test_cap_below_one_cycle_is_reported() -> None:
    """The journal is never merged at all — it just grows as a plain list."""
    warns = validate_tuning(_settings(max_attempts_per_challenge=3, fresh_restart_rounds=5))
    assert len(warns) == 1
    assert "never merged" in warns[0]
    assert "at least 6" in warns[0]


def test_cap_allows_only_one_cycle_is_reported() -> None:
    """The exact shipped-before configuration: merge once, use it for one round."""
    warns = validate_tuning(_settings(max_attempts_per_challenge=6, fresh_restart_rounds=5))
    assert len(warns) == 1
    assert "only 1 merge cycle" in warns[0]
    assert "at least 11" in warns[0]


def test_cap_exactly_at_one_cycle_boundary() -> None:
    warns = validate_tuning(_settings(max_attempts_per_challenge=10, fresh_restart_rounds=5))
    assert len(warns) == 1
    assert "at least 11" in warns[0]


def test_warning_names_both_knobs_so_the_fix_is_obvious() -> None:
    for w in validate_tuning(_settings(max_attempts_per_challenge=6, fresh_restart_rounds=5)):
        assert "MAX_ATTEMPTS_PER_CHALLENGE" in w
        assert "FRESH_RESTART_ROUNDS" in w


# ---------------------------------------------------------------------------
# Unlimited rounds
# ---------------------------------------------------------------------------

def test_unlimited_cap_with_a_time_budget_is_reported() -> None:
    warns = validate_tuning(_settings(max_attempts_per_challenge=0, solver_time_budget_s=1500))
    assert any("unlimited rounds" in w for w in warns)


def test_unlimited_cap_without_a_time_budget_is_not_flagged() -> None:
    """Coordinator mode against a live CTFd legitimately runs until the CTF ends."""
    warns = validate_tuning(_settings(max_attempts_per_challenge=0, solver_time_budget_s=0))
    assert not any("unlimited rounds" in w for w in warns)


# ---------------------------------------------------------------------------
# Summarizer availability
# ---------------------------------------------------------------------------

def test_missing_gateway_warns_that_distillation_degrades() -> None:
    """The mechanism still works — it just loses the model-driven merge."""
    warns = validate_tuning(_settings(gateway_base_url="", gateway_api_key=""))
    assert any("cannot call a summarizer" in w for w in warns)


def test_missing_api_key_alone_also_warns() -> None:
    warns = validate_tuning(_settings(gateway_api_key=""))
    assert any("cannot call a summarizer" in w for w in warns)


def test_no_summarizer_warning_when_distillation_is_disabled() -> None:
    """fresh_restart_rounds=0 means no journal, so no summarizer is needed."""
    warns = validate_tuning(_settings(fresh_restart_rounds=0, gateway_base_url=""))
    assert not any("summarizer" in w for w in warns)


def test_gateway_configured_produces_no_summarizer_warning() -> None:
    assert not any("summarizer" in w for w in validate_tuning(_settings()))


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_partial_settings_object_does_not_crash() -> None:
    """Callers build Settings-like namespaces; validate_tuning must not assume
    every field is present."""
    from types import SimpleNamespace

    assert validate_tuning(SimpleNamespace()) == []
    warns = validate_tuning(SimpleNamespace(max_attempts_per_challenge=3,
                                            fresh_restart_rounds=5))
    assert any("never merged" in w for w in warns)


def test_multiple_problems_are_all_reported() -> None:
    warns = validate_tuning(_settings(
        max_attempts_per_challenge=3, fresh_restart_rounds=5,
        gateway_base_url="", gateway_api_key="",
    ))
    # Three independent defects in one config: the journal is never merged, the
    # summarizer cannot be reached, and neither can the strategist.
    assert len(warns) == 3, warns
    assert any("never merged" in w for w in warns)
    assert any("cannot call a summarizer" in w for w in warns)
    assert any("Strategist can never be consulted" in w for w in warns)


def test_no_strategist_warning_when_it_is_disabled() -> None:
    """strategist_every_bumps=0 means no call is ever attempted, so a missing
    gateway defeats nothing."""
    warns = validate_tuning(_settings(strategist_every_bumps=0,
                                      fresh_restart_rounds=0, gateway_base_url=""))
    assert not any("Strategist" in w for w in warns)


def test_a_short_cap_does_not_warn_about_the_strategist() -> None:
    """Benchmarks deliberately run tiny caps. A cap below the strategist interval
    is a tuning choice, not a silently defeated mechanism, so it must stay quiet
    while the genuine cap/fresh-restart coupling is still reported."""
    warns = validate_tuning(_settings(max_attempts_per_challenge=2, fresh_restart_rounds=0))
    assert not any("STRATEGIST_EVERY_BUMPS" in w for w in warns)
