"""Cost tracking and cache-rate maths.

Two provider conventions collide here. Anthropic — and therefore the claude-sdk
path that the default lineup uses — reports `input_tokens` and
`cache_read_input_tokens` DISJOINTLY, so cache_read can exceed input. The old
rate calculation divided by `input_tokens` alone and rendered that as "378% hit"
in a real run. Pricing has the same hazard: subtracting cache_read from input
under the disjoint convention zeroes out the full-price portion.
"""

from __future__ import annotations

from pydantic_ai.usage import RunUsage

from backend.cost_tracker import (
    FALLBACK_PRICING,
    _cache_rate,
    _uncached_input,
    calc_cost,
)


def _usage(inp: int, cached: int, out: int) -> RunUsage:
    return RunUsage(input_tokens=inp, output_tokens=out, cache_read_tokens=cached)


# ---------------------------------------------------------------------------
# Cache hit rate
# ---------------------------------------------------------------------------

def test_disjoint_provider_never_exceeds_100_percent() -> None:
    """The exact figures from a live run: 11.6k in / 43.9k cached."""
    rate = _cache_rate(_usage(11_600, 43_900, 722))
    assert rate.endswith("%")
    assert int(rate.rstrip("%")) <= 100, rate
    # 43900 / (11600 + 43900) = 79%
    assert rate == "79%", rate


def test_inclusive_provider_uses_input_as_denominator() -> None:
    # cached <= input → treated as a subset of input
    assert _cache_rate(_usage(1000, 800, 10)) == "80%"
    assert _cache_rate(_usage(1000, 500, 10)) == "50%"


def test_zero_input_is_not_a_division_error() -> None:
    assert _cache_rate(_usage(0, 0, 0)) == "n/a"
    assert _cache_rate(_usage(0, 0, 500)) == "n/a"


def test_all_cached_reads_100_percent() -> None:
    assert _cache_rate(_usage(0, 5000, 10)) == "100%"


def test_no_cache_reads_0_percent() -> None:
    assert _cache_rate(_usage(5000, 0, 10)) == "0%"


# ---------------------------------------------------------------------------
# Uncached input (pricing)
# ---------------------------------------------------------------------------

def test_uncached_input_follows_the_same_convention() -> None:
    # disjoint: every input token is billed at full price
    assert _uncached_input(_usage(11_600, 43_900, 0)) == 11_600
    # inclusive: cached portion is billed at the cheaper rate
    assert _uncached_input(_usage(1000, 800, 0)) == 200
    assert _uncached_input(_usage(1000, 0, 0)) == 1000
    assert _uncached_input(_usage(0, 500, 0)) == 0


def test_disjoint_convention_does_not_zero_out_the_bill() -> None:
    """Under the old formula, cache_read > input made uncached 0 and the whole
    prompt looked free."""
    u = _usage(11_600, 43_900, 722)
    assert _uncached_input(u) > 0
    cost = calc_cost(u, "claude-opus-4-6", provider_spec="claude-sdk")
    assert cost > 0, "a 55k-token prompt was priced at zero"


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

def test_reported_cost_wins_over_the_price_table() -> None:
    """Gateway models are priced at 0.0 in the table, so without the
    provider-reported figure every run totals $0.00."""
    assert FALLBACK_PRICING["kimi-k3"]["input"] == 0.0
    from backend.cost_tracker import CostTracker

    t = CostTracker()
    t.record_tokens("chal/kimi-k3", "kimi-k3", input_tokens=10_000,
                    output_tokens=500, provider_spec="gateway", cost_usd=0.42)
    assert round(t.by_agent["chal/kimi-k3"].cost_usd, 2) == 0.42
    assert round(t.total_cost_usd, 2) == 0.42


def test_zero_reported_cost_falls_back_to_the_table() -> None:
    from backend.cost_tracker import CostTracker

    t = CostTracker()
    t.record_tokens("chal/opus", "claude-opus-4-6", input_tokens=1_000_000,
                    output_tokens=10_000, provider_spec="claude-sdk", cost_usd=0.0)
    assert t.by_agent["chal/opus"].cost_usd > 0


def test_record_cost_only_accumulates() -> None:
    """The coordinator's spend used to be logged and then thrown away."""
    from backend.cost_tracker import CostTracker

    t = CostTracker()
    t.record_cost_only("coordinator", "qwen3.8-max", 0.10)
    t.record_cost_only("coordinator", "qwen3.8-max", 0.05)
    assert round(t.by_agent["coordinator"].cost_usd, 3) == 0.15
    assert round(t.total_cost_usd, 3) == 0.15


def test_record_cost_only_ignores_nonpositive() -> None:
    from backend.cost_tracker import CostTracker

    t = CostTracker()
    t.record_cost_only("coordinator", "m", 0.0)
    t.record_cost_only("coordinator", "m", -1.0)
    assert "coordinator" not in t.by_agent


def test_total_tokens_sums_across_agents() -> None:
    from backend.cost_tracker import CostTracker

    t = CostTracker()
    t.record_tokens("a/m1", "m1", input_tokens=100, output_tokens=10)
    t.record_tokens("b/m2", "m2", input_tokens=200, output_tokens=20)
    assert t.total_tokens == 330


def test_get_usage_by_model_groups_across_challenges() -> None:
    from backend.cost_tracker import CostTracker

    t = CostTracker()
    t.record_tokens("chalA/kimi-k3", "kimi-k3", input_tokens=100, output_tokens=10,
                    provider_spec="gateway", cost_usd=0.01)
    t.record_tokens("chalB/kimi-k3", "kimi-k3", input_tokens=200, output_tokens=20,
                    provider_spec="gateway", cost_usd=0.02)
    by_model = t.get_usage_by_model()
    assert by_model["kimi-k3"]["input"] == 300
    assert by_model["kimi-k3"]["output"] == 30
    assert round(by_model["kimi-k3"]["cost"], 3) == 0.03
