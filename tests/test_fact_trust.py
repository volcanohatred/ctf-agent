"""Fact-trust gate: corroboration, not assertion, makes a fact.

The blackboard is the swarm's only shared memory. Every sibling reads it mid-run
and every fresh restart is seeded from it, so an entry is not "one model's note"
— it is inherited as ground truth by everything that comes after. The failure this
guards against was observed live: one solver reported that a class had migrated
successfully, the claim landed on the board looking exactly like a fact three
models had established, and the rest of the swarm spent hours deriving payloads
from a premise that was never true.

These tests pin the gate itself. It is purely structural (backend/dedup.py), so
none of them need a gateway, a sandbox or an LLM.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agents.swarm import ChallengeSwarm
from backend.message_bus import (
    KIND_BROADCAST,
    KIND_CLAIM,
    KIND_EXCLUDED,
    KIND_FACT,
    KIND_TAGS,
    KIND_VERIFIED,
    ChallengeMessageBus,
    Finding,
)


def _settings(**over) -> SimpleNamespace:
    base = dict(
        skills_dir="", skills_host_dir="", host_project_dir="", workspace_root="",
        sandbox_image="ctf-sandbox", container_memory_limit="1g", container_cpu_limit=1.0,
        sandbox_command_timeout_s=60, solver_time_budget_s=0, conclude_timeout_s=5,
        max_attempts_per_challenge=12, max_infra_failures=0, skill_after_bumps=1,
        escalate_after_bumps=0, fresh_restart_rounds=0, summarize_threshold_chars=900,
        summarizer_model="", gateway_base_url="", gateway_api_key="",
        corroborate_facts=True, strategist_every_bumps=0, stale_round_limit=0,
        bump_cooldown_base_s=0, bump_cooldown_step_s=0, bump_cooldown_max_s=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _swarm(**settings_over) -> ChallengeSwarm:
    return ChallengeSwarm(
        challenge_dir=".",
        meta=SimpleNamespace(name="fatjar-relic", category="web", value=500,
                             description="", tags=[], connection_info="", hints=[]),
        ctfd=SimpleNamespace(),
        # record_tokens is what the summarizer path calls; a no-op stub keeps the
        # swarm usable without a CostTracker.
        cost_tracker=SimpleNamespace(by_agent={}, total_cost_usd=0.0,
                                     record_tokens=lambda *a, **k: None),
        settings=_settings(**settings_over),
        model_specs=[],
        no_submit=True,
    )


def _kinds(swarm: ChallengeSwarm) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in swarm.message_bus.findings:
        out.setdefault(f.kind, []).append(f)
    return out


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_single_source_assertion_is_a_claim_not_a_fact() -> None:
    swarm = _swarm()

    await swarm._publish_harvest("m1", "[fact] the uploaded JAR is loaded by the migrator")

    kinds = _kinds(swarm)
    assert kinds.get(KIND_CLAIM), kinds
    assert KIND_FACT not in kinds, "one model's say-so must not become ground truth"


@pytest.mark.asyncio
async def test_independent_corroboration_promotes_both_to_fact() -> None:
    swarm = _swarm()
    line = "[fact] the uploaded JAR is loaded by the migrator"

    await swarm._publish_harvest("m1", line)
    await swarm._publish_harvest("m2", line)

    kinds = _kinds(swarm)
    assert not kinds.get(KIND_CLAIM), "the original claim must be promoted in place"
    # Both entries are facts: m1's promoted, m2's posted as one. The board must
    # never carry the same statement at two different trust levels.
    assert len(kinds[KIND_FACT]) == 2, [(f.model, f.kind) for f in swarm.message_bus.findings]


@pytest.mark.asyncio
async def test_a_solver_cannot_corroborate_itself() -> None:
    """The exact shape a hallucination takes across a session wipe.

    The model restarts with its own claim in the injected state, re-asserts it as
    a "confirmed fact", and — if self-agreement counted — promotes its own
    invention to ground truth. Three rounds of that must still leave a claim.
    """
    swarm = _swarm()
    line = "[fact] com.evil.MarkerException migrated successfully"

    for _ in range(3):
        await swarm._publish_harvest("m1", line)

    kinds = _kinds(swarm)
    assert KIND_FACT not in kinds, "self-agreement is not corroboration"
    # ...and re-asserting must not grow the board either.
    assert len(kinds[KIND_CLAIM]) == 1


@pytest.mark.asyncio
async def test_corroboration_survives_a_volatile_literal() -> None:
    """Two models observing the same thing report different addresses.

    dedup.normalize collapses hex literals and long digit runs, so an ASLR slide
    or a fresh offset does not defeat the gate — otherwise corroboration would
    only ever fire on copy-paste, which is the one case that proves nothing.
    """
    swarm = _swarm()

    await swarm._publish_harvest("m1", "[fact] libc base resolved to 0x7f8a1c000000")
    await swarm._publish_harvest("m2", "[fact] libc base resolved to 0x7f9b2d000000")

    kinds = _kinds(swarm)
    assert KIND_FACT in kinds, [(f.model, f.kind, f.content) for f in swarm.message_bus.findings]
    assert not kinds.get(KIND_CLAIM)


@pytest.mark.asyncio
async def test_different_findings_do_not_corroborate_each_other() -> None:
    """The gate must not be so loose that anything two models said agrees."""
    swarm = _swarm()

    await swarm._publish_harvest("m1", "[fact] the allowlist blocks FileOutputStream")
    await swarm._publish_harvest("m2", "[fact] the JAR is re-signed on upload")

    kinds = _kinds(swarm)
    assert KIND_FACT not in kinds
    assert len(kinds[KIND_CLAIM]) == 2


@pytest.mark.asyncio
async def test_third_agreement_does_not_repost_the_fact() -> None:
    swarm = _swarm()
    line = "[fact] the worker keeps FD 12-48 open across exec"

    for spec in ("m1", "m2", "m3"):
        await swarm._publish_harvest(spec, line)

    facts = [f for f in swarm.message_bus.findings if f.kind == KIND_FACT]
    assert len(facts) == 2, "a fact is established once, not restated per agreement"


@pytest.mark.asyncio
async def test_the_gate_can_be_turned_off() -> None:
    swarm = _swarm(corroborate_facts=False)

    await swarm._publish_harvest("m1", "[fact] something observed directly")

    kinds = _kinds(swarm)
    assert kinds.get(KIND_FACT), "corroborate_facts=false must restore the old behaviour"
    assert KIND_CLAIM not in kinds


# ---------------------------------------------------------------------------
# Dead ends and untagged output are unaffected by the gate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dead_ends_are_still_published_immediately() -> None:
    """A ruled-out direction needs no corroboration — it is negative knowledge,
    and holding it back until a second model wastes a round rediscovering it
    would defeat the entire dead-end ledger."""
    swarm = _swarm()

    await swarm._publish_harvest("m1", "[excluded] $ref chain is blocked by the allowlist")

    kinds = _kinds(swarm)
    assert kinds.get(KIND_EXCLUDED)
    assert KIND_CLAIM not in kinds and KIND_FACT not in kinds


@pytest.mark.asyncio
async def test_untagged_output_is_a_claim_never_a_fact() -> None:
    swarm = _swarm()

    await swarm._publish_harvest("m1", "I think the deserialization sink is reachable")

    kinds = _kinds(swarm)
    assert kinds.get(KIND_CLAIM)
    assert KIND_FACT not in kinds
    assert KIND_EXCLUDED not in kinds, "prose must never be labelled a ruled-out direction"


@pytest.mark.asyncio
async def test_a_corroborated_fact_mixed_with_a_new_claim_splits_correctly() -> None:
    swarm = _swarm()
    shared = "the migrate endpoint returns 422 on a blocked class"

    await swarm._publish_harvest("m1", f"[fact] {shared}")
    await swarm._publish_harvest(
        "m2", f"[fact] {shared}\n[fact] the allowlist is fastjson checkAutoType"
    )

    kinds = _kinds(swarm)
    assert len(kinds[KIND_FACT]) == 2
    # The second line is m2's alone — still unverified.
    lone = [f.content for f in kinds.get(KIND_CLAIM, [])]
    assert any("checkAutoType" in c for c in lone), lone


# ---------------------------------------------------------------------------
# What siblings are actually shown
# ---------------------------------------------------------------------------

def test_claims_get_their_own_section_with_a_reverify_instruction() -> None:
    """The wording is the mechanism. A flat "unverified" label reads as noise and
    gets ignored; naming the failure mode is what changes behaviour."""
    bus = ChallengeMessageBus()
    rendered = bus.format_unread([
        Finding(model="m1", content="JAR is loaded", kind=KIND_CLAIM),
        Finding(model="m2", content="422 on blocked class", kind=KIND_FACT),
        Finding(model="m3", content="probe showed the JAR loads", kind=KIND_VERIFIED),
        Finding(model="m4", content="$ref chain blocked", kind=KIND_EXCLUDED),
    ])

    assert "UNVERIFIED single-agent claims" in rendered
    assert "NOT confirmed" in rendered
    assert "Re-check" in rendered
    assert "nobody has tested them" in rendered
    # Every tier must be its own section, or the distinction is cosmetic only — and
    # ordered strongest first so the weakest evidence is the last thing read.
    assert rendered.index("Dead ends") < rendered.index("Probe-VERIFIED")
    assert rendered.index("Probe-VERIFIED") < rendered.index("Corroborated facts")
    assert rendered.index("Corroborated facts") < rendered.index("UNVERIFIED")
    assert rendered.index("JAR is loaded") > rendered.index("UNVERIFIED")
    assert rendered.index("422 on blocked class") < rendered.index("UNVERIFIED")


def test_a_broadcast_is_never_tagged_as_a_confirmed_fact() -> None:
    """Regression: broadcasts fell through to the default "✓" and arrived in every
    restart payload labelled as a corroborated fact — which is how one
    coordinator's hunch about a gadget chain became the swarm's premise."""
    assert KIND_TAGS[KIND_BROADCAST] != KIND_TAGS[KIND_FACT]
    assert KIND_TAGS[KIND_CLAIM] != KIND_TAGS[KIND_FACT]


@pytest.mark.asyncio
async def test_the_team_digest_carries_the_trust_markers() -> None:
    """The digest is what a restarted solver actually reads. If the markers do not
    survive into it, the gate is defeated one step downstream."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the JAR is loaded by the migrator")
    await swarm._publish_harvest("m2", "[excluded] brute forcing the draft id")

    reports = "\n".join(swarm._collect_team_reports())

    assert f"{KIND_TAGS[KIND_CLAIM]} the JAR is loaded" in reports
    # Dead ends are published as one bulleted blob, so assert on the marker and
    # the content rather than an exact joined form.
    assert KIND_TAGS[KIND_EXCLUDED] in reports and "brute forcing" in reports
    assert f"{KIND_TAGS[KIND_FACT]} " not in reports, "nothing here was corroborated"


# ---------------------------------------------------------------------------
# Bus-level promotion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_claims_reports_how_many_it_promoted() -> None:
    from backend.dedup import normalize

    bus = ChallengeMessageBus()
    await bus.post("m1", "the JAR is loaded", kind=KIND_CLAIM)
    await bus.post("m1", "something else entirely", kind=KIND_CLAIM)

    assert await bus.promote_claims({normalize("the JAR is loaded")}) == 1
    kinds = [f.kind for f in bus.findings]
    assert kinds == [KIND_FACT, KIND_CLAIM]


@pytest.mark.asyncio
async def test_promote_claims_is_a_noop_on_empty_input() -> None:
    bus = ChallengeMessageBus()
    await bus.post("m1", "x", kind=KIND_CLAIM)

    assert await bus.promote_claims(set()) == 0
    assert bus.findings[0].kind == KIND_CLAIM


@pytest.mark.asyncio
async def test_promotion_never_touches_a_dead_end() -> None:
    from backend.dedup import normalize

    bus = ChallengeMessageBus()
    await bus.post("m1", "$ref chain blocked", kind=KIND_EXCLUDED)

    assert await bus.promote_claims({normalize("$ref chain blocked")}) == 0
    assert bus.findings[0].kind == KIND_EXCLUDED


# ---------------------------------------------------------------------------
# The distinction must survive all the way to the post-mortem export
# ---------------------------------------------------------------------------

def _exportable_swarm(bus: ChallengeMessageBus) -> SimpleNamespace:
    import asyncio

    return SimpleNamespace(
        meta=SimpleNamespace(category="web", value=500, description="fatjar relic", tags=[]),
        model_specs=["m1"], solvers={}, cancel_event=asyncio.Event(),
        winner=None, winner_spec="", _total_bumps=24,
        _failed_approaches=["fastjson-gadgets"], _stale_breaks=5, _strategist_pivots=2,
        findings={}, message_bus=bus,
    )


@pytest.mark.asyncio
async def test_the_export_marks_a_claim_differently_from_a_fact() -> None:
    """This is where the gate either pays off or does not. A writeup written from
    an export that renders ⚠ and ✓ with the same bullet re-launders the
    hallucination into the permanent record."""
    from backend.export import build_report, render_markdown

    bus = ChallengeMessageBus()
    await bus.post("m1", "the JAR is loaded by the migrator", kind=KIND_CLAIM)
    await bus.post("m2", "422 on every blocked class", kind=KIND_FACT)

    md = render_markdown(build_report("fatjar-relic", _exportable_swarm(bus), None))

    assert f"{KIND_TAGS[KIND_CLAIM]} **[m1]** the JAR is loaded" in md
    assert f"{KIND_TAGS[KIND_FACT]} **[m2]** 422 on every blocked class" in md
    assert "未经证实" in md, "the export must explain what ⚠ means"


@pytest.mark.asyncio
async def test_the_export_carries_the_audit_counters() -> None:
    from backend.export import build_report

    bus = ChallengeMessageBus()
    await bus.post("m1", "only one model said this", kind=KIND_CLAIM)

    report = build_report("fatjar-relic", _exportable_swarm(bus), None)

    assert report["stale_breaks"] == 5
    assert report["strategist_pivots"] == 2
    assert report["unverified_claims"] == 1
    assert report["corroborated_facts"] == 0


@pytest.mark.asyncio
async def test_the_export_warns_when_almost_nothing_was_confirmed() -> None:
    from backend.export import build_report, render_markdown

    bus = ChallengeMessageBus()
    for i in range(9):
        await bus.post("m1", f"unverified assertion number {i} about the target", kind=KIND_CLAIM)

    md = render_markdown(build_report("fatjar-relic", _exportable_swarm(bus), None))

    assert "从未被第二个模型证实" in md


@pytest.mark.asyncio
async def test_a_healthy_board_produces_no_such_warning() -> None:
    from backend.export import build_report, render_markdown

    bus = ChallengeMessageBus()
    for i in range(9):
        await bus.post("m1", f"corroborated finding number {i} about the target", kind=KIND_FACT)

    md = render_markdown(build_report("fatjar-relic", _exportable_swarm(bus), None))

    assert "从未被第二个模型证实" not in md


def test_the_archived_export_path_keeps_the_counters() -> None:
    """A finished swarm is gone; the archive is all a post-mortem has."""
    from backend.export import build_report, render_markdown

    archived = {
        "category": "web", "value": 500, "winner": None, "bumps": 24,
        "stale_breaks": 5, "strategist_pivots": 2,
        "corroborated_facts": 1, "unverified_claims": 40,
        "shared_feed": [{"model": "m1", "kind": KIND_CLAIM,
                         "content": "the JAR is loaded", "ts": 0.0}],
    }

    report = build_report("fatjar-relic", None, archived)

    assert report["archived"] is True
    assert report["stale_breaks"] == 5
    assert report["unverified_claims"] == 40
    assert "从未被第二个模型证实" in render_markdown(report)
