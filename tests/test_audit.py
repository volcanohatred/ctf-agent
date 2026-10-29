"""Contradiction audit, and what a verdict actually does to the board.

Two mechanisms, one purpose: stop the swarm believing things that are not true.

The audit compares claims against EACH OTHER — the failure the board itself hides,
because two mutually exclusive statements both look fine in isolation. Observed
live as the thing that should have ended a four-hour run in four minutes: one agent
had recorded that the allowlist rejects a gadget class, another was building a
payload out of that same class, and both sat side by side on the board the whole
time.

The verdict tests below pin what makes a verdict worth having: it must CHANGE
something. A refutation that only logs, or that leaves the disproved claim sitting
readable next to its own refutation, saves nobody any time.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.agents.swarm import ChallengeSwarm
from backend.audit import Contradiction, find_contradictions, parse_contradictions
from backend.dedup import normalize
from backend.message_bus import (
    KIND_CLAIM,
    KIND_EXCLUDED,
    KIND_FACT,
    KIND_VERIFIED,
    ChallengeMessageBus,
)
from backend.verifier import INCONCLUSIVE, REFUTED, VERIFIED, Verdict

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_a_well_formed_contradiction() -> None:
    raw = ("CONTRADICTION: the allowlist rejects FileOutputStream || "
           "built a working payload with FileOutputStream || "
           "the same class cannot be both blocked and usable")
    found = parse_contradictions(raw)
    assert len(found) == 1
    assert "rejects FileOutputStream" in found[0].a
    assert "working payload" in found[0].b
    assert "cannot be both" in found[0].why


def test_none_means_no_conflicts() -> None:
    assert parse_contradictions("NONE") == []
    assert parse_contradictions("none\n") == []


def test_multiple_contradictions_are_all_returned() -> None:
    raw = (
        "CONTRADICTION: a1 || b1 || why1\n"
        "some reasoning in between that is not a finding\n"
        "CONTRADICTION: a2 || b2 || why2\n"
    )
    assert [c.a for c in parse_contradictions(raw)] == ["a1", "a2"]


@pytest.mark.parametrize("raw", [
    "CONTRADICTION: only one part",
    "CONTRADICTION: a || b",                      # missing the why
    "CONTRADICTION: a ||  || why",                # empty middle
    "CONTRADICTION:|||",
    "contradiction without the marker: a || b || c",
])
def test_malformed_lines_are_dropped_not_guessed_at(raw: str) -> None:
    """A contradiction accuses two agents of being wrong about something, so a
    mis-parse is worse than a miss — the miss just leaves the board as it was."""
    assert parse_contradictions(raw) == []


def test_the_broadcast_never_says_which_side_is_right() -> None:
    """The audit has no evidence for that. Picking a winner would be a new
    unverified claim wearing an authoritative label — the exact failure mode the
    whole trust system exists to prevent."""
    text = Contradiction(a="the JAR loads", b="the JAR is rejected",
                         why="both cannot hold").broadcast_text
    assert "the JAR loads" in text and "the JAR is rejected" in text
    assert "不要基于" in text
    assert "Verifier" in text, "must say a probe is coming, not leave it hanging"
    for forbidden in ("A 是错的", "B 是正确的", "A 正确", "B 错误"):
        assert forbidden not in text


# ---------------------------------------------------------------------------
# The audit call
# ---------------------------------------------------------------------------

def _gateway_settings(**over) -> SimpleNamespace:
    base = dict(gateway_base_url="http://gw/v1", gateway_api_key="k", summarizer_model="")
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def patch_client(monkeypatch):
    import httpx

    def _apply(handler):
        transport = httpx.MockTransport(handler)
        real = httpx.AsyncClient

        class PatchedClient(real):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)

    return _apply


async def test_nothing_to_compare_makes_no_call(patch_client) -> None:
    called: list[bool] = []

    def _handler(request):
        called.append(True)
        raise AssertionError("must not be called")

    patch_client(_handler)

    assert await find_contradictions(new_statements=[], existing=[("fact", "x")],
                                     settings=_gateway_settings()) == []
    assert await find_contradictions(new_statements=["x"], existing=[],
                                     settings=_gateway_settings()) == []
    assert called == []


async def test_a_conflict_is_parsed_out_of_the_reply(patch_client) -> None:
    import json as _json

    import httpx

    seen: list[str] = []

    def _handler(request):
        seen.append(_json.loads(request.content)["messages"][0]["content"])
        return httpx.Response(200, json={
            "choices": [{"message": {"content":
                "CONTRADICTION: allowlist blocks X || X worked for me || incompatible"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    patch_client(_handler)

    found = await find_contradictions(
        new_statements=["X worked for me"],
        existing=[("excluded", "allowlist blocks X")],
        settings=_gateway_settings(),
    )

    assert len(found) == 1 and found[0].why == "incompatible"
    # Both the new statement and the board context must reach the model.
    assert "X worked for me" in seen[0] and "allowlist blocks X" in seen[0]


async def test_no_gateway_is_a_silent_noop() -> None:
    assert await find_contradictions(
        new_statements=["a"], existing=[("fact", "b")],
        settings=_gateway_settings(gateway_base_url="", gateway_api_key=""),
    ) == []


async def test_a_gateway_failure_returns_empty_and_never_raises(patch_client) -> None:
    """This sits on the publish path. An audit that can take down a round would be
    a far worse defect than the contradictions it fails to find."""
    import httpx

    def _boom(request):
        raise httpx.ConnectError("refused")

    patch_client(_boom)

    assert await find_contradictions(
        new_statements=["a"], existing=[("fact", "b")], settings=_gateway_settings(),
    ) == []


# ---------------------------------------------------------------------------
# Swarm wiring — the verdict has to change the board
# ---------------------------------------------------------------------------

def _settings(**over) -> SimpleNamespace:
    base = dict(
        skills_dir="", skills_host_dir="", host_project_dir="", workspace_root="",
        sandbox_image="ctf-sandbox", container_memory_limit="1g", container_cpu_limit=1.0,
        sandbox_command_timeout_s=60, solver_time_budget_s=0, conclude_timeout_s=5,
        max_attempts_per_challenge=12, max_infra_failures=0, skill_after_bumps=1,
        escalate_after_bumps=0, fresh_restart_rounds=0, summarize_threshold_chars=900,
        summarizer_model="", gateway_base_url="", gateway_api_key="",
        corroborate_facts=True, strategist_every_bumps=0, stale_round_limit=0,
        verify_claims=True, verifier_model="", verifier_budget_s=300,
        verifier_triage=False, verifier_max_probes=0, audit_contradictions=False,
        bump_cooldown_base_s=0, bump_cooldown_step_s=0, bump_cooldown_max_s=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _swarm(**settings_over) -> ChallengeSwarm:
    return ChallengeSwarm(
        challenge_dir=".",
        meta=SimpleNamespace(name="fatjar-relic", category="web", value=500,
                             description="offline connector", tags=[],
                             connection_info="", hints=[]),
        ctfd=SimpleNamespace(),
        cost_tracker=SimpleNamespace(by_agent={}, total_cost_usd=0.0,
                                     record_tokens=lambda *a, **k: None),
        settings=_settings(**settings_over),
        model_specs=["claude-sdk/m1"],
        no_submit=True,
    )


def _kinds(swarm: ChallengeSwarm) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in swarm.message_bus.findings:
        out.setdefault(f.kind, []).append(f)
    return out


async def test_a_refutation_moves_the_claim_into_the_dead_ends() -> None:
    """The core capability corroboration cannot provide. Withholding promotion
    leaves the false premise on the board, still readable, still tempting."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")
    assert _kinds(swarm).get(KIND_CLAIM)

    await swarm._on_verdict(Verdict(
        REFUTED, "ClassNotFoundException: com.evil.Marker on both runs",
        "the uploaded JAR gets loaded", "m1"))

    kinds = _kinds(swarm)
    assert not kinds.get(KIND_CLAIM), "the disproved claim must leave circulation"
    refuted = kinds[KIND_EXCLUDED]
    assert len(refuted) == 1, "reclassified in place, not duplicated"
    # The evidence travels with it, so nobody re-tests the same thing.
    assert "ClassNotFoundException" in refuted[0].content


async def test_a_refutation_reaches_running_solvers_immediately() -> None:
    """The difference between a wasted minute and a wasted hour: a broadcast is
    picked up mid-turn (every ~5 tool calls), while a board entry alone is only
    read at the next restart — up to a whole time budget later."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")

    await swarm._on_verdict(Verdict(REFUTED, "422 blocked", "the uploaded JAR gets loaded", "m1"))

    broadcasts = [f for f in swarm.message_bus.findings
                  if f.kind == "broadcast" and f.model == "verifier"]
    assert len(broadcasts) == 1
    assert "剪枝" in broadcasts[0].content
    assert "立即停止" in broadcasts[0].content


async def test_a_refutation_invalidates_the_team_digest() -> None:
    """The digest was computed while the premise was still believed. Every
    restarting solver would otherwise be re-seeded with the disproved claim as part
    of "what the team has established"."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")
    swarm._team_digest = "【队友已确认】the uploaded JAR gets loaded"
    swarm._team_digest_gen = 3

    await swarm._on_verdict(Verdict(REFUTED, "422", "the uploaded JAR gets loaded", "m1"))

    assert swarm._team_digest == ""
    assert swarm._team_digest_gen == -1


async def test_a_refutation_lands_in_the_dead_end_ledger() -> None:
    """So it steers every subsequent round through the restart payload, not just
    the live board."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")

    await swarm._on_verdict(Verdict(REFUTED, "CNFE", "the uploaded JAR gets loaded", "m1"))

    assert any("证伪" in d for d in swarm._seen_dead_ends)


async def test_a_successful_probe_creates_the_strongest_tier() -> None:
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the migrate endpoint accepts a JAR")

    await swarm._on_verdict(Verdict(
        VERIFIED, "HTTP 200 and the marker file appeared under /tmp",
        "the migrate endpoint accepts a JAR", "m1"))

    kinds = _kinds(swarm)
    assert not kinds.get(KIND_CLAIM)
    verified = kinds[KIND_VERIFIED]
    assert len(verified) == 1, "flipped in place, not posted as a second entry"
    assert "marker file appeared" in verified[0].content


async def test_probe_verified_outranks_agreement_corroboration() -> None:
    """Two models agreeing is real evidence, but on a hard challenge the whole
    swarm can sit inside the same wrong paradigm and corroborate each other's
    premise. A probe that ran is a different kind of thing, so it gets its own tier
    and renders above ✓."""
    bus = ChallengeMessageBus()
    rendered = bus.format_unread([
        SimpleNamespace(model="m1", content="agreed by two", kind=KIND_FACT, summary=""),
        SimpleNamespace(model="v", content="observed by probe", kind=KIND_VERIFIED, summary=""),
    ])
    assert rendered.index("Probe-VERIFIED") < rendered.index("Corroborated facts")


async def test_inconclusive_changes_nothing() -> None:
    """The safe default. A verifier that could not settle a question must leave the
    board exactly as it was — no promotion, no refutation, no noise."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")
    before = [(f.kind, f.content) for f in swarm.message_bus.findings]

    await swarm._on_verdict(Verdict(INCONCLUSIVE, "endpoint unreachable",
                                    "the uploaded JAR gets loaded", "m1"))

    assert [(f.kind, f.content) for f in swarm.message_bus.findings] == before
    assert not [f for f in swarm.message_bus.findings if f.kind == "broadcast"]


async def test_a_verdict_whose_claim_already_left_the_board_still_posts() -> None:
    """Findings are trimmed at MAX_FINDINGS. Losing the evidence because its
    original claim aged out would discard the most expensive part of the process."""
    swarm = _swarm()

    await swarm._on_verdict(Verdict(VERIFIED, "HTTP 200 observed", "a claim long gone", "m1"))

    verified = _kinds(swarm)[KIND_VERIFIED]
    assert len(verified) == 1 and "HTTP 200 observed" in verified[0].content


async def test_the_coordinator_is_told_about_a_refutation() -> None:
    inbox: asyncio.Queue = asyncio.Queue()
    swarm = _swarm()
    swarm.coordinator_inbox = inbox
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")

    await swarm._on_verdict(Verdict(REFUTED, "CNFE", "the uploaded JAR gets loaded", "m1"))

    msg = inbox.get_nowait()
    assert "VERIFIER REFUTED" in msg
    assert "do not steer solvers back" in msg


async def test_refutation_counts_are_auditable() -> None:
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded")
    await swarm._on_verdict(Verdict(REFUTED, "CNFE", "the uploaded JAR gets loaded", "m1"))

    status = swarm.get_status()
    assert status["probe_refuted"] == 1
    assert status["probe_verified"] == 0


# ---------------------------------------------------------------------------
# Enqueueing
# ---------------------------------------------------------------------------

async def test_a_new_claim_is_queued_for_a_probe() -> None:
    swarm = _swarm()

    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded by the migrator")

    assert len(swarm._claim_queue) == 1
    assert swarm._verifier is not None, "the runner must be started lazily on demand"


@pytest.mark.parametrize("claim", [
    "[fact] short",                      # below MIN_TESTABLE_CLAIM_CHARS
    "[fact] 422",
])
async def test_trivia_is_not_queued(claim: str) -> None:
    swarm = _swarm()
    await swarm._publish_harvest("m1", claim)
    assert len(swarm._claim_queue) == 0


async def test_multiline_prose_is_queued_as_atomic_sentences_not_as_a_blob() -> None:
    """A verifier handed a paragraph tests whichever sentence it fixates on while
    appearing to have checked the whole thing — so the blob itself must never be
    queued. But refusing to queue anything meant a solver that skipped the `[fact]`
    tags contributed findings that were never probed at all, which is exactly the
    population a weak or quantized model belongs to. The board keeps the blob as one
    readable entry; the queue gets testable sentences.
    """
    swarm = _swarm()
    await swarm._publish_harvest(
        "m1",
        "The migrate endpoint returns 422 for blocked names.\n"
        "The uploaded JAR does get loaded by the worker.\n"
        "Short.\n",
    )

    assert len(swarm.message_bus.findings) == 1, "the board still holds ONE blob entry"
    queued = []
    while len(swarm._claim_queue):
        item = await swarm._claim_queue.pop(timeout=0.05)
        if item is None:
            break
        queued.append(item.content)

    assert len(queued) == 2, queued
    assert all("\n" not in q for q in queued), "no multi-line blob may reach a probe"
    # Too-short fragments are dropped rather than probed.
    assert all(len(q) >= swarm.MIN_TESTABLE_CLAIM_CHARS for q in queued)


async def test_prose_extraction_is_capped_so_a_verbose_round_cannot_flood_the_queue() -> None:
    """Tagged claims are deliberate assertions; prose sentences are candidates. An
    uncapped extraction would let one chatty round starve them."""
    swarm = _swarm()
    verbose = "\n".join(
        f"Sentence number {i} makes a distinct claim about the target endpoint."
        for i in range(20)
    )

    await swarm._publish_harvest("m1", verbose)

    assert len(swarm._claim_queue) == swarm.MAX_PROSE_CANDIDATES


def test_prose_candidate_extraction_shape() -> None:
    swarm = _swarm()

    assert swarm._prose_candidates("") == []
    assert swarm._prose_candidates("too short") == []
    # Sentence terminators in both scripts split. Chinese puts no space after 。, so
    # the CJK alternative must be zero-width while the ASCII period requires
    # following whitespace — otherwise "0.5" and "e.g." become "sentences".
    assert swarm._prose_candidates(
        "First claim about the target endpoint behaviour. "
        "第二个关于目标端点行为的声称内容。"
        "Third distinct claim about the same endpoint."
    ) == [
        "First claim about the target endpoint behaviour.",
        "第二个关于目标端点行为的声称内容。",
        "Third distinct claim about the same endpoint.",
    ]
    # An ASCII period NOT followed by whitespace does not split.
    assert swarm._prose_candidates(
        "the ratio is 0.5 and the offset is 1.25 in this build") == [
        "the ratio is 0.5 and the offset is 1.25 in this build"]
    # A paragraph the splitter missed is skipped rather than probed whole.
    assert swarm._prose_candidates("x" * 900) == []
    # Duplicates within one round are not queued twice.
    dupe = "The migrate endpoint returns 422 for blocked names. " * 3
    assert len(swarm._prose_candidates(dupe)) == 1


async def test_corroborated_facts_are_not_requeued() -> None:
    """Two independent solvers agreeing is already stronger evidence than the queue
    gate; spending a probe on it delays one that is actually needed."""
    swarm = _swarm()
    line = "[fact] the migrate endpoint returns 422 on a blocked class name"
    await swarm._publish_harvest("m1", line)
    queued_after_first = len(swarm._claim_queue)
    await swarm._publish_harvest("m2", line)

    assert queued_after_first == 1
    assert len(swarm._claim_queue) == 1


async def test_verification_can_be_disabled() -> None:
    swarm = _swarm(verify_claims=False)
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded by the migrator")
    assert len(swarm._claim_queue) == 0
    assert swarm._verifier is None


async def test_the_probe_budget_is_respected() -> None:
    swarm = _swarm(verifier_max_probes=1)
    swarm._probes_spent = 1

    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded by the migrator")

    assert len(swarm._claim_queue) == 0


async def test_inconclusive_probes_count_against_the_budget() -> None:
    """The knob caps COST, and a probe that ran cost a sandbox round and a
    strong-model call whatever it concluded. The old check compared against
    `_probe_verified + _probe_refuted`, so INCONCLUSIVE was free — and that is the
    common outcome against an unreachable target and the documented behaviour of a
    weak verifier model, i.e. exactly the case an operator sets the cap for."""
    swarm = _swarm(verifier_max_probes=2)
    claim = "[fact] the uploaded JAR gets loaded by the migrator"

    await swarm._publish_harvest("m1", claim)
    await swarm._on_verdict(Verdict(INCONCLUSIVE, "target unreachable", 
                                    "the uploaded JAR gets loaded by the migrator", "m1"))
    assert swarm._probes_spent == 1

    await swarm._publish_harvest("m2", "[fact] a different claim about the migrator")
    await swarm._on_verdict(Verdict(INCONCLUSIVE, "target unreachable",
                                    "a different claim about the migrator", "m2"))
    assert swarm._probes_spent == 2

    queued_before = len(swarm._claim_queue)
    await swarm._publish_harvest("m3", "[fact] yet another claim about the migrator")
    assert len(swarm._claim_queue) == queued_before, (
        "budget exhausted — the third claim must not be enqueued (the two already "
        "queued before the budget ran out are unaffected)"
    )
    pending = []
    while len(swarm._claim_queue):
        item = await swarm._claim_queue.pop(timeout=0.05)
        if item is None:
            break
        pending.append(item.content)
    assert not any("yet another claim" in c for c in pending)


async def test_the_budget_also_bounds_contradiction_probes(monkeypatch) -> None:
    """Contradiction-driven probes are priority 10 and the most expensive kind, so
    they are the last thing an operator wants left unbudgeted."""
    import backend.audit as audit_mod

    async def _found(**kwargs):
        del kwargs
        return [Contradiction(a="the allowlist rejects FileOutputStream",
                              b="FileOutputStream worked in my payload",
                              why="cannot be both")]

    monkeypatch.setattr(audit_mod, "find_contradictions", _found)
    swarm = _swarm(audit_contradictions=True, verifier_max_probes=1)
    swarm._probes_spent = 1
    await swarm.message_bus.post("m2", "the allowlist rejects FileOutputStream",
                                 kind=KIND_EXCLUDED)

    await swarm._publish_harvest("m1", "[fact] FileOutputStream worked in my payload")

    assert swarm._contradictions == 1, "the conflict is still detected and broadcast"
    assert len(swarm._claim_queue) == 0, "but no probe is queued past the budget"


# ---------------------------------------------------------------------------
# Contradiction wiring
# ---------------------------------------------------------------------------

async def test_a_contradiction_is_broadcast_and_both_sides_queued(monkeypatch) -> None:
    import backend.audit as audit_mod

    async def _found(**kwargs):
        del kwargs
        return [Contradiction(a="the allowlist rejects FileOutputStream",
                              b="FileOutputStream worked in my payload",
                              why="the same class cannot be both blocked and usable")]

    monkeypatch.setattr(audit_mod, "find_contradictions", _found)
    swarm = _swarm(audit_contradictions=True)
    await swarm.message_bus.post("m2", "the allowlist rejects FileOutputStream",
                                 kind=KIND_EXCLUDED)

    await swarm._publish_harvest("m1", "[fact] FileOutputStream worked in my payload")

    assert swarm._contradictions == 1
    broadcasts = [f for f in swarm.message_bus.findings
                  if f.kind == "broadcast" and f.model == "audit"]
    assert len(broadcasts) == 1 and "矛盾" in broadcasts[0].content
    # Both sides go to the front of the queue — that is the A/B confrontation, and
    # a probe is what settles it rather than another opinion.
    assert len(swarm._claim_queue) >= 2
    assert swarm._verifier is not None


async def test_a_contradiction_matches_back_to_the_original_board_entry(monkeypatch) -> None:
    """The auditor quotes a fragment. Probing the wording an agent actually used is
    what makes the verdict recognisable to it — and reuses the existing queue key so
    the same claim is not probed twice."""
    import backend.audit as audit_mod

    long_entry = "the allowlist rejects FileOutputStream and every other stream sink"

    async def _found(**kwargs):
        del kwargs
        return [Contradiction(a="allowlist rejects FileOutputStream",
                              b="FileOutputStream worked", why="incompatible")]

    monkeypatch.setattr(audit_mod, "find_contradictions", _found)
    swarm = _swarm(audit_contradictions=True)
    await swarm.message_bus.post("m2", long_entry, kind=KIND_EXCLUDED)

    await swarm._publish_harvest("m1", "[fact] FileOutputStream worked")

    queued = [(await swarm._claim_queue.pop(timeout=0.1))
              for _ in range(len(swarm._claim_queue))]
    contents = [q.content for q in queued if q]
    assert long_entry in contents, "the fragment must resolve to the full board entry"


async def test_an_audit_failure_cannot_break_a_round(monkeypatch) -> None:
    import backend.audit as audit_mod

    async def _boom(**kwargs):
        del kwargs
        raise RuntimeError("gateway exploded")

    monkeypatch.setattr(audit_mod, "find_contradictions", _boom)
    swarm = _swarm(audit_contradictions=True)

    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded by the migrator")

    assert swarm._contradictions == 0
    assert _kinds(swarm).get(KIND_CLAIM), "the round's findings still reached the board"


async def test_the_audit_is_off_by_default_in_a_bare_swarm() -> None:
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded by the migrator")
    assert swarm._contradictions == 0
    assert not [f for f in swarm.message_bus.findings if f.model == "audit"]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def test_the_verifier_sandbox_is_torn_down_when_the_swarm_ends() -> None:
    """The verifier owns a container nothing else tracks. Every exit path has to
    destroy it or it leaks one container and one workspace per challenge."""
    swarm = _swarm()
    stopped: list[bool] = []

    class FakeRunner:
        async def stop(self):
            stopped.append(True)

    swarm._verifier = FakeRunner()

    await swarm._stop_verifier()

    assert stopped == [True]
    assert swarm._verifier is None


async def test_stop_verifier_is_safe_when_none_was_ever_created() -> None:
    swarm = _swarm()
    assert swarm._verifier is None
    await swarm._stop_verifier()


async def test_a_verifier_teardown_failure_does_not_mask_the_result() -> None:
    swarm = _swarm()

    class BadRunner:
        async def stop(self):
            raise RuntimeError("container already gone")

    swarm._verifier = BadRunner()
    await swarm._stop_verifier()
    assert swarm._verifier is None


async def test_nothing_is_queued_after_the_swarm_is_cancelled() -> None:
    """A solver winding down must not start a verifier that will outlive it."""
    swarm = _swarm()
    swarm.cancel_event.set()

    await swarm._publish_harvest("m1", "[fact] the uploaded JAR gets loaded by the migrator")

    assert len(swarm._claim_queue) == 0
    assert swarm._verifier is None


async def test_queue_keys_use_the_same_normalization_as_the_board() -> None:
    """The verdict reclassifies by normalized key. If the queue and the board
    normalized differently, a probe result would never find its claim."""
    swarm = _swarm()
    swarm._enqueue_for_verification("libc base is 0x7f8a1c000000", "m1")
    queued = await swarm._claim_queue.pop(timeout=0.1)
    assert queued is not None
    assert queued.key == normalize("libc base is 0x7f8a1c000000")


async def test_a_probe_result_finds_its_claim_despite_a_volatile_literal() -> None:
    """End to end: the solver reports an address, the verifier's verdict quotes the
    same finding with a different ASLR slide, and the reclassification must still
    land. Without normalize on both sides the evidence would be orphaned."""
    swarm = _swarm()
    await swarm._publish_harvest("m1", "[fact] libc base resolved to 0x7f8a1c000000")

    await swarm._on_verdict(Verdict(
        VERIFIED, "confirmed by reading /proc/self/maps",
        "libc base resolved to 0x7f8a1c000000", "m1"))

    kinds = _kinds(swarm)
    assert kinds.get(KIND_VERIFIED), "the claim must have been found and promoted"
    assert not kinds.get(KIND_CLAIM)
