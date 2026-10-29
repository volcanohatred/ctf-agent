"""Approach rotation — when a handbook counts as tried-and-failed.

The defect this pins: `_next_approach` used to mark the assigned approach as failed
because "the solver is back for a bump", and a solver comes back for a bump when the
wall clock cut it off mid-work, when its context overflowed, when it honestly said
still_working, and on an infrastructure error. None of those is evidence against the
technique.

With SOLVER_TIME_BUDGET_S=1500 and a round cap of 12, one solver interrupted twelve
times produced twelve handbooks recorded as exhausted. That ledger feeds the bump
payload ("已试且失败，勿重复"), the escalation briefing, and the Strategist — so the
Strategist was told twelve techniques had been tried and failed when in reality the
same one had been cut off twelve times, and pivoted the swarm away from it.

These tests need a populated `_approach_queue`, which the rest of the suite never
has (it runs with skills_dir empty), so the rotation logic had no coverage at all.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from backend.agents.swarm import (
    APPROACH_FAILED,
    APPROACH_INTERRUPTED,
    APPROACH_UNENGAGED,
    MAX_SAME_APPROACH_ROUNDS,
    ChallengeSwarm,
)
from backend.approaches import Approach
from backend.solver_base import GAVE_UP, INFRA_ERROR, SolverResult
from backend.trace_read import tool_args_mention

QUEUE = [
    Approach(path="/skills/ctf-pwn/a-first.md", title="A first", pack="ctf-pwn"),
    Approach(path="/skills/ctf-pwn/b-second.md", title="B second", pack="ctf-pwn"),
    Approach(path="/skills/ctf-pwn/c-third.md", title="C third", pack="ctf-pwn"),
    Approach(path="/skills/ctf-pwn/d-fourth.md", title="D fourth", pack="ctf-pwn"),
]


def _settings(**over) -> SimpleNamespace:
    base = dict(
        skills_dir="", skills_host_dir="", host_project_dir="", workspace_root="",
        sandbox_image="ctf-sandbox", container_memory_limit="1g", container_cpu_limit=1.0,
        sandbox_command_timeout_s=60, solver_time_budget_s=0, conclude_timeout_s=5,
        max_attempts_per_challenge=12, max_infra_failures=0, skill_after_bumps=1,
        skill_hints=True, skill_rotate_approaches=True,
        escalate_after_bumps=0, fresh_restart_rounds=0,
        summarize_threshold_chars=900, summarizer_model="", gateway_base_url="",
        gateway_api_key="", corroborate_facts=True, strategist_every_bumps=0,
        stale_round_limit=0, verify_claims=False, audit_contradictions=False,
        bump_cooldown_base_s=0, bump_cooldown_step_s=0, bump_cooldown_max_s=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _swarm(queue: list[Approach] | None = None, **settings_over) -> ChallengeSwarm:
    swarm = ChallengeSwarm(
        challenge_dir=".",
        meta=SimpleNamespace(name="chal", category="pwn", value=100, description="",
                             tags=[], connection_info="", hints=[]),
        ctfd=SimpleNamespace(),
        cost_tracker=SimpleNamespace(by_agent={}, total_cost_usd=0.0,
                                     record_tokens=lambda *a, **k: None),
        settings=_settings(**settings_over),
        model_specs=["m1"],
        no_submit=True,
    )
    swarm._approach_queue = list(queue if queue is not None else QUEUE)
    return swarm


def _result(status: str = GAVE_UP, tool_calls: int = 5, interrupted: bool = False,
            overflow: bool = False, api_error: str = "") -> SolverResult:
    return SolverResult(flag=None, status=status, findings_summary="worked on it",
                        step_count=4, cost_usd=0.01, log_path="", tool_calls=tool_calls,
                        interrupted=interrupted, context_overflow=overflow,
                        api_error=api_error)


def _solver_with_trace(tmp_path, commands: list[str]):
    """A fake solver whose trace records the given tool calls."""
    path = tmp_path / "trace.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for cmd in commands:
            fh.write(json.dumps({"type": "tool_call", "tool": "Bash",
                                 "args": {"command": cmd}}) + "\n")
    return SimpleNamespace(tracer=SimpleNamespace(path=str(path)))


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("result,kwargs", [
    (_result(interrupted=True), dict(timed_out=True, overflowed=False)),
    (_result(overflow=True), dict(timed_out=False, overflowed=True)),
    (_result(status=INFRA_ERROR, api_error="Connection refused"), dict(timed_out=False, overflowed=False)),
    (_result(tool_calls=0), dict(timed_out=False, overflowed=False)),
])
def test_a_round_that_was_cut_off_is_never_a_failure(tmp_path, result, kwargs) -> None:
    """The whole defect. Being interrupted says nothing about the technique."""
    swarm = _swarm()
    swarm._assigned_approach["m1"] = QUEUE[0].path
    solver = _solver_with_trace(tmp_path, [f"cat {QUEUE[0].path}"])

    assert swarm._approach_outcome(solver, "m1", result, **kwargs) == APPROACH_INTERRUPTED


def test_a_completed_round_that_read_the_handbook_is_a_failure(tmp_path) -> None:
    swarm = _swarm()
    swarm._assigned_approach["m1"] = QUEUE[0].path
    solver = _solver_with_trace(tmp_path, [f"sed -n '1,160p' {QUEUE[0].path}", "./exploit.py"])

    assert swarm._approach_outcome(
        solver, "m1", _result(), timed_out=False, overflowed=False) == APPROACH_FAILED


def test_a_handbook_that_was_never_opened_is_unengaged_not_failed(tmp_path) -> None:
    """Handing a model a path plus "先读再做" and then recording that handbook as
    exhausted when the model never opened it is how the ledger fills up with
    techniques nobody tried — and the Strategist reads that ledger."""
    swarm = _swarm()
    swarm._assigned_approach["m1"] = QUEUE[0].path
    solver = _solver_with_trace(tmp_path, ["gdb ./binary", "checksec --file=./binary"])

    assert swarm._approach_outcome(
        solver, "m1", _result(), timed_out=False, overflowed=False) == APPROACH_UNENGAGED


def test_no_assigned_approach_defaults_to_failed(tmp_path) -> None:
    """Nothing to be unengaged with; a completed empty round is just a failure."""
    swarm = _swarm()
    solver = _solver_with_trace(tmp_path, ["ls"])

    assert swarm._approach_outcome(
        solver, "m1", _result(), timed_out=False, overflowed=False) == APPROACH_FAILED


def test_absent_evidence_never_blacklists_a_handbook(tmp_path) -> None:
    """An unreadable trace cannot prove the file was opened — and must not be read
    as proof it was tried and failed either. Of the two errors, blacklisting a
    technique nobody refuted is the worse one: it is the exact corruption this whole
    change exists to remove, and it propagates into the Strategist. So absence of
    evidence resolves to UNENGAGED, which rotates on without recording a failure."""
    swarm = _swarm()
    swarm._assigned_approach["m1"] = QUEUE[0].path
    solver = SimpleNamespace(tracer=SimpleNamespace(path=str(tmp_path / "nope.jsonl")))

    assert swarm._approach_outcome(
        solver, "m1", _result(), timed_out=False, overflowed=False) == APPROACH_UNENGAGED

    swarm._next_approach("m1", APPROACH_UNENGAGED)
    assert swarm._failed_approaches == []


# ---------------------------------------------------------------------------
# Rotation behaviour
# ---------------------------------------------------------------------------

def test_an_interrupt_keeps_the_same_approach_and_records_nothing() -> None:
    swarm = _swarm()
    first = swarm._next_approach("m1", APPROACH_INTERRUPTED)
    assert first is QUEUE[0]

    again = swarm._next_approach("m1", APPROACH_INTERRUPTED)

    assert again is QUEUE[0], "an interrupt is not a reason to abandon the direction"
    assert swarm._failed_approaches == []
    assert swarm._failed_approach_paths == []


def test_a_real_failure_records_and_rotates() -> None:
    swarm = _swarm()
    assert swarm._next_approach("m1", APPROACH_FAILED) is QUEUE[0]

    second = swarm._next_approach("m1", APPROACH_FAILED)

    assert second is QUEUE[1]
    assert swarm._failed_approaches == ["A first"]
    assert swarm._failed_approach_paths == [QUEUE[0].path]


def test_an_unengaged_handbook_rotates_without_being_recorded_as_failed() -> None:
    swarm = _swarm()
    swarm._next_approach("m1", APPROACH_FAILED)      # assign A

    second = swarm._next_approach("m1", APPROACH_UNENGAGED)

    assert second is QUEUE[1], "move on — but do not blacklist what was never tried"
    assert swarm._failed_approaches == []
    assert swarm._failed_approach_paths == []


def test_interrupts_are_bounded_so_a_wrong_handbook_cannot_be_kept_forever() -> None:
    swarm = _swarm()
    seen = [swarm._next_approach("m1", APPROACH_INTERRUPTED)]
    for _ in range(MAX_SAME_APPROACH_ROUNDS + 2):
        seen.append(swarm._next_approach("m1", APPROACH_INTERRUPTED))

    assert seen[0] is QUEUE[0]
    assert seen[MAX_SAME_APPROACH_ROUNDS] is QUEUE[0], "still within the bound"
    assert seen[MAX_SAME_APPROACH_ROUNDS + 1] is QUEUE[1], "rotated after the bound"
    # Rotating away is not the same as refuting it.
    assert swarm._failed_approaches == []


def test_a_success_after_interrupts_resets_the_streak() -> None:
    swarm = _swarm()
    swarm._next_approach("m1", APPROACH_INTERRUPTED)
    swarm._next_approach("m1", APPROACH_FAILED)      # resets the interrupt counter

    assert swarm._approach_interrupts["m1"] == 0
    # And it can now be held again for a fresh run of interrupts.
    assert swarm._next_approach("m1", APPROACH_INTERRUPTED) is QUEUE[1]


def test_twelve_interrupts_do_not_produce_twelve_failed_approaches() -> None:
    """The exact shape of the reported defect, asserted end to end."""
    swarm = _swarm(queue=[Approach(path=f"/skills/p/{i}.md", title=f"t{i}", pack="p")
                          for i in range(20)])

    swarm._next_approach("m1", APPROACH_FAILED)      # first assignment
    for _ in range(12):
        swarm._next_approach("m1", APPROACH_INTERRUPTED)

    assert swarm._failed_approaches == [], (
        "twelve interrupts recorded as twelve exhausted techniques — this is what "
        "the Strategist was being told"
    )


def test_the_old_behaviour_would_have_failed_all_twelve() -> None:
    """Guard the guard: with FAILED outcomes the ledger really does fill up, so the
    test above is proving the distinction and not just an empty code path."""
    swarm = _swarm(queue=[Approach(path=f"/skills/p/{i}.md", title=f"t{i}", pack="p")
                          for i in range(20)])

    for _ in range(13):
        swarm._next_approach("m1", APPROACH_FAILED)

    assert len(swarm._failed_approaches) == 12


def test_rotation_can_be_disabled_independently_of_hints() -> None:
    """The knob that makes the A/B answerable. SKILLS_DIR="" used to turn rotation
    and hint injection off together, so there was no way to tell which one hurt."""
    swarm = _swarm(skill_rotate_approaches=False)

    assert swarm._next_approach("m1", APPROACH_FAILED) is None
    assert swarm._next_approach("m1", APPROACH_INTERRUPTED) is None
    assert swarm._failed_approaches == []
    assert swarm._assigned_approach == {}, "nothing is assigned when rotation is off"


def test_disabled_rotation_leaves_hints_alone() -> None:
    """Separation is the point: turning rotation off must not also silence the
    knowledge-base hint, or the A/B compares the wrong two things."""
    swarm = _swarm(skill_rotate_approaches=False)
    swarm._skill_pack_ids = ["ctf-pwn"]

    assert swarm._next_approach("m1", APPROACH_FAILED) is None
    # _skill_hint only needs pack ids; it does not consult the queue.
    assert isinstance(swarm._skill_hint(1), str)


def test_the_master_switch_disables_triage_so_nothing_is_injected() -> None:
    """SKILL_HINTS=false has to stop the machinery at the source. Gating only
    _skill_hint would still run triage and still build the rotation queue, leaving
    the imperative "本轮专攻新方向" in every bump — the half most suspected of hurting
    weak models would survive the switch meant to measure it."""
    swarm = ChallengeSwarm(
        challenge_dir=".",
        meta=SimpleNamespace(name="chal", category="pwn", value=100, description="heap tcache",
                             tags=["heap"], connection_info="", hints=[]),
        ctfd=SimpleNamespace(),
        cost_tracker=SimpleNamespace(by_agent={}, total_cost_usd=0.0,
                                     record_tokens=lambda *a, **k: None),
        settings=_settings(skill_hints=False),
        model_specs=["m1"],
        no_submit=True,
    )

    assert swarm._skill_pack_ids == [], "triage must not run"
    assert swarm._approach_queue == [], "the rotation queue must not be built"
    assert swarm._skill_hint(3) == ""
    assert swarm._next_approach("m1", APPROACH_FAILED) is None


def test_the_master_switch_defaults_to_on() -> None:
    """Off-by-default would silently change every existing deployment's behaviour
    and make the A/B baseline meaningless."""
    swarm = _swarm()
    assert getattr(swarm.settings, "skill_hints", True) is True


@pytest.mark.asyncio
async def test_no_direction_is_emitted_when_the_master_switch_is_off() -> None:
    swarm = _swarm(skill_hints=False)

    payload = await swarm._build_bump_payload("m1", "no flag", 3, APPROACH_FAILED)

    assert payload == "no flag", "nothing skill-related may be injected"


def test_an_empty_queue_assigns_nothing() -> None:
    swarm = _swarm(queue=[])
    assert swarm._next_approach("m1", APPROACH_FAILED) is None
    assert swarm._failed_approaches == []


# ---------------------------------------------------------------------------
# SKILL_AFTER_BUMPS — 0 means off
# ---------------------------------------------------------------------------

def _spy_swarm(**settings_over):
    """A swarm whose skill router records every render_hint call and its level.

    A spy rather than a real library: SkillRouter.render_hint returns "" when the
    router is disabled, so asserting on the return value cannot tell "the threshold
    gated it off" from "there was no library to render". The call is the evidence.
    """
    swarm = _swarm(**settings_over)
    swarm._skill_pack_ids = ["ctf-pwn"]
    calls: list[int] = []

    def _render_hint(pack_ids, level=1, keywords=None):
        calls.append(level)
        return f"HINT-L{level}"

    swarm.skill_router = SimpleNamespace(enabled=True, render_hint=_render_hint)
    return swarm, calls


def test_zero_disables_hint_injection_entirely() -> None:
    """The requested semantics, and the opposite of what 0 used to do.

    The old gate was `bump_count < threshold`, which can never fire at 0, and the
    level formula `1 + (bump - 0)` then started at L2 — so 0 injected the HEAVIEST
    tier from the very first retry. It was the one value that could not express the
    most common intent.
    """
    swarm, calls = _spy_swarm(skill_after_bumps=0)

    for bump in range(0, 12):
        assert swarm._skill_hint(bump) == "", f"bump {bump} must inject nothing"
    assert calls == [], "the router must not even be consulted"


def test_negative_also_disables() -> None:
    """A negative threshold is nonsense; treating it as "off" beats letting the level
    formula produce a tier index below L1."""
    swarm, calls = _spy_swarm(skill_after_bumps=-1)

    assert swarm._skill_hint(5) == ""
    assert calls == []


def test_one_injects_from_the_first_retry_at_level_one() -> None:
    swarm, calls = _spy_swarm(skill_after_bumps=1)

    assert swarm._skill_hint(1) == "HINT-L1"
    assert calls == [1], "the first retry gets the cheap pointer, not the full menu"


def test_the_ladder_escalates_from_the_configured_threshold() -> None:
    swarm, calls = _spy_swarm(skill_after_bumps=3)

    assert swarm._skill_hint(1) == ""
    assert swarm._skill_hint(2) == ""
    assert calls == [], "nothing before the threshold"

    assert swarm._skill_hint(3) == "HINT-L1"
    assert swarm._skill_hint(4) == "HINT-L2"
    assert swarm._skill_hint(5) == "HINT-L3"
    # Capped: a 12-round run must not keep escalating past the heaviest tier.
    assert swarm._skill_hint(11) == "HINT-L3"
    assert calls == [1, 2, 3, 3]


def test_a_later_threshold_delays_the_heavy_tiers() -> None:
    """The point of the knob: let a solver work alone for its first few retries,
    which is exactly what a strong model wants and what the old 0 made impossible."""
    early, early_calls = _spy_swarm(skill_after_bumps=1)
    late, late_calls = _spy_swarm(skill_after_bumps=4)

    assert early._skill_hint(2) == "HINT-L2"
    assert late._skill_hint(2) == ""
    assert late._skill_hint(4) == "HINT-L1"
    assert early_calls == [2] and late_calls == [1]


def test_zero_still_allows_rotation() -> None:
    """The two switches are independent, and that separation is measurable: assign a
    direction each retry, inject no handbook text."""
    swarm, calls = _spy_swarm(skill_after_bumps=0, skill_rotate_approaches=True)

    assert swarm._next_approach("m1", APPROACH_FAILED) is QUEUE[0]
    assert swarm._skill_hint(5) == ""
    assert calls == []


@pytest.mark.asyncio
async def test_zero_produces_a_bump_payload_with_a_direction_but_no_hint() -> None:
    swarm, calls = _spy_swarm(skill_after_bumps=0)

    payload = await swarm._build_bump_payload("m1", "no flag", 5, APPROACH_FAILED)

    assert "本轮专攻新方向" in payload, "rotation is still on"
    assert "HINT-L" not in payload and "KNOWLEDGE BASE" not in payload
    assert calls == []


def test_the_master_switch_still_wins_over_a_nonzero_threshold() -> None:
    swarm, calls = _spy_swarm(skill_hints=False, skill_after_bumps=1)

    assert swarm._skill_hint(5) == ""
    assert calls == []


# ---------------------------------------------------------------------------
# What the solver is actually told
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_interrupted_solver_is_told_to_continue_not_to_switch() -> None:
    swarm = _swarm()
    swarm._next_approach("m1", APPROACH_FAILED)      # assign A

    payload = await swarm._build_bump_payload(
        "m1", "budget expired", 1, APPROACH_INTERRUPTED)

    assert "继续上一轮的方向" in payload
    assert "A first" in payload
    assert "不是这个方向错了" in payload
    assert "先读再做" not in payload, "an interrupt must not be narrated as a failure"


@pytest.mark.asyncio
async def test_a_failed_solver_is_told_to_switch() -> None:
    swarm = _swarm()
    swarm._next_approach("m1", APPROACH_FAILED)      # assign A

    payload = await swarm._build_bump_payload("m1", "no flag", 1, APPROACH_FAILED)

    assert "本轮专攻新方向" in payload
    assert "B second" in payload
    assert "已试且失败" in payload and "A first" in payload


@pytest.mark.asyncio
async def test_an_unengaged_solver_may_skip_the_handbook() -> None:
    """A weak model that obeyed "先读再做" on an irrelevant file is exactly the
    harm being investigated, so the unengaged path has to permit declining."""
    swarm = _swarm()
    swarm._next_approach("m1", APPROACH_FAILED)

    payload = await swarm._build_restart_payload(
        "m1", "no flag", 1, APPROACH_UNENGAGED)

    assert "可以跳过它" in payload
    assert "并不算被否证" in payload or "不算被否证" in payload


@pytest.mark.asyncio
async def test_the_restart_payload_also_distinguishes_an_interrupt() -> None:
    """A session wipe makes this matter more: the restarted solver no longer
    remembers the work it was cut off in the middle of."""
    swarm = _swarm()
    swarm._next_approach("m1", APPROACH_FAILED)

    payload = await swarm._build_restart_payload(
        "m1", "budget expired", 1, APPROACH_INTERRUPTED)

    assert "接着上一轮的方向做" in payload
    assert "不是这个方向被否证了" in payload


@pytest.mark.asyncio
async def test_disabled_rotation_emits_no_direction_at_all() -> None:
    swarm = _swarm(skill_rotate_approaches=False)

    payload = await swarm._build_bump_payload("m1", "no flag", 1, APPROACH_FAILED)

    assert "本轮专攻新方向" not in payload
    assert "继续上一轮的方向" not in payload
    assert "no flag" in payload


# ---------------------------------------------------------------------------
# The trace scanner
# ---------------------------------------------------------------------------

def test_tool_args_mention_finds_the_path_in_any_command_shape(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([
        json.dumps({"type": "tool_call", "tool": "Bash",
                    "args": {"command": "sed -n '1,160p' /skills/ctf-pwn/heap.md"}}),
        json.dumps({"type": "tool_result", "tool": "Bash", "result": "..."}),
    ]), encoding="utf-8")

    assert tool_args_mention(str(path), "/skills/ctf-pwn/heap.md")
    assert not tool_args_mention(str(path), "/skills/ctf-pwn/other.md")


def test_tool_args_mention_is_case_insensitive_and_matches_substrings(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"type": "tool_call", "tool": "Bash",
                                "args": {"command": "grep -n tcache /SKILLS/CTF-PWN/Heap.md"}})
                    + "\n", encoding="utf-8")

    assert tool_args_mention(str(path), "/skills/ctf-pwn/heap.md")


def test_tool_args_mention_handles_string_args_and_bad_json(tmp_path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([
        json.dumps({"type": "tool_call", "tool": "Read", "args": "/skills/x.md"}),
        "{not json",
        json.dumps({"type": "tool_call", "tool": "Bash"}),          # no args at all
        "",
    ]), encoding="utf-8")

    assert tool_args_mention(str(path), "/skills/x.md")


def test_tool_args_mention_is_safe_on_a_missing_file_and_empty_needle(tmp_path) -> None:
    assert not tool_args_mention(str(tmp_path / "nope.jsonl"), "/skills/x.md")
    assert not tool_args_mention(str(tmp_path / "nope.jsonl"), "")
    assert not tool_args_mention(str(tmp_path / "nope.jsonl"), "   ")
