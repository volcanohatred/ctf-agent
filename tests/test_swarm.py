"""Swarm main-loop tests with a fake solver — no Docker, no LLM, no network.

This is the code that iterated the most and previously had zero coverage: the
round loop, the two-phase timeout harvest, the dead-end ledger, the escalation
ladder, flag verification and the infrastructure-failure guard. Cairn covers the
equivalent scheduler logic with a mock worker; these tests play that role here.

Every scripted solver is bounded: `max_attempts_per_challenge` is set in the
helper so a script that runs out cannot spin forever.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from backend.agents.swarm import ChallengeSwarm, publishable, split_tagged
from backend.message_bus import KIND_CLAIM, KIND_EXCLUDED, KIND_FACT, ChallengeMessageBus
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    INFRA_ERROR,
    SolverResult,
)

# Every test swarm gets a hard round cap so an exhausted script cannot spin.
DEFAULT_CAP = 12


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

@dataclass
class FakeSandbox:
    """Stand-in for DockerSandbox — records stop() so leaks are visible."""

    stopped: bool = False
    is_running: bool = True

    async def stop(self) -> None:
        self.stopped = True


@dataclass
class ScriptedSolver:
    """A solver that replays a scripted list of round outcomes.

    Once the script is exhausted the LAST entry repeats, so a test that forgets
    to make its solver terminate is caught by the swarm's round cap rather than
    hanging the suite.
    """

    model_spec: str
    script: list = field(default_factory=list)
    conclude_results: list = field(default_factory=list)
    sandbox: FakeSandbox = field(default_factory=FakeSandbox)

    agent_name: str = ""
    rounds: int = 0
    bumps: list[str] = field(default_factory=list)
    conclude_calls: int = 0
    resets: int = 0
    forgets: int = 0
    started: bool = False
    stopped: bool = False
    _findings_text: str = ""

    def __post_init__(self) -> None:
        self.agent_name = self.agent_name or self.model_spec

    async def start(self) -> None:
        self.started = True

    async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
        idx = min(self.rounds, len(self.script) - 1) if self.script else 0
        self.rounds += 1
        entry = self.script[idx] if self.script else _gave_up()
        result = entry(self.rounds - 1) if callable(entry) else entry
        # Mirror the real backends: findings are readable as an attribute.
        self._findings_text = result.findings_summary
        return result

    async def conclude(self, budget_s: int) -> str:
        self.conclude_calls += 1
        if budget_s <= 0 or not self.conclude_results:
            return ""
        idx = min(self.conclude_calls - 1, len(self.conclude_results) - 1)
        return self.conclude_results[idx]

    @property
    def findings(self) -> str:
        return self._findings_text

    def bump(self, insights: str) -> None:
        self.bumps.append(insights)

    async def forget_session(self) -> None:
        self.forgets += 1

    async def reset_session(self) -> None:
        self.resets += 1

    async def stop(self) -> None:
        self.stopped = True
        await self.sandbox.stop()


def _gave_up(findings: str = "tried things", tool_calls: int = 4) -> SolverResult:
    return SolverResult(flag=None, status=GAVE_UP, findings_summary=findings,
                        step_count=5, cost_usd=0.01, log_path="", tool_calls=tool_calls)


def _flag(flag: str = "flag{win}") -> SolverResult:
    return SolverResult(flag=flag, status=FLAG_FOUND, findings_summary=f"got {flag}",
                        step_count=9, cost_usd=0.02, log_path="", tool_calls=9)


def _infra(detail: str = "API Error: Connection refused (ECONNREFUSED)") -> SolverResult:
    return SolverResult(flag=None, status=INFRA_ERROR, findings_summary=detail,
                        step_count=0, cost_usd=0.0, log_path="", tool_calls=0,
                        api_error=detail)


def _interrupted(findings: str = "half finished work") -> SolverResult:
    return SolverResult(flag=None, status=GAVE_UP, findings_summary=findings,
                        step_count=7, cost_usd=0.01, log_path="", tool_calls=7,
                        interrupted=True)


def _settings(**over) -> SimpleNamespace:
    base = dict(
        skills_dir="", skills_host_dir="", host_project_dir="", workspace_root="",
        sandbox_image="ctf-sandbox", container_memory_limit="1g", container_cpu_limit=1.0,
        sandbox_command_timeout_s=60, solver_time_budget_s=0, conclude_timeout_s=5,
        max_attempts_per_challenge=DEFAULT_CAP, max_infra_failures=0, skill_after_bumps=1,
        escalate_after_bumps=0, fresh_restart_rounds=0, summarize_threshold_chars=900,
        summarizer_model="", gateway_base_url="", gateway_api_key="",
        coordinator_model="",
        # Zero backoffs so the loop tests run in milliseconds instead of the
        # minutes the production cadence would take.
        bump_cooldown_base_s=0, bump_cooldown_step_s=0, bump_cooldown_max_s=0,
        quota_backoff_base_s=0, quota_backoff_max_s=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _swarm(solvers: dict[str, ScriptedSolver], **settings_over) -> ChallengeSwarm:
    """Build a swarm whose solvers are the supplied fakes (no Docker, no LLM)."""
    swarm = ChallengeSwarm(
        challenge_dir=".",
        meta=SimpleNamespace(name="chal", category="misc", value=100,
                             description="", tags=[], connection_info="", hints=[]),
        ctfd=SimpleNamespace(),
        cost_tracker=SimpleNamespace(by_agent={}, total_cost_usd=0.0),
        settings=_settings(**settings_over),
        model_specs=list(solvers),
        no_submit=True,
        expected_flags=["flag{win}"],
    )
    # Bypass _create_solver: the fakes are pre-built so tests can assert on them.
    swarm._create_solver = lambda spec: solvers[spec]  # type: ignore[method-assign]
    return swarm


def _posted(swarm: ChallengeSwarm) -> list:
    return list(swarm.message_bus.findings)


# ---------------------------------------------------------------------------
# Winner path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flag_found_cancels_siblings_and_stops_sandboxes() -> None:
    winner = ScriptedSolver("m1", [_gave_up(), _flag()])
    loser = ScriptedSolver("m2", [_gave_up()])
    swarm = _swarm({"m1": winner, "m2": loser})

    result = await swarm.run()

    assert result is not None and result.status == FLAG_FOUND
    assert result.flag == "flag{win}"
    assert swarm.winner_spec == "m1"
    assert swarm.cancel_event.is_set()
    # Every solver is stopped and its sandbox destroyed — no orphan containers.
    assert winner.stopped and loser.stopped
    assert winner.sandbox.stopped and loser.sandbox.stopped


@pytest.mark.asyncio
async def test_rejected_flag_claim_does_not_end_the_swarm() -> None:
    """A FLAG_FOUND whose flag the verifier rejected must not cancel siblings.

    Regression for the hallucinated-flag bug: the structured-output path used to
    self-confirm in --no-submit mode, killing three healthy solvers. The real
    backends now clear `flag` unless the verifier accepted it, which is exactly
    the shape asserted here.
    """
    rejected = SolverResult(flag=None, status=GAVE_UP,
                            findings_summary="Claimed flag flag{made_up} was REJECTED",
                            step_count=3, cost_usd=0.0, log_path="", tool_calls=3)
    liar = ScriptedSolver("m1", [rejected])
    honest = ScriptedSolver("m2", [_gave_up(), _flag()])
    swarm = _swarm({"m1": liar, "m2": honest})

    result = await swarm.run()

    assert result is not None and result.flag == "flag{win}"
    assert swarm.winner_spec == "m2"


@pytest.mark.asyncio
async def test_try_submit_flag_is_the_only_confirmation_path() -> None:
    swarm = _swarm({})
    assert swarm.confirmed_flag is None
    display, ok = await swarm.try_submit_flag("flag{win}", "m1")
    assert ok and "CORRECT" in display
    assert swarm.confirmed_flag == "flag{win}"


# ---------------------------------------------------------------------------
# Round cap — the infinite-loop fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_round_cap_retires_a_solver_that_never_wins() -> None:
    """max_attempts_per_challenge must actually bound the round loop.

    The setting existed but was never read, so single-challenge mode looped
    forever on a solver that kept returning GAVE_UP.
    """
    spinner = ScriptedSolver("m1", [_gave_up()])
    swarm = _swarm({"m1": spinner}, max_attempts_per_challenge=3)

    result = await swarm.run()

    assert result is None or result.status != FLAG_FOUND
    assert spinner.rounds == 3, f"expected exactly 3 rounds, got {spinner.rounds}"
    # 3 attempts means 2 transitions — the cap is checked before the next bump.
    assert len(spinner.bumps) == 2
    assert "round cap" in swarm.retire_reason
    assert spinner.stopped


@pytest.mark.asyncio
async def test_round_cap_zero_means_unlimited_but_still_terminates_on_flag() -> None:
    a = ScriptedSolver("m1", [_gave_up()] * 5 + [_flag()])
    swarm = _swarm({"m1": a}, max_attempts_per_challenge=0)
    result = await swarm.run()
    assert result is not None and result.status == FLAG_FOUND
    assert a.rounds == 6


@pytest.mark.asyncio
async def test_zero_cooldown_still_yields_to_the_event_loop() -> None:
    """A round loop with no backoff must not starve cancellation.

    Without an explicit yield, an unlimited-round solver with zero cooldown ran
    the event loop into the ground: cancel(), the patrol and the dashboard all
    stopped getting scheduled.
    """
    spinner = ScriptedSolver("m1", [_gave_up()])
    swarm = _swarm({"m1": spinner}, max_attempts_per_challenge=0)

    task = asyncio.create_task(swarm.run())
    await asyncio.sleep(0.02)          # only runs if the loop yields
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert spinner.stopped and spinner.sandbox.stopped


# ---------------------------------------------------------------------------
# Infrastructure failure — the ECONNREFUSED-as-GAVE_UP fix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_infra_failures_retire_the_swarm_without_polluting_the_blackboard() -> None:
    dead = ScriptedSolver("m1", [_infra()])
    dead2 = ScriptedSolver("m2", [_infra()])
    swarm = _swarm({"m1": dead, "m2": dead2}, max_infra_failures=3)

    await swarm.run()

    assert "infrastructure" in swarm.retire_reason
    assert swarm.cancel_event.is_set()
    # The gateway error text must never be published as a "ruled out direction".
    assert _posted(swarm) == [], "infra noise reached the blackboard"
    # And it must not be burned as a bump/technique rotation either.
    assert dead.rounds + dead2.rounds <= 4


@pytest.mark.asyncio
async def test_zero_tool_call_round_is_treated_as_infra_not_gave_up() -> None:
    """A GAVE_UP with no tool calls is a broken endpoint, not a hard challenge."""
    empty = ScriptedSolver("m1", [_gave_up(findings="", tool_calls=0)])
    swarm = _swarm({"m1": empty}, max_infra_failures=2)

    await swarm.run()

    assert "infrastructure" in swarm.retire_reason
    assert empty.rounds == 2


@pytest.mark.asyncio
async def test_infra_counter_resets_after_a_healthy_round() -> None:
    script = [_infra(), _gave_up(), _infra(), _flag()]
    solver = ScriptedSolver("m1", script)
    swarm = _swarm({"m1": solver}, max_infra_failures=2)

    result = await swarm.run()

    assert result is not None and result.status == FLAG_FOUND
    assert solver.rounds == 4


# ---------------------------------------------------------------------------
# Two-phase timeout harvest (Cairn-style conclude)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_runs_conclude_before_any_session_reset() -> None:
    """Phase 2 needs session continuity, so conclude must run BEFORE the reset."""
    order: list[str] = []

    class OrderTracking(ScriptedSolver):
        async def conclude(self, budget_s: int) -> str:
            out = await super().conclude(budget_s)
            order.append(f"conclude:{out[:12]}")
            return out

        async def forget_session(self) -> None:
            order.append("forget")
            await super().forget_session()

    solver = OrderTracking("m1", [_interrupted(), _flag()],
                           conclude_results=["[fact] libc base 0x7f00\n[excluded] fmtstr"])
    swarm = _swarm({"m1": solver}, conclude_timeout_s=7, fresh_restart_rounds=5)

    result = await swarm.run()

    assert result is not None and result.status == FLAG_FOUND
    assert solver.conclude_calls == 1
    assert order and order[0].startswith("conclude:"), f"conclude must come first: {order}"
    assert "forget" in order


@pytest.mark.asyncio
async def test_conclude_harvest_is_published_to_the_blackboard() -> None:
    solver = ScriptedSolver("m1", [_interrupted(), _flag()],
                            conclude_results=["[fact] key = 0xDEADBEEF\n[excluded] brute force"])
    swarm = _swarm({"m1": solver}, conclude_timeout_s=7)

    await swarm.run()

    posted = _posted(swarm)
    kinds = {f.kind for f in posted}
    # One solver asserting something is a CLAIM, not a fact. Nothing has checked
    # it, and every sibling plus every future restart payload is built from this
    # board — promoting it on assertion alone is how a hallucination becomes the
    # swarm's ground truth.
    assert KIND_CLAIM in kinds and KIND_EXCLUDED in kinds
    assert KIND_FACT not in kinds
    claim = next(f.content for f in posted if f.kind == KIND_CLAIM)
    excluded = next(f.content for f in posted if f.kind == KIND_EXCLUDED)
    assert "0xDEADBEEF" in claim
    assert "brute force" in excluded


@pytest.mark.asyncio
async def test_failed_conclude_publishes_no_dead_end() -> None:
    """Cairn invariant: a failed conclude phase must not leave a partial fact,
    and untagged fallback text must never be labelled a ruled-out direction."""
    solver = ScriptedSolver("m1", [_interrupted(), _flag()], conclude_results=[""])
    swarm = _swarm({"m1": solver}, conclude_timeout_s=7)

    await swarm.run()

    assert solver.conclude_calls == 1
    assert all(f.kind != KIND_EXCLUDED for f in _posted(swarm))


@pytest.mark.asyncio
async def test_conclude_budget_is_forwarded_to_the_solver() -> None:
    seen: list[int] = []

    class Recording(ScriptedSolver):
        async def conclude(self, budget_s: int) -> str:
            seen.append(budget_s)
            return ""

    solver = Recording("m1", [_interrupted(), _flag()])
    swarm = _swarm({"m1": solver}, conclude_timeout_s=42)

    await swarm.run()

    assert seen == [42]


# ---------------------------------------------------------------------------
# Blackboard hygiene
# ---------------------------------------------------------------------------

def test_split_tagged_routes_lines_by_tag() -> None:
    excluded, facts = split_tagged(
        "[excluded] format string — no write primitive\n"
        "[fact] libc base at 0x7f1234\n"
        "[next] try tcache poisoning\n"
    )
    assert excluded == ["format string — no write primitive"]
    assert facts == ["libc base at 0x7f1234"]
    # [next] is guidance for the next round, not blackboard data
    assert not any("tcache" in f for f in facts + excluded)


def test_split_tagged_handles_multiline_sections() -> None:
    excluded, facts = split_tagged(
        "[excluded] first dead end\n  continuation of it\n[fact] a fact\n"
    )
    assert excluded == ["first dead end", "continuation of it"]
    assert facts == ["a fact"]


def test_untagged_text_is_never_labelled_a_dead_end() -> None:
    """The old behaviour posted every GAVE_UP as 'excluded — do NOT retry',
    which mislabelled confirmed facts (and API errors) as ruled-out directions."""
    excluded, facts = split_tagged("found the libc base at 0x7f00, no leak yet")
    assert excluded == []
    assert facts == ["found the libc base at 0x7f00, no leak yet"]


def test_publishable_rejects_api_errors_and_tracebacks() -> None:
    assert not publishable("API Error: Connection refused — a firewall may be blocking it")
    assert not publishable("Error: something exploded")
    assert not publishable("Traceback (most recent call last): ...")
    assert not publishable("")
    assert not publishable("   ")
    assert publishable("[fact] the key is 0xDEADBEEF")


@pytest.mark.asyncio
async def test_api_error_text_never_reaches_siblings_as_a_dead_end() -> None:
    """Regression for the exact contamination seen in the 2026-10-26 bench logs,
    where "API Error: Connection refused" was published as a ruled-out direction
    and every sibling was told not to retry it."""
    noisy = ScriptedSolver("m1", [_infra("API Error: Connection refused (ECONNREFUSED)")])
    healthy = ScriptedSolver("m2", [_gave_up("the PNG magic bytes are wrong"), _flag()])
    swarm = _swarm({"m1": noisy, "m2": healthy}, max_infra_failures=0,
                   max_attempts_per_challenge=4)

    await swarm.run()

    posted = _posted(swarm)
    for f in posted:
        assert "Connection refused" not in f.content
        assert f.kind != KIND_EXCLUDED or "refused" not in f.content


# ---------------------------------------------------------------------------
# Escalation ladder
# ---------------------------------------------------------------------------

def test_escalation_disabled_at_zero() -> None:
    """`escalate_after_bumps = 0` must disable the ladder, not fall back to 8."""
    swarm = _swarm({}, escalate_after_bumps=0)
    swarm._total_bumps = 999
    swarm._maybe_escalate()
    assert swarm._pending_extra == []


def test_escalation_picks_a_model_not_already_running() -> None:
    """The old code always resolved to DEFAULT_MODELS[0], which was already in
    the lineup — so the ladder silently never fired."""
    swarm = _swarm({}, escalate_after_bumps=1, coordinator_model="kimi-k3")
    swarm._all_specs = ["claude-sdk/qwen3.8-max"]

    swarm._total_bumps = 1
    swarm._maybe_escalate()

    assert swarm._pending_extra == ["claude-sdk/kimi-k3"]
    assert swarm._escalated is True


def test_escalation_skips_a_candidate_that_is_already_running() -> None:
    swarm = _swarm({}, escalate_after_bumps=1, coordinator_model="kimi-k3")
    swarm._all_specs = ["claude-sdk/qwen3.8-max", "claude-sdk/kimi-k3"]

    swarm._total_bumps = 3
    swarm._maybe_escalate()

    assert "claude-sdk/kimi-k3" not in swarm._pending_extra
    # it must fall through to the next strongest candidate instead of giving up
    assert swarm._pending_extra, "escalation should have picked another model"


def test_escalation_reports_exhausted_when_no_stronger_model_exists() -> None:
    from backend.models import DEFAULT_MODELS, ESCALATION_CANDIDATES

    specs = list({*DEFAULT_MODELS, *(f"claude-sdk/{c}" for c in ESCALATION_CANDIDATES)})
    swarm = _swarm({}, escalate_after_bumps=1)
    swarm._all_specs = specs

    swarm._total_bumps = 5
    swarm._maybe_escalate()

    assert swarm._pending_extra == []
    assert swarm._escalation_exhausted is True
    # and it does not keep re-deciding on every subsequent bump
    swarm._total_bumps = 50
    swarm._maybe_escalate()
    assert swarm._pending_extra == []


def test_escalation_waits_for_the_bump_threshold() -> None:
    swarm = _swarm({}, escalate_after_bumps=4, coordinator_model="kimi-k3")
    swarm._total_bumps = 3
    swarm._maybe_escalate()
    assert swarm._pending_extra == []
    swarm._total_bumps = 4
    swarm._maybe_escalate()
    assert swarm._pending_extra == ["claude-sdk/kimi-k3"]


# ---------------------------------------------------------------------------
# Flag submission gating
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wrong_flag_submission_triggers_escalating_cooldown() -> None:
    swarm = _swarm({})
    display, ok = await swarm.try_submit_flag("flag{nope}", "m1")
    assert ok is False and "INCORRECT" in display

    display2, ok2 = await swarm.try_submit_flag("flag{nope2}", "m1")
    assert ok2 is False and "COOLDOWN" in display2


@pytest.mark.asyncio
async def test_duplicate_flag_submission_is_deduped() -> None:
    swarm = _swarm({})
    await swarm.try_submit_flag("flag{dupe}", "m1")
    display, ok = await swarm.try_submit_flag("flag{dupe}", "m2")
    assert ok is False and "already tried" in display


@pytest.mark.asyncio
async def test_answer_key_confirms_and_blocks_further_submissions() -> None:
    swarm = _swarm({})
    display, ok = await swarm.try_submit_flag("flag{win}", "m1")
    assert ok is True and "CORRECT" in display

    display2, ok2 = await swarm.try_submit_flag("flag{other}", "m2")
    assert ok2 is True and "ALREADY SOLVED" in display2


@pytest.mark.asyncio
async def test_answer_key_is_case_insensitive() -> None:
    swarm = _swarm({})
    _display, ok = await swarm.try_submit_flag("FLAG{WIN}", "m1")
    assert ok is True


@pytest.mark.asyncio
async def test_empty_flag_is_rejected_without_consuming_a_submission() -> None:
    swarm = _swarm({})
    display, ok = await swarm.try_submit_flag("   ", "m1")
    assert ok is False and "Empty" in display
    assert swarm._submit_count == {}


# ---------------------------------------------------------------------------
# Context overflow — recoverable, and must not be counted as an error
# ---------------------------------------------------------------------------

def _overflow(findings: str = "partial work before the window filled") -> SolverResult:
    r = SolverResult(flag=None, status=ERROR, findings_summary=findings,
                     step_count=4, cost_usd=0.01, log_path="", tool_calls=4,
                     api_error="prompt exceeds the context window")
    r.context_overflow = True
    return r


@pytest.mark.asyncio
async def test_context_overflow_does_not_count_toward_the_error_retirement() -> None:
    """The solver already self-healed by dropping history, so five overflows must
    not retire a healthy solver via the consecutive-error guard."""
    solver = ScriptedSolver("m1", [_overflow()] * 4 + [_flag()])
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=0, fresh_restart_rounds=0)

    result = await swarm.run()

    assert result is not None and result.status == FLAG_FOUND, swarm.retire_reason
    assert solver.rounds == 5
    assert "consecutive errors" not in swarm.retire_reason


@pytest.mark.asyncio
async def test_context_overflow_forces_the_fresh_restart_path() -> None:
    """After an overflow the history is already gone, so a continue-session bump
    would talk to an empty session carrying no state."""
    solver = ScriptedSolver("m1", [_overflow(), _flag()])
    # fresh_restart_rounds=0 means the plain bump path — overflow must override it
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=DEFAULT_CAP,
                   fresh_restart_rounds=0)

    await swarm.run()

    assert solver.forgets >= 1, "history was not reset after a context overflow"
    assert any("context window" in b for b in solver.bumps), solver.bumps


@pytest.mark.asyncio
async def test_overflow_bump_explains_how_to_avoid_repeating_it() -> None:
    solver = ScriptedSolver("m1", [_overflow(), _flag()])
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=DEFAULT_CAP,
                   fresh_restart_rounds=0)

    await swarm.run()

    payload = "\n".join(solver.bumps)
    # actionable guidance, not just "you failed"
    assert "/challenge/workspace/" in payload


@pytest.mark.asyncio
async def test_genuine_errors_still_retire_after_five() -> None:
    """The overflow exemption must not weaken the real error guard."""
    broken = SolverResult(flag=None, status=ERROR, findings_summary="Error: backend died",
                          step_count=1, cost_usd=0.01, log_path="", tool_calls=1,
                          api_error="backend died")
    solver = ScriptedSolver("m1", [broken])
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=0, fresh_restart_rounds=0)

    await swarm.run()

    assert "consecutive errors" in swarm.retire_reason, swarm.retire_reason
    assert solver.rounds == 5


@pytest.mark.asyncio
async def test_restart_payload_states_why_the_previous_round_ended() -> None:
    """`reason` was accepted by _build_restart_payload and then ignored, so a
    restarted solver was never told it had timed out or blown its context window
    and could not adjust its behaviour."""
    solver = ScriptedSolver("m1", [_interrupted("work in progress"), _flag()])
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=DEFAULT_CAP,
                   fresh_restart_rounds=5, solver_time_budget_s=1500,
                   conclude_timeout_s=5)

    await swarm.run()

    payload = "\n".join(solver.bumps)
    assert "上一轮为何结束" in payload, payload[:400]
    assert "wall-clock time budget" in payload


@pytest.mark.asyncio
async def test_plain_failure_restart_says_so_too() -> None:
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=DEFAULT_CAP,
                   fresh_restart_rounds=5)

    await swarm.run()

    payload = "\n".join(solver.bumps)
    assert "上一轮为何结束" in payload
    assert "did not find the flag" in payload


# ---------------------------------------------------------------------------
# Bump queue — coordinator guidance must survive the swarm's own bump
# ---------------------------------------------------------------------------

def test_bump_queue_does_not_overwrite_pending_guidance() -> None:
    """Regression: bump() stored a single payload, so whoever wrote last won and
    the coordinator's targeted guidance was silently discarded."""
    from backend.agents.claude_solver import ClaudeSolver

    solver = ScriptedSolver("m1", [])
    queue: list[str] = []
    solver.bump = queue.append  # type: ignore[method-assign]

    solver.bump("[operator] the JWT kid is an enum — try SQLi there")
    solver.bump("[swarm] next rotation direction")

    assert len(queue) == 2
    assert "JWT kid" in queue[0]

    # The real backends keep a FIFO; assert the contract on one of them.
    real = ClaudeSolver.__new__(ClaudeSolver)
    from collections import deque

    from backend.tracing import SolverTracer
    real._bump_queue = deque(maxlen=8)
    real.loop_detector = SimpleNamespace(reset=lambda: None)
    real.tracer = SolverTracer("unit-test-bump", "fake")
    real.agent_name = "unit-test-bump/fake"
    real.bump("first")
    real.bump("second")
    assert real.pending_bumps == 2
    assert real._next_prompt().count("first") == 1
    assert real.pending_bumps == 1
    real.tracer.close()


@pytest.mark.asyncio
async def test_kill_records_why_the_swarm_stopped() -> None:
    """An externally killed swarm used to vanish with no winner, no retire_reason
    and no log line — indistinguishable from a crash. Observed live: a coordinator
    killed a redundant swarm and the dashboard could not say why."""
    solver = ScriptedSolver("m1", [_gave_up()])
    swarm = _swarm({"m1": solver}, max_attempts_per_challenge=0)

    task = asyncio.create_task(swarm.run())
    await asyncio.sleep(0.05)
    swarm.kill("challenge already solved on CTFd")
    await task

    assert swarm.cancel_event.is_set()
    assert "already solved" in swarm.retire_reason
    assert swarm.get_status()["retire_reason"] == swarm.retire_reason


@pytest.mark.asyncio
async def test_kill_reason_does_not_overwrite_a_more_specific_one() -> None:
    """A swarm that retired itself on infra failure must keep that diagnosis."""
    dead = ScriptedSolver("m1", [_infra()])
    swarm = _swarm({"m1": dead}, max_infra_failures=1)

    await swarm.run()
    specific = swarm.retire_reason
    assert "infrastructure" in specific

    swarm.kill("some later, vaguer reason")
    assert swarm.retire_reason == specific, "the specific diagnosis was clobbered"


@pytest.mark.asyncio
async def test_advice_mailbox_delivers_once_per_solver() -> None:
    bus = ChallengeMessageBus()
    await bus.post_advice("m1", "first")
    await bus.post_advice("m1", "second")
    await bus.post_advice("m2", "other")

    assert await bus.check_advice("m1") == ["first", "second"]
    assert await bus.check_advice("m1") == []
    assert await bus.check_advice("m2") == ["other"]


@pytest.mark.asyncio
async def test_advice_mailbox_is_bounded() -> None:
    bus = ChallengeMessageBus()
    for i in range(40):
        await bus.post_advice("m1", f"n{i}")
    got = await bus.check_advice("m1")
    assert len(got) == bus.max_advice_per_model
    assert got[-1] == "n39"


# ---------------------------------------------------------------------------
# Dead-end ledger identity
# ---------------------------------------------------------------------------

def test_failed_approach_ledger_uses_path_identity_not_title() -> None:
    """Two handbooks sharing an H1 must not blacklist each other."""
    from backend.approaches import Approach

    swarm = _swarm({})
    swarm._approach_queue = [
        Approach(path="/skills/p/a.md", title="Same Title", pack="p"),
        Approach(path="/skills/p/b.md", title="Same Title", pack="p"),
        Approach(path="/skills/p/c.md", title="Other", pack="p"),
    ]
    first = swarm._next_approach("m1")
    assert first.path == "/skills/p/a.md"

    second = swarm._next_approach("m1")
    assert second.path == "/skills/p/b.md", "b.md was blacklisted by a shared title"
    assert swarm._failed_approach_paths == ["/skills/p/a.md"]


def test_rotation_exhausts_the_queue_then_stops() -> None:
    from backend.approaches import Approach

    swarm = _swarm({})
    swarm._approach_queue = [Approach(path=f"/skills/p/{n}.md", title=n, pack="p")
                             for n in "abc"]
    seen = [swarm._next_approach("m1").path for _ in range(3)]
    assert seen == ["/skills/p/a.md", "/skills/p/b.md", "/skills/p/c.md"]
    assert swarm._next_approach("m1") is None


# ---------------------------------------------------------------------------
# Cancellation hygiene
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_external_cancel_stops_every_solver_and_sandbox() -> None:
    slow = ScriptedSolver("m1", [_gave_up()])
    slow2 = ScriptedSolver("m2", [_gave_up()])
    swarm = _swarm({"m1": slow, "m2": slow2}, max_attempts_per_challenge=0)

    task = asyncio.create_task(swarm.run())
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert slow.stopped and slow2.stopped
    assert slow.sandbox.stopped and slow2.sandbox.stopped


@pytest.mark.asyncio
async def test_solver_raising_is_contained_and_reported() -> None:
    class Exploding(ScriptedSolver):
        async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
            self.rounds += 1
            raise RuntimeError("backend exploded")

    boom = Exploding("m1", [])
    ok = ScriptedSolver("m2", [_flag()])
    swarm = _swarm({"m1": boom, "m2": ok})

    result = await swarm.run()

    assert result is not None and result.status == FLAG_FOUND
    assert boom.stopped


@pytest.mark.asyncio
async def test_cancelled_solver_is_not_rebumped_or_published_as_a_dead_end() -> None:
    """A solver that noticed cancel_event mid-turn must stop cleanly: no extra
    round, no bump, and its partial work must never be labelled a dead end."""

    class SelfCancelling(ScriptedSolver):
        async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
            self.rounds += 1
            return SolverResult(flag=None, status=CANCELLED, findings_summary="was working",
                                step_count=2, cost_usd=0.0, log_path="", tool_calls=2)

    solver = SelfCancelling("m1", [])
    swarm = _swarm({"m1": solver})

    await swarm.run()

    assert solver.rounds == 1, "a cancelled solver must not be re-bumped"
    assert solver.bumps == []
    assert all(f.kind != KIND_EXCLUDED for f in _posted(swarm))


# ---------------------------------------------------------------------------
# Quota fallback
# ---------------------------------------------------------------------------

def test_quota_fallback_mapping_strips_the_effort_suffix() -> None:
    from backend.agents.swarm import _quota_fallback_spec

    assert _quota_fallback_spec("claude-sdk/kimi-k3") == "gateway/kimi-k3"
    assert _quota_fallback_spec("claude-sdk/kimi-k3/high") == "gateway/kimi-k3"
    assert _quota_fallback_spec("claude-sdk/unknown-model") is None


@pytest.mark.asyncio
async def test_the_handed_over_container_is_stopped_after_a_late_failure() -> None:
    """The quota fallback hands the LIVE container to a replacement solver:
    `detach_sandbox()` clears the old solver's `_owns_sandbox`, so its `stop()`
    deliberately leaves the container running and the replacement owns it.

    `_run_solver_loop` returns the replacement, but only on a normal return. If
    anything raises after the handover, the caller's local still refers to the OLD
    solver — whose stop() is now a no-op for the container — and nothing else holds
    a reference. One orphaned container and workspace per occurrence.
    """
    old = ScriptedSolver("m1", [_gave_up()])
    new = ScriptedSolver("m1", [_gave_up()])
    swarm = _swarm({"m1": old})
    # Simulate the post-handover state: the registry points at the replacement
    # while the caller's local still holds the original.
    swarm.solvers["m1"] = new

    await swarm._stop_registered_solver("m1", old)

    assert new.stopped, "the solver that OWNS the container must be stopped"
    assert old.stopped, "the stale reference must also be stopped, in case the handover was partial"


@pytest.mark.asyncio
async def test_stopping_is_idempotent_when_there_was_no_handover() -> None:
    solver = ScriptedSolver("m1", [_gave_up()])
    swarm = _swarm({"m1": solver})

    await swarm._stop_registered_solver("m1", solver)

    assert solver.stopped
    # Stopped exactly once, not twice — the registered solver and the local are the
    # same object, and a second stop() would run teardown on a dead container.
    assert solver.sandbox.stopped


@pytest.mark.asyncio
async def test_a_failing_stop_does_not_prevent_the_other_one() -> None:
    """Teardown of the stale reference must not be able to skip the one that owns
    the container, and must not mask the round's real outcome."""
    class Exploding(ScriptedSolver):
        async def stop(self) -> None:
            raise RuntimeError("container already gone")

    old = Exploding("m1", [_gave_up()])
    new = ScriptedSolver("m1", [_gave_up()])
    swarm = _swarm({"m1": old})
    swarm.solvers["m1"] = new

    await swarm._stop_registered_solver("m1", old)   # must not raise

    assert new.stopped, "the owner was stopped even though the stale one exploded"


@pytest.mark.asyncio
async def test_stopping_an_unregistered_spec_is_safe() -> None:
    swarm = _swarm({})
    await swarm._stop_registered_solver("nobody", None)


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------

def test_get_status_includes_escalated_solvers() -> None:
    """Escalated solvers are not in model_specs; they used to be invisible."""
    solver = ScriptedSolver("m1", [_flag()])
    swarm = _swarm({"m1": solver})
    swarm._all_specs = ["m1", "claude-sdk/escalated"]

    status = swarm.get_status()

    assert set(status["agents"]) == {"m1", "claude-sdk/escalated"}
    assert "retire_reason" in status and "infra_failures" in status


def test_steps_are_ints_for_both_backends() -> None:
    """The Pydantic backend keeps _step_count as a list; the JSON must be an int."""
    from backend.web_dashboard import _as_int

    assert _as_int([42]) == 42
    assert _as_int(42) == 42
    assert _as_int([]) == 0
    assert _as_int(None) == 0
