"""The Verifier — active falsification.

`corroborate_facts` can only withhold trust. It cannot refute a false premise (the
claim stays on the board, still readable, still tempting) and it cannot confirm a
true one that is hard to rediscover. On the run this was built after, "the uploaded
JAR does get loaded" was never corroborated and never contradicted, and four solvers
spent hours deriving payloads from it.

The safety property that matters most is the asymmetry: an ambiguous, crashed,
evidence-free or zero-tool-call probe must resolve to INCONCLUSIVE and never to a
verdict with consequences. A wrong REFUTED kills the one line of attack that might
have worked; INCONCLUSIVE costs nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from backend.agents.verifier_agent import VerificationRunner
from backend.solver_base import GAVE_UP, SolverResult
from backend.verifier import (
    INCONCLUSIVE,
    REFUTED,
    VERIFIED,
    ClaimQueue,
    PendingClaim,
    Verdict,
    build_claim_task,
    parse_triage,
    parse_verdict,
)

# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("VERDICT: VERIFIED | got HTTP 200 and the marker file appeared", VERIFIED),
    ("VERDICT: REFUTED | ClassNotFoundException: com.evil.Marker", REFUTED),
    ("VERDICT: INCONCLUSIVE | endpoint unreachable", INCONCLUSIVE),
    ("verdict: verified | lowercase still parses", VERIFIED),
    ("VERDICT：REFUTED｜中文冒号和分隔符", REFUTED),
    ("VERDICT: VERIFIED - dash separator", VERIFIED),
    ("VERDICT:VERIFIED|no spaces", VERIFIED),
])
def test_parse_verdict_reads_the_marker(raw: str, expected: str) -> None:
    assert parse_verdict(raw).verdict == expected


def test_parse_verdict_keeps_the_observable_evidence() -> None:
    v = parse_verdict("VERDICT: REFUTED | 422 with body 'legacy extension blocked'")
    assert v.evidence == "422 with body 'legacy extension blocked'"


def test_the_last_verdict_line_wins_not_the_first() -> None:
    """The role prompt asks for deliberation then a verdict. A verifier that
    discusses all three options before choosing would have its reasoning read as
    its conclusion if the first match won."""
    raw = (
        "VERIFIED would be wrong here — the class never loaded.\n"
        "INCONCLUSIVE is tempting but I did get a clean discriminator.\n"
        "Ran the probe twice. Both times: 422 with 'blocked by allowlist'.\n"
        "VERDICT: REFUTED | 422 'blocked by allowlist' on both runs"
    )
    v = parse_verdict(raw)
    assert v.verdict == REFUTED
    assert "both runs" in v.evidence


def test_a_bare_verdict_word_on_the_last_line_is_accepted() -> None:
    assert parse_verdict("did the probe\nREFUTED | no such class").verdict == REFUTED


def test_a_verdict_with_no_evidence_is_downgraded() -> None:
    """This is the whole point of the module. "VERIFIED" with nothing observed is
    exactly the hallucination the Verifier exists to catch — laundering it onto the
    board as ✅ would make the immune system the infection."""
    v = parse_verdict("VERDICT: VERIFIED")
    assert v.verdict == INCONCLUSIVE
    assert "no observable evidence" in v.evidence


@pytest.mark.parametrize("raw", ["", "   ", "I ran some commands and looked at things",
                                 "the class seems to load probably", "VERDICT:"])
def test_unparseable_output_is_always_inconclusive(raw: str) -> None:
    assert parse_verdict(raw).verdict == INCONCLUSIVE


def test_parse_verdict_only_scans_the_tail() -> None:
    """A long transcript with an early stray verdict word must not be mistaken for
    the conclusion — and scanning the whole thing backwards is wasted work."""
    raw = "\n".join([f"noise line {i}" for i in range(40)]
                    + ["VERDICT: VERIFIED | marker file written"])
    assert parse_verdict(raw).verdict == VERIFIED


def test_the_claim_and_source_travel_with_the_verdict() -> None:
    v = parse_verdict("VERDICT: REFUTED | ClassNotFoundException on both runs",
                      claim="the JAR loads", source="m1")
    assert v.claim == "the JAR loads" and v.source == "m1"


# ---------------------------------------------------------------------------
# Verdict shaping
# ---------------------------------------------------------------------------

def test_inconclusive_never_touches_the_board() -> None:
    assert Verdict(INCONCLUSIVE, "x", "claim").is_verdict is False
    assert Verdict(VERIFIED, "x", "claim").is_verdict is True
    assert Verdict(REFUTED, "x", "claim").is_verdict is True


def test_board_text_quotes_both_the_claim_and_the_evidence() -> None:
    """A bare "REFUTED" tells a solver nothing about what to stop doing, and a bare
    evidence string gets re-interpreted by whoever reads it."""
    text = Verdict(REFUTED, "ClassNotFoundException: com.evil.Marker",
                   "the JAR's classes get loaded").board_text
    assert "the JAR's classes get loaded" in text
    assert "ClassNotFoundException" in text
    assert "证伪" in text


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def test_parse_triage_reads_all_three_lines() -> None:
    t = parse_triage(
        "TESTABLE: yes\nLOAD_BEARING: high\nPROBE: upload a JAR with a marker class"
    )
    assert t.testable and t.load_bearing and t.should_verify
    assert "marker class" in t.probe_hint


@pytest.mark.parametrize("raw,should_verify", [
    ("TESTABLE: no\nLOAD_BEARING: high\nPROBE: none", False),
    ("TESTABLE: yes\nLOAD_BEARING: low\nPROBE: do a thing", False),
    ("TESTABLE: no\nLOAD_BEARING: low\nPROBE: none", False),
    ("", False),
    ("the model rambled without using the format", False),
])
def test_triage_defaults_to_not_spending_a_probe(raw: str, should_verify: bool) -> None:
    assert parse_triage(raw).should_verify is should_verify


def test_probe_none_becomes_an_empty_hint() -> None:
    assert parse_triage("TESTABLE: yes\nLOAD_BEARING: high\nPROBE: none").probe_hint == ""


# ---------------------------------------------------------------------------
# Claim task rendering
# ---------------------------------------------------------------------------

def test_claim_task_carries_the_claim_source_and_hint() -> None:
    task = build_claim_task("the JAR loads", "m1", "upload a marker class")
    assert "the JAR loads" in task and "m1" in task and "marker class" in task


def test_claim_task_without_a_hint_says_to_design_one() -> None:
    task = build_claim_task("the JAR loads", "m1", "")
    assert "自己设计" in task


# ---------------------------------------------------------------------------
# Claim queue
# ---------------------------------------------------------------------------

def _claim(content: str, key: str | None = None, priority: int = 0) -> PendingClaim:
    return PendingClaim(content=content, source="m1", key=key or content, priority=priority)


def test_queue_dedupes_by_key() -> None:
    q = ClaimQueue()
    assert q.push(_claim("a", key="k")) is True
    assert q.push(_claim("a again", key="k")) is False
    assert len(q) == 1


async def test_queue_pops_highest_priority_first() -> None:
    """Not FIFO: a claim sitting in a contradiction is worth testing before one
    nobody has disagreed with, whatever order they arrived in."""
    q = ClaimQueue()
    q.push(_claim("low", key="1"))
    q.push(_claim("high", key="2", priority=10))
    q.push(_claim("mid", key="3", priority=5))

    order = [(await q.pop(timeout=0.1)).key for _ in range(3)]
    assert order == ["2", "3", "1"]


def test_queue_bump_priority_reorders() -> None:
    q = ClaimQueue()
    q.push(_claim("first", key="1"))
    q.push(_claim("second", key="2"))
    q.bump_priority("2")

    item = asyncio.run(q.pop(timeout=0.1))
    assert item is not None and item.key == "2"


def test_queue_is_bounded_and_evicts_the_oldest() -> None:
    """An unsolvable challenge produces claims faster than one verifier can probe
    them. Newest claims are more relevant — older ones have usually already been
    superseded or excluded."""
    q = ClaimQueue(maxlen=3)
    for i in range(6):
        q.push(_claim(f"claim {i}", key=str(i)))

    assert len(q) == 3
    keys = {asyncio.run(q.pop(timeout=0.1)).key for _ in range(3)}
    assert keys == {"3", "4", "5"}


def test_release_lets_a_claim_be_requeued() -> None:
    q = ClaimQueue()
    q.push(_claim("a", key="k"))
    asyncio.run(q.pop(timeout=0.1))
    assert q.push(_claim("a", key="k")) is False, "key still reserved while pending"
    q.release("k")
    assert q.push(_claim("a", key="k")) is True


async def test_pop_waits_then_times_out() -> None:
    q = ClaimQueue()
    assert await q.pop(timeout=0.05) is None


async def test_pop_wakes_when_something_is_pushed() -> None:
    q = ClaimQueue()

    async def _later():
        await asyncio.sleep(0.02)
        q.push(_claim("late", key="k"))

    task = asyncio.create_task(_later())
    item = await q.pop(timeout=2.0)
    await task
    assert item is not None and item.key == "k"


def test_clear_empties_everything() -> None:
    q = ClaimQueue()
    q.push(_claim("a", key="k"))
    q.clear()
    assert len(q) == 0
    assert q.push(_claim("a", key="k")) is True


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

@dataclass
class FakeVerifierSolver:
    """Stands in for the solver the runner drives."""

    model_spec: str = "claude-sdk/verifier"
    script: list = field(default_factory=list)
    bumps: list[str] = field(default_factory=list)
    rounds: int = 0
    started: bool = False
    stopped: bool = False
    forgets: int = 0
    conclude_calls: int = 0
    _findings: str = ""

    async def start(self) -> None:
        self.started = True

    async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
        idx = min(self.rounds, len(self.script) - 1) if self.script else 0
        self.rounds += 1
        entry = self.script[idx] if self.script else "VERDICT: INCONCLUSIVE | nothing"
        if isinstance(entry, Exception):
            raise entry
        self._findings = entry
        return SolverResult(flag=None, status=GAVE_UP, findings_summary=entry,
                            step_count=3, cost_usd=0.01, log_path="", tool_calls=4)

    async def conclude(self, budget_s: int) -> str:
        """Phase 2, tools denied. The base fake returns nothing: a test that wants a
        recovered verdict has to say so explicitly, otherwise the forced-commit path
        would look like it worked when it was never exercised."""
        self.conclude_calls += 1
        return ""

    @property
    def findings(self) -> str:
        return self._findings

    def bump(self, insights: str) -> None:
        self.bumps.append(insights)

    async def forget_session(self) -> None:
        self.forgets += 1

    async def stop(self) -> None:
        self.stopped = True


def _fake_swarm(solver: FakeVerifierSolver | None, **settings) -> SimpleNamespace:
    from backend.message_bus import ChallengeMessageBus

    base = dict(verify_claims=True, verifier_model="", verifier_budget_s=300,
                verifier_triage=False, verifier_max_probes=0,
                audit_contradictions=False, gateway_base_url="", gateway_api_key="")
    base.update(settings)
    meta = SimpleNamespace(name="fatjar-relic", category="web", value=500,
                           description="offline connector", tags=[])
    created: list[str] = []

    swarm = SimpleNamespace(
        meta=meta, settings=SimpleNamespace(**base), cancel_event=asyncio.Event(),
        message_bus=ChallengeMessageBus(), cost_tracker=None,
        model_specs=["claude-sdk/m1"], _all_specs=["claude-sdk/m1"],
        coordinator_inbox=None,
        create_verifier=lambda spec, role_prompt: (created.append(spec), solver)[1],
        created_specs=created,
    )
    return swarm


class VerdictCollector:
    """Event-driven sink for `on_verdict`.

    An Event rather than a polling loop: the runner is a long-lived background task
    with no natural completion point, so the test has to be woken by the thing it is
    waiting for instead of asking whether it happened yet.
    """

    def __init__(self, want: int = 1) -> None:
        self.seen: list[Verdict] = []
        self._enough = asyncio.Event()
        self._want = want

    async def __call__(self, verdict: Verdict) -> None:
        self.seen.append(verdict)
        if len(self.seen) >= self._want:
            self._enough.set()

    async def wait(self, timeout: float = 5.0) -> list[Verdict]:
        await asyncio.wait_for(self._enough.wait(), timeout)
        return self.seen


async def test_a_turn_that_ends_with_no_verdict_gets_one_forced_commit() -> None:
    """The defect a live run exposed, and the reason `explicit` exists.

    Observed: the verifier ended its turn mid-exploration ("Let me look at main's
    disassembly") after four tool calls, 74s into a 240s budget, never emitting a
    verdict. Narrating an intention and stopping is normal model behaviour. Without
    a forced commit the probe work is done and the answer thrown away — and a fake
    solver that always returns a verdict line cannot surface this at all.

    The retry must go through conclude(), which denies every tool. A second
    tools-enabled round was tried first and just kept exploring until its own budget
    expired, recovering nothing.
    """
    seen = VerdictCollector()

    class Rambler(FakeVerifierSolver):
        """Explores without concluding, then commits once tools are denied."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.conclude_calls = 0

        async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
            self.rounds += 1
            text = "Let me look at main's disassembly around the show branch."
            self._findings = text
            return SolverResult(flag=None, status=GAVE_UP, findings_summary=text,
                                step_count=3, cost_usd=0.01, log_path="", tool_calls=4)

        async def conclude(self, budget_s: int) -> str:
            self.conclude_calls += 1
            return "VERDICT: REFUTED | write(1, notes[i]) never reached, size is 0"

    solver = Rambler()
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("show leaks the note contents", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == REFUTED, "the forced commit must recover the verdict"
    assert runner.forced_commits == 1
    assert solver.conclude_calls == 1
    assert solver.rounds == 1, "no second tools-enabled round — conclude() only"


async def test_an_explicit_inconclusive_is_not_retried() -> None:
    """Re-asking a model that already decided it could not tell just pressures it
    into guessing — the one outcome worse than no verdict."""
    seen = VerdictCollector()
    solver = FakeVerifierSolver(script=["VERDICT: INCONCLUSIVE | the service is unreachable"])
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("the JAR loads", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == INCONCLUSIVE
    assert runner.forced_commits == 0
    assert solver.conclude_calls == 0, "a deliberate INCONCLUSIVE must not be re-asked"


async def test_a_forced_commit_that_also_fails_stays_inconclusive() -> None:
    """Two passes with no verdict is a real INCONCLUSIVE, not a licence to keep
    pushing until the model says something."""
    seen = VerdictCollector()

    class NeverCommits(FakeVerifierSolver):
        async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
            self.rounds += 1
            text = "Let me check something else first."
            self._findings = text
            return SolverResult(flag=None, status=GAVE_UP, findings_summary=text,
                                step_count=2, cost_usd=0.01, log_path="", tool_calls=3)

        async def conclude(self, budget_s: int) -> str:
            self.conclude_calls += 1
            return ""          # phase 2 also produced nothing

    solver = NeverCommits()
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("the JAR loads", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == INCONCLUSIVE
    assert runner.forced_commits == 1
    assert solver.conclude_calls == 1, "exactly one retry, never a loop"


async def test_a_verdict_buried_in_the_first_pass_is_recovered() -> None:
    """Phase 2 can come back empty even when the probe round did produce a verdict
    line. Concatenating both passes means that answer is not lost."""
    seen = VerdictCollector()

    class BuriedVerdict(FakeVerifierSolver):
        async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
            self.rounds += 1
            text = "I ran the probe.\nSome trailing narration with no marker."
            self._findings = "VERDICT: VERIFIED | marker file appeared\n" + text
            return SolverResult(flag=None, status=GAVE_UP, findings_summary=text,
                                step_count=3, cost_usd=0.01, log_path="", tool_calls=4)

        async def conclude(self, budget_s: int) -> str:
            self.conclude_calls += 1
            return ""

    swarm = _fake_swarm(BuriedVerdict())
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("the note is written out", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == VERIFIED


def test_parse_verdict_marks_whether_a_verdict_line_was_actually_found() -> None:
    assert parse_verdict("VERDICT: INCONCLUSIVE | the service is unreachable").explicit is True
    assert parse_verdict("VERDICT: REFUTED | 422 with body 'legacy extension'").explicit is True
    assert parse_verdict("let me look at the disassembly").explicit is False
    assert parse_verdict("").explicit is False


def test_a_verdict_with_no_evidence_stays_retryable() -> None:
    """explicit=False on purpose. Marking it True would suppress the forced-commit
    pass, permanently losing a verdict the model was one line away from giving —
    which is worse than spending one tools-denied conclude call to ask again."""
    verdict = parse_verdict("VERDICT: VERIFIED")
    assert verdict.verdict == INCONCLUSIVE
    assert verdict.explicit is False
    assert "no observable evidence" in verdict.evidence


@pytest.mark.parametrize("text", [
    # Ordinary prose that happens to start with a verdict word. Measured on the
    # previous regex: parsed as VERIFIED with explicit=True, went straight onto the
    # board at the strongest trust tier.
    "Verified that the uploaded JAR is loaded by the worker.",
    "refuted by nothing in particular",
    "I verified the claim is true",
    # The prompt's own format template echoed back — both ROLE_PROMPT and
    # VERDICT_FORCE spell it out verbatim, so this is a likely reply.
    "VERDICT: VERIFIED | <the observable evidence, quoting the actual output>",
    "VERDICT: REFUTED | <与声称矛盾的观察证据>",
    # Too short to be an observation.
    "VERDICT: VERIFIED | 422",
    "VERDICT: REFUTED | ok",
])
def test_narration_and_template_echoes_never_become_a_verdict(text: str) -> None:
    """The single most damaging hole in the trust gate: ✅ is the tier every other
    agent is told to build on freely, and it was reachable by prose alone."""
    verdict = parse_verdict(text)
    assert verdict.verdict == INCONCLUSIVE, text
    assert not verdict.is_verdict


@pytest.mark.parametrize("text,expected,evidence_has", [
    ("VERDICT: VERIFIED | HTTP 200 and the marker file appeared at /tmp/x",
     VERIFIED, "marker file"),
    ("VERDICT: REFUTED | ClassNotFoundException: com.evil.Marker on both runs",
     REFUTED, "ClassNotFoundException"),
    ("VERIFIED | write(1, notes[i]) never reached, size is 0", VERIFIED, "never reached"),
    ("verdict: refuted | 422 with body 'legacy extension not permitted'",
     REFUTED, "legacy extension"),
])
def test_a_real_verdict_with_real_evidence_is_accepted(text, expected, evidence_has) -> None:
    """The hardening must not over-correct into refusing everything, or the
    verifier becomes decorative in the other direction."""
    verdict = parse_verdict(text)
    assert verdict.verdict == expected
    assert verdict.explicit is True
    assert evidence_has in verdict.evidence


def test_evidence_on_the_following_line_is_recovered() -> None:
    """`(.*)` cannot span lines. Losing the verdict here used to be permanent —
    it was also marked explicit, which suppressed the one retry that could have
    supplied the evidence."""
    verdict = parse_verdict(
        "VERDICT: VERIFIED\nthe marker file appeared at /tmp/x with size 41 bytes")
    assert verdict.verdict == VERIFIED
    assert verdict.explicit is True
    assert "marker file" in verdict.evidence


async def test_the_sandbox_is_created_lazily_on_the_first_claim() -> None:
    """A challenge solved in 40 seconds must never pay for a verifier container."""
    swarm = _fake_swarm(FakeVerifierSolver())
    runner = VerificationRunner(swarm, ClaimQueue(), VerdictCollector())

    assert runner._solver is None
    assert not runner.running


async def test_a_verdict_reaches_the_callback() -> None:
    seen = VerdictCollector()
    solver = FakeVerifierSolver(script=["probe ran\nVERDICT: REFUTED | 422 blocked"])
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)

    runner.queue.push(_claim("the JAR gets loaded", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == REFUTED
    assert "422 blocked" in seen.seen[0].evidence
    assert runner.refuted == 1 and runner.probes_run == 1
    assert solver.started and solver.stopped


async def test_a_verdict_after_zero_tool_calls_is_downgraded() -> None:
    """The same guard the swarm applies to solvers, and it matters more here: an
    unverifiable VERIFIED is worse than no verification at all."""
    seen = VerdictCollector()

    class NoWork(FakeVerifierSolver):
        async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
            return SolverResult(flag=None, status=GAVE_UP,
                                findings_summary="VERDICT: VERIFIED | it works",
                                step_count=0, cost_usd=0.0, log_path="", tool_calls=0)

    swarm = _fake_swarm(NoWork())
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("the JAR gets loaded", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == INCONCLUSIVE
    assert "without running any command" in seen.seen[0].evidence


async def test_a_crashing_probe_reports_inconclusive_and_keeps_going() -> None:
    """One bad claim must not end verification for the whole challenge."""
    seen = VerdictCollector(want=2)
    solver = FakeVerifierSolver(script=[
        RuntimeError("sandbox died"),
        "VERDICT: VERIFIED | marker file appeared",
    ])
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("claim one", key="k1"))
    runner.queue.push(_claim("claim two", key="k2"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    kinds = [v.verdict for v in seen.seen]
    assert INCONCLUSIVE in kinds and VERIFIED in kinds


async def test_triage_filters_a_claim_before_any_sandbox_is_created(monkeypatch) -> None:
    """Triage runs first, so an untestable claim costs no container at all."""
    import backend.verifier as verifier_mod
    from backend.verifier import Triage

    filtered = asyncio.Event()
    calls: list[str] = []

    async def _not_testable(**kwargs):
        calls.append(kwargs["claim"])
        filtered.set()
        return Triage(testable=False, load_bearing=False, probe_hint="")

    # The gateway call is patched, not `_triage`: the runner imports it by name at
    # call time, so this exercises the real triage path including its guard.
    monkeypatch.setattr(verifier_mod, "triage_claim", _not_testable)

    solver = FakeVerifierSolver()
    swarm = _fake_swarm(solver, verifier_triage=True)
    runner = VerificationRunner(swarm, ClaimQueue(), VerdictCollector())
    runner.queue.push(_claim("the author probably intended X", key="k1"))
    runner.ensure_started()
    await asyncio.wait_for(filtered.wait(), timeout=5)
    await runner.stop()

    assert calls == ["the author probably intended X"]
    assert runner.filtered == 1
    assert runner.probes_run == 0
    assert solver.started is False, "no sandbox for a claim triage rejected"


async def test_a_failing_triage_verifies_anyway(monkeypatch) -> None:
    """The opposite of the usual fail-safe direction, deliberately: triage is an
    optimisation, and one that silently disabled verification whenever the gateway
    hiccuped would remove the only mechanism that can catch a false premise.

    The gateway call is patched, NOT `_triage` itself — the guard under test lives
    inside `_triage`, so replacing that method would test nothing.
    """
    import backend.verifier as verifier_mod

    async def _boom(**kwargs):
        del kwargs
        raise RuntimeError("gateway down")

    monkeypatch.setattr(verifier_mod, "triage_claim", _boom)

    seen = VerdictCollector()
    solver = FakeVerifierSolver(script=["VERDICT: REFUTED | 422 with body 'legacy extension'"])
    swarm = _fake_swarm(solver, verifier_triage=True)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("the JAR loads", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert seen.seen[0].verdict == REFUTED, "triage failure must not skip the probe"
    assert runner.filtered == 0


async def test_triage_that_says_testable_supplies_the_probe_hint(monkeypatch) -> None:
    """The hint is handed to the verifier as a starting point — and the task text
    must still tell it to reject a hint that cannot discriminate."""
    import backend.verifier as verifier_mod
    from backend.verifier import Triage

    async def _yes(**kwargs):
        del kwargs
        return Triage(testable=True, load_bearing=True,
                      probe_hint="upload a JAR containing a marker class")

    monkeypatch.setattr(verifier_mod, "triage_claim", _yes)

    seen = VerdictCollector()
    solver = FakeVerifierSolver(script=["VERDICT: VERIFIED | marker loaded"])
    swarm = _fake_swarm(solver, verifier_triage=True)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("the JAR loads", key="k1"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert "marker class" in solver.bumps[0]
    assert "不能真正区分真假" in solver.bumps[0], "a hint is a starting point, not a verdict"


async def test_each_claim_gets_a_fresh_context() -> None:
    """Without the reset the verifier accumulates every earlier transcript and
    starts answering from memory of the last claim — which looks like verification
    and is not."""
    seen = VerdictCollector(want=2)
    solver = FakeVerifierSolver(script=[
        "VERDICT: VERIFIED | marker file appeared at /tmp/x",
        "VERDICT: REFUTED | ClassNotFoundException on both runs",
    ])
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("claim one", key="k1"))
    runner.queue.push(_claim("claim two", key="k2"))
    runner.ensure_started()
    await seen.wait()
    await runner.stop()

    assert solver.forgets == 2
    assert len(solver.bumps) == 2


async def test_the_verifier_spec_is_resolvable() -> None:
    swarm = _fake_swarm(FakeVerifierSolver())
    runner = VerificationRunner(swarm, ClaimQueue(), VerdictCollector())
    assert runner._resolve_spec() == "claude-sdk/m1"

    swarm.settings.verifier_model = "claude-sdk/strong"
    assert runner._resolve_spec() == "claude-sdk/strong"


async def test_stop_tears_the_sandbox_down_even_with_no_claims() -> None:
    """The verifier owns a container nothing else tracks. Every exit path has to
    destroy it or it leaks one container and one workspace per challenge."""
    seen = VerdictCollector()
    solver = FakeVerifierSolver(script=["VERDICT: VERIFIED | marker file appeared at /tmp/x"])
    swarm = _fake_swarm(solver)
    runner = VerificationRunner(swarm, ClaimQueue(), seen)
    runner.queue.push(_claim("c", key="k"))
    runner.ensure_started()
    await seen.wait()

    await runner.stop()

    assert solver.stopped
    assert runner._solver is None
    assert len(runner.queue) == 0
    assert not runner.running


async def test_stop_is_safe_when_nothing_ever_ran() -> None:
    swarm = _fake_swarm(None)
    runner = VerificationRunner(swarm, ClaimQueue(), VerdictCollector())
    await runner.stop()
