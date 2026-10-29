"""Tests for ClaudeSolver's stream consumption and two-phase turn handling.

`_consume_stream` is the most intricate code in the solver: a pump task drains
the SDK's async generator into a queue so that deadline/cancel polling only ever
cancels `Queue.get()` (cancelling an async generator mid-yield can corrupt its
state). It also owns the phase-1 interrupt that makes the phase-2 conclude
harvest possible at all. None of that is reachable without a real CLI, so these
tests drive it with a fake client.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.agents import claude_solver as cs
from backend.agents.claude_solver import ClaudeSolver
from backend.loop_detect import LoopDetector
from backend.tracing import SolverTracer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeTextBlock:
    """Duck-typed stand-in for claude_agent_sdk.TextBlock."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeThinkingBlock:
    def __init__(self, thinking: str) -> None:
        self.thinking = thinking


class FakeAssistantMessage:
    def __init__(self, *blocks) -> None:
        self.content = list(blocks)


class FakeResultMessage:
    def __init__(self, session_id="sess-1", total_cost_usd=0.0, **kw) -> None:
        self.session_id = session_id
        self.total_cost_usd = total_cost_usd
        self.usage = kw.pop("usage", None)
        self.model_usage = kw.pop("model_usage", None)
        self.structured_output = kw.pop("structured_output", None)
        self.is_error = kw.pop("is_error", False)
        self.subtype = kw.pop("subtype", "success")
        self.api_error_status = kw.pop("api_error_status", None)
        self.errors = kw.pop("errors", None)
        self.terminal_reason = kw.pop("terminal_reason", None)
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def _patch_sdk_types(monkeypatch):
    """Swap the SDK message types inside the module under test.

    `_consume_stream` dispatches with `isinstance`, so handing it lookalike
    classes is not enough — the names it checks against have to BE those
    classes. Patching the module namespace keeps the real control flow under
    test without needing a live Claude CLI.
    """
    monkeypatch.setattr(cs, "AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr(cs, "ResultMessage", FakeResultMessage)
    monkeypatch.setattr(cs, "TextBlock", FakeTextBlock)
    monkeypatch.setattr(cs, "ThinkingBlock", FakeThinkingBlock)
    yield


class FakeClient:
    """Yields a scripted message list, then a ResultMessage.

    `stall_after` makes the stream hang instead of finishing, to exercise the
    interrupt/grace paths. `interrupt_delay` is how long the CLI takes to
    produce its ResultMessage after being interrupted.
    """

    def __init__(self, messages=None, result=None, stall_after=None,
                 interrupt_delay=0.0, raise_exc=None):
        self.messages = list(messages or [])
        self.result = result if result is not None else FakeResultMessage()
        self.stall_after = stall_after
        self.interrupt_delay = interrupt_delay
        self.raise_exc = raise_exc
        self.interrupts = 0
        self.queries: list[str] = []
        # An Event rather than a polling sleep: the fake has to model "the CLI
        # winds down after being interrupted" without adding latency or tripping
        # ASYNC110.
        self._interrupt_ev = asyncio.Event()
        self.auto_respond_to_interrupt = True

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def interrupt(self) -> None:
        self.interrupts += 1
        if self.auto_respond_to_interrupt:
            self._interrupt_ev.set()

    async def receive_response(self):
        for i, msg in enumerate(self.messages):
            if self.stall_after is not None and i >= self.stall_after:
                break
            yield msg
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.stall_after is not None:
            # Emulate the CLI: after interrupt it winds down, then reports.
            await self._interrupt_ev.wait()
            if self.interrupt_delay:
                await asyncio.sleep(self.interrupt_delay)
        yield self.result


def make_solver(client, cancel_event=None, session_id="sess-0") -> ClaudeSolver:
    """A ClaudeSolver wired to a fake client, bypassing __init__ (no Docker)."""
    s = ClaudeSolver.__new__(ClaudeSolver)
    s.model_spec = "claude-sdk/kimi-k3"
    s.model_id = "kimi-k3"
    s.agent_name = "unit-stream/kimi-k3"
    s._client = client
    s._session_id = session_id
    s._step_count = 0
    s._tool_calls = 0
    s._flag = None
    s._confirmed = False
    from collections import deque

    s._finding_blocks = deque(maxlen=24)
    s._cost_usd = 0.0
    s._session_cost_hwm = 0.0
    s._last_usage = (0, 0, 0)
    s._bump_queue = deque(maxlen=8)
    s._concluding = False
    s._interrupted = False
    s._externally_cancelled = False
    # Bypassing __init__ means every instance attribute the real constructor sets
    # has to be declared here too, or the method under test raises AttributeError
    # partway through. These two are the Verifier's hooks; empty = stock behaviour.
    s.role_prompt = ""
    s.conclude_prompt = ""
    s._context_overflow = False
    s.cancel_event = cancel_event or asyncio.Event()
    s.loop_detector = LoopDetector()
    s.tracer = SolverTracer("unit-stream", "kimi-k3")
    s.cost_tracker = SimpleNamespace(
        record_tokens=lambda *a, **k: None,
        by_agent={},
    )
    s.message_bus = None
    s.submit_fn = None
    s.no_submit = True
    return s


@pytest.fixture(autouse=True)
def _fast_grace(monkeypatch):
    """Shrink the interrupt grace so the abandon path is testable in ms."""
    monkeypatch.setattr(cs, "INTERRUPT_GRACE_S", 0.25)
    monkeypatch.setattr(cs, "STREAM_POLL_S", 0.02)
    yield


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consumes_until_result_and_collects_text() -> None:
    client = FakeClient(messages=[
        FakeAssistantMessage(FakeTextBlock("first finding")),
        FakeAssistantMessage(FakeTextBlock("second finding")),
    ])
    s = make_solver(client)
    sink: list[str] = []

    result = await s._consume_stream(None, "execute", sink)

    assert result is client.result
    assert sink == ["first finding", "second finding"]
    assert client.interrupts == 0
    assert s._interrupted is False


@pytest.mark.asyncio
async def test_all_text_blocks_are_accumulated_not_overwritten() -> None:
    """Regression: `_findings = block.text` kept only the LAST block, so timeout
    harvesting lost everything the model had confirmed earlier in the round."""
    client = FakeClient(messages=[
        FakeAssistantMessage(FakeTextBlock("libc base 0x7f00")),
        FakeAssistantMessage(FakeTextBlock("pie enabled, no canary")),
        FakeAssistantMessage(FakeTextBlock("overflow at offset 72")),
    ])
    s = make_solver(client)
    sink: list[str] = []

    await s._consume_stream(None, "execute", sink)
    for block in sink:
        s._add_finding(block)

    assert "libc base 0x7f00" in s.findings
    assert "pie enabled" in s.findings
    assert "overflow at offset 72" in s.findings
    s.tracer.close()


@pytest.mark.asyncio
async def test_no_pump_task_is_left_running() -> None:
    client = FakeClient(messages=[FakeAssistantMessage(FakeTextBlock("x"))])
    s = make_solver(client)
    before = len(all_tasks())

    await s._consume_stream(None, "execute", [])
    await asyncio.sleep(0)  # let the finally-block cancellation settle

    assert len(all_tasks()) <= before, "the stream pump task leaked"


def all_tasks() -> set:
    return {t for t in asyncio.all_tasks() if not t.done()}


# ---------------------------------------------------------------------------
# Phase 1: budget interrupt
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_expired_deadline_interrupts_instead_of_cancelling() -> None:
    """The session must survive the budget, or phase 2 has nothing to resume."""
    client = FakeClient(
        messages=[FakeAssistantMessage(FakeTextBlock("partial work"))],
        stall_after=1,
        result=FakeResultMessage(session_id="sess-kept", terminal_reason="aborted_streaming"),
    )
    s = make_solver(client, session_id="sess-kept")
    sink: list[str] = []

    loop = asyncio.get_running_loop()
    already_past = loop.time() - 1.0
    result = await s._consume_stream(already_past, "execute", sink)

    assert client.interrupts == 1, "the turn was not interrupted through the SDK"
    assert s._interrupted is True
    assert s._externally_cancelled is False
    # session continuity is the whole point of interrupting rather than cancelling
    assert s._session_id == "sess-kept"
    assert result is not None
    assert sink == ["partial work"]
    s.tracer.close()


@pytest.mark.asyncio
async def test_external_cancel_interrupts_and_flags_itself() -> None:
    cancel = asyncio.Event()
    client = FakeClient(
        messages=[FakeAssistantMessage(FakeTextBlock("working"))],
        stall_after=1,
    )
    s = make_solver(client, cancel_event=cancel)
    cancel.set()

    await s._consume_stream(None, "execute", [])

    assert client.interrupts == 1
    assert s._externally_cancelled is True, "an external cancel must not look like a budget hit"
    s.tracer.close()


@pytest.mark.asyncio
async def test_interrupted_only_once_per_turn() -> None:
    client = FakeClient(messages=[], stall_after=0, interrupt_delay=0.05)
    s = make_solver(client)
    loop = asyncio.get_running_loop()

    await s._consume_stream(loop.time() - 1.0, "execute", [])

    assert client.interrupts == 1, f"interrupt fired {client.interrupts}x"
    s.tracer.close()


@pytest.mark.asyncio
async def test_abandoned_stream_drops_the_untrusted_session() -> None:
    """If the CLI never answers the interrupt, the session state is unknown and
    must NOT be resumed by phase 2."""
    client = FakeClient(messages=[], stall_after=0)
    client.auto_respond_to_interrupt = False  # the CLI never winds down
    s = make_solver(client, session_id="sess-doomed")

    loop = asyncio.get_running_loop()
    result = await s._consume_stream(loop.time() - 1.0, "execute", [])

    assert result is None, "an abandoned stream must not fabricate a ResultMessage"
    assert s._session_id is None, "an untrusted session was kept for resume"
    s.tracer.close()


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generator_exception_reaches_the_caller() -> None:
    """The pump must not swallow transport errors into a silent empty turn."""
    client = FakeClient(messages=[], raise_exc=RuntimeError("stream blew up"))
    s = make_solver(client)

    with pytest.raises(RuntimeError, match="stream blew up"):
        await s._consume_stream(None, "execute", [])
    s.tracer.close()


@pytest.mark.asyncio
async def test_cancellation_propagates_out_of_the_consumer() -> None:
    client = FakeClient(messages=[], stall_after=0)
    client.auto_respond_to_interrupt = False  # keep the stream open
    s = make_solver(client)

    task = asyncio.create_task(s._consume_stream(None, "execute", []))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not all_tasks() & {task}
    s.tracer.close()


# ---------------------------------------------------------------------------
# Phase 2: conclude
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conclude_uses_a_separate_sink_and_keeps_phase_one_text() -> None:
    """Phase-1 findings and the phase-2 harvest must not overwrite each other."""
    phase1 = FakeClient(messages=[FakeAssistantMessage(FakeTextBlock("phase one work"))])
    s = make_solver(phase1)
    sink: list[str] = []
    await s._consume_stream(None, "execute", sink)
    for b in sink:
        s._add_finding(b)

    s._client = FakeClient(
        messages=[FakeAssistantMessage(FakeTextBlock("[fact] key = 0xBEEF"))],
        result=FakeResultMessage(session_id="sess-1"),
    )
    harvested = await s.conclude(budget_s=5)

    assert harvested == "[fact] key = 0xBEEF"
    # both survive: the round's work AND the distilled state card
    assert "phase one work" in s.findings
    assert "0xBEEF" in s.findings
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_refuses_without_session_continuity() -> None:
    """Cairn invariant: no session id ⇒ nothing to resume ⇒ harvest nothing."""
    s = make_solver(FakeClient(), session_id=None)
    assert await s.conclude(budget_s=5) == ""
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_refuses_a_zero_budget() -> None:
    s = make_solver(FakeClient())
    assert await s.conclude(budget_s=0) == ""
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_returns_empty_when_the_model_says_nothing() -> None:
    s = make_solver(FakeClient(messages=[FakeAssistantMessage(FakeTextBlock("   "))]))
    assert await s.conclude(budget_s=5) == ""
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_returns_empty_on_an_api_error() -> None:
    """A harvest that is itself an error message must never be published as fact."""
    client = FakeClient(messages=[
        FakeAssistantMessage(FakeTextBlock("API Error: Connection refused (ECONNREFUSED)"))
    ])
    s = make_solver(client)
    assert await s.conclude(budget_s=5) == ""
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_denies_tools() -> None:
    """Phase 2 is text-only by construction, enforced in the hook, not just asked for."""
    s = make_solver(FakeClient())
    s._concluding = True
    out = await s._sandbox_redirect_inner(
        {"hook_event_name": "PreToolUse", "tool_name": "Bash",
         "tool_input": {"command": "rm -rf /"}},
        "call-1",
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_failure_leaves_the_solver_usable() -> None:
    s = make_solver(FakeClient(messages=[], raise_exc=RuntimeError("nope")))
    assert await s.conclude(budget_s=5) == ""
    assert s._concluding is False, "the conclude flag was left stuck on"
    s.tracer.close()


# ---------------------------------------------------------------------------
# Sandbox command wrapping
# ---------------------------------------------------------------------------

def test_commands_are_timeout_wrapped_so_they_cannot_outlive_the_round() -> None:
    s = ClaudeSolver.__new__(ClaudeSolver)
    s.settings = SimpleNamespace(sandbox_command_timeout_s=900)
    s._container_id = "abc123"

    wrapped = s._wrap_for_sandbox("sleep 100000")

    assert wrapped.startswith("docker exec -i abc123 bash -c ")
    assert "timeout -k 5 900" in wrapped
    assert "sleep 100000" in wrapped


def test_wrapping_survives_shell_metacharacters() -> None:
    """The payload passes through two quoting layers (docker bash -c, then
    timeout bash -c). Verify it round-trips instead of eyeballing the escapes."""
    import shlex

    s = ClaudeSolver.__new__(ClaudeSolver)
    s.settings = SimpleNamespace(sandbox_command_timeout_s=60)
    s._container_id = "c1"

    nasty = """python3 -c 'print("hi"); import os' && echo `date` $HOME "q" 's'"""
    wrapped = s._wrap_for_sandbox(nasty)

    # Outer layer: `docker exec -i c1 bash -c <quoted>`
    outer = shlex.split(wrapped)
    assert outer[:4] == ["docker", "exec", "-i", "c1"]
    assert outer[4] == "bash" and outer[5] == "-c"
    inner = outer[6]

    # Inner layer: `timeout -k 5 60 bash -c <quoted>`
    parts = shlex.split(inner)
    assert parts[:4] == ["timeout", "-k", "5", "60"]
    assert parts[4] == "bash" and parts[5] == "-c"
    assert parts[6] == nasty, "the command did not survive the two quoting layers"


def test_command_timeout_can_be_disabled() -> None:
    s = ClaudeSolver.__new__(ClaudeSolver)
    s.settings = SimpleNamespace(sandbox_command_timeout_s=0)
    s._container_id = "c1"
    assert "timeout" not in s._wrap_for_sandbox("ls")


# ---------------------------------------------------------------------------
# Restart payloads must be auditable from the trace
# ---------------------------------------------------------------------------

def _realistic_restart_payload(target_chars: int = 10_500) -> str:
    """A payload sized like the ones measured live (7.8-10.5 KB).

    Sections appear in the order `_build_restart_payload` emits them. The skill
    hint is the bulky one: at escalation level 3 it embeds a whole pack index,
    bounded by skills.MAX_INDEX_CHARS. Because the team digest comes AFTER it, an
    undersized trace window loses the digest specifically — which is the section
    you most need when a restart misbehaves.
    """
    from backend.skills import MAX_INDEX_CHARS

    journal = "\n".join(
        f"{i}. 【已排除的死路】direction {i} ruled out after probing it thoroughly "
        f"and finding no write primitive; 【已确认的事实】offset {i * 7} leaks a libc pointer"
        for i in range(1, 6)
    )
    # Pad the handbook index to the real cap rather than guessing a size.
    index_body = "handbook entry line describing one technique\n"
    repeats = (MAX_INDEX_CHARS // len(index_body)) + 1
    skill_hint = (
        "KNOWLEDGE BASE (escalation level 3):\n"
        "A CTF technique library is mounted read-only at `/skills/`.\n"
        f"=== Pack index: /skills/ctf-misc/INDEX.md ===\n{(index_body * repeats)[:MAX_INDEX_CHARS]}\n"
        "Full file list for ctf-misc: " + ", ".join(f"/skills/ctf-misc/f{i}.md" for i in range(40))
    )
    # Size the digest from the ACTUAL entry length rather than an estimate, so
    # the fixture reliably lands at the measured payload size.
    sample_entry = "finding number 0 confirmed; "
    n_findings = max(1, (target_chars - len(journal) - len(skill_hint)) // len(sample_entry))
    team_digest = "【队友已确认】" + "; ".join(
        f"finding number {i} confirmed" for i in range(n_findings)
    )
    payload = "\n".join([
        "这是一次全新的尝试（第 4 轮）。",
        "【上一轮为何结束】",
        "Your previous attempt hit the 90s wall-clock time budget and was interrupted.",
        "【探索状态】",
        journal,
        "🎯 本轮主攻方向（未试过）：【CTF Misc - DNS Exploitation Techniques】"
        "（手册 /skills/ctf-misc/dns.md，先读再做）",
        skill_hint,
        "【全队进展（含你自己之前的发现，已去重蒸馏）】",
        team_digest,
        "开始吧。工作区里你之前创建的文件还在（/challenge/workspace/）。",
    ])
    return payload


def test_fixture_matches_the_measured_payload_size() -> None:
    """Guard the guard: if the fixture shrinks below what was observed live, the
    window test stops testing anything."""
    payload = _realistic_restart_payload()
    assert len(payload) >= 10_000, f"fixture is only {len(payload)} chars; live payloads reached 10500"


def test_trace_window_holds_a_realistic_restart_payload() -> None:
    """The team digest sits AFTER the bulky L3 skill hint, so a window that is
    even slightly too small loses precisely the section you need when a restart
    misbehaves. Measured payloads reached 10.5 KB against a 4 KB window."""
    payload = _realistic_restart_payload()
    assert len(payload) <= cs.BUMP_TRACE_CHARS, (
        f"BUMP_TRACE_CHARS={cs.BUMP_TRACE_CHARS} cannot hold a {len(payload)}-char "
        "payload; the team digest would be truncated out of the trace"
    )


def test_bump_event_records_the_true_payload_length() -> None:
    """So truncation is visible rather than silent, whatever the window is."""
    import json
    import tempfile

    payload = _realistic_restart_payload()
    with tempfile.TemporaryDirectory() as td:
        s = ClaudeSolver.__new__(ClaudeSolver)
        from collections import deque

        s._bump_queue = deque(maxlen=8)
        s.loop_detector = LoopDetector()
        s.tracer = SolverTracer("window-check", "fake", log_dir=td)
        s.agent_name = "window-check/fake"
        s.bump(payload)
        s.tracer.close()

        events = [
            json.loads(line)
            for line in (Path(td) / Path(s.tracer.path).name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    bump_events = [e for e in events if e.get("type") == "bump"]
    assert len(bump_events) == 1
    ev = bump_events[0]
    assert ev["insights_chars"] == len(payload), "the true length was not recorded"
    assert len(ev["insights"]) == min(len(payload), cs.BUMP_TRACE_CHARS)
    # The strongest proof nothing was cut: the LAST emitted section is present.
    # Asserting a specific digest entry is brittle — its index depends on how the
    # fixture was sized — while the trailing line is always the final one.
    assert "全队进展" in ev["insights"], "the team digest header was cut off"
    assert payload.rstrip().splitlines()[-1] in ev["insights"], (
        "the payload tail was truncated, so the team digest is not auditable"
    )


def test_pydantic_backend_uses_the_same_window() -> None:
    """Both backends write the same event shape; a window fixed in only one would
    leave the other blind."""
    from backend.agents import solver as pyd_solver

    assert pyd_solver.BUMP_TRACE_CHARS == cs.BUMP_TRACE_CHARS


# ---------------------------------------------------------------------------
# Mid-turn injection: advice and blackboard findings must not crowd each other out
# ---------------------------------------------------------------------------

class _FakeBus:
    """Records what was drained and hands back canned advice/findings."""

    def __init__(self, advice: list[str], findings_text: str) -> None:
        self._advice = advice
        self._findings_text = findings_text
        self.advice_calls = 0
        self.findings_calls = 0

    async def check_advice(self, model: str):
        self.advice_calls += 1
        out, self._advice = self._advice, []
        return out

    async def check(self, model: str):
        self.findings_calls += 1
        return [SimpleNamespace(model="other", content="x", kind="fact", summary="")]

    def format_unread(self, findings):
        return self._findings_text


def _hook_solver(bus) -> ClaudeSolver:
    s = make_solver(FakeClient())
    s.message_bus = bus
    s.model_spec = "claude-sdk/kimi-k3"
    return s


def _post_tool_event(step: int) -> dict:
    return {"hook_event_name": "PostToolUse", "tool_name": "Bash",
            "tool_response": "ok", "tool_use_id": f"call-{step}"}


@pytest.mark.asyncio
async def test_advice_and_findings_are_both_delivered_on_a_shared_step() -> None:
    """Advice is drained every 3 steps, findings every 5. Both branches used to
    `return` immediately, so on any step divisible by 15 the advice branch won and
    that round's blackboard check never ran.

    That is not a cosmetic ordering issue: the messages most likely to land there are
    exactly the urgent ones — a ⛔ prune notice after a refutation, or a ⚡
    contradiction — and deferring them to the next multiple of five is deferring the
    one thing that stops a solver from building on a disproved premise.
    """
    bus = _FakeBus(["stop what you are doing"], "**Findings:** something new")
    s = _hook_solver(bus)
    s._step_count = 15

    out = await s._trace_post_tool_inner(_post_tool_event(15))

    context = out["hookSpecificOutput"]["additionalContext"]
    assert "stop what you are doing" in context, "advice missing"
    assert "something new" in context, "findings were crowded out by the advice"
    assert bus.advice_calls == 1 and bus.findings_calls == 1
    s.tracer.close()


@pytest.mark.asyncio
async def test_advice_alone_still_delivers_on_its_own_step() -> None:
    bus = _FakeBus(["a nudge"], "**Findings:** something new")
    s = _hook_solver(bus)
    s._step_count = 9                      # divisible by 3, not by 5

    out = await s._trace_post_tool_inner(_post_tool_event(9))

    context = out["hookSpecificOutput"]["additionalContext"]
    assert "a nudge" in context
    assert "something new" not in context
    assert bus.findings_calls == 0, "the findings channel must not fire off-cycle"
    s.tracer.close()


@pytest.mark.asyncio
async def test_findings_alone_still_delivers_on_its_own_step() -> None:
    bus = _FakeBus([], "**Findings:** something new")
    s = _hook_solver(bus)
    s._step_count = 10                     # divisible by 5, not by 3

    out = await s._trace_post_tool_inner(_post_tool_event(10))

    assert "something new" in out["hookSpecificOutput"]["additionalContext"]
    assert bus.advice_calls == 0
    s.tracer.close()


@pytest.mark.asyncio
async def test_nothing_to_inject_returns_an_empty_hook_response() -> None:
    """An empty additionalContext would still be injected as a tool result and burn
    the model's attention for nothing."""
    bus = _FakeBus([], "No new findings from other agents.")
    s = _hook_solver(bus)
    s._step_count = 15

    assert await s._trace_post_tool_inner(_post_tool_event(15)) == {}
    s.tracer.close()


@pytest.mark.asyncio
async def test_the_conclude_phase_injects_nothing() -> None:
    """Phase 2 is tools-denied and text-only by construction; a mid-turn injection
    there would compete with the verdict the solver is being asked to produce."""
    bus = _FakeBus(["a nudge"], "**Findings:** something new")
    s = _hook_solver(bus)
    s._concluding = True
    s._step_count = 15

    assert await s._trace_post_tool_inner(_post_tool_event(15)) == {}
    assert bus.advice_calls == 0
    s.tracer.close()


# ---------------------------------------------------------------------------
# Teardown order — the container is the thing that leaks
# ---------------------------------------------------------------------------

class _RecordingSandbox:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


class _CancellingClient:
    """Stands in for an SDK client whose __aexit__ is interrupted by a cancel."""

    def __init__(self) -> None:
        self.closed = False

    async def __aexit__(self, *exc) -> None:
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_the_sandbox_is_destroyed_even_when_client_teardown_is_cancelled() -> None:
    """`stop()` awaited `_drop_client()` BEFORE `sandbox.stop()`, and `_drop_client`
    catches Exception but not CancelledError. A second cancel landing during the SDK
    client's `__aexit__` therefore propagated out and the container was never
    destroyed.

    That cancel is not hypothetical: `asyncio.run`'s shutdown cancels all tasks, a
    bench `--time-limit` fires `wait_for`, and `_wind_down` cancels stragglers after
    its grace period — each can land on a task already inside its finally-block stop.
    `DockerSandbox.stop()` shields its own teardown, so the exposure was purely the
    await sitting in front of it.
    """
    s = make_solver(FakeClient())
    sandbox = _RecordingSandbox()
    s.sandbox = sandbox
    s._owns_sandbox = True
    s._client = _CancellingClient()

    with pytest.raises(asyncio.CancelledError):
        await s.stop()

    assert sandbox.stopped, "the container must be destroyed before the client is dropped"
    s.tracer.close()


@pytest.mark.asyncio
async def test_stop_destroys_the_sandbox_on_the_happy_path_too() -> None:
    s = make_solver(FakeClient())
    sandbox = _RecordingSandbox()
    s.sandbox = sandbox
    s._owns_sandbox = True

    await s.stop()

    assert sandbox.stopped
    s.tracer.close()


@pytest.mark.asyncio
async def test_a_detached_sandbox_is_left_running_by_stop() -> None:
    """The quota-fallback handover depends on this: `detach_sandbox()` clears
    ownership so the old solver's stop() does not destroy the container the
    replacement is about to reuse."""
    s = make_solver(FakeClient())
    sandbox = _RecordingSandbox()
    s.sandbox = sandbox
    s._owns_sandbox = True

    assert s.detach_sandbox() is sandbox
    await s.stop()

    assert not sandbox.stopped, "a handed-over container must survive its old owner"
    s.tracer.close()


# ---------------------------------------------------------------------------
# Context overflow — the default backend used to misclassify it as a dead gateway
# ---------------------------------------------------------------------------

OVERFLOW_TEXTS = [
    "API Error: prompt is too long: 210000 tokens > 200000 maximum",
    "API Error: This model's maximum context length is 200000 tokens",
    "request failed: input length exceeds the context window",
]


@pytest.mark.parametrize("text", OVERFLOW_TEXTS)
def test_a_context_overflow_is_not_an_infrastructure_failure(text: str) -> None:
    """The defect, and why it was expensive.

    The CLI reports an overflow as ordinary assistant text, `_API_ERROR_RE` matches
    the "API Error" prefix, there is no HTTP status and no quota keyword — so it fell
    through to `return INFRA_ERROR`. On the DEFAULT backend that incremented
    `_infra_failures` (three retire the whole swarm, telling the operator to check
    ANTHROPIC_BASE_URL) and `consecutive_errors`, while `context_overflow` stayed
    False so the overflow coaching and the forced clean restart never ran.
    """
    from backend.solver_base import INFRA_ERROR, QUOTA_ERROR

    s = make_solver(FakeClient())
    s._finding_blocks.append(text)
    message = FakeResultMessage(is_error=True, subtype="error_during_execution")

    api_error, status_override = s._classify_result(message)

    assert status_override not in (INFRA_ERROR, QUOTA_ERROR), (
        f"an overflow was classified as {status_override}"
    )
    assert s._context_overflow is True
    assert api_error, "the reason must still be recorded for the trace"
    s.tracer.close()


@pytest.mark.parametrize("text", OVERFLOW_TEXTS)
def test_an_overflow_drops_the_session_so_the_next_round_can_succeed(text: str) -> None:
    """Overflow is recoverable ONLY by dropping the conversation. Leaving the session
    id in place means the next round resumes the same over-long history and fails
    identically forever — which is what made this look like a persistent outage."""
    s = make_solver(FakeClient(), session_id="sess-overlong")
    s._finding_blocks.append(text)

    s._classify_result(FakeResultMessage(is_error=True, subtype="error_during_execution"))

    assert s._session_id is None
    s.tracer.close()


def test_the_overflow_flag_reaches_the_result() -> None:
    """The swarm branches on `result.context_overflow` to force a clean restart and
    to keep the round out of the consecutive-error counter. A flag set on the solver
    but not carried onto the result is the same as not detecting it."""
    s = make_solver(FakeClient())
    s._context_overflow = True

    result = s._result("error", {"steps": 0, "calls": 0, "cost": 0.0})

    assert result.context_overflow is True
    s.tracer.close()


def test_a_genuine_gateway_failure_is_still_infra() -> None:
    """The fix must not blind the classifier to real outages — the ECONNREFUSED run
    that motivated this whole module still has to be caught."""
    from backend.solver_base import INFRA_ERROR

    s = make_solver(FakeClient())
    s._finding_blocks.append("API Error: Connection refused (ECONNREFUSED)")

    _api_error, status = s._classify_result(
        FakeResultMessage(is_error=True, subtype="error_during_execution"))

    assert status == INFRA_ERROR
    assert s._context_overflow is False
    s.tracer.close()


def test_hitting_the_turn_ceiling_is_not_an_infrastructure_failure() -> None:
    """`error_max_turns` used to return a non-empty api_error, which makes
    `is_infra_failure` true, so `_account_infra` rewrote the status to INFRA_ERROR
    and counted a PRODUCTIVE round toward retiring the swarm — three of them told
    the operator to check their gateway."""
    from backend.solver_base import INFRA_ERROR

    s = make_solver(FakeClient())
    s._tool_calls = 40

    api_error, status = s._classify_result(
        FakeResultMessage(is_error=True, subtype="error_max_turns"))

    assert status not in (INFRA_ERROR,)
    assert api_error == "", "a non-empty api_error would mark the round as infra"
    s.tracer.close()


def test_an_overflow_exception_is_also_recognised() -> None:
    """The SDK can raise instead of reporting in a ResultMessage; the Pydantic
    backend already handled that path and this one did not."""
    from backend.solver_base import ERROR, INFRA_ERROR

    class FakeContextOverflow(Exception):
        pass

    FakeContextOverflow.__name__ = "ModelContextWindowExceededError"

    s = make_solver(FakeClient(), session_id="sess-x")
    exc = FakeContextOverflow("conversation no longer fits")

    from backend.solver_base import is_context_overflow

    assert is_context_overflow(exc), "the class name must match across the MRO"
    assert s._result(ERROR, {"steps": 0, "calls": 0, "cost": 0.0}).status == ERROR
    # And a plain transport exception is still classified by CLASS, not just text:
    from backend.solver_base import classify_exception

    class FakeTimeout(Exception):
        pass

    FakeTimeout.__name__ = "APITimeoutError"
    assert classify_exception(FakeTimeout("x")) == INFRA_ERROR
    s.tracer.close()


# ── pseudo-command argument parsing ───────────────────────────────────────────
# Found by a real run, not by reading code: a solver on a 500-point crypto
# challenge recovered the correct key and submitted it as
# `submit_flag 'flag{...}'; echo "---"`. The old pattern anchored a non-greedy
# `(.+?)` to end-of-line, so the payload became `flag{...}'; echo "---` and the
# answer key rejected a correct solve. The run reported "No flag found".


def test_a_chained_command_does_not_pollute_the_flag() -> None:
    """The exact string observed in the rejected submission."""
    got = cs._pseudo_arg(
        "submit_flag 'flag{ca651cdb35a467acf263}'; echo \"---\"", "submit_flag")
    assert got == "flag{ca651cdb35a467acf263}"


def test_quoting_styles_and_separators_all_yield_the_bare_argument() -> None:
    cases = {
        "submit_flag flag{plain}": "flag{plain}",
        "submit_flag 'flag{sq}'": "flag{sq}",
        'submit_flag "flag{dq}"': "flag{dq}",
        "submit_flag 'flag{with space inside}'": "flag{with space inside}",
        "submit_flag 'flag{x}' && echo done": "flag{x}",
        "submit_flag flag{x}|tee out": "flag{x}",
        "submit_flag   'flag{x}'   ": "flag{x}",
    }
    for cmd, want in cases.items():
        assert cs._pseudo_arg(cmd, "submit_flag") == want, cmd


def test_unbalanced_quoting_falls_back_rather_than_dropping_the_submit() -> None:
    """A dirty flag can still match the key; a lost submission never can."""
    assert cs._pseudo_arg("submit_flag 'flag{unbalanced}", "submit_flag") == "flag{unbalanced}"


def test_a_missing_or_separator_only_argument_is_no_submission_at_all() -> None:
    for cmd in ("submit_flag", "submit_flag   ", "submit_flag ; echo hi", "submit_flag |"):
        assert cs._pseudo_arg(cmd, "submit_flag") is None, cmd


def test_a_command_that_merely_starts_with_the_name_is_not_intercepted() -> None:
    """`submit_flagged` must fall through to the sandbox, not be eaten."""
    assert cs._pseudo_arg("submit_flagged something", "submit_flag") is None
    assert cs._pseudo_arg("echo submit_flag 'x'", "submit_flag") is None


def test_all_three_pseudo_commands_route_through_the_same_parser() -> None:
    """Guard against the regex being reintroduced at one call site only."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "backend" / "agents" / "claude_solver.py").read_text(encoding="utf-8")
    for name in ("submit_flag", "notify_coordinator", "request_help"):
        assert f'_pseudo_arg(command, "{name}")' in src, f"{name} is not using the shared parser"
    assert r"(.+?)['\"]?\s*$" not in src, "the end-of-line capture pattern is back"


def test_the_notify_and_help_payloads_are_not_polluted_either() -> None:
    assert cs._pseudo_arg(
        "notify_coordinator 'q divides p-1?'; grep x", "notify_coordinator") == "q divides p-1?"
    assert cs._pseudo_arg(
        "request_help 'stuck on LLL' && ls", "request_help") == "stuck on LLL"
