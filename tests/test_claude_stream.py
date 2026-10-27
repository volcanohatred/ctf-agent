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
