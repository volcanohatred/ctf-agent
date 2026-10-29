"""The Pydantic backend's phase-2 harvest must describe the round that just ended.

`_messages` was assigned only on a successful run, so after a `RunTimedOut` it still
held the PREVIOUS round's conversation — or nothing at all on a first-round timeout.
`conclude()` then distilled that, and `_harvest_timeout` published the result to the
blackboard, where every sibling and every restart treats it as ground truth. So the
swarm could be handed a confident state card about work this round never did, or an
invented one built from the system prompt alone.

pydantic-ai attaches the partial conversation to the exception, so the right fix is to
recover it. When it cannot be recovered, the harvest must refuse — "nothing" beats
"plausible", which is the same rule ClaudeSolver's conclude already follows.
"""

from __future__ import annotations

import asyncio
from collections import deque
from types import SimpleNamespace

import pytest

from backend.agents.solver import Solver, _partial_messages
from backend.loop_detect import LoopDetector
from backend.solver_base import GAVE_UP
from backend.tracing import SolverTracer


class RunTimedOut(Exception):
    """Duck-typed stand-in. `_handle_run_exception` matches on the class NAME as well
    as isinstance, precisely so this does not have to import pydantic-ai internals."""


def _msg(text: str = "m"):
    return SimpleNamespace(text=text)


def _make_solver(tmp_path, messages: list | None = None) -> Solver:
    """A Solver wired to fakes, bypassing __init__ (no Docker, no model)."""
    s = Solver.__new__(Solver)
    s.agent_name = "unit/pyd"
    s.model_id = "pyd"
    s.model_spec = "gateway/pyd"
    s.tracer = SolverTracer("unit-pyd", "pyd", log_dir=str(tmp_path))
    s.loop_detector = LoopDetector()
    s.cost_tracker = SimpleNamespace(by_agent={}, total_cost_usd=0.0,
                                     record=lambda *a, **k: None,
                                     record_tokens=lambda *a, **k: None)
    s._step_count = [0]
    s._tool_calls = [0]
    s._flag = None
    s._confirmed = False
    s._finding_blocks = deque(maxlen=24)
    s._cost_usd = 0.0
    s._last_usage = (0, 0, 0)
    s._last_input_tokens = 0
    s._bump_queue = deque(maxlen=8)
    s._interrupted = False
    s._messages = list(messages or [])
    s._messages_current = True
    s._agent = SimpleNamespace()      # conclude() only checks truthiness
    s._model = SimpleNamespace()
    s._conclude_agent = None
    s.deps = SimpleNamespace(no_submit=True, submit_fn=None, message_bus=None,
                             confirmed_flag=None)
    s.cancel_event = SimpleNamespace(is_set=lambda: False)
    return s


# ---------------------------------------------------------------------------
# Recovering the partial conversation
# ---------------------------------------------------------------------------

def test_partial_messages_are_recovered_from_all_messages() -> None:
    exc = RunTimedOut("timed out")
    exc.all_messages = lambda: [_msg("a"), _msg("b")]

    assert len(_partial_messages(exc)) == 2


def test_partial_messages_fall_back_to_new_messages() -> None:
    exc = RunTimedOut("timed out")
    exc.all_messages = lambda: []
    exc.new_messages = lambda: [_msg("only-new")]

    assert len(_partial_messages(exc)) == 1


def test_nothing_recoverable_returns_empty_not_a_raise() -> None:
    assert _partial_messages(RunTimedOut("timed out")) == []
    assert _partial_messages(RuntimeError("plain")) == []


def test_a_broken_accessor_does_not_mask_the_timeout() -> None:
    """The timeout is the fact worth reporting; a pydantic-ai version that changes
    the accessor must not turn it into a different error."""
    exc = RunTimedOut("timed out")

    def _boom():
        raise ValueError("accessor changed shape")

    exc.all_messages = _boom
    exc.new_messages = _boom

    assert _partial_messages(exc) == []


# ---------------------------------------------------------------------------
# The timeout path
# ---------------------------------------------------------------------------

def test_a_timeout_captures_the_interrupted_rounds_history(tmp_path) -> None:
    """Without this the conclude phase distils the PREVIOUS round and the swarm
    publishes that as ground truth."""
    previous = [_msg("round-one work")]
    s = _make_solver(tmp_path, messages=previous)
    interrupted = [_msg("round-two partial"), _msg("more round-two")]

    exc = RunTimedOut("timed out")
    exc.all_messages = lambda: interrupted

    result = s._handle_run_exception(exc, {"steps": 0, "calls": 0, "cost": 0.0})

    assert result.interrupted is True
    assert result.status == GAVE_UP
    assert s._messages == interrupted, "must hold THIS round, not the previous one"
    assert s._messages_current is True
    s.tracer.close()


def test_an_unrecoverable_timeout_marks_the_history_stale(tmp_path) -> None:
    previous = [_msg("round-one work")]
    s = _make_solver(tmp_path, messages=previous)

    s._handle_run_exception(RunTimedOut("timed out"), {"steps": 0, "calls": 0, "cost": 0.0})

    assert s._messages_current is False, "the history still describes round one"
    s.tracer.close()


def test_a_first_round_timeout_leaves_no_history_to_distil(tmp_path) -> None:
    """The worst case: `_messages` is empty, so conclude() would run on the system
    prompt plus the conclude prompt alone and the model would invent a state card."""
    s = _make_solver(tmp_path, messages=[])

    s._handle_run_exception(RunTimedOut("timed out"), {"steps": 0, "calls": 0, "cost": 0.0})

    assert s._messages == []
    assert s._messages_current is False
    s.tracer.close()


# ---------------------------------------------------------------------------
# conclude() must refuse rather than distil the wrong round
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_conclude_refuses_when_the_history_is_from_a_prior_round(tmp_path) -> None:
    s = _make_solver(tmp_path, messages=[_msg("stale round")])
    s._messages_current = False

    assert await s.conclude(budget_s=30) == ""
    s.tracer.close()


@pytest.mark.asyncio
async def test_conclude_still_refuses_on_a_zero_budget(tmp_path) -> None:
    s = _make_solver(tmp_path, messages=[_msg("current round")])

    assert await s.conclude(budget_s=0) == ""
    s.tracer.close()


def test_reset_history_makes_the_empty_conversation_current_again(tmp_path) -> None:
    """A hard reset means the next round genuinely starts from nothing, so a later
    conclude must not be refused on a stale flag."""
    s = _make_solver(tmp_path, messages=[_msg("old")])
    s._messages_current = False

    s.reset_history()

    assert s._messages == []
    assert s._messages_current is True
    s.tracer.close()


def test_the_timeout_records_how_much_was_recovered(tmp_path) -> None:
    """Diagnostics: "0 partial messages" and "40 partial messages" mean very
    different things when triaging why a harvest was empty."""
    s = _make_solver(tmp_path, messages=[])
    exc = RunTimedOut("timed out")
    exc.all_messages = lambda: [_msg("a"), _msg("b"), _msg("c")]

    s._handle_run_exception(exc, {"steps": 0, "calls": 0, "cost": 0.0})

    events = [e for e in _read_events(s.tracer.path) if e.get("type") == "budget_interrupt"]
    assert events and events[-1]["partial_messages"] == 3
    s.tracer.close()


def _read_events(path: str) -> list[dict]:
    import json
    from pathlib import Path

    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


# ---------------------------------------------------------------------------
# Mid-turn cancellation
#
# `_wind_down`'s contract is that solvers watch cancel_event and break out of their
# stream loop within a couple of seconds, so each can run its own container and
# workspace teardown uninterrupted. ClaudeSolver polls every 2s and honours it; this
# backend awaited `agent.run` to completion and only read the event afterwards, so it
# ignored cancellation for the rest of a turn that can run to SOLVER_TIME_BUDGET_S
# (1500s), ate the whole 60s grace, and was then hard-task-cancelled — the exact path
# `_wind_down` exists to avoid, and it lost the round's findings.
# ---------------------------------------------------------------------------

def _real_cancel_event() -> asyncio.Event:
    return asyncio.Event()


async def test_a_run_that_completes_returns_its_result(tmp_path) -> None:
    s = _make_solver(tmp_path)
    s.cancel_event = _real_cancel_event()

    async def _run():
        return "the-result"

    task = asyncio.create_task(_run())
    assert await s._await_run_or_cancel(task) == "the-result"
    s.tracer.close()


async def test_an_already_cancelled_swarm_does_not_start_the_turn(tmp_path) -> None:
    s = _make_solver(tmp_path)
    s.cancel_event = _real_cancel_event()
    s.cancel_event.set()

    started = []

    async def _run():
        started.append(True)
        return "should-not-be-returned"

    task = asyncio.create_task(_run())
    assert await s._await_run_or_cancel(task) is None
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    s.tracer.close()


async def test_a_cancel_mid_turn_abandons_the_run_promptly(tmp_path) -> None:
    """The whole point: a long turn must not ride out its budget after the swarm is
    done with it."""
    s = _make_solver(tmp_path)
    s.cancel_event = _real_cancel_event()

    abandoned = asyncio.Event()

    async def _long_run():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            abandoned.set()
            raise
        return "never"

    task = asyncio.create_task(_long_run())
    waiter = asyncio.create_task(s._await_run_or_cancel(task))
    await asyncio.sleep(0.02)
    s.cancel_event.set()

    assert await asyncio.wait_for(waiter, timeout=2) is None
    assert abandoned.is_set(), "the abandoned run must actually be cancelled"
    assert s._interrupted is True
    s.tracer.close()


async def test_a_failing_run_still_raises_so_the_handler_maps_it(tmp_path) -> None:
    """The race must not swallow the run's exception — `_handle_run_exception` is
    what turns a RunTimedOut into the interrupted/harvest path."""
    s = _make_solver(tmp_path)
    s.cancel_event = _real_cancel_event()

    class Boom(Exception):
        pass

    async def _run():
        raise Boom("the run failed")

    task = asyncio.create_task(_run())
    with pytest.raises(Boom):
        await s._await_run_or_cancel(task)
    s.tracer.close()


async def test_cancelling_the_awaiter_propagates_and_cleans_up_the_run(tmp_path) -> None:
    """This is `_wind_down`'s last-resort hard cancel of a straggler. Swallowing it
    here would defeat the only mechanism that bounds a wedged solver."""
    s = _make_solver(tmp_path)
    s.cancel_event = _real_cancel_event()

    abandoned = asyncio.Event()

    async def _long_run():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            abandoned.set()
            raise
        return "never"

    task = asyncio.create_task(_long_run())
    awaiter = asyncio.create_task(s._await_run_or_cancel(task))
    await asyncio.sleep(0.02)
    awaiter.cancel()

    with pytest.raises(asyncio.CancelledError):
        await awaiter
    assert abandoned.is_set(), "the inner run must not be left pending"
    s.tracer.close()


async def test_no_task_is_left_pending_after_any_outcome(tmp_path) -> None:
    """An unawaited task is destroyed at loop close with a warning, and if it owned a
    container that container leaks."""
    s = _make_solver(tmp_path)
    s.cancel_event = _real_cancel_event()

    for cancel_first in (False, True):
        s.cancel_event.clear()

        async def _run():
            await asyncio.sleep(10)
            return "x"

        task = asyncio.create_task(_run())
        waiter = asyncio.create_task(s._await_run_or_cancel(task))
        await asyncio.sleep(0.01)
        if cancel_first:
            s.cancel_event.set()
            await asyncio.wait_for(waiter, timeout=2)
        else:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        assert task.done(), "the inner run task must always be awaited"
    s.tracer.close()
