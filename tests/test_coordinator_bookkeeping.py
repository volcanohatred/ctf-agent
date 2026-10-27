"""Coordinator bookkeeping that the operator's dashboard depends on.

The regression pinned here was observed live: a full 14-challenge run ended with
CTFd and the coordinator's own STATUS line both reporting 14 solved, while the
dashboard said 13. One challenge (reg-misc-encoding) was missing from
`deps.results` because the poller observed the solve and killed the swarm before
`swarm.run()` could return FLAG_FOUND — and `_run_and_cleanup` only records a
result on that path.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.agents.coordinator_loop import (
    _auto_spawn_unsolved,
    record_challenge_solved,
)


def _deps(**over):
    base = dict(
        swarms={}, swarm_tasks={}, results={}, challenge_dirs={}, challenge_metas={},
        spawn_attempts={}, swarms_history={}, coordinator_inbox=asyncio.Queue(),
        max_concurrent_challenges=10, settings=SimpleNamespace(),
    )
    base.update(over)
    return SimpleNamespace(**base)


def _swarm(name: str, flag: str | None = None, confirmed: str | None = None):
    killed_with: list[str] = []
    swarm = SimpleNamespace(
        meta=SimpleNamespace(name=name),
        cancel_event=asyncio.Event(),
        winner=SimpleNamespace(flag=flag) if flag else None,
        winner_spec="claude-sdk/deepseek-v4-flash" if flag else "",
        confirmed_flag=confirmed,
        killed_with=killed_with,
    )
    swarm.kill = lambda reason="": (killed_with.append(reason),
                                    swarm.cancel_event.set())
    return swarm


# ---------------------------------------------------------------------------
# record_challenge_solved
# ---------------------------------------------------------------------------

def test_solve_is_recorded_even_when_the_swarm_never_set_a_winner() -> None:
    """The exact live failure: CTFd says solved, the swarm has no winner yet."""
    swarm = _swarm("chal")
    deps = _deps(swarms={"chal": swarm})

    record_challenge_solved(deps, "chal")

    assert "chal" in deps.results, "the solve was lost"
    assert deps.results["chal"]["submit"] == "confirmed on CTFd (observed by poller)"
    assert swarm.killed_with == ["challenge confirmed solved on CTFd"]


def test_recorded_solve_keeps_the_flag_when_the_swarm_had_one() -> None:
    swarm = _swarm("chal", flag="flag{real}")
    deps = _deps(swarms={"chal": swarm})

    record_challenge_solved(deps, "chal")

    assert deps.results["chal"]["flag"] == "flag{real}"
    assert deps.results["chal"]["submit"] == "confirmed by solver"


def test_confirmed_flag_is_used_when_winner_is_unset() -> None:
    """submit_fn confirms on the swarm before the round loop stores a winner."""
    swarm = _swarm("chal", flag=None, confirmed="flag{via_submit}")
    deps = _deps(swarms={"chal": swarm})

    record_challenge_solved(deps, "chal")

    assert deps.results["chal"]["flag"] == "flag{via_submit}"
    assert deps.results["chal"]["submit"] == "confirmed by solver"


def test_solve_without_any_swarm_is_still_recorded() -> None:
    """Another team/instance solved it, or the swarm was already archived."""
    deps = _deps(swarms={})

    record_challenge_solved(deps, "chal")

    assert "chal" in deps.results
    assert deps.results["chal"]["flag"] == ""


def test_an_existing_result_is_not_overwritten() -> None:
    """The solver's own record is more detailed than the poller's; keep it."""
    swarm = _swarm("chal", flag="flag{real}")
    deps = _deps(swarms={"chal": swarm},
                 results={"chal": {"flag": "flag{real}", "submit": "confirmed by solver"}})

    record_challenge_solved(deps, "chal")

    assert deps.results["chal"]["flag"] == "flag{real}"
    assert deps.results["chal"]["submit"] == "confirmed by solver"


def test_already_cancelled_swarm_is_not_killed_again() -> None:
    swarm = _swarm("chal")
    swarm.cancel_event.set()
    deps = _deps(swarms={"chal": swarm})

    record_challenge_solved(deps, "chal")

    assert "chal" in deps.results
    assert swarm.killed_with == [], "a finished swarm was re-killed"


def test_kill_reason_is_recorded_on_the_swarm() -> None:
    """A killed swarm must say why, or it is indistinguishable from a crash."""
    swarm = _swarm("chal")
    deps = _deps(swarms={"chal": swarm})

    record_challenge_solved(deps, "chal")

    assert swarm.cancel_event.is_set()
    assert swarm.killed_with[0] == "challenge confirmed solved on CTFd"


# ---------------------------------------------------------------------------
# Periodic re-spawn (self-healing)
# ---------------------------------------------------------------------------

class _FakePoller:
    def __init__(self, known: set[str], solved: set[str]):
        self._known, self._solved = known, solved

    @property
    def known_challenges(self) -> set[str]:
        return set(self._known)

    @property
    def known_solved(self) -> set[str]:
        return set(self._solved)


@pytest.mark.asyncio
async def test_respawn_picks_up_unsolved_with_no_swarm(monkeypatch) -> None:
    spawned: list[str] = []

    async def fake_spawn(deps, name):
        spawned.append(name)
        deps.swarm_tasks[name] = _done_task()
        return "spawned"

    monkeypatch.setattr("backend.agents.coordinator_core.do_spawn_swarm", fake_spawn)
    deps = _deps()
    poller = _FakePoller({"a", "b"}, set())

    n = await _auto_spawn_unsolved(deps, poller)

    assert n == 2
    assert sorted(spawned) == ["a", "b"]
    assert deps.spawn_attempts == {"a": 1, "b": 1}


@pytest.mark.asyncio
async def test_respawn_skips_challenges_that_already_have_a_swarm(monkeypatch) -> None:
    spawned: list[str] = []

    async def fake_spawn(deps, name):
        spawned.append(name)
        return "spawned"

    monkeypatch.setattr("backend.agents.coordinator_core.do_spawn_swarm", fake_spawn)
    deps = _deps(swarms={"a": _swarm("a")}, swarm_tasks={"a": _running_task()})
    poller = _FakePoller({"a", "b"}, set())

    await _auto_spawn_unsolved(deps, poller)

    assert "a" not in spawned, "a challenge with a live swarm was re-spawned"
    assert spawned == ["b"], spawned


@pytest.mark.asyncio
async def test_respawn_skips_solved_challenges(monkeypatch) -> None:
    spawned: list[str] = []

    async def fake_spawn(deps, name):
        spawned.append(name)
        deps.swarm_tasks[name] = _done_task()
        return "spawned"

    monkeypatch.setattr("backend.agents.coordinator_core.do_spawn_swarm", fake_spawn)
    deps = _deps()
    poller = _FakePoller({"a", "b"}, {"a"})

    await _auto_spawn_unsolved(deps, poller)

    assert spawned == ["b"]


@pytest.mark.asyncio
async def test_respawn_is_bounded_per_challenge(monkeypatch) -> None:
    """A hopeless challenge must not be re-queued forever now that the round cap
    actually terminates swarms."""
    calls: list[str] = []

    async def fake_spawn(deps, name):
        calls.append(name)
        deps.swarm_tasks[name] = _done_task()
        return "spawned"

    monkeypatch.setattr("backend.agents.coordinator_core.do_spawn_swarm", fake_spawn)
    deps = _deps()
    poller = _FakePoller({"a"}, set())

    for _ in range(6):
        # each cycle the previous swarm has finished and been reaped
        deps.swarm_tasks.clear()
        deps.swarms.clear()
        await _auto_spawn_unsolved(deps, poller, max_spawns=3)

    assert len(calls) == 3, f"re-spawned {len(calls)}x, expected the cap of 3"
    assert deps.spawn_attempts == {"a": 3}


@pytest.mark.asyncio
async def test_respawn_respects_the_concurrency_cap(monkeypatch) -> None:
    spawned: list[str] = []

    async def fake_spawn(deps, name):
        spawned.append(name)
        deps.swarm_tasks[name] = _running_task()
        return "spawned"

    monkeypatch.setattr("backend.agents.coordinator_core.do_spawn_swarm", fake_spawn)
    deps = _deps(max_concurrent_challenges=2)
    poller = _FakePoller({"a", "b", "c", "d"}, set())

    await _auto_spawn_unsolved(deps, poller)

    assert len(spawned) == 2, spawned


def _done_task():
    async def _noop():
        return None

    t = asyncio.ensure_future(_noop())
    return t


def _running_task():
    async def _forever():
        await asyncio.sleep(3600)

    return asyncio.ensure_future(_forever())
