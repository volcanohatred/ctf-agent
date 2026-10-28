"""Live HTTP tests for the dashboard server.

The auth gate is the security-critical change here: an earlier revision listened
on 0.0.0.0 with no authentication and exposed endpoints that inject instructions
into solvers (which run arbitrary code in privileged sandboxes), spawn
containers, and export confirmed flags. Solver sandboxes can reach the host via
host.docker.internal, so that was a remote-control and flag-disclosure endpoint.

These tests start a real aiohttp server on an ephemeral loopback port and drive
it over HTTP, so the gate is exercised exactly as a client would.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import ClientSession

from backend.message_bus import ChallengeMessageBus
from backend.web_dashboard import DashboardServer

TOKEN = "test-token-abc123"


def _fake_swarm(name: str = "chal", flag: str | None = None):
    bus = ChallengeMessageBus()
    swarm = SimpleNamespace(
        meta=SimpleNamespace(category="crypto", value=100, description="d", tags=[]),
        model_specs=["claude-sdk/kimi-k3"],
        _all_specs=["claude-sdk/kimi-k3"],
        solvers={},
        cancel_event=asyncio.Event(),
        winner=SimpleNamespace(flag=flag) if flag else None,
        winner_spec="claude-sdk/kimi-k3" if flag else "",
        _total_bumps=0,
        _failed_approaches=[],
        _assigned_approach={},
        _infra_failures=0,
        retire_reason="",
        findings={},
        message_bus=bus,
    )
    return swarm, bus


def _deps(swarms: dict | None = None):
    return SimpleNamespace(
        swarms=swarms if swarms is not None else {},
        swarms_history={},
        challenge_metas={},
        results={},
        cost_tracker=SimpleNamespace(total_cost_usd=0.0, by_agent={}),
        settings=SimpleNamespace(dashboard_feed_chars=4000),
    )


def _bound_port(srv: DashboardServer) -> int:
    """The kernel-assigned port when constructed with port=0."""
    for site in getattr(srv._runner, "sites", ()) if srv._runner else ():
        for sock in getattr(site, "_server", None).sockets if getattr(site, "_server", None) else ():
            return sock.getsockname()[1]
    return srv.port


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_every_route_rejects_an_unauthenticated_client() -> None:
    swarm, _bus = _fake_swarm()
    deps = _deps({"chal": swarm})
    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        port = _bound_port(srv)
        base = f"http://127.0.0.1:{port}"
        async with ClientSession() as s:
            for method, path, body in [
                ("GET", "/api/state", None),
                ("GET", "/api/trace?name=chal&spec=x", None),
                ("GET", "/api/export?name=chal&fmt=json", None),
                ("POST", "/api/advise", {"challenge": "chal", "model_spec": "m", "advice": "x"}),
                ("POST", "/api/bump", {"challenge": "chal", "model_spec": "m"}),
                ("POST", "/api/broadcast", {"challenge": "chal", "message": "x"}),
                ("POST", "/api/ask", {"challenge": "chal", "question": "progress?"}),
                ("POST", "/api/spawn", {"challenge": "chal"}),
                ("GET", "/", None),
            ]:
                if method == "GET":
                    resp = await s.get(base + path)
                else:
                    resp = await s.post(base + path, json=body)
                assert resp.status == 401, f"{method} {path} -> {resp.status}, expected 401"
                await resp.read()
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_a_wrong_token_is_rejected() -> None:
    srv = DashboardServer(_deps({}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.get(base + "/api/state", headers={"Authorization": "Bearer nope"})
            assert r.status == 401
            r2 = await s.get(base + "/api/state?token=nope")
            assert r2.status == 401
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_bearer_and_query_token_both_work() -> None:
    srv = DashboardServer(_deps({}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r1 = await s.get(base + "/api/state",
                             headers={"Authorization": f"Bearer {TOKEN}"})
            assert r1.status == 200
            await r1.read()
            # The query form is what the browser needs for <a href> export links.
            r2 = await s.get(base + f"/api/state?token={TOKEN}")
            assert r2.status == 200
            await r2.read()
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_healthz_needs_no_token_but_leaks_nothing() -> None:
    srv = DashboardServer(_deps({}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.get(base + "/healthz")
            assert r.status == 200
            assert await r.json() == {"ok": True}
    finally:
        await srv.stop()


def test_token_is_generated_when_unset() -> None:
    """It must never silently run unauthenticated."""
    srv = DashboardServer(_deps({}), token="")
    assert srv.token and len(srv.token) >= 20
    other = DashboardServer(_deps({}), token="")
    assert srv.token != other.token, "generated tokens must be unique per process"


# ---------------------------------------------------------------------------
# Payload shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_returns_normalized_agent_counters() -> None:
    """The Pydantic backend keeps _step_count as a list; the JSON must be an int."""
    swarm, _bus = _fake_swarm()
    swarm.solvers["claude-sdk/kimi-k3"] = SimpleNamespace(
        _step_count=[42], _tool_calls=[7], tracer=None,
    )
    srv = DashboardServer(_deps({"chal": swarm}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.get(base + f"/api/state?token={TOKEN}")
            data = await r.json()
    finally:
        await srv.stop()

    agent = data["swarms"]["chal"]["agents"]["claude-sdk/kimi-k3"]
    assert agent["steps"] == 42, agent
    assert agent["tool_calls"] == 7, agent
    assert isinstance(agent["steps"], int) and isinstance(agent["tool_calls"], int)


@pytest.mark.asyncio
async def test_state_includes_help_requests_and_feed() -> None:
    swarm, bus = _fake_swarm()
    await bus.post("claude-sdk/kimi-k3", "libc base 0x7f00", kind="fact")
    await bus.post_help("claude-sdk/kimi-k3", "unknown custom protocol")
    srv = DashboardServer(_deps({"chal": swarm}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    ch = data["swarms"]["chal"]
    assert any("libc base" in f["content"] for f in ch["shared_feed"])
    assert any("unknown custom protocol" in h["question"] for h in ch["help_requests"])


@pytest.mark.asyncio
async def test_state_clips_oversized_findings() -> None:
    """Findings are deliberately never truncated at rest, so streaming dozens of
    them in full every 2.5s was megabytes per poll."""
    swarm, bus = _fake_swarm()
    await bus.post("claude-sdk/kimi-k3", "x" * 50_000, kind="fact")
    deps = _deps({"chal": swarm})
    deps.settings = SimpleNamespace(dashboard_feed_chars=500)
    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    entry = data["swarms"]["chal"]["shared_feed"][0]
    assert len(entry["content"]) < 1000, len(entry["content"])
    assert entry["chars"] == 50_000, "the true size must still be reported"
    assert "truncated" in entry["content"]


@pytest.mark.asyncio
async def test_export_returns_the_full_untruncated_report() -> None:
    """The live feed clips; the export must not — it is the writeup source."""
    swarm, bus = _fake_swarm(flag="flag{secret}")
    await bus.post("claude-sdk/kimi-k3", "y" * 50_000, kind="fact")
    deps = _deps({"chal": swarm})
    deps.settings = SimpleNamespace(dashboard_feed_chars=500)
    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.get(base + f"/api/export?name=chal&fmt=json&token={TOKEN}")
            body = await r.text()
    finally:
        await srv.stop()

    assert r.status == 200
    report = json.loads(body)
    assert report["winner"] == "flag{secret}"
    assert "y" * 50_000 in body, "export truncated the blackboard content"
    assert "truncated" not in report["shared_feed"][0]["content"]


@pytest.mark.asyncio
async def test_export_filename_is_ascii_safe() -> None:
    """Challenge names come from CTFd and can contain anything."""
    swarm, _bus = _fake_swarm()
    deps = _deps({'chäl "x"/y': swarm})
    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.get(base + "/api/export?fmt=json&token=" + TOKEN
                            + "&name=" + "ch%C3%A4l%20%22x%22%2Fy")
            cd = r.headers.get("Content-Disposition", "")
            await r.read()
    finally:
        await srv.stop()

    assert r.status == 200
    header_value = cd.split("filename=")[1].strip('"')
    header_value.encode("ascii")  # must not raise
    assert all(c not in header_value for c in '"/\\')


@pytest.mark.asyncio
async def test_advise_rejects_empty_and_malformed_bodies() -> None:
    swarm, _bus = _fake_swarm()
    srv = DashboardServer(_deps({"chal": swarm}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            h = {"Authorization": f"Bearer {TOKEN}"}
            r1 = await s.post(base + "/api/advise", json={"challenge": "chal"}, headers=h)
            assert r1.status == 400
            await r1.read()
            r2 = await s.post(base + "/api/advise", data=b"not json", headers=h)
            assert r2.status == 400
            await r2.read()
            r3 = await s.post(base + "/api/advise",
                              json={"challenge": "nope", "advice": "x"}, headers=h)
            assert r3.status == 404
            await r3.read()
    finally:
        await srv.stop()


@pytest.mark.asyncio
async def test_advise_delivers_into_the_solver_mailbox() -> None:
    swarm, bus = _fake_swarm()
    srv = DashboardServer(_deps({"chal": swarm}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.post(base + "/api/advise", headers={"Authorization": f"Bearer {TOKEN}"},
                             json={"challenge": "chal", "model_spec": "claude-sdk/kimi-k3",
                                   "advice": "try the tcache poison"})
            assert r.status == 200
            await r.read()
    finally:
        await srv.stop()

    got = await bus.check_advice("claude-sdk/kimi-k3")
    assert got == ["[operator] try the tcache poison"]


@pytest.mark.asyncio
async def test_index_page_carries_the_token() -> None:
    srv = DashboardServer(_deps({}), host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            r = await s.get(base + f"/?token={TOKEN}")
            html = await r.text()
    finally:
        await srv.stop()

    assert r.status == 200
    assert f'const TOKEN = "{TOKEN}"' in html


# ---------------------------------------------------------------------------
# Solved state must come from CTFd, not from this session's results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_solved_reflects_ctfd_after_a_restart() -> None:
    """Regression observed live: a full 14-challenge run finished, the
    coordinator was restarted, and the dashboard then reported 0 solved with all
    22 challenges as "待解题" — while the coordinator's own STATUS line, which
    reads the poller, correctly said 14 solved.

    `deps.results` only accumulates during the current session, so it is empty
    after a restart. CTFd is the authority.
    """
    deps = _deps({})
    deps.results = {}                      # fresh session: nothing recorded yet
    deps.ctfd_solved = {"chal-a", "chal-b"}
    deps.ctfd_challenges = {"chal-a", "chal-b", "chal-c"}
    deps.challenge_metas = {
        n: SimpleNamespace(category="misc", value=10)
        for n in ("chal-a", "chal-b", "chal-c", "local-only")
    }

    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    assert sorted(data["solved"]) == ["chal-a", "chal-b"]
    assert data["ctfd_solved"] == ["chal-a", "chal-b"]


@pytest.mark.asyncio
async def test_session_results_and_ctfd_solved_are_unioned() -> None:
    """A solve confirmed this session must survive even if the poller has not
    picked it up yet."""
    deps = _deps({})
    deps.results = {"session-win": {"flag": "flag{x}", "submit": "confirmed by solver"}}
    deps.ctfd_solved = {"poller-win"}
    deps.ctfd_challenges = {"session-win", "poller-win"}

    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    assert sorted(data["solved"]) == ["poller-win", "session-win"]


@pytest.mark.asyncio
async def test_local_only_challenges_are_not_marked_pending() -> None:
    """challenges/ can hold sets that were never seeded to CTFd. Showing them as
    "待解题" implied the coordinator was neglecting work it could never see."""
    deps = _deps({})
    deps.ctfd_solved = set()
    deps.ctfd_challenges = {"on-ctfd"}
    deps.challenge_metas = {
        "on-ctfd": SimpleNamespace(category="misc", value=10),
        "local-only": SimpleNamespace(category="misc", value=10),
    }

    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    assert data["swarms"]["on-ctfd"]["pending"] is True
    assert data["swarms"]["on-ctfd"].get("local_only") is False
    assert data["swarms"]["local-only"]["pending"] is False
    assert data["swarms"]["local-only"]["local_only"] is True


@pytest.mark.asyncio
async def test_unknown_ctfd_set_does_not_mark_everything_local_only() -> None:
    """Before the poller's first tick ctfd_challenges is empty; treating that as
    "nothing is on CTFd" would flag the whole board as local-only."""
    deps = _deps({})
    deps.ctfd_solved = set()
    deps.ctfd_challenges = set()          # not yet populated
    deps.challenge_metas = {"chal": SimpleNamespace(category="misc", value=10)}

    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    assert data["swarms"]["chal"]["pending"] is True
    assert data["swarms"]["chal"]["local_only"] is False


@pytest.mark.asyncio
async def test_solved_challenges_are_not_also_marked_pending() -> None:
    """A solved challenge with no live swarm is done, not waiting.

    Marking it pending left the card badge correct (the UI tests `solved` first)
    but made the detail panel read "待解题（尚未 spawn）" and sorted it below the
    running ones — so after a restart every already-solved challenge looked like
    work the coordinator had not got to yet.
    """
    deps = _deps({})
    deps.results = {}
    deps.ctfd_solved = {"solved-chal"}
    deps.ctfd_challenges = {"solved-chal", "unsolved-chal"}
    deps.challenge_metas = {
        "solved-chal": SimpleNamespace(category="misc", value=10),
        "unsolved-chal": SimpleNamespace(category="misc", value=10),
    }

    srv = DashboardServer(deps, host="127.0.0.1", port=0, token=TOKEN)
    await srv.start()
    try:
        base = f"http://127.0.0.1:{_bound_port(srv)}"
        async with ClientSession() as s:
            data = await (await s.get(base + f"/api/state?token={TOKEN}")).json()
    finally:
        await srv.stop()

    solved_entry = data["swarms"]["solved-chal"]
    assert solved_entry["solved"] is True
    assert solved_entry["pending"] is False, "a solved challenge was also marked pending"

    unsolved_entry = data["swarms"]["unsolved-chal"]
    assert unsolved_entry["solved"] is False
    assert unsolved_entry["pending"] is True


def test_ui_prefers_solved_over_pending_in_status_and_sort() -> None:
    from backend.dashboard_ui import render_dashboard

    html = render_dashboard(TOKEN)
    # statusTxt must test solved before pending
    assert html.index("x.solved") < html.index("'⏸ 待解题")
    # and the sort rank must treat solved as ahead of pending
    assert "x.winner || x.solved" in html


def test_ui_distinguishes_local_only_from_pending() -> None:
    from backend.dashboard_ui import render_dashboard

    html = render_dashboard(TOKEN)
    assert "local_only" in html, "the UI never reads the local_only flag"
    assert "未上 CTFd" in html
    # solved must take precedence over pending in the badge, or a solved
    # challenge that is also still in challenge_metas renders as 待解题
    assert html.index("已解") < html.index("待解题")
