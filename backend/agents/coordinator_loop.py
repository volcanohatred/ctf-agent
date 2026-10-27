"""Shared coordinator event loop — used by the Claude SDK coordinator."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from backend.agents.coordinator_core import reap_finished_swarms
from backend.config import Settings
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.deps import CoordinatorDeps
from backend.models import DEFAULT_MODELS
from backend.poller import CTFdPoller
from backend.prompts import ChallengeMeta
from backend.trace_read import trace_health

logger = logging.getLogger(__name__)

# A coordinator turn returns the provider-reported cost of that turn in USD
# (0.0 when unknown), so the coordinator's own spend lands in CostTracker
# instead of being logged and thrown away.
TurnFn = Callable[[str], Coroutine[Any, Any, float]]

# Consecutive failed/timed-out turns before the loop backs off instead of
# spinning against a stalling gateway.
TURN_FAILURE_BACKOFF = 5
TURN_BACKOFF_SECONDS = 60.0


def build_deps(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    challenge_dirs: dict[str, str] | None = None,
    challenge_metas: dict[str, ChallengeMeta] | None = None,
) -> tuple[CTFdClient, CostTracker, CoordinatorDeps]:
    """Create CTFd client, cost tracker, and coordinator deps."""
    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )
    cost_tracker = CostTracker()
    specs = model_specs or list(DEFAULT_MODELS)
    Path(challenges_root).mkdir(parents=True, exist_ok=True)

    deps = CoordinatorDeps(
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=specs,
        challenges_root=challenges_root,
        no_submit=no_submit,
        max_concurrent_challenges=getattr(settings, "max_concurrent_challenges", 10),
        msg_port=getattr(settings, "msg_port", 0),
        msg_host=getattr(settings, "msg_host", "127.0.0.1"),
        challenge_dirs=challenge_dirs or {},
        challenge_metas=challenge_metas or {},
    )

    # Pre-load already-pulled challenges
    for d in Path(challenges_root).iterdir():
        meta_path = d / "metadata.yml"
        if not meta_path.exists():
            continue
        try:
            meta = ChallengeMeta.from_yaml(meta_path)
        except Exception as e:  # noqa: BLE001 — one bad dir must not stop startup
            logger.warning("Skipping %s (bad metadata.yml: %s)", d, e)
            continue
        if meta.name not in deps.challenge_dirs:
            deps.challenge_dirs[meta.name] = str(d)
            deps.challenge_metas[meta.name] = meta

    return ctfd, cost_tracker, deps


async def run_event_loop(
    deps: CoordinatorDeps,
    ctfd: CTFdClient,
    cost_tracker: CostTracker,
    turn_fn: TurnFn,
    status_interval: int = 60,
    coordinator_model: str = "coordinator",
) -> dict[str, Any]:
    """Run the shared coordinator event loop.

    Args:
        deps: Coordinator dependencies (shared state).
        ctfd: CTFd client (for poller).
        cost_tracker: Cost tracker.
        turn_fn: Async function that sends a message to the coordinator LLM and
            returns that turn's cost in USD.
        status_interval: Seconds between status updates.
        coordinator_model: Label used when accounting the coordinator's spend.
    """
    poller = CTFdPoller(ctfd=ctfd, interval_s=5.0)
    await poller.start()

    # Start operator message HTTP endpoint
    msg_server = await _start_msg_server(deps.operator_inbox, deps.msg_port, deps.msg_host)

    logger.info(
        "Coordinator starting: %d models, %d challenges, %d solved",
        len(deps.model_specs),
        len(poller.known_challenges),
        len(poller.known_solved),
    )

    unsolved = poller.known_challenges - poller.known_solved
    # Seed the dashboard's authoritative view before the first loop iteration, so
    # a freshly restarted coordinator does not briefly show every already-solved
    # challenge as pending.
    deps.ctfd_solved = poller.known_solved
    deps.ctfd_challenges = poller.known_challenges
    initial_msg = (
        f"CTF is LIVE. {len(poller.known_challenges)} challenges, "
        f"{len(poller.known_solved)} solved.\n"
        f"Unsolved: {sorted(unsolved) if unsolved else 'NONE'}\n"
        "Fetch challenges and spawn swarms for all unsolved."
    )

    patrol_task: asyncio.Task | None = None
    dashboard = None
    turn_timeout = float(getattr(deps.settings, "coordinator_turn_timeout_s", 300) or 300)
    consecutive_turn_failures = 0

    async def _turn(msg: str) -> bool:
        """Run one coordinator turn under a hard timeout, accounting its cost.

        Without the timeout, a gateway that accepts the connection and then
        stalls blocks the ENTIRE event loop — poller, patrol, operator messages
        and dashboard all freeze with it, and nothing logs why.
        """
        nonlocal consecutive_turn_failures
        t0 = time.monotonic()
        try:
            cost = await asyncio.wait_for(turn_fn(msg), timeout=turn_timeout)
            consecutive_turn_failures = 0
            cost_tracker.record_cost_only(
                "coordinator", coordinator_model, float(cost or 0.0),
                duration_seconds=time.monotonic() - t0,
            )
            return True
        except TimeoutError:
            consecutive_turn_failures += 1
            logger.error(
                "Coordinator turn timed out after %.0fs (%d consecutive) — "
                "the gateway may be stalling", turn_timeout, consecutive_turn_failures,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — keep the loop alive
            consecutive_turn_failures += 1
            logger.error("Coordinator turn failed (%d consecutive): %s",
                         consecutive_turn_failures, e)
        return False

    try:
        await _turn(initial_msg)

        # Auto-spawn swarms for unsolved challenges if coordinator LLM didn't
        await _auto_spawn_unsolved(deps, poller)

        # Operator dashboard (live swarm view + guidance mailbox UI)
        dash_port = int(getattr(deps.settings, "dashboard_port", 8080) or 0)
        if dash_port > 0:
            try:
                from backend.web_dashboard import DashboardServer
                dashboard = DashboardServer(
                    deps,
                    host=getattr(deps.settings, "dashboard_host", "127.0.0.1"),
                    port=dash_port,
                    token=getattr(deps.settings, "dashboard_token", "") or "",
                )
                await dashboard.start()
            except Exception as e:  # noqa: BLE001 — the dashboard is optional
                logger.warning("Dashboard failed to start: %s", e)

        # Start the proactive patrol (stuck-solver detection + fast nudges)
        patrol_interval = int(getattr(deps.settings, "patrol_interval_s", 120) or 120)
        if patrol_interval > 0:
            patrol_task = asyncio.create_task(
                _patrol_loop(
                    deps,
                    interval_s=patrol_interval,
                    stuck_quiet_s=int(getattr(deps.settings, "patrol_stuck_quiet_s", 240) or 240),
                ),
                name="coordinator-patrol",
            )

        last_status = asyncio.get_event_loop().time()

        while True:
            events = []
            evt = await poller.get_event(timeout=5.0)
            if evt:
                events.append(evt)
            events.extend(poller.drain_events())

            # Publish CTFd's authoritative view for the dashboard. Doing it every
            # iteration rather than at startup is what makes the UI correct after
            # a restart, and keeps it correct as solves accumulate.
            deps.ctfd_solved = poller.known_solved
            deps.ctfd_challenges = poller.known_challenges

            # Auto-kill swarms for solved challenges, and record the solve.
            for evt in events:
                if evt.kind == "challenge_solved":
                    record_challenge_solved(deps, evt.challenge_name)

            parts: list[str] = []
            for evt in events:
                if evt.kind == "new_challenge":
                    parts.append(f"NEW CHALLENGE: '{evt.challenge_name}' appeared. Spawn a swarm.")
                    # Auto-spawn for new challenges
                    await _auto_spawn_one(deps, evt.challenge_name)
                elif evt.kind == "challenge_solved":
                    parts.append(f"SOLVED: '{evt.challenge_name}' — swarm auto-killed.")

            # Retire genuinely-finished swarms (archives them, surfaces errors)
            for name in await reap_finished_swarms(deps):
                parts.append(
                    f"SOLVER FINISHED: Swarm for '{name}' completed. "
                    "Check results, or spawn_swarm again to retry."
                )

            # Drain solver-to-coordinator messages
            while True:
                try:
                    solver_msg = deps.coordinator_inbox.get_nowait()
                    parts.append(f"SOLVER MESSAGE: {solver_msg}")
                except asyncio.QueueEmpty:
                    break

            # Drain operator messages
            while True:
                try:
                    op_msg = deps.operator_inbox.get_nowait()
                    parts.append(f"OPERATOR MESSAGE: {op_msg}")
                    logger.info("Operator message: %s", op_msg[:200])
                except asyncio.QueueEmpty:
                    break

            # Periodic status update — only when there are active swarms or other events
            now = asyncio.get_event_loop().time()
            if now - last_status >= status_interval:
                last_status = now
                # Re-check for unsolved challenges with no swarm. This is what
                # makes the coordinator self-heal: a swarm that retired (round
                # cap, quota, infrastructure) leaves its challenge unsolved and
                # nothing else would ever retry it, and a CTFd solve-history reset
                # emits no event at all.
                try:
                    n = await _auto_spawn_unsolved(
                        deps, poller,
                        max_spawns=int(getattr(deps.settings, "max_spawns_per_challenge", 3) or 3),
                    )
                    if n:
                        logger.info("Re-spawned %d unsolved challenge(s)", n)
                        parts.append(
                            f"AUTO-SPAWN: {n} unsolved challenge(s) had no active swarm "
                            "(retired or newly reset) and were re-queued."
                        )
                except Exception as e:  # noqa: BLE001 — never let this kill the loop
                    logger.warning("Periodic re-spawn check failed: %s", e)

                active = [n for n, t in deps.swarm_tasks.items() if not t.done()]
                solved_set = poller.known_solved
                unsolved_set = poller.known_challenges - solved_set
                status_line = (
                    f"STATUS: {len(solved_set)} solved, {len(unsolved_set)} unsolved, "
                    f"{len(active)} active swarms. Cost: ${cost_tracker.total_cost_usd:.2f}"
                )
                # Only send to coordinator if there's something happening
                if active or parts:
                    parts.append(status_line)
                else:
                    logger.info("Event -> coordinator: %s", status_line)

            if parts:
                msg = "\n\n".join(parts)
                logger.info("Event -> coordinator: %s", msg[:200])
                await _turn(msg)
                if consecutive_turn_failures >= TURN_FAILURE_BACKOFF:
                    logger.error(
                        "Coordinator failed %d turns in a row — backing off %.0fs "
                        "instead of spinning", consecutive_turn_failures, TURN_BACKOFF_SECONDS,
                    )
                    await asyncio.sleep(TURN_BACKOFF_SECONDS)
                    consecutive_turn_failures = 0

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Coordinator shutting down...")
    except Exception as e:  # noqa: BLE001 — log, then run the same teardown
        logger.error("Coordinator fatal: %s", e, exc_info=True)
    finally:
        if patrol_task:
            patrol_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await patrol_task
        if dashboard:
            with contextlib.suppress(Exception):
                await dashboard.stop()
        if msg_server:
            msg_server.close()
            with contextlib.suppress(Exception):
                await msg_server.wait_closed()
        await poller.stop()
        for swarm in deps.swarms.values():
            swarm.kill("coordinator shutting down")
        for task in deps.swarm_tasks.values():
            task.cancel()
        if deps.swarm_tasks:
            await asyncio.gather(*deps.swarm_tasks.values(), return_exceptions=True)
        cost_tracker.log_summary()
        with contextlib.suppress(Exception):
            await ctfd.close()

    return {
        "results": deps.results,
        "total_cost_usd": cost_tracker.total_cost_usd,
        "total_tokens": cost_tracker.total_tokens,
        "archived": sorted(deps.swarms_history),
    }


def record_challenge_solved(deps: CoordinatorDeps, name: str) -> None:
    """Record a CTFd-confirmed solve and retire any swarm still working on it.

    CTFd is the authority, and the poller can observe a solve BEFORE the winning
    swarm has stored its own winner. Killing the swarm at that moment used to lose
    the result entirely — observed live as the dashboard reporting 13 solved while
    CTFd and the coordinator's own STATUS line both said 14, because
    `_run_and_cleanup` only records a result when `swarm.run()` returns FLAG_FOUND.
    """
    swarm = deps.swarms.get(name)
    if name not in deps.results:
        flag = ""
        if swarm is not None:
            winner = getattr(swarm, "winner", None)
            flag = getattr(winner, "flag", "") or ""
            if not flag:
                flag = getattr(swarm, "confirmed_flag", "") or ""
        deps.results[name] = {
            "flag": flag,
            "submit": ("confirmed by solver" if flag
                       else "confirmed on CTFd (observed by poller)"),
        }
    if swarm is not None and not swarm.cancel_event.is_set():
        swarm.kill("challenge confirmed solved on CTFd")
        logger.info("Auto-killed swarm for: %s", name)


async def _auto_spawn_one(deps: CoordinatorDeps, challenge_name: str) -> None:
    """Auto-spawn a swarm for a single challenge if not already running."""
    if challenge_name in deps.swarms or challenge_name in deps.swarm_tasks:
        return
    active = sum(1 for t in deps.swarm_tasks.values() if not t.done())
    if active >= deps.max_concurrent_challenges:
        return
    try:
        from backend.agents.coordinator_core import do_spawn_swarm
        result = await do_spawn_swarm(deps, challenge_name)
        logger.info("Auto-spawn %s: %s", challenge_name, result[:100])
    except Exception as e:  # noqa: BLE001 — one failure must not stop the others
        logger.warning("Auto-spawn failed for %s: %s", challenge_name, e)


async def _auto_spawn_unsolved(deps: CoordinatorDeps, poller, max_spawns: int = 3) -> int:
    """Spawn swarms for unsolved challenges that have no active swarm.

    Called at startup AND periodically thereafter. The periodic call matters for
    two cases that otherwise leave a challenge permanently unattended:
      * its solve history was reset (CTFd emits no "became unsolved" event, so
        nothing else notices), and
      * its swarm retired — round cap, quota exhaustion or infrastructure
        failures — without solving it.
    Bounded by `max_spawns` per challenge so a genuinely hopeless challenge is
    not re-queued forever.
    """
    unsolved = poller.known_challenges - poller.known_solved
    spawned = 0
    for name in sorted(unsolved):
        if deps.spawn_attempts.get(name, 0) >= max_spawns:
            continue
        before = len(deps.swarm_tasks)
        await _auto_spawn_one(deps, name)
        if len(deps.swarm_tasks) > before:
            deps.spawn_attempts[name] = deps.spawn_attempts.get(name, 0) + 1
            spawned += 1
    return spawned


# ---------------------------------------------------------------------------
# Proactive patrol — detect stuck solvers, nudge them via the fast mailbox.
# ---------------------------------------------------------------------------

# Signals that a solver is stuck on something it doesn't understand — used by
# the patrol to auto-surface a help point (so the operator sees the gap even
# if the model never calls request_help itself).
_CONFUSION_SIGNALS = (
    "不知道", "不懂", "不理解", "不明白", "不认识", "看不懂", "不了解", "不熟悉", "没见过",
    "unknown", "unfamiliar", "don't know", "not sure", "no idea",
    "confusing", "unclear", "unrecogniz", "what is this", "custom protocol",
    "proprietary", "never seen", "makes no sense",
)

# Signals that the failure is infrastructure, not reasoning — nudging a solver
# to "try a different technique" when its endpoint is down is useless noise.
_INFRA_SIGNALS = (
    "api error", "connection refused", "econnrefused", "econnreset", "etimedout",
    "overloaded", "rate limit", "502 bad gateway", "503 service", "504 gateway",
)


def _looks_confused(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in _CONFUSION_SIGNALS)


def _looks_like_infra_failure(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in _INFRA_SIGNALS)


async def _patrol_loop(deps: CoordinatorDeps, interval_s: int, stuck_quiet_s: int) -> None:
    """Periodically scan active solvers; alert + advise the stuck ones.

    A solver counts as stuck when its trace shows:
      * no new event for `stuck_quiet_s` seconds, OR
      * the same tool call repeated with no variety, OR
      * turns completing with ZERO tool calls — a dead endpoint or broken tool
        wiring. This is invisible to a tool-signature-only check, which is how a
        benchmark once ground through 40 empty rounds per challenge while the
        patrol reported everything healthy.

    Detection writes a template nudge to the solver's fast mailbox (picked up
    mid-run) AND alerts the coordinator inbox so the coordinator LLM can follow
    up with real guidance via advise_agent / bump_agent.

    The scan is pure file I/O, so it runs in a worker thread: doing it inline
    stalled the event loop every patrol interval.
    """
    reported: set[str] = set()  # solver keys currently in a reported-stuck episode
    loop = asyncio.get_running_loop()
    while True:
        try:
            await asyncio.sleep(interval_s)
            await asyncio.to_thread(_patrol_scan, deps, stuck_quiet_s, reported, loop)
        except asyncio.CancelledError:
            return
        except Exception as e:  # noqa: BLE001 — the patrol must never die
            logger.debug("patrol loop error: %s", e)


def _patrol_scan(deps: CoordinatorDeps, stuck_quiet_s: int, reported: set[str],
                 loop: asyncio.AbstractEventLoop) -> None:
    """One synchronous patrol pass. Runs in a worker thread — file I/O only."""
    now = time.time()
    live_keys: set[str] = set()

    for ch_name, swarm in list(deps.swarms.items()):
        if swarm.cancel_event.is_set():
            continue
        for spec, solver in list(swarm.solvers.items()):
            key = f"{ch_name}|{spec}"
            live_keys.add(key)
            trace_path = getattr(getattr(solver, "tracer", None), "path", None)
            if not trace_path or not Path(trace_path).exists():
                continue
            try:
                health = trace_health(trace_path)
            except Exception:  # noqa: BLE001 — unreadable trace means "unknown"
                continue

            quiet_s = now - health.last_ts if health.last_ts else 0.0
            if quiet_s >= stuck_quiet_s:
                reason = f"no activity for {int(quiet_s)}s"
            elif health.spinning:
                reason = (f"{health.turns} turns completed with 0 tool calls "
                          "(endpoint or tool wiring is broken)")
            elif health.looping:
                reason = "repeating the same command"
            else:
                reported.discard(key)  # recovered → allow future alerts
                continue

            if key in reported:
                continue
            reported.add(key)

            evidence = ""
            if health.looping and health.tool_sigs:
                from collections import Counter

                counts = Counter(health.tool_sigs[-6:])
                top, top_n = counts.most_common(1)[0]
                evidence = f" You ran this {top_n}x in a row: `{top[:120]}`."
            logger.warning("[patrol] %s stuck (%s)", key, reason)

            # An infrastructure failure is not something the solver can reason
            # its way out of — tell the operator, don't nag the model.
            if _looks_like_infra_failure(health.last_text):
                _enqueue(deps.coordinator_inbox,
                         f"INFRA ALERT: solver '{spec}' on '{ch_name}' is getting API/transport "
                         f"errors ({health.last_text[:160]}). This is NOT a technique problem — "
                         "do not bump it with new directions. Check gateway reachability.")
                continue

            bus = getattr(swarm, "message_bus", None)
            if bus is not None and _looks_confused(health.last_text):
                gap = (f"模型 {spec} 卡在了它不懂的东西上。它的原话思路："
                       f"「{health.last_text[:280]}」")
                _schedule(bus.post_help(spec, gap), loop)
                logger.info("[patrol] %s auto-help-point posted", key)

            nudge_parts = [
                f"You seem stuck ({reason}).{evidence} STOP and reset: "
                "write down what you've confirmed so far, then pick a "
                "DIFFERENT technique.",
            ]
            try:
                queue = getattr(swarm, "_approach_queue", [])
                failed = set(getattr(swarm, "_failed_approach_paths", []))
                nxt = next((a for a in queue if a.path not in failed), None)
                if nxt:
                    nudge_parts.append(
                        f"Suggested untried direction: **{nxt.title}** — "
                        f"read `{nxt.path}` first, then apply it."
                    )
            except Exception:  # noqa: BLE001 — guidance is best-effort
                logger.debug("[patrol] could not pick a direction for %s", key)
            nudge_parts.append(
                "If a task-specific handbook exists under /skills/, read it "
                "before retrying."
            )
            if bus is not None:
                _schedule(bus.post_advice(spec, "\n".join(nudge_parts)), loop)
            _enqueue(deps.coordinator_inbox,
                     f"PATROL ALERT: solver '{spec}' on '{ch_name}' looks stuck "
                     f"({reason}).{evidence} Use read_solver_trace to see what it's doing, "
                     f"then advise_agent with specific next steps.")

    # Forget keys for swarms/solvers that no longer exist, so `reported` cannot
    # grow for the lifetime of the process.
    reported.intersection_update(live_keys)


def _enqueue(inbox: asyncio.Queue | None, message: str) -> None:
    if inbox is None:
        return
    try:
        inbox.put_nowait(message)
    except asyncio.QueueFull:
        logger.warning("Coordinator inbox full — dropping: %s", message[:120])


def _schedule(coro, loop: asyncio.AbstractEventLoop) -> None:
    """Fire a coroutine on the main loop from the patrol's worker thread."""
    if loop.is_closed():
        coro.close()
        return
    asyncio.run_coroutine_threadsafe(coro, loop)


async def _start_msg_server(inbox: asyncio.Queue, port: int = 0,
                            host: str = "127.0.0.1") -> asyncio.Server | None:
    """Start a tiny HTTP server that accepts operator messages via POST."""

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            headers: dict[str, str] = {}
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5)
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    k, v = line.decode(errors="replace").split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            method = request_line.decode(errors="replace").split()[0] if request_line else ""
            try:
                content_length = int(headers.get("content-length", 0))
            except ValueError:
                content_length = 0

            if method == "POST" and 0 < content_length <= 65536:
                body = await asyncio.wait_for(reader.read(content_length), timeout=5)
                message = _decode_message(body)
                inbox.put_nowait(message)
                await _respond(writer, 200, {"ok": True, "queued": message[:200]})
            elif method == "POST":
                await _respond(writer, 413, {"error": "body must be 1..65536 bytes"})
            else:
                await _respond(writer, 400, {
                    "error": "POST with a JSON body required",
                    "usage": 'POST {"message": "..."}',
                })
        except Exception as e:  # noqa: BLE001 — one bad request must not kill the server
            logger.debug("msg server request error: %s", e)
        finally:
            with contextlib.suppress(Exception):
                writer.close()

    try:
        server = await asyncio.start_server(_handle, host, port)
    except OSError as e:
        logger.warning("Could not start operator message endpoint on %s:%s — %s",
                       host, port, e)
        return None
    actual = server.sockets[0].getsockname()[1] if server.sockets else port
    logger.info("Operator message endpoint listening on http://%s:%d", host, actual)
    if port == 0:
        logger.warning(
            "msg_port=0 picked port %d, but `ctf-msg` defaults to 9400 — pass "
            "`ctf-msg --port %d` or set MSG_PORT=9400", actual, actual,
        )
    return server


def _decode_message(body: bytes) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body.decode("utf-8", errors="replace")
    if isinstance(data, dict) and data.get("message"):
        return str(data["message"])
    return body.decode("utf-8", errors="replace")


async def _respond(writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    reason = {200: "OK", 400: "Bad Request", 413: "Payload Too Large"}.get(status, "OK")
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
    )
    await writer.drain()
