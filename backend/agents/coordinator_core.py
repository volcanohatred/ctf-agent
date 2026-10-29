"""Shared coordinator tool logic — called by the Claude coordinator."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from backend.deps import CoordinatorDeps
from backend.message_bus import KIND_CLAIM, KIND_FACT, KIND_VERIFIED
from backend.prompts import ChallengeMeta
from backend.solver_base import FLAG_FOUND

logger = logging.getLogger(__name__)

# Archived swarm summaries are kept for the dashboard/export. Bounded so a
# long competition cannot grow this without limit.
MAX_SWARMS_HISTORY = 200


def _counter(solver, name: str) -> int:
    """Read a solver's step/tool counter as an int, whatever the backend stores.

    Prefers the typed public property; falls back to the private attribute (which is
    a one-element list on the Pydantic backend) so an older or partial solver object
    still yields a number rather than a list that silently breaks downstream
    comparisons.
    """
    value = getattr(solver, name, None)
    if value is None:
        value = getattr(solver, f"_{name}", 0)
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _archive_swarm(deps: CoordinatorDeps, name: str, swarm) -> None:
    """Snapshot a finished swarm into deps.swarms_history (bounded)."""
    try:
        feed = []
        bus = getattr(swarm, "message_bus", None)
        if bus is not None:
            for f in bus.findings:
                feed.append({"model": f.model, "kind": f.kind,
                             "content": f.content, "ts": f.timestamp})
        agents_hist = {}
        for spec in getattr(swarm, "_all_specs", None) or swarm.model_specs:
            solver = swarm.solvers.get(spec)
            agents_hist[spec] = {
                "status": "finished",
                "findings": swarm.findings.get(spec, ""),
                # Typed properties, not the private attributes: those are a plain int
                # on the Claude backend and a one-element LIST on the Pydantic one, so
                # a raw getattr archived `[0]` for half the lineup and the dashboard's
                # `!a.tool_calls` zero-work check never fired for them.
                "steps": _counter(solver, "step_count"),
                "tool_calls": _counter(solver, "tool_calls"),
            }
        entry = {
            "challenge": name,
            "category": getattr(swarm.meta, "category", ""),
            "value": getattr(swarm.meta, "value", 0),
            "description": getattr(swarm.meta, "description", ""),
            "tags": getattr(swarm.meta, "tags", []),
            "winner": swarm.winner.flag if (swarm.winner and swarm.winner.flag) else None,
            "winner_spec": swarm.winner_spec,
            "bumps": getattr(swarm, "_total_bumps", 0),
            "failed_approaches": getattr(swarm, "_failed_approaches", []),
            "retire_reason": getattr(swarm, "retire_reason", ""),
            # Kept in the archive so a post-mortem on a finished challenge can
            # still tell "explored widely, never converged" from "grinded one
            # dead paradigm for hours" — the live counters die with the swarm.
            "stale_breaks": getattr(swarm, "_stale_breaks", 0),
            "strategist_pivots": getattr(swarm, "_strategist_pivots", 0),
            "corroborated_facts": sum(1 for f in feed if f["kind"] == KIND_FACT),
            "probe_verified": sum(1 for f in feed if f["kind"] == KIND_VERIFIED),
            "unverified_claims": sum(1 for f in feed if f["kind"] == KIND_CLAIM),
            "probe_refuted": getattr(swarm, "_probe_refuted", 0),
            "contradictions": getattr(swarm, "_contradictions", 0),
            "audit_failures": getattr(swarm, "_audit_failures", 0),
            "strategist_failures": getattr(swarm, "_strategist_failures", 0),
            "agents": agents_hist,
            "shared_feed": feed,
            "finished_at": time.time(),
        }
    except Exception as e:  # noqa: BLE001 — archiving must never block spawning
        logger.warning("Could not archive swarm %s: %s", name, e)
        entry = {"challenge": name, "winner": None}

    deps.swarms_history[name] = entry
    # Evict oldest-first once over the cap (insertion order == archive order).
    while len(deps.swarms_history) > MAX_SWARMS_HISTORY:
        oldest = next(iter(deps.swarms_history))
        deps.swarms_history.pop(oldest, None)


async def reap_finished_swarms(deps: CoordinatorDeps) -> list[str]:
    """Retire swarms whose run task has actually completed.

    Two bugs fixed here:
      * Retirement used to trigger on `cancel_event.is_set()` alone. That flag
        means "cancellation requested", not "finished" — the task can still be
        inside `finally: await solver.stop()`. Archiving and dropping it at that
        point left an un-awaited task (orphan containers) and under-counted
        capacity, since `len(deps.swarms)` no longer included a live swarm.
      * Finished tasks were popped without ever reading `task.result()`, so any
        exception inside `swarm.run()` was silently swallowed.
    """
    retired: list[str] = []
    for name, task in list(deps.swarm_tasks.items()):
        if not task.done():
            continue
        swarm = deps.swarms.get(name)
        if swarm is not None:
            _archive_swarm(deps, name, swarm)
            del deps.swarms[name]
        deps.swarm_tasks.pop(name, None)
        retired.append(name)

        if task.cancelled():
            logger.info("Swarm %s was cancelled", name)
            continue
        exc = task.exception()
        if exc is not None:
            logger.error("Swarm %s failed: %s", name, exc, exc_info=exc)
    return retired


async def do_fetch_challenges(deps: CoordinatorDeps) -> str:
    challenges = await deps.ctfd.fetch_all_challenges()
    solved = await deps.ctfd.fetch_solved_names()
    result = [
        {
            "name": ch.get("name", "?"),
            "category": ch.get("category", "?"),
            "value": ch.get("value", 0),
            "solves": ch.get("solves", 0),
            "status": "SOLVED" if ch.get("name") in solved else "unsolved",
            "description": (ch.get("description") or "")[:200],
        }
        for ch in challenges
    ]
    return json.dumps(result, indent=2)


async def do_get_solve_status(deps: CoordinatorDeps) -> str:
    solved = await deps.ctfd.fetch_solved_names()
    await reap_finished_swarms(deps)
    swarm_status = {name: swarm.get_status() for name, swarm in deps.swarms.items()}
    return json.dumps(
        {"solved": sorted(solved), "active_swarms": swarm_status,
         "archived": sorted(deps.swarms_history)},
        indent=2,
    )


async def do_spawn_swarm(deps: CoordinatorDeps, challenge_name: str) -> str:
    # Retire genuinely-finished swarms first — archiving their summaries so the
    # dashboard and export keep showing them as completed.
    await reap_finished_swarms(deps)

    if challenge_name in deps.swarms:
        return f"Swarm still running for {challenge_name}"

    active_count = len(deps.swarm_tasks)
    if active_count >= deps.max_concurrent_challenges:
        return (f"At capacity ({active_count}/{deps.max_concurrent_challenges} "
                "challenges running). Wait for one to finish.")

    # Auto-pull challenge if needed
    if challenge_name not in deps.challenge_dirs:
        challenges = await deps.ctfd.fetch_all_challenges()
        ch_data = next((c for c in challenges if c.get("name") == challenge_name), None)
        if not ch_data:
            return f"Challenge '{challenge_name}' not found on CTFd"
        output_dir = str(Path(deps.challenges_root))
        ch_dir = await deps.ctfd.pull_challenge(ch_data, output_dir)
        deps.challenge_dirs[challenge_name] = ch_dir
        deps.challenge_metas[challenge_name] = ChallengeMeta.from_yaml(Path(ch_dir) / "metadata.yml")

    from backend.agents.swarm import ChallengeSwarm

    swarm = ChallengeSwarm(
        challenge_dir=deps.challenge_dirs[challenge_name],
        meta=deps.challenge_metas[challenge_name],
        ctfd=deps.ctfd,
        cost_tracker=deps.cost_tracker,
        settings=deps.settings,
        model_specs=list(deps.model_specs),
        no_submit=deps.no_submit,
        coordinator_inbox=deps.coordinator_inbox,
    )
    deps.swarms[challenge_name] = swarm

    async def _run_and_cleanup() -> None:
        try:
            result = await swarm.run()
        finally:
            # A swarm that retired itself (round cap, infra failure, quota) must
            # stay visible to the operator instead of looking "still running".
            reason = getattr(swarm, "retire_reason", "")
            if reason:
                logger.warning("Swarm %s retired: %s", challenge_name, reason)
        # Flag already submitted/confirmed by solver's submit_fn — just record the result
        if result and result.status == FLAG_FOUND and result.flag:
            deps.results[challenge_name] = {
                "flag": result.flag,
                "submit": "DRY RUN" if deps.no_submit else "confirmed by solver",
            }

    task = asyncio.create_task(_run_and_cleanup(), name=f"swarm-{challenge_name}")
    deps.swarm_tasks[challenge_name] = task
    return f"Swarm spawned for {challenge_name} with {len(deps.model_specs)} models"


async def do_check_swarm_status(deps: CoordinatorDeps, challenge_name: str) -> str:
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    return json.dumps(swarm.get_status(), indent=2)


async def do_submit_flag(deps: CoordinatorDeps, challenge_name: str, flag: str) -> str:
    if deps.no_submit:
        return f'DRY RUN — would submit "{flag.strip()}" for {challenge_name}'
    try:
        result = await deps.ctfd.submit_flag(challenge_name, flag)
        return result.display
    except Exception as e:
        return f"submit_flag error: {e}"


async def do_kill_swarm(deps: CoordinatorDeps, challenge_name: str, reason: str = "") -> str:
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    why = (reason or "").strip() or "killed by the coordinator (no reason given)"
    swarm.kill(f"operator/coordinator kill: {why}")
    # Logged explicitly: this used to be the one swarm-termination path that left
    # no trace anywhere, so a killed swarm was indistinguishable from a crash.
    logger.warning("Swarm %s killed on request: %s", challenge_name, why)
    return f"Swarm for {challenge_name} cancelled ({why})"


async def do_bump_agent(deps: CoordinatorDeps, challenge_name: str, model_spec: str, insights: str) -> str:
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    solver = swarm.solvers.get(model_spec)
    if not solver:
        return f"No solver for {model_spec} in {challenge_name}"
    solver.bump(insights)
    return f"Bumped {model_spec} on {challenge_name}"


async def do_advise_agent(deps: CoordinatorDeps, challenge_name: str, model_spec: str, advice: str) -> str:
    """Drop guidance into a RUNNING solver's mailbox (picked up mid-run via its
    tool hook — no need to wait for the current attempt to finish)."""
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    solver = swarm.solvers.get(model_spec)
    if not solver:
        return f"No solver for {model_spec} in {challenge_name}"
    await swarm.message_bus.post_advice(model_spec, advice)
    return f"Advised {model_spec} on {challenge_name} (delivered at its next tool call)"


async def do_read_solver_trace(deps: CoordinatorDeps, challenge_name: str, model_spec: str, last_n: int = 20) -> str:
    """Read the last N trace events from a solver's JSONL log."""
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm for {challenge_name}"
    solver = swarm.solvers.get(model_spec)
    if not solver:
        known = ", ".join(sorted(swarm.solvers)) or "(none)"
        return f"No solver for {model_spec}. Running: {known}"
    tracer = getattr(solver, "tracer", None)
    path = getattr(tracer, "path", None) or (str(tracer) if tracer else "")
    if not path:
        return "No tracer on solver"
    last_n = max(1, min(int(last_n or 20), 400))
    try:
        # Blocking file I/O off the event loop: traces grow large and the
        # coordinator loop must keep servicing the poller and patrol.
        lines = (await asyncio.to_thread(_read_tail_lines, path, last_n * 4))[-last_n:]
    except FileNotFoundError:
        return f"Trace file not found: {path}"
    except OSError as e:
        return f"Error reading trace: {e}"

    summary = []
    for line in lines:
        try:
            d = json.loads(line)
            t = d.get("type", "?")
            if t == "tool_call":
                summary.append(f"step {d.get('step','?')} CALL {d.get('tool','?')}: "
                               f"{str(d.get('args', ''))[:100]}")
            elif t == "tool_result":
                summary.append(f"step {d.get('step','?')} RESULT {d.get('tool','?')}: "
                               f"{str(d.get('result', ''))[:100]}")
            elif t in ("finish", "error", "bump", "turn_failed", "budget_interrupt",
                       "conclude_ok", "conclude_failed", "loop_break", "flag_rejected",
                       "structured_flag_verified"):
                summary.append(f"** {t}: "
                               f"{json.dumps({k: v for k, v in d.items() if k != 'ts'})[:300]}")
            elif t == "usage":
                summary.append(f"usage: in={d.get('input_tokens',0)} "
                               f"out={d.get('output_tokens',0)} cost=${d.get('cost_usd',0):.4f}")
            else:
                summary.append(f"{t}: {str(d)[:80]}")
        except json.JSONDecodeError:
            summary.append(line[:100])
    return "\n".join(summary) or "(trace is empty)"


def _read_tail_lines(path: str, n: int, chunk: int = 65536) -> list[str]:
    """Return the last `n` non-empty lines without loading the whole file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - chunk * max(1, n // 50)))
        tail = fh.read()
    lines = [line for line in tail.splitlines() if line.strip()]
    return lines[-n:]


async def do_broadcast(deps: CoordinatorDeps, challenge_name: str, message: str) -> str:
    """Broadcast a message to all solvers working on a challenge."""
    swarm = deps.swarms.get(challenge_name)
    if not swarm:
        return f"No swarm running for {challenge_name}"
    await swarm.message_bus.broadcast(message)
    return f"Broadcast to all solvers on {challenge_name}"
