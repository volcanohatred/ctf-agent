"""Claude Agent SDK coordinator — uses the shared event loop with a Claude SDK client."""

from __future__ import annotations

import logging
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    create_sdk_mcp_server,
    tool,
)

from backend.agents.coordinator_core import (
    do_advise_agent,
    do_broadcast,
    do_bump_agent,
    do_check_swarm_status,
    do_fetch_challenges,
    do_get_solve_status,
    do_kill_swarm,
    do_read_solver_trace,
    do_spawn_swarm,
    do_submit_flag,
)
from backend.agents.coordinator_loop import build_deps, run_event_loop
from backend.config import Settings
from backend.deps import CoordinatorDeps

logger = logging.getLogger(__name__)

COORDINATOR_PROMPT = """\
You are a CTF competition coordinator running for the ENTIRE duration of a live competition.
Your job is to maximize the number of challenges solved.

Strategy:
- Spawn swarms for unsolved challenges, prioritizing by solve count (easy first)
- Use read_solver_trace to monitor what each solver is doing and where it's stuck
- When agents are stuck, read their traces, then craft targeted bumps with specific technical guidance
- Use broadcast to share cross-solver insights (e.g. flag format discovery, shared vulnerabilities)

COACHING TOOLS (use these to steer solvers — many solvers here are weaker models
that need active guidance):
- bump_agent: restart-level nudge with new insights (takes effect on the solver's next attempt)
- advise_agent: FAST channel — drops a message into a RUNNING solver's mailbox,
  picked up at its next tool call. Use this to redirect a solver mid-turn without
  waiting for its current attempt to finish.
- You will receive "PATROL ALERT: solver X on challenge Y looks stuck" events.
  When you see one: read that solver's trace, diagnose the exact problem, and
  send a specific advise_agent message telling it precisely what to try next.

KNOWLEDGE BASE:
- Every solver sandbox has a curated CTF technique library mounted read-only at /skills/
  (packs: ctf-web, ctf-pwn, ctf-reverse, ctf-crypto, ctf-forensics, ctf-misc, ctf-osint,
  ctf-malware, ctf-ai-ml, plus zh-knowledge/ Chinese writeups).
- Solvers already get auto-escalating skill hints when they repeatedly fail. Your job is
  to go further: read the trace, identify the EXACT technique that applies, and bump the
  solver with a specific file path + what to try from it (e.g. "read
  /skills/ctf-pwn/heap-techniques.md — the tcache poisoning section matches your case").

CRITICAL RULES:
- Do NOT kill a swarm because it looks stuck. Solvers rotate techniques, harvest
  their own findings on timeout, and often unstick themselves after several
  bumps. Your job is to HELP them, not give up on them.
- A swarm CAN retire itself, and you will see why in check_swarm_status
  (`retire_reason`): "round cap reached", "quota retries exhausted", or
  "N consecutive infrastructure failures". Round-cap retirement means the
  challenge is genuinely hard — re-spawn only with a concrete new idea.
- On an "INFRA ALERT" the gateway itself is failing. Do NOT bump that solver
  with new techniques; that wastes turns on a networking problem. Report it to
  the operator and leave the swarm alone.
- When a solver seems stuck, bump it with very specific technical guidance based
  on its trace. Tell it exactly what to try next — specific tools, techniques,
  approaches. Prefer advise_agent (reaches a RUNNING solver) over bump_agent
  (only takes effect on its next round).
- Solvers may end a turn with `still_working` instead of a flag. That is normal
  and honest — it is NOT a failure, and it is never a reason to kill a swarm.
- Spend deliberately: keep all swarms running while they are making progress,
  but do not re-spawn the same dead direction indefinitely.

You will receive event messages. Respond with tool calls to manage the competition.
"""


def _text(s: str) -> dict:
    """Wrap a string in the Claude SDK MCP tool return format."""
    return {"content": [{"type": "text", "text": s}]}


def _build_coordinator_mcp(deps: CoordinatorDeps):
    """Build MCP server — thin wrappers around coordinator_core functions."""

    @tool("fetch_challenges", "List all challenges with category, points, solve count, and status.", {})
    async def fetch_challenges(args: dict) -> dict:
        return _text(await do_fetch_challenges(deps))

    @tool("get_solve_status", "Check which challenges are solved and which swarms are running.", {})
    async def get_solve_status(args: dict) -> dict:
        return _text(await do_get_solve_status(deps))

    @tool("spawn_swarm", "Launch all solver models on a challenge.", {"challenge_name": str})
    async def spawn_swarm(args: dict) -> dict:
        return _text(await do_spawn_swarm(deps, args["challenge_name"]))

    @tool("check_swarm_status", "Get per-agent progress for a swarm.", {"challenge_name": str})
    async def check_swarm_status(args: dict) -> dict:
        return _text(await do_check_swarm_status(deps, args["challenge_name"]))

    @tool("submit_flag", "Submit a flag to CTFd.", {"challenge_name": str, "flag": str})
    async def submit_flag(args: dict) -> dict:
        return _text(await do_submit_flag(deps, args["challenge_name"], args["flag"]))

    @tool("kill_swarm", "Cancel all agents for a challenge. Requires a reason, which is "
          "recorded and shown to the operator — a killed swarm must never look like a crash.",
          {"challenge_name": str, "reason": str})
    async def kill_swarm(args: dict) -> dict:
        return _text(await do_kill_swarm(deps, args["challenge_name"], args.get("reason", "")))

    @tool("bump_agent", "Send targeted insights to a stuck agent.", {"challenge_name": str, "model_spec": str, "insights": str})
    async def bump_agent(args: dict) -> dict:
        return _text(await do_bump_agent(deps, args["challenge_name"], args["model_spec"], args["insights"]))

    @tool("advise_agent", "Drop guidance into a RUNNING solver's mailbox — picked up at its next tool call, no restart needed. Use for timely coaching when a solver is mid-turn.", {"challenge_name": str, "model_spec": str, "advice": str})
    async def advise_agent(args: dict) -> dict:
        return _text(await do_advise_agent(deps, args["challenge_name"], args["model_spec"], args["advice"]))

    @tool("broadcast", "Broadcast a strategic hint to ALL solvers on a challenge.", {"challenge_name": str, "message": str})
    async def broadcast(args: dict) -> dict:
        return _text(await do_broadcast(deps, args["challenge_name"], args["message"]))

    @tool("read_solver_trace", "Read recent trace events from a specific solver. Use this to understand what a solver is doing, what it tried, and where it's stuck.", {"challenge_name": str, "model_spec": str, "last_n": int})
    async def read_solver_trace(args: dict) -> dict:
        return _text(await do_read_solver_trace(deps, args["challenge_name"], args["model_spec"], args.get("last_n", 20)))

    return create_sdk_mcp_server(
        name="coordinator", version="1.0.0",
        tools=[fetch_challenges, get_solve_status, spawn_swarm, check_swarm_status,
               submit_flag, kill_swarm, bump_agent, advise_agent, broadcast, read_solver_trace],
    )


async def run_claude_coordinator(
    settings: Settings,
    model_specs: list[str] | None = None,
    challenges_root: str = "challenges",
    no_submit: bool = False,
    coordinator_model: str | None = None,
    msg_port: int = 0,
) -> dict[str, Any]:
    """Run the Claude Agent SDK coordinator with the shared event loop."""
    ctfd, cost_tracker, deps = build_deps(
        settings, model_specs, challenges_root, no_submit,
    )
    deps.msg_port = msg_port

    mcp_server = _build_coordinator_mcp(deps)
    # Coordinator = the "coach": default to the strongest configured gateway
    # model. CLI --coordinator-model overrides; COORDINATOR_MODEL env next.
    resolved_model = (
        coordinator_model
        or getattr(settings, "coordinator_model", "")
        or "qwen3.8-max"
    )

    allowed = {
        "mcp__coordinator__fetch_challenges", "mcp__coordinator__get_solve_status",
        "mcp__coordinator__spawn_swarm", "mcp__coordinator__check_swarm_status",
        "mcp__coordinator__submit_flag", "mcp__coordinator__kill_swarm",
        "mcp__coordinator__bump_agent", "mcp__coordinator__advise_agent",
        "mcp__coordinator__broadcast", "mcp__coordinator__read_solver_trace",
        "ToolSearch",
        "TaskCreate", "TaskUpdate", "TaskGet", "TaskList", "TaskOutput", "TaskStop",
    }

    async def enforce_allowlist(input_data, tool_use_id, context):
        if input_data.get("hook_event_name") != "PreToolUse":
            return {}
        tool = input_data.get("tool_name", "")
        if tool in allowed:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"{tool} not available to coordinator.",
            }
        }

    from backend.models import claude_cli_env
    options = ClaudeAgentOptions(
        model=resolved_model,
        system_prompt=COORDINATOR_PROMPT,
        env=claude_cli_env(settings),
        mcp_servers={"coordinator": mcp_server},
        allowed_tools=list(allowed),
        permission_mode="bypassPermissions",
        hooks={
            "PreToolUse": [HookMatcher(hooks=[enforce_allowlist])],
        },
    )

    async with ClaudeSDKClient(options=options) as client:
        # Session-high-water mark for the CLI's cumulative cost field.
        cost_hwm: list[float] = [0.0]

        async def turn_fn(msg: str) -> float:
            """Send one message to the coordinator LLM; return this turn's cost.

            The reported cost is a session high-water mark in current CLI
            versions and per-turn in older ones, so it is tracked adaptively —
            the same rule the solvers use — to avoid double-counting.
            """
            logger.debug("Coordinator query: %s", msg[:200])
            await client.query(msg)
            msg_count = 0
            cost = 0.0
            async for message in client.receive_response():
                msg_count += 1
                logger.debug("Coordinator received: %s", type(message).__name__)
                if isinstance(message, ResultMessage):
                    reported = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
                    if reported >= cost_hwm[0]:
                        cost = reported - cost_hwm[0]
                        cost_hwm[0] = reported
                    else:
                        cost = reported
                    logger.info(
                        "Coordinator turn done (messages=%d, turn=$%.4f, total=$%.4f, session=%s)",
                        msg_count, cost, cost_hwm[0], getattr(message, "session_id", None),
                    )
                    if getattr(message, "is_error", False):
                        logger.error(
                            "Coordinator turn reported an error: status=%s errors=%s",
                            getattr(message, "api_error_status", None),
                            getattr(message, "errors", None),
                        )
            if msg_count == 0:
                logger.warning("Coordinator turn produced no messages!")
            return cost

        # Assigned then returned outside the `async with` so the client is fully
        # closed before the result escapes, and every code path returns a dict.
        results = await run_event_loop(
            deps, ctfd, cost_tracker, turn_fn, coordinator_model=resolved_model,
        )
    return results
