"""Claude Agent SDK solver  — native tools with execution hooks.

Uses Claude's native Bash tool, but intercepts every command via a PreToolUse
hook and rewrites it to run inside the Docker sandbox via `docker exec`. Read,
Write, and Edit are blocked  — the model uses bash for all file operations.
Flag submission is intercepted from bash commands matching `submit_flag <flag>`.

Turn lifecycle (Cairn-style two phases):
  phase 1  run_until_done_or_gave_up(budget_s) — on budget expiry the turn is
           INTERRUPTED through the SDK (not task-cancelled), so the session
           stays alive and resumable.
  phase 2  conclude(budget_s) — the same session is asked to distil what it had
           confirmed, under its own independent budget, with all tools denied.
           Any failure returns "" and the caller must then write NOTHING to the
           blackboard.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shlex
import time
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
)

from backend.config import int_setting, str_setting
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.loop_detect import LoopDetector
from backend.models import model_id_from_spec
from backend.output_types import solver_output_json_schema
from backend.prompts import CONCLUDE_PROMPT, ChallengeMeta, build_prompt, list_distfiles
from backend.sandbox import sandbox_from_settings
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    INFRA_ERROR,
    SolverResult,
    classify_api_error,
    classify_api_status,
    classify_exception,
    is_context_overflow,
    is_context_overflow_text,
)
from backend.tracing import SolverTracer

if TYPE_CHECKING:
    from backend.config import Settings

logger = logging.getLogger(__name__)

# How long a phase-1 interrupt is given to produce a ResultMessage before the
# stream is abandoned. The CLI must unwind an in-flight tool call first.
INTERRUPT_GRACE_S = 45.0
# Poll granularity while waiting for the next stream message.
STREAM_POLL_S = 2.0
# Cap on retained assistant text blocks. Findings are accumulated rather than
# overwritten, but they must not grow without bound either.
MAX_FINDING_BLOCKS = 24
MAX_FINDING_CHARS = 24_000
# Restart payloads carry the distilled journal, the team digest, an escalating
# skill hint (whose L3 form embeds a whole pack index, up to MAX_INDEX_CHARS) and
# the next untried technique. Measured live at 7.8-10.5 KB, so a 4 KB window cut
# the tail — and because the team digest sits after the bulky skill hint, the
# digest was exactly what got lost. Size the window to hold a real payload;
# traces are pruned to the newest 500 files anyway, so this is bounded.
BUMP_TRACE_CHARS = 16_000

_STREAM_END = object()


class ClaudeSolver:
    """Claude Agent SDK solver using native tools redirected to Docker sandbox."""

    def __init__(
        self,
        model_spec: str,
        challenge_dir: str,
        meta: ChallengeMeta,
        ctfd: CTFdClient,
        cost_tracker: CostTracker,
        settings: Settings,
        cancel_event: asyncio.Event | None = None,
        no_submit: bool = False,
        submit_fn=None,
        message_bus=None,
        notify_coordinator=None,
        role_prompt: str = "",
        conclude_prompt: str = "",
    ) -> None:
        self.model_spec = model_spec
        self.model_id = model_id_from_spec(model_spec)
        self.challenge_dir = challenge_dir
        self.meta = meta
        self.ctfd = ctfd
        self.cost_tracker = cost_tracker
        self.settings = settings
        self.cancel_event = cancel_event or asyncio.Event()
        self.no_submit = no_submit
        self.submit_fn = submit_fn
        self.message_bus = message_bus
        self.notify_coordinator = notify_coordinator
        # Replaces the generated solver system prompt entirely. This is what lets
        # the Verifier reuse every piece of hard-won machinery below — sandbox
        # redirect, loop detection, tracing, budget interruption, cost accounting,
        # shielded teardown — while doing a completely different job. Without it a
        # verifier would either have to be a second solver implementation or run
        # under "find the flag" instructions it is not trying to follow.
        self.role_prompt = (role_prompt or "").strip()
        # Replaces CONCLUDE_PROMPT for the tools-denied phase-2 harvest. The
        # Verifier needs this: its phase 2 must demand a VERDICT line, not a
        # [excluded]/[fact] state card, and phase 2 is the only reliable way to get
        # an answer at all — it denies every tool, so a model that would otherwise
        # keep exploring cannot.
        self.conclude_prompt = (conclude_prompt or "").strip()

        self.sandbox = sandbox_from_settings(challenge_dir, settings)
        # Cleared by detach_sandbox() when the quota fallback hands this
        # solver's live container over to the Pydantic backend.
        self._owns_sandbox = True
        self.loop_detector = LoopDetector()
        self.tracer = SolverTracer(meta.name, self.model_id)
        self.agent_name = f"{meta.name}/{self.model_id}"

        self._client: ClaudeSDKClient | None = None
        self._session_id: str | None = None
        self._container_id: str = ""
        self._step_count = 0
        # Tool calls actually dispatched (denied/intercepted calls excluded).
        # The swarm uses this — not _step_count — to detect a broken solver.
        self._tool_calls = 0
        self._flag: str | None = None
        self._confirmed = False
        # Accumulated assistant text, newest last. Overwriting this with only
        # the final block (as an earlier revision did) discarded everything the
        # model had confirmed earlier in the round, which made timeout
        # harvesting nearly useless.
        self._finding_blocks: deque[str] = deque(maxlen=MAX_FINDING_BLOCKS)
        self._cost_usd = 0.0
        self._session_cost_hwm = 0.0
        self._last_usage: tuple[int, int, int] = (0, 0, 0)
        # Pending bump payloads, FIFO. A single slot meant the swarm's own
        # end-of-round bump silently overwrote guidance the coordinator or the
        # operator had queued mid-turn.
        self._bump_queue: deque[str] = deque(maxlen=8)
        self._concluding = False
        self._interrupted = False
        self._externally_cancelled = False
        # Set when this round ended because the conversation no longer fits. Read by
        # `_result()`; see `_classify_result` for why this backend needs a text-based
        # check rather than the exception-based one the Pydantic backend uses.
        self._context_overflow = False

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    @staticmethod
    def _join_blocks(blocks: Iterator[str] | list[str] | deque[str]) -> str:
        text = "\n\n".join(b for b in blocks if b and b.strip())
        if len(text) > MAX_FINDING_CHARS:
            text = text[-MAX_FINDING_CHARS:]
        return text.strip()

    @property
    def step_count(self) -> int:
        """Public, typed view of the step counter.

        Both backends expose this so consumers stop reading `_step_count` /
        `_tool_calls` directly: those are a plain int here but a one-element LIST on
        the Pydantic backend, and the raw value leaked into the archive and
        get_status. In the dashboard `!a.tool_calls` is false for `[0]`, so the
        "this agent ran zero tools" red flag never fired for a Pydantic agent.
        """
        return int(self._step_count)

    @property
    def tool_calls(self) -> int:
        """Public, typed view of the dispatched-tool counter. See `step_count`."""
        return int(self._tool_calls)

    @property
    def findings(self) -> str:
        """All text this solver produced in the current round."""
        return self._join_blocks(self._finding_blocks)

    # Back-compat: older call sites read `solver._findings`.
    @property
    def _findings(self) -> str:  # pragma: no cover - compatibility shim
        return self.findings

    def _add_finding(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._finding_blocks.append(text)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _build_options(self, system_prompt: str, allowed_tools: list[str]) -> ClaudeAgentOptions:
        from backend.models import claude_cli_env, effort_from_spec

        # Effort levels are a Claude-native feature  — gateway models (kimi,
        # deepseek, qwen, ...) may reject the parameter, so only pass it for
        # genuine Claude models.
        effort = effort_from_spec(self.model_spec) if self.model_id.startswith("claude") else None

        return ClaudeAgentOptions(
            model=self.model_id,
            system_prompt=system_prompt,
            effort=effort,
            env=claude_cli_env(self.settings),
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            output_format={"type": "json_schema", "schema": solver_output_json_schema()},
            # Resume the existing session when we have one — this is what makes
            # the phase-2 conclude harvest possible after an interrupt.
            resume=self._session_id,
            hooks={
                "PreToolUse": [HookMatcher(hooks=[self._pre_tool_use])],
                "PostToolUse": [HookMatcher(hooks=[self._post_tool_use])],
            },
        )

    async def start(self) -> None:
        if self._client:
            return  # idempotent  — _drop_client() forces a rebuild
        if not self.sandbox.is_running:
            await self.sandbox.start()

        self._container_id = self.sandbox.container_id

        arch_result = await self.sandbox.exec("uname -m", timeout_s=10)
        container_arch = arch_result.stdout.strip() or "unknown"

        distfile_names = list_distfiles(self.challenge_dir)
        sandbox_preamble = (
            "IMPORTANT: You are running inside a Docker sandbox. "
            "All files are under /challenge/  — distfiles at /challenge/distfiles/, "
            "workspace at /challenge/workspace/. Do NOT use any paths outside /challenge/. "
            "All bash commands run inside the container via docker exec. "
            "Use bash for everything: cat/head to read files, tee/echo> to write, find/grep to search. "
            "submit_flag 'FLAG' to submit. notify_coordinator 'MSG' to message the coordinator.\n\n"
        )
        if self.role_prompt:
            # A role prompt (the Verifier) still gets the challenge context and the
            # sandbox preamble — it has to know the target and where it is running —
            # but not the "find the flag" instructions, which are not its job.
            system_prompt = sandbox_preamble + build_prompt(
                self.meta, distfile_names, container_arch=container_arch,
                has_named_tools=False,
                skills_enabled=bool(str_setting(self.settings, "skills_dir")),
            ) + "\n\n" + self.role_prompt
        else:
            system_prompt = sandbox_preamble + build_prompt(
                self.meta, distfile_names, container_arch=container_arch,
                has_named_tools=False,
                skills_enabled=bool(str_setting(self.settings, "skills_dir")),
            )

        self._client = ClaudeSDKClient(
            options=self._build_options(system_prompt, ["Bash", "WebFetch", "WebSearch"])
        )
        await self._client.__aenter__()
        self.tracer.event("start", challenge=self.meta.name, model=self.model_id,
                          resumed=bool(self._session_id))
        logger.info(f"[{self.agent_name}] Claude SDK solver started")

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    async def _pre_tool_use(self, input_data, tool_use_id, context):
        try:
            return await self._sandbox_redirect_inner(input_data, tool_use_id)
        except Exception as e:  # noqa: BLE001 — a broken hook must not kill the turn
            logger.warning(f"[{self.agent_name}] PreToolUse hook error: {e}")
            return {}

    async def _post_tool_use(self, input_data, tool_use_id, context):
        try:
            return await self._trace_post_tool_inner(input_data, tool_use_id)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.agent_name}] PostToolUse hook error: {e}")
            return {}

    @staticmethod
    def _deny(tool_name: str, reason: str, system_message: str = "") -> dict:
        # Name the blocked tool in the reason. It was accepted and then ignored,
        # so a loop-break denial read as a generic "try a different approach"
        # with no indication of which call had been rejected.
        out: dict = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"[{tool_name}] {reason}" if tool_name else reason,
            }
        }
        if system_message:
            out["systemMessage"] = system_message
        return out

    @staticmethod
    def _allow(tool_input: dict, command: str | None = None,
               system_message: str = "") -> dict:
        updated = dict(tool_input)
        if command is not None:
            updated["command"] = command
        out: dict = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": updated,
            }
        }
        if system_message:
            out["systemMessage"] = system_message
        return out

    def _wrap_for_sandbox(self, command: str) -> str:
        """Rewrite a Bash command to run inside the sandbox container.

        Every command is wrapped in coreutils `timeout` so it cannot outlive the
        round. When the wall-clock budget interrupts a turn the CLI is torn
        down, but a process it started inside the container would otherwise keep
        burning that container's CPU/memory quota and compete with the next
        round. `-k 5` sends TERM then KILL 5s later — the same shape as Cairn's
        `timeout -k 5s <n>s` wrapper.
        """
        limit = int_setting(self.settings, "sandbox_command_timeout_s")
        inner = command
        if limit > 0:
            inner = f"timeout -k 5 {limit} bash -c {shlex.quote(command)}"
        return f"docker exec -i {self._container_id} bash -c {shlex.quote(inner)}"

    async def _sandbox_redirect_inner(self, input_data, tool_use_id: str = "") -> dict:
        if input_data.get("hook_event_name") != "PreToolUse":
            return {}

        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", {}) or {}
        call_id = str(tool_use_id or input_data.get("tool_use_id") or "")

        # Phase 2 (conclude) is text-only by construction: deny every tool so a
        # model that ignores the prompt still cannot start new work. The wording
        # points at whatever this instance was asked to produce, because the
        # Verifier concludes with a VERDICT line rather than a solver state card.
        if self._concluding:
            return self._deny(
                tool_name,
                "Conclude phase — no tool calls are allowed. Reply with the exact "
                "plain-text answer your last instruction asked for, and nothing else.",
            )

        self._step_count += 1
        self.tracer.tool_call(tool_name, tool_input, self._step_count)
        loop_status = self.loop_detector.begin(tool_name, tool_input, call_id=call_id)
        if loop_status == "break":
            from backend.loop_detect import LOOP_BLOCKED_MESSAGE

            self.tracer.event("loop_break", tool=tool_name, step=self._step_count)
            self.loop_detector.cancel(call_id)
            return self._deny(tool_name, LOOP_BLOCKED_MESSAGE)
        warn_msg = ""
        if loop_status == "warn":
            from backend.loop_detect import LOOP_WARNING_MESSAGE

            warn_msg = LOOP_WARNING_MESSAGE

        if tool_name == "Bash":
            command = (tool_input.get("command") or "").strip()

            flag_match = re.match(r"submit_flag\s+['\"]?(.+?)['\"]?\s*$", command)
            if flag_match:
                return await self._handle_submit(flag_match.group(1).strip(), tool_input)

            notify_match = re.match(r"notify_coordinator\s+['\"]?(.+?)['\"]?\s*$", command)
            if notify_match:
                if not self.notify_coordinator:
                    return self._allow(tool_input, "echo 'No coordinator connected.'")
                await self.notify_coordinator(notify_match.group(1).strip())
                return self._allow(tool_input, "echo 'Message sent to coordinator.'")

            help_match = re.match(r"request_help\s+['\"]?(.+?)['\"]?\s*$", command)
            if help_match:
                from backend.tools.core import do_request_help

                result_msg = await do_request_help(
                    self.message_bus, self.model_spec, help_match.group(1).strip(),
                    notify_coordinator=self.notify_coordinator,
                )
                return self._allow(tool_input, f"echo {shlex.quote(result_msg)}")

            # A real command is about to run in the sandbox — count it. Denied
            # and intercepted pseudo-commands deliberately do NOT count, so
            # `tool_calls == 0` reliably means "this solver never did anything".
            self._tool_calls += 1
            return self._allow(tool_input, self._wrap_for_sandbox(command), warn_msg)

        if tool_name in ("WebFetch", "WebSearch"):
            self._tool_calls += 1
            return {"systemMessage": warn_msg} if warn_msg else {}

        # StructuredOutput is the SDK's answer-submission channel — safe and
        # required for flag reporting; allow it (no filesystem side effects).
        if tool_name == "StructuredOutput":
            self.loop_detector.cancel(call_id)
            return {}

        # Everything else is denied  — Glob/Grep/Read/Write/Edit/Agent/etc.
        # would run on the host filesystem, breaking sandbox isolation.
        if tool_name in ("Glob", "Grep"):
            hint = " Use `find` or `grep` via bash instead  — those run in the container."
        elif tool_name in ("Read", "Write", "Edit", "NotebookEdit"):
            hint = " Use cat/head/tail to read, and tee/cat>file to write via bash."
        elif tool_name in ("Agent", "Task", "TaskCreate", "TaskStop"):
            hint = (" Sub-agents are not available here — do the work directly "
                    "with your own bash commands instead of spawning a sub-agent.")
        else:
            hint = ""
        self.loop_detector.cancel(call_id)
        return self._deny(
            tool_name,
            f"{tool_name} blocked  — use bash for all operations inside the sandbox.",
            f"{tool_name} is not available  — all work happens inside the Docker "
            f"container.{hint}" if hint else "",
        )

    async def _handle_submit(self, flag_val: str, tool_input: dict) -> dict:
        """Verify a candidate flag through the swarm (dedup/cooldown/answer key).

        A flag is only ever marked confirmed by the verifier — never by the
        model asserting one. Claiming FLAG_FOUND from an unverified structured
        output used to cancel every sibling solver on a hallucination.
        """
        display, confirmed = await self._verify_flag(flag_val)
        if confirmed:
            self.tracer.event("flag_confirmed", flag=flag_val, step=self._step_count)
        else:
            self.tracer.event("flag_rejected", flag=flag_val, step=self._step_count,
                              detail=display[:200])
        return self._allow(tool_input, f"echo {shlex.quote(display)}")

    async def _verify_flag(self, candidate: str, method: str = "") -> tuple[str, bool]:
        """Run one candidate through the swarm verifier and record the outcome."""
        if self.submit_fn:
            display, confirmed = await self.submit_fn(candidate)
        elif self.no_submit:
            display, confirmed = f'DRY RUN  — would submit "{candidate}"', False
        else:
            from backend.tools.core import FlagSubmissionError, do_submit_flag

            try:
                display, confirmed = await do_submit_flag(
                    self.ctfd, self.meta.name, candidate)
            except FlagSubmissionError as e:
                # Never judged, so never "REJECTED" — recording it that way would send
                # the model off to fix a flag that was correct as far as anyone knows.
                display = (
                    f"SUBMISSION FAILED — CTFd did not answer ({e}). The flag was NOT "
                    "judged and this is not a wrong attempt."
                )
                self._add_finding(display)
                self.tracer.event("submit_undelivered", flag=candidate[:200],
                                  detail=str(e)[:200])
                return display, False
        if confirmed:
            self._confirmed = True
            self._flag = candidate
            self._add_finding(f"Flag found via {method or '?'}: {candidate}")
        else:
            self._add_finding(f"Claimed flag {candidate} was REJECTED: {display}")
        return display, confirmed

    async def _trace_post_tool_inner(self, input_data, tool_use_id: str = "") -> dict:
        if input_data.get("hook_event_name") != "PostToolUse":
            return {}
        response_str = str(input_data.get("tool_response", ""))
        self.tracer.tool_result(input_data.get("tool_name", "?"), response_str[:500],
                                self._step_count)
        # Feed the result back to the loop detector: a repeat only counts as a
        # loop when the SAME command produced the SAME output.
        call_id = str(tool_use_id or input_data.get("tool_use_id") or "")
        self.loop_detector.finish(call_id=call_id, result=response_str)
        if self._concluding or not self.message_bus:
            return {}

        # Both channels are collected and returned as ONE additionalContext. They used
        # to early-return, so on any step divisible by both 3 and 5 the advice branch
        # won and that round's blackboard findings were never checked — the cursor is
        # only advanced by `check()`, so they were not lost, just deferred to the next
        # multiple-of-5 step. A prune notice or a refutation arriving on a %15 step is
        # exactly the message that must not wait.
        parts: list[str] = []

        # Fast mailbox: coordinator advice picked up mid-run (every 3 steps)
        if self._step_count % 3 == 0:
            advice = await self.message_bus.check_advice(self.model_spec)
            if advice:
                self.tracer.event("advice_picked", count=len(advice))
                parts.append(
                    "📨 **Guidance from your coordinator (read now, it overrides "
                    "your current plan if relevant):**\n\n" + "\n\n".join(advice)
                )

        if self._step_count % 5 == 0:
            from backend.tools.core import do_check_findings

            findings = await do_check_findings(self.message_bus, self.model_spec)
            if findings and "No new findings" not in findings:
                self.tracer.event("findings_injected", step=self._step_count)
                parts.append(findings)

        if not parts:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "\n\n".join(parts),
            }
        }

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def _next_prompt(self) -> str:
        if self._bump_queue:
            insights = self._bump_queue.popleft()
            return (
                "Your previous attempt did not find the flag. First output a "
                "3-line tagged summary: [excluded] dead ends ruled out, "
                "[fact] confirmed findings, [next] your new hypothesis. "
                "Then continue with a DIFFERENT approach. "
                f"Insights from other agents:\n\n{insights}\n\n"
                "Do NOT repeat what was tried."
            )
        if self._session_id:
            return "Continue solving. Try a different approach."
        return "Solve this CTF challenge."

    def _record_usage(self, message: ResultMessage, t0: float) -> None:
        """Accumulate per-turn cost + tokens.

        `total_cost_usd` is a session high-water mark in current CLI versions
        but was per-turn in older ones, so it is handled adaptively: a value at
        or above the mark is a new cumulative total (charge the delta); a value
        below it means the field is per-turn (charge it directly). Either way
        the running sum is correct and nothing is double-counted.
        """
        reported = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
        if reported >= self._session_cost_hwm:
            turn_cost = reported - self._session_cost_hwm
            self._session_cost_hwm = reported
        else:
            turn_cost = reported
        turn_cost = max(0.0, turn_cost)
        self._cost_usd += turn_cost

        usage = getattr(message, "usage", None) or {}
        if not isinstance(usage, dict):
            usage = vars(usage) if hasattr(usage, "__dict__") else {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        cache_tok = int(
            usage.get("cache_read_input_tokens", usage.get("cache_read_tokens", 0)) or 0
        )
        if not (in_tok or out_tok):
            # Fall back to the per-model breakdown when the CLI omits `usage`.
            for entry in (getattr(message, "model_usage", None) or {}).values():
                in_tok += int(entry.get("inputTokens", 0) or 0)
                out_tok += int(entry.get("outputTokens", 0) or 0)
                cache_tok += int(entry.get("cacheReadInputTokens", 0) or 0)
        self._last_usage = (in_tok, out_tok, cache_tok)
        self.cost_tracker.record_tokens(
            self.agent_name, self.model_id,
            input_tokens=in_tok, output_tokens=out_tok, cache_read_tokens=cache_tok,
            provider_spec="claude-sdk",
            duration_seconds=time.monotonic() - t0,
            cost_usd=turn_cost,
        )

    async def _consume_stream(
        self, deadline: float | None, phase: str, sink: list[str]
    ) -> ResultMessage | None:
        """Read one turn's messages until its ResultMessage.

        The SDK's async generator is drained by a dedicated pump task into a
        queue, so the deadline/cancel polling only ever cancels `Queue.get()` —
        never the generator itself (cancelling an async generator mid-yield can
        corrupt its state).

        On `deadline` expiry the turn is INTERRUPTED through the SDK rather than
        task-cancelled, so the session survives and phase 2 can resume it.
        """
        assert self._client is not None
        # Bind the client to a local: the pump closure outlives this frame's
        # narrowing, and re-reading the Optional attribute inside it would
        # re-introduce the None case.
        client = self._client
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        result_msg: ResultMessage | None = None
        interrupted_at: float | None = None
        abandoned = False

        async def _pump() -> None:
            try:
                async for msg in client.receive_response():
                    await queue.put(msg)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — surface to the consumer
                await queue.put(e)
            finally:
                await queue.put(_STREAM_END)

        pump = asyncio.create_task(_pump(), name=f"{self.agent_name}-stream-{phase}")
        try:
            while True:
                if deadline is not None and not self._interrupted and loop.time() >= deadline:
                    self._interrupted = True
                    interrupted_at = loop.time()
                    self.tracer.event("budget_interrupt", phase=phase)
                    logger.warning(f"[{self.agent_name}] Time budget hit — interrupting turn")
                    await self._try_interrupt()

                if self.cancel_event.is_set() and not self._interrupted:
                    # External cancel (sibling won / operator kill): interrupt so
                    # the CLI stops cleanly instead of us walking away mid-turn.
                    self._interrupted = True
                    self._externally_cancelled = True
                    interrupted_at = loop.time()
                    await self._try_interrupt()

                wait_s = STREAM_POLL_S
                if self._interrupted and interrupted_at is not None:
                    remaining = INTERRUPT_GRACE_S - (loop.time() - interrupted_at)
                    if remaining <= 0:
                        abandoned = True
                        self.tracer.event("interrupt_grace_expired", phase=phase)
                        break
                    wait_s = min(wait_s, remaining)

                try:
                    item = await asyncio.wait_for(queue.get(), timeout=wait_s)
                except TimeoutError:
                    continue

                if item is _STREAM_END:
                    break
                if isinstance(item, BaseException):
                    raise item

                if isinstance(item, AssistantMessage):
                    for block in item.content:
                        if isinstance(block, TextBlock):
                            sink.append(block.text)
                            self.tracer.model_response(block.text[:800], self._step_count)
                        elif isinstance(block, ThinkingBlock):
                            # Reasoning models surface their plan as a
                            # ThinkingBlock — capture it for the dashboard's
                            # live "思路" view.
                            self.tracer.model_response(
                                f"[思考] {block.thinking[:800]}", self._step_count
                            )
                elif isinstance(item, ResultMessage):
                    result_msg = item
                    break
        finally:
            if not pump.done():
                pump.cancel()
                with contextlib.suppress(BaseException):
                    await pump
            if abandoned:
                # The stream was walked away from; its state is unknown, so the
                # session cannot be trusted for a phase-2 resume.
                self._session_id = None

        return result_msg

    async def _try_interrupt(self) -> None:
        if not self._client:
            return
        try:
            await self._client.interrupt()
        except Exception as e:  # noqa: BLE001 — fall back to abandoning the stream
            logger.warning(f"[{self.agent_name}] interrupt() failed: {e}")
            self.tracer.event("interrupt_failed", error=str(e)[:200])

    async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
        if not self._client:
            await self.start()
        assert self._client is not None

        t0 = time.monotonic()
        cost_before = self._cost_usd
        steps_before = self._step_count
        calls_before = self._tool_calls
        self._interrupted = False
        self._externally_cancelled = False
        self._context_overflow = False
        self._last_usage = (0, 0, 0)
        self._finding_blocks.clear()
        deadline = (asyncio.get_running_loop().time() + budget_s) if budget_s > 0 else None
        deltas = {"steps": steps_before, "calls": calls_before, "cost": cost_before}

        try:
            await self._client.query(self._next_prompt())
            sink: list[str] = []
            message = await self._consume_stream(deadline, "execute", sink)
            for block in sink:
                self._add_finding(block)

            api_error = ""
            status_override: str | None = None
            claimed: tuple[str, str] | None = None
            if message is not None:
                self._session_id = message.session_id or self._session_id
                self._record_usage(message, t0)
                api_error, status_override = self._classify_result(message)
                claimed = self._handle_structured_output(message)

            # A flag asserted through StructuredOutput is verified here, through
            # the same swarm path submit_flag uses — never trusted on assertion.
            if status_override is None and claimed is not None:
                candidate, method = claimed
                self.tracer.event("structured_flag", flag=candidate[:200],
                                  method=method[:200])
                await self._verify_flag(candidate, method)

            self.tracer.event(
                "turn_complete",
                duration=round(time.monotonic() - t0, 1),
                cost=round(self._cost_usd, 4),
                interrupted=self._interrupted,
                tool_calls=self._tool_calls - calls_before,
            )

            if self._externally_cancelled:
                return self._result(CANCELLED, deltas, api_error=api_error)
            if status_override:
                return self._result(status_override, deltas, api_error=api_error)
            if self._confirmed and self._flag:
                return self._result(FLAG_FOUND, deltas)
            return self._result(GAVE_UP, deltas)

        except asyncio.CancelledError:
            # Correct cooperative-cancel semantics: record, then propagate. The
            # swarm owns the task boundary and converts this into CANCELLED.
            self.tracer.event("cancelled", step_count=self._step_count)
            raise
        except Exception as e:  # noqa: BLE001
            error_str = str(e)
            # Mirror the Pydantic backend's ordering: cancel, then overflow, then
            # exception-class classification, then text classification. This path
            # used to go straight to `classify_api_error(error_str) or ERROR`, so an
            # SDK exception whose CLASS name identifies it (`APITimeoutError`,
            # `APIConnectionError`) but whose `str()` does not match the conservative
            # text regexes was demoted to a plain ERROR — feeding the consecutive-
            # error counter instead of the infra guard's backoff semantics.
            if is_context_overflow(e):
                logger.warning(f"[{self.agent_name}] Context overflow — dropping session")
                self.tracer.event("context_overflow_reset", error=error_str[:200])
                self._session_id = None
                self._context_overflow = True
                return self._result(ERROR, deltas, api_error=error_str[:300])
            logger.error(f"[{self.agent_name}] Error: {e}", exc_info=True)
            self._add_finding(f"Error: {e}")
            self.tracer.event("error", error=error_str)
            kind = classify_exception(e) or classify_api_error(error_str) or ERROR
            return self._result(kind, deltas, api_error=error_str)

    def _classify_result(self, message: ResultMessage) -> tuple[str, str | None]:
        """Detect API/infrastructure failures reported inside a ResultMessage.

        The CLI does not raise on a failed API call — it emits an assistant text
        block ("API Error: Connection refused ...") and/or a ResultMessage with
        `is_error` / `api_error_status` set. Treating that as an ordinary GAVE_UP
        is what let a dead gateway masquerade as a hard challenge for hours.
        """
        # An interrupted turn is a budget expiry or an external cancel — not a
        # failure, and never a reason to pollute the blackboard.
        if self._interrupted:
            return "", None

        api_status = getattr(message, "api_error_status", None)
        subtype = getattr(message, "subtype", "") or ""
        terminal = getattr(message, "terminal_reason", "") or ""
        errors = list(getattr(message, "errors", None) or [])
        is_error = bool(getattr(message, "is_error", False))
        detail = "; ".join(str(e) for e in errors)[:400]

        if terminal in ("aborted_streaming", "aborted_tools"):
            return "", None

        from_status = classify_api_status(api_status)
        text = self.findings
        # Context overflow must be recognised BEFORE the generic classification. The
        # CLI reports it as ordinary assistant text ("API Error: prompt is too long:
        # 210000 tokens > 200000 maximum"), which `_API_ERROR_RE` matches, and it
        # carries no HTTP status and no quota keyword — so it fell all the way
        # through to `return INFRA_ERROR`. On the default backend that meant a real
        # overflow incremented `_infra_failures` (three retire the swarm, blaming the
        # gateway) and `consecutive_errors`, while `context_overflow` stayed False so
        # the overflow coaching and the forced clean restart never ran.
        if is_context_overflow_text(text) or is_context_overflow_text(detail):
            self._context_overflow = True
            # Recoverable only by dropping the conversation, so drop it now — the
            # next round must not fail identically forever.
            self._session_id = None
            return (detail or text[:300] or "context window exceeded"), None
        if from_status:
            return detail or f"api_error_status={api_status}", from_status

        if is_error or subtype.startswith("error"):
            kind = classify_api_error(text) or classify_api_error(detail)
            if kind:
                return detail or text[:400], kind
            if subtype == "error_max_turns" or terminal == "max_turns":
                # Deliberately NOT reported as api_error. A non-empty api_error makes
                # `is_infra_failure` true, so `_account_infra` rewrote the status to
                # INFRA_ERROR and counted the round toward retiring the swarm — a
                # productive round that ran out of turns was booked as a dead
                # gateway, and three of them retired a healthy swarm telling the
                # operator to check ANTHROPIC_BASE_URL. Hitting the turn ceiling is
                # an ordinary GAVE_UP: the model worked, the round ended.
                self.tracer.event("max_turns_reached", steps=self._step_count)
                return "", None
            return detail or f"turn failed ({subtype or 'unknown'})", INFRA_ERROR

        # No error flag, but the text itself may be a bare API error message —
        # only believable when the solver never managed a single tool call.
        kind = classify_api_error(text)
        if kind and self._tool_calls == 0:
            return text[:400], kind
        return "", None

    def _handle_structured_output(self, message: ResultMessage) -> tuple[str, str] | None:
        """Interpret the turn's structured output.

        Returns a (flag, method) pair to be VERIFIED by the caller — never a
        confirmation. `still_working` exists so a model can end a turn honestly:
        when `flag_found` was the only allowed output, an unsolved turn still had
        to produce one, so the model invented a flag, and that hallucination
        used to cancel every sibling solver.
        """
        output = getattr(message, "structured_output", None)
        if not isinstance(output, dict):
            return None
        kind = output.get("type")
        if kind == "flag_found":
            candidate = (output.get("flag") or "").strip()
            if not candidate:
                self.tracer.event("structured_flag_empty")
                return None
            return candidate, str(output.get("method", ""))
        if kind == "still_working":
            note = "\n".join(
                p for p in (
                    f"[fact] {output['confirmed']}" if output.get("confirmed") else "",
                    f"[excluded] {output['excluded']}" if output.get("excluded") else "",
                    f"[next] {output['next_step']}" if output.get("next_step") else "",
                ) if p
            )
            if note:
                self._add_finding(note)
            self.tracer.event("still_working", step=self._step_count)
        return None

    async def conclude(self, budget_s: int) -> str:
        """Phase 2: distil the interrupted session into a state card.

        Requires session continuity — with no session id there is nothing to
        resume, so "" is returned and the caller writes nothing. Any failure
        (timeout, transport error, empty answer, API error) also returns "": a
        partial or invented harvest is worse than none, because siblings treat
        harvested [fact] lines as ground truth.
        """
        if budget_s <= 0 or not self._session_id or not self._client:
            self.tracer.event("conclude_skipped", budget_s=budget_s,
                              has_session=bool(self._session_id),
                              has_client=bool(self._client))
            return ""
        self._concluding = True
        self._interrupted = False
        t0 = time.monotonic()
        sink: list[str] = []
        self.tracer.event("conclude_start", budget_s=budget_s, session=self._session_id)
        try:
            await self._client.query(self.conclude_prompt or CONCLUDE_PROMPT)
            deadline = asyncio.get_running_loop().time() + budget_s
            message = await self._consume_stream(deadline, "conclude", sink)
            if message is None:
                self.tracer.event("conclude_no_result")
                return ""
            self._session_id = message.session_id or self._session_id
            self._record_usage(message, t0)
            text = self._join_blocks(sink)
            if not text:
                self.tracer.event("conclude_empty")
                return ""
            if classify_api_error(text):
                self.tracer.event("conclude_api_error", detail=text[:200])
                return ""
            self._add_finding(text)
            self.tracer.event("conclude_ok", chars=len(text),
                              duration=round(time.monotonic() - t0, 1))
            return text
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.agent_name}] conclude failed: {e}")
            self.tracer.event("conclude_failed", error=str(e)[:300])
            return ""
        finally:
            self._concluding = False
            self._interrupted = False

    async def reset_session(self) -> None:
        """Drop the client but KEEP the session id, so the next start() resumes.

        Use after an interrupt whose stream had to be abandoned, or whenever the
        in-process client is in an unknown state but the conversation is worth
        keeping.
        """
        await self._drop_client()
        self.tracer.event("session_reset", keep_session=bool(self._session_id))

    async def forget_session(self) -> None:
        """Hard reset: drop the client AND the session id (clean context).

        Used by the fresh-restart retry path, where the whole point is to stop
        carrying an ever-growing conversation.
        """
        await self._drop_client()
        self._session_id = None
        self._session_cost_hwm = 0.0
        self.tracer.event("session_forgotten")

    async def _drop_client(self) -> None:
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"[{self.agent_name}] client teardown: {e}")
            self._client = None

    def bump(self, insights: str) -> None:
        """Queue guidance for the next turn. FIFO — nothing is overwritten."""
        insights = (insights or "").strip()
        if not insights:
            return
        self._bump_queue.append(insights)
        self.loop_detector.reset()
        # Payloads are multi-KB now (distilled journal + team digest + skill
        # hint), so a 500-char window hid everything past the journal's first
        # line and made the restart content unauditable from the trace. Record a
        # usable window plus the true size so truncation is visible.
        self.tracer.event("bump", insights=insights[:BUMP_TRACE_CHARS],
                          insights_chars=len(insights), queued=len(self._bump_queue))
        logger.info(f"[{self.agent_name}] Bump queued ({len(self._bump_queue)} pending)")

    @property
    def pending_bumps(self) -> int:
        return len(self._bump_queue)

    def _result(self, status: str, deltas: dict, api_error: str = "") -> SolverResult:
        in_tok, out_tok, cache_tok = self._last_usage
        self.tracer.event("finish", status=status, flag=self._flag if self._confirmed else None,
                          confirmed=self._confirmed, cost_usd=round(self._cost_usd, 4),
                          interrupted=self._interrupted, api_error=api_error[:200])
        return SolverResult(
            # Only ever report a flag the verifier actually accepted.
            flag=self._flag if self._confirmed else None,
            status=status,
            findings_summary=self.findings,
            step_count=self._step_count - deltas.get("steps", 0),
            cost_usd=self._cost_usd - deltas.get("cost", 0.0),
            log_path=self.tracer.path,
            tool_calls=self._tool_calls - deltas.get("calls", 0),
            api_error=api_error,
            interrupted=self._interrupted,
            context_overflow=self._context_overflow,
            input_tokens=in_tok, output_tokens=out_tok, cache_read_tokens=cache_tok,
        )

    def detach_sandbox(self) -> object:
        """Hand sandbox ownership back to the caller so stop() leaves it alive.

        Ownership is a flag rather than `sandbox = None`: keeping the attribute
        non-Optional means every `self.sandbox.<x>` use stays type-safe, and a
        detached solver still reports which container it was using.
        """
        self._owns_sandbox = False
        return self.sandbox

    async def stop(self) -> None:
        self.tracer.event("stop", step_count=self._step_count, tool_calls=self._tool_calls)
        self.tracer.close()
        # Sandbox FIRST, client second. `_drop_client` catches Exception but not
        # CancelledError, and the SDK client's `__aexit__` awaits — so a second cancel
        # landing there (asyncio.run's shutdown, a bench time-limit, `_wind_down`
        # cancelling a straggler) propagated out before the container was destroyed.
        # `DockerSandbox.stop()` shields its own teardown, so once we reach it the
        # cleanup completes; the exposure was purely the await in front of it. The
        # Pydantic backend already had this order.
        if self._owns_sandbox and self.sandbox:
            await self.sandbox.stop()
        await self._drop_client()
