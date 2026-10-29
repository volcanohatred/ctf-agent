"""Per-model solver agent  — one model, one container, one challenge."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.wrapper import WrapperToolset

from backend.config import str_setting
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.deps import SolverDeps
from backend.loop_detect import LOOP_BLOCKED_MESSAGE, LOOP_WARNING_MESSAGE, LoopDetector
from backend.models import (
    context_window,
    model_id_from_spec,
    provider_from_spec,
    resolve_model,
    resolve_model_settings,
    supports_vision,
)
from backend.output_types import FlagFound, SolverOutput, StillWorking
from backend.prompts import CONCLUDE_PROMPT, ChallengeMeta, build_prompt, list_distfiles
from backend.sandbox import DockerSandbox, sandbox_from_settings
from backend.solver_base import (
    CANCELLED,
    CORRECT_MARKERS,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    SolverResult,
    classify_api_error,
    classify_exception,
    is_context_overflow,
)
from backend.tools.flag import submit_flag
from backend.tools.sandbox import (
    bash,
    check_findings,
    list_files,
    notify_coordinator,
    read_file,
    request_help,
    web_fetch,
    webhook_create,
    webhook_get_requests,
    write_file,
)
from backend.tools.vision import view_image
from backend.tracing import SolverTracer

if TYPE_CHECKING:
    from backend.config import Settings

logger = logging.getLogger(__name__)

MAX_FINDING_BLOCKS = 24
MAX_FINDING_CHARS = 24_000

# Tools that talk to the swarm or the operator rather than to the challenge. They
# must not count toward `tool_calls`, because `did_work` (tool_calls > 0) is what
# `_account_infra` uses to decide a round was a broken endpoint rather than a
# legitimate give-up — and a round whose only activity was submitting flags and
# reading the board never touched the target.
#
# The Claude backend already behaved this way by accident: it intercepts
# submit_flag / notify_coordinator / request_help as pseudo-commands and returns
# before incrementing. This backend counted every non-loop-broken call, so the SAME
# round was `did_work=False` on one backend and True on the other — published and
# exempt from the infra guard on one, rewritten to INFRA_ERROR and counted toward
# retiring the swarm on the other.
BOOKKEEPING_TOOLS = frozenset({
    "submit_flag", "check_findings", "notify_coordinator", "request_help",
})
# Restart payloads carry the distilled journal, the team digest, an escalating
# skill hint (whose L3 form embeds a whole pack index) and the next untried
# technique. Measured live at 7.8-10.5 KB, so the window must hold a real payload
# or the tail — where the team digest sits — is what gets lost.
BUMP_TRACE_CHARS = 16_000


@dataclass
class TracingToolset(WrapperToolset[SolverDeps]):
    """Wraps a toolset to add per-call tracing, loop detection and mailbox delivery."""

    tracer: SolverTracer = field(repr=False)
    loop_detector: LoopDetector = field(repr=False)
    step_counter: list[int] = field(repr=False)
    # Real dispatched calls (loop-broken calls excluded). The swarm uses this to
    # tell "the solver did nothing" apart from "the solver tried and failed".
    tool_counter: list[int] = field(repr=False)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[SolverDeps], tool: ToolsetTool[SolverDeps]
    ) -> Any:
        self.step_counter[0] += 1
        step = self.step_counter[0]

        self.tracer.tool_call(name, tool_args, step)

        # Loop detection: decided before the call, refined with its result below.
        call_id = f"s{step}"
        loop_status = self.loop_detector.begin(name, tool_args, call_id=call_id)
        if loop_status == "break":
            logger.warning(f"Loop break on {name} at step {step}")
            self.tracer.event("loop_break", tool=name, step=step)
            self.loop_detector.cancel(call_id)
            # Inject loop warning by returning it as the tool result
            return LOOP_BLOCKED_MESSAGE

        self.tool_counter[0] += 1 if name not in BOOKKEEPING_TOOLS else 0
        result = await self.wrapped.call_tool(name, tool_args, ctx, tool)

        result_str = str(result) if result is not None else ""
        self.tracer.tool_result(name, result_str, step)
        # A repeat only counts as a loop when the SAME call produced the SAME
        # output, so legitimately repeated commands are never blocked.
        self.loop_detector.finish(call_id=call_id, result=result_str)

        # Inject loop warning alongside result on "warn" level
        if loop_status == "warn" and isinstance(result, str):
            result = f"{result}\n\n{LOOP_WARNING_MESSAGE}"

        # Check for confirmed flag (avoid matching "INCORRECT"  — CORRECT substring)
        if name == "submit_flag" and "INCORRECT" not in result_str and any(m in result_str for m in CORRECT_MARKERS):
            self.tracer.event("flag_confirmed", tool=name, step=step)

        # Mailbox injection only happens when the result can actually carry
        # text. Draining first and checking afterwards (as an earlier revision
        # did) silently DISCARDED coordinator advice and sibling findings
        # whenever a tool returned a non-string payload such as an image.
        if isinstance(result, str) and ctx.deps.message_bus:
            if step % 3 == 0:
                advice = await ctx.deps.message_bus.check_advice(ctx.deps.model_spec)
                if advice:
                    self.tracer.event("advice_picked", count=len(advice))
                    result += (
                        "\n\n---\n📨 **Guidance from your coordinator (read now, it "
                        "overrides your current plan if relevant):**\n\n"
                        + "\n\n".join(advice)
                    )

            if step % 5 == 0:
                from backend.tools.core import do_check_findings
                findings_text = await do_check_findings(ctx.deps.message_bus, ctx.deps.model_spec)
                if findings_text and "No new findings" not in findings_text:
                    result = f"{result}\n\n---\n{findings_text}"
                    self.tracer.event("findings_injected", step=step)

        return result


def _build_toolset(deps: SolverDeps) -> FunctionToolset[SolverDeps]:
    """Build the raw toolset for a solver agent."""
    tools = [bash, read_file, write_file, list_files, submit_flag, web_fetch,
             webhook_create, webhook_get_requests, check_findings, notify_coordinator,
             request_help]
    if deps.use_vision:
        tools.append(view_image)
    return FunctionToolset(tools=tools, max_retries=4)


def _partial_messages(exc: BaseException) -> list:
    """Recover the interrupted round's message history from a RunTimedOut.

    pydantic-ai attaches `all_messages()` to the exception, which is the only way to
    reach the partial conversation — `Agent.run` never returned a result. Without it
    `_messages` keeps the PREVIOUS round's history, and the conclude phase then
    distils a state card about work that is not this round's.

    Returns [] when nothing is recoverable, which the caller must treat as "no
    harvest possible" rather than "harvest whatever is in _messages".
    """
    for accessor in ("all_messages", "new_messages"):
        fn = getattr(exc, accessor, None)
        if fn is None:
            continue
        try:
            messages = list(fn())
        except Exception:  # noqa: BLE001 — a broken accessor must not mask the timeout
            continue
        if messages:
            return messages
    return []


class Solver:
    """A single solver: one model, one container, one challenge."""

    def __init__(
        self,
        model_spec: str,
        challenge_dir: str,
        meta: ChallengeMeta,
        ctfd: CTFdClient,
        cost_tracker: CostTracker,
        settings: Settings,
        cancel_event: asyncio.Event | None = None,
        sandbox: DockerSandbox | None = None,
        owns_sandbox: bool | None = None,
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
        self._owns_sandbox = owns_sandbox if owns_sandbox is not None else (sandbox is None)
        # Appended to the generated system prompt; see ClaudeSolver.role_prompt.
        self.role_prompt = (role_prompt or "").strip()
        # Replaces CONCLUDE_PROMPT for the tools-denied phase-2 harvest; see
        # ClaudeSolver.conclude_prompt.
        self.conclude_prompt = (conclude_prompt or "").strip()

        self.sandbox = sandbox or sandbox_from_settings(challenge_dir, settings)
        self.use_vision = supports_vision(model_spec)
        self.deps = SolverDeps(
            sandbox=self.sandbox,
            ctfd=ctfd,
            challenge_dir=challenge_dir,
            challenge_name=meta.name,
            workspace_dir="",
            use_vision=self.use_vision,
            cost_tracker=cost_tracker,
            web_fetch_allow_hosts=str_setting(settings, "web_fetch_allow_hosts"),
        )
        self.loop_detector = LoopDetector()
        self.tracer = SolverTracer(meta.name, self.model_id)
        self.agent_name = f"{meta.name}/{self.model_id}"
        self._agent: Agent[SolverDeps, SolverOutput] | None = None
        self._conclude_agent: Agent[SolverDeps, str] | None = None
        self._model = None
        self._model_settings = None
        self._system_prompt = ""
        self._messages: list = []
        # False when `_messages` does not describe the round that just ended — i.e. a
        # timeout whose partial history could not be recovered. `conclude()` must then
        # refuse rather than distil the PREVIOUS round (or, on a first-round timeout,
        # invent a state card from the system prompt alone), because the swarm
        # publishes that text to the blackboard as ground truth.
        self._messages_current = True
        self._step_count = [0]  # mutable ref shared with TracingToolset
        self._tool_calls = [0]  # mutable ref shared with TracingToolset
        self._flag: str | None = None
        self._confirmed: bool = False
        self._finding_blocks: deque[str] = deque(maxlen=MAX_FINDING_BLOCKS)
        self._bump_queue: deque[str] = deque(maxlen=8)
        self._last_input_tokens: int = 0  # context-size probe for compaction
        self._interrupted = False

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """Public, typed view of the step counter.

        `_step_count` is a one-element LIST on this backend because `TracingToolset`
        needs a mutable reference it shares with the agent, and a plain int on the
        Claude backend. Consumers that read the private attribute directly got `[N]`
        from one backend and `N` from the other, which leaked into the archive and
        get_status: in the dashboard `!a.tool_calls` is false for `[0]`, so the
        "this agent ran zero tools" red flag never fired for a Pydantic agent — the
        exact symptom that once let a benchmark grind through 40 empty rounds.
        """
        return int(self._step_count[0])

    @property
    def tool_calls(self) -> int:
        """Public, typed view of the dispatched-tool counter. See `step_count`."""
        return int(self._tool_calls[0])

    @property
    def findings(self) -> str:
        text = "\n\n".join(b for b in self._finding_blocks if b and b.strip())
        if len(text) > MAX_FINDING_CHARS:
            text = text[-MAX_FINDING_CHARS:]
        return text.strip()

    @property
    def _findings(self) -> str:  # pragma: no cover - compatibility shim
        return self.findings

    def _add_finding(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self._finding_blocks.append(text)

    @property
    def pending_bumps(self) -> int:
        return len(self._bump_queue)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the sandbox and build the agent."""
        if not self.sandbox.is_running:
            await self.sandbox.start()
        self.deps.workspace_dir = self.sandbox.workspace_dir

        arch_result = await self.sandbox.exec("uname -m", timeout_s=10)
        container_arch = arch_result.stdout.strip() or "unknown"

        distfile_names = list_distfiles(self.challenge_dir)
        system_prompt = build_prompt(
            self.meta,
            distfile_names,
            container_arch=container_arch,
            skills_enabled=bool(str_setting(self.settings, "skills_dir")),
        )
        if self.role_prompt:
            system_prompt = f"{system_prompt}\n\n{self.role_prompt}"

        model = resolve_model(self.model_spec, self.settings)
        model_settings = resolve_model_settings(self.model_spec)
        self._model = model
        self._model_settings = model_settings
        self._system_prompt = system_prompt
        raw_toolset = _build_toolset(self.deps)
        toolset = TracingToolset(
            wrapped=raw_toolset,
            tracer=self.tracer,
            loop_detector=self.loop_detector,
            step_counter=self._step_count,
            tool_counter=self._tool_calls,
        )

        self._agent = Agent(
            model,
            deps_type=SolverDeps,
            system_prompt=system_prompt,
            model_settings=model_settings,
            toolsets=[toolset],
            output_type=SolverOutput,
        )

        self.tracer.event("start", challenge=self.meta.name, model=self.model_id)
        logger.info(f"[{self.agent_name}] Solver started")

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------

    def _next_prompt(self) -> str:
        if self._bump_queue:
            insights = self._bump_queue.popleft()
            return (
                "Your previous attempt did not find the flag. First, output a "
                "3-line tagged summary so teammates can learn from it:\n"
                "[excluded] dead ends you ruled out (be specific)\n"
                "[fact] confirmed findings (files, behavior, credentials)\n"
                "[next] your new hypothesis\n"
                "Then continue with a DIFFERENT approach. Here are insights "
                "from other agents working on the same challenge:\n\n"
                f"{insights}\n\n"
                "Do NOT repeat what has already been tried."
            )
        return "Solve this CTF challenge." if not self._messages else "Continue solving."

    async def run_until_done_or_gave_up(self, budget_s: int = 0) -> SolverResult:
        """Run one solver turn, bounded by `budget_s` (0 = unbounded)."""
        if not self._agent:
            await self.start()
        assert self._agent is not None

        t0 = time.monotonic()
        steps_before = self._step_count[0]
        calls_before = self._tool_calls[0]
        cost_before = self._agent_cost()
        self._interrupted = False
        self._finding_blocks.clear()
        deltas = {"steps": steps_before, "calls": calls_before, "cost": cost_before}

        try:
            from pydantic_ai.usage import UsageLimits
            # `timeout` is pydantic-ai's own per-run wall clock: it raises
            # RunTimedOut and leaves the agent usable, which is what makes the
            # phase-2 conclude possible on this backend too.
            #
            # The run is raced against cancel_event rather than simply awaited.
            # `_wind_down`'s contract is that solvers watch the event and break out
            # of their stream loop within a couple of seconds, letting each run its
            # own container/workspace teardown uninterrupted — ClaudeSolver polls
            # every 2s and honours it. This backend awaited `agent.run` to
            # completion and only read the event afterwards, so on a flag-found or
            # a coordinator shutdown it ignored the cancellation for the rest of a
            # turn that can run to SOLVER_TIME_BUDGET_S (1500s), ate the whole 60s
            # grace, and was then hard-task-cancelled — which is the exact path
            # `_wind_down` exists to avoid, and it lost the round's findings.
            run_task = asyncio.create_task(self._agent.run(
                self._next_prompt(),
                deps=self.deps,
                message_history=self._messages if self._messages else None,
                usage_limits=UsageLimits(request_limit=None),
                timeout=float(budget_s) if budget_s > 0 else None,
            ))
            result = await self._await_run_or_cancel(run_task)
            if result is None:
                return self._result(CANCELLED, deltas)

            duration = time.monotonic() - t0
            usage = result.usage

            self.cost_tracker.record(
                self.agent_name, usage, self.model_id,
                provider_spec=provider_from_spec(self.model_spec),
                duration_seconds=duration,
            )

            agent_usage = self.cost_tracker.by_agent.get(self.agent_name)
            self.tracer.usage(
                usage.input_tokens, usage.output_tokens,
                usage.cache_read_tokens,
                agent_usage.cost_usd if agent_usage else 0.0,
            )

            self._messages = result.all_messages()
            self._messages_current = True
            self._trace_new_messages(result)
            self._maybe_compact_messages()
            await self._handle_output(result.output)

            if self.deps.confirmed_flag:
                self._confirmed = True
                self._flag = self._flag or self.deps.confirmed_flag

            self.tracer.event("turn_complete", duration=round(duration, 1),
                              tool_calls=self._tool_calls[0] - calls_before)
            if self._confirmed and self._flag:
                return self._result(FLAG_FOUND, deltas)
            return self._result(GAVE_UP, deltas)

        except asyncio.CancelledError:
            self.tracer.event("cancelled", step_count=self._step_count[0])
            raise
        except Exception as e:  # noqa: BLE001 — mapped onto solver statuses below
            return self._handle_run_exception(e, deltas)

    async def _await_run_or_cancel(self, run_task: asyncio.Task):
        """Await an agent run, abandoning it if the swarm is cancelled first.

        Returns the run's result, or None when cancel_event won the race. The
        abandoned task is always cancelled AND awaited, so it cannot linger as
        "Task was destroyed but it is pending" at loop close.

        A cancel of THIS task (not the swarm's) still propagates: that is the
        hard-cancel `_wind_down` applies to a straggler after its grace period, and
        swallowing it here would defeat the last resort that makes the grace safe.
        """
        if self.cancel_event.is_set():
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run_task
            self.tracer.event("cancelled", phase="execute", reason="swarm already cancelled")
            return None

        cancel_waiter = asyncio.create_task(self.cancel_event.wait())
        try:
            done, _pending = await asyncio.wait(
                {run_task, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run_task
            raise
        finally:
            cancel_waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await cancel_waiter

        if run_task not in done:
            # cancel_event fired first: leave the turn, let teardown run.
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await run_task
            self._interrupted = True
            self.tracer.event("cancelled", phase="execute", reason="cancel_event")
            return None
        # Re-raises whatever the run raised, so the existing handler still maps it.
        return run_task.result()

    def _handle_run_exception(self, e: Exception, deltas: dict) -> SolverResult:
        """Map a pydantic-ai/model exception onto a solver status."""
        from pydantic_ai.exceptions import RunTimedOut

        text = str(e)
        if isinstance(e, RunTimedOut) or type(e).__name__ == "RunTimedOut":
            # Phase 1 budget expiry: the session survives, phase 2 can resume it.
            self._interrupted = True
            # ...but only if the history actually reflects THIS round. `_messages` is
            # assigned on success only, so without capturing the partial messages the
            # conclude phase would distil the PREVIOUS round — or, on a first-round
            # timeout, nothing at all, in which case the model invents a state card
            # from the system prompt alone. `_harvest_timeout` publishes that text to
            # the blackboard and every sibling then treats it as ground truth, so a
            # stale or fabricated harvest is worse than none.
            partial = _partial_messages(e)
            if partial:
                self._messages = partial
                self._messages_current = True
                self._trace_new_messages_from(partial)
            else:
                self._messages_current = False
            self.tracer.event("budget_interrupt", phase="execute", error=text[:200],
                              partial_messages=len(partial or ()))
            logger.warning(
                f"[{self.agent_name}] Time budget hit — turn interrupted "
                f"({len(partial or ())} message(s) recoverable for the conclude phase)"
            )
            return self._result(GAVE_UP, deltas, api_error="")

        if type(e).__name__ == "RunCancelled" or self.cancel_event.is_set():
            self.tracer.event("cancelled", error=text[:200])
            return self._result(CANCELLED, deltas)

        if is_context_overflow(e):
            # Recoverable only by dropping the conversation; do it here so the
            # next round starts clean instead of failing identically forever.
            logger.warning(f"[{self.agent_name}] Context overflow — resetting history")
            self.tracer.event("context_overflow_reset", error=text[:200])
            self.reset_history()
            result = self._result(ERROR, deltas, api_error=text[:300])
            result.context_overflow = True
            return result

        kind = classify_exception(e) or ERROR
        logger.error(f"[{self.agent_name}] Error ({kind}): {e}", exc_info=True)
        self._add_finding(f"Error: {e}")
        self.tracer.event("error", error=text, status=kind)
        return self._result(kind, deltas, api_error=text[:300])

    def _trace_new_messages(self, result) -> None:
        self._trace_messages(result.new_messages())

    def _trace_new_messages_from(self, messages) -> None:
        """Same tracing, for a message list obtained from an exception rather than a
        completed run."""
        self._trace_messages(messages)

    def _trace_messages(self, messages) -> None:
        from pydantic_ai.messages import ModelResponse, TextPart

        for msg in messages:
            if isinstance(msg, ModelResponse):
                text_parts = [p.content for p in msg.parts if isinstance(p, TextPart)]
                text = " ".join(text_parts)
                msg_usage = msg.usage
                if msg_usage and msg_usage.input_tokens:
                    self._last_input_tokens = msg_usage.input_tokens
                if text.strip():
                    self._add_finding(text)
                self.tracer.model_response(
                    text[:500], self._step_count[0],
                    input_tokens=msg_usage.input_tokens if msg_usage else 0,
                    output_tokens=msg_usage.output_tokens if msg_usage else 0,
                )

    async def _handle_output(self, output: Any) -> None:
        """Interpret the structured output.

        `StillWorking` exists so a model can end a turn honestly. When
        `FlagFound` was the only allowed output, an unsolved turn still had to
        produce one — so the model invented a flag, and that hallucination used
        to be accepted outright in --no-submit mode, cancelling every sibling.
        A claimed flag is therefore always routed through the same verifier the
        submit_flag tool uses.
        """
        if isinstance(output, FlagFound):
            candidate = (output.flag or "").strip()
            self.tracer.event("structured_flag", flag=candidate[:200],
                              method=(output.method or "")[:200])
            if not candidate:
                return
            if self.deps.submit_fn:
                display, confirmed = await self.deps.submit_fn(candidate)
            elif self.deps.no_submit:
                display, confirmed = f'DRY RUN — would submit "{candidate}"', False
            else:
                from backend.tools.core import FlagSubmissionError, do_submit_flag

                try:
                    display, confirmed = await do_submit_flag(
                        self.ctfd, self.meta.name, candidate
                    )
                except FlagSubmissionError as e:
                    # Never judged, so not a rejection. Recording it as one would send
                    # the model off to "fix" a flag that may well have been correct.
                    display = (
                        f"SUBMISSION FAILED — CTFd did not answer ({e}). The flag was "
                        "NOT judged and this is not a wrong attempt."
                    )
                    self._add_finding(display)
                    self.tracer.event("submit_undelivered", flag=candidate[:200],
                                      detail=str(e)[:200])
                    return
            self.tracer.event("structured_flag_verified", flag=candidate[:200],
                              confirmed=confirmed, detail=display[:200])
            if confirmed:
                self._confirmed = True
                self._flag = candidate
                self.deps.confirmed_flag = candidate
                self._add_finding(f"Flag found via {output.method or '?'}: {candidate}")
                self.tracer.event("flag_confirmed", flag=candidate,
                                  step=self._step_count[0])
            else:
                self._add_finding(f"Claimed flag {candidate} was REJECTED: {display}")
        elif isinstance(output, StillWorking):
            note = "\n".join(
                p for p in (
                    f"[fact] {output.confirmed}" if output.confirmed else "",
                    f"[excluded] {output.excluded}" if output.excluded else "",
                    f"[next] {output.next_step}" if output.next_step else "",
                ) if p
            )
            if note:
                self._add_finding(note)
            self.tracer.event("still_working", step=self._step_count[0])

    def _agent_cost(self) -> float:
        agent_usage = self.cost_tracker.by_agent.get(self.agent_name)
        return agent_usage.cost_usd if agent_usage else 0.0

    def _maybe_compact_messages(self) -> None:
        """Drop middle history when approaching the context window.

        Keeps the initial request plus a tail that starts at a clean user-turn
        boundary (a ModelRequest without tool parts), so tool_call/tool_return
        pairs are never orphaned. Falls back to [initial, summary] when no
        clean boundary exists.
        """
        window = context_window(self.model_spec)
        if not self._messages or self._last_input_tokens < int(window * 0.72):
            return

        from pydantic_ai.messages import RetryFeedbackPart, ToolReturnPart

        msgs = self._messages

        def _is_clean_boundary(m) -> bool:
            if not isinstance(m, ModelRequest):
                return False
            has_user = any(isinstance(p, UserPromptPart) for p in m.parts)
            # RetryPromptPart was split in pydantic-ai 2.x into a ToolReturnPart
            # with outcome='retried' (answers a tool call) and a
            # RetryFeedbackPart (answers none). Both still mean "this request is
            # glued to a preceding tool call", so neither is a safe cut point —
            # trimming there would orphan the call and some providers reject it.
            has_tool = any(isinstance(p, (ToolReturnPart, RetryFeedbackPart)) for p in m.parts)
            return has_user and not has_tool

        tail_start = 0
        for i in range(len(msgs) - 1, 0, -1):
            if _is_clean_boundary(msgs[i]) and len(msgs) - i >= 2:
                tail_start = i
                break

        summary_text = (
            "[Context compacted] Older tool outputs were dropped to free context space. "
            f"Your progress so far: {self.findings[:1500] or '(no summary available)'}\n"
            "Files you created persist under /challenge/workspace/  — "
            "re-run cheap read commands if you need their contents again."
        )
        summary_msg = ModelRequest(parts=[UserPromptPart(content=summary_text)])
        if tail_start:
            self._messages = [msgs[0], summary_msg, *msgs[tail_start:]]
        else:
            self._messages = [msgs[0], summary_msg]
        self.tracer.event(
            "context_compact",
            input_tokens=self._last_input_tokens,
            kept_messages=len(self._messages),
        )
        self._last_input_tokens = 0
        logger.info(f"[{self.agent_name}] Context compacted (window={window})")

    def _get_conclude_agent(self) -> Agent[SolverDeps, str]:
        """A tool-less, plain-text agent used only for the phase-2 harvest.

        Built separately rather than passing `toolsets=[]` to the main agent so
        the "no new work during conclude" guarantee does not depend on how a
        given pydantic-ai version interprets an empty toolset override.
        """
        if self._conclude_agent is None:
            self._conclude_agent = Agent(
                self._model,
                deps_type=SolverDeps,
                system_prompt=self._system_prompt,
                model_settings=self._model_settings,
                output_type=str,
            )
        return self._conclude_agent

    async def conclude(self, budget_s: int) -> str:
        """Phase 2: distil the interrupted round into a state card.

        Runs with NO tools and a plain-text output type, so the model cannot
        start new work even if it ignores the prompt. Returns "" on any failure
        — the caller then writes nothing to the blackboard.
        """
        if budget_s <= 0 or not self._agent or self._model is None:
            self.tracer.event("conclude_skipped", budget_s=budget_s,
                              has_agent=bool(self._agent))
            return ""
        if not self._messages_current:
            # `_messages` describes an EARLIER round: the last turn timed out and its
            # partial history could not be recovered from the exception. Distilling
            # that would publish a state card about work this round never did — and on
            # a first-round timeout there is no history at all, so the model would
            # invent one from the system prompt alone. The swarm publishes this text
            # to the blackboard and every sibling treats it as ground truth, which is
            # exactly the case where "nothing" beats "plausible".
            self.tracer.event("conclude_skipped", reason="history is from a prior round")
            logger.warning(
                f"[{self.agent_name}] refusing to conclude: no recoverable history for "
                "the interrupted round, so any state card would describe the wrong work"
            )
            return ""
        t0 = time.monotonic()
        self.tracer.event("conclude_start", budget_s=budget_s, phase="conclude")
        try:
            result = await self._get_conclude_agent().run(
                self.conclude_prompt or CONCLUDE_PROMPT,
                deps=self.deps,
                message_history=self._messages or None,
                timeout=float(budget_s),
            )
            text = result.output.strip() if isinstance(result.output, str) else ""
            if not text:
                self.tracer.event("conclude_empty")
                return ""
            # A gateway that returns error prose inside a 200 body produces a
            # "successful" conclude whose output is the error text. Published, that
            # becomes a state card on the blackboard which every sibling and every
            # restart treats as ground truth — the same failure the Claude backend
            # already guards against, and worse here because the text looks like a
            # harvest rather than a failure.
            if classify_api_error(text):
                self.tracer.event("conclude_api_error", detail=text[:200])
                logger.warning(
                    f"[{self.agent_name}] conclude returned an API error as its output "
                    "— discarding rather than publishing it as a state card"
                )
                return ""
            # The conclude turn's messages are deliberately NOT merged back into
            # _messages: the round is over, and the swarm decides whether the
            # next round continues this session or starts a fresh one.
            self.cost_tracker.record(
                self.agent_name, result.usage, self.model_id,
                provider_spec=provider_from_spec(self.model_spec),
                duration_seconds=time.monotonic() - t0,
            )
            self._add_finding(text)
            self.tracer.event("conclude_ok", chars=len(text),
                              duration=round(time.monotonic() - t0, 1))
            return text
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — harvesting is best-effort
            logger.warning(f"[{self.agent_name}] conclude failed: {e}")
            self.tracer.event("conclude_failed", error=str(e)[:300])
            return ""

    def bump(self, insights: str) -> None:
        """Queue guidance for the next turn. FIFO — nothing is overwritten."""
        insights = (insights or "").strip()
        if not insights:
            return
        self._bump_queue.append(insights)
        self.loop_detector.reset()
        # See ClaudeSolver.bump: payloads outgrew the old 500-char window.
        self.tracer.event("bump", insights=insights[:BUMP_TRACE_CHARS],
                          insights_chars=len(insights), queued=len(self._bump_queue))
        logger.info(f"[{self.agent_name}] Bump queued ({len(self._bump_queue)} pending)")

    def reset_history(self) -> None:
        """Hard-reset the conversation for a FRESH attempt (fresh-restart mode).

        The sandbox (files in /challenge/workspace) persists — only the LLM
        message history is cleared. Used with the distilled journal so each
        round starts with a clean context instead of an ever-growing history.
        """
        self._messages = []
        # After a hard reset the empty history IS the current state, so a later
        # conclude is not refused on a stale flag. A timeout in the next round
        # re-evaluates this from whether its partial messages were recoverable.
        self._messages_current = True
        self._last_input_tokens = 0
        self.loop_detector.reset()
        self.tracer.event("session_forgotten")

    # Alias so the swarm can reset either backend through one name.
    forget_session = reset_history

    def _result(self, status: str, deltas: dict, api_error: str = "") -> SolverResult:
        cost = self._agent_cost()
        in_tok = out_tok = cache_tok = 0
        agent_usage = self.cost_tracker.by_agent.get(self.agent_name)
        if agent_usage:
            in_tok = agent_usage.usage.input_tokens
            out_tok = agent_usage.usage.output_tokens
            cache_tok = agent_usage.usage.cache_read_tokens
        self.tracer.event("finish", status=status,
                          flag=self._flag if self._confirmed else None,
                          confirmed=self._confirmed, cost_usd=round(cost, 4),
                          interrupted=self._interrupted, api_error=api_error[:200])
        return SolverResult(
            # Only ever report a flag the verifier actually accepted.
            flag=self._flag if self._confirmed else None,
            status=status,
            findings_summary=self.findings,
            step_count=self._step_count[0] - deltas.get("steps", 0),
            cost_usd=max(0.0, cost - deltas.get("cost", 0.0)),
            log_path=self.tracer.path,
            tool_calls=self._tool_calls[0] - deltas.get("calls", 0),
            api_error=api_error,
            interrupted=self._interrupted,
            input_tokens=in_tok, output_tokens=out_tok, cache_read_tokens=cache_tok,
        )

    def detach_sandbox(self) -> object:
        """Hand sandbox ownership back to the caller so stop() leaves it alive."""
        self._owns_sandbox = False
        return self.sandbox

    async def stop(self) -> None:
        self.tracer.event("stop", step_count=self._step_count[0],
                          tool_calls=self._tool_calls[0])
        self.tracer.close()
        if self._owns_sandbox and self.sandbox:
            await self.sandbox.stop()
