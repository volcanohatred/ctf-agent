"""Per-model solver agent  — one model, one container, one challenge."""

from __future__ import annotations

import asyncio
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

        self.tool_counter[0] += 1
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
            result = await self._agent.run(
                self._next_prompt(),
                deps=self.deps,
                message_history=self._messages if self._messages else None,
                usage_limits=UsageLimits(request_limit=None),
                timeout=float(budget_s) if budget_s > 0 else None,
            )

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

    def _handle_run_exception(self, e: Exception, deltas: dict) -> SolverResult:
        """Map a pydantic-ai/model exception onto a solver status."""
        from pydantic_ai.exceptions import RunTimedOut

        text = str(e)
        if isinstance(e, RunTimedOut) or type(e).__name__ == "RunTimedOut":
            # Phase 1 budget expiry: the session survives, phase 2 can resume it.
            self._interrupted = True
            self.tracer.event("budget_interrupt", phase="execute", error=text[:200])
            logger.warning(f"[{self.agent_name}] Time budget hit — turn interrupted")
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
        from pydantic_ai.messages import ModelResponse, TextPart

        for msg in result.new_messages():
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
