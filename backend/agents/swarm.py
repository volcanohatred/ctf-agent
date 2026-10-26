"""ChallengeSwarm — Parallel solvers racing on one challenge."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.agents.solver import Solver
from backend.approaches import build_approach_queue
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.message_bus import ChallengeMessageBus
from backend.models import DEFAULT_MODELS, provider_from_spec
from backend.prompts import ChallengeMeta, list_distfiles
from backend.skills import SkillRouter
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    QUOTA_ERROR,
    SolverProtocol,
    SolverResult,
)

if TYPE_CHECKING:
    from backend.config import Settings

logger = logging.getLogger(__name__)


# Quota fallback: when a claude-cli (Anthropic-protocol) model exhausts quota,
# retry the SAME model through the OpenAI-compatible gateway endpoint (the
# pydantic solver reuses the existing sandbox container).
QUOTA_FALLBACK: dict[str, str] = {
    "claude-sdk/kimi-k3": "gateway/kimi-k3",
    "claude-sdk/deepseek-v4-pro": "gateway/deepseek-v4-pro",
    "claude-sdk/deepseek-v4-flash": "gateway/deepseek-v4-flash",
    "claude-sdk/qwen3.8-max": "gateway/qwen3.8-max",
    "claude-sdk/glm-5.3": "gateway/glm-5.3",
    # Legacy native-Claude fallback (requires AWS Bedrock credentials).
    "claude-sdk/claude-opus-4-6": "bedrock/us.anthropic.claude-opus-4-6-v1",
    "claude-sdk/claude-sonnet-4-5": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1",
}


def _quota_fallback_spec(model_spec: str) -> str | None:
    base = "/".join(model_spec.split("/")[:2])  # strip effort suffix
    return QUOTA_FALLBACK.get(base)


@dataclass
class ChallengeSwarm:
    """Parallel solvers racing on one challenge."""

    challenge_dir: str
    meta: ChallengeMeta
    ctfd: CTFdClient
    cost_tracker: CostTracker
    settings: Settings
    model_specs: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    no_submit: bool = False
    coordinator_inbox: asyncio.Queue | None = None
    expected_flags: list[str] = field(default_factory=list)  # offline answer key

    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    solvers: dict[str, SolverProtocol] = field(default_factory=dict)
    findings: dict[str, str] = field(default_factory=dict)
    winner: SolverResult | None = None
    winner_spec: str = ""
    confirmed_flag: str | None = None
    _flag_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _submit_count: dict[str, int] = field(default_factory=dict)  # per-model wrong submission count
    _submitted_flags: set[str] = field(default_factory=set)  # dedup exact flags
    _last_submit_time: dict[str, float] = field(default_factory=dict)  # per-model last submit timestamp
    message_bus: ChallengeMessageBus = field(default_factory=ChallengeMessageBus)
    skill_router: SkillRouter = field(init=False)
    _skill_pack_ids: list[str] = field(init=False, default_factory=list)
    # Direction-rotation state (anti-loop): per-challenge queue of candidate
    # techniques + per-solver assignment/failure tracking.
    _approach_queue: list = field(init=False, default_factory=list)
    _failed_approaches: list[str] = field(init=False, default_factory=list)
    _assigned_approach: dict[str, str] = field(init=False, default_factory=dict)
    _total_bumps: int = field(init=False, default=0)
    _escalated: bool = field(init=False, default=False)
    _pending_extra: list[str] = field(init=False, default_factory=list)  # escalation specs to spawn

    def __post_init__(self) -> None:
        self.skill_router = SkillRouter(skills_dir=getattr(self.settings, "skills_dir", "") or "")
        self._skill_pack_ids = []
        if self.skill_router.enabled:
            try:
                self._skill_pack_ids = self.skill_router.triage(
                    category=self.meta.category,
                    description=self.meta.description,
                    tags=self.meta.tags,
                    distfiles=list_distfiles(self.challenge_dir),
                )
                if self._skill_pack_ids:
                    logger.info(
                        f"[{self.meta.name}] Skill triage → {self._skill_pack_ids}"
                    )
                    self._approach_queue = build_approach_queue(
                        self._skill_pack_ids,
                        getattr(self.settings, "skills_dir", "") or "",
                    )
                    logger.info(
                        f"[{self.meta.name}] Approach queue: {len(self._approach_queue)} techniques"
                    )
            except Exception as e:
                logger.warning(f"[{self.meta.name}] Skill triage failed: {e}")

    def _create_solver(self, model_spec: str):
        """Create the right solver type based on provider.

        - claude-sdk/* → ClaudeSolver (Claude Agent SDK, subscription-first)
        - everything else → Pydantic AI Solver (API: bedrock/azure/zen/google/gateway)
        """
        provider = provider_from_spec(model_spec)

        def _submit_fn(flag): return self.try_submit_flag(flag, model_spec)
        _notify = self._make_notify_fn(model_spec)

        if provider == "claude-sdk":
            from backend.agents.claude_solver import ClaudeSolver
            return ClaudeSolver(
                model_spec=model_spec,
                challenge_dir=self.challenge_dir,
                meta=self.meta,
                ctfd=self.ctfd,
                cost_tracker=self.cost_tracker,
                settings=self.settings,
                cancel_event=self.cancel_event,
                no_submit=self.no_submit,
                submit_fn=_submit_fn,
                message_bus=self.message_bus,
                notify_coordinator=_notify,
            )

        return self._create_pydantic_solver(model_spec)

    def _make_notify_fn(self, model_spec: str):
        """Create a callback that pushes solver messages to the coordinator inbox."""
        async def _notify(message: str) -> None:
            if self.coordinator_inbox:
                self.coordinator_inbox.put_nowait(
                    f"[{self.meta.name}/{model_spec}] {message}"
                )
        return _notify

    def _create_pydantic_solver(self, model_spec: str, sandbox=None, owns_sandbox: bool | None = None) -> Solver:
        """Create a Pydantic AI solver. Pass sandbox to reuse an existing container (quota fallback)."""
        solver = Solver(
            model_spec=model_spec,
            challenge_dir=self.challenge_dir,
            meta=self.meta,
            ctfd=self.ctfd,
            cost_tracker=self.cost_tracker,
            settings=self.settings,
            cancel_event=self.cancel_event,
            sandbox=sandbox,
            owns_sandbox=owns_sandbox,
        )
        solver.deps.message_bus = self.message_bus
        solver.deps.model_spec = model_spec
        solver.deps.no_submit = self.no_submit
        solver.deps.submit_fn = lambda flag: self.try_submit_flag(flag, model_spec)
        solver.deps.notify_coordinator = self._make_notify_fn(model_spec)
        return solver

    def _gather_sibling_insights(self, exclude_model: str) -> str:
        """Sibling findings for bump payloads — bounded per finding so the
        bump context stays light (the blackboard stores them in full)."""
        def _clip(text: str, limit: int = 400) -> str:
            text = text.strip()
            return text if len(text) <= limit else text[:limit] + " …"

        parts: list[str] = []
        for model, finding in self.findings.items():
            if model != exclude_model and finding:
                parts.append(f"[{model}]: {_clip(finding)}")
        return "\n\n".join(parts) if parts else "No sibling insights available yet."

    def _skill_hint(self, bump_count: int) -> str:
        """Escalating knowledge-base hint, activated after repeated failures."""
        threshold = getattr(self.settings, "skill_after_bumps", 1)
        if not self._skill_pack_ids or bump_count < threshold:
            return ""
        level = min(1 + (bump_count - threshold), 3)
        keywords = [self.meta.category, *self.meta.tags, *self.meta.name.replace("-", " ").split()]
        try:
            return self.skill_router.render_hint(self._skill_pack_ids, level=level, keywords=keywords)
        except Exception as e:
            logger.warning(f"[{self.meta.name}] Skill hint render failed: {e}")
            return ""

    def _next_approach(self, model_spec: str):
        """Assign the next untried technique to this solver.

        The previously assigned approach is marked as failed (the solver is
        back for a bump), so retries systematically walk the technique space
        instead of repeating a dead end.
        """
        prev_path = self._assigned_approach.get(model_spec)
        if prev_path:
            prev = next((a for a in self._approach_queue if a.path == prev_path), None)
            if prev and prev.title not in self._failed_approaches:
                self._failed_approaches.append(prev.title)
        for a in self._approach_queue:
            if a.title not in self._failed_approaches and a.path != prev_path:
                self._assigned_approach[model_spec] = a.path
                return a
        return None

    def _build_bump_payload(self, model_spec: str, reason: str, bump_count: int) -> str:
        """Lean rotation bump: short reason + next untried technique + skill hint
        only from the configured failure threshold onward."""
        parts = [reason]

        if self._failed_approaches:
            failed = "、".join(self._failed_approaches)
            parts.append(f"\n🚫 已试且失败（勿重复）：{failed}")

        nxt = self._next_approach(model_spec)
        if nxt:
            parts.append(
                f"\n🎯 本轮专攻新方向：【{nxt.title}】（手册 {nxt.path}，先读再做）"
            )

        insights = self._gather_sibling_insights(model_spec)
        if "No sibling insights" not in insights:
            parts += ["", insights]

        hint = self._skill_hint(bump_count)
        if hint:
            parts += ["", hint]
        return "\n".join(p for p in parts if p.strip())

    # Escalation: after this many total failed bumps across the swarm with no
    # winner, add a stronger solver instead of spinning in place forever.
    # Overridable via settings.escalate_after_bumps.
    ESCALATE_AFTER_BUMPS = 8

    def _maybe_escalate(self) -> None:
        """Queue a stronger-model solver when the swarm keeps failing."""
        if self._escalated or self.cancel_event.is_set():
            return
        threshold = int(getattr(self.settings, "escalate_after_bumps", 0) or self.ESCALATE_AFTER_BUMPS)
        if self._total_bumps < threshold:
            return
        strong = getattr(self.settings, "coordinator_model", "") or "qwen3.8-max"
        spec = f"claude-sdk/{strong}"
        if spec in self.solvers or spec in self._pending_extra:
            return
        self._escalated = True
        logger.warning(
            f"[{self.meta.name}] {self._total_bumps} bumps without a flag — escalating to {spec}"
        )
        self._pending_extra.append(spec)

    async def _run_escalated_solver(self, model_spec: str) -> None:
        """Run an escalation solver seeded with the full dead-end ledger."""
        solver = self._create_solver(model_spec)
        self.solvers[model_spec] = solver
        try:
            await solver.start()
            solver.bump(self._escalation_briefing())  # seed it with everything we learned
            result, solver = await self._run_solver_loop(solver, model_spec)
            if result and result.status == FLAG_FOUND:
                self.winner = result
                self.winner_spec = model_spec
                self.cancel_event.set()
        except Exception as e:
            logger.error(f"[{self.meta.name}/{model_spec}] escalation solver error: {e}")
        finally:
            await solver.stop()

    def _escalation_briefing(self) -> str:
        failed = "\n".join(f"  ✗ {t}" for t in self._failed_approaches) or "  (none recorded)"
        tried_files = "\n".join(f"  - {a.path}" for a in self._approach_queue
                                if a.title in self._failed_approaches) or "  (none)"
        return (
            "You are a STRONGER solver brought in after others failed this challenge.\n\n"
            "Directions already tried and FAILED (do not repeat):\n" + failed + "\n\n"
            "Skill handbooks consulted so far:\n" + tried_files + "\n\n"
            "Bring a genuinely different, more sophisticated technique. "
            "If the library has an advanced file, use it."
        )

    # Escalating cooldowns after incorrect submissions (per model)
    SUBMISSION_COOLDOWNS = [0, 30, 120, 300, 600]  # 0s, 30s, 2min, 5min, 10min

    async def try_submit_flag(self, flag: str, model_spec: str) -> tuple[str, bool]:
        """Cooldown-gated, deduplicated flag submission. Returns (display, is_confirmed)."""
        async with self._flag_lock:
            if self.confirmed_flag:
                return f"ALREADY SOLVED — flag already confirmed: {self.confirmed_flag}", True

            normalized = flag.strip()

            # Dedup exact flags across all models
            if normalized in self._submitted_flags:
                return "INCORRECT — already tried this exact flag.", False

            # Escalating cooldown after incorrect submissions
            wrong_count = self._submit_count.get(model_spec, 0)
            cooldown_idx = min(wrong_count, len(self.SUBMISSION_COOLDOWNS) - 1)
            cooldown = self.SUBMISSION_COOLDOWNS[cooldown_idx]
            if cooldown > 0:
                last_time = self._last_submit_time.get(model_spec, 0)
                elapsed = time.monotonic() - last_time
                if elapsed < cooldown:
                    remaining = int(cooldown - elapsed)
                    return (
                        f"COOLDOWN — wait {remaining}s before submitting again. "
                        f"You have {wrong_count} incorrect submissions. "
                        "Use this time to do deeper analysis and verify your flag.",
                        False,
                    )

            self._submitted_flags.add(normalized)

            if self.no_submit:
                # Offline mode: verify against the local answer key if present.
                if self.expected_flags:
                    is_confirmed = any(
                        normalized.casefold() == ef.strip().casefold()
                        for ef in self.expected_flags
                    )
                    display = (
                        "CORRECT — matches the local answer key."
                        if is_confirmed
                        else "INCORRECT — does not match the local answer key."
                    )
                else:
                    display, is_confirmed = f'DRY RUN — would submit "{normalized}"', False
            else:
                from backend.tools.core import do_submit_flag
                display, is_confirmed = await do_submit_flag(self.ctfd, self.meta.name, flag)
            if is_confirmed:
                self.confirmed_flag = normalized
            else:
                self._submit_count[model_spec] = wrong_count + 1
                self._last_submit_time[model_spec] = time.monotonic()
            return display, is_confirmed

    async def _post_finding(self, model_spec: str, content: str, kind: str) -> None:
        """Post a finding to the blackboard; LLM-summarize long ones so siblings
        get the key info (never truncated) without the verbose process log."""
        content = (content or "").strip()
        if not content:
            return
        summary = ""
        threshold = int(getattr(self.settings, "summarize_threshold_chars", 900) or 900)
        if len(content) > threshold:
            try:
                from backend.summarize import summarize_for_siblings
                summary = await summarize_for_siblings(content, self.settings)
            except Exception:
                summary = ""
        await self.message_bus.post(model_spec, content, kind=kind, summary=summary)

    async def _run_solver(self, model_spec: str) -> SolverResult | None:
        solver = self._create_solver(model_spec)
        self.solvers[model_spec] = solver

        try:
            result, final_solver = await self._run_solver_loop(solver, model_spec)
            solver = final_solver
            return result
        except Exception as e:
            logger.error(f"[{self.meta.name}/{model_spec}] Fatal: {e}", exc_info=True)
            return None
        finally:
            await solver.stop()

    async def _harvest_timeout(self, solver, model_spec: str) -> None:
        """Cairn-style conclude on wall-clock timeout: harvest, reset, move on."""
        logger.warning(f"[{self.meta.name}/{model_spec}] Time budget exhausted — harvesting")
        reset = getattr(solver, "reset_session", None)
        if reset:
            try:
                await reset()
            except Exception:
                pass
        findings = getattr(solver, "_findings", "") or ""
        if findings:
            self.findings[model_spec] = findings
            await self._post_finding(model_spec, f"[excluded] (time budget hit) {findings}", "excluded")

    async def _run_solver_loop(self, solver, model_spec: str) -> tuple[SolverResult, SolverProtocol]:
        """Inner loop: start → run → (timeout harvest) → bump → run → ..."""
        bump_count = 0
        consecutive_errors = 0
        quota_retries = 0
        budget_s = int(getattr(self.settings, "solver_time_budget_s", 0) or 0)
        result = SolverResult(
            flag=None, status=CANCELLED, findings_summary="",
            step_count=0, cost_usd=0.0, log_path="",
        )
        await solver.start()

        while not self.cancel_event.is_set():
            timed_out = False
            if budget_s > 0:
                # asyncio.wait does NOT cancel the task on timeout (wait_for does,
                # and the solvers' internal CancelledError handlers would swallow
                # it into a clean CANCELLED, defeating the budget). Wait, then
                # cancel explicitly and take the harvest path.
                run_task = asyncio.create_task(solver.run_until_done_or_gave_up())
                done, pending = await asyncio.wait({run_task}, timeout=budget_s)
                if pending:
                    timed_out = True
                    run_task.cancel()
                    try:
                        await run_task
                    except BaseException:
                        pass  # task was cancelled mid-flight; harvest below
                else:
                    try:
                        result = run_task.result()
                    except Exception as e:
                        logger.error(f"[{self.meta.name}/{model_spec}] Run task error: {e}")
                        result = SolverResult(
                            flag=None, status=ERROR, findings_summary=f"Error: {e}",
                            step_count=1, cost_usd=0.01, log_path="",
                        )
            else:
                result = await solver.run_until_done_or_gave_up()

            if timed_out:
                await self._harvest_timeout(solver, model_spec)
                result = SolverResult(
                    flag=None, status=GAVE_UP,
                    findings_summary="time budget exhausted",
                    step_count=1, cost_usd=0.01, log_path="",
                )
            else:
                # Only broadcast useful findings — skip errors and broken solvers
                if (result.status not in (ERROR, QUOTA_ERROR)
                        and not (result.step_count == 0 and result.cost_usd == 0)
                        and result.findings_summary
                        and not result.findings_summary.startswith(("Error:", "Turn failed:"))):
                    self.findings[model_spec] = result.findings_summary
                    kind = "excluded" if result.status == GAVE_UP else "fact"
                    # Store the FULL finding on the blackboard (dashboard shows it);
                    # long ones get an LLM key-info summary for sibling injection.
                    await self._post_finding(model_spec, result.findings_summary, kind)

                if result.status == FLAG_FOUND:
                    self.cancel_event.set()
                    self.winner = result
                    self.winner_spec = model_spec
                    logger.info(
                        f"[{self.meta.name}] Flag found by {model_spec}: {result.flag}"
                    )
                    return result, solver

                if result.status == CANCELLED:
                    break

                # Quota exhaustion: fall back to API-backed Pydantic AI solver
                if result.status == QUOTA_ERROR:
                    fallback_spec = _quota_fallback_spec(model_spec)
                    if fallback_spec:
                        logger.warning(
                            f"[{self.meta.name}/{model_spec}] Quota exhausted — falling back to {fallback_spec}"
                        )
                        existing_sandbox = solver.sandbox
                        # Detach sandbox from old solver so stop() doesn't destroy it
                        solver.sandbox = None  # type: ignore[assignment]
                        await solver.stop()
                        solver = self._create_pydantic_solver(fallback_spec, sandbox=existing_sandbox, owns_sandbox=True)
                        self.solvers[model_spec] = solver
                        await solver.start()
                        continue
                    # No fallback — quota errors are often transient; back off and retry
                    quota_retries += 1
                    if quota_retries > 5:
                        logger.warning(
                            f"[{self.meta.name}/{model_spec}] Quota retries exhausted — giving up"
                        )
                        break
                    try:
                        await asyncio.wait_for(
                            self.cancel_event.wait(),
                            timeout=min(60 * quota_retries, 300),
                        )
                        break  # cancelled during backoff
                    except TimeoutError:
                        pass
                    continue

                if result.status == ERROR:
                    # Broken solver (0 steps, $0) will never recover — drop it
                    if result.step_count == 0 and result.cost_usd == 0:
                        logger.warning(
                            f"[{self.meta.name}/{model_spec}] Broken (0 steps, $0) — not bumping"
                        )
                        break
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        logger.warning(
                            f"[{self.meta.name}/{model_spec}] {consecutive_errors} consecutive errors — giving up"
                        )
                        break
                else:
                    consecutive_errors = 0

            # ---- bump path: GAVE_UP, recoverable ERROR, or time budget hit ----
            bump_count += 1
            self._total_bumps += 1
            self._maybe_escalate()
            # Cooldown between bumps — check cancellation during wait
            try:
                await asyncio.wait_for(
                    self.cancel_event.wait(),
                    timeout=min(5 + bump_count * 10, 120),
                )
                break  # cancelled during cooldown
            except TimeoutError:
                pass  # cooldown elapsed, proceed with bump

            if timed_out:
                reason = (
                    f"Your attempt hit the {budget_s}s wall-clock time budget and was "
                    "interrupted. Your previous session state was harvested. Work in "
                    "smaller increments, and pick a FASTER, more direct strategy."
                )
            else:
                reason = "Your previous attempt did not find the flag."
            solver.bump(self._build_bump_payload(model_spec, reason, bump_count))
            logger.info(
                f"[{self.meta.name}/{model_spec}] Bumped ({bump_count}), resuming"
            )

        return result, solver

    async def run(self) -> SolverResult | None:
        """Run all solvers in parallel. Returns the winner's result or None."""
        tasks: set[asyncio.Task] = {
            asyncio.create_task(self._run_solver(spec), name=f"solver-{spec}")
            for spec in self.model_specs
        }

        try:
            while tasks:
                # Timeout so escalations queued mid-run get picked up.
                done, pending = await asyncio.wait(
                    tasks, timeout=10, return_when=asyncio.FIRST_COMPLETED
                )

                # Spawn any escalation solvers queued since last wake
                while self._pending_extra:
                    spec = self._pending_extra.pop(0)
                    pending.add(asyncio.create_task(
                        self._run_escalated_solver(spec), name=f"solver-escalate-{spec}"
                    ))

                for task in done:
                    try:
                        result = task.result()
                    except Exception:
                        continue
                    if result and result.status == FLAG_FOUND:
                        self.cancel_event.set()
                        for p in pending:
                            p.cancel()
                        await asyncio.gather(*pending, return_exceptions=True)
                        return result

                tasks = set(pending)

            self.cancel_event.set()
            return self.winner
        except asyncio.CancelledError:
            # External cancellation (e.g. offline benchmark time limit) —
            # make sure child solver tasks don't linger as orphans.
            self.cancel_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception as e:
            logger.error(f"[{self.meta.name}] Swarm error: {e}", exc_info=True)
            self.cancel_event.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return None

    def kill(self) -> None:
        """Cancel all agents for this challenge."""
        self.cancel_event.set()

    def get_status(self) -> dict:
        """Get per-agent progress and findings."""
        return {
            "challenge": self.meta.name,
            "cancelled": self.cancel_event.is_set(),
            "winner": self.winner.flag if self.winner else None,
            "agents": {
                spec: {
                    "findings": self.findings.get(spec, ""),
                    "status": "running" if spec in self.solvers and not self.cancel_event.is_set()
                              else ("won" if self.winner and self.winner.flag else "finished"),
                }
                for spec in self.model_specs
            },
        }
