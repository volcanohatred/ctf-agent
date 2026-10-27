"""ChallengeSwarm — Parallel solvers racing on one challenge."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.agents.solver import Solver
from backend.approaches import build_approach_queue
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.message_bus import KIND_EXCLUDED, KIND_FACT, ChallengeMessageBus
from backend.models import DEFAULT_MODELS, escalation_spec, provider_from_spec
from backend.prompts import ChallengeMeta, list_distfiles
from backend.skills import SkillRouter
from backend.solver_base import (
    CANCELLED,
    ERROR,
    FLAG_FOUND,
    GAVE_UP,
    INFRA_ERROR,
    QUOTA_ERROR,
    SolverProtocol,
    SolverResult,
    classify_api_error,
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


_TAG_RE = re.compile(
    r"^[\[\(]?\s*(excluded|facts?|next|artifacts?)\s*[\]\)]?\s*[:：\-]?\s*(.*)$",
    re.IGNORECASE,
)
# A tag must be either bracketed ([fact] ...) or delimited (fact: ...). Without
# this, ordinary prose starting with one of the words ("facts about the binary")
# would be misread as a section header.
_BRACKETED_RE = re.compile(r"^\s*[\[\(]\s*(excluded|facts?|next|artifacts?)\s*[\]\)]",
                           re.IGNORECASE)
_DELIMITED_RE = re.compile(r"^\s*(excluded|facts?|next|artifacts?)\s*[:：\-]",
                            re.IGNORECASE)


def _match_tag(line: str) -> tuple[str, str] | None:
    """Return (tag, body) when the line opens a tagged section, else None."""
    if not (_BRACKETED_RE.match(line) or _DELIMITED_RE.match(line)):
        return None
    m = _TAG_RE.match(line)
    if not m:
        return None
    return m.group(1).lower().rstrip("s"), m.group(2).strip()


def split_tagged(text: str) -> tuple[list[str], list[str]]:
    """Split a harvested finding into (excluded_lines, fact_lines).

    Solvers are asked to tag their round summary with [excluded] / [fact] /
    [next]. Only lines the model explicitly ruled out may be published as dead
    ends — publishing an entire failed round as "do NOT retry this" (the
    previous behaviour) mislabels confirmed facts as dead ends, and once an API
    error text got published that way it told every sibling that a *network
    problem* was a ruled-out technique.

    Untagged text (including any preamble before the first tag) is treated as
    observation, i.e. facts — never as a dead end.
    """
    excluded: list[str] = []
    facts: list[str] = []
    current: list[str] = facts
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        matched = _match_tag(line)
        if matched:
            tag, body = matched
            if tag == "excluded":
                current = excluded
            elif tag in ("fact", "artifact"):
                current = facts
            else:
                # "next" is guidance for the following round, not blackboard data.
                current = []
            if body:
                current.append(body)
            continue
        current.append(line)
    return excluded, facts


def publishable(text: str) -> bool:
    """False for content that must never reach the blackboard."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.startswith(("Error:", "Turn failed:", "Traceback")):
        return False
    return classify_api_error(stripped) is None


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
    # Identity for the dead-end ledger is the handbook PATH, not its title: two
    # files can share an H1, and title-based exclusion used to blacklist an
    # untried technique along with the one that actually failed.
    _failed_approach_paths: list[str] = field(init=False, default_factory=list)
    _assigned_approach: dict[str, str] = field(init=False, default_factory=dict)
    _total_bumps: int = field(init=False, default=0)
    _escalated: bool = field(init=False, default=False)
    _escalation_exhausted: bool = field(init=False, default=False)
    _pending_extra: list[str] = field(init=False, default_factory=list)  # escalation specs to spawn
    _journals: dict[str, list[str]] = field(init=False, default_factory=dict)  # per-solver distilled rounds
    # Every spec this swarm has run, including escalations. `model_specs` alone
    # hid escalated solvers from get_status(), the dashboard and the export.
    _all_specs: list[str] = field(init=False, default_factory=list)
    # Consecutive infrastructure failures across the swarm. A dead gateway must
    # retire the swarm, not grind through the whole time limit.
    _infra_failures: int = field(init=False, default=0)
    _retire_reason: str = field(init=False, default="")
    # Dead-end ledger of *published* excluded lines, normalized, so a restated
    # dead end cannot accumulate independently of what the LLM merge catches.
    _seen_dead_ends: list[str] = field(init=False, default_factory=list)
    # Swarm-wide distilled state, recomputed at most once per bump generation and
    # shared by every solver restarting in that generation.
    _team_digest: str = field(init=False, default="")
    _team_digest_gen: int = field(init=False, default=-1)
    _digest_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._all_specs = list(self.model_specs)
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
            except Exception as e:  # noqa: BLE001 — skills are an optional enhancement
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

    def _collect_team_reports(self) -> list[str]:
        """Raw material for the team digest: the whole blackboard, grouped.

        Read from the message bus rather than `self.findings` — the bus keeps up
        to max_findings entries with their kind, while `self.findings` only ever
        holds each model's LATEST finding, so early discoveries silently dropped
        out of every restart payload.
        """
        grouped: dict[str, list[str]] = {}
        for f in self.message_bus.findings:
            if not publishable(f.content):
                continue
            tag = {"excluded": "✗", "help": "?"}.get(f.kind, "✓")
            grouped.setdefault(f.model, []).append(f"{tag} {f.content.strip()}")
        return [
            f"[{model}]\n" + "\n".join(items)
            for model, items in grouped.items()
        ]

    def _raw_sibling_fallback(self, exclude_model: str) -> str:
        """Bounded raw form, used when no digest is available."""
        def _clip(text: str, limit: int = 400) -> str:
            text = text.strip()
            return text if len(text) <= limit else text[:limit] + " …"

        parts = [
            f"[{model}]: {_clip(finding)}"
            for model, finding in self.findings.items()
            if model != exclude_model and finding and publishable(finding)
        ]
        return "\n".join(parts)

    async def _team_state(self, generation: int, exclude_model: str) -> str:
        """Distilled, de-duplicated picture of what the swarm has established.

        Recomputed at most once per bump generation and shared by every solver
        restarting in that generation, so four solvers cost one summarizer call
        instead of four. A restarting solver has had its session wiped, so its
        own earlier findings are included deliberately — it no longer remembers
        them, and re-reading them is the point.
        """
        async with self._digest_lock:
            refresh_every = max(1, int(getattr(self.settings, "sibling_digest_refresh_bumps", 1) or 1))
            age = generation - self._team_digest_gen
            if self._team_digest and 0 <= age < refresh_every:
                return self._team_digest

            reports = self._collect_team_reports()
            digest = ""
            if reports:
                try:
                    from backend.round_summary import summarize_siblings

                    every = int(getattr(self.settings, "sibling_digest_min_reports", 2) or 2)
                    if len(reports) >= every:
                        res = await summarize_siblings(
                            reports, self.settings,
                            cost_tracker=self.cost_tracker,
                            agent_name=f"{self.meta.name}/summarizer",
                        )
                        if res.condensed:
                            digest = res.text
                except Exception as e:  # noqa: BLE001 — the digest is an enhancement
                    logger.debug(f"[{self.meta.name}] team digest failed: {e}")

            if not digest:
                digest = self._raw_sibling_fallback(exclude_model)
            self._team_digest = digest
            self._team_digest_gen = generation
            return digest

    def _gather_sibling_insights(self, exclude_model: str) -> str:
        """Synchronous, undigested sibling findings.

        Kept for the escalation briefing and any caller that must not await.
        Returns "" when there is nothing to say — callers test truthiness, and an
        earlier sentinel string made that test always pass, injecting a
        meaningless placeholder section into every payload.
        """
        return self._raw_sibling_fallback(exclude_model)

    def _skill_hint(self, bump_count: int) -> str:
        """Escalating knowledge-base hint, activated after repeated failures."""
        threshold = getattr(self.settings, "skill_after_bumps", 1)
        if not self._skill_pack_ids or bump_count < threshold:
            return ""
        level = min(1 + (bump_count - threshold), 3)
        keywords = [self.meta.category, *self.meta.tags, *self.meta.name.replace("-", " ").split()]
        try:
            return self.skill_router.render_hint(self._skill_pack_ids, level=level, keywords=keywords)
        except Exception as e:  # noqa: BLE001 — hints are best-effort
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
            if prev and prev.path not in self._failed_approach_paths:
                self._failed_approach_paths.append(prev.path)
                if prev.title not in self._failed_approaches:
                    self._failed_approaches.append(prev.title)
        for a in self._approach_queue:
            if a.path not in self._failed_approach_paths and a.path != prev_path:
                self._assigned_approach[model_spec] = a.path
                return a
        return None

    async def _build_bump_payload(self, model_spec: str, reason: str, bump_count: int) -> str:
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

        team = await self._team_state(bump_count, model_spec)
        if team:
            parts += ["", "【全队进展（已去重蒸馏）】", team]

        hint = self._skill_hint(bump_count)
        if hint:
            parts += ["", hint]
        return "\n".join(p for p in parts if p.strip())

    def _maybe_escalate(self) -> None:
        """Queue a stronger-model solver when the swarm keeps failing.

        `escalate_after_bumps = 0` genuinely disables the ladder. The candidate
        must NOT already be in the lineup — resolving to a model that is already
        running used to make this a silent no-op forever, because the default
        coordinator model was also DEFAULT_MODELS[0].
        """
        if self._escalated or self._escalation_exhausted or self.cancel_event.is_set():
            return
        threshold = int(getattr(self.settings, "escalate_after_bumps", 0) or 0)
        if threshold <= 0 or self._total_bumps < threshold:
            return
        spec = escalation_spec(self._all_specs, self.settings)
        if not spec:
            self._escalation_exhausted = True
            logger.info(
                f"[{self.meta.name}] Escalation requested but every stronger model "
                "is already running — continuing with the current lineup"
            )
            return
        if spec in self.solvers or spec in self._pending_extra:
            self._escalation_exhausted = True
            return
        self._escalated = True
        logger.warning(
            f"[{self.meta.name}] {self._total_bumps} bumps without a flag — escalating to {spec}"
        )
        self._pending_extra.append(spec)

    async def _run_escalated_solver(self, model_spec: str) -> SolverResult | None:
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
            return result
        except asyncio.CancelledError:
            logger.info(f"[{self.meta.name}/{model_spec}] escalation cancelled")
            return None
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.meta.name}/{model_spec}] escalation solver error: {e}",
                         exc_info=True)
            return None
        finally:
            await solver.stop()

    def _escalation_briefing(self) -> str:
        failed = "\n".join(f"  ✗ {t}" for t in self._failed_approaches) or "  (none recorded)"
        tried_files = "\n".join(f"  - {p}" for p in self._failed_approach_paths) or "  (none)"
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

            normalized = (flag or "").strip()
            if not normalized:
                return "Empty flag — nothing to submit.", False

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
                logger.info(f"[{self.meta.name}] Flag CONFIRMED via {model_spec}")
            else:
                self._submit_count[model_spec] = wrong_count + 1
                self._last_submit_time[model_spec] = time.monotonic()
                logger.info(
                    f"[{self.meta.name}/{model_spec}] Rejected submission "
                    f"#{wrong_count + 1}: {normalized[:80]}"
                )
            return display, is_confirmed

    async def _post_finding(self, model_spec: str, content: str, kind: str) -> None:
        """Post a finding to the blackboard; LLM-summarize long ones so siblings
        get the key info (never truncated) without the verbose process log."""
        content = (content or "").strip()
        if not content or not publishable(content):
            return
        summary = ""
        threshold = int(getattr(self.settings, "summarize_threshold_chars", 900) or 900)
        if len(content) > threshold:
            try:
                from backend.summarize import summarize_for_siblings
                summary = await summarize_for_siblings(
                    content, self.settings,
                    cost_tracker=self.cost_tracker,
                    agent_name=f"{self.meta.name}/summarizer",
                )
                # A summary that came back unchanged is not a summary.
                if summary.strip() == content:
                    summary = ""
            except Exception as e:  # noqa: BLE001 — summarizing is best-effort
                logger.debug(f"[{self.meta.name}] summarize failed: {e}")
                summary = ""
        await self.message_bus.post(model_spec, content, kind=kind, summary=summary)

    async def _publish_harvest(self, model_spec: str, text: str) -> None:
        """Publish a round's output, routing tagged lines to the right kind.

        Untagged output is published as a neutral FACT. Labeling a whole failed
        round as an excluded dead end — as an earlier revision did — tells
        siblings "do NOT retry this" about content that is mostly confirmed
        observations.

        Dead ends are additionally de-duplicated structurally, before the LLM
        ever sees them: a restated dead end must not grow the "do not retry"
        list, and the ledger is what steers every subsequent round.
        """
        if not publishable(text):
            return
        excluded, facts = split_tagged(text)
        posted = False
        if excluded:
            from backend.dedup import new_lines

            fresh = new_lines(excluded, self._seen_dead_ends)
            if fresh:
                self._seen_dead_ends.extend(fresh)
                body = "\n".join(f"- {e}" for e in fresh)
                await self._post_finding(model_spec, body, KIND_EXCLUDED)
                posted = True
            elif excluded:
                logger.debug(
                    f"[{self.meta.name}/{model_spec}] {len(excluded)} dead end(s) "
                    "already recorded — not republishing"
                )
        if facts:
            from backend.dedup import dedup_lines

            body = "\n".join(f"- {f}" for f in dedup_lines(facts))
            await self._post_finding(model_spec, body, KIND_FACT)
            posted = True
        if not posted:
            await self._post_finding(model_spec, text, KIND_FACT)

    async def _fresh_restart(self, solver, model_spec: str, result: SolverResult,
                             reason: str, bump_count: int) -> None:
        """Fresh-restart retry: distill this round into the journal, optionally
        meta-summarize it (dedup) every N rounds, reset the session, and seed a
        clean new attempt carrying ONLY the distilled state — never an
        ever-growing conversation history."""
        from backend.round_summary import meta_summarize, summarize_round

        fresh_rounds = int(getattr(self.settings, "fresh_restart_rounds", 5) or 0)
        # 1. distill this round — prefer the phase-2 harvest, then the solver's
        #    accumulated findings, then the synthetic placeholder.
        raw = (result.concluded or getattr(solver, "findings", "")
               or result.findings_summary or reason)
        note = await summarize_round(raw, self.settings,
                                     cost_tracker=self.cost_tracker,
                                     agent_name=f"{self.meta.name}/summarizer")
        if note:
            journal = self._journals.setdefault(model_spec, [])
            journal.append(note)

            # 2. meta-summarize (dedup) every N rounds so the state never bloats.
            #    Entries are de-duplicated STRUCTURALLY first, so the model is
            #    handed an already-unique set and a restated round cannot consume
            #    one of the N slots.
            #    On failure the bounded LIST is kept: collapsing it into one giant
            #    joined string would make the carried state grow without bound,
            #    which is the exact problem this mechanism exists to prevent.
            if fresh_rounds > 0 and len(journal) >= fresh_rounds:
                from backend.dedup import dedup_entries

                unique = dedup_entries(journal)
                state = await meta_summarize(unique, self.settings,
                                             cost_tracker=self.cost_tracker,
                                             agent_name=f"{self.meta.name}/summarizer")
                if state.condensed:
                    self._journals[model_spec] = dedup_entries([state.text])
                else:
                    # Keep the newest entries only, so the payload stays bounded
                    # even when the summarizer is unavailable.
                    self._journals[model_spec] = unique[-fresh_rounds:]

        # 3. hard-reset the session (workspace files persist; history clears)
        await self._reset_solver_session(solver)

        # 4. build the restart seed and queue it into the fresh session
        solver.bump(await self._build_restart_payload(model_spec, reason, bump_count))

    @staticmethod
    async def _reset_solver_session(solver) -> None:
        """Forget the conversation on either backend (sync or async, by name)."""
        for name in ("forget_session", "reset_history", "reset_session"):
            fn = getattr(solver, name, None)
            if fn is None:
                continue
            try:
                out = fn()
                if asyncio.iscoroutine(out):
                    await out
                return
            except Exception as e:  # noqa: BLE001 — a failed reset must not kill the round
                logger.warning("session reset via %s failed: %s", name, e)

    async def _build_restart_payload(self, model_spec: str, reason: str, bump_count: int) -> str:
        """The seed message for a freshly restarted session."""
        journal = self._journals.get(model_spec, [])
        state_txt = "\n".join(f"{i+1}. {j}" for i, j in enumerate(journal)) or "(暂无)"

        parts = [
            f"这是一次全新的尝试（第 {bump_count + 1} 轮）。你之前的探索状态已被蒸馏如下，"
            "不要重复已失败的方向，直接基于它继续：",
            "",
            "【探索状态】",
            state_txt,
        ]
        # Rotation: assign the next untried technique
        nxt = self._next_approach(model_spec)
        if nxt:
            parts.append(
                f"\n🎯 本轮主攻方向（未试过）：【{nxt.title}】（手册 {nxt.path}，先读再做）"
            )
        hint = self._skill_hint(bump_count)
        if hint:
            parts += ["", hint]
        # The session was just wiped, so this solver no longer remembers its own
        # earlier work either — the team digest intentionally includes it.
        team = await self._team_state(bump_count, model_spec)
        if team:
            parts += ["", "【全队进展（含你自己之前的发现，已去重蒸馏）】", team]
        parts.append("\n开始吧。工作区里你之前创建的文件还在（/challenge/workspace/）。")
        return "\n".join(p for p in parts if p.strip())

    async def _harvest_timeout(self, solver, model_spec: str, budget_s: int) -> str:
        """Cairn-style conclude on wall-clock timeout.

        Phase 1 already interrupted the turn without destroying the session, so
        phase 2 can resume that same session and ask for a distilled state card
        under its own independent budget. The session is deliberately left
        ALIVE: the continue-session bump path reuses it, and the fresh-restart
        path performs its own hard reset. If the conclude phase fails, "" is
        returned and NOTHING is written to the blackboard — a partial or invented
        harvest is worse than none, because siblings treat it as ground truth.
        """
        logger.warning(f"[{self.meta.name}/{model_spec}] Time budget exhausted — concluding")
        concluded = ""
        try:
            concluded = await solver.conclude(budget_s)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — harvesting is best-effort
            logger.warning(f"[{self.meta.name}/{model_spec}] conclude failed: {e}")

        if concluded:
            self.findings[model_spec] = concluded
            await self._publish_harvest(model_spec, concluded)
        else:
            fallback = getattr(solver, "findings", "") or ""
            if fallback and publishable(fallback):
                # No phase-2 answer: fall back to whatever phase 1 confirmed,
                # published as facts (never as a dead end).
                self.findings[model_spec] = fallback
                await self._publish_harvest(model_spec, fallback)
            else:
                logger.info(
                    f"[{self.meta.name}/{model_spec}] Nothing harvestable — "
                    "blackboard left untouched"
                )
        return concluded

    async def _account_infra(self, result: SolverResult, model_spec: str) -> None:
        """Classify a round as infrastructure failure vs real work.

        A round that ended with zero tool calls is never "the model gave up" — it
        means the endpoint never served a usable turn. Counting those separately
        is what lets a dead gateway retire the swarm in a few rounds instead of
        grinding through the entire time limit while the report blames the
        challenge. Sets cancel_event when the swarm-level threshold is reached.
        """
        max_infra = int(getattr(self.settings, "max_infra_failures", 3) or 0)
        infra = result.is_infra_failure or (
            result.status == GAVE_UP and not result.did_work
        )
        if not infra:
            self._infra_failures = 0
            return

        if result.status not in (INFRA_ERROR, QUOTA_ERROR):
            result.status = INFRA_ERROR
            result.api_error = result.api_error or "no tool calls executed"
        self._infra_failures += 1
        logger.warning(
            f"[{self.meta.name}/{model_spec}] infrastructure failure "
            f"({self._infra_failures} consecutive): "
            f"{result.api_error[:160] or result.status}"
        )
        if max_infra > 0 and self._infra_failures >= max_infra:
            self._retire_reason = (
                f"{self._infra_failures} consecutive infrastructure failures "
                f"(last: {result.api_error[:200] or result.status})"
            )
            logger.error(
                f"[{self.meta.name}] Retiring swarm — {self._retire_reason}. "
                "Check ANTHROPIC_BASE_URL / GATEWAY_BASE_URL reachability."
            )
            self.cancel_event.set()

    async def _run_solver(self, model_spec: str) -> SolverResult | None:
        solver = self._create_solver(model_spec)
        self.solvers[model_spec] = solver

        try:
            result, final_solver = await self._run_solver_loop(solver, model_spec)
            solver = final_solver
            return result
        except asyncio.CancelledError:
            # Task boundary: convert to a clean CANCELLED result so the swarm's
            # gather never sees a bare CancelledError from a child task.
            logger.info(f"[{self.meta.name}/{model_spec}] cancelled")
            return SolverResult(flag=None, status=CANCELLED, findings_summary="",
                                step_count=0, cost_usd=0.0, log_path="")
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.meta.name}/{model_spec}] Fatal: {e}", exc_info=True)
            return None
        finally:
            await solver.stop()

    async def _run_solver_loop(self, solver, model_spec: str) -> tuple[SolverResult, SolverProtocol]:
        """Inner loop: start → run → (interrupt + conclude) → bump → run → ...

        Terminates on: flag found, external cancel, the configured round cap,
        repeated infrastructure failure, an unrecoverable/broken solver, or
        exhausted quota retries.
        """
        bump_count = 0
        consecutive_errors = 0
        quota_retries = 0
        budget_s = int(getattr(self.settings, "solver_time_budget_s", 0) or 0)
        conclude_s = int(getattr(self.settings, "conclude_timeout_s", 90) or 90)
        max_rounds = int(getattr(self.settings, "max_attempts_per_challenge", 0) or 0)
        result = SolverResult(
            flag=None, status=CANCELLED, findings_summary="",
            step_count=0, cost_usd=0.0, log_path="",
        )
        await solver.start()

        while not self.cancel_event.is_set():
            try:
                # The budget is enforced INSIDE the solver so the turn can be
                # interrupted through the SDK (session preserved, phase 2
                # possible) instead of task-cancelled from out here.
                result = await solver.run_until_done_or_gave_up(budget_s)
            except asyncio.CancelledError:
                result = SolverResult(flag=None, status=CANCELLED, findings_summary="",
                                      step_count=0, cost_usd=0.0, log_path="")
                break
            timed_out = bool(result.interrupted) and not self.cancel_event.is_set()

            if timed_out:
                concluded = await self._harvest_timeout(solver, model_spec, conclude_s)
                result = SolverResult(
                    flag=None, status=GAVE_UP,
                    findings_summary=concluded or "time budget exhausted",
                    step_count=result.step_count, cost_usd=result.cost_usd,
                    log_path=result.log_path, tool_calls=result.tool_calls,
                    interrupted=True, concluded=concluded,
                )
            else:
                await self._account_infra(result, model_spec)
                if self.cancel_event.is_set():
                    break
                if publishable(result.findings_summary) and result.did_work:
                    # Only broadcast rounds that did real work. Errors, API
                    # failures and empty spins stay off the blackboard entirely.
                    self.findings[model_spec] = result.findings_summary
                    await self._publish_harvest(model_spec, result.findings_summary)

            if result.status == FLAG_FOUND:
                self.cancel_event.set()
                self.winner = result
                self.winner_spec = model_spec
                logger.info(
                    f"[{self.meta.name}] Flag found by {model_spec}: {result.flag}"
                )
                return result, solver

            if result.status == CANCELLED or self.cancel_event.is_set():
                break

            # Quota exhaustion: fall back to API-backed Pydantic AI solver
            if result.status == QUOTA_ERROR:
                fallback_spec = _quota_fallback_spec(model_spec)
                if fallback_spec:
                    logger.warning(
                        f"[{self.meta.name}/{model_spec}] Quota exhausted — falling back to {fallback_spec}"
                    )
                    # Hand the live sandbox over so the old solver's stop()
                    # leaves it running, and the fallback reuses it instead of
                    # paying to rebuild the container and re-upload distfiles.
                    existing_sandbox = solver.detach_sandbox()
                    await solver.stop()
                    solver = self._create_pydantic_solver(fallback_spec, sandbox=existing_sandbox,
                                                          owns_sandbox=True)
                    self.solvers[model_spec] = solver
                    if model_spec not in self._all_specs:
                        self._all_specs.append(model_spec)
                    await solver.start()
                    continue
                # No fallback — quota errors are often transient; back off and retry
                quota_retries += 1
                if quota_retries > 5:
                    self._retire_reason = "quota retries exhausted"
                    logger.warning(
                        f"[{self.meta.name}/{model_spec}] {self._retire_reason} — giving up"
                    )
                    break
                if not await self._sleep_or_cancel(self._quota_backoff(quota_retries)):
                    break  # cancelled during backoff
                continue

            if result.status in (ERROR, INFRA_ERROR):
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    self._retire_reason = f"{consecutive_errors} consecutive errors"
                    logger.warning(
                        f"[{self.meta.name}/{model_spec}] {self._retire_reason} — giving up"
                    )
                    break
            else:
                consecutive_errors = 0

            # ---- bump path: GAVE_UP, recoverable ERROR, or time budget hit ----
            bump_count += 1
            self._total_bumps += 1

            if max_rounds > 0 and bump_count >= max_rounds:
                self._retire_reason = f"round cap reached ({max_rounds})"
                logger.warning(
                    f"[{self.meta.name}/{model_spec}] {self._retire_reason} — retiring solver"
                )
                break

            self._maybe_escalate()

            # Cooldown between bumps — check cancellation during wait
            if not await self._sleep_or_cancel(self._bump_cooldown(bump_count)):
                break  # cancelled during cooldown

            if timed_out:
                reason = (
                    f"Your attempt hit the {budget_s}s wall-clock time budget and was "
                    "interrupted. Your confirmed findings were harvested. Work in "
                    "smaller increments, and pick a FASTER, more direct strategy."
                )
            else:
                reason = "Your previous attempt did not find the flag."

            # Fresh-restart mode: distill this round, reset the session, and
            # restart clean carrying only the distilled state (no history bloat).
            if int(getattr(self.settings, "fresh_restart_rounds", 5) or 0) > 0:
                await self._fresh_restart(solver, model_spec, result, reason, bump_count)
            else:
                solver.bump(await self._build_bump_payload(model_spec, reason, bump_count))
            logger.info(
                f"[{self.meta.name}/{model_spec}] Bumped ({bump_count}), resuming"
            )

        return result, solver

    def _bump_cooldown(self, bump_count: int) -> float:
        """Backoff before the next round: base + step*round, capped at max."""
        base = float(getattr(self.settings, "bump_cooldown_base_s", 5) or 0)
        step = float(getattr(self.settings, "bump_cooldown_step_s", 10) or 0)
        cap = float(getattr(self.settings, "bump_cooldown_max_s", 120) or 0)
        wait = base + step * bump_count
        return min(wait, cap) if cap > 0 else wait

    def _quota_backoff(self, retries: int) -> float:
        base = float(getattr(self.settings, "quota_backoff_base_s", 60) or 0)
        cap = float(getattr(self.settings, "quota_backoff_max_s", 300) or 0)
        wait = base * retries
        return min(wait, cap) if cap > 0 else wait

    async def _sleep_or_cancel(self, seconds: float) -> bool:
        """Wait `seconds`, returning False if the swarm was cancelled midway.

        Always yields at least once, even for a zero cooldown: with no yield the
        round loop can starve the event loop entirely, so cancellation, the
        patrol and the dashboard would never get to run.
        """
        if seconds <= 0:
            await asyncio.sleep(0)
            return not self.cancel_event.is_set()
        try:
            await asyncio.wait_for(self.cancel_event.wait(), timeout=seconds)
            return False  # the event fired → cancelled
        except TimeoutError:
            return not self.cancel_event.is_set()

    async def _wind_down(self, pending: set[asyncio.Task]) -> None:
        """Let siblings finish on their own, cancelling only stragglers.

        Solvers already watch `cancel_event` and break out of their stream loop
        within a couple of seconds, which lets each one run its own teardown
        (container + workspace) uninterrupted. Cancelling the tasks immediately —
        the previous behaviour — interrupted teardown mid-way and orphaned a
        container and a workspace directory per sibling on every solved
        challenge. The grace period is what makes that safe; the cancel is only a
        last resort for a solver wedged inside a single long tool call.
        """
        if not pending:
            return
        grace = float(getattr(self.settings, "solver_teardown_grace_s", 60) or 60)
        _done, still = await asyncio.wait(pending, timeout=grace)
        if still:
            logger.warning(
                f"[{self.meta.name}] {len(still)} solver(s) did not wind down within "
                f"{grace:.0f}s — cancelling (teardown is shielded, so cleanup still runs)"
            )
            for t in still:
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

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
                    if spec not in self._all_specs:
                        self._all_specs.append(spec)
                    pending.add(asyncio.create_task(
                        self._run_escalated_solver(spec), name=f"solver-escalate-{spec}"
                    ))

                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        logger.error(f"[{self.meta.name}] {task.get_name()} failed: {exc}",
                                     exc_info=exc)
                        continue
                    result = task.result()
                    if result and result.status == FLAG_FOUND:
                        self.cancel_event.set()
                        await self._wind_down(pending)
                        return result

                tasks = set(pending)

            self.cancel_event.set()
            return self.winner
        except asyncio.CancelledError:
            # External cancellation (e.g. offline benchmark time limit) —
            # make sure child solver tasks don't linger as orphans.
            self.cancel_event.set()
            await self._wind_down(tasks)
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{self.meta.name}] Swarm error: {e}", exc_info=True)
            self.cancel_event.set()
            await self._wind_down(tasks)
            return None

    def kill(self, reason: str = "") -> None:
        """Cancel all agents for this challenge.

        `reason` is recorded so the operator can tell afterwards WHY a swarm
        stopped. Without it an externally killed swarm vanished from the
        dashboard with no winner, no retire_reason and no log line —
        indistinguishable from a crash.
        """
        if reason and not self._retire_reason:
            self._retire_reason = reason
        if not self.cancel_event.is_set():
            logger.info(f"[{self.meta.name}] Swarm killed: {reason or '(no reason given)'}")
        self.cancel_event.set()

    @property
    def retire_reason(self) -> str:
        return self._retire_reason

    def get_status(self) -> dict:
        """Get per-agent progress and findings (escalated solvers included)."""
        return {
            "challenge": self.meta.name,
            "cancelled": self.cancel_event.is_set(),
            "winner": self.winner.flag if (self.winner and self.winner.flag) else None,
            "winner_spec": self.winner_spec,
            "bumps": self._total_bumps,
            "retire_reason": self._retire_reason,
            "infra_failures": self._infra_failures,
            "agents": {
                spec: {
                    "findings": self.findings.get(spec, ""),
                    "tool_calls": getattr(self.solvers.get(spec), "_tool_calls", 0),
                    "status": (
                        "won" if (self.winner and self.winner_spec == spec)
                        else "running" if (spec in self.solvers and not self.cancel_event.is_set())
                        else "finished"
                    ),
                }
                for spec in self._all_specs
            },
        }
