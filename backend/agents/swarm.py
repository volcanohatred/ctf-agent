"""ChallengeSwarm — Parallel solvers racing on one challenge."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.agents.solver import Solver
from backend.agents.verifier_agent import VerificationRunner
from backend.approaches import build_approach_queue
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.dedup import dedup_lines, new_lines, normalize
from backend.message_bus import (
    KIND_CLAIM,
    KIND_EXCLUDED,
    KIND_FACT,
    KIND_TAGS,
    KIND_VERIFIED,
    ChallengeMessageBus,
)
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
from backend.verifier import VERDICT_FORCE, ClaimQueue, PendingClaim

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


# How the previous round ended, as far as the assigned handbook is concerned.
#
# The rotation used to have one outcome — "the solver is back for a bump, so that
# approach failed" — which is wrong three ways over. A solver comes back for a bump
# when the wall clock cut it off mid-work, when its context overflowed, when it
# honestly said still_working, and on an infrastructure error. None of those is
# evidence against the technique, and with SOLVER_TIME_BUDGET_S=1500 on a hard
# challenge a solver interrupted 12 times had 12 handbooks recorded as exhausted.
# That ledger feeds the bump payload ("已试且失败，勿重复"), the escalation briefing,
# and the Strategist — so the Strategist was told twelve techniques had been tried
# and failed when in fact the same one had been cut off twelve times, and pivoted
# the whole swarm away from it.
APPROACH_FAILED = "failed"          # did real work on it and came back empty
APPROACH_INTERRUPTED = "interrupted"  # round was cut off — never got a fair trial
APPROACH_UNENGAGED = "unengaged"    # handed the handbook and never opened it

# Consecutive interrupts tolerated before rotating away anyway. Keeping the
# direction is right — an interrupt is not a refutation — but unbounded it would
# retry a genuinely wrong handbook until the round cap.
MAX_SAME_APPROACH_ROUNDS = 2


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
    # Consecutive rounds this solver was cut off while holding the same handbook.
    # Bounds how long an interrupted direction is kept before rotating anyway.
    _approach_interrupts: dict[str, int] = field(init=False, default_factory=dict)
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
    # Authorship ledger for asserted fact lines: normalized line → the specs that
    # reported it. A line one model asserted is a CLAIM; the first time a
    # *different* spec reports the same line it becomes a corroborated FACT. This
    # is the gate that stops one model's hallucination from becoming the swarm's
    # ground truth — the blackboard is injected into every restart as such.
    _claim_sources: dict[str, set[str]] = field(init=False, default_factory=dict)
    # Circuit breaker state. `_last_round_novel` is set by _publish_harvest and
    # consumed once per bump; `_stale_rounds` counts consecutive rounds in which a
    # solver added NOTHING new to the team (every dead end it reported was already
    # recorded, every fact already asserted). That is the signature of a solver
    # micro-tweaking one payload, which no per-command loop detector can see
    # because each tweak is a genuinely different command.
    _last_round_novel: dict[str, bool] = field(init=False, default_factory=dict)
    _stale_rounds: dict[str, int] = field(init=False, default_factory=dict)
    _stale_breaks: int = field(init=False, default=0)
    # Strategist: fires at most once per `strategist_every_bumps` swarm-wide bumps.
    # The generation counter plus the lock is what keeps four solvers bumping
    # concurrently from paying for four consultations.
    _strategist_gen: int = field(init=False, default=0)
    _strategist_pivots: int = field(init=False, default=0)
    _strategist_failures: int = field(init=False, default=0)
    _strategist_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)
    # Active verification. Claims awaiting a probe, the runner that drains them,
    # and the audit counters. The runner is created lazily and started on the
    # first queued claim so a challenge solved in 40s never pays for a sandbox.
    _claim_queue: ClaimQueue = field(init=False, default_factory=ClaimQueue)
    _verifier: VerificationRunner | None = field(init=False, default=None)
    _probe_verified: int = field(init=False, default=0)
    _probe_refuted: int = field(init=False, default=0)
    _contradictions: int = field(init=False, default=0)
    # Swarm-wide distilled state, recomputed at most once per bump generation and
    # shared by every solver restarting in that generation.
    _team_digest: str = field(init=False, default="")
    _team_digest_gen: int = field(init=False, default=-1)
    _digest_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self._all_specs = list(self.model_specs)
        self.skill_router = SkillRouter(skills_dir=getattr(self.settings, "skills_dir", "") or "")
        self._skill_pack_ids = []
        # Master switch. Gating here rather than in _skill_hint means triage never
        # runs and the rotation queue is never built, so both injection and rotation
        # are off from a single flag.
        #
        # This deliberately does NOT stop solvers being told that /skills exists —
        # that comes from settings.skills_dir in the solver's own prompt builder and
        # describes the passive mount, which is the one mechanism with no plausible
        # downside. Turning off the push without turning off the availability is
        # exactly the configuration worth measuring.
        if not bool(getattr(self.settings, "skill_hints", True)):
            logger.info(
                f"[{self.meta.name}] SKILL_HINTS=false — no triage, no hint injection, "
                "no approach rotation (the library stays mounted for solvers to read "
                "on their own)"
            )
            return
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

    def create_verifier(self, model_spec: str, role_prompt: str):
        """Build a solver instance wearing the Verifier role.

        Reuses the real solver backends rather than a second implementation, so the
        verifier gets sandbox redirection, loop detection, tracing, the wall-clock
        budget and shielded teardown for free. Two things are deliberately NOT
        wired up: no submit_fn and no message_bus, because a verifier must not be
        able to end the swarm on a flag it found by accident, and it does not post
        to the board itself — the swarm publishes its verdicts.
        """
        provider = provider_from_spec(model_spec)
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
                no_submit=True,
                role_prompt=role_prompt,
                conclude_prompt=VERDICT_FORCE,
            )
        return Solver(
            model_spec=model_spec,
            challenge_dir=self.challenge_dir,
            meta=self.meta,
            ctfd=self.ctfd,
            cost_tracker=self.cost_tracker,
            settings=self.settings,
            cancel_event=self.cancel_event,
            role_prompt=role_prompt,
            conclude_prompt=VERDICT_FORCE,
        )

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
            tag = KIND_TAGS.get(f.kind, KIND_TAGS[KIND_FACT])
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
        """Escalating knowledge-base hint, activated after repeated failures.

        `0` disables injection entirely. That is a deliberate semantic choice and
        matches the sibling knob ESCALATE_AFTER_BUMPS, whose 0 also means "off" —
        the old behaviour was the opposite, where the gate `bump_count < threshold`
        could never fire at 0 and the level formula `1 + (bump - 0)` started at L2,
        so 0 meant "inject the heaviest tier from the very first retry". Nobody
        wants that, and it made 0 the one value that could not express the most
        common intent.

        Rotation is NOT gated here: SKILL_AFTER_BUMPS=0 with
        SKILL_ROTATE_APPROACHES=true is a real configuration (assign a direction,
        inject no handbook text), and it is one of the arms worth measuring.
        """
        # Checked explicitly rather than relying on __post_init__ having left
        # _skill_pack_ids empty: the switch that measures this mechanism has to hold
        # even if pack ids are populated from somewhere else later.
        if not bool(getattr(self.settings, "skill_hints", True)):
            return ""
        threshold = int(getattr(self.settings, "skill_after_bumps", 1) or 0)
        if threshold <= 0:
            return ""
        if not self._skill_pack_ids or bump_count < threshold:
            return ""
        level = min(1 + (bump_count - threshold), 3)
        keywords = [self.meta.category, *self.meta.tags, *self.meta.name.replace("-", " ").split()]
        try:
            return self.skill_router.render_hint(self._skill_pack_ids, level=level, keywords=keywords)
        except Exception as e:  # noqa: BLE001 — hints are best-effort
            logger.warning(f"[{self.meta.name}] Skill hint render failed: {e}")
            return ""

    def _next_approach(self, model_spec: str, outcome: str = APPROACH_FAILED):
        """Assign this solver's next technique, recording how the last one ended.

        Only a genuine failure is written to the dead-end ledger:

          failed      — the solver did real work on it and came back empty. The
                        technique was tried and did not pan out; rotate away.
          interrupted — the round was cut off (wall clock, context overflow, infra).
                        The technique never got a fair trial, so it is NOT recorded
                        as failed and the solver KEEPS it, up to
                        MAX_SAME_APPROACH_ROUNDS consecutive interrupts.
          unengaged   — the solver was handed the handbook and never opened it.
                        Rotating is fine, but recording it as a refuted technique
                        would be a lie, and the Strategist reads this ledger.

        Returns None when rotation is disabled or the queue is exhausted; callers
        then emit no direction and the solver keeps its own.
        """
        # Both switches are checked here, not only at construction. __post_init__
        # already skips building the queue when SKILL_HINTS is false, but relying on
        # construction order for a safety property is how it gets undone later by
        # someone who populates the queue from elsewhere. The imperative
        # "本轮专攻新方向…先读再做" is the single most suspected source of harm for
        # weak models, so the switch that measures it must hold unconditionally.
        if not bool(getattr(self.settings, "skill_hints", True)):
            return None
        if not bool(getattr(self.settings, "skill_rotate_approaches", True)):
            return None

        prev_path = self._assigned_approach.get(model_spec)
        if prev_path:
            prev = next((a for a in self._approach_queue if a.path == prev_path), None)
            if outcome == APPROACH_INTERRUPTED:
                held = self._approach_interrupts.get(model_spec, 0) + 1
                self._approach_interrupts[model_spec] = held
                if prev is not None and held <= MAX_SAME_APPROACH_ROUNDS:
                    logger.debug(
                        f"[{self.meta.name}/{model_spec}] round was cut off, not a "
                        f"failure — keeping approach {prev.title} ({held}/"
                        f"{MAX_SAME_APPROACH_ROUNDS} consecutive interrupts)"
                    )
                    return prev
            else:
                self._approach_interrupts[model_spec] = 0

            if outcome == APPROACH_FAILED and prev is not None:
                if prev.path not in self._failed_approach_paths:
                    self._failed_approach_paths.append(prev.path)
                    if prev.title not in self._failed_approaches:
                        self._failed_approaches.append(prev.title)
            elif prev is not None and prev.path not in self._failed_approach_paths:
                # Interrupted past the bound, or never engaged: move on, but do not
                # add it to the ledger — nothing established that it does not work.
                logger.debug(
                    f"[{self.meta.name}/{model_spec}] rotating off {prev.title} "
                    f"without recording it as failed (outcome={outcome})"
                )

        for a in self._approach_queue:
            if a.path not in self._failed_approach_paths and a.path != prev_path:
                self._assigned_approach[model_spec] = a.path
                return a
        return None

    def _approach_outcome(self, solver, model_spec: str, result: SolverResult,
                          timed_out: bool, overflowed: bool) -> str:
        """Classify how the last round ended, as far as the assigned handbook goes.

        Order matters: an interrupted or infra round is checked first, because those
        tell us nothing about the technique. Only a round that ran to completion with
        real work behind it can be called a failure — and even then only if the solver
        actually opened the handbook it was handed.
        """
        if timed_out or overflowed or result.is_infra_failure or not result.did_work:
            return APPROACH_INTERRUPTED

        assigned = self._assigned_approach.get(model_spec)
        if not assigned:
            return APPROACH_FAILED
        # Did it ever touch the file? Handing a weak model a handbook path plus
        # "先读再做" and then recording that handbook as exhausted when the model
        # never opened it is how the ledger fills up with techniques nobody tried.
        path = getattr(getattr(solver, "tracer", None), "path", "") or ""
        if path:
            try:
                from backend.trace_read import tool_args_mention

                if not tool_args_mention(path, assigned):
                    return APPROACH_UNENGAGED
            except OSError as e:
                logger.debug(f"[{self.meta.name}/{model_spec}] could not check handbook "
                             f"engagement: {e}")
        return APPROACH_FAILED

    async def _build_bump_payload(self, model_spec: str, reason: str, bump_count: int,
                                  outcome: str = APPROACH_FAILED) -> str:
        """Lean rotation bump: short reason + next untried technique + skill hint
        only from the configured failure threshold onward."""
        parts = [reason]

        # Resolve the rotation FIRST, then render the ledger. The other order made
        # the message permanently one step behind itself: it announced "本轮专攻新方向 B"
        # while the 已试且失败 list still did not contain the A that had just failed,
        # so the solver was told to switch away from a direction nothing said was dead.
        nxt = self._next_approach(model_spec, outcome)

        if self._failed_approaches:
            failed = "、".join(self._failed_approaches)
            parts.append(f"\n🚫 已试且失败（勿重复）：{failed}")

        if nxt:
            if outcome == APPROACH_INTERRUPTED:
                # It was cut off, not refuted. Telling a solver to abandon a
                # direction it was making progress on is how an interrupt becomes a
                # permanent detour.
                parts.append(
                    f"\n🎯 继续上一轮的方向：【{nxt.title}】（手册 {nxt.path}）——"
                    "上一轮是被时间打断的，不是这个方向错了。这轮用更小的步长推进，"
                    "把中间结果写进 /challenge/workspace/ 以便再次被打断时能接上。"
                )
            else:
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

    async def _run_strategist(self, force: bool = False) -> str:
        """Consult the strategist and broadcast its verdict.

        Fires at most once per `strategist_every_bumps` swarm-wide bumps. `force`
        (the stale-round circuit breaker) bypasses that interval — a solver that
        has stopped teaching the team anything is precisely when a step-back is
        overdue, and waiting for the next tick is what let a swarm grind the same
        payload for hours.

        Concurrency: several solvers bump at once. The generation check sits
        inside the lock so only the first one pays for the call; the others see
        the advanced generation and reuse the verdict already on the board.

        Returns "" when the strategist is disabled or produced nothing worth
        broadcasting. A failed gateway call is silent by design — an invented
        PIVOT would yank every solver off a working line of attack, which is
        strictly worse than no opinion at all.
        """
        every = int(getattr(self.settings, "strategist_every_bumps", 0) or 0)
        if every <= 0:
            return ""

        async with self._strategist_lock:
            if self.cancel_event.is_set():
                return ""
            gen = self._total_bumps // every
            if not force and gen <= self._strategist_gen:
                return ""
            if not force:
                self._strategist_gen = gen

            from backend.strategist import consult_strategist

            logger.info(
                f"[{self.meta.name}] Strategist consult "
                f"({'FORCED by stale-round breaker' if force else f'bump {self._total_bumps}'})"
            )
            try:
                verdict = await consult_strategist(
                    name=self.meta.name,
                    category=self.meta.category,
                    value=self.meta.value,
                    tags=self.meta.tags,
                    connection=self.meta.connection_info,
                    distfiles=list_distfiles(self.challenge_dir),
                    description=self.meta.description,
                    dead_ends=list(self._seen_dead_ends),
                    failed_approaches=list(self._failed_approaches),
                    bumps=self._total_bumps,
                    settings=self.settings,
                    cost_tracker=self.cost_tracker,
                    agent_name=f"{self.meta.name}/strategist",
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — the strategist is an enhancement
                logger.warning(f"[{self.meta.name}] strategist failed: {e}")
                return ""

            if not verdict.available:
                # A dead strategist and a strategist that said "carry on" must not
                # look the same. Measured against a live gateway, the original 60s
                # timeout failed on every single call while logging at debug level,
                # so the mechanism was silently dead and the operator had no way to
                # know the strategy was never being reviewed.
                self._strategist_failures += 1
                logger.warning(
                    f"[{self.meta.name}] Strategist produced NO verdict "
                    f"({verdict.reason}) — failure #{self._strategist_failures}. "
                    "The strategy was not reviewed this round; if this repeats, check "
                    "STRATEGIST_TIMEOUT_S / STRATEGIST_MAX_TOKENS / the model name."
                )
                return ""

            text = verdict.broadcast_text
            await self.message_bus.broadcast(text, source="strategist")
            if verdict.pivot:
                self._strategist_pivots += 1
                logger.warning(f"[{self.meta.name}] Strategist PIVOT → {verdict.text[:200]}")
                if self.coordinator_inbox is not None:
                    # Best-effort: a coordinator notification must never be able to
                    # kill the solver round that asked for it.
                    try:
                        self.coordinator_inbox.put_nowait(
                            f"STRATEGIST PIVOT on '{self.meta.name}' after {self._total_bumps} "
                            f"bumps: {verdict.text[:300]} — broadcast to all solvers. Do not "
                            "steer them back to the excluded directions."
                        )
                    except asyncio.QueueFull:
                        logger.warning(
                            f"[{self.meta.name}] coordinator inbox full — pivot notice dropped"
                        )
            else:
                logger.info(f"[{self.meta.name}] Strategist CONTINUE ({verdict.text[:120]})")
            return text

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

    # Bound on the authorship ledger. Keys are short normalized lines, so this is
    # a few hundred KB at worst; the cap exists so an unlimited-rounds coordinator
    # run cannot grow it for the lifetime of the process. Oldest keys are evicted,
    # which can only ever demote a very old corroborated fact back to a claim.
    MAX_CLAIM_KEYS = 2000

    def _classify_claims(
        self, model_spec: str, facts: list[str]
    ) -> tuple[list[str], list[str], bool]:
        """Split asserted fact lines into corroborated vs single-source.

        Returns `(corroborated, claims, taught_something_new)`.

        A line is corroborated the first time a DIFFERENT spec reports it — two
        models independently reaching the same conclusion is evidence, one model
        asserting it is not. Two rules make that hold:

          * the same spec re-asserting its own line (routine after a session wipe)
            is skipped, not counted — otherwise a model could promote its own
            hallucination simply by surviving a restart and repeating it;
          * a line already corroborated by an earlier pair is skipped, so a third
            agreement does not post the same fact to the board a second time.
        """
        corroborate = bool(getattr(self.settings, "corroborate_facts", True))
        corroborated: list[str] = []
        claims: list[str] = []
        novel = False

        for line in dedup_lines(facts):
            key = normalize(line)
            if not key:
                continue
            sources = self._claim_sources.get(key)
            first_time = sources is None
            if first_time:
                sources = set()
                self._claim_sources[key] = sources
                novel = True

            if model_spec in sources:
                continue
            if not first_time and len(sources) >= 2:
                sources.add(model_spec)
                continue

            if first_time and corroborate:
                claims.append(line)
            else:
                # Either the trust gate is off (post as a fact, as before) or a
                # different solver has now independently reported this line.
                corroborated.append(line)
                if not first_time:
                    novel = True
            sources.add(model_spec)

        while len(self._claim_sources) > self.MAX_CLAIM_KEYS:
            self._claim_sources.pop(next(iter(self._claim_sources)), None)
        return corroborated, claims, novel

    async def _publish_harvest(self, model_spec: str, text: str) -> None:
        """Publish a round's output, routing tagged lines to the right kind.

        Untagged output is published as a CLAIM, never as an excluded dead end
        (labeling a whole failed round "do NOT retry this" tells siblings that
        mostly-confirmed observations were ruled out) and never as a corroborated
        fact either (one model's prose is not ground truth).

        Dead ends are additionally de-duplicated structurally, before the LLM
        ever sees them: a restated dead end must not grow the "do not retry"
        list, and the ledger is what steers every subsequent round.

        Records whether the round taught the team anything new — the input to the
        stale-round circuit breaker in `_run_solver_loop`.
        """
        if not publishable(text):
            return
        excluded, facts = split_tagged(text)
        posted = False
        novel = False
        # Everything this round asserts, collected for the contradiction audit.
        statements: list[str] = []
        if excluded:
            fresh = new_lines(excluded, self._seen_dead_ends)
            if fresh:
                self._seen_dead_ends.extend(fresh)
                body = "\n".join(f"- {e}" for e in fresh)
                await self._post_finding(model_spec, body, KIND_EXCLUDED)
                statements.extend(fresh)
                posted = True
                novel = True
            else:
                logger.debug(
                    f"[{self.meta.name}/{model_spec}] {len(excluded)} dead end(s) "
                    "already recorded — not republishing"
                )
        if facts:
            # Untagged prose is classified as ONE unit rather than line by line:
            # it is a narrative, not a set of discrete assertions, and splitting
            # it would put a dozen half-sentences on the board as separate claims.
            # It still goes through the same ledger, so a solver re-reporting the
            # same summary after a session wipe cannot grow the board with it.
            tagged = any(_match_tag(line.strip()) for line in text.splitlines())
            units = facts if tagged else ["\n".join(f.strip() for f in facts if f.strip())]
            corroborated, claims, fact_novel = self._classify_claims(model_spec, units)
            novel = novel or fact_novel
            statements.extend(corroborated)
            statements.extend(claims)
            if corroborated:
                keys = {normalize(line) for line in corroborated}
                # Promote the ORIGINAL claim too, so the board never carries the
                # same statement at two trust levels.
                promoted = await self.message_bus.promote_claims(keys)
                if promoted:
                    logger.info(
                        f"[{self.meta.name}] {promoted} single-source claim(s) "
                        f"corroborated by {model_spec} → promoted to fact"
                    )
                for line in corroborated:
                    await self._post_finding(model_spec, line, KIND_FACT)
                posted = True
            # One finding per claim: promotion flips a whole finding, so a
            # multi-line blob could never be promoted without dragging its
            # still-unverified neighbours along with it.
            for line in claims:
                await self._post_finding(model_spec, line, KIND_CLAIM)
                posted = True
                self._enqueue_for_verification(line, model_spec)
            if not corroborated and not claims:
                logger.debug(
                    f"[{self.meta.name}/{model_spec}] every reported finding was "
                    "already on the board — nothing to publish"
                )
        if not posted and not excluded and not facts:
            # Publishable but structurally empty. Posting it would put noise on the
            # board; staying silent is what the "nothing harvestable" path already
            # does for a failed conclude.
            logger.debug(f"[{self.meta.name}/{model_spec}] harvest had no content")
        self._last_round_novel[model_spec] = novel
        await self._audit_contradictions(model_spec, statements)

    async def _audit_contradictions(self, model_spec: str, new_statements: list[str]) -> None:
        """Check what this round just wrote against what the board already says.

        Two agents holding mutually exclusive beliefs both look productive from the
        inside; only a comparison across the whole board surfaces it. Detection is
        worthless unless it triggers a test, so each conflict is broadcast AND both
        sides are pushed to the front of the verification queue — that is the A/B
        confrontation, and the probe is what settles it rather than another opinion.
        """
        if not new_statements or not bool(getattr(self.settings, "audit_contradictions", True)):
            return
        if self.cancel_event.is_set():
            return

        from backend.audit import find_contradictions

        existing = [
            (f.kind, f.summary or f.content)
            for f in self.message_bus.findings
            if f.kind in (KIND_VERIFIED, KIND_FACT, KIND_CLAIM, KIND_EXCLUDED)
        ]
        try:
            found = await find_contradictions(
                new_statements=new_statements,
                existing=existing,
                settings=self.settings,
                cost_tracker=self.cost_tracker,
                agent_name=f"{self.meta.name}/audit",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — the audit must never break a round
            logger.debug(f"[{self.meta.name}] contradiction audit failed: {e}")
            return

        for conflict in found:
            self._contradictions += 1
            logger.warning(
                f"[{self.meta.name}] CONTRADICTION between {model_spec}'s report and the "
                f"board: {conflict.a[:80]!r} vs {conflict.b[:80]!r} ({conflict.why[:100]})"
            )
            # Queue both sides BEFORE broadcasting. The notice quotes both
            # statements, so once it is on the board it becomes the longest
            # containment match for either fragment — resolving the originals first
            # keeps the probe pointed at what an agent actually said.
            for side, who in ((conflict.a, model_spec), (conflict.b, "board")):
                self._queue_contradiction_side(side, who)
            await self.message_bus.broadcast(conflict.broadcast_text, source="audit")
            if self.coordinator_inbox is not None:
                try:
                    self.coordinator_inbox.put_nowait(
                        f"CONTRADICTION on '{self.meta.name}': {conflict.a[:150]} "
                        f"VS {conflict.b[:150]} ({conflict.why[:150]}). Both sides queued "
                        "for a probe. Do not tell solvers which one is right — nobody "
                        "knows yet."
                    )
                except asyncio.QueueFull:
                    logger.debug(f"[{self.meta.name}] inbox full — contradiction notice dropped")

    def _queue_contradiction_side(self, statement: str, source: str) -> None:
        """Push one side of a conflict to the front of the verification queue.

        The auditor's text may be a paraphrase, so it is matched back to the actual
        board entry first: probing the wording an agent really used is what makes
        the result recognisable to it, and reuses the existing queue key so the same
        claim is not probed twice.
        """
        if not bool(getattr(self.settings, "verify_claims", True)):
            return
        text = self._board_original(statement) or (statement or "").strip()
        if not text:
            return
        key = normalize(text)
        if not key:
            return
        if not self._claim_queue.push(PendingClaim(
            content=text[:1500], source=source, key=key, priority=10,
            enqueued_at=time.time(),
        )):
            # Already queued — a conflict is the strongest possible reason to test
            # it now rather than in FIFO order.
            self._claim_queue.bump_priority(key)
        self._ensure_verifier()

    def _board_original(self, paraphrase: str) -> str:
        """Find the board entry an auditor's quote most likely refers to.

        Broadcasts and help requests are excluded from the search: a contradiction
        notice quotes BOTH sides of the conflict, so once it is on the board it is
        the longest containment match for either fragment. Matching it would queue
        the notice's own text for probing instead of the statement that caused it.
        """
        needle = normalize(paraphrase or "")
        if not needle:
            return ""
        candidates = [
            f for f in self.message_bus.findings
            if f.kind in (KIND_VERIFIED, KIND_FACT, KIND_CLAIM, KIND_EXCLUDED)
        ]
        for f in candidates:
            if normalize(f.content) == needle:
                return f.content
        # Containment, longest first: the auditor routinely quotes a fragment of a
        # longer entry, and the longest match is the most specific one.
        best = ""
        for f in candidates:
            hay = normalize(f.content)
            if (needle and hay and (needle in hay or hay in needle)
                    and len(f.content) > len(best)):
                best = f.content
        return best

    # ------------------------------------------------------------------
    # Active verification
    # ------------------------------------------------------------------

    # Below this a "claim" is too short to be a testable assertion ("422", "no
    # flag") and a probe would just burn a sandbox round on noise.
    MIN_TESTABLE_CLAIM_CHARS = 16

    def _enqueue_for_verification(self, claim: str, source: str) -> None:
        """Queue a single-source claim for an active probe.

        Multi-line untagged prose is skipped: it is a narrative, not one testable
        assertion, and a verifier handed a paragraph tests whichever sentence it
        happens to fixate on while appearing to have checked the whole thing.
        """
        if not bool(getattr(self.settings, "verify_claims", True)):
            return
        if self.cancel_event.is_set():
            return
        text = (claim or "").strip()
        if not text or "\n" in text or len(text) < self.MIN_TESTABLE_CLAIM_CHARS:
            return
        budget = int(getattr(self.settings, "verifier_max_probes", 0) or 0)
        if budget > 0 and self._probe_verified + self._probe_refuted >= budget:
            return
        key = normalize(text)
        if not key:
            return
        if self._claim_queue.push(PendingClaim(content=text, source=source, key=key,
                                              enqueued_at=time.time())):
            self._ensure_verifier()

    def _ensure_verifier(self) -> None:
        """Create and start the runner on first queued claim.

        Lazy on purpose: a challenge solved in 40 seconds, or one whose claims are
        all untestable, never pays for a verifier container.
        """
        if self.cancel_event.is_set():
            return
        if self._verifier is None:
            self._verifier = VerificationRunner(self, self._claim_queue, self._on_verdict)
        self._verifier.ensure_started()

    async def _on_verdict(self, verdict) -> None:
        """Apply one probe result to the board.

        The refutation path does three things, and all three are needed: move the
        claim into the dead-end ledger (so it steers every later round), record the
        evidence (so nobody re-tests it), and broadcast a prune notice (so RUNNING
        solvers stop within a few tool calls instead of at their next restart —
        that difference is what turns a wasted hour into a wasted minute).
        """
        from backend.verifier import INCONCLUSIVE, VERIFIED

        key = normalize(verdict.claim)
        if verdict.verdict == INCONCLUSIVE:
            return

        if verdict.verdict == VERIFIED:
            self._probe_verified += 1
            moved = await self.message_bus.verify_claims({key}, note=verdict.evidence)
            if not moved:
                # The original claim was trimmed or already reclassified; the
                # evidence is still worth having on the board.
                await self._post_finding("verifier", verdict.board_text, KIND_VERIFIED)
            logger.info(f"[{self.meta.name}] ✅ probe-verified: {verdict.claim[:90]!r}")
            return

        self._probe_refuted += 1
        moved = await self.message_bus.refute_claims({key}, note=verdict.evidence)
        line = verdict.board_text
        if line not in self._seen_dead_ends:
            self._seen_dead_ends.append(line)
        if not moved:
            await self._post_finding("verifier", line, KIND_EXCLUDED)
        # Invalidate the team digest: it was computed while this premise was still
        # believed, and every restarting solver would otherwise be re-seeded with it.
        self._team_digest_gen = -1
        self._team_digest = ""
        await self.message_bus.broadcast(
            "⛔ **剪枝通知（探针实测，非推测）**：以下前提已被证伪，所有基于它的工作立即停止。\n"
            f"被证伪的声称：「{verdict.claim}」\n"
            f"实测证据：{verdict.evidence}\n"
            "如果你正在基于这个前提构造 payload 或推导链，现在换方向；已经推出来的结论"
            "一并作废。",
            source="verifier",
        )
        logger.warning(
            f"[{self.meta.name}] ⛔ REFUTED and pruned: {verdict.claim[:90]!r} "
            f"({verdict.evidence[:120]})"
        )
        if self.coordinator_inbox is not None:
            try:
                self.coordinator_inbox.put_nowait(
                    f"VERIFIER REFUTED a premise on '{self.meta.name}': "
                    f"{verdict.claim[:200]} — evidence: {verdict.evidence[:200]}. "
                    "Prune notice broadcast; do not steer solvers back to it."
                )
            except asyncio.QueueFull:
                logger.warning(f"[{self.meta.name}] coordinator inbox full — refutation dropped")

    async def _fresh_restart(self, solver, model_spec: str, result: SolverResult,
                             reason: str, bump_count: int,
                             outcome: str = APPROACH_FAILED) -> None:
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
        solver.bump(await self._build_restart_payload(model_spec, reason, bump_count, outcome))

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

    async def _build_restart_payload(self, model_spec: str, reason: str, bump_count: int,
                                     outcome: str = APPROACH_FAILED) -> str:
        """The seed message for a freshly restarted session.

        `reason` explains WHY the previous round ended — time budget exhausted,
        context window exceeded, or simply no flag. It used to be accepted and
        then ignored, so a restarted solver was never told it had timed out or
        blown its context, and could not adjust its behaviour accordingly.

        `outcome` decides whether the handbook it was holding counts as tried-and-
        failed or as interrupted. A session wipe makes this matter more, not less:
        the restarted solver no longer remembers the work it was cut off in the
        middle of, so being told "that direction failed, here is a new one" loses
        it twice over.
        """
        journal = self._journals.get(model_spec, [])
        state_txt = "\n".join(f"{i+1}. {j}" for i, j in enumerate(journal)) or "(暂无)"

        parts = [
            f"这是一次全新的尝试（第 {bump_count + 1} 轮）。",
        ]
        if reason:
            parts += ["【上一轮为何结束】", reason, ""]
        parts += [
            "你之前的探索状态已被蒸馏如下，不要重复已失败的方向，直接基于它继续：",
            "",
            "【探索状态】",
            state_txt,
        ]
        # Rotation: assign the next untried technique
        nxt = self._next_approach(model_spec, outcome)
        if nxt:
            if outcome == APPROACH_INTERRUPTED:
                parts.append(
                    f"\n🎯 接着上一轮的方向做：【{nxt.title}】（手册 {nxt.path}）——"
                    "上一轮是被时间/上下文打断的，**不是这个方向被否证了**。"
                    "你的会话已被重置，所以先看上面【探索状态】找回进度，"
                    "然后用更小的步长继续，中间结果写进 /challenge/workspace/。"
                )
            elif outcome == APPROACH_UNENGAGED:
                parts.append(
                    f"\n🎯 本轮主攻方向：【{nxt.title}】（手册 {nxt.path}）——"
                    "上一本手册你并没有真的打开过，所以它不算被否证。"
                    "这本如果你觉得与当前情况无关，可以跳过它、直接说明理由并按自己的判断走。"
                )
            else:
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
            # Per-round breaker state, reset up front so a round that never
            # reaches _publish_harvest cannot inherit the previous round's
            # verdict and so the `stale`/`stale_break` reads below are always
            # bound on the first iteration.
            self._last_round_novel[model_spec] = False
            stale = 0
            stale_break = False
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

            # Context overflow is recoverable and the solver has already dropped
            # its history by the time we get here, so it must not be counted
            # alongside genuine errors — otherwise five overflows would retire a
            # solver that is perfectly healthy. The flag existed but nothing read
            # it, so the distinction was invisible.
            overflowed = bool(result.context_overflow)
            if overflowed:
                logger.warning(
                    f"[{self.meta.name}/{model_spec}] context window exceeded — "
                    "history was dropped, restarting clean"
                )

            if result.status in (ERROR, INFRA_ERROR) and not overflowed:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    self._retire_reason = f"{consecutive_errors} consecutive errors"
                    logger.warning(
                        f"[{self.meta.name}/{model_spec}] {self._retire_reason} — giving up"
                    )
                    break
            elif not overflowed:
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

            # ---- stale-round circuit breaker + periodic strategist ----
            # Consumed here rather than in _publish_harvest so a round that
            # published nothing at all still counts as teaching the team nothing.
            # Infra failures and empty spins are excluded: those belong to the
            # infra guard, and tripping the breaker on a flaky gateway would
            # redirect a solver that never got to try anything.
            if result.is_infra_failure or not result.did_work or overflowed:
                self._last_round_novel.pop(model_spec, None)
            else:
                limit = int(getattr(self.settings, "stale_round_limit", 0) or 0)
                stale = (
                    0 if self._last_round_novel.pop(model_spec, False)
                    else self._stale_rounds.get(model_spec, 0) + 1
                )
                self._stale_rounds[model_spec] = stale
                if limit > 0 and stale >= limit:
                    self._stale_rounds[model_spec] = 0
                    self._stale_breaks += 1
                    stale_break = True
                    logger.warning(
                        f"[{self.meta.name}/{model_spec}] STALE-ROUND BREAKER: {stale} "
                        "consecutive rounds added nothing new to the team (every dead "
                        "end already recorded, every fact already asserted) — forcing "
                        "a strategist consult and a hard redirect"
                    )
            await self._run_strategist(force=stale_break)

            # Cooldown between bumps — check cancellation during wait
            if not await self._sleep_or_cancel(self._bump_cooldown(bump_count)):
                break  # cancelled during cooldown

            if timed_out:
                reason = (
                    f"Your attempt hit the {budget_s}s wall-clock time budget and was "
                    "interrupted. Your confirmed findings were harvested. Work in "
                    "smaller increments, and pick a FASTER, more direct strategy."
                )
            elif overflowed:
                reason = (
                    "Your previous attempt ran out of context window and your "
                    "conversation history was dropped. Everything you confirmed is "
                    "in the state below. Read large files in slices (sed -n, head), "
                    "write intermediate results to /challenge/workspace/ instead of "
                    "keeping them in context, and avoid re-printing big outputs."
                )
            else:
                reason = "Your previous attempt did not find the flag."

            if stale_break:
                # Folded into `reason` rather than passed as a new argument: both
                # bump paths already surface it under 【上一轮为何结束】, which is
                # exactly where a solver looks for "what do I do differently now".
                reason += (
                    f"\n\n⛔ **熔断触发**：你连续 {stale} 轮没有为团队贡献任何新信息——"
                    "你报告的每个死路都已被记录过，每条结论都已有人断言过。这说明你在"
                    "微调同一个方向，而不是在探索。\n"
                    "**本轮硬性要求**：\n"
                    "1. 禁止再改动当前 payload 的字段名/编码/嵌套层数——这类微调已被证明无效。\n"
                    "2. 重读题面描述原文，逐句问：哪个细节至今没有被我解释过？\n"
                    "3. 选一个与【已排除的死路】**正交**的攻击面。若战略家刚刚广播了 PIVOT，"
                    "优先按它走。\n"
                    "4. 如果确证所有面都已排除，就用 `request_help` 说明你不懂的具体技术点。"
                )

            # An overflow already wiped the history, so the continue-session bump
            # would be talking to an empty session with no state carried over —
            # force the distilled fresh-restart path regardless of the setting.
            force_restart = overflowed

            # How the round ended decides whether the handbook this solver was
            # holding counts as tried-and-failed. Computed here, where timed_out /
            # overflowed / infra are all in scope, rather than re-derived downstream.
            outcome = self._approach_outcome(solver, model_spec, result, timed_out, overflowed)

            # Fresh-restart mode: distill this round, reset the session, and
            # restart clean carrying only the distilled state (no history bloat).
            # Also forced after a context overflow, where the history is already
            # gone and only the distilled state can carry anything forward.
            if force_restart or int(getattr(self.settings, "fresh_restart_rounds", 5) or 0) > 0:
                await self._fresh_restart(solver, model_spec, result, reason, bump_count, outcome)
            else:
                solver.bump(await self._build_bump_payload(model_spec, reason, bump_count,
                                                           outcome))
            logger.info(
                f"[{self.meta.name}/{model_spec}] Bumped ({bump_count}, approach={outcome}), "
                "resuming"
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
        finally:
            # The verifier owns a sandbox that nothing else tracks. Every exit path
            # — win, retire, external cancel, crash — has to tear it down or it
            # leaks a container and a workspace per challenge, which is the exact
            # failure `_wind_down` exists to prevent for solvers.
            await self._stop_verifier()

    async def _stop_verifier(self) -> None:
        verifier, self._verifier = self._verifier, None
        if verifier is None:
            return
        try:
            await verifier.stop()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — teardown must not mask the real outcome
            logger.warning(f"[{self.meta.name}] verifier stop failed: {e}")

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
            # Audit trail for the two anti-hallucination mechanisms. Without these
            # a run that ground for hours looks identical in the report to one
            # that was making progress the whole time.
            "stale_breaks": self._stale_breaks,
            "strategist_pivots": self._strategist_pivots,
            "strategist_failures": self._strategist_failures,
            "corroborated_facts": sum(
                1 for f in self.message_bus.findings if f.kind == KIND_FACT
            ),
            "probe_verified": sum(
                1 for f in self.message_bus.findings if f.kind == KIND_VERIFIED
            ),
            "unverified_claims": sum(
                1 for f in self.message_bus.findings if f.kind == KIND_CLAIM
            ),
            "probe_refuted": self._probe_refuted,
            "contradictions": self._contradictions,
            "claims_queued": len(self._claim_queue),
            "agents": {
                spec: {
                    "findings": self.findings.get(spec, ""),
                    "tool_calls": getattr(self.solvers.get(spec), "_tool_calls", 0),
                    "stale_rounds": self._stale_rounds.get(spec, 0),
                    "status": (
                        "won" if (self.winner and self.winner_spec == spec)
                        else "running" if (spec in self.solvers and not self.cancel_event.is_set())
                        else "finished"
                    ),
                }
                for spec in self._all_specs
            },
        }
