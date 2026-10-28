"""Per-challenge message bus for inter-agent communication.

Findings carry a `kind` so solvers can tell confirmed facts apart from
excluded dead-ends (Cairn-style negative knowledge) and coordinator
broadcasts. Kinds:
  - "fact"      : a discovery CORROBORATED by two independent solvers
  - "claim"     : one solver's unverified assertion — not yet ground truth
  - "excluded"  : a direction was tried and ruled out — do NOT retry it
  - "broadcast" : message from the coordinator / strategist / operator
  - "help"      : a solver's request for human help (knowledge gap)

The fact/claim split exists because the blackboard is the swarm's only shared
memory and every restarting solver is told to trust it. A lone model asserting
"the uploaded JAR does get loaded" used to land on the board looking exactly
like a fact three models had established, and siblings then built hours of work
on it. A claim is promoted to a fact only when a DIFFERENT solver independently
reports the same thing — see `promote_claims`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from backend.dedup import normalize

KIND_FACT = "fact"
KIND_VERIFIED = "verified"
KIND_CLAIM = "claim"
KIND_EXCLUDED = "excluded"
KIND_BROADCAST = "broadcast"
KIND_HELP = "help"  # a solver's explicit request for human help (knowledge gap)

# Marker used when a finding is folded into the team digest. A broadcast is NOT
# a confirmed fact — it used to fall through to the default "✓" and arrive in
# every restart payload labelled as one, which is how a coordinator's hunch
# became ground truth for the whole swarm.
KIND_TAGS = {
    KIND_VERIFIED: "✅",
    KIND_FACT: "✓",
    KIND_CLAIM: "⚠",
    KIND_EXCLUDED: "✗",
    KIND_BROADCAST: "📢",
    KIND_HELP: "?",
}

# Ordered strongest → weakest, for rendering. Agreement between two models is
# real evidence but it is still agreement: on a hard challenge the whole swarm
# can be inside the same wrong paradigm and corroborate each other's premise. A
# probe that actually ran against the target is a different kind of thing, so it
# gets its own tier rather than sharing ✓.
TRUST_ORDER = [KIND_VERIFIED, KIND_FACT, KIND_CLAIM]


@dataclass
class Finding:
    model: str
    content: str
    kind: str = KIND_FACT
    timestamp: float = field(default_factory=time.time)
    # LLM-generated key-info summary (for long findings) — used when injecting
    # into sibling prompts. The full `content` is NEVER truncated.
    summary: str = ""


MAX_FINDINGS = 200


@dataclass
class ChallengeMessageBus:
    """Append-only shared findings list with per-model cursors.

    Also carries a per-solver "advice" mailbox: the coordinator can drop
    guidance that a RUNNING solver picks up mid-turn (via its tool hook),
    instead of waiting for the next bump/restart. This is the fast channel
    for coaching weaker models.
    """

    findings: list[Finding] = field(default_factory=list)
    cursors: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _advice: dict[str, list[str]] = field(default_factory=dict)
    max_findings: int = MAX_FINDINGS
    max_advice_per_model: int = 10

    async def post(self, model: str, content: str, kind: str = KIND_FACT, summary: str = "") -> None:
        """Post a finding from a solver. `summary` (optional) is the LLM
        key-info extraction used when injecting into siblings."""
        async with self._lock:
            self.findings.append(Finding(model=model, content=content, kind=kind, summary=summary))
            if len(self.findings) > self.max_findings:
                trim = len(self.findings) - self.max_findings
                self.findings = self.findings[trim:]
                self.cursors = {k: max(0, v - trim) for k, v in self.cursors.items()}
                # Cursors for models that have not polled in a while are now
                # stale; drop them so they re-read from the start of the list
                # rather than holding a reference into trimmed history.
                self.cursors = {k: v for k, v in self.cursors.items() if v >= 0}

    async def check(self, model: str) -> list[Finding]:
        """Get unread findings from other models. Advances the cursor."""
        async with self._lock:
            cursor = self.cursors.get(model, 0)
            unread = [f for f in self.findings[cursor:] if f.model != model]
            self.cursors[model] = len(self.findings)
            return unread

    async def broadcast(self, content: str, source: str = "coordinator") -> None:
        """Coordinator broadcasts a message to all solvers."""
        await self.post(source, content, kind=KIND_BROADCAST)

    async def promote_claims(self, keys: set[str]) -> int:
        """Flip already-posted single-source claims to corroborated facts.

        `keys` are `dedup.normalize`d line keys. Called when a second, DIFFERENT
        solver independently reports something that is on the board only as one
        model's claim — that is the moment it becomes evidence rather than
        assertion. Returns how many findings were promoted.

        The earlier claim is promoted in place rather than left stale so the
        board never shows the same statement twice at two different trust levels.
        """
        return await self._reclassify_claims(keys, KIND_FACT)

    async def verify_claims(self, keys: set[str], note: str = "") -> int:
        """Flip claims a probe actually confirmed to the strongest tier.

        `note` carries the observable evidence and is appended to the finding, so
        the board holds ONE entry that says both what is known and how it was
        established. Posting a second "verified" finding instead would duplicate
        every probe result into the team digest.
        """
        return await self._reclassify_claims(keys, KIND_VERIFIED, note)

    async def refute_claims(self, keys: set[str], note: str = "") -> int:
        """Flip claims a probe DISPROVED into excluded dead ends.

        This is the capability corroboration structurally cannot provide. Withholding
        promotion only stops a false premise from being *labelled* trusted; it stays
        on the board, still readable, still tempting. Refutation moves it into the
        "do NOT retry" section, which is what actually stops four solvers from each
        spending an hour on it.

        The finding is reclassified rather than deleted so the audit trail shows
        what was believed, by whom, and what disproved it.
        """
        return await self._reclassify_claims(keys, KIND_EXCLUDED, note)

    async def _reclassify_claims(self, keys: set[str], new_kind: str, note: str = "") -> int:
        """Move matching claims to `new_kind`, optionally annotating them.

        Returns how many moved, so the caller can tell "updated the board" from
        "the claim had already been trimmed or superseded" and post a fresh entry
        only in the second case.
        """
        if not keys:
            return 0
        note = (note or "").strip()
        async with self._lock:
            moved = 0
            for f in self.findings:
                # Only claims move. A line already corroborated by two solvers or
                # confirmed by a probe is stronger evidence than one verifier's
                # opinion, and silently downgrading it would throw away work.
                if f.kind == KIND_CLAIM and normalize(f.content) in keys:
                    f.kind = new_kind
                    if note:
                        f.content = f"{f.content.rstrip()}\n  ↳ {note}"
                    moved += 1
            return moved

    async def post_help(self, model: str, question: str) -> None:
        """A solver raises a knowledge-gap help request for the human operator."""
        await self.post(model, question, kind=KIND_HELP)

    async def post_advice(self, model: str, content: str) -> None:
        """Drop guidance addressed to ONE running solver (mid-run pickup)."""
        content = (content or "").strip()
        if not content:
            return
        async with self._lock:
            queue = self._advice.setdefault(model, [])
            queue.append(content)
            # keep bounded
            if len(queue) > self.max_advice_per_model:
                self._advice[model] = queue[-self.max_advice_per_model:]

    async def check_advice(self, model: str) -> list[str]:
        """Drain pending advice for a solver (each item delivered once)."""
        async with self._lock:
            items = self._advice.get(model, [])
            if items:
                self._advice[model] = []
            return items

    def format_unread(self, findings: list[Finding]) -> str:
        """Format findings for injection into a solver prompt, grouped by kind.

        No truncation anywhere: short findings go in full; long findings use
        their LLM-generated key-info `summary` (extracted at post time) so
        siblings get the load-bearing facts without the verbose process logs.
        """
        if not findings:
            return ""

        def render(f: Finding) -> str:
            body = (f.summary or f.content).strip()
            return f"- [{f.model}] {body}"

        excluded = [f for f in findings if f.kind == KIND_EXCLUDED]
        verified = [f for f in findings if f.kind == KIND_VERIFIED]
        facts = [f for f in findings if f.kind == KIND_FACT]
        claims = [f for f in findings if f.kind == KIND_CLAIM]
        broadcasts = [f for f in findings if f.kind == KIND_BROADCAST]
        # Help requests used to be consumed by the cursor and then dropped by
        # every branch below — silently discarding the fact that a peer is
        # blocked. They are a real coordination signal, so surface them.
        helps = [f for f in findings if f.kind == KIND_HELP]

        parts: list[str] = ["**Findings from other agents:**"]
        if broadcasts:
            parts.append("\nCoordinator / strategist messages:")
            parts.extend(render(f) for f in broadcasts)
        if excluded:
            parts.append("\nDead ends already ruled out — do NOT retry these:")
            parts.extend(render(f) for f in excluded)
        if verified:
            parts.append(
                "\nProbe-VERIFIED facts (an agent ran a test against the target and "
                "observed this — strongest evidence available, build on it freely):"
            )
            parts.extend(render(f) for f in verified)
        if facts:
            parts.append("\nCorroborated facts (independently reported by 2+ agents):")
            parts.extend(render(f) for f in facts)
        if claims:
            # The wording matters more than the section header. A flat "unverified"
            # label reads as noise and gets ignored; naming the concrete failure
            # mode (building a long chain on one model's say-so) is what actually
            # changes behaviour.
            parts.append(
                "\nUNVERIFIED single-agent claims — NOT confirmed, and nobody has "
                "tested them. Re-check the ones you intend to build on with your own "
                "minimal probe before deriving anything from them; a claim that turns "
                "out false invalidates every step stacked on top of it:"
            )
            parts.extend(render(f) for f in claims)
        if helps:
            parts.append("\nTeammates currently blocked on a knowledge gap "
                         "(a human may answer soon — don't duplicate their work):")
            parts.extend(render(f) for f in helps)
        return "\n".join(parts)

    def help_requests(self, limit: int = 50) -> list[Finding]:
        """Outstanding human-help requests, newest last (for the dashboard)."""
        return [f for f in self.findings if f.kind == KIND_HELP][-limit:]
