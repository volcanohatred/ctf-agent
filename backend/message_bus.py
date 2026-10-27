"""Per-challenge message bus for inter-agent communication.

Findings carry a `kind` so solvers can tell confirmed facts apart from
excluded dead-ends (Cairn-style negative knowledge) and coordinator
broadcasts. Kinds:
  - "fact"      : confirmed discovery (default)
  - "excluded"  : a direction was tried and ruled out — do NOT retry it
  - "broadcast" : message from the coordinator / operator
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

KIND_FACT = "fact"
KIND_EXCLUDED = "excluded"
KIND_BROADCAST = "broadcast"
KIND_HELP = "help"  # a solver's explicit request for human help (knowledge gap)


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
        facts = [f for f in findings if f.kind == KIND_FACT]
        broadcasts = [f for f in findings if f.kind == KIND_BROADCAST]
        # Help requests used to be consumed by the cursor and then dropped by
        # every branch below — silently discarding the fact that a peer is
        # blocked. They are a real coordination signal, so surface them.
        helps = [f for f in findings if f.kind == KIND_HELP]

        parts: list[str] = ["**Findings from other agents:**"]
        if broadcasts:
            parts.append("\nCoordinator messages:")
            parts.extend(render(f) for f in broadcasts)
        if excluded:
            parts.append("\nDead ends already ruled out — do NOT retry these:")
            parts.extend(render(f) for f in excluded)
        if facts:
            parts.append("\nConfirmed facts:")
            parts.extend(render(f) for f in facts)
        if helps:
            parts.append("\nTeammates currently blocked on a knowledge gap "
                         "(a human may answer soon — don't duplicate their work):")
            parts.extend(render(f) for f in helps)
        return "\n".join(parts)

    def help_requests(self, limit: int = 50) -> list[Finding]:
        """Outstanding human-help requests, newest last (for the dashboard)."""
        return [f for f in self.findings if f.kind == KIND_HELP][-limit:]
