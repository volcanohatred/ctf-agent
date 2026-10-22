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


@dataclass
class Finding:
    model: str
    content: str
    kind: str = KIND_FACT
    timestamp: float = field(default_factory=time.time)


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

    async def post(self, model: str, content: str, kind: str = KIND_FACT) -> None:
        """Post a finding from a solver."""
        async with self._lock:
            self.findings.append(Finding(model=model, content=content, kind=kind))
            if len(self.findings) > MAX_FINDINGS:
                trim = len(self.findings) - MAX_FINDINGS
                self.findings = self.findings[trim:]
                self.cursors = {k: max(0, v - trim) for k, v in self.cursors.items()}

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

    async def post_advice(self, model: str, content: str) -> None:
        """Drop guidance addressed to ONE running solver (mid-run pickup)."""
        async with self._lock:
            self._advice.setdefault(model, []).append(content)
            # keep bounded
            if len(self._advice[model]) > 10:
                self._advice[model] = self._advice[model][-10:]

    async def check_advice(self, model: str) -> list[str]:
        """Drain pending advice for a solver (each item delivered once)."""
        async with self._lock:
            items = self._advice.get(model, [])
            if items:
                self._advice[model] = []
            return items

    def format_unread(self, findings: list[Finding]) -> str:
        """Format findings for injection into a solver prompt, grouped by kind."""
        if not findings:
            return ""
        excluded = [f for f in findings if f.kind == KIND_EXCLUDED]
        facts = [f for f in findings if f.kind == KIND_FACT]
        broadcasts = [f for f in findings if f.kind == KIND_BROADCAST]

        parts: list[str] = ["**Findings from other agents:**"]
        if broadcasts:
            parts.append("\nCoordinator messages:")
            parts.extend(f"- [{f.model}] {f.content}" for f in broadcasts)
        if excluded:
            parts.append("\nDead ends already ruled out — do NOT retry these:")
            parts.extend(f"- [{f.model}] {f.content}" for f in excluded)
        if facts:
            parts.append("\nConfirmed facts:")
            parts.extend(f"- [{f.model}] {f.content}" for f in facts)
        return "\n".join(parts)
