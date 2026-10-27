"""Lightweight solver-trace readers (stdlib only — safe to import anywhere).

Separated from coordinator_loop so unit tests don't drag in the coordinator's
heavy dependency chain (genai-prices, aiodocker, ...).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class TraceHealth:
    """What a solver's recent trace says about whether it is making progress."""

    last_ts: float = 0.0
    tool_sigs: list[str] = field(default_factory=list)
    tool_calls: int = 0
    turns: int = 0
    errors: int = 0
    last_status: str = ""
    last_text: str = ""

    @property
    def looping(self) -> bool:
        """The same call repeated with no variety in the recent window."""
        return len(self.tool_sigs) >= 6 and len(set(self.tool_sigs[-6:])) <= 2

    @property
    def spinning(self) -> bool:
        """Turns are completing but no tool ever ran — a dead endpoint or a
        model that cannot use tools. Invisible to a tool-signature-only check,
        which is how a whole benchmark run once ground through 40 empty rounds.
        """
        return self.turns >= 2 and self.tool_calls == 0


def _read_tail(trace_path: str, tail_bytes: int) -> str:
    with open(trace_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        return f.read().decode("utf-8", errors="replace")


def _events(trace_path: str, tail_bytes: int):
    try:
        tail = _read_tail(trace_path, tail_bytes)
    except OSError:
        return
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def trace_activity(trace_path: str, tail_bytes: int = 8192) -> tuple[float, list[str]]:
    """Read a solver's JSONL trace → (last_event_wall_ts, recent tool signatures).

    Returns (0.0, []) on any read/parse problem — callers treat the solver as
    "unknown", never crash.
    """
    health = trace_health(trace_path, tail_bytes=tail_bytes)
    return health.last_ts, health.tool_sigs


def trace_health(trace_path: str, tail_bytes: int = 65536) -> TraceHealth:
    """Fuller progress picture: tools, turns, errors, last status, last text."""
    health = TraceHealth()
    for ev in _events(trace_path, tail_bytes):
        try:
            ts = float(ev.get("ts", 0))
        except (TypeError, ValueError):
            ts = 0.0
        if ts:
            health.last_ts = max(health.last_ts, ts)
        kind = ev.get("type")
        if kind == "tool_call":
            health.tool_calls += 1
            health.tool_sigs.append(f"{ev.get('tool')}:{str(ev.get('args'))[:120]}")
        elif kind == "turn_complete":
            health.turns += 1
        elif kind in ("error", "conclude_failed"):
            health.errors += 1
        elif kind == "finish":
            health.last_status = str(ev.get("status", ""))
        elif kind == "model_response" and ev.get("text"):
            health.last_text = str(ev["text"])[:600]
    # Keep only the recent window for loop detection.
    health.tool_sigs = health.tool_sigs[-15:]
    return health


def last_model_text(trace_path: str, tail_bytes: int = 65536) -> str:
    """The most recent reasoning text the model produced (model_response event).

    Used by the dashboard to show "what is the solver currently thinking".
    Returns "" when absent.
    """
    try:
        tail = _read_tail(trace_path, tail_bytes)
    except OSError:
        return ""
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "model_response" and ev.get("text"):
            return str(ev["text"])[:600]
    return ""
