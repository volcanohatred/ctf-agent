"""Lightweight solver-trace readers (stdlib only — safe to import anywhere).

Separated from coordinator_loop so unit tests don't drag in the coordinator's
heavy dependency chain (genai-prices, aiodocker, ...).
"""

from __future__ import annotations

import json
from pathlib import Path


def trace_activity(trace_path: str, tail_bytes: int = 8192) -> tuple[float, list[str]]:
    """Read a solver's JSONL trace → (last_event_wall_ts, recent tool signatures).

    Returns (0.0, []) on any read/parse problem — callers treat the solver as
    "unknown", never crash.
    """
    try:
        with open(trace_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", errors="replace")
        lines = [l for l in tail.strip().splitlines() if l.strip()]
        last_ts = 0.0
        sigs: list[str] = []
        for line in lines[-15:]:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            try:
                ts = float(ev.get("ts", 0))
            except Exception:
                ts = 0.0
            last_ts = max(last_ts, ts)
            if ev.get("type") == "tool_call":
                sigs.append(f"{ev.get('tool')}:{str(ev.get('args'))[:120]}")
        return last_ts, sigs
    except Exception:
        return 0.0, []


def last_model_text(trace_path: str, tail_bytes: int = 65536) -> str:
    """The most recent reasoning text the model produced (model_response event).

    Used by the dashboard to show "what is the solver currently thinking".
    Returns "" when absent.
    """
    try:
        with open(trace_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.strip().splitlines()):
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "model_response" and ev.get("text"):
                return str(ev["text"])[:600]
        return ""
    except Exception:
        return ""
