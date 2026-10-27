"""Per-tool-call JSONL event tracing — one file per solver, streamable via tail -f."""

from __future__ import annotations

import atexit
import itertools
import json
import logging
import os
import time
import weakref
from pathlib import Path

logger = logging.getLogger(__name__)

# Tracers are created once per solver attempt, and a long coordinator run creates
# hundreds of them. Registering `atexit` per instance kept a strong reference to
# every tracer (and its open file handle) for the life of the process, so none of
# them could ever be collected. One process-wide weak registry avoids that.
_LIVE_TRACERS: weakref.WeakSet[SolverTracer] = weakref.WeakSet()
_ATEXIT_REGISTERED = False
_SEQ = itertools.count(1)


def _flush_all_tracers() -> None:
    for tracer in list(_LIVE_TRACERS):
        try:
            tracer.close()
        except Exception:  # noqa: BLE001 — best effort at interpreter shutdown
            pass


def _ensure_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_flush_all_tracers)
        _ATEXIT_REGISTERED = True


def _sanitize(s: str) -> str:
    return s.replace("/", "_").replace(" ", "_")


class SolverTracer:
    """Append-only JSONL event tracer. Flushes every write for tail -f streaming."""

    def __init__(self, challenge_name: str, model_id: str, log_dir: str = "logs") -> None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        # Seconds alone collide when a swarm re-spawns the same model for the
        # same challenge within one second; appending pid+seq keeps every run in
        # its own file instead of appending into a previous run's trace.
        uniq = f"{os.getpid()}-{next(_SEQ)}"
        self.path = str(
            Path(log_dir)
            / f"trace-{_sanitize(challenge_name)}-{_sanitize(model_id)}-{ts}-{uniq}.jsonl"
        )
        self._fh = open(self.path, "a", encoding="utf-8")  # noqa: SIM115 - lifecycle is explicit
        self._closed = False
        _LIVE_TRACERS.add(self)
        _ensure_atexit()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Explicitly close the trace file. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("trace close failed: %s", e)

    # Kept as an alias: older call sites (and atexit) used `_close`.
    _close = close

    def _write(self, event: dict) -> None:
        if self._closed:
            return
        try:
            self._fh.write(json.dumps({"ts": time.time(), **event}) + "\n")
            self._fh.flush()
        except Exception as e:  # noqa: BLE001 — tracing must never break solving
            logger.debug("trace write failed: %s", e)
            self._closed = True

    def tool_call(self, tool_name: str, args: dict | str, step: int) -> None:
        args_str = args if isinstance(args, str) else json.dumps(args)
        self._write({"type": "tool_call", "tool": tool_name, "args": args_str[:2000], "step": step})

    def tool_result(self, tool_name: str, result: str, step: int) -> None:
        self._write({"type": "tool_result", "tool": tool_name, "result": result[:2000], "step": step})

    def model_response(self, text: str, step: int, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._write({"type": "model_response", "text": text[:1000], "step": step,
                      "input_tokens": input_tokens, "output_tokens": output_tokens})

    def usage(self, input_tokens: int, output_tokens: int, cache_read: int, cost_usd: float) -> None:
        self._write({"type": "usage", "input_tokens": input_tokens, "output_tokens": output_tokens,
                     "cache_read_tokens": cache_read, "cost_usd": round(cost_usd, 6)})

    def event(self, kind: str, **kwargs) -> None:
        self._write({"type": kind, **kwargs})


def prune_old_traces(log_dir: str = "logs", keep: int = 500, max_age_days: float = 14.0) -> int:
    """Delete old trace files so `logs/` cannot grow without bound.

    Keeps the newest `keep` files and anything younger than `max_age_days`.
    Benchmark reports (bench-*.json) are never touched.
    """
    root = Path(log_dir)
    if not root.is_dir() or keep <= 0:
        return 0
    traces = sorted(
        (p for p in root.glob("trace-*.jsonl")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    cutoff = time.time() - max_age_days * 86400
    removed = 0
    for idx, path in enumerate(traces):
        try:
            if idx < keep and path.stat().st_mtime > cutoff:
                continue
            path.unlink(missing_ok=True)
            removed += 1
        except OSError as e:
            logger.debug("could not prune %s: %s", path, e)
    if removed:
        logger.info("Pruned %d old trace file(s) from %s", removed, log_dir)
    return removed
