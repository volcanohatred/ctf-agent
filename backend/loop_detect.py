"""Tool signature tracking for loop detection."""

from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import dataclass, field

LOOP_WARNING_MESSAGE = (
    "⚠️ **You are stuck in a loop** — you have run the exact same command multiple times "
    "with identical results. STOP repeating this command. Step back, reconsider your approach, "
    "and try a **completely different** technique or tool. "
    "If you were grepping/searching, try a Python script instead. "
    "If you were analyzing one aspect of the file, switch to another. "
    "What other angles haven't you explored?"
)

# Shown when a call is actually blocked rather than merely warned about.
LOOP_BLOCKED_MESSAGE = (
    "⛔ **BLOCKED — repeated identical call with identical output.** "
    "This exact command has already been run and returned the same result. "
    "Running it again cannot produce new information. Pick a DIFFERENT technique: "
    "another tool, another file, another hypothesis. "
    "If you believe the command's output has changed on the server side, say why "
    "and vary the command."
)

# After this many consecutive blocks of the SAME signature the detector stops
# blocking and only warns. A hard, permanent block used to lock a solver out of
# a legitimately repeated command (polling a service, `ls`, re-running a script
# after editing it) for the rest of the round, because every denied retry
# re-appended the signature and kept the window saturated.
MAX_CONSECUTIVE_BLOCKS = 3


def _signature(tool_name: str, args: dict | str | None) -> str:
    if not args:
        return tool_name
    raw = json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args)
    return f"{tool_name}:{raw[:500]}"


@dataclass
class LoopDetector:
    """Detect genuinely unproductive repetition.

    A repeat only counts toward the threshold when the SAME call produced the
    SAME output. Identical commands with different results (polling a service,
    re-running a script after editing it, `ls` while files are being created)
    are legitimate and are never blocked.
    """

    window: int = 12
    warn_threshold: int = 3
    break_threshold: int = 5
    # (signature, result-digest) pairs, oldest first
    _recent: deque[tuple[str, str]] = field(init=False)
    _counts: Counter[tuple[str, str]] = field(init=False)
    # call_id -> signature, so a PostToolUse hook can attach the result
    _inflight: dict[str, str] = field(init=False)
    _blocked_streak: Counter[str] = field(init=False)
    # Signature of the most recent begin(). Lets a caller that has no stable
    # call id (hooks that only see one call at a time) still close the loop.
    _pending_sig: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self._recent = deque(maxlen=self.window)
        self._counts = Counter()
        self._inflight = {}
        self._blocked_streak = Counter()
        self._pending_sig = ""

    # ------------------------------------------------------------------
    # One-shot API (used where the result is available at the same call site)
    # ------------------------------------------------------------------

    def check(self, tool_name: str, args: dict | str | None = None,
              result: str = "") -> str | None:
        """Record a completed call and classify it.

        Returns None (no loop), "warn", or "break".
        """
        sig = _signature(tool_name, args)
        status = self._record(sig, _digest(result))
        # The one-shot API's verdict IS the decision served to the model, so it
        # counts toward the release valve.
        return self._block_decision(sig) if status == "break" else status

    # ------------------------------------------------------------------
    # Split API (PreToolUse decides, PostToolUse supplies the result)
    # ------------------------------------------------------------------

    def begin(self, tool_name: str, args: dict | str | None = None,
              call_id: str = "") -> str | None:
        """Classify a call BEFORE it runs, using its previous results.

        Without a result digest yet, the decision is based on how often this
        signature has already repeated with an identical outcome.
        """
        sig = _signature(tool_name, args)
        self._pending_sig = sig
        if call_id:
            self._inflight[call_id] = sig
        peak = self._peak_repeat(sig)
        if peak >= self.break_threshold:
            return self._block_decision(sig)
        if peak >= self.warn_threshold:
            return "warn"
        return None

    def finish(self, call_id: str = "", result: str = "", sig: str = "") -> str | None:
        """Attach the result to a call started with `begin()`.

        Resolution order: explicit signature, then the call id, then the most
        recent `begin()`. The last fallback matters because hooks do not always
        get a stable id, and dropping the pairing silently disabled detection.

        Returns the raw classification for the completed (sig, result) pair. It
        deliberately does NOT advance the block streak: the decision actually
        served to the model was already made by `begin()`, and counting a
        discarded verdict here made the release valve fire far too early.
        """
        signature = sig
        if not signature and call_id:
            signature = self._inflight.pop(call_id, "")
        if not signature:
            signature = self._pending_sig
        if call_id:
            self._inflight.pop(call_id, None)
        self._pending_sig = ""
        if not signature:
            return None
        return self._record(signature, _digest(result))

    def cancel(self, call_id: str = "") -> None:
        """Forget a `begin()` whose call was denied and will never produce a result."""
        if call_id:
            self._inflight.pop(call_id, None)
        self._pending_sig = ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _peak_repeat(self, sig: str) -> int:
        return max((c for (s, _), c in self._counts.items() if s == sig), default=0)

    def _record(self, sig: str, digest: str) -> str | None:
        pair = (sig, digest)
        if self._recent:
            evicted_sig, evicted_digest = self._recent[0]
            if len(self._recent) == self.window:
                self._counts[(evicted_sig, evicted_digest)] -= 1
                if self._counts[(evicted_sig, evicted_digest)] <= 0:
                    del self._counts[(evicted_sig, evicted_digest)]
        self._recent.append(pair)
        self._counts[pair] += 1

        count = self._counts[pair]
        if count >= self.break_threshold:
            return "break"
        if count >= self.warn_threshold:
            return "warn"
        return None

    def _block_decision(self, sig: str) -> str:
        """Block, but never permanently: every MAX_CONSECUTIVE_BLOCKS-th repeat
        is downgraded to a warning and the streak restarts.

        A hard, permanent block used to lock a solver out of a legitimately
        repeated command for the rest of the round, because every denied retry
        re-appended the signature and kept the window saturated. Cycling keeps
        the pressure on a real loop while guaranteeing a false positive can
        always make progress.
        """
        self._blocked_streak[sig] += 1
        if self._blocked_streak[sig] > MAX_CONSECUTIVE_BLOCKS:
            self._blocked_streak[sig] = 0
            return "warn"
        return "break"

    @staticmethod
    def blocked_message() -> str:
        return LOOP_BLOCKED_MESSAGE

    @property
    def last_sig(self) -> str:
        return self._recent[-1][0] if self._recent else ""

    def reset(self) -> None:
        self._recent.clear()
        self._counts.clear()
        self._inflight.clear()
        self._blocked_streak.clear()
        self._pending_sig = ""


def _digest(text: str, limit: int = 512) -> str:
    """Stable, cheap fingerprint of a tool result (no hashing dependency)."""
    if not text:
        return ""
    flat = " ".join(str(text).split())
    return flat[:limit]
