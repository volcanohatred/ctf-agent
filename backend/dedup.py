"""Structural de-duplication for distilled knowledge — no LLM involved.

The fresh-restart journal and the dead-end ledger both ask a model to "merge and
de-duplicate". That is a reasonable division of labour, but it is also the only
guard, so a model that misses a restatement carries the duplicate forward
forever — and the summarizer being unavailable (the common case against a flaky
gateway) removed the guard entirely.

These helpers give a deterministic floor underneath the LLM pass: identical and
near-identical lines cannot accumulate, regardless of what the model does.

Normalization is deliberately aggressive but language-agnostic: CJK characters
are kept (they carry the meaning in Chinese writeups), ASCII is case-folded, and
punctuation/whitespace is collapsed. Two lines that differ only in punctuation,
capitalisation or a trailing address format collapse to the same key.
"""

from __future__ import annotations

import re

# Keep letters (any script, including CJK) and digits; drop everything else.
_NON_WORD_RE = re.compile(r"[^\w]", re.UNICODE)
# Volatile-literal collapsing, and why it is deliberately narrow.
#
# The original rule collapsed EVERY hex literal of 3+ digits and every digit run of
# 5+, on the reasoning that "libc base 0x7f8a1c00" and "0x7f8a2000" are the same
# finding under a different ASLR slide. That is true for dead-end dedup, where you
# want the SHAPE of what was tried. It is false for corroboration, which shares this
# function: "offset of win() is 123456" and "offset of win() is 789012" collapsed to
# one key, so two models reporting DIFFERENT offsets were recorded as independently
# agreeing and promoted to a ✓ corroborated fact that every later round was told to
# build on. audit.py meanwhile lists "two mutually exclusive concrete values pointing
# at the same thing" as a contradiction — so the two modules gave opposite answers to
# the same input.
#
# The distinction that actually matters is whether the literal CAN differ between two
# correct observations:
#   * a runtime address (mmap/PIE base) necessarily differs per process → collapse
#   * a static value taken from the artifact must match if both are right → preserve
# Runtime addresses are recognisable: long, and in the kernel/mmap/PIE ranges.
_RUNTIME_HEX_PREFIXES = ("7f", "55", "56", "ff")
_HEX_RE = re.compile(r"\b0x([0-9a-fA-F]{3,})\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{5,}\b")
# Digit runs this long are address- or timestamp-scale, not a small static offset or
# an index. Below it, two models reporting different numbers are disagreeing.
_DIGIT_COLLAPSE_MIN = 9
# Hex literals at least this many digits long, with a runtime prefix, are bases.
_HEX_COLLAPSE_MIN = 10


def _collapse_hex(match: re.Match[str]) -> str:
    """Collapse a runtime address, preserve a static one."""
    digits = match.group(1)
    if len(digits) >= _HEX_COLLAPSE_MIN and digits[:2].lower() in _RUNTIME_HEX_PREFIXES:
        return "0xaddr"
    return match.group(0)


def _collapse_digits(match: re.Match[str]) -> str:
    return "n" if len(match.group(0)) >= _DIGIT_COLLAPSE_MIN else match.group(0)


def normalize(text: str) -> str:
    """Canonical form used for duplicate detection.

    Case, punctuation and whitespace are always folded. Numeric literals are folded
    ONLY when they are volatile by nature — see `_RUNTIME_HEX_PREFIXES` and
    `_DIGIT_COLLAPSE_MIN` for why the blanket version was wrong.
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _HEX_RE.sub(_collapse_hex, s)
    s = _LONG_DIGITS_RE.sub(_collapse_digits, s)
    s = _NON_WORD_RE.sub("", s)
    return s

# A line whose normalized form is contained in an already-recorded line adds
# nothing; this catches "tried fmtstr" vs "tried fmtstr, no write primitive".
SUBSUMPTION_MIN_LEN = 12


def dedup_lines(lines: list[str]) -> list[str]:
    """Drop duplicate and subsumed lines, preserving first-seen order.

    A line is dropped when its normalized form was already seen, or when it is
    fully contained in an earlier line (the earlier, longer one carries strictly
    more information). The reverse case — a new line that subsumes an old one —
    replaces the old, so detail is never lost to an earlier terse version.
    """
    kept: list[str] = []
    kept_keys: list[str] = []
    seen: set[str] = set()

    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        key = normalize(line)
        if not key or key in seen:
            continue

        # Does an already-kept line subsume this one?
        if len(key) >= SUBSUMPTION_MIN_LEN and any(
            len(k) >= SUBSUMPTION_MIN_LEN and key in k for k in kept_keys
        ):
            continue

        # Does this line subsume an already-kept one? Replace it.
        replaced = False
        if len(key) >= SUBSUMPTION_MIN_LEN:
            for i, k in enumerate(kept_keys):
                if len(k) >= SUBSUMPTION_MIN_LEN and k in key:
                    kept[i] = line
                    kept_keys[i] = key
                    replaced = True
                    break
        if not replaced:
            kept.append(line)
            kept_keys.append(key)
        seen.add(key)

    return kept


def dedup_entries(entries: list[str]) -> list[str]:
    """De-duplicate multi-line journal entries against each other.

    Applied before the LLM merge so the model is handed an already-unique set,
    and applied again on the merged output so a model that re-lists something is
    still bounded.
    """
    out: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        text = (entry or "").strip()
        if not text:
            continue
        # Compare on the entry's own line set, so re-ordering does not defeat it.
        key = "|".join(sorted({normalize(ln) for ln in text.splitlines() if ln.strip()}))
        if not key or key in seen:
            continue
        seen.add(key)
        # Also collapse repeated lines *within* one entry.
        out.append("\n".join(dedup_lines(text.splitlines())))
    return out


def new_lines(candidate: list[str], existing: list[str]) -> list[str]:
    """Return only the candidates that are not already covered by `existing`.

    Used for the dead-end ledger: a direction already ruled out must not be
    re-added (which would grow the "do not retry" list with restatements), but a
    genuinely new one must get through.
    """
    existing_keys = {normalize(e) for e in existing if e and e.strip()}
    joined_existing = " ".join(sorted(existing_keys))
    out: list[str] = []
    for raw in candidate:
        line = (raw or "").strip()
        if not line:
            continue
        key = normalize(line)
        if not key or key in existing_keys:
            continue
        if len(key) >= SUBSUMPTION_MIN_LEN and key in joined_existing:
            continue
        out.append(line)
        existing_keys.add(key)
    return out
