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
# Hex addresses and long digit runs vary between rounds for the same finding
# ("libc base 0x7f8a1c00" vs "0x7f8a2000"). Collapse them so the *shape* of the
# finding is what gets compared, not the volatile literal.
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]{3,}\b")
_LONG_DIGITS_RE = re.compile(r"\b\d{5,}\b")

# A line whose normalized form is contained in an already-recorded line adds
# nothing; this catches "tried fmtstr" vs "tried fmtstr, no write primitive".
SUBSUMPTION_MIN_LEN = 12


def normalize(text: str) -> str:
    """Canonical form used for duplicate detection."""
    if not text:
        return ""
    s = text.strip().lower()
    s = _HEX_RE.sub("0xaddr", s)
    s = _LONG_DIGITS_RE.sub("n", s)
    s = _NON_WORD_RE.sub("", s)
    return s


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
