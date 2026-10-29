"""Structural de-duplication — the deterministic floor under the LLM merge.

Both the fresh-restart journal and the dead-end ledger ask a model to "merge and
de-duplicate". That is a fine division of labour, but it was the ONLY guard: a
model that missed a restatement carried the duplicate forward forever, and when
the summarizer was unreachable (the common case against a flaky gateway) the
guard disappeared entirely. These tests pin the deterministic layer down.
"""

from __future__ import annotations

import pytest

from backend.dedup import dedup_entries, dedup_lines, new_lines, normalize

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ("Tried format string", "tried format string"),
    ("tried  format   string", "tried format string"),
    ("tried format string!", "tried format string"),
    ("【已排除】格式化字符串", "已排除格式化字符串"),
])
def test_normalize_ignores_case_punctuation_and_whitespace(a: str, b: str) -> None:
    assert normalize(a) == normalize(b)


def test_normalize_collapses_volatile_addresses() -> None:
    """The same finding reported with a different ASLR slide is the same finding.

    A runtime base address necessarily differs between two observations of the same
    truth, so collapsing it is what lets two solvers corroborate each other instead
    of looking like they disagree.
    """
    assert normalize("libc base at 0x7f8a1c000000") == normalize("libc base at 0x7f9b2d000000")
    assert normalize("pie base 0x555555554000") == normalize("pie base 0x55f3a2b1c000")
    assert normalize("leaked at 1700000001") == normalize("leaked at 1700000002")


@pytest.mark.parametrize("a,b", [
    ("offset of win() is 123456", "offset of win() is 789012"),
    ("system() is at 0x401234", "system() is at 0x409999"),
    ("the draft id is 100001", "the draft id is 200002"),
    ("write(2) to fd 12", "write(2) to fd 48"),
])
def test_normalize_preserves_static_values_that_must_agree(a: str, b: str) -> None:
    """The other half, and the one the blanket rule got wrong.

    A static offset or address is read out of the artifact: if two solvers report
    different values they are DISAGREEING, not confirming each other. Collapsing
    those literals made `offset 123456` and `offset 789012` share a key, so the
    second report "corroborated" the first and promoted a wrong number to a
    ✓ corroborated fact that every later round was told to build on. audit.py lists
    exactly this shape as a contradiction, so the two modules disagreed about the
    same input.

    The previous version of this file asserted the collapsed behaviour for
    "offset 123456 bytes", which is why the defect shipped: the test encoded it.
    """
    assert normalize(a) != normalize(b), f"{a!r} and {b!r} must stay distinct"


def test_the_two_normalize_purposes_do_not_conflict() -> None:
    """Dead-end dedup wants shape ("tried fmtstr" == "tried fmtstr, no write
    primitive"); corroboration wants value identity. Both read through normalize, so
    the literal rule has to serve the stricter of the two without breaking the
    looser one."""
    # Shape still works for prose without load-bearing numbers.
    assert normalize("tried fmtstr") == normalize("tried  fmtstr!")
    # ...while a numeric disagreement is preserved.
    assert normalize("tried offset 123456") != normalize("tried offset 789012")


def test_normalize_keeps_cjk() -> None:
    """Chinese writeups carry their meaning in the characters — stripping them
    would collapse every Chinese line to the same empty key."""
    assert normalize("格式化字符串不可行") != normalize("堆溢出不可行")
    assert "格式化字符串不可行" in normalize("【已排除】格式化字符串不可行")


def test_normalize_handles_empty() -> None:
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize("!!!") == ""


# ---------------------------------------------------------------------------
# dedup_lines
# ---------------------------------------------------------------------------

def test_exact_duplicates_collapse() -> None:
    out = dedup_lines(["tried fmtstr", "tried fmtstr", "tried fmtstr"])
    assert out == ["tried fmtstr"]


def test_punctuation_and_case_variants_collapse() -> None:
    out = dedup_lines([
        "Tried format string — no write primitive",
        "tried format string - no write primitive!",
        "TRIED FORMAT STRING — NO WRITE PRIMITIVE",
    ])
    assert len(out) == 1, out


def test_address_variants_collapse() -> None:
    out = dedup_lines([
        "libc base leaked at 0x7f8a1c000000",
        "libc base leaked at 0x7f9b2d000000",
    ])
    assert len(out) == 1, out


def test_distinct_lines_all_survive() -> None:
    lines = ["tried fmtstr", "tried heap overflow", "tried ret2libc", "found the canary leak"]
    assert dedup_lines(lines) == lines


def test_order_is_preserved() -> None:
    out = dedup_lines(["c", "a", "b", "a", "c"])
    assert out == ["c", "a", "b"]


def test_subsumed_line_is_dropped() -> None:
    """A terse restatement of an already-recorded longer finding adds nothing."""
    out = dedup_lines([
        "tried format string but there is no write primitive available",
        "tried format string",
    ])
    assert len(out) == 1
    assert "no write primitive" in out[0]


def test_longer_line_replaces_the_terse_one() -> None:
    """Detail must not be lost to an earlier terse version."""
    out = dedup_lines([
        "tried format string",
        "tried format string but there is no write primitive available",
    ])
    assert len(out) == 1
    assert "no write primitive" in out[0], out


def test_empty_lines_are_skipped() -> None:
    assert dedup_lines(["", "   ", "real finding", ""]) == ["real finding"]


def test_short_lines_are_not_subsumption_matched() -> None:
    """Below the length floor, containment is coincidence rather than restatement."""
    out = dedup_lines(["abc", "xabcy"])
    assert out == ["abc", "xabcy"]


# ---------------------------------------------------------------------------
# dedup_entries (multi-line journal entries)
# ---------------------------------------------------------------------------

def test_identical_entries_collapse() -> None:
    entries = ["[fact] key=0xBEEF\n[excluded] fmtstr"] * 4
    out = dedup_entries(entries)
    assert len(out) == 1


def test_reordered_entries_collapse() -> None:
    """A model that lists the same facts in a different order is not new information."""
    a = "[fact] key=0xBEEF\n[excluded] fmtstr failed"
    b = "[excluded] fmtstr failed\n[fact] key=0xBEEF"
    assert len(dedup_entries([a, b])) == 1


def test_genuinely_different_entries_survive() -> None:
    out = dedup_entries([
        "[fact] key=0xBEEF",
        "[fact] found the canary leak at offset 72",
    ])
    assert len(out) == 2


def test_repeated_lines_within_one_entry_collapse() -> None:
    out = dedup_entries(["tried X\ntried X\ntried X"])
    assert out == ["tried X"]


def test_entries_are_trimmed_not_dropped() -> None:
    out = dedup_entries(["", "   ", "real"])
    assert out == ["real"]


# ---------------------------------------------------------------------------
# new_lines — the dead-end ledger
# ---------------------------------------------------------------------------

def test_already_recorded_dead_end_is_not_readded() -> None:
    existing = ["tried format string, no write primitive"]
    assert new_lines(["tried format string, no write primitive"], existing) == []
    assert new_lines(["Tried format string — no write primitive!"], existing) == []


def test_new_dead_end_passes_through() -> None:
    existing = ["tried format string"]
    assert new_lines(["tried heap overflow"], existing) == ["tried heap overflow"]


def test_partial_batch_keeps_only_the_new_part() -> None:
    existing = ["tried format string"]
    out = new_lines(["tried format string", "tried ret2libc", "tried format string"], existing)
    assert out == ["tried ret2libc"]


def test_repeated_calls_accumulate_without_duplicates() -> None:
    ledger: list[str] = []
    for round_no in range(5):
        fresh = new_lines(["tried fmtstr", f"tried direction {round_no}"], ledger)
        ledger.extend(fresh)
    assert ledger.count("tried fmtstr") == 1, ledger
    assert len(ledger) == 6  # fmtstr once + 5 distinct directions


def test_empty_input_is_safe() -> None:
    assert new_lines([], ["x"]) == []
    assert new_lines(["", "  "], []) == []
