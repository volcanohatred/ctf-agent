"""Loop detector tests.

Two regressions are pinned here:
  1. A repeat only counts as a loop when the SAME call produced the SAME output,
     so legitimately repeated commands (polling a service, re-running a script
     after editing it, `ls` while files are being created) are never blocked.
  2. Blocking is never permanent — an earlier revision denied the signature
     forever, and each denied retry re-appended it, keeping the window
     saturated. A false positive therefore bricked the rest of the round.
"""

from __future__ import annotations

from backend.loop_detect import MAX_CONSECUTIVE_BLOCKS, LoopDetector


def test_no_loop_for_varied_commands() -> None:
    d = LoopDetector()
    for cmd in ("ls /challenge", "cat flag.txt", "file chal", "strings chal", "checksec chal"):
        assert d.begin("Bash", {"command": cmd}) is None
        d.finish(result="some output")


def test_identical_call_with_identical_result_escalates() -> None:
    d = LoopDetector()
    statuses = []
    for _ in range(6):
        statuses.append(d.begin("Bash", {"command": "id"}))
        d.finish(result="uid=0(root)")
    assert "warn" in statuses
    assert "break" in statuses


def test_identical_command_with_different_results_is_never_a_loop() -> None:
    """Polling a service, or re-running a script after editing it, is progress."""
    d = LoopDetector()
    for i in range(20):
        assert d.begin("Bash", {"command": "curl -s http://x/status"}) is None, i
        d.finish(result=f"progress: {i}%")


def test_block_is_not_permanent() -> None:
    """After MAX_CONSECUTIVE_BLOCKS the detector must let the call through with a
    warning, so a false positive cannot brick the round."""
    d = LoopDetector()
    decisions = []
    for _ in range(40):
        status = d.begin("Bash", {"command": "ls /challenge/workspace"})
        d.finish(result="solve.py  notes.txt")
        decisions.append(status)

    breaks = decisions.count("break")
    assert breaks > 0, "the loop was never blocked"
    # It must have downgraded to warn at least once, and must not block forever.
    assert decisions.count("warn") > 0
    tail = decisions[-MAX_CONSECUTIVE_BLOCKS * 2:]
    assert tail.count("break") < len(tail), f"still hard-blocked at the end: {tail}"


def test_blocking_resumes_after_the_release_valve() -> None:
    """The downgrade is a valve, not a surrender: pressure comes back."""
    d = LoopDetector()
    decisions = []
    for _ in range(20):
        decisions.append(d.begin("Bash", {"command": "same"}))
        d.finish(result="same")
    assert "break" in decisions[:10]
    assert "warn" in decisions
    assert "break" in decisions[10:], "blocking never resumed"


def test_reset_clears_everything() -> None:
    d = LoopDetector()
    for _ in range(6):
        d.begin("Bash", {"command": "id"})
        d.finish(result="uid=0")
    assert d.begin("Bash", {"command": "id"}) == "break"

    d.reset()
    assert d.begin("Bash", {"command": "id"}) is None
    assert d.last_sig == ""


def test_cancel_forgets_a_denied_call() -> None:
    """A denied call never produces a PostToolUse, so it must not leak an entry."""
    d = LoopDetector()
    d.begin("Read", {"path": "/etc/passwd"}, call_id="c1")
    d.cancel("c1")
    assert d._inflight == {}
    # and the cancelled call did not count toward a loop
    assert d.begin("Read", {"path": "/etc/passwd"}, call_id="c2") is None


def test_one_shot_check_api() -> None:
    """The pydantic backend classifies with the result already in hand."""
    d = LoopDetector()
    out = [d.check("bash", {"command": "id"}, result="uid=0") for _ in range(6)]
    assert "break" in out

    d2 = LoopDetector()
    out2 = [d2.check("bash", {"command": "id"}, result=f"uid={i}") for i in range(6)]
    assert all(o is None for o in out2), out2


def test_args_are_normalized_before_comparison() -> None:
    """Dict key order must not make two identical calls look different."""
    d = LoopDetector()
    d.begin("Bash", {"command": "id", "timeout": 5}, call_id="a")
    d.finish(call_id="a", result="uid=0")
    for _ in range(5):
        status = d.begin("Bash", {"timeout": 5, "command": "id"}, call_id="b")
        d.finish(call_id="b", result="uid=0")
        if status == "break":
            break
    else:
        raise AssertionError("key order defeated signature matching")


def test_long_args_are_truncated_consistently() -> None:
    d = LoopDetector()
    long_a = "x" * 900
    long_b = "x" * 900 + "TAIL-DIFFERS"
    # Both truncate to the same 500-char signature, which is intentional: the
    # head of a command is what identifies it.
    assert d.check("Bash", long_a, result="r") == d.check("Bash", long_b, result="r")
