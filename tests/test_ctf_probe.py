"""ctf_probe — the sandbox-side telemetry helper.

This ships into the solver image, so it is real product code even though it lives
outside `backend/` and is never imported by the agent. Its whole reason to exist
is that a measurement made in one bash call is gone by the next one; if the query
half is subtly wrong, a solver draws the wrong conclusion from its own data and
there is nothing upstream to catch it.

No network and no `requests` install required: the transport is faked and the
clock is scripted, so the median/percentile arithmetic is asserted exactly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROBE_PATH = Path(__file__).resolve().parent.parent / "sandbox" / "ctf_probe.py"


def _load_probe() -> object:
    """Import sandbox/ctf_probe.py as a module.

    It must be registered in sys.modules BEFORE exec_module: @dataclass resolves
    `from __future__ import annotations` strings through sys.modules[cls.__module__],
    and an unregistered module raises AttributeError at class-creation time. That
    is an artifact of loading by path, not something the sandbox ever hits.
    """
    spec = importlib.util.spec_from_file_location("ctf_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ctf_probe"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe(tmp_path, monkeypatch):
    module = _load_probe()
    monkeypatch.setenv("CTF_PROBE_LOG", str(tmp_path / "telemetry.jsonl"))
    return module


def _write_log(probe, rows: list[dict]) -> Path:
    path = probe.log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def _row(label: str, ms: float, status: int = 200, url: str = "http://h:32002/migrate",
         body: str = "", flags: list[str] | None = None) -> dict:
    return {"ts": 1.0, "url": url, "method": "POST", "status": status, "ms": ms,
            "label": label, "error": "", "flags": flags or [],
            "body_chars": len(body), "body": body}


# ---------------------------------------------------------------------------
# Percentile arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("values,pct,expected", [
    ([2100.0], 95, 2100.0),                          # n=1: the only sample IS the tail
    ([12.0, 300.0, 2100.0], 95, 2100.0),             # the bug this pins: was 300
    ([12.0, 300.0, 2100.0], 50, 300.0),
    ([1.0, 2.0, 3.0, 4.0, 5.0], 95, 5.0),
    ([], 95, 0.0),
])
def test_percentile_is_nearest_rank_not_a_disguised_median(probe, values, pct, expected) -> None:
    """`int(n*pct)-1` reports the median while claiming to be p95. On a timing
    sweep that means presenting the typical case as the tail — the one number a
    solver uses to decide whether an oracle fired."""
    assert probe._percentile(values, pct) == expected


def test_percentile_never_indexes_out_of_range(probe) -> None:
    for n in range(1, 40):
        values = [float(i) for i in range(n)]
        for pct in (50, 90, 95, 99, 100):
            assert probe._percentile(values, pct) in values


# ---------------------------------------------------------------------------
# Querying the log back
# ---------------------------------------------------------------------------

@pytest.fixture
def logged(probe):
    _write_log(probe, [
        _row("a", 12.0, status=422, body="legacy extension rejected", flags=["ALLOWLIST_BLOCKED"]),
        _row("b", 2100.0, status=200, body="ok", flags=["SLOW", "TIMING_ORACLE_HIT"]),
        _row("c", 300.0, status=500, url="http://h:32002/other"),
    ])
    return probe


def test_unfiltered_read_returns_everything(logged) -> None:
    assert len(logged.rows()) == 3


@pytest.mark.parametrize("filters,expected", [
    ({"min_ms": 1000}, ["b"]),
    ({"max_ms": 100}, ["a"]),
    ({"status": 422}, ["a"]),
    ({"label": "c"}, ["c"]),
    ({"flag": "SLOW"}, ["b"]),
    ({"contains": "legacy"}, ["a"]),
    ({"url_has": "other"}, ["c"]),
    ({"status": 422, "contains": "legacy"}, ["a"]),
    ({"status": 422, "contains": "nomatch"}, []),
    ({"min_ms": 100, "max_ms": 1000}, ["c"]),
])
def test_filters(logged, filters, expected) -> None:
    """This is the feature's entire point: "show me every slow migrate request"
    must be a query, not a re-run of the sweep."""
    assert [r["label"] for r in logged.rows(**filters)] == expected


def test_slowest_is_ordered_and_limited(logged) -> None:
    assert [(r["label"], r["ms"]) for r in logged.slowest(2)] == [("b", 2100.0), ("c", 300.0)]


def test_slowest_accepts_filters(logged) -> None:
    assert [r["label"] for r in logged.slowest(5, status=422)] == ["a"]


def test_a_missing_log_is_empty_not_an_error(probe) -> None:
    assert probe.rows() == []
    assert probe.slowest() == []
    assert "no telemetry recorded" in probe.summary()


def test_corrupt_lines_are_skipped(probe) -> None:
    """A half-written line (a killed sandbox, a concurrent append) must not make
    the whole history unreadable — losing the sweep costs more than losing a row."""
    path = _write_log(probe, [_row("a", 12.0)])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n{not json at all}\n\n")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_row("b", 900.0)) + "\n")

    assert [r["label"] for r in probe.rows()] == ["a", "b"]


def test_summary_reports_the_tail_and_the_flags(logged) -> None:
    text = logged.summary()
    assert "3 request(s)" in text
    assert "h:32002" in text
    assert "max 2100ms" in text
    assert "TIMING_ORACLE_HIT" in text
    assert "422" in text and "500" in text


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def test_probe_records_the_median_attempt_not_the_fastest(probe, monkeypatch) -> None:
    """For a timing oracle the median is the value that survives one GC pause or
    one retransmit. Reporting the min would make every request look fast and
    reporting the max would make every request look like a hit."""
    calls: list[str] = []
    clock = iter([0.000, 0.010,   # attempt 1 → 10ms
                  1.000, 1.500,   # attempt 2 → 500ms
                  2.000, 2.050])  # attempt 3 → 50ms

    monkeypatch.setattr(probe, "time", SimpleNamespace(
        perf_counter=lambda: next(clock), time=lambda: 1700000000.0))

    def _request(method, url, **kwargs):
        calls.append(f"{method} {url}")
        return SimpleNamespace(status_code=200, headers={"Content-Type": "text/plain"},
                               text="migrated")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_request))

    result = probe.probe("post", "http://h:32002/migrate", label="variant", repeat=3)

    assert len(calls) == 3
    assert result.ms == pytest.approx(50.0), "the median of 10/500/50 is 50"
    assert result.status == 200
    assert result.body == "migrated"
    assert result.method == "POST"
    # Every attempt is recorded, not just the reported one — the spread is the
    # evidence that the median means anything.
    assert len(probe.rows()) == 3
    assert sorted(r["ms"] for r in probe.rows()) == [10.0, 50.0, 500.0]


def test_probe_records_the_spread_so_noise_is_visible(probe, monkeypatch) -> None:
    """Bridge-network jitter is routinely larger than the signal being measured.
    A solver that cannot see the spread cannot tell an oracle from noise."""
    clock = iter([0.0, 0.010, 1.0, 1.500, 2.0, 2.050])
    monkeypatch.setattr(probe, "time", SimpleNamespace(
        perf_counter=lambda: next(clock), time=lambda: 1700000000.0))
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=lambda *a, **k: (
        SimpleNamespace(status_code=200, headers={}, text=""))))

    result = probe.probe("GET", "http://h/x", repeat=3)

    assert any(f.startswith("spread=") for f in result.flags), result.flags


def test_a_single_sample_carries_no_spread_claim(probe, monkeypatch) -> None:
    """repeat=1 is required for state-changing requests; asserting a spread over
    one sample would be a fabricated statistic."""
    clock = iter([0.0, 0.250])
    monkeypatch.setattr(probe, "time", SimpleNamespace(
        perf_counter=lambda: next(clock), time=lambda: 1700000000.0))
    sent: list[str] = []

    def _request(method, url, **kwargs):
        sent.append(method)
        return SimpleNamespace(status_code=201, headers={}, text="")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_request))

    result = probe.probe("POST", "http://h/x", repeat=1)

    assert sent == ["POST"], "a state-changing request must be sent exactly once"
    assert not [f for f in result.flags if f.startswith("spread=")]


def test_slow_and_content_oracles_attach_flags(probe, monkeypatch) -> None:
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(probe, "time", SimpleNamespace(
        perf_counter=lambda: next(clock), time=lambda: 1700000000.0))
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=lambda *a, **k: (
        SimpleNamespace(status_code=200, headers={}, text="legacy extension applied"))))

    result = probe.probe("GET", "http://h/x", repeat=1, slow_ms=1500,
                         body_contains="legacy extension")

    assert "SLOW" in result.flags
    assert "TIMING_ORACLE_HIT" in result.flags
    assert "HIT" in result.flags


def test_a_connection_error_is_recorded_as_a_result_not_raised(probe, monkeypatch) -> None:
    """A timeout IS a measurement on a timing oracle — discarding it would delete
    precisely the slowest, most interesting samples."""
    clock = iter([0.0, 30.0])
    monkeypatch.setattr(probe, "time", SimpleNamespace(
        perf_counter=lambda: next(clock), time=lambda: 1700000000.0))

    def _refuse(*args, **kwargs):
        raise OSError("Connection refused")

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_refuse))

    result = probe.probe("GET", "http://h/x", repeat=1)

    assert result.error
    assert result.status == 0
    assert not result.ok
    assert len(probe.rows()) == 1


def test_recording_survives_an_unwritable_log(probe, monkeypatch, tmp_path) -> None:
    """Telemetry must never break an exploit that was otherwise working.

    The log path is pointed at a DIRECTORY: opening one for append raises
    IsADirectoryError on Linux and PermissionError on Windows, both OSError, so
    this exercises the guard on every platform without touching a real path.
    """
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    monkeypatch.setenv("CTF_PROBE_LOG", str(blocked))

    probe._record(probe.Result(url="u", method="GET", status=200, ms=1.0))


def test_the_body_written_to_the_log_is_capped(probe) -> None:
    """A large download must not be written into a log that every later query
    re-reads from disk."""
    result = probe.Result(url="u", method="GET", status=200, ms=1.0, body="x" * 9000)
    row = result.as_row()
    assert row["body_chars"] == 9000, "the true size is still reported"
    assert len(row["body"]) == probe.MAX_BODY_CHARS


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def test_flag_is_idempotent_and_chainable(probe) -> None:
    result = probe.Result(url="u", method="GET", status=200, ms=1.0)
    assert result.flag("A").flag("A").flag_if(True, "B").flag_if(False, "C").flagged == "A,B"


def test_ok_reflects_both_status_and_error(probe) -> None:
    assert probe.Result(url="u", method="GET", status=200, ms=1.0).ok
    assert not probe.Result(url="u", method="GET", status=422, ms=1.0).ok
    assert not probe.Result(url="u", method="GET", status=200, ms=1.0, error="x").ok


def test_str_includes_what_a_solver_scans_for(probe) -> None:
    text = str(probe.Result(url="http://h/x", method="POST", status=422, ms=1234.0,
                            label="v3", flags=["SLOW"]))
    for needed in ("POST", "http://h/x", "422", "1234ms", "[v3]", "SLOW"):
        assert needed in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_summary_exits_zero(logged, capsys) -> None:
    assert logged._main(["--summary"]) == 0
    assert "3 request(s)" in capsys.readouterr().out


def test_cli_query_prints_matching_rows(logged, capsys) -> None:
    assert logged._main(["--min-ms", "1000"]) == 0
    out = capsys.readouterr().out
    assert "2100.0ms" in out and "SLOW" in out


def test_cli_returns_nonzero_when_nothing_matched(logged, capsys) -> None:
    assert logged._main(["--status", "418"]) == 1
    assert "no matching rows" in capsys.readouterr().out
