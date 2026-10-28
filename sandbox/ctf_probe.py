#!/usr/bin/env python3
"""ctf_probe — a tiny HTTP/timing telemetry helper for WEB challenges.

WHY THIS EXISTS
---------------
Every solver on a timing-oracle or blind-injection challenge re-derives the same
thirty lines: build a request, measure the delta, decide whether it is a hit,
then lose the measurement because it was printed to stdout and the next bash call
is a fresh process. Across a long challenge that produces r1.py … r9.py in the
workspace, each with its own slightly different notion of "slow".

This is NOT a framework and NOT mandatory. It is deliberately generic — it knows
nothing about any challenge's endpoints — and it exists so that:

  * a measurement is recorded where the NEXT command can still read it, and
  * "which of my 200 requests were slow?" is a query, not a re-run.

Usage (any solver, any challenge):

    from ctf_probe import probe, slowest

    r = probe("POST", "http://host.docker.internal:32002/migrate",
              json={"draft_id": 7, "payload": "..."}, label="ref-chain")
    print(r.status, r.ms, r.flagged)

    # later, in a DIFFERENT bash call — the log persists:
    for row in slowest(limit=10):
        print(row["ms"], row["label"], row["status"])

    python3 -m ctf_probe --min-ms 1000          # CLI form of the same query
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Per-sandbox, inside the writable workspace, so it survives across bash calls and
# is cleaned up with the workspace. Override with CTF_PROBE_LOG.
DEFAULT_LOG = "/challenge/workspace/.telemetry.jsonl"

# Body capture is capped: a 40 MB download must not be written into the telemetry
# log that every later query re-reads.
MAX_BODY_CHARS = 4000

# A repeat count for timing work. One sample proves nothing — network jitter on a
# bridge network is routinely 50 ms, which is larger than many oracles.
DEFAULT_REPEATS = 3


def log_path() -> Path:
    return Path(os.environ.get("CTF_PROBE_LOG") or DEFAULT_LOG)


@dataclass
class Result:
    """One recorded request.

    `ms` is the wall-clock time of the request itself. `flags` are the labels the
    caller (or a rule) attached — these are what make a run queryable afterwards,
    e.g. every request flagged TIMING_ORACLE_HIT.
    """

    url: str
    method: str
    status: int
    ms: float
    label: str = ""
    error: str = ""
    body: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> str:
        return ",".join(self.flags)

    @property
    def ok(self) -> bool:
        return not self.error and 200 <= self.status < 400

    def flag(self, name: str) -> Result:
        """Attach a label; returns self so it can be used inline."""
        if name and name not in self.flags:
            self.flags.append(name)
        return self

    def flag_if(self, condition: bool, name: str) -> Result:
        return self.flag(name) if condition else self

    def as_row(self) -> dict[str, Any]:
        return {
            "ts": round(time.time(), 3),
            "url": self.url,
            "method": self.method,
            "status": self.status,
            "ms": round(self.ms, 1),
            "label": self.label,
            "error": self.error,
            "flags": self.flags,
            "body_chars": len(self.body),
            "body": self.body[:MAX_BODY_CHARS],
        }

    def __str__(self) -> str:
        bits = [f"{self.method} {self.url}", f"→ {self.status or 'ERR'}", f"{self.ms:.0f}ms"]
        if self.label:
            bits.append(f"[{self.label}]")
        if self.error:
            bits.append(f"error={self.error[:120]}")
        if self.flags:
            bits.append(f"flags={self.flagged}")
        return " ".join(bits)


def _record(result: Result) -> None:
    """Append one row. Never raises — telemetry must not break an exploit."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.as_row(), ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"ctf_probe: could not write telemetry log: {e}", file=sys.stderr)


def probe(
    method: str,
    url: str,
    *,
    label: str = "",
    repeat: int = DEFAULT_REPEATS,
    timeout: float = 30.0,
    slow_ms: float = 0.0,
    body_contains: str = "",
    flag_name: str = "HIT",
    verify: bool = False,
    **kwargs: Any,
) -> Result:
    """Send one request `repeat` times and record every attempt.

    Returns the MEDIAN attempt, not the fastest or the last: for a timing oracle
    the median is the value that survives a single GC pause or a retransmit, and
    reporting the max would make every request look like a hit.

    `repeat=1` for anything that is not a timing measurement (a state-changing
    request must not be sent three times).

    Convenience rules — both optional, both just attach a flag:
      slow_ms        → flag when the median exceeds it (a timing oracle)
      body_contains  → flag when the response body contains it (a content oracle)
    """
    import requests  # imported lazily: only WEB work pays for it

    attempts: list[Result] = []
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        error = ""
        status = 0
        body = ""
        headers: dict[str, str] = {}
        try:
            resp = requests.request(
                method.upper(), url, timeout=timeout, verify=verify, **kwargs
            )
            status = resp.status_code
            headers = {k.lower(): v for k, v in resp.headers.items()}
            body = resp.text or ""
        except Exception as e:  # noqa: BLE001 — a connection error IS a result
            error = f"{type(e).__name__}: {e}"
        attempts.append(Result(
            url=url, method=method.upper(), status=status,
            ms=(time.perf_counter() - started) * 1000.0,
            label=label, error=error, body=body, headers=headers,
        ))

    attempts.sort(key=lambda r: r.ms)
    median = attempts[len(attempts) // 2]
    median.flag_if(slow_ms > 0 and median.ms >= slow_ms, "SLOW")
    median.flag_if(bool(body_contains) and body_contains in median.body, flag_name)
    if slow_ms > 0 and median.ms >= slow_ms:
        median.flag("TIMING_ORACLE_HIT")
    # Spread is the whole point of repeating: a 5 ms spread over 3 samples means
    # the measurement is trustworthy, a 900 ms spread means the oracle is noise.
    if len(attempts) > 1:
        times = [a.ms for a in attempts]
        median.flag(f"spread={statistics.pstdev(times):.0f}ms")
    for attempt in attempts:
        _record(attempt)
    return median


def rows(**filters: Any) -> list[dict[str, Any]]:
    """Read the telemetry log back, optionally filtered.

    Supported filters: `min_ms`, `max_ms`, `status`, `label`, `flag`, `contains`
    (substring of the recorded body), `url_has`. All optional and combinable.
    """
    path = log_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "min_ms" in filters and row.get("ms", 0) < filters["min_ms"]:
                continue
            if "max_ms" in filters and row.get("ms", 0) > filters["max_ms"]:
                continue
            if "status" in filters and row.get("status") != filters["status"]:
                continue
            if "label" in filters and row.get("label") != filters["label"]:
                continue
            if "flag" in filters and filters["flag"] not in (row.get("flags") or []):
                continue
            if "contains" in filters and filters["contains"] not in (row.get("body") or ""):
                continue
            if "url_has" in filters and filters["url_has"] not in (row.get("url") or ""):
                continue
            out.append(row)
    return out


def slowest(limit: int = 10, **filters: Any) -> list[dict[str, Any]]:
    """The N slowest recorded requests — the first thing to look at after a sweep."""
    return sorted(rows(**filters), key=lambda r: r.get("ms", 0), reverse=True)[:limit]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    `int(n * pct) - 1` is not a percentile: on three samples it returns the
    MEDIAN while claiming to be p95, which on a timing sweep means reporting the
    typical case as the tail — exactly the number that decides whether an oracle
    fired. Nearest-rank is correct at n=1 (the only sample) and monotonic after.
    """
    if not sorted_values:
        return 0.0
    rank = math.ceil(pct / 100.0 * len(sorted_values))
    return sorted_values[min(max(rank - 1, 0), len(sorted_values) - 1)]


def summary() -> str:
    """One-paragraph overview: how much was measured and what stood out."""
    all_rows = rows()
    if not all_rows:
        return f"no telemetry recorded yet at {log_path()}"
    times = sorted(r.get("ms", 0) for r in all_rows)
    hosts = {urlsplit(r.get("url", "")).netloc for r in all_rows}
    statuses: dict[int, int] = {}
    for r in all_rows:
        statuses[r.get("status", 0)] = statuses.get(r.get("status", 0), 0) + 1
    flagged = [r for r in all_rows if r.get("flags")]
    lines = [
        f"{len(all_rows)} request(s) to {', '.join(sorted(h for h in hosts if h)) or '?'}",
        f"timing: median {statistics.median(times):.0f}ms, "
        f"p95 {_percentile(times, 95):.0f}ms, max {times[-1]:.0f}ms",
        "status codes: " + ", ".join(f"{k}×{v}" for k, v in sorted(statuses.items())),
    ]
    if flagged:
        counts: dict[str, int] = {}
        for r in flagged:
            for f in r["flags"]:
                counts[f] = counts.get(f, 0) + 1
        lines.append("flags: " + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ctf-probe",
        description="Query the telemetry log written by ctf_probe.probe(). "
                    "Import the module to make requests; use this to read them back.",
    )
    p.add_argument("--min-ms", type=float, help="only rows slower than this")
    p.add_argument("--max-ms", type=float, help="only rows faster than this")
    p.add_argument("--status", type=int, help="only this HTTP status")
    p.add_argument("--label", help="only this label")
    p.add_argument("--flag", help="only rows carrying this flag")
    p.add_argument("--contains", help="only rows whose body contains this")
    p.add_argument("--url-has", help="only rows whose URL contains this")
    p.add_argument("--limit", type=int, default=25, help="max rows to print")
    p.add_argument("--summary", action="store_true", help="print an overview instead")
    p.add_argument("--full-body", action="store_true", help="print the whole recorded body")
    args = p.parse_args(argv)

    if args.summary:
        print(summary())
        return 0

    filters: dict[str, Any] = {}
    for key in ("min_ms", "max_ms", "status", "label", "flag", "contains", "url_has"):
        value = getattr(args, key)
        if value is not None:
            filters[key] = value

    matched = rows(**filters)
    if not matched:
        print(f"no matching rows in {log_path()}")
        return 1
    for row in matched[-args.limit:]:
        head = (f"{row.get('ms', 0):8.1f}ms  {row.get('status', 0):>4}  "
                f"{row.get('method', '')} {row.get('url', '')}")
        if row.get("label"):
            head += f"  [{row['label']}]"
        if row.get("flags"):
            head += f"  flags={','.join(row['flags'])}"
        print(head)
        body = (row.get("body") or "").strip()
        if body:
            print("    " + (body if args.full_body else body[:400]).replace("\n", "\n    "))
        if row.get("error"):
            print(f"    error: {row['error'][:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
