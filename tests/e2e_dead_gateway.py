"""End-to-end proof that a dead gateway fails fast instead of grinding.

Reproduces the exact 2026-10-26 failure: ANTHROPIC_BASE_URL points at a closed
port, so every solver turn came back "API Error: Connection refused" and the
benchmark reported 0/10 with `error: timeout after 600s` on every challenge.

Run:  python tests/e2e_dead_gateway.py
Expected: exits in a few seconds with a health-check error, NOT a 600s grind.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# A closed port on loopback: connect() fails immediately with ECONNREFUSED,
# exactly what the original incident looked like from inside the agent container.
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9/anthropic"
os.environ["ANTHROPIC_AUTH_TOKEN"] = "not-a-real-token"
os.environ["GATEWAY_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["GATEWAY_API_KEY"] = "not-a-real-key"
os.environ["STARTUP_HEALTH_CHECK"] = "true"


def _ascii(text: str) -> str:
    """Keep the report printable on a CJK-codepage console."""
    return text.encode("ascii", "replace").decode("ascii")


def _find_challenge_root() -> str:
    """Pick any bundled challenge dir — the point is the gateway, not the题."""
    for candidate in ("challenges/regression", "challenges/hard", "challenges"):
        if (ROOT / candidate).is_dir():
            return candidate
    return "challenges"


async def main() -> int:
    from backend.config import Settings
    from backend.offline import run_benchmark

    settings = Settings(sandbox_image="ctf-sandbox")
    root = _find_challenge_root()
    print(f"anthropic_base_url = {settings.anthropic_base_url}")
    print(f"startup_health_check = {settings.startup_health_check}")
    print(f"challenges_root = {root}")

    t0 = time.monotonic()
    report = await run_benchmark(
        settings=settings,
        challenges_root=root,
        concurrency=1,
        only="misc-encoding" if root.endswith("regression") else "",
        time_limit_s=600,
        output_path=str(ROOT / "logs" / "e2e-dead-gateway.json"),
    )
    elapsed = time.monotonic() - t0

    print(f"\nelapsed: {elapsed:.1f}s")
    if "error" in report:
        print("FAILED FAST with:")
        print(_ascii(str(report["error"]))[:800])
    else:
        summary = report["benchmark"]
        print(f"solved={summary['solved']}/{summary['total_challenges']} "
              f"infra_failures={summary.get('infra_failures')} "
              f"usable_run={summary.get('usable_run')}")

    # The whole point: this must be seconds, not the 600s time limit.
    ok = elapsed < 60 and "error" in report and "health check" in str(report["error"])
    print(f"\n{'PASS' if ok else 'FAIL'}: dead gateway was caught in "
          f"{elapsed:.1f}s (budget was 600s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
