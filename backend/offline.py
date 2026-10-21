"""Offline challenge-set runner & benchmark harness.

Runs swarms over a local challenges directory (each subdir must contain a
metadata.yml), without needing a live CTFd instance. If the metadata embeds
an answer key, submissions are verified locally:

    name: baby-pwn
    category: pwn
    flag: "flag{r3al_answ3r}"            # single expected flag
    # or:  flags: ["flag{a}", "flag{b}"] # multiple accepted flags

Results are appended to a JSON report for regression tracking (M4).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.config import Settings
from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.prompts import ChallengeMeta
from backend.solver_base import FLAG_FOUND

logger = logging.getLogger(__name__)


@dataclass
class OfflineChallenge:
    name: str
    directory: str
    meta: ChallengeMeta


def discover_challenges(challenges_root: str, only: str = "") -> list[OfflineChallenge]:
    """Find challenge directories containing metadata.yml under challenges_root."""
    root = Path(challenges_root)
    found: list[OfflineChallenge] = []
    for meta_path in sorted(root.rglob("metadata.yml")):
        ch_dir = meta_path.parent
        try:
            meta = ChallengeMeta.from_yaml(meta_path)
        except Exception as e:
            logger.warning("Skipping %s (bad metadata.yml: %s)", ch_dir, e)
            continue
        if only and only.lower() not in meta.name.lower() and only.lower() not in ch_dir.name.lower():
            continue
        found.append(OfflineChallenge(name=meta.name, directory=str(ch_dir), meta=meta))
    return found


@dataclass
class BenchResult:
    name: str
    category: str
    solved: bool = False
    verified: bool = False  # flag matched the embedded answer key
    flag: str | None = None
    winner_model: str = ""
    duration_s: float = 0.0
    cost_usd: float = 0.0
    has_answer_key: bool = False
    error: str = ""


async def _solve_one(
    ch: OfflineChallenge,
    settings: Settings,
    model_specs: list[str],
    cost_tracker: CostTracker,
    time_limit_s: int,
) -> BenchResult:
    """Run a single-challenge swarm offline."""
    from backend.agents.swarm import ChallengeSwarm

    result = BenchResult(
        name=ch.name,
        category=ch.meta.category,
        has_answer_key=bool(ch.meta.flags),
    )
    # A throwaway client — never contacted in offline mode.
    ctfd = CTFdClient(base_url="http://offline.invalid", token="", username="", password="")
    swarm = ChallengeSwarm(
        challenge_dir=ch.directory,
        meta=ch.meta,
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=model_specs,
        no_submit=True,
        expected_flags=list(ch.meta.flags),
    )

    t0 = time.monotonic()
    try:
        coro = swarm.run()
        winner = await asyncio.wait_for(coro, timeout=time_limit_s) if time_limit_s > 0 else await coro
        result.duration_s = time.monotonic() - t0
        if winner and winner.status == FLAG_FOUND and winner.flag:
            result.solved = True
            result.flag = winner.flag
            result.winner_model = swarm.winner_spec
            if ch.meta.flags:
                result.verified = any(
                    winner.flag.strip().casefold() == ef.strip().casefold()
                    for ef in ch.meta.flags
                )
                result.solved = result.verified  # offline: only verified counts
    except TimeoutError:
        result.duration_s = time.monotonic() - t0
        result.error = f"timeout after {time_limit_s}s"
        swarm.kill()
    except Exception as e:
        result.duration_s = time.monotonic() - t0
        result.error = str(e)
        swarm.kill()
    finally:
        await ctfd.close()
    return result


async def run_benchmark(
    settings: Settings,
    challenges_root: str = "challenges",
    model_specs: list[str] | None = None,
    concurrency: int = 2,
    only: str = "",
    time_limit_s: int = 1800,
    output_path: str = "",
) -> dict:
    """Run all discovered challenges with bounded concurrency; write a JSON report."""
    from backend.models import DEFAULT_MODELS
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore

    specs = model_specs or list(DEFAULT_MODELS)
    challenges = discover_challenges(challenges_root, only=only)
    if not challenges:
        return {"error": f"no challenges found under {challenges_root}", "results": []}

    configure_semaphore(concurrency * len(specs))
    await cleanup_orphan_containers()

    cost_tracker = CostTracker()
    sem = asyncio.Semaphore(concurrency)

    async def _guarded(ch: OfflineChallenge) -> BenchResult:
        async with sem:
            logger.info("[bench] start %s (%s)", ch.name, ch.meta.category)
            r = await _solve_one(ch, settings, specs, cost_tracker, time_limit_s)
            status = "SOLVED" if r.solved else ("unsolved" if not r.error else f"error: {r.error}")
            logger.info("[bench] done  %s → %s (%.0fs, $%.2f)", r.name, status, r.duration_s, r.cost_usd)
            return r

    started = time.monotonic()
    results = await asyncio.gather(*(_guarded(ch) for ch in challenges))
    total_s = time.monotonic() - started

    per_result_cost: dict[str, float] = {}
    for agent_name, usage in cost_tracker.by_agent.items():
        ch_name = agent_name.rsplit("/", 1)[0]
        per_result_cost[ch_name] = per_result_cost.get(ch_name, 0.0) + usage.cost_usd
    for r in results:
        r.cost_usd = round(per_result_cost.get(r.name, 0.0), 4)

    solved = sum(1 for r in results if r.solved)
    report = {
        "benchmark": {
            "challenges_root": challenges_root,
            "models": specs,
            "concurrency": concurrency,
            "time_limit_s": time_limit_s,
            "total_challenges": len(results),
            "solved": solved,
            "solve_rate": round(solved / len(results), 4) if results else 0.0,
            "total_wall_time_s": round(total_s, 1),
            "total_cost_usd": round(cost_tracker.total_cost_usd, 4),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": [
            {
                "name": r.name,
                "category": r.category,
                "solved": r.solved,
                "verified": r.verified,
                "flag": r.flag,
                "winner_model": r.winner_model,
                "duration_s": round(r.duration_s, 1),
                "cost_usd": r.cost_usd,
                "has_answer_key": r.has_answer_key,
                "error": r.error,
            }
            for r in results
        ],
    }

    out = output_path or str(
        Path("logs") / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("[bench] report written to %s", out)
    return report
