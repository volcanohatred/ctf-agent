"""Offline challenge-set runner & benchmark harness.

Runs swarms over a local challenges directory (each subdir must contain a
metadata.yml), without needing a live CTFd instance. If the metadata embeds an
answer key, submissions are verified locally:

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
from dataclasses import dataclass
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
        except Exception as e:  # noqa: BLE001 — one bad file must not stop discovery
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
    # Why the swarm stopped, when it retired itself (round cap / infra failure /
    # quota). Without this, a dead gateway and an unsolvable challenge both show
    # up as a bare "timeout" and the report is useless for triage.
    retire_reason: str = ""
    infra_failure: bool = False
    bumps: int = 0
    # Audit trail for the trust/strategy mechanisms. Without these a run that
    # spent four hours inside one dead paradigm reports exactly the same shape as
    # one that explored widely: `solved: false`, a high bump count, no reason.
    stale_breaks: int = 0          # circuit-breaker trips (rounds adding nothing new)
    strategist_pivots: int = 0     # PIVOT verdicts broadcast
    corroborated_facts: int = 0    # findings two solvers independently confirmed
    probe_verified: int = 0        # findings a Verifier probe confirmed on the target
    unverified_claims: int = 0     # findings still resting on one solver's word
    probe_refuted: int = 0         # premises a probe DISPROVED — these were believed
    contradictions: int = 0        # mutually exclusive statement pairs detected


def _write_report(out: str, payload: str) -> None:
    """Blocking report write — driven through asyncio.to_thread by the caller."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(payload, encoding="utf-8")


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
            result.flag = winner.flag
            result.winner_model = swarm.winner_spec
            if ch.meta.flags:
                result.verified = any(
                    winner.flag.strip().casefold() == ef.strip().casefold()
                    for ef in ch.meta.flags
                )
                # offline: only a flag that matches the answer key counts
                result.solved = result.verified
            else:
                # Unreachable in practice, and that is the point: with no answer
                # key, try_submit_flag can only answer DRY RUN, so a solver never
                # reaches FLAG_FOUND and this challenge cannot be scored at all.
                # Say so instead of reporting a bare "unsolved".
                result.solved = False
                result.error = ("no answer key in metadata.yml — offline scoring is "
                                "impossible (every submission answers DRY RUN, so no "
                                "flag can ever be confirmed)")
                logger.error("[bench] %s has no answer key — its result is meaningless",
                             ch.name)
    except TimeoutError:
        result.duration_s = time.monotonic() - t0
        result.error = f"timeout after {time_limit_s}s"
        swarm.kill(f"bench --time-limit {time_limit_s}s reached")
    except asyncio.CancelledError:
        result.duration_s = time.monotonic() - t0
        result.error = "cancelled"
        swarm.kill("bench run cancelled")
        raise
    except Exception as e:  # noqa: BLE001 — record and keep the benchmark going
        result.duration_s = time.monotonic() - t0
        result.error = f"{type(e).__name__}: {e}"
        logger.error("[bench] %s crashed: %s", ch.name, e, exc_info=True)
        swarm.kill(f"bench harness error: {type(e).__name__}")
    finally:
        result.retire_reason = swarm.retire_reason
        result.infra_failure = "infrastructure" in swarm.retire_reason
        # Read through the public status snapshot rather than poking privates: it
        # carries the audit counters too, and those are what let a post-mortem
        # tell "ground for hours inside one paradigm" apart from "explored widely
        # and never got close".
        #
        # Every access is a .get() with a default, including the pre-existing
        # `bumps`. This block runs in a `finally`, i.e. usually while handling a
        # crash — a KeyError raised here would replace the real error and destroy
        # the only diagnostic anybody was going to get.
        status = swarm.get_status()
        result.bumps = int(status.get("bumps", 0) or 0)
        result.stale_breaks = int(status.get("stale_breaks", 0) or 0)
        result.strategist_pivots = int(status.get("strategist_pivots", 0) or 0)
        result.corroborated_facts = int(status.get("corroborated_facts", 0) or 0)
        result.probe_verified = int(status.get("probe_verified", 0) or 0)
        result.unverified_claims = int(status.get("unverified_claims", 0) or 0)
        result.probe_refuted = int(status.get("probe_refuted", 0) or 0)
        result.contradictions = int(status.get("contradictions", 0) or 0)
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
    from backend.config import validate_tuning
    from backend.models import DEFAULT_MODELS
    from backend.sandbox import (
        cleanup_orphan_containers,
        cleanup_orphan_workspaces,
        configure_semaphore,
        default_workspace_root,
    )
    from backend.tracing import prune_old_traces

    specs = model_specs or list(DEFAULT_MODELS)
    for w in validate_tuning(settings):
        logger.warning("[bench] config: %s", w)

    # Health check FIRST. A dead gateway is a far more useful thing to report
    # than "no challenges found", and probing before any container work means
    # the failure costs seconds rather than the whole time limit.
    from backend.health import HealthCheckError, ensure_healthy
    try:
        await ensure_healthy(settings)
    except HealthCheckError as e:
        return {"error": f"gateway health check failed:\n{e.report.render()}", "results": []}

    challenges = discover_challenges(challenges_root, only=only)
    if not challenges:
        return {"error": f"no challenges found under {challenges_root}", "results": []}

    # A challenge with no answer key cannot be scored offline at all: every
    # submission answers DRY RUN, so no solver can ever reach FLAG_FOUND. Warn up
    # front rather than letting it surface as a mysterious "unsolved" — and note
    # that the directory name is not always the challenge name, so a typo'd path
    # can quietly select a keyless duplicate of a keyed challenge.
    keyless = [c.name for c in challenges if not c.meta.flags]
    if keyless:
        logger.warning(
            "[bench] %d/%d challenge(s) have NO answer key and cannot be scored "
            "offline: %s", len(keyless), len(challenges), ", ".join(keyless[:12]),
        )
        if len(keyless) == len(challenges):
            return {
                "error": "none of the selected challenges have an answer key "
                         "(flag:/flags: in metadata.yml), so nothing can be verified "
                         "offline. Check the path — the challenge NAME and its "
                         "directory name are not always the same.",
                "results": [],
            }

    # Escalation can add a solver beyond the configured lineup, so leave headroom.
    configure_semaphore(max(1, concurrency * (len(specs) + 1)))
    prune_old_traces()
    if getattr(settings, "cleanup_orphans_on_start", True):
        cleanup_orphan_workspaces(default_workspace_root(settings))
        # Label-matched, not owner-matched: a concurrent agent's live sandboxes
        # would be deleted too. See CLEANUP_ORPHANS_ON_START.
        await cleanup_orphan_containers()

    cost_tracker = CostTracker()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(ch: OfflineChallenge) -> BenchResult:
        async with sem:
            logger.info("[bench] start %s (%s)", ch.name, ch.meta.category)
            r = await _solve_one(ch, settings, specs, cost_tracker, time_limit_s)
            status = ("SOLVED" if r.solved
                      else f"INFRA ({r.retire_reason})" if r.infra_failure
                      else f"unsolved ({r.error or r.retire_reason or 'no flag'})")
            logger.info("[bench] done  %s → %s (%.0fs, $%.2f, %d bumps)",
                        r.name, status, r.duration_s, r.cost_usd, r.bumps)
            return r

    started = time.monotonic()
    results = list(await asyncio.gather(*(_guarded(ch) for ch in challenges)))
    total_s = time.monotonic() - started

    per_result_cost: dict[str, float] = {}
    for agent_name, usage in cost_tracker.by_agent.items():
        ch_name = agent_name.rsplit("/", 1)[0]
        per_result_cost[ch_name] = per_result_cost.get(ch_name, 0.0) + usage.cost_usd
    for r in results:
        r.cost_usd = round(per_result_cost.get(r.name, 0.0), 4)

    solved = sum(1 for r in results if r.solved)
    infra = sum(1 for r in results if r.infra_failure)
    unscoreable = sum(1 for r in results if not r.has_answer_key)
    # Only challenges that could actually be scored belong in the denominator;
    # including unscoreable ones understates the solve rate and hides why.
    scoreable = len(results) - unscoreable
    report = {
        "benchmark": {
            "challenges_root": challenges_root,
            "models": specs,
            "concurrency": concurrency,
            "time_limit_s": time_limit_s,
            "solver_time_budget_s": settings.solver_time_budget_s,
            "max_attempts_per_challenge": settings.max_attempts_per_challenge,
            "total_challenges": len(results),
            "solved": solved,
            "solve_rate": round(solved / scoreable, 4) if scoreable else 0.0,
            # Challenges with no answer key cannot be verified offline at all —
            # every submission answers DRY RUN, so FLAG_FOUND is unreachable.
            # They are excluded from the denominator and reported separately.
            "unscoreable_no_answer_key": unscoreable,
            # A run where infrastructure decided the outcome is not a measurement
            # of the models. The test is a proportion, not a count with a floor:
            # `infra <= max(1, n//4)` rated a single-challenge run that died
            # entirely on infra failures as usable, because the floor allowed one.
            "infra_failures": infra,
            "usable_run": (infra * 4 <= len(results)) and scoreable > 0,
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
                "retire_reason": r.retire_reason,
                "infra_failure": r.infra_failure,
                "bumps": r.bumps,
                "stale_breaks": r.stale_breaks,
                "strategist_pivots": r.strategist_pivots,
                "corroborated_facts": r.corroborated_facts,
                "probe_verified": r.probe_verified,
                "unverified_claims": r.unverified_claims,
                "probe_refuted": r.probe_refuted,
                "contradictions": r.contradictions,
            }
            for r in results
        ],
    }

    out = output_path or str(
        Path("logs") / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    await asyncio.to_thread(_write_report, out, payload)
    logger.info("[bench] report written to %s", out)
    if not report["benchmark"]["usable_run"]:
        logger.error(
            "[bench] %d/%d challenges retired on infrastructure failures — "
            "this run does NOT measure model capability. Fix the gateway first.",
            infra, len(results),
        )
    return report
