"""Click CLI entry points: ctf-solve / ctf-bench / ctf-msg."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console

from backend.config import Settings
from backend.models import DEFAULT_MODELS

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    for noisy in ("httpx", "httpcore", "botocore", "urllib3", "aiodocker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%X"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _print_settings(settings: Settings, model_specs: list[str], max_challenges: int) -> None:
    console.print("[bold]CTF Agent v2[/bold]")
    console.print(f"  CTFd: {settings.ctfd_url}")
    console.print(f"  Models: {', '.join(model_specs)}")
    console.print(f"  Image: {settings.sandbox_image}")
    console.print(f"  Max challenges: {max_challenges}")
    console.print(f"  Round cap / solver: {settings.max_attempts_per_challenge or 'unlimited'}")
    console.print(f"  Time budget / round: {settings.solver_time_budget_s or 'none'}s "
                  f"(conclude {settings.conclude_timeout_s}s)")
    if settings.skills_dir:
        console.print(f"  Skills: {settings.skills_dir}")
    if settings.host_project_dir:
        console.print(f"  DooD host root: {settings.host_project_dir}")
    console.print()


async def _preflight(settings: Settings, model_specs: list[str], *, prune_logs: bool = True) -> None:
    """Fail fast on an unreachable gateway, and clean up previous runs' litter.

    Without the health probe, a misconfigured ANTHROPIC_BASE_URL shows up as
    thousands of "gave up" rounds spread over hours; with it, the run stops in a
    couple of seconds and says exactly what is wrong.
    """
    from backend.config import validate_tuning
    from backend.health import HealthCheckError, ensure_healthy
    from backend.sandbox import (
        cleanup_orphan_containers,
        cleanup_orphan_workspaces,
        configure_semaphore,
        default_workspace_root,
    )
    from backend.tracing import prune_old_traces

    for w in validate_tuning(settings):
        console.print(f"[yellow]Config warning:[/yellow] {w}")

    configure_semaphore(max(1, settings.max_concurrent_challenges * max(1, len(model_specs))))

    if prune_logs:
        try:
            prune_old_traces()
        except Exception as e:  # noqa: BLE001 — pruning is a courtesy
            logging.getLogger(__name__).debug("trace pruning failed: %s", e)

    if getattr(settings, "cleanup_orphans_on_start", True):
        try:
            cleanup_orphan_workspaces(default_workspace_root(settings))
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).debug("workspace cleanup failed: %s", e)

        # The sweep matches by label, not by owner: it clears a crashed previous
        # run, but it would also delete a *concurrently running* agent's live
        # sandboxes. That is why CLEANUP_ORPHANS_ON_START exists.
        await cleanup_orphan_containers()
    else:
        logging.getLogger(__name__).info(
            "CLEANUP_ORPHANS_ON_START=false — skipping container/workspace sweep "
            "(safe to run alongside another agent)"
        )

    # Always last, and never skipped by the cleanup branch above: this is the
    # probe that turns a dead gateway into a two-second error.
    try:
        await ensure_healthy(settings)
    except HealthCheckError as e:
        console.print("[bold red]Startup health check FAILED[/bold red]")
        # The exception carries a hint tailored to the actual failure kind
        # (loopback addressing vs container DNS vs bad credentials vs nothing
        # listening). A fixed hint here used to contradict it.
        console.print(str(e))
        raise SystemExit(2) from None


@click.command()
@click.option("--ctfd-url", default=None, help="CTFd URL (overrides .env)")
@click.option("--ctfd-token", default=None, help="CTFd API token (overrides .env)")
@click.option("--image", default="ctf-sandbox", help="Docker sandbox image name")
@click.option("--models", multiple=True, help="Model specs (default: all configured)")
@click.option("--challenge", default=None, help="Solve a single challenge directory")
@click.option("--challenges-dir", default="challenges", help="Directory for challenge files")
@click.option("--no-submit", is_flag=True, help="Dry run — don't submit flags")
@click.option("--coordinator-model", default=None,
              help="Model for the coordinator (default: COORDINATOR_MODEL or qwen3.8-max; "
                   "use your strongest)")
@click.option("--max-challenges", default=None, type=int,
              help="Max challenges solved concurrently (default: MAX_CONCURRENT_CHALLENGES)")
@click.option("--time-limit", default=0, type=int,
              help="Wall-clock cap for --challenge mode, in seconds (0 = no cap)")
@click.option("--msg-port", default=None, type=int,
              help="Operator message port (default: MSG_PORT or 9400; 0 = auto)")
@click.option("--skip-health-check", is_flag=True, help="Don't probe the gateway at startup")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def main(
    ctfd_url: str | None,
    ctfd_token: str | None,
    image: str,
    models: tuple[str, ...],
    challenge: str | None,
    challenges_dir: str,
    no_submit: bool,
    coordinator_model: str | None,
    max_challenges: int | None,
    time_limit: int,
    msg_port: int | None,
    skip_health_check: bool,
    verbose: bool,
) -> None:
    """CTF Agent — multi-model solver swarm.

    Run without --challenge to start the full coordinator (Ctrl+C to stop).
    """
    _setup_logging(verbose)

    settings = Settings(sandbox_image=image)
    if ctfd_url:
        settings.ctfd_url = ctfd_url
    if ctfd_token:
        settings.ctfd_token = ctfd_token
    if max_challenges is not None:
        settings.max_concurrent_challenges = max_challenges
    if msg_port is not None:
        settings.msg_port = msg_port
    # The coordinator model must reach BOTH the coordinator itself and the
    # swarm's escalation ladder; keeping it only in a local variable left the
    # ladder escalating to a different model than the one the operator chose.
    if coordinator_model:
        settings.coordinator_model = coordinator_model
    if skip_health_check:
        settings.startup_health_check = False

    max_challenges = settings.max_concurrent_challenges
    model_specs = list(models) if models else list(DEFAULT_MODELS)
    _print_settings(settings, model_specs, max_challenges)

    if challenge:
        asyncio.run(_run_single(settings, challenge, model_specs, no_submit, time_limit))
    else:
        asyncio.run(_run_coordinator(settings, model_specs, challenges_dir, no_submit,
                                     coordinator_model, max_challenges))


async def _run_single(
    settings: Settings,
    challenge_dir: str,
    model_specs: list[str],
    no_submit: bool,
    time_limit: int = 0,
) -> None:
    """Run a single challenge with a swarm."""
    from backend.agents.swarm import ChallengeSwarm
    from backend.cost_tracker import CostTracker
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta
    from backend.solver_base import FLAG_FOUND

    await _preflight(settings, model_specs)

    challenge_path = Path(challenge_dir)
    meta_path = challenge_path / "metadata.yml"
    if not meta_path.exists():
        console.print(f"[red]No metadata.yml found in {challenge_dir}[/red]")
        raise SystemExit(1)

    meta = ChallengeMeta.from_yaml(meta_path)
    console.print(f"[bold]Challenge:[/bold] {meta.name} ({meta.category}, {meta.value} pts)")
    if time_limit > 0:
        console.print(f"[dim]Wall-clock cap: {time_limit}s[/dim]")

    # Without an answer key, --no-submit cannot confirm anything: every submission
    # answers DRY RUN, so FLAG_FOUND is unreachable and the run always ends in
    # "No flag found" no matter how well the models did. Say so up front instead
    # of letting it look like a capability failure. (challenges/ holds keyless
    # flat duplicates of the keyed canonical sets, and a challenge's NAME is not
    # always its directory name, so this is easy to hit by accident.)
    if no_submit and not meta.flags:
        console.print(
            "[bold red]No answer key in metadata.yml[/bold red] — with --no-submit nothing "
            "can ever be confirmed, so this run cannot report a solve regardless of "
            "what the models find.\n"
            f"  [dim]path: {meta_path}\n"
            "  Add `flag: \"...\"` to that metadata.yml, or point --challenge at the "
            "keyed copy (e.g. challenges/regression/<dir> rather than a flat "
            "challenges/<name> duplicate), or drop --no-submit to verify against CTFd.[/dim]"
        )

    ctfd = CTFdClient(
        base_url=settings.ctfd_url,
        token=settings.ctfd_token,
        username=settings.ctfd_user,
        password=settings.ctfd_pass,
    )
    cost_tracker = CostTracker()

    swarm = ChallengeSwarm(
        challenge_dir=str(challenge_path),
        meta=meta,
        ctfd=ctfd,
        cost_tracker=cost_tracker,
        settings=settings,
        model_specs=model_specs,
        no_submit=no_submit,
        expected_flags=list(meta.flags),
    )

    try:
        coro = swarm.run()
        if time_limit > 0:
            # Without this, single-challenge mode had no termination guarantee at
            # all: the round loop only ends on a flag, a cancel, or the (previously
            # unused) attempt cap.
            try:
                result = await asyncio.wait_for(coro, timeout=time_limit)
            except TimeoutError:
                console.print(f"[yellow]Time limit {time_limit}s reached.[/yellow]")
                swarm.kill(f"--time-limit {time_limit}s reached")
                result = swarm.winner
        else:
            result = await coro

        if result and result.status == FLAG_FOUND and result.flag:
            console.print(f"\n[bold green]FLAG FOUND:[/bold green] {result.flag}")
            if meta.flags:
                ok = result.flag.strip().casefold() in {f.strip().casefold() for f in meta.flags}
                style = "bold green" if ok else "bold red"
                console.print(f"[{style}]Answer-key check: {'MATCH' if ok else 'MISMATCH'}[/{style}]")
        else:
            console.print("\n[bold red]No flag found.[/bold red]")
            if swarm.retire_reason:
                console.print(f"[yellow]Retired because:[/yellow] {swarm.retire_reason}")

        console.print("\n[bold]Cost Summary:[/bold]")
        for agent_name in cost_tracker.by_agent:
            console.print(f"  {agent_name}: {cost_tracker.format_usage(agent_name)}")
        console.print(f"  [bold]Total: ${cost_tracker.total_cost_usd:.2f}[/bold]")
    finally:
        await ctfd.close()


async def _run_coordinator(
    settings: Settings,
    model_specs: list[str],
    challenges_dir: str,
    no_submit: bool,
    coordinator_model: str | None,
    max_challenges: int,
) -> None:
    """Run the full coordinator (continuous until Ctrl+C)."""
    from backend.agents.claude_coordinator import run_claude_coordinator

    await _preflight(settings, model_specs)
    console.print("[bold]Starting coordinator (claude, Ctrl+C to stop)...[/bold]\n")

    results = await run_claude_coordinator(
        settings=settings,
        model_specs=model_specs,
        challenges_root=challenges_dir,
        no_submit=no_submit,
        coordinator_model=coordinator_model,
        msg_port=settings.msg_port,
    )

    console.print("\n[bold]Final Results:[/bold]")
    for challenge, data in results.get("results", {}).items():
        console.print(f"  {challenge}: {data.get('flag', 'no flag')}")
    console.print(f"\n[bold]Total cost: ${results.get('total_cost_usd', 0):.2f}[/bold]")


@click.command()
@click.option("--challenges-dir", default="challenges", help="Directory with challenge subdirs")
@click.option("--models", multiple=True, help="Model specs (default: all configured)")
@click.option("--image", default="ctf-sandbox", help="Docker sandbox image name")
@click.option("--concurrency", default=2, type=int, help="Challenges solved in parallel")
@click.option("--only", default="", help="Substring filter for challenge name/dir")
@click.option("--time-limit", default=1800, type=int, help="Per-challenge wall-clock limit (s, 0=none)")
@click.option("--output", default="", help="JSON report path (default: logs/bench-<ts>.json)")
@click.option("--skip-health-check", is_flag=True, help="Don't probe the gateway at startup")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def bench(
    challenges_dir: str,
    models: tuple[str, ...],
    image: str,
    concurrency: int,
    only: str,
    time_limit: int,
    output: str,
    skip_health_check: bool,
    verbose: bool,
) -> None:
    """Offline benchmark — solve a local challenge set and write a JSON report.

    Challenges live in subdirectories with metadata.yml. If the metadata has a
    `flag:` (or `flags:`) answer key, submissions are verified locally.
    """
    _setup_logging(verbose)
    from backend.offline import run_benchmark

    settings = Settings(sandbox_image=image)
    if skip_health_check:
        settings.startup_health_check = False
    model_specs = list(models) if models else list(DEFAULT_MODELS)

    console.print("[bold]CTF Agent — offline benchmark[/bold]")
    console.print(f"  Challenges: {challenges_dir}" + (f" (filter: {only})" if only else ""))
    console.print(f"  Models: {', '.join(model_specs)}")
    limit_label = f"{time_limit}s" if time_limit > 0 else "none"
    console.print(f"  Concurrency: {concurrency} | Time limit: {limit_label}")
    console.print()

    # A per-challenge limit below the per-round budget means the round budget can
    # never fire, so every challenge ends as an outer "timeout" and the report
    # cannot tell a slow solve from a dead gateway. Surface that up front.
    if 0 < time_limit < settings.solver_time_budget_s:
        console.print(
            f"[yellow]Warning:[/yellow] --time-limit {time_limit}s is below "
            f"SOLVER_TIME_BUDGET_S={settings.solver_time_budget_s}s, so the per-round "
            "budget (and its conclude harvest) will never trigger."
        )

    report = asyncio.run(run_benchmark(
        settings=settings,
        challenges_root=challenges_dir,
        model_specs=model_specs,
        concurrency=concurrency,
        only=only,
        time_limit_s=time_limit,
        output_path=output,
    ))

    if "error" in report:
        console.print(f"[red]{report['error']}[/red]")
        raise SystemExit(1)

    summary = report["benchmark"]
    console.print("\n[bold]Results:[/bold]")
    for r in report["results"]:
        mark = "[green]SOLVED[/green]" if r["solved"] else "[red]fail[/red]"
        note = f" ({r['error']})" if r.get("error") else ""
        console.print(f"  {mark} {r['name']} ({r['category']}) — {r['duration_s']:.0f}s, "
                      f"${r['cost_usd']:.2f} {r['winner_model']}{note}")
    console.print(
        f"\n[bold]{summary['solved']}/{summary['total_challenges']} solved "
        f"({summary['solve_rate']:.0%})[/bold] | "
        f"wall {summary['total_wall_time_s']:.0f}s | ${summary['total_cost_usd']:.2f}"
    )
    if summary.get("unscoreable_no_answer_key"):
        console.print(
            f"[bold yellow]{summary['unscoreable_no_answer_key']} challenge(s) had NO answer "
            "key[/bold yellow] — offline scoring is impossible for them (every submission "
            "answers DRY RUN), so they are excluded from the solve rate. Add "
            "`flag:` to their metadata.yml, or check that the path selects the keyed "
            "copy: a challenge's NAME and its directory name are not always the same."
        )
    if summary.get("infra_failures"):
        console.print(
            f"[bold red]{summary['infra_failures']} challenge(s) retired on infrastructure "
            "failures[/bold red] — the gateway, not the models, was the problem."
        )
    if not summary.get("usable_run", True):
        console.print(
            "[bold red]usable_run=false[/bold red] — this run does NOT measure model "
            "capability. Fix the infrastructure or the answer keys first."
        )


@click.command()
@click.argument("message")
@click.option("--port", default=None, type=int,
              help="Coordinator message port (default: MSG_PORT or 9400)")
@click.option("--host", default="127.0.0.1", help="Coordinator host")
def msg(message: str, port: int | None, host: str) -> None:
    """Send a message to the running coordinator."""
    import json
    import urllib.error
    import urllib.request

    if port is None:
        # Resolve the same default the coordinator binds, instead of hardcoding a
        # port that only matched when the container CMD passed it explicitly.
        try:
            port = int(Settings().msg_port or 9400)
        except Exception:  # noqa: BLE001 — fall back to the documented default
            port = 9400

    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/msg",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - loopback only
            data = json.loads(resp.read())
            console.print(f"[green]Sent:[/green] {data.get('queued', message[:200])}")
    except urllib.error.HTTPError as e:
        console.print(f"[red]Rejected (HTTP {e.code}):[/red] {e.read().decode(errors='replace')[:200]}")
        raise SystemExit(1) from None
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed:[/red] {e}")
        console.print(f"Is the coordinator running and listening on {host}:{port}?")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
