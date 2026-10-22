"""Click CLI entry points: ctf-solve / ctf-bench / ctf-msg."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from backend.config import Settings
from backend.models import DEFAULT_MODELS

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiodocker").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%X"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _print_settings(settings: Settings, model_specs: list[str], max_challenges: int) -> None:
    console.print("[bold]CTF Agent v2[/bold]")
    console.print(f"  CTFd: {settings.ctfd_url}")
    console.print(f"  Models: {', '.join(model_specs)}")
    console.print(f"  Image: {settings.sandbox_image}")
    console.print(f"  Max challenges: {max_challenges}")
    if settings.skills_dir:
        console.print(f"  Skills: {settings.skills_dir}")
    if settings.host_project_dir:
        console.print(f"  DooD host root: {settings.host_project_dir}")
    console.print()


@click.command()
@click.option("--ctfd-url", default=None, help="CTFd URL (overrides .env)")
@click.option("--ctfd-token", default=None, help="CTFd API token (overrides .env)")
@click.option("--image", default="ctf-sandbox", help="Docker sandbox image name")
@click.option("--models", multiple=True, help="Model specs (default: all configured)")
@click.option("--challenge", default=None, help="Solve a single challenge directory")
@click.option("--challenges-dir", default="challenges", help="Directory for challenge files")
@click.option("--no-submit", is_flag=True, help="Dry run — don't submit flags")
@click.option("--coordinator-model", default=None, help="Model for coordinator (default: qwen3.8-max; use your strongest)")
@click.option("--max-challenges", default=10, type=int, help="Max challenges solved concurrently")
@click.option("--msg-port", default=0, type=int, help="Operator message port (0 = auto)")
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
    max_challenges: int,
    msg_port: int,
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
    settings.max_concurrent_challenges = max_challenges

    model_specs = list(models) if models else list(DEFAULT_MODELS)
    _print_settings(settings, model_specs, max_challenges)

    if challenge:
        asyncio.run(_run_single(settings, challenge, model_specs, no_submit, max_challenges))
    else:
        asyncio.run(_run_coordinator(settings, model_specs, challenges_dir, no_submit, coordinator_model, max_challenges, msg_port))


async def _run_single(
    settings: Settings,
    challenge_dir: str,
    model_specs: list[str],
    no_submit: bool,
    max_challenges: int,
) -> None:
    """Run a single challenge with a swarm."""
    from backend.agents.swarm import ChallengeSwarm
    from backend.cost_tracker import CostTracker
    from backend.ctfd import CTFdClient
    from backend.prompts import ChallengeMeta
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore
    from backend.solver_base import FLAG_FOUND

    max_containers = max_challenges * len(model_specs)
    configure_semaphore(max_containers)
    await cleanup_orphan_containers()

    challenge_path = Path(challenge_dir)
    meta_path = challenge_path / "metadata.yml"
    if not meta_path.exists():
        console.print(f"[red]No metadata.yml found in {challenge_dir}[/red]")
        sys.exit(1)

    meta = ChallengeMeta.from_yaml(meta_path)
    console.print(f"[bold]Challenge:[/bold] {meta.name} ({meta.category}, {meta.value} pts)")

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
        result = await swarm.run()
        if result and result.status == FLAG_FOUND:
            console.print(f"\n[bold green]FLAG FOUND:[/bold green] {result.flag}")
            if meta.flags:
                ok = result.flag.strip().casefold() in {f.strip().casefold() for f in meta.flags}
                style = "bold green" if ok else "bold red"
                console.print(f"[{style}]Answer-key check: {'MATCH' if ok else 'MISMATCH'}[/{style}]")
        else:
            console.print("\n[bold red]No flag found.[/bold red]")

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
    msg_port: int = 0,
) -> None:
    """Run the full coordinator (continuous until Ctrl+C)."""
    from backend.agents.claude_coordinator import run_claude_coordinator
    from backend.sandbox import cleanup_orphan_containers, configure_semaphore

    max_containers = max_challenges * len(model_specs)
    configure_semaphore(max_containers)
    await cleanup_orphan_containers()
    console.print("[bold]Starting coordinator (claude, Ctrl+C to stop)...[/bold]\n")

    results = await run_claude_coordinator(
        settings=settings,
        model_specs=model_specs,
        challenges_root=challenges_dir,
        no_submit=no_submit,
        coordinator_model=coordinator_model,
        msg_port=msg_port,
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
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def bench(
    challenges_dir: str,
    models: tuple[str, ...],
    image: str,
    concurrency: int,
    only: str,
    time_limit: int,
    output: str,
    verbose: bool,
) -> None:
    """Offline benchmark — solve a local challenge set and write a JSON report.

    Challenges live in subdirectories with metadata.yml. If the metadata has a
    `flag:` (or `flags:`) answer key, submissions are verified locally.
    """
    _setup_logging(verbose)
    from backend.offline import run_benchmark

    settings = Settings(sandbox_image=image)
    model_specs = list(models) if models else list(DEFAULT_MODELS)

    console.print("[bold]CTF Agent — offline benchmark[/bold]")
    console.print(f"  Challenges: {challenges_dir}" + (f" (filter: {only})" if only else ""))
    console.print(f"  Models: {', '.join(model_specs)}")
    limit_label = f"{time_limit}s" if time_limit > 0 else "none"
    console.print(f"  Concurrency: {concurrency} | Time limit: {limit_label}")
    console.print()

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
        sys.exit(1)

    summary = report["benchmark"]
    console.print("\n[bold]Results:[/bold]")
    for r in report["results"]:
        mark = "[green]SOLVED[/green]" if r["solved"] else "[red]fail[/red]"
        console.print(f"  {mark} {r['name']} ({r['category']}) — {r['duration_s']:.0f}s, ${r['cost_usd']:.2f} {r['winner_model']}")
    console.print(
        f"\n[bold]{summary['solved']}/{summary['total_challenges']} solved "
        f"({summary['solve_rate']:.0%})[/bold] | "
        f"wall {summary['total_wall_time_s']:.0f}s | ${summary['total_cost_usd']:.2f}"
    )


@click.command()
@click.argument("message")
@click.option("--port", default=9400, type=int, help="Coordinator message port")
@click.option("--host", default="127.0.0.1", help="Coordinator host")
def msg(message: str, port: int, host: str) -> None:
    """Send a message to the running coordinator."""
    import json
    import urllib.request

    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/msg",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            console.print(f"[green]Sent:[/green] {data.get('queued', message[:200])}")
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        console.print("Is the coordinator running?")
        sys.exit(1)


if __name__ == "__main__":
    main()
