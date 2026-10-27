"""Real-Docker integration tests for the sandbox layer.

These need a reachable Docker engine and the `ctf-sandbox` image, so they skip
themselves when either is missing (CI has neither). Run them locally with:

    pytest tests/test_sandbox_integration.py -q -rs

They cover the parts that unit tests with fakes cannot: that the bind mounts
actually land, that the answer key really is not readable from inside, that the
`timeout -k 5` wrapper really reaps a runaway process, and that stopping a
sandbox really removes its container and workspace.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.sandbox import (
    CONTAINER_LABEL,
    RUN_ID,
    DockerSandbox,
    cleanup_orphan_containers,
    default_workspace_root,
    new_docker_client,
    resolve_docker_host,
    sandbox_from_settings,
)

IMAGE = os.environ.get("CTF_SANDBOX_IMAGE", "ctf-sandbox")


async def _docker_available() -> bool:
    try:
        docker = new_docker_client()
    except Exception:  # noqa: BLE001
        return False
    try:
        await docker.version()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            await docker.close()
        except Exception:  # noqa: BLE001
            pass


async def _image_present() -> bool:
    docker = new_docker_client()
    try:
        images = await docker.images.list()
    except Exception:  # noqa: BLE001
        return False
    finally:
        await docker.close()
    return any(IMAGE in (tag or "") for img in images for tag in (img.get("RepoTags") or []))


@pytest.fixture(scope="module")
def docker_ready():
    """Skip the module unless a real engine and the sandbox image are available.

    Deliberately a SYNC fixture that drives its own loop: a module-scoped async
    fixture conflicts with pytest-asyncio's function-scoped loop and fails at
    setup rather than skipping.
    """
    if not asyncio.run(_docker_available()):
        pytest.skip("no reachable Docker engine (set DOCKER_HOST, or run on the host)")
    if not asyncio.run(_image_present()):
        pytest.skip(f"image '{IMAGE}' not built — run: "
                    "docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .")
    return True


@pytest.fixture
def challenge_dir():
    """A throwaway challenge whose metadata embeds an answer key."""
    td = Path(tempfile.mkdtemp(prefix="ctf-it-chal-"))
    try:
        (td / "distfiles").mkdir()
        (td / "distfiles" / "note.txt").write_text("hello from distfiles\n", encoding="utf-8")
        (td / "metadata.yml").write_text(
            'name: it-chal\ncategory: misc\nvalue: 100\n'
            'description: "中文描述 — UTF-8 must survive"\n'
            'flag: "flag{SUPER_SECRET_ANSWER_KEY}"\n',
            encoding="utf-8",
        )
        yield td
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _sandbox(challenge_dir: Path, workspace_root: str) -> DockerSandbox:
    return DockerSandbox(
        image=IMAGE,
        challenge_dir=str(challenge_dir),
        memory_limit="1g",
        cpu_limit=1.0,
        command_timeout_s=30,
        workspace_root=workspace_root,
    )


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

def test_windows_named_pipe_is_resolved_when_docker_host_unset(monkeypatch) -> None:
    """aiodocker's unset-DOCKER_HOST default is https://127.0.0.1:2376, which
    Docker Desktop does not expose — local mode failed on every Windows host."""
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr("backend.sandbox.sys.platform", "win32")
    monkeypatch.setattr(os, "listdir", lambda p: ["docker_engine", "other"])

    assert resolve_docker_host() == "npipe:////./pipe/docker_engine"


def test_explicit_docker_host_always_wins(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_HOST", "tcp://example.invalid:2375")
    monkeypatch.setattr("backend.sandbox.sys.platform", "win32")
    monkeypatch.setattr(os, "listdir", lambda p: ["docker_engine"])

    assert resolve_docker_host() is None, "an explicit DOCKER_HOST must not be overridden"


def test_non_windows_uses_the_default_socket(monkeypatch) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr("backend.sandbox.sys.platform", "linux")

    assert resolve_docker_host() is None


def test_missing_pipe_namespace_is_not_fatal(monkeypatch) -> None:
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.setattr("backend.sandbox.sys.platform", "win32")

    def boom(_p):
        raise OSError("no pipe namespace")

    monkeypatch.setattr(os, "listdir", boom)
    assert resolve_docker_host() is None


# ---------------------------------------------------------------------------
# Live container behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_teardown_completes_when_cancelled_midway(docker_ready, challenge_dir) -> None:
    """Cancellation must not strand the container, client session or workspace.

    The swarm cancels every sibling the instant one solver wins, so `stop()` is
    routinely interrupted. An unshielded await left 2 orphaned workspace dirs per
    solved challenge and logged "Unclosed client session" in a real 4-solver run.
    """
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        await sb.start()
        workspace, container_id = sb.workspace_dir, sb.container_id

        stop_task = asyncio.create_task(sb.stop())
        await asyncio.sleep(0)          # let it enter teardown
        stop_task.cancel()              # interrupt mid-teardown
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        # Give any shielded/detached work a moment to finish.
        for _ in range(50):
            await asyncio.sleep(0.05)
            if not Path(workspace).exists():
                break

        assert not await _container_exists(container_id), "container survived teardown"
        assert not Path(workspace).exists(), "workspace survived a cancelled teardown"
        assert not list(Path(ws_root).iterdir()), "workspace root not reclaimed"
        assert sb._container is None and sb._docker is None


@pytest.mark.asyncio
async def test_stop_is_idempotent(docker_ready, challenge_dir) -> None:
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        await sb.start()
        await sb.stop()
        await sb.stop()   # must not raise on a second call
        assert not sb.is_running


@pytest.mark.asyncio
async def test_sandbox_starts_mounts_and_stops_cleanly(docker_ready, challenge_dir) -> None:
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        workspace = None
        try:
            await sb.start()
            workspace = sb.workspace_dir
            assert sb.is_running and sb.container_id
            assert Path(workspace).is_dir()

            # distfiles are mounted read-only at the documented path
            r = await sb.exec("cat /challenge/distfiles/note.txt")
            assert r.exit_code == 0 and "hello from distfiles" in r.stdout

            listing = await sb.exec("ls -la /challenge/distfiles")
            assert "note.txt" in listing.stdout

            # the workspace is writable and persists across separate exec calls
            w = await sb.exec("echo step1 > /challenge/workspace/a.txt")
            assert w.exit_code == 0
            r2 = await sb.exec("cat /challenge/workspace/a.txt")
            assert "step1" in r2.stdout
        finally:
            await sb.stop()

        assert not sb.is_running
        # the workspace must be reclaimed — leaked ones were piling up in the repo
        assert workspace is None or not Path(workspace).exists()
        assert not list(Path(ws_root).iterdir()), "workspace dir was left behind"


@pytest.mark.asyncio
async def test_answer_key_is_not_readable_from_inside(docker_ready, challenge_dir) -> None:
    """The whole point of mounting a sanitized metadata copy."""
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        try:
            await sb.start()
            meta = await sb.exec("cat /challenge/metadata.yml")
            assert "SUPER_SECRET_ANSWER_KEY" not in meta.stdout, meta.stdout
            assert "flag" not in meta.stdout.lower(), meta.stdout
            # but the rest of the metadata (including UTF-8) must survive
            assert "it-chal" in meta.stdout
            assert "中文描述" in meta.stdout

            # nor anywhere else in the sandbox
            sweep = await sb.exec("grep -r SUPER_SECRET /challenge/ || true")
            assert "SUPER_SECRET" not in sweep.stdout, sweep.stdout
        finally:
            await sb.stop()


@pytest.mark.asyncio
async def test_metadata_mount_is_read_only(docker_ready, challenge_dir) -> None:
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        try:
            await sb.start()
            r = await sb.exec("echo tampered > /challenge/metadata.yml")
            assert r.exit_code != 0, "a solver must not be able to rewrite its own metadata"
        finally:
            await sb.stop()


@pytest.mark.asyncio
async def test_command_timeout_actually_reaps_a_runaway(docker_ready, challenge_dir) -> None:
    """`timeout -k 5` must bound every command, so an interrupted round cannot
    leave a process burning the container's quota."""
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        try:
            await sb.start()
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            # request 300s of sleep against a 30s ceiling
            r = await sb.exec("sleep 300", timeout_s=3)
            elapsed = loop.time() - t0

            assert elapsed < 25, f"the ceiling was not enforced ({elapsed:.1f}s)"
            # 124 is coreutils timeout's exit code; 137 is SIGKILL
            assert r.exit_code in (124, 137, -1), r.exit_code
        finally:
            await sb.stop()


@pytest.mark.asyncio
async def test_caller_cannot_exceed_the_configured_ceiling(docker_ready, challenge_dir) -> None:
    """A model-supplied timeout_seconds must be clamped to the configured cap."""
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        sb.command_timeout_s = 4
        try:
            await sb.start()
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await sb.exec("sleep 300", timeout_s=9999)
            assert loop.time() - t0 < 30, "the configured ceiling was ignored"
        finally:
            await sb.stop()


@pytest.mark.asyncio
async def test_containers_are_labelled_with_the_run_id(docker_ready, challenge_dir) -> None:
    """Run-id labelling is what lets two concurrent agents coexist."""
    with tempfile.TemporaryDirectory(prefix="ctf-it-ws-") as ws_root:
        sb = _sandbox(challenge_dir, ws_root)
        try:
            await sb.start()
            docker = new_docker_client()
            try:
                container = await docker.containers.get(sb.container_id)
                info = await container.show()
            finally:
                await docker.close()
            labels = (info.get("Config") or {}).get("Labels") or {}
            assert labels.get(CONTAINER_LABEL) == "true"
            assert labels.get("ctf-agent-run") == RUN_ID
        finally:
            await sb.stop()


@pytest.mark.asyncio
async def test_scoped_orphan_cleanup_spares_another_runs_container(docker_ready) -> None:
    """A bench run next to a coordinator must not delete the other's sandboxes.

    Docker labels are immutable on a live container, so the "foreign" container
    is created directly with someone else's run id rather than relabelled.
    """
    foreign_id = await _create_labelled_container({"ctf-agent-run": "someone-elses-run"})
    mine_id = await _create_labelled_container({"ctf-agent-run": RUN_ID})
    try:
        # Scoped sweep: only this run's containers are eligible.
        await cleanup_orphan_containers(only_this_run=True)

        assert await _container_exists(foreign_id), "another run's container was deleted"
        assert not await _container_exists(mine_id), "this run's orphan was not cleaned"
    finally:
        await _force_remove(foreign_id)
        await _force_remove(mine_id)


@pytest.mark.asyncio
async def test_unscoped_orphan_cleanup_takes_both(docker_ready) -> None:
    """The default (unscoped) sweep is the documented startup behaviour, and it
    is exactly why CLEANUP_ORPHANS_ON_START exists."""
    foreign_id = await _create_labelled_container({"ctf-agent-run": "someone-elses-run"})
    mine_id = await _create_labelled_container({"ctf-agent-run": RUN_ID})
    try:
        await cleanup_orphan_containers(only_this_run=False)
        assert not await _container_exists(foreign_id)
        assert not await _container_exists(mine_id)
    finally:
        await _force_remove(foreign_id)
        await _force_remove(mine_id)


async def _create_labelled_container(labels: dict[str, str]) -> str:
    docker = new_docker_client()
    try:
        config = {
            "Image": IMAGE,
            "Cmd": ["sleep", "60"],
            "Labels": {CONTAINER_LABEL: "true", **labels},
        }
        container = await docker.containers.create(config)
        await container.start()
        return container.id
    finally:
        await docker.close()


async def _force_remove(container_id: str) -> None:
    docker = new_docker_client()
    try:
        container = await docker.containers.get(container_id)
        await container.delete(force=True)
    except Exception:  # noqa: BLE001 — already gone is fine
        pass
    finally:
        await docker.close()


async def _container_exists(container_id: str) -> bool:
    docker = new_docker_client()
    try:
        # containers.get() is a coroutine in aiodocker >= 0.26; forgetting the
        # await silently produced a RuntimeWarning and a bogus False here.
        container = await docker.containers.get(container_id)
        await container.show()
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        await docker.close()


def test_sandbox_from_settings_honours_dood_translation() -> None:
    """Docker-outside-of-docker: bind mounts must use HOST paths.

    Verified without starting a container — the translated host prefix is a path
    that does not exist on this machine, so create() would (correctly) fail.
    """
    settings = SimpleNamespace(
        sandbox_image=IMAGE,
        container_memory_limit="1g",
        container_cpu_limit=1.0,
        sandbox_command_timeout_s=30,
        host_project_dir="/host/proj",
        skills_dir="/skills",
        skills_host_dir="/host/skills",
        workspace_root="",
        sandbox_proxy_url="",
    )
    sb = sandbox_from_settings(str(Path.cwd() / "challenges"), settings)

    assert sb.host_prefix_map == (str(Path.cwd().resolve()), "/host/proj")
    # an explicit host dir wins over translating skills_dir
    assert sb.skills_host_dir == "/host/skills"
    # workspaces land under the project so the host can bind-mount them
    assert sb.workspace_root == str(Path.cwd() / ".workspaces")
    assert sb.cpu_limit == 1.0 and sb.command_timeout_s == 30

    translated = sb._to_host(str(Path.cwd().resolve() / "challenges" / "x" / "distfiles"))
    assert translated == "/host/proj/challenges/x/distfiles"


def test_skills_dir_is_translated_when_no_host_dir_is_given() -> None:
    """Setting only SKILLS_DIR under DooD must still produce a host path."""
    settings = SimpleNamespace(
        sandbox_image=IMAGE, container_memory_limit="1g", container_cpu_limit=1.0,
        sandbox_command_timeout_s=30, host_project_dir="/host/proj",
        skills_dir=str(Path.cwd() / "skills"), skills_host_dir="",
        workspace_root="", sandbox_proxy_url="",
    )
    sb = sandbox_from_settings(str(Path.cwd()), settings)
    assert sb.skills_host_dir == "/host/proj/skills"


def test_default_workspace_root_is_inside_the_project_under_dood() -> None:
    assert default_workspace_root(SimpleNamespace(
        workspace_root="", host_project_dir="/host/proj")) == str(Path.cwd() / ".workspaces")
    assert default_workspace_root(SimpleNamespace(
        workspace_root="", host_project_dir="")) == tempfile.gettempdir()
    assert default_workspace_root(SimpleNamespace(
        workspace_root="/explicit", host_project_dir="/host")) == "/explicit"
