"""Docker sandbox for CTF challenge solving — native async via aiodocker."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import shlex
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiodocker

from backend.config import float_setting, int_setting

logger = logging.getLogger(__name__)

CONTAINER_LABEL = "ctf-agent"
# Per-process run id, stamped on every container and workspace this agent
# creates. Without it, `cleanup_orphan_containers()` deletes containers belonging
# to a *concurrently running* agent (e.g. a bench run next to a coordinator).
RUN_ID = uuid.uuid4().hex[:8]
RUN_LABEL = "ctf-agent-run"
WORKSPACE_PREFIX = "ctf-workspace-"

# A flag-shaped literal: `word{...}`. Used to verify the answer key really left the
# sanitized metadata before it is bind-mounted into a solver's sandbox. Deliberately
# narrow — matching the word "flag" anywhere would reject every challenge whose
# description mentions recovering one.
_FLAG_LITERAL_RE = re.compile(r"\b[A-Za-z0-9_]{2,32}\{[^}\n]{1,200}\}")


def _find_flag_literal(data: Any, path: str = "") -> str | None:
    """Return the key path of the first flag-shaped value in a parsed YAML tree.

    Recurses through mappings and sequences so an answer key nested under something
    other than a top-level `flag:`/`flags:` is still caught — which is the whole
    point of checking the value rather than the field name.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            found = _find_flag_literal(value, f"{path}.{key}" if path else str(key))
            if found:
                return found
        return None
    if isinstance(data, (list, tuple)):
        for i, value in enumerate(data):
            found = _find_flag_literal(value, f"{path}[{i}]")
            if found:
                return found
        return None
    if isinstance(data, str) and _FLAG_LITERAL_RE.search(data):
        return path or "<root>"
    return None
# Workspaces older than this are considered leaked (a live solver refreshes its
# directory mtime by writing into it, and a round is bounded by the time budget).
WORKSPACE_MAX_AGE_S = 12 * 3600

# On Windows, Docker Desktop serves the engine over a named pipe. aiodocker's
# fallback when DOCKER_HOST is unset is https://127.0.0.1:2376, which Docker
# Desktop does NOT expose — so local (non-container) mode failed on every Windows
# host with "Cannot connect to Docker Engine". The containerized DooD path is
# unaffected: it uses /var/run/docker.sock.
WINDOWS_PIPE_ROOT = r"\\.\pipe"
WINDOWS_NPIPE_URL = "npipe:////./pipe/docker_engine"


def resolve_docker_host() -> str | None:
    """The Docker endpoint aiodocker should use, or None to let it decide.

    An explicit DOCKER_HOST always wins (remote engines, custom contexts).
    """
    if os.environ.get("DOCKER_HOST"):
        return None
    if sys.platform != "win32":
        return None
    try:
        # os.path.exists() is False for named pipes, and Path.iterdir() does not
        # work on the \\.\pipe namespace at all — enumerating it with os.listdir
        # is the only reliable check.
        if "docker_engine" in os.listdir(WINDOWS_PIPE_ROOT):  # noqa: PTH208
            return WINDOWS_NPIPE_URL
    except OSError as e:
        logger.debug("Could not enumerate the Windows pipe namespace: %s", e)
    return None


def new_docker_client() -> aiodocker.Docker:
    """Build an aiodocker client against the right endpoint for this platform."""
    url = resolve_docker_host()
    if url:
        logger.debug("Docker endpoint: %s", url)
        return aiodocker.Docker(url=url)
    return aiodocker.Docker()

# Concurrency control
_start_semaphore: asyncio.Semaphore | None = None
_active_count: int = 0
_count_lock = asyncio.Lock()

# Detached teardown tasks: a `stop()` whose caller was cancelled leaves its shielded
# inner task running, and asyncio holds only a WEAK reference to tasks. Without this
# registry such a task could be garbage collected mid-`container.delete()`, and
# `asyncio.run`'s shutdown cancels all pending tasks — destroying it between deleting
# the container and closing the client, or before the workspace rmtree. The
# coordinator awaits these before its loop closes.
_DETACHED_TEARDOWNS: set[asyncio.Task] = set()


def _register_detached(task: asyncio.Task) -> None:
    """Hold a strong reference until the teardown finishes."""
    _DETACHED_TEARDOWNS.add(task)
    task.add_done_callback(_DETACHED_TEARDOWNS.discard)


async def await_detached_teardowns(timeout: float = 30.0) -> int:
    """Wait for teardowns that outlived their caller. Returns how many were pending.

    Called from the coordinator's shutdown path, after the swarms are cancelled but
    before the loop closes. Bounded, because a wedged Docker daemon must not hang
    shutdown forever — but the bound is generous, since abandoning a teardown here is
    precisely what leaks a container.
    """
    pending = {t for t in _DETACHED_TEARDOWNS if not t.done()}
    if not pending:
        return 0
    logger.info("awaiting %d detached sandbox teardown(s)", len(pending))
    _done, still = await asyncio.wait(pending, timeout=timeout)
    if still:
        logger.warning(
            "%d sandbox teardown(s) did not finish within %.0fs — those containers "
            "may be orphaned and will need the startup sweep", len(still), timeout,
        )
    for task in _done:
        if not task.cancelled() and task.exception() is not None:
            logger.debug("detached teardown raised: %s", task.exception())
    return len(pending)


_WARN_THRESHOLDS = {100, 200, 500}


def configure_semaphore(max_concurrent: int = 50) -> None:
    """Set the max concurrent container starts. Call once at startup."""
    global _start_semaphore
    _start_semaphore = asyncio.Semaphore(max_concurrent)


async def _track_start() -> None:
    global _active_count
    async with _count_lock:
        _active_count += 1
        if _active_count in _WARN_THRESHOLDS:
            logger.warning("Active containers: %d", _active_count)


async def _track_stop() -> None:
    global _active_count
    async with _count_lock:
        _active_count = max(0, _active_count - 1)


async def cleanup_orphan_containers(only_this_run: bool = False) -> int:
    """Kill leftover ctf-agent containers from a previous run.

    `only_this_run=True` restricts the sweep to containers stamped with this
    process's RUN_ID, which is what a second concurrent agent must use so it
    does not delete the first agent's live sandboxes.
    """
    filters: dict[str, Any] = {"label": [CONTAINER_LABEL]}
    if only_this_run:
        filters = {"label": [CONTAINER_LABEL, f"{RUN_LABEL}={RUN_ID}"]}
    removed = 0
    docker: aiodocker.Docker | None = None
    try:
        docker = new_docker_client()
        containers = await docker.containers.list(all=True, filters=filters)
        for c in containers:
            try:
                await c.delete(force=True)
                removed += 1
            except Exception as e:  # noqa: BLE001 — best effort, keep sweeping
                logger.debug("Could not remove orphan container: %s", e)
        if removed:
            logger.info("Cleaned up %d orphan container(s)", removed)
    except Exception as e:  # noqa: BLE001 — no docker socket is not fatal here
        logger.warning("Orphan cleanup failed: %s", e)
    finally:
        if docker is not None:
            try:
                await docker.close()
            except Exception:  # noqa: BLE001
                pass
    return removed


def cleanup_orphan_workspaces(workspace_root: str, max_age_s: float = WORKSPACE_MAX_AGE_S) -> int:
    """Delete leaked per-sandbox workspace directories.

    `DockerSandbox.stop()` removes its own workspace, but a hard crash (or a
    container force-kill) leaves them behind; they accumulate inside the repo
    because the default root is <project>/.workspaces. Only directories matching
    the workspace naming scheme and older than `max_age_s` are touched.
    """
    if not workspace_root:
        return 0
    root = Path(workspace_root)
    if not root.is_dir():
        return 0
    cutoff = time.time() - max_age_s
    removed = 0
    import shutil

    for entry in root.iterdir():
        if not entry.is_dir() or not entry.name.startswith(WORKSPACE_PREFIX):
            continue
        try:
            if entry.stat().st_mtime > cutoff:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except OSError as e:
            logger.debug("Could not remove stale workspace %s: %s", entry, e)
    if removed:
        logger.info("Cleaned up %d stale workspace(s) under %s", removed, root)
    return removed


def default_workspace_root(settings: object) -> str:
    """The workspace base dir this configuration will use ("" = system temp)."""
    explicit = getattr(settings, "workspace_root", "") or ""
    if explicit:
        return explicit
    if getattr(settings, "host_project_dir", ""):
        return str(Path.cwd() / ".workspaces")
    return tempfile.gettempdir()


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


def to_host_path(path: str, host_prefix_map: tuple[str, str] | None) -> str:
    """Translate a container-local path to a docker-host path for bind mounts."""
    if not host_prefix_map:
        return path
    container_prefix, host_prefix = host_prefix_map
    norm = path.replace("\\", "/")
    cp = container_prefix.replace("\\", "/").rstrip("/")
    if cp and norm.startswith(cp):
        return host_prefix.replace("\\", "/").rstrip("/") + norm[len(cp):]
    return path


def sandbox_from_settings(challenge_dir: str, settings: object) -> DockerSandbox:
    """Build a DockerSandbox honoring docker-in-docker path translation.

    - When settings.host_project_dir is set (agent runs in a container), all
      bind mounts are translated to host paths and workspaces are created
      under <project>/.workspaces (which must be volume-mounted from the host).
    - Skills library (settings.skills_dir / skills_host_dir) is mounted ro.
    """
    host_root = getattr(settings, "host_project_dir", "") or ""
    container_root = str(Path.cwd().resolve())
    prefix_map = (container_root, host_root) if host_root else None

    skills_dir = getattr(settings, "skills_host_dir", "") or getattr(settings, "skills_dir", "") or ""
    if skills_dir and host_root and not getattr(settings, "skills_host_dir", ""):
        skills_dir = to_host_path(skills_dir, prefix_map)

    workspace_root = getattr(settings, "workspace_root", "") or ""
    if not workspace_root and host_root:
        workspace_root = str(Path.cwd() / ".workspaces")

    return DockerSandbox(
        image=getattr(settings, "sandbox_image", "ctf-sandbox"),
        challenge_dir=challenge_dir,
        memory_limit=getattr(settings, "container_memory_limit", "16g"),
        cpu_limit=float_setting(settings, "container_cpu_limit"),
        command_timeout_s=int_setting(settings, "sandbox_command_timeout_s"),
        workspace_root=workspace_root,
        skills_host_dir=skills_dir,
        host_prefix_map=prefix_map,
        proxy_url=getattr(settings, "sandbox_proxy_url", "") or "",
    )


@dataclass
class DockerSandbox:
    """Isolated Docker container for a single solver agent.

    host_prefix_map: optional (container_prefix, host_prefix) pair used to
    translate local paths to DOCKER HOST paths when the agent itself runs
    inside a container (docker-outside-of-docker via /var/run/docker.sock).
    """

    image: str
    challenge_dir: str
    memory_limit: str = "16g"
    cpu_limit: float = 2.0
    command_timeout_s: int = 900
    workspace_dir: str = ""
    workspace_root: str = ""  # base dir for workspaces (default: system temp)
    skills_host_dir: str = ""  # host path of skill library, mounted at /skills:ro
    host_prefix_map: tuple[str, str] | None = None
    proxy_url: str = ""  # HTTP(S) proxy injected into the container env
    _container: Any = field(default=None, repr=False)
    _docker: Any = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Teardown runs at most once, even if several solvers share a sandbox and
    # each calls stop() as the swarm winds down.
    _stopping: bool = field(default=False, repr=False)

    @property
    def is_running(self) -> bool:
        """True once a container exists. Public replacement for poking `_container`."""
        return self._container is not None

    @property
    def container_id(self) -> str:
        """The Docker container ID, available after start()."""
        if not self._container:
            raise RuntimeError("Sandbox not started")
        return self._container.id

    def _parse_memory_limit(self) -> int:
        s = self.memory_limit.strip().lower()
        try:
            if s.endswith("g"):
                return int(s[:-1]) * 1024 * 1024 * 1024
            if s.endswith("m"):
                return int(s[:-1]) * 1024 * 1024
            return int(s)
        except (ValueError, IndexError):
            logger.warning("Invalid memory_limit %r, defaulting to 16GB", self.memory_limit)
            return 16 * 1024 * 1024 * 1024

    def _to_host(self, path: str) -> str:
        """Translate a container-local path to a docker-host path for bind mounts."""
        return to_host_path(path, self.host_prefix_map)

    def _make_workspace(self) -> str:
        if self.workspace_root:
            root = Path(self.workspace_root)
            root.mkdir(parents=True, exist_ok=True)
            ws = root / f"{WORKSPACE_PREFIX}{uuid.uuid4().hex[:12]}-{RUN_ID}"
            ws.mkdir()
            return str(ws)
        return tempfile.mkdtemp(prefix=WORKSPACE_PREFIX)

    def _sanitized_metadata(self, meta_yml: str) -> str | None:
        """Copy metadata.yml into the workspace dir without flag/flags keys.

        Returns None when sanitization could not be completed, meaning "do not mount
        anything". The previous fallback returned the ORIGINAL path, so a failed
        write — a full disk, a permissions error, an unparseable YAML — put the answer
        key inside the solver's sandbox after one warning line. In offline/bench mode
        that silently invalidates the whole run: a solver can `cat
        /challenge/metadata.yml` and read the flag, and the reported solve rate then
        measures nothing.

        Omitting the mount is a graceful degradation by comparison: the challenge
        description is already in the solver's system prompt (build_prompt renders it
        from ChallengeMeta), so the file inside the sandbox is a convenience, not the
        only copy. Losing a convenience beats leaking the answer.
        """
        try:
            import yaml

            data = yaml.safe_load(Path(meta_yml).read_text(encoding="utf-8")) or {}
            data.pop("flag", None)
            data.pop("flags", None)
            dest = Path(self.workspace_dir) / ".metadata.sanitized.yml"
            dest.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — must not mount the original instead
            logger.error(
                "Could not sanitize metadata (%s: %s) — NOT mounting it. The answer key "
                "would otherwise be readable from inside the sandbox, which invalidates "
                "offline scoring. The solver still has the description in its prompt.",
                type(e).__name__, e,
            )
            return None
        # Verify the key really is gone before mounting. A future edit that renames
        # the answer-key field would otherwise leak it silently, and this is the one
        # check whose failure mode is "the benchmark result is meaningless".
        #
        # The check inspects VALUES for a flag-shaped literal, not the document text
        # for the word "flag": challenge descriptions legitimately discuss flags
        # ("recover the flag from the PNG"), and a substring test rejected those
        # challenges outright, which is a worse failure than the leak it guards.
        try:
            sanitized_data = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
        except Exception as e:  # noqa: BLE001 — unreadable or unparseable
            logger.error("Sanitized metadata unreadable (%s) — not mounting it", e)
            return None
        leaked = _find_flag_literal(sanitized_data)
        if leaked is not None:
            logger.error(
                "Sanitized metadata still contains a flag-shaped value (%s) — not "
                "mounting it. The answer-key field name is probably not `flag`/`flags`; "
                "add it to the pop list in _sanitized_metadata().", leaked,
            )
            return None
        return str(dest)

    def _build_config(self, binds: list[str], env: list[str], with_devices: bool) -> dict:
        host_config: dict[str, Any] = {
            "Binds": binds,
            "ExtraHosts": ["host.docker.internal:host-gateway"],
            "CapAdd": ["SYS_ADMIN", "SYS_PTRACE"],
            "SecurityOpt": ["seccomp=unconfined"],
            "Memory": self._parse_memory_limit(),
            "NanoCpus": int(max(0.1, self.cpu_limit) * 1e9),
            # Fork bombs are a real hazard when running untrusted challenge
            # binaries; cap the process count instead of relying on the OOM killer.
            "PidsLimit": 4096,
        }
        if with_devices:
            # Needed for losetup/mount-based forensics. Absent on some Docker
            # Desktop hosts, so creation is retried without it (see start()).
            host_config["Devices"] = [{
                "PathOnHost": "/dev/loop-control",
                "PathInContainer": "/dev/loop-control",
                "CgroupPermissions": "rwm",
            }]
        return {
            "Image": self.image,
            "Cmd": ["sleep", "infinity"],
            "WorkingDir": "/challenge",
            "Tty": False,
            "Labels": {CONTAINER_LABEL: "true", RUN_LABEL: RUN_ID},
            "Env": env,
            "HostConfig": host_config,
        }

    def _prepare_binds(self) -> list[str]:
        """Blocking filesystem prep: workspace, distfile/metadata probes, sanitizing.

        Kept synchronous and driven through `asyncio.to_thread` — with 40 sandboxes
        starting at once these stat/read/write calls would otherwise stall the
        coordinator's event loop.
        """
        self.workspace_dir = self._make_workspace()

        challenge_root = Path(self.challenge_dir).resolve()
        distfiles = str(challenge_root / "distfiles")
        meta_yml = str(challenge_root / "metadata.yml")

        binds: list[str] = [f"{self._to_host(self.workspace_dir)}:/challenge/workspace:rw"]
        if Path(distfiles).exists():
            binds.append(f"{self._to_host(distfiles)}:/challenge/distfiles:ro")
        if Path(meta_yml).exists():
            # Mount a SANITIZED copy: strip answer-key fields (flag/flags)
            # so offline solvers can't read the solution from the sandbox.
            # None means sanitization failed — mount nothing rather than leak the key.
            sanitized = self._sanitized_metadata(meta_yml)
            if sanitized is not None:
                binds.append(f"{self._to_host(str(sanitized))}:/challenge/metadata.yml:ro")
        if self.skills_host_dir:
            binds.append(f"{self.skills_host_dir}:/skills:ro")
        return binds

    async def start(self) -> None:
        sem = _start_semaphore or asyncio.Semaphore(50)
        async with sem:
            self._docker = new_docker_client()
            try:
                binds = await asyncio.to_thread(self._prepare_binds)

                env: list[str] = []
                if self.proxy_url:
                    env += [
                        f"HTTP_PROXY={self.proxy_url}",
                        f"HTTPS_PROXY={self.proxy_url}",
                        "NO_PROXY=localhost,127.0.0.1,host.docker.internal,.local",
                    ]

                try:
                    self._container = await self._docker.containers.create(
                        self._build_config(binds, env, with_devices=True)
                    )
                except aiodocker.exceptions.DockerError as e:
                    # /dev/loop-control is not available on every Docker host.
                    if "loop-control" not in str(e):
                        raise
                    logger.warning("Retrying sandbox creation without /dev/loop-control")
                    self._container = await self._docker.containers.create(
                        self._build_config(binds, env, with_devices=False)
                    )
                await self._container.start()
                await _track_start()

                info = await self._container.show()
                logger.info("Sandbox started: %s", info["Id"][:12])
            except BaseException:
                # Never leak a docker client, a half-created container, or a
                # workspace directory when startup fails partway through.
                await self.stop()
                raise

    async def exec(self, command: str, timeout_s: int = 300) -> ExecResult:
        if not self._container:
            raise RuntimeError("Sandbox not started")

        # Never let a caller (or a model-supplied timeout_seconds) exceed the
        # configured per-command ceiling.
        ceiling = int(self.command_timeout_s or 0)
        if ceiling > 0 and (timeout_s <= 0 or timeout_s > ceiling):
            timeout_s = ceiling

        async with self._lock:
            try:
                return await self._exec_inner(command, timeout_s)
            except aiodocker.exceptions.DockerError as e:
                # Container was deleted (e.g., sibling solver found the flag)
                return ExecResult(exit_code=-1, stdout="", stderr=f"Container gone: {e}")

    async def _exec_inner(self, command: str, timeout_s: int) -> ExecResult:
        # Wrap with coreutils `timeout` so the container kills the process on
        # expiry: TERM first (lets tools clean up), KILL 5s later as the safety
        # net. Mirrors the wrapper the claude-sdk path applies in
        # ClaudeSolver._wrap_for_sandbox, so neither backend can leave an
        # orphaned process burning the container's quota.
        wrapped = f"timeout -k 5 {timeout_s} bash -c {shlex.quote(command)}"
        exec_instance = await self._container.exec(
            cmd=["bash", "-c", wrapped],
            stdout=True,
            stderr=True,
            tty=False,
        )

        stream = exec_instance.start(detach=False)
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        async def _collect() -> None:
            while True:
                msg = await stream.read_out()
                if msg is None:
                    break
                if msg.stream == 1:
                    stdout_chunks.append(msg.data)
                else:
                    stderr_chunks.append(msg.data)

        try:
            # Give extra margin beyond the container-side timeout
            await asyncio.wait_for(_collect(), timeout=timeout_s + 30)
        except TimeoutError:
            try:
                await stream.close()
            except Exception:
                pass
            return ExecResult(
                exit_code=-1,
                stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                stderr="Command timed out",
            )

        inspect = await exec_instance.inspect()
        exit_code = inspect.get("ExitCode", 0)

        return ExecResult(
            exit_code=exit_code,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        )

    async def read_file(self, path: str) -> str | bytes:
        """Read a file from the container. Returns str for text, bytes for binary."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        try:
            tar = await asyncio.wait_for(
                self._container.get_archive(path),
                timeout=30,
            )
        except TimeoutError as e:
            raise TimeoutError(f"Timed out reading {path}") from e

        # aiodocker 0.26.0 returns tarfile.TarFile directly
        with tar:
            for member in tar:
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        data = f.read()
                        try:
                            return data.decode("utf-8")
                        except UnicodeDecodeError:
                            return data
        raise FileNotFoundError(f"No file found at {path}")

    async def read_file_bytes(self, path: str) -> bytes:
        """Read a file from the container as raw bytes."""
        result = await self.read_file(path)
        if isinstance(result, str):
            return result.encode("utf-8")
        return result

    async def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file into the container via tar archive."""
        if not self._container:
            raise RuntimeError("Sandbox not started")

        if isinstance(content, str):
            content = content.encode("utf-8")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=Path(path).name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        buf.seek(0)

        try:
            await asyncio.wait_for(
                self._container.put_archive(str(Path(path).parent), buf.getvalue()),
                timeout=30,
            )
        except TimeoutError as e:
            raise TimeoutError(f"Timed out writing {path}") from e

    async def copy_from(self, container_path: str, host_path: str) -> None:
        """Copy a file from the container to the host."""
        data = await self.read_file_bytes(container_path)
        await asyncio.to_thread(_write_bytes, host_path, data)

    async def stop(self) -> None:
        """Tear down container, client and workspace exactly once.

        The whole teardown runs as ONE shielded unit. Its steps are
        order-dependent — deleting the container needs the aiodocker client's
        connector, which the next step closes — so a cancellation must never
        land between them. Shielding per-step (an earlier attempt) made things
        worse: swallowing the cancel let step 2 close the connector out from
        under step 1's still-in-flight delete, so the container was left running.

        The swarm cancels siblings the instant one solver wins, so this path is
        hit on every solved challenge.
        """
        if self._stopping:
            return
        self._stopping = True
        # Created explicitly rather than letting `shield()` wrap the coroutine, so
        # there is a handle to register. asyncio holds only a WEAK reference to
        # tasks: a shielded inner task whose caller was cancelled had nothing
        # referencing it, so it could be garbage collected mid-`container.delete()`,
        # and `asyncio.run`'s shutdown cancels all pending tasks — destroying it
        # between deleting the container and closing the client, or before the
        # workspace rmtree. Registering it keeps it alive and lets the coordinator
        # await outstanding teardowns before the loop closes.
        inner = asyncio.ensure_future(self._stop_inner())
        _register_detached(inner)
        try:
            await asyncio.shield(inner)
        except asyncio.CancelledError:
            # The shielded inner task keeps running to completion; there is
            # nothing left for this frame to do but let the cancel through.
            logger.warning(
                "sandbox teardown detached (caller cancelled) — it will keep running "
                "and must be awaited before the event loop closes"
            )
            raise

    async def _stop_inner(self) -> None:
        if self._container is not None:
            container, self._container = self._container, None
            try:
                await container.delete(force=True)
            except Exception as e:  # noqa: BLE001 — teardown is best-effort
                logger.debug("container delete failed: %s", e)
            await _track_stop()

        if self._docker is not None:
            docker, self._docker = self._docker, None
            try:
                await docker.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("docker client close failed: %s", e)

        if self.workspace_dir:
            workspace, self.workspace_dir = self.workspace_dir, ""
            try:
                # Off the loop: a workspace can hold gigabytes of carved or
                # extracted challenge data.
                await asyncio.to_thread(_remove_workspace, workspace)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not remove workspace %s: %s", workspace, e)
        logger.info("Sandbox stopped")


def _write_bytes(host_path: str, data: bytes) -> None:
    Path(host_path).parent.mkdir(parents=True, exist_ok=True)
    Path(host_path).write_bytes(data)


def _remove_workspace(workspace: str) -> None:
    import shutil

    try:
        shutil.rmtree(workspace, ignore_errors=True)
    except OSError as e:
        logger.warning("Could not remove workspace %s: %s", workspace, e)
