"""Shared dependency types — avoids circular imports between agents and tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from backend.cost_tracker import CostTracker
from backend.ctfd import CTFdClient
from backend.sandbox import DockerSandbox

if TYPE_CHECKING:
    from backend.message_bus import ChallengeMessageBus

# Type for the deduped submit callback: (flag) -> (display, is_confirmed)
SubmitFn = Callable[[str], Coroutine[Any, Any, tuple[str, bool]]]


@dataclass
class SolverDeps:
    sandbox: DockerSandbox
    ctfd: CTFdClient
    challenge_dir: str
    challenge_name: str
    workspace_dir: str
    use_vision: bool
    cost_tracker: CostTracker | None = None
    confirmed_flag: str | None = None
    message_bus: ChallengeMessageBus | None = None
    model_spec: str = ""
    submit_fn: SubmitFn | None = None  # Deduped flag submission via swarm
    no_submit: bool = False
    notify_coordinator: Callable[[str], Coroutine[Any, Any, None]] | None = None
    # Hostnames web_fetch may reach even though they resolve privately.
    web_fetch_allow_hosts: str = ""


@dataclass
class CoordinatorDeps:
    ctfd: CTFdClient
    cost_tracker: CostTracker
    settings: Any
    model_specs: list[str] = field(default_factory=list)
    challenges_root: str = "challenges"
    no_submit: bool = False
    max_concurrent_challenges: int = 10

    msg_port: int = 0  # 0 = auto-pick free port (breaks ctf-msg's 9400 default)
    msg_host: str = "127.0.0.1"

    # Runtime state
    coordinator_inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    operator_inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    swarms: dict[str, Any] = field(default_factory=dict)
    swarm_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    results: dict[str, dict] = field(default_factory=dict)
    challenge_dirs: dict[str, str] = field(default_factory=dict)
    challenge_metas: dict[str, Any] = field(default_factory=dict)
    # How many times each challenge has been auto-spawned. Bounds the periodic
    # re-spawn so a genuinely hopeless challenge is not re-queued forever.
    spawn_attempts: dict[str, int] = field(default_factory=dict)
    # Archived (finished) swarm summaries — kept so the dashboard still shows
    # solved/completed challenges after their live swarm is retired.
    swarms_history: dict[str, dict] = field(default_factory=dict)
