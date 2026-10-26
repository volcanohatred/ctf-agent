"""Pydantic Settings — credentials from .env file + environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # CTFd
    ctfd_url: str = "http://localhost:8000"
    ctfd_user: str = "admin"
    ctfd_pass: str = "admin"
    ctfd_token: str = ""

    # API Keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Claude Code CLI routing (claude-sdk backend).
    # Set anthropic_base_url to route ALL claude-cli traffic through your
    # unified gateway (Anthropic-protocol endpoint). Leave empty for the
    # official Anthropic API. anthropic_auth_token defaults to the gateway key.
    anthropic_base_url: str = ""
    anthropic_auth_token: str = ""

    # Provider-specific (optional, for Bedrock/Azure/Zen fallback)
    aws_region: str = "us-east-1"
    aws_bearer_token: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    opencode_zen_api_key: str = ""

    # Gateway provider (OpenAI-compatible unified gateway, e.g. deepseek/kimi/glm/qwen)
    gateway_base_url: str = ""
    gateway_api_key: str = ""

    # Skill knowledge base (see backend/skills.py)
    # skills_dir: path readable by the AGENT process (triage/index rendering)
    # skills_host_dir: same directory as seen by the DOCKER HOST (bind-mounted
    #   into sandboxes at /skills:ro). Only differs when the agent itself runs
    #   in a container (docker-outside-of-docker). Empty = use skills_dir.
    skills_dir: str = ""
    skills_host_dir: str = ""

    # Docker-outside-of-docker path translation. When the agent runs inside a
    # container with /var/run/docker.sock mounted, sandbox bind mounts must use
    # HOST paths. Set to the absolute project root on the host (e.g. D:/code/ctf).
    # Empty = agent runs directly on the host (no translation).
    host_project_dir: str = ""
    # Base directory for per-sandbox workspaces. Empty = system temp (local
    # mode) or <project>/.workspaces (when host_project_dir is set).
    workspace_root: str = ""

    # Infra
    sandbox_image: str = "ctf-sandbox"
    max_concurrent_challenges: int = 10
    max_attempts_per_challenge: int = 3
    container_memory_limit: str = "16g"
    # Optional HTTP(S) proxy injected into sandbox containers (e.g.
    # "http://host.docker.internal:7890" so solvers can reach github etc.).
    sandbox_proxy_url: str = ""

    # Solver behavior
    # Wall-clock budget per solver attempt (seconds). On expiry the attempt is
    # interrupted, findings are harvested (Cairn-style conclude), and the solver
    # is bumped with fresh direction. 0 = no budget.
    solver_time_budget_s: int = 1500
    # Start injecting skill-library hints after this many failed attempts (bumps).
    # Default 2 = first retry is lean (rotation only); skills join from the 2nd failure.
    skill_after_bumps: int = 2
    # Coordinator model — use your STRONGEST model here (it coaches weaker
    # solvers). Empty = CLI --coordinator-model, else built-in default.
    coordinator_model: str = ""
    # Proactive coordinator patrol: scan active solvers every N seconds for
    # stuck patterns (quiet too long / looping), alert + advise them.
    patrol_interval_s: int = 120
    # A solver with no new trace activity for this many seconds counts as stuck.
    patrol_stuck_quiet_s: int = 240
    # After this many failed bump attempts across a swarm with no winner,
    # spawn a stronger-model solver (escalation ladder). 0 disables.
    escalate_after_bumps: int = 8
    # Long findings posted to the blackboard get an LLM key-info summary for
    # sibling injection (full text is always kept; nothing is truncated).
    summarize_threshold_chars: int = 900
    # Model used for blackboard summarization (fast/cheap gateway model).
    summarizer_model: str = "deepseek-v4-flash"
    # Operator web dashboard (live swarm view + guidance mailbox UI).
    # 0 disables. Default 8080.
    dashboard_port: int = 8080
    dashboard_host: str = "0.0.0.0"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
