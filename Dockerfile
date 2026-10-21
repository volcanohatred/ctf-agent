# CTF Agent — orchestrator image (coordinator + solver drivers).
# Runs the agent in Docker and spawns sibling sandbox containers via the
# host's docker socket (docker-outside-of-docker).
#
# Build:  docker build -t ctf-agent .
# Run:    see docker-compose.yml (recommended) or README.md

# Node.js runtime for the Claude Code CLI (pulled via the docker mirror —
# faster and more reliable than the NodeSource apt repo).
FROM node:22-slim AS node

FROM python:3.14-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Claude Code CLI: allow bypassPermissions as root (agent is the sandbox)
    IS_SANDBOX=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI (static binary) — the Claude solver rewrites Bash calls to
# `docker exec` against sandbox containers via the mounted host socket.
ARG DOCKER_CLI_VERSION=27.3.1
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
      | tar -xz --strip-components=1 -C /usr/local/bin docker/docker \
    && docker --version

# Node.js + npm (copied from the node image)
COPY --from=node /usr/local /usr/local

# Claude Code CLI (required by claude-agent-sdk solvers/coordinator).
# npmmirror for fast downloads; fall back silently to the default registry.
RUN npm config set registry https://registry.npmmirror.com || true \
    && npm install -g @anthropic-ai/claude-code \
    && claude --version

# uv (fast python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
# China PyPI mirror for fast, reliable builds (override with --build-arg).
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_INDEX_URL=${UV_INDEX_URL}

WORKDIR /app

# Install python dependencies first (better layer caching)
COPY pyproject.toml ./
COPY backend ./backend
RUN uv sync --no-dev

# Runtime layout (mount points):
#   /app/challenges   — challenge dirs (metadata.yml + distfiles/)
#   /app/logs         — traces, benchmark reports
#   /app/.workspaces  — per-sandbox workspaces (host-visible for DooD binds)
#   /skills           — read-only CTF knowledge library (optional)
RUN mkdir -p /app/challenges /app/logs /app/.workspaces /skills

ENTRYPOINT ["uv", "run", "--no-sync"]
# Default: start the coordinator. Override the command, e.g.:
#   docker compose run --rm agent ctf-solve --challenge challenges/baby-pwn -v
#   docker compose run --rm agent ctf-bench --challenges-dir challenges -v
CMD ["ctf-solve", "--msg-port", "9400"]
