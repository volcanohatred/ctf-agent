# CTF Agent

Autonomous CTF (Capture The Flag) solver that races multiple AI models against challenges in parallel — now with a **mounted skill knowledge base**, **wall-clock degradation**, and a **fully Dockerized control plane**.

Forked from Veria Labs' ctf-agent (52/52, 1st place at BSidesSF 2026 CTF), extended with ideas from Cairn (dual-phase timeout harvesting, structured negative-knowledge blackboard).

## What's Different in This Fork

| Area | Original | This fork |
|---|---|---|
| Model lineup | Claude + OpenAI (Codex) | **Claude Code CLI × unified gateway** — all solvers drive gateway models (kimi-k3 / deepseek-v4-pro / qwen3.8-max) through the Anthropic protocol; OpenAI-compatible gateway endpoint as automatic fallback |
| Codex backend | ✅ | ❌ removed |
| Knowledge | none | **Skill library** mounted at `/skills` — model works alone first, escalating hints are injected when stuck |
| Long runs | context grows unbounded | **Context compaction** (Pydantic backend) + **wall-clock budget** with Cairn-style conclude→bump |
| Findings | plain text bus | **Structured blackboard** (fact / excluded dead-ends / coordinator) |
| Deployment | run on host | **Agent runs in Docker** (docker-outside-of-docker), or still locally |
| Evaluation | none | **`ctf-bench`** offline benchmark with local answer-key verification |

## How It Works

```
                        +-----------------+
                        |  CTFd Platform  |        (or offline challenge dir)
                        +--------+--------+
                                 |
                        +--------v--------+
                        |  Poller (5s)    |
                        +--------+--------+
                                 |
                        +--------v--------+
                        | Coordinator LLM |
                        | (Claude CLI →   |
                        |  unified gateway)|
                        +--------+--------+
                                 |
               +-----------------+-----------------+
               |                 |                 |
      +--------v-------+ +-------v--------+ +------v----------+
      | Swarm: chall-1 | | Swarm: chall-2  | |      ...        |
      |  kimi-k3       | |                | |                 |
      |  deepseek-v4-pro | |              | |                 |
      |  qwen3.8-max   | |                | |                 |
      |  (all via Claude| |               | |                 |
      |   CLI + gateway)| |               | |                 |
      +--------+-------+ +--------+-------+ +-----------------+
               |                  |
      +--------v-------+ +--------v-------+
      | Docker Sandbox | | Docker Sandbox |     /skills (ro) mounted
      | ctf-sandbox    | | ctf-sandbox    | <-- in every sandbox
      +----------------+ +----------------+
```

**Model-first, skills-second**: solvers start with raw capability. After repeated
failures (`SKILL_AFTER_BUMPS`, default 1) the swarm injects escalating hints from
the skill library: L1 pack index → L2 full file menu → L3 similar Chinese
writeups. Solvers then read concrete techniques from `/skills/` themselves.

**Wall-clock degradation (Cairn-style)**: each solver attempt has a time budget
(`SOLVER_TIME_BUDGET_S`, default 1500s). On expiry the attempt is interrupted,
findings are harvested to the blackboard as *excluded dead-ends*, and the solver
resumes with a fresh direction — no silent 6-hour stalls.

## Quick Start (Docker — recommended)

```bash
# 0. Pull prebuilt images (fastest — Aliyun ACR, great for CN networks):
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest
docker tag  crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest   ctf-agent:latest
docker tag  crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest ctf-sandbox:latest
#    (alternative: ctf-agent is also on GHCR — ghcr.io/volcanohatred/ctf-agent:latest)
#    (or build locally: docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .)

# 1. Configure
cp .env.example .env
# Edit .env:
#   ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN  — unified gateway (Anthropic protocol)
#   GATEWAY_BASE_URL / GATEWAY_API_KEY         — OpenAI-compatible endpoint (fallback)
#   SKILLS_HOST_DIR — host path of your skill library root (contains ctf-web/ etc., optional)
#   HOST_PROJECT_DIR — absolute path of THIS repo on the docker host
#     (the agent spawns sandbox containers via the host docker socket)

# 2a. Full competition (coordinator mode)
docker compose run --rm --service-ports agent

# 2b. Single challenge (bundled smoke test — no CTFd needed)
docker compose run --rm agent ctf-solve --challenge challenges/example-static --no-submit -v

# 2c. Offline benchmark over a local challenge set
docker compose run --rm agent ctf-bench --challenges-dir challenges --concurrency 2 -v

# Operator hint to the running coordinator (second terminal)
docker compose exec agent uv run --no-sync ctf-msg "the JWT uses an enum kid — try SQLi there"
```

## Quick Start (local, no agent container)

```bash
uv sync
docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .
cp .env.example .env   # set SKILLS_DIR to the skill library path
uv run ctf-solve --ctfd-url https://ctf.example.com --ctfd-token ... -v
```

## Solver Models

Default lineup (edit `backend/models.py: DEFAULT_MODELS`, or pass `--models`):

| Spec | Backend | Notes |
|---|---|---|
| `claude-sdk/qwen3.8-max` | Claude Code CLI → gateway (Anthropic protocol) | Solver A |
| `claude-sdk/deepseek-v4-flash` | Claude Code CLI → gateway | Solver B (fast) |
| `claude-sdk/kimi-k3` | Claude Code CLI → gateway | Solver C |
| `claude-sdk/glm-5.3` | Claude Code CLI → gateway | Solver D |

All four run through the **unified gateway**: set `ANTHROPIC_BASE_URL` +
`ANTHROPIC_AUTH_TOKEN` in `.env` (if the gateway runs on the host, use
`http://host.docker.internal:<port>/anthropic` — the agent is containerized).
On quota/rate-limit, a solver automatically retries the SAME model via the
OpenAI-compatible endpoint (`GATEWAY_BASE_URL`) reusing its sandbox
(`QUOTA_FALLBACK` in `backend/agents/swarm.py`).

Verified on the regression suite (10 easy + 4 hard + 4 harder + 3 boss-level,
incl. glibc-2.39 safe-linking tcache poison, tcache double-free with key-clear
bypass, DSA nonce-MSB hidden-number-problem, and an MBA-obfuscated reverse
checker): 19/20 on a live local CTFd (one intentionally-unsolvable knapsack
excepted — it's the control case that drives solvers into the skill library).

## Verification results (local CTFd, gateway models)

| Suite | Result | Notes |
|---|---|---|
| challenges/example-static | ✅ | smoke |
| challenges/regression (10) | 10/10 | 18s–10min each |
| challenges/hard (4) | 4/4 | coppersmith / anti-debug / flask-cookie / heap-tcache |
| challenges/hard2 (4) | 4/4 | franklin-reiter / custom-VM reverse / multi-layer forensics pcap / tcache double-free (key bypass) |
| challenges/hard3 (3) | 3/3 | crypto-hnp (DSA nonce MSB leak) / reverse-mba / web-pickle-restricted |
| Skill engagement | ✅ | when genuinely stuck, solvers grep & read `/skills/` docs on their own (knapsack → crypto lattice doc; mba → mba-simplification.md); bump-triggered hints escalate L1→L2→L3 with real file paths |

Rebuild/regenerate: `python3 scripts/gen_regression.py` (inside the sandbox),
`gen_hard.py`, `gen_hard2.py`, `gen_hard3.py`; seed a local CTFd with
`scripts/seed_ctfd.py`.

## Skill Library

Point `SKILLS_DIR` (local) / `SKILLS_HOST_DIR` (docker) at a directory containing
the skill packs (`ctf-web/INDEX.md`, `ctf-pwn/INDEX.md`, ... `zh-knowledge/`).
The router (`backend/skills.py`) triages each challenge by category, keywords,
and distfile extensions; hints are injected on failure bumps at escalating
levels. If the directory is missing or has no packs, the feature disables
itself gracefully.

## Offline Challenge Format & Benchmark

Each challenge is a directory with `metadata.yml` (+ optional `distfiles/`):

```yaml
name: baby-pwn
category: pwn
value: 100
description: "nc to the service and pwn it"
connection_info: "nc localhost 31337"   # rewritten to host.docker.internal in sandboxes
flag: "flag{r3al_answ3r}"                # optional answer key → local verification
```

`ctf-bench` runs swarms with bounded concurrency, verifies flags against answer
keys, and writes a JSON report (`logs/bench-<ts>.json`) with per-challenge
time/cost and overall solve rate — use it for regression testing after changes.

## Sandbox Tooling

| Category | Tools |
|----------|-------|
| **Binary** | radare2, GDB, objdump, binwalk, pyghidra |
| **Pwn** | pwntools, ROPgadget, angr, unicorn, capstone |
| **Crypto** | SageMath, RsaCtfTool, z3, gmpy2, cado-nfs, flatter |
| **Forensics** | volatility3, sleuthkit, foremost, exiftool |
| **Stego** | steghide, stegseek, zsteg, ImageMagick, tesseract |
| **Web** | curl, nmap, requests, flask |
| **Misc** | ffmpeg, sox, Pillow, PyTorch, podman |

## Configuration

See `.env.example` for all settings. Key tuning knobs:

| Var | Default | Meaning |
|---|---|---|
| `SOLVER_TIME_BUDGET_S` | `1500` | Per-attempt wall clock; 0 disables |
| `SKILL_AFTER_BUMPS` | `1` | Inject skill hints after N failed attempts |
| `CONTAINER_MEMORY_LIMIT` | `16g` | Per-sandbox memory |
| `MAX_CONCURRENT_CHALLENGES` | `10` | Parallel swarms |

## Requirements

- Docker (sandbox containers; agent itself can also run containerized)
- Python 3.14+ (local mode), `uv`
- The unified gateway: `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` (Anthropic
  protocol, used by the Claude Code CLI solvers) and `GATEWAY_BASE_URL` +
  `GATEWAY_API_KEY` (OpenAI-compatible fallback)
- `claude` CLI (bundled in the agent Docker image; needed on host for local mode)

## Acknowledgements

- [Veria Labs ctf-agent](https://verialabs.com) — original architecture
- Cairn — dual-phase timeout harvesting and blackboard coordination ideas
- [es3n1n/Eruditus](https://github.com/es3n1n/Eruditus) — CTFd interaction helpers
