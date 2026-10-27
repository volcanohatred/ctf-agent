# CTF Agent

Autonomous CTF (Capture The Flag) solver that races multiple AI models against challenges in parallel — now with a **mounted skill knowledge base**, **wall-clock degradation**, and a **fully Dockerized control plane**.

> 📦 **安装指南（中文，含国内镜像一键拉取）请见 [INSTALL.md](INSTALL.md)。**

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

**Wall-clock degradation (Cairn-style, two phases)**: each solver attempt has a
time budget (`SOLVER_TIME_BUDGET_S`, default 1500s). On expiry the turn is
**interrupted through the SDK** — not task-cancelled, so the session survives —
and a second phase resumes that same session with a conclude prompt under its own
independent budget (`CONCLUDE_TIMEOUT_S`, default 90s), with all tools denied.
The distilled `[excluded]` / `[fact]` / `[artifacts]` / `[next]` state card is
what gets harvested to the blackboard. If the conclude phase fails, **nothing is
written** — a partial or invented harvest is worse than none, because siblings
treat it as ground truth. The solver then resumes with a fresh direction.

**Fresh restart with a distilled journal**: after a GAVE_UP the round is distilled
into one journal entry, the session is *forgotten* (not merely reconnected — the
Claude backend's `reset_session` keeps the session id and would resume the old
history, defeating the point), and a clean attempt starts carrying only that
state plus the team digest, the next untried technique, and the skill hint. Every
`FRESH_RESTART_ROUNDS` (default 5) the journal is merged by a second model call
that de-duplicates it, so the carried state stays bounded instead of growing.

Two guards make that bound hold when the summarizer is unavailable, which against
a flaky gateway is the normal case rather than the exception:

- **Structural de-duplication** (`backend/dedup.py`) runs *before* the model
  merge, on the journal, on the dead-end ledger, and on the team digest. It
  normalizes case, punctuation and CJK-preserving word content, and collapses
  volatile literals (`0x7f8a1c000000` and `0x7f9b2d000000` are the same finding
  under a different ASLR slide). A terse restatement of a recorded finding is
  dropped; a longer one *replaces* it, so detail is never lost to an earlier
  summary. Identical or reordered entries cannot accumulate no matter what the
  model does.
- **A failed merge keeps the bounded list** rather than collapsing it into one
  joined string. The previous behaviour did the opposite of the mechanism's
  purpose: with the gateway down, every round appended to a single ever-growing
  entry.

**Team digest**: what all solvers on a challenge have established is merged into
one de-duplicated state and injected into every restart. It is read from the
blackboard (which retains history) rather than the per-model latest finding, so
early discoveries no longer drop out. It is recomputed at most once per bump
generation and shared, so four restarting solvers cost one summarizer call
instead of four. A restarting solver has had its session wiped, so its own
earlier findings are included deliberately — it no longer remembers them.

**Coupling worth knowing**: one distil→merge cycle costs `FRESH_RESTART_ROUNDS`
bumps, so `MAX_ATTEMPTS_PER_CHALLENGE` must be at least
`2 × FRESH_RESTART_ROUNDS + 1` for the merged state to drive more than a single
round. The defaults (12 and 5) allow two full cycles. A misalignment is reported
at startup by `validate_tuning` instead of silently wasting the merge.

**Fail fast on a dead gateway**: `STARTUP_HEALTH_CHECK` probes both endpoints
before anything launches. This matters more than it sounds — the Claude CLI
reports `API Error: Connection refused` as an *ordinary assistant text block*,
so an unreachable gateway used to look exactly like a hard challenge: every
round "gave up", the swarm rotated techniques and escalated skill hints at a
networking problem, and the benchmark report said `timeout after 600s`. Those
texts are now classified as `INFRA_ERROR`/`QUOTA_ERROR`, never published to the
blackboard, and `MAX_INFRA_FAILURES` consecutive ones retire the swarm. A round
that executed **zero tool calls** is treated the same way, and `ctf-bench`
reports `usable_run: false` when infrastructure rather than the models decided
the outcome.

## Quick Start (Docker — recommended)

```bash
# 0. Pull prebuilt images (Aliyun ACR — fast in CN):
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest
docker tag  crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest   ctf-agent:latest
docker tag  crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest ctf-sandbox:latest
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

**Windows hosts**: `backend/sandbox.py` auto-detects Docker Desktop's named pipe
(`npipe:////./pipe/docker_engine`) when `DOCKER_HOST` is unset. This matters
because aiodocker's own fallback is `https://127.0.0.1:2376`, which Docker
Desktop does not expose — without the detection, local mode failed on every
Windows host with `Cannot connect to Docker Engine`. If you use a remote engine
or a non-default context, set `DOCKER_HOST` explicitly; it always wins.

Single-challenge mode also takes a hard wall-clock cap, so it cannot run away:

```bash
uv run ctf-solve --challenge challenges/regression/misc-encoding --no-submit --time-limit 600 -v
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

The report also carries `retire_reason`, `infra_failure` and `bumps` per
challenge, plus an overall `infra_failures` count and a `usable_run` flag.
**Check `usable_run` before reading a solve rate**: when it is `false`,
infrastructure rather than the models decided the outcome, and the number means
nothing. `ctf-bench` also warns up front when `--time-limit` is below
`SOLVER_TIME_BUDGET_S`, a combination in which the per-round budget and its
conclude harvest can never fire, so every challenge ends as a bare outer timeout.

## Tests

```bash
uv sync                      # or: pip install -e ".[dev]"
uv run pytest -q             # 337 tests, no Docker / no LLM / no network
uv run ruff check .
uv run ty check backend
uv run python tests/import_check.py     # every module imports
uv run python tests/wiring_check.py     # ~80 wiring assertions (auth, hooks, wrapping)
uv run python tests/e2e_dead_gateway.py # a dead gateway must fail in seconds
```

The suite is deliberately dependency-light: swarms are driven by scripted fake
solvers, `httpx` goes through a `MockTransport` subclass, and DNS cases use
numeric literals. That is what makes the scheduler logic testable at all — the
round loop, two-phase timeout harvest, dead-end ledger, escalation ladder, flag
verification and infrastructure-failure guard in `tests/test_swarm.py` are the
parts that changed most often and previously had no coverage. Each file also
passes when run on its own, so there are no hidden import-order dependencies.

CI runs all of the above on every push (`.github/workflows/ci.yml`), including a
`uv lock --check` gate so `pyproject.toml` and the committed lockfile cannot
drift apart.

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

See `.env.example` for all settings (every knob is documented there). Key ones:

| Var | Default | Meaning |
|---|---|---|
| `STARTUP_HEALTH_CHECK` | `true` | Probe both gateway endpoints before launching; fail in seconds instead of grinding |
| `SOLVER_TIME_BUDGET_S` | `1500` | Per-attempt wall clock; on expiry the turn is interrupted and phase 2 harvests it. 0 disables |
| `CONCLUDE_TIMEOUT_S` | `90` | Independent budget for the phase-2 conclude harvest |
| `MAX_ATTEMPTS_PER_CHALLENGE` | `12` | Hard cap on retry rounds per solver. 0 = unlimited. Must be ≥ `2×FRESH_RESTART_ROUNDS+1` for two merge cycles |
| `MAX_INFRA_FAILURES` | `3` | Consecutive infra failures before a swarm retires itself |
| `SKILL_AFTER_BUMPS` | `1` | Inject skill hints after N failed attempts |
| `ESCALATE_AFTER_BUMPS` | `8` | Add a stronger solver after N total failed bumps. **0 disables** |
| `FRESH_RESTART_ROUNDS` | `5` | Meta-summarize the journal every N rounds so carried state stays bounded |
| `SIBLING_DIGEST_MIN_REPORTS` | `2` | Distinct reports needed before spending a model call on the team digest |
| `SIBLING_DIGEST_REFRESH_BUMPS` | `1` | Recompute the team digest at most once every N bumps (shared by all solvers) |
| `SUMMARIZER_MODEL` | `deepseek-v4-flash` | Model for journal distillation, merges and the team digest — all cost-tracked |
| `SANDBOX_COMMAND_TIMEOUT_S` | `900` | Per-command `timeout -k 5` inside the sandbox — no orphaned processes |
| `COORDINATOR_TURN_TIMEOUT_S` | `300` | Cap on one coordinator turn; a stalling gateway would otherwise freeze the whole event loop |
| `CONTAINER_MEMORY_LIMIT` | `16g` | Per-sandbox memory |
| `CONTAINER_CPU_LIMIT` | `2.0` | Per-sandbox CPU quota |
| `MAX_CONCURRENT_CHALLENGES` | `10` | Parallel swarms |
| `DASHBOARD_HOST` / `DASHBOARD_TOKEN` | `127.0.0.1` / auto | Loopback + bearer token by default (see below) |
| `MSG_PORT` | `9400` | Must match what `ctf-msg` dials |

### Operator dashboard

The dashboard can inject instructions into solvers (which execute arbitrary code
inside privileged sandboxes), spawn containers, and export confirmed flags. It is
therefore bound to `127.0.0.1` and requires a bearer token on every route. With
`DASHBOARD_TOKEN` unset, a random token is generated at startup and logged once
together with the ready-to-open URL:

```
Dashboard access token (no DASHBOARD_TOKEN was set): <token>
  → open http://127.0.0.1:8080/?token=<token>
```

Under Docker the listeners bind `0.0.0.0` *inside* the container (otherwise the
port cannot be forwarded) and compose publishes them as `127.0.0.1:8080:8080`, so
neither the LAN nor a sandbox container can reach them via
`host.docker.internal`. Bind to `0.0.0.0` on the host only behind a reverse proxy
that adds its own auth.

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
