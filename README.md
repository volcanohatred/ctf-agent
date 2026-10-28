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
| Trust | every report is a fact | **Four tiers** — ✅ probe-verified > ✓ corroborated by 2+ solvers > ⚠ single-source claim > ✗ excluded/refuted |
| False premises | undetectable | **Verifier agent** runs a real probe per claim and can REFUTE it, pruning every dependent line of attack |
| Internal inconsistency | invisible | **Contradiction audit** — mutually exclusive board entries are surfaced and both sides sent to a probe |
| Strategy | solvers rotate tactics only | **Strategist** — periodic step-back that reads the题面 + dead-ends and can force a PIVOT |
| Stagnation | per-command loop detector | **+ stale-round circuit breaker** — catches repeated *outcomes*, not repeated commands |
| WEB measurements | re-derived per round, then lost | **`ctf_probe`** telemetry log in the workspace + `ctf-probe` query CLI |
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

## Not believing yourselves: trust, strategy, stagnation

Three mechanisms were added after a long WEB/Java run exposed a failure mode the
rest of the architecture could not see. Every solver ground away tactically for
hours — fuzzing payload variants, enumerating class names, collecting hundreds of
422 responses — while one model's unverified claim ("the uploaded JAR does get
loaded") sat on the blackboard indistinguishable from a fact three models had
established, and every restart inherited it as ground truth. Nothing in the
system ever re-read the challenge description.

**Corroboration gate** (`CORROBORATE_FACTS`, default on). A finding one solver
asserts is published as an **unverified claim** (⚠) and is promoted to a
**corroborated fact** (✓) only when a *different* solver independently reports the
same thing. Two details make it hold:

- a solver re-asserting its own line after a session wipe does **not** count —
  that is exactly how a hallucination survives a restart, and counting it would
  let a model promote its own invention by repeating it;
- matching is structural (`backend/dedup.py`), so an ASLR slide or a fresh offset
  does not defeat it. Two models observing the same thing report different
  addresses; requiring literal equality would mean corroboration only ever fired
  on copy-paste, which is the one case that proves nothing.

It costs no LLM calls. Claims and facts are separate sections in the sibling
prompt, and the claim section names the failure mode rather than just labelling
it — a flat "unverified" reads as noise and gets ignored. The markers survive
into the team digest (`_SIBLING_PROMPT` is explicitly forbidden from upgrading
them), because a gate defeated one step downstream is not a gate. Dead ends need
no corroboration: they are negative knowledge, and holding one back until a
second model wastes a round rediscovering it would defeat the ledger.

Corroboration has two holes, and both cost solve rate, which is why it is the
floor and not the ceiling:

- **It can never refute anything.** A false premise is not labelled trusted, but
  it stays on the board, still readable, still tempting.
- **It can never confirm anything either.** A true finding that is hard to
  rediscover stays ⚠ forever and gets under-weighted — a false negative on the
  one thing the swarm most needs.

**The Verifier** (`VERIFY_CLAIMS`, default on) closes both by doing what
agreement cannot: running a test. It is a real agent with a real sandbox — one
per swarm, reused across claims, created lazily on the first testable claim, and
running concurrently with the solvers so it costs no wall-clock. It is built on
the existing solver backends through a `role_prompt` override, so it inherits
sandbox redirection, loop detection, tracing, the wall-clock budget and shielded
teardown rather than being a second implementation.

Its verdict has three values and the asymmetry is the design:

| Verdict | Effect on the board |
|---|---|
| `VERIFIED` | claim → ✅ probe-verified, the strongest tier, evidence appended in place |
| `REFUTED` | claim → ✗ excluded, added to the dead-end ledger, **prune notice broadcast**, team digest invalidated, coordinator told |
| `INCONCLUSIVE` | nothing changes |

Anything ambiguous resolves to INCONCLUSIVE: an unparsable reply, a crashed probe,
a timeout, a verdict with no observable evidence, or a verdict returned after
zero tool calls. INCONCLUSIVE is free. A wrong VERIFIED makes four agents build
on sand; a wrong REFUTED kills the one line of attack that might have worked.

The refutation path is the one that saves hours, and it does three things because
all three are needed: moving the claim out of circulation (a log line alone leaves
it readable), recording the evidence (so nobody re-tests it), and broadcasting a
prune notice — which running solvers pick up within ~5 tool calls via the existing
mid-turn hook, instead of at their next restart up to a whole time budget later.
The team digest is also invalidated, since it was computed while the premise was
still believed and would otherwise re-seed every restarting solver with it.

The role prompt is written against the specific ways verification fails: it
demands a **discriminator** (two cases that must differ if the claim is true) over
a direct test, forbids reporting anything but observed output, and states plainly
that **absence of an error is not confirmation** — a missing `ClassNotFoundException`
may mean the class loaded, or that the path never ran, or that the exception was
swallowed. That last rule is the FatJar failure written down as an instruction.

A cheap triage call filters claims before any sandbox time is spent: is it
testable by a probe at all, and would its falsity invalidate other agents' work?
This is a wall-clock control, not a cost control. If the triage call itself fails
the claim is verified anyway — an optimisation must not be able to silently
disable the only mechanism that can catch a false premise.

**Contradiction audit** (`AUDIT_CONTRADICTIONS`, default on). The Verifier tests a
claim against the *target*; this tests claims against *each other*. Each round's
new statements are compared against the whole board, and a pair that cannot both
be true is broadcast plus pushed to the front of the verification queue — both
sides, which is the A/B confrontation, settled by a probe rather than by another
opinion. The broadcast deliberately does **not** say which side is wrong: the
audit has no evidence for that, and picking one would be a new unverified claim
wearing an authoritative label.

This catches what the board itself hides, because two mutually exclusive
statements each look fine in isolation. It is also the cheapest of the mechanisms:
one summarizer-class call per published round, no sandbox.

**Strategist** (`STRATEGIST_EVERY_BUMPS`, default 4). One model call, not a fifth
swarm member — it owns no sandbox, runs no tools and cannot submit a flag, so it
cannot contribute a dead end of its own. It is shown the challenge metadata and
the dead-end ledger and **deliberately nothing else**: the pile of tactical
findings is precisely what drowned the宏观 clues. It must answer `CONTINUE` or
`PIVOT: <new attack surface>`; the verdict is broadcast as a 📢 strategist
message, kept visually distinct from a ✓ fact, and a pivot is worded as defeasible
— solvers are told to check it against their own confirmed evidence rather than
obey it. It runs on a strong model (`STRATEGIST_MODEL` → `COORDINATOR_MODEL` →
built-in default), never `SUMMARIZER_MODEL`: a flash model answers CONTINUE to
everything, which makes the whole mechanism decorative. Four solvers bumping at
once cost one call.

**Stale-round circuit breaker** (`STALE_ROUND_LIMIT`, default 3). After N
consecutive rounds in which a solver added *nothing new* to the team — every dead
end it reported was already recorded, every fact already asserted by someone —
it forces a Strategist consult and hands that solver a hard redirect forbidding
further micro-tweaks of the same payload. This catches what `loop_detect.py`
structurally cannot: each payload tweak is a genuinely different command with a
different signature, so no per-command detector ever fires. What repeats is the
*outcome*. Infrastructure failures, zero-tool-call rounds and context overflows
are excluded — those belong to `MAX_INFRA_FAILURES`, and tripping the breaker on
a flaky gateway would redirect a solver that never got to try anything. The
counter resets after firing, so a stuck solver gets a full N rounds of fresh
stagnation before being interrupted again rather than spending its budget reading
redirects.

Every mechanism above is auditable. `check_swarm_status`, the operator dashboard,
the Markdown/JSON export and the benchmark report all carry `probe_verified`,
`corroborated_facts`, `unverified_claims`, `probe_refuted`, `contradictions`,
`stale_breaks` and `strategist_pivots`, so a run that ground for hours is
distinguishable in the report from one that was making progress the whole time.
`ctf-bench` prints a **Trust audit** block and warns on the two shapes that look
like a model-capability problem and are not: far more claims than confirmations
(solvers exploring disjoint surfaces, never cross-checking), and contradictions
detected but never settled by a probe (both statements still on the board).

The export renders ✅ / ✓ / ⚠ / ✗ / 📢 with the same markers the solvers see, and
flags a run where most of what the team "knew" was never tested. A writeup built
from an export that rendered a ⚠ claim like a ✓ fact would re-launder the
hallucination into the permanent record.

**The coordinator is constrained to match.** It may not launder a claim into a
fact, and it steers by direction rather than by payload — it cannot test an
exploit string, and a wrong one dictated with authority is adopted verbatim by
every solver that reads the board. A `STRATEGIST PIVOT` outranks its own read of
the tactics. A `VERIFIER REFUTED` notice is final: the probe ran against the real
target, so a solver's insistence is not counter-evidence. And on a contradiction
notice it is forbidden from picking a side — nobody knows yet which is wrong, and
guessing is how a coordinator launders its own hunch into a fact.

**Sandbox telemetry** (`sandbox/ctf_probe.py`, preinstalled in the image). WEB
solvers used to re-derive the same timing code every round and then lose the
measurement, because the next bash call is a fresh process. `ctf_probe` records
every request's status/elapsed/body to a JSONL log inside the workspace, so a
later command can *query* the sweep instead of re-running it
(`ctf-probe --min-ms 1000`). It reports the **median** of a repeat set and records
the spread, because bridge-network jitter is routinely larger than the signal
being measured and a solver that cannot see the spread cannot tell an oracle from
noise. It is generic and optional — it knows nothing about any challenge's
endpoints, and plain `curl` is still the right tool for a one-off request.
`sandbox/AGENTS.md` also tells solvers to `ls /challenge/workspace/` and extend an
existing script rather than adding `r7.py` beside it, and the Verifier writes its
probes to `/challenge/workspace/verify/` so solvers reuse them instead of
re-deriving the same measurement.

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

# Ask the command centre what is actually going on with a challenge that has been
# grinding for hours (sends its whole live state to a reasoning model)
docker compose exec agent uv run --no-sync ctf-ask "FatJar Relic" "这题现在什么进展？还该继续吗？"
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

The bundled suite has **22 keyed challenges**: `example-static` (1) +
`regression` (10) + `hard` (4) + `hard2` (4) + `hard3` (3), covering glibc-2.39
safe-linking tcache poison, tcache double-free with key-clear bypass, DSA
nonce-MSB hidden-number-problem and an MBA-obfuscated reverse checker.

> An earlier revision of this file claimed "19/20" while its own table summed to
> 21/21 and its prose described 21 challenges, and cited an "intentionally
> unsolvable knapsack control case" that is not present anywhere in `challenges/`.
> The numbers below are what was actually measured, with the distinction between
> re-verified and inherited made explicit.

## Verification results

**Re-measured on this branch** (`feature/coach-mode`, gateway models, live local
CTFd, `MAX_CONCURRENT_CHALLENGES=2`):

| Run | Result | Cost | Notes |
|---|---|---|---|
| Full live CTFd run — `regression` (10) + `hard2` (4) | **14/14** | $8.72 | Coordinator mode end to end: auto-spawn, solve, submit, auto-kill, self-healing re-spawn. 0 leaked containers, 0 leaked workspaces |
| `regression/misc-encoding` via `ctf-solve` | ✅ MATCH | $0.25–0.30 | 37s and 146s on separate runs |
| `regression/forensics-magic` via `ctf-bench` | ✅ MATCH | $1.82 | 205s |
| `regression/reverse-xor-binary` | ✅ MATCH | $0.40 | 30s |
| `hard2/reverse-vm` | ✅ MATCH | $0.15 | 14s |
| `hard3/reverse-mba` | ✅ MATCH | $0.22 | 55s |
| `example-static` (smoke) | ✅ MATCH | $0.06–0.07 | 4/4 sandboxes torn down cleanly |
| Two-phase harvest under a live LLM | ✅ | $1.54 | Forced with a 12s round budget on an unsolvable synthetic challenge: 8 `budget_interrupt`, **6/6 `conclude_ok`** (0 failures), 6 `session_forgotten`, 3 full restart cycles |

That covers 16 of the 22 challenges by direct observation.

**Inherited from earlier revisions, not re-measured here**: `hard` (4 —
coppersmith / anti-debug / flask-cookie / heap-tcache) and `hard3`'s crypto-hnp
and web-pickle-restricted (2). Treat those rows as unverified until re-run.

| Suite | Status |
|---|---|
| `challenges/example-static` (1) | re-verified ✅ |
| `challenges/regression` (10) | re-verified 10/10 ✅ |
| `challenges/hard2` (4) | re-verified 4/4 ✅ |
| `challenges/hard3` (3) | 1 of 3 re-verified; 2 inherited |
| `challenges/hard` (4) | inherited, not re-verified |

**Skill engagement**: bump-triggered hints escalate L1→L2→L3 with real file
paths, and the approach queue rotates through triaged handbook files — observed
live (`Skill triage → ['ctf-misc']`, `Approach queue: 12 techniques`, and
`['ctf-forensics', 'ctf-reverse']` → 24 techniques). The previously documented
"knapsack → crypto lattice doc" example cannot be reproduced: that challenge is
not in the tree.

Rebuild/regenerate: `python3 scripts/gen_regression.py` (inside the sandbox),
`gen_hard.py`, `gen_hard2.py`, `gen_hard3.py`; seed a local CTFd with
`scripts/seed_ctfd.py`.

> **Scoring caveat**: 21 keyless flat duplicates of the canonical challenges also
> sit directly under `challenges/` (e.g. `challenges/reg-misc-encoding/` beside
> the keyed `challenges/regression/misc-encoding/`). A challenge with no `flag:`
> cannot be scored offline at all — every submission answers DRY RUN, so
> FLAG_FOUND is unreachable and the run always reports "No flag found" no matter
> how well the models did. Both `ctf-bench` and `ctf-solve` now warn about this
> instead of failing silently. Their `distfiles/` are byte-identical to the
> canonical copies, so they are redundant rather than additional coverage.

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
uv run pytest -q             # 587 tests, no Docker / no LLM / no network
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
| **Web** | curl, nmap, requests, flask, **`ctf_probe`** (request/timing telemetry log + `ctf-probe` query CLI) |
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
| `CORROBORATE_FACTS` | `true` | One solver's assertion is an unverified claim until a second solver independently reports it |
| `VERIFY_CLAIMS` | `true` | Verifier agent runs a real probe per testable claim; can REFUTE it and broadcast a prune notice |
| `VERIFIER_MODEL` | `` | Model spec for the Verifier. Empty = first solver in the lineup |
| `VERIFIER_BUDGET_S` | `300` | Wall clock for ONE probe — a "minimal test" needing 25 min is not minimal |
| `VERIFIER_TRIAGE` | `true` | Cheap pre-filter: testable AND load-bearing, else skip. Wall-clock control, not cost control |
| `VERIFIER_MAX_PROBES` | `0` | Hard cap on decisive probes per challenge. 0 = unlimited |
| `AUDIT_CONTRADICTIONS` | `true` | Compare each round's statements against the board; conflicting pairs go to a probe |
| `STRATEGIST_EVERY_BUMPS` | `4` | Consult the Strategist (题面 + dead-ends only) every N swarm-wide bumps. 0 disables |
| `STRATEGIST_MODEL` | `` | Strong model for the Strategist; falls back to `COORDINATOR_MODEL`, never to the summarizer |
| `STRATEGIST_TIMEOUT_S` | `300` | Measured 100–150s per consult. **Do not lower without measuring** — at 60s every call timed out and the mechanism was silently dead |
| `STRATEGIST_MAX_TOKENS` | `8000` | Reasoning models burn most of it thinking; at 1200 the verdict line never appeared and the tail read as CONTINUE |
| `REASONING_MODEL` | `` | Model for judgement calls (Strategist + audit). Never `SUMMARIZER_MODEL` — measured: flash answered "no contradiction" to an unambiguous conflict |
| `AUDIT_MODEL` / `AUDIT_TIMEOUT_S` / `AUDIT_MAX_TOKENS` | `` / `120` / `4000` | Contradiction-audit overrides; empty = `REASONING_MODEL` |
| `BRIEFING_MODEL` / `BRIEFING_TIMEOUT_S` / `BRIEFING_MAX_TOKENS` | `` / `300` / `8000` | Command-centre chat; ~70s measured on a full dossier |
| `STALE_ROUND_LIMIT` | `3` | Rounds adding nothing new before the circuit breaker forces a pivot + hard redirect. 0 disables |
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

### Command-centre chat (`/api/ask`, `ctf-ask`)

Four hours into a hard challenge there is more state than a human can read: four
solvers, dozens of bumps, a blackboard of findings and dead ends, per-solver traces,
audit counters. Ask instead of reading — in the dashboard's 🎙 panel, or:

```bash
uv run ctf-ask "FatJar Relic" "这题现在什么进展？我们还该继续吗？"
uv run ctf-ask "FatJar Relic" "有哪些说法还没被验证？" --token <dashboard-token>
```

Every question assembles that challenge's **entire live state** and sends it to a
reasoning model: the blackboard *grouped by trust tier*, the dead-end ledger, the
failed handbooks, per-solver trace health (spinning / looping / last reasoning
text), and every audit counter. The dossier is rebuilt per request rather than
cached, because the swarm is live and an answer about "the current state" built on
a ten-minute-old snapshot is confidently wrong in exactly the way that gets an
operator to issue a stale instruction. It works for archived swarms too — "what
happened on the one we lost four hours on?" is asked most often after the fact.

The design constraint is the same one the rest of the trust system enforces, and it
is **more** important here, not less: the briefing must never upgrade a ⚠
single-source claim into a fact. An operator told "the team has established that the
JAR gets loaded" will type "go with the JAR-loading approach" into the advice box —
and operator advice outranks anything an agent says to itself. Laundering a
hallucination through a summary does not dilute it, it *authorises* it. So the
dossier arrives with its ✅/✓/⚠/✗/📢 markers intact, the system prompt forbids
collapsing them and explains why, and the answer must lead with what is DEAD before
what is live (the most expensive thing an operator can do is suggest something
already disproved), must say "档案中没有这项" rather than speculate, and must
distinguish an infrastructure failure from a strategy failure.

It ends with 1–3 paste-ready 【建议指令】; in the dashboard each one is a button that
fills the guidance mailbox. Over-budget dossiers trim the ⚠ claim pile first and
never the verified or refuted material, and say out loud what was cut — silently
dropping the refuted premises is the one truncation that would make a briefing
actively dangerous.

Runs on `BRIEFING_MODEL` → `REASONING_MODEL` → `COORDINATOR_MODEL` → built-in
default. Measured ~70s on a full dossier, hence the 300s default timeout.

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
