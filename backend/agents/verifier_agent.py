"""VerificationRunner — drives the Verifier role on a shared sandbox.

One runner per swarm. It owns a single solver instance created with the verifier's
`role_prompt`, reuses that sandbox across every claim, and runs concurrently with
the solvers so verification costs no wall-clock.

The sandbox is created LAZILY, on the first claim that survives triage. A challenge
solved in 40 seconds never pays for a verifier container, and a challenge with
nothing testable on the board never spins one up either.
"""

from __future__ import annotations

import asyncio
import logging

from backend.config import bool_setting, int_setting
from backend.verifier import (
    INCONCLUSIVE,
    ROLE_PROMPT,
    ClaimQueue,
    PendingClaim,
    Triage,
    Verdict,
    build_claim_task,
    parse_verdict,
)

logger = logging.getLogger(__name__)


def _joined(*texts: str) -> str:
    """Concatenate non-empty text sources for verdict scanning.

    Never `a or b`: the accumulated transcript and the round summary carry
    different slices of what the model said, and a verdict line can land in either.
    """
    return "\n".join(t for t in texts if t and t.strip())


def _base_model(spec: str) -> str:
    """Strip the provider and any effort suffix, so the same weights reached by two
    routes are recognised as one adjudicator.

    This matters because the quota fallback rebinds a solver mid-run
    (`claude-sdk/kimi-k3` -> `gateway/kimi-k3`); comparing full specs would treat
    those as two independent models when they are the same one.
    """
    parts = (spec or "").split("/")
    return parts[1] if len(parts) >= 2 else (spec or "")


class VerificationRunner:
    """Drains a ClaimQueue by running real probes in a sandbox.

    `on_verdict` is awaited with every non-trivial outcome; the swarm owns what
    happens to the board. Keeping that as a callback rather than reaching into the
    swarm is what makes this testable without Docker.
    """

    def __init__(self, swarm, queue: ClaimQueue, on_verdict) -> None:
        self.swarm = swarm
        self.queue = queue
        self.on_verdict = on_verdict
        self._solver = None
        self._spec = ""
        self._task: asyncio.Task | None = None
        self._lock = asyncio.Lock()
        self.probes_run = 0
        self.verified = 0
        self.refuted = 0
        self.inconclusive = 0
        # Claims triaged out before any sandbox time was spent on them.
        self.filtered = 0
        # Triage calls that produced no usable answer. Distinct from `filtered`:
        # these claims WERE verified, and a nonzero count means the pre-filter is
        # not actually filtering. Neither counter was exported anywhere, which is
        # how a fully dead verifier came to look like an idle healthy one.
        self.triage_failures = 0
        # Probes that ended a turn without a verdict and needed the forced-commit
        # round. A high count means the role prompt is not landing, which is worth
        # seeing rather than hiding behind INCONCLUSIVE.
        self.forced_commits = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def ensure_started(self) -> None:
        """Start the drain loop if it is not already running. Idempotent."""
        if self.running:
            return
        self._task = asyncio.create_task(self._loop(), name=f"verifier-{self.swarm.meta.name}")

    async def stop(self) -> None:
        """Cancel the loop and tear the sandbox down.

        Teardown is awaited rather than cancelled through: an interrupted
        `solver.stop()` is how this codebase historically leaked a container and a
        workspace directory per sibling.
        """
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 — shutting down
                pass
        self.queue.clear()
        solver, self._solver = self._solver, None
        if solver is None:
            return
        try:
            await solver.stop()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — teardown must not mask the real error
            logger.warning(f"[{self.swarm.meta.name}] verifier teardown failed: {e}")

    async def _loop(self) -> None:
        """Drain claims until the swarm is done."""
        swarm = self.swarm
        while not swarm.cancel_event.is_set():
            try:
                claim = await self.queue.pop(timeout=2.0)
            except asyncio.CancelledError:
                raise
            if claim is None:
                continue
            try:
                await self._verify_one(claim)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — one bad probe must not end verification
                logger.warning(
                    f"[{swarm.meta.name}] verification of {claim.key[:40]!r} crashed: {e}"
                )
                self.queue.release(claim.key)
                await self._safe_report(Verdict(
                    INCONCLUSIVE, f"verifier crashed: {type(e).__name__}: {e}"[:300],
                    claim.content, claim.source,
                ))

    # ------------------------------------------------------------------
    # One probe
    # ------------------------------------------------------------------

    def _resolve_spec(self, prefer_not: str = "") -> str:
        """Which model verifies. Explicit knob wins, else the strongest solver in
        the lineup — verification is judgement-heavy and a weak model returns
        INCONCLUSIVE for everything, which is safe but useless.

        `prefer_not` is the spec that asserted the claim currently being probed. A
        probe is empirical, so the same model running a real command is far better
        than the same model merely agreeing with itself — but it still shares the
        failure mode that matters here: designing a non-discriminating test and then
        misreading its output in the same way twice. With the default
        `VERIFIER_MODEL` unset and `specs[0]` as the fallback, the most prolific
        claimant was also the adjudicator of its own claims, and a ✅ is the tier
        every other agent is told to build on freely. Picking a different model when
        one is available costs nothing.
        """
        configured = str(getattr(self.swarm.settings, "verifier_model", "") or "").strip()
        if configured:
            return configured
        specs = list(getattr(self.swarm, "_all_specs", None) or self.swarm.model_specs)
        if not specs:
            return ""
        avoid = _base_model(prefer_not)
        if avoid:
            for spec in specs:
                if _base_model(spec) != avoid:
                    return spec
        return specs[0]

    async def _ensure_solver(self, prefer_not: str = ""):
        """Create the verifier solver + sandbox on first use.

        `prefer_not` only influences the FIRST claim, since one sandbox is then
        reused for the whole swarm by design. That is a deliberate trade: spinning up
        a container per claim to guarantee adjudicator independence would cost more
        wall-clock than the residual risk it removes, and a probe is empirical
        evidence rather than a second opinion.

        `self._solver` is assigned BEFORE `start()` and torn down on any failure.
        Both backends bring the container up first and then await more work
        (`sandbox.exec("uname -m")`, `resolve_model`, the SDK client handshake), so
        a raise or a cancel in that window leaves a live container that nothing can
        reach: `stop()` bails out on `self._solver is None`, and because `_loop`
        catches the error and moves to the next claim, a persistent cause — a typo'd
        VERIFIER_MODEL, a spec that will not resolve — leaked one container and one
        workspace per claim for the life of the swarm.
        """
        async with self._lock:
            if self._solver is not None:
                return self._solver
            spec = self._resolve_spec(prefer_not)
            if not spec:
                return None
            self._spec = spec
            solver = self.swarm.create_verifier(spec, ROLE_PROMPT)
            # Publish first so a concurrent stop() can still find and tear it down.
            self._solver = solver
            try:
                await solver.start()
            except BaseException:
                self._solver = None
                try:
                    await solver.stop()
                except Exception as e:  # noqa: BLE001 — already failing; do not mask
                    logger.warning(
                        f"[{self.swarm.meta.name}] verifier start failed AND its "
                        f"sandbox could not be torn down (possible leak): {e}"
                    )
                raise
            logger.info(f"[{self.swarm.meta.name}] Verifier online ({spec})")
            return solver

    async def _triage(self, claim: PendingClaim) -> Triage:
        """Cheap pre-filter: is this claim testable AND load-bearing?

        Both a raise and a None return resolve to "still verify". That is the
        opposite of the usual fail-safe direction here, and deliberate: triage is an
        optimisation, and one that silently disabled verification whenever the
        gateway hiccuped would remove the only mechanism that can catch a false
        premise. Skipping a probe costs one sandbox round; skipping verification
        costs the run.

        The None case is the one that matters in production. `_call_gateway`
        swallows every transport failure and returns "", which `parse_triage` reads
        as "not testable" — so a dead gateway used to look exactly like a triage
        model that had examined every claim and declined all of them, with the only
        counter (`filtered`) exported nowhere.
        """
        from backend.verifier import triage_claim

        meta = self.swarm.meta
        name = getattr(meta, "name", "") or "chal"
        try:
            triage = await triage_claim(
                claim=claim.content,
                name=name,
                category=getattr(meta, "category", ""),
                description=getattr(meta, "description", ""),
                settings=self.swarm.settings,
                cost_tracker=self.swarm.cost_tracker,
                agent_name=f"{name}/triage",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — triage is an optimisation
            logger.warning(
                f"[{name}] triage raised, verifying anyway: {type(e).__name__}: {e}"
            )
            self.triage_failures += 1
            return Triage(testable=True, load_bearing=True, probe_hint="")
        if triage is None:
            self.triage_failures += 1
            logger.warning(
                f"[{name}] triage returned NO answer (gateway unreachable, timeout or "
                f"truncated) — verifying the claim anyway: {claim.content[:70]!r}. "
                "If this repeats, verification is running unfiltered; check the gateway."
            )
            return Triage(testable=True, load_bearing=True, probe_hint="")
        return triage

    async def _verify_one(self, claim: PendingClaim) -> None:
        swarm = self.swarm
        if swarm.cancel_event.is_set():
            return

        # Triage runs BEFORE the sandbox is touched. Money is not the constraint —
        # wall-clock and sandbox contention are, and a probe that cannot falsify
        # anything still occupies the verifier for a whole round while solvers wait
        # on the answer. Filtering untestable and non-load-bearing claims here is
        # what keeps the verifier pointed at premises whose falsity would actually
        # invalidate other agents' work.
        if bool_setting(swarm.settings, "verifier_triage"):
            triage = await self._triage(claim)
            if not triage.should_verify:
                self.filtered += 1
                logger.debug(
                    f"[{swarm.meta.name}] claim skipped by triage "
                    f"(testable={triage.testable}, load_bearing={triage.load_bearing}): "
                    f"{claim.content[:60]!r}"
                )
                # Release the key. `pop()` deliberately leaves it reserved so a claim
                # cannot be re-queued while pending, but this one is no longer
                # pending — and without the release a later CONTRADICTION on the same
                # statement could never be probed: push() returns False on the
                # reserved key, bump_priority() then finds nothing in _items, and the
                # strongest signal in the whole design is dropped without a log line.
                self.queue.release(claim.key)
                return
            claim.probe_hint = triage.probe_hint or claim.probe_hint
            # Re-check after triage: that call can take 25s, and starting a sandbox
            # for a swarm that was killed while we waited leaves teardown dependent on
            # whoever eventually awaits run()'s finally.
            if swarm.cancel_event.is_set():
                self.queue.release(claim.key)
                return

        solver = await self._ensure_solver(prefer_not=claim.source)
        if solver is None:
            self.queue.release(claim.key)
            return

        # Fresh context per claim. Without the reset the verifier accumulates every
        # earlier probe's transcript and starts answering from memory of the last
        # claim instead of testing this one — which looks like verification and is
        # not. Workspace files persist, so earlier probes are still reusable.
        for name in ("forget_session", "reset_history", "reset_session"):
            fn = getattr(solver, name, None)
            if fn is None:
                continue
            try:
                out = fn()
                if asyncio.iscoroutine(out):
                    await out
                break
            except Exception as e:  # noqa: BLE001 — a failed reset degrades, not dies
                logger.debug(f"[{swarm.meta.name}] verifier reset via {name} failed: {e}")

        solver.bump(build_claim_task(claim.content, claim.source, claim.probe_hint))

        budget = int_setting(swarm.settings, "verifier_budget_s")
        try:
            result = await solver.run_until_done_or_gave_up(budget)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — report and move to the next claim
            logger.warning(f"[{swarm.meta.name}] verifier round failed: {e}")
            self.queue.release(claim.key)
            await self._safe_report(Verdict(
                INCONCLUSIVE, f"verifier round failed: {type(e).__name__}"[:300],
                claim.content, claim.source,
            ))
            return

        self.probes_run += 1
        # Scan BOTH sources, never `a or b`. On the real backends `findings` is the
        # accumulated assistant text (every block the model emitted, and where
        # conclude() appends its answer) while `findings_summary` is only this
        # round's summary — so a verdict line can exist in one and not the other.
        # Short-circuiting on the first non-empty string silently discarded verdicts
        # that had actually been produced.
        text = _joined(result.findings_summary, getattr(solver, "findings", ""))
        verdict = parse_verdict(text, claim.content, claim.source)

        # Observed live: the verifier ended its turn mid-exploration ("Let me look
        # at main's disassembly") after four tool calls, never emitting a verdict.
        # Narrating an intention and then stopping is normal model behaviour, and
        # unlike a solver the verifier had no second phase to force it to commit —
        # so the probe work was done and the answer thrown away.
        #
        # The retry goes through conclude(), NOT another open round. conclude()
        # denies every tool by construction, so the model physically cannot start
        # new work and must answer from what it already observed. A second
        # tools-enabled round was tried first and simply kept exploring until its
        # own budget expired — recovering nothing.
        #
        # Deliberately skipped when the model explicitly said INCONCLUSIVE:
        # re-asking a model that already decided it could not tell just pressures it
        # into guessing, which is the one outcome worse than no verdict.
        if not verdict.explicit and not swarm.cancel_event.is_set():
            self.forced_commits += 1
            logger.info(
                f"[{swarm.meta.name}] verifier produced no verdict — forcing a "
                "tools-denied commit from what it already observed"
            )
            try:
                commit_text = await solver.conclude(max(30, budget // 3))
                # Parse the commit answer ALONE first. parse_verdict only scans the
                # last 15 lines, so concatenating commit_text ahead of a full
                # transcript pushes it out of the window — measured: a correct
                # "VERDICT: REFUTED | 422 blocked" followed by 30 lines of narration
                # parsed as INCONCLUSIVE, so the recovery billed a strong-model
                # conclude call and then threw the answer away. That is worse than
                # not retrying, because `forced_commits` rose while nothing was
                # recovered and the premise stayed ⚠ on the board.
                retry = parse_verdict(commit_text or "", claim.content, claim.source)
                if not retry.explicit:
                    # Fall back to everything: conclude() also appends its answer to
                    # the solver's accumulated findings, and a verdict may sit in the
                    # first round's output rather than the second's.
                    retry = parse_verdict(
                        _joined(getattr(solver, "findings", ""), text, commit_text or ""),
                        claim.content, claim.source)
                if retry.explicit:
                    verdict = retry
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — the first pass already ran
                logger.debug(f"[{swarm.meta.name}] forced commit failed: {e}")

        # A probe that never ran a single command cannot have observed anything,
        # whatever the model concluded. This is the same guard the swarm applies to
        # solvers, and it matters more here: an unverifiable VERIFIED is worse than
        # no verification at all.
        if verdict.verdict != INCONCLUSIVE and not getattr(result, "did_work", False):
            logger.info(
                f"[{swarm.meta.name}] verifier returned {verdict.verdict} after 0 tool "
                "calls — downgrading to INCONCLUSIVE"
            )
            verdict = Verdict(INCONCLUSIVE, "verdict returned without running any command",
                              claim.content, claim.source)

        if verdict.verdict == "VERIFIED":
            self.verified += 1
        elif verdict.verdict == INCONCLUSIVE:
            self.inconclusive += 1
            # Retryable: a transient failure should not permanently mark a claim
            # untested. VERIFIED/REFUTED are terminal, so their keys stay reserved.
            self.queue.release(claim.key)
        else:
            self.refuted += 1
        logger.info(
            f"[{swarm.meta.name}] VERIFY {verdict.verdict} ({self._spec}): "
            f"{claim.content[:70]!r} → {verdict.evidence[:120]}"
        )
        await self._safe_report(verdict)

    async def _safe_report(self, verdict: Verdict) -> None:
        """A board-write failure must not kill the drain loop."""
        try:
            await self.on_verdict(verdict)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[{self.swarm.meta.name}] could not publish verdict: {e}")
