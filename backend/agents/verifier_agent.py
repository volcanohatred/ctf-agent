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

from backend.verifier import (
    INCONCLUSIVE,
    ROLE_PROMPT,
    ClaimQueue,
    PendingClaim,
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

    def _resolve_spec(self) -> str:
        """Which model verifies. Explicit knob wins, else the strongest solver in
        the lineup — verification is judgement-heavy and a weak model returns
        INCONCLUSIVE for everything, which is safe but useless."""
        configured = str(getattr(self.swarm.settings, "verifier_model", "") or "").strip()
        if configured:
            return configured
        specs = list(getattr(self.swarm, "_all_specs", None) or self.swarm.model_specs)
        return specs[0] if specs else ""

    async def _ensure_solver(self):
        """Create the verifier solver + sandbox on first use."""
        async with self._lock:
            if self._solver is not None:
                return self._solver
            spec = self._resolve_spec()
            if not spec:
                return None
            self._spec = spec
            solver = self.swarm.create_verifier(spec, ROLE_PROMPT)
            await solver.start()
            self._solver = solver
            logger.info(f"[{self.swarm.meta.name}] Verifier online ({spec})")
            return solver

    async def _triage(self, claim: PendingClaim):
        """Cheap pre-filter: is this claim testable AND load-bearing?

        Any failure resolves to "still verify". That is the opposite of the usual
        fail-safe direction here, and deliberate: triage is an optimisation, and an
        optimisation that silently disables verification when the gateway hiccups
        would remove the one mechanism that can catch a false premise. Skipping a
        probe costs one sandbox round; skipping verification costs the run.
        """
        from backend.verifier import Triage, triage_claim

        meta = self.swarm.meta
        try:
            return await triage_claim(
                claim=claim.content,
                name=getattr(meta, "name", ""),
                category=getattr(meta, "category", ""),
                description=getattr(meta, "description", ""),
                settings=self.swarm.settings,
                cost_tracker=self.swarm.cost_tracker,
                agent_name=f"{getattr(meta, 'name', 'chal')}/triage",
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — triage is an optimisation
            logger.debug(f"[{getattr(meta, 'name', '')}] triage failed, verifying anyway: {e}")
            return Triage(testable=True, load_bearing=True, probe_hint="")

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
        if bool(getattr(swarm.settings, "verifier_triage", True)):
            triage = await self._triage(claim)
            if not triage.should_verify:
                self.filtered += 1
                logger.debug(
                    f"[{swarm.meta.name}] claim skipped by triage "
                    f"(testable={triage.testable}, load_bearing={triage.load_bearing}): "
                    f"{claim.content[:60]!r}"
                )
                return
            claim.probe_hint = triage.probe_hint or claim.probe_hint

        solver = await self._ensure_solver()
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

        budget = int(getattr(swarm.settings, "verifier_budget_s", 300) or 300)
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
                # conclude() appends its answer to the solver's accumulated
                # findings, so scan that too — it is the fullest text available.
                retry = parse_verdict(
                    _joined(commit_text, getattr(solver, "findings", ""), text),
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
