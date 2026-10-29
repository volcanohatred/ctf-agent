"""The fresh-restart retry loop — the original design, pinned down.

The intended mechanism:
  1. after a GAVE_UP, distil that round's attempts and findings into one entry
  2. restart clean, seeded with the distilled state + sibling summaries + the
     skill-library hint and the next untried technique
  3. every N rounds (default 5), call the model AGAIN to merge those entries into
     one condensed state, de-duplicating, and restart from that

These tests assert each step, plus the failure modes that silently defeated it:
a summarizer outage used to collapse the journal into one ever-growing string,
and the sibling section used to inject an English placeholder every round.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from test_swarm import DEFAULT_CAP, ScriptedSolver, _flag, _gave_up, _swarm

import backend.round_summary as rs
import backend.summarize as summarize_mod
import backend.verifier as verifier_mod
from backend.round_summary import SummaryResult, _call_gateway

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class RecordingSummarizer:
    """Stands in for the gateway-backed summarizers."""

    def __init__(self, meta_condensed: bool = True):
        self.round_calls: list[str] = []
        self.meta_calls: list[list[str]] = []
        self.meta_condensed = meta_condensed

    async def summarize_round(self, text, settings, cost_tracker=None, agent_name=""):
        self.round_calls.append(text)
        # Numbered so successive rounds produce DISTINCT entries. An earlier
        # version keyed the output only on len(text), which made every round
        # identical — the structural de-dup then (correctly) collapsed all five
        # into one and the merge test measured nothing.
        return f"<distilled#{len(self.round_calls)}:{len(text)}chars>"

    async def meta_summarize(self, journal, settings, cost_tracker=None, agent_name=""):
        self.meta_calls.append(list(journal))
        if self.meta_condensed:
            return SummaryResult(text="<MERGED-STATE>", condensed=True)
        return SummaryResult(text="\n---\n".join(journal), condensed=False)

    def install(self, monkeypatch):
        monkeypatch.setattr(rs, "summarize_round", self.summarize_round)
        monkeypatch.setattr(rs, "meta_summarize", self.meta_summarize)
        return self


def _restart_swarm(monkeypatch, rounds: int, fresh: int, cap: int, rec=None):
    """A single-solver swarm that GAVE_UPs `rounds` times then wins."""
    rec = rec or RecordingSummarizer().install(monkeypatch)
    script = [_gave_up(findings=f"round-{i} work product") for i in range(rounds)] + [_flag()]
    solver = ScriptedSolver("m1", script)
    swarm = _swarm({"m1": solver},
                   fresh_restart_rounds=fresh,
                   max_attempts_per_challenge=cap)
    return swarm, solver, rec


# ---------------------------------------------------------------------------
# Step 1 — each GAVE_UP is distilled into exactly one journal entry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_gave_up_round_produces_one_journal_entry(monkeypatch) -> None:
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=3, fresh=5, cap=DEFAULT_CAP)

    await swarm.run()

    assert len(rec.round_calls) == 3, rec.round_calls
    journal = swarm._journals["m1"]
    assert len(journal) == 3
    assert all(j.startswith("<distilled#") for j in journal)


@pytest.mark.asyncio
async def test_the_distilled_input_is_the_round_output_not_a_placeholder(monkeypatch) -> None:
    """Regression: findings were overwritten by the last text block, so the
    "attempts and experience" being distilled were often just the tail."""
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=1, fresh=5, cap=DEFAULT_CAP)

    await swarm.run()

    assert "round-0 work product" in rec.round_calls[0]
    assert rec.round_calls[0] != "Your previous attempt did not find the flag."


@pytest.mark.asyncio
async def test_a_phase2_conclude_harvest_is_preferred_over_raw_findings(monkeypatch) -> None:
    from test_swarm import _interrupted

    rec = RecordingSummarizer().install(monkeypatch)
    solver = ScriptedSolver("m1", [_interrupted("raw phase-1 tail"), _flag()],
                            conclude_results=["[fact] key=0xBEEF\n[excluded] fmtstr"])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, conclude_timeout_s=7,
                   max_attempts_per_challenge=DEFAULT_CAP)

    await swarm.run()

    assert rec.round_calls, "nothing was distilled"
    assert "0xBEEF" in rec.round_calls[0], rec.round_calls[0]
    assert "raw phase-1 tail" not in rec.round_calls[0]


@pytest.mark.asyncio
async def test_empty_distillation_does_not_create_a_journal_entry(monkeypatch) -> None:
    """An empty entry would otherwise trigger the N-round merge early."""
    async def empty_round(text, settings, cost_tracker=None, agent_name=""):
        return ""

    async def meta(journal, settings, cost_tracker=None, agent_name=""):
        return SummaryResult(text="should not run", condensed=True)

    monkeypatch.setattr(rs, "summarize_round", empty_round)
    called = []

    async def spy_meta(journal, settings, cost_tracker=None, agent_name=""):
        called.append(list(journal))
        return await meta(journal, settings)

    monkeypatch.setattr(rs, "meta_summarize", spy_meta)

    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=1, max_attempts_per_challenge=DEFAULT_CAP)
    await swarm.run()

    assert swarm._journals.get("m1", []) == []
    assert called == [], "meta-summarize ran on an empty journal"


# ---------------------------------------------------------------------------
# Step 3 — the N-round merge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_merge_fires_exactly_at_n_rounds(monkeypatch) -> None:
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=6, fresh=5, cap=0)

    await swarm.run()

    assert len(rec.meta_calls) == 1, f"expected one merge, got {len(rec.meta_calls)}"
    assert len(rec.meta_calls[0]) == 5, "merge must see all 5 entries"
    # The merge replaced the 5 entries with one condensed state; the following
    # bump then appends to it, so the journal is [merged, newer].
    assert swarm._journals["m1"][0] == "<MERGED-STATE>"


@pytest.mark.asyncio
async def test_merge_does_not_fire_before_n_rounds(monkeypatch) -> None:
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=4, fresh=5, cap=DEFAULT_CAP)

    await swarm.run()

    assert rec.meta_calls == []
    assert len(swarm._journals["m1"]) == 4


@pytest.mark.asyncio
async def test_merged_state_is_what_the_next_round_is_seeded_with(monkeypatch) -> None:
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=6, fresh=5, cap=0)

    await swarm.run()

    # The bump queued right after the merge must carry the condensed state.
    merged_bump = solver.bumps[5]
    assert "<MERGED-STATE>" in merged_bump, merged_bump[:400]


@pytest.mark.asyncio
async def test_merge_cycle_can_repeat(monkeypatch) -> None:
    """With enough rounds the distil->merge cycle runs more than once."""
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=11, fresh=5, cap=0)

    await swarm.run()

    assert len(rec.meta_calls) == 2, f"expected two merge cycles, got {len(rec.meta_calls)}"
    # second cycle merges the previous merged state plus 4 new entries
    assert "<MERGED-STATE>" in rec.meta_calls[1][0]


# ---------------------------------------------------------------------------
# Failure mode — a summarizer outage must not defeat the bound
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failed_merge_keeps_the_journal_bounded(monkeypatch) -> None:
    """The critical regression.

    On gateway failure meta_summarize returned the full joined text and the
    caller stored it as ONE entry — so the carried state grew every cycle,
    which is exactly the context explosion this mechanism exists to prevent.
    And the gateway being down is the normal case here, not the rare one.
    """
    swarm, solver, rec = _restart_swarm(
        monkeypatch, rounds=11, fresh=5, cap=0,
        rec=RecordingSummarizer(meta_condensed=False).install(monkeypatch),
    )

    await swarm.run()

    journal = swarm._journals["m1"]
    assert len(journal) <= 5, f"journal grew unbounded: {len(journal)} entries"
    assert all(len(e) < 200 for e in journal), [len(e) for e in journal]


# ---------------------------------------------------------------------------
# Step 2 — what the restart payload carries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_restart_payload_carries_state_direction_and_skills(monkeypatch) -> None:
    from backend.approaches import Approach

    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=2, fresh=5, cap=DEFAULT_CAP)
    swarm._skill_pack_ids = ["ctf-misc"]
    swarm._approach_queue = [
        Approach(path="/skills/ctf-misc/encoding.md", title="Encoding Tricks", pack="ctf-misc"),
        Approach(path="/skills/ctf-misc/stego.md", title="Stego", pack="ctf-misc"),
    ]

    await swarm.run()

    payload = solver.bumps[0]
    assert "【探索状态】" in payload
    assert "<distilled#" in payload, "the distilled state is missing"
    assert "本轮主攻方向" in payload and "Encoding Tricks" in payload
    assert "/challenge/workspace/" in payload, "must tell the solver its files persist"


@pytest.mark.asyncio
async def test_restart_payload_rotates_to_the_next_untried_technique(monkeypatch) -> None:
    from backend.approaches import Approach

    swarm, solver, _rec = _restart_swarm(monkeypatch, rounds=3, fresh=5, cap=DEFAULT_CAP)
    swarm._approach_queue = [
        Approach(path=f"/skills/p/{n}.md", title=n, pack="p") for n in ("aaa", "bbb", "ccc")
    ]

    await swarm.run()

    import re

    # The payload's first bracketed section is 【探索状态】; extract the technique
    # from the 本轮主攻方向 marker specifically.
    assigned = [
        m.group(1)
        for b in solver.bumps
        for m in [re.search(r"本轮主攻方向（未试过）：【(.+?)】", b)]
        if m
    ]
    assert assigned == ["aaa", "bbb", "ccc"], assigned


@pytest.mark.asyncio
async def test_no_placeholder_when_siblings_have_nothing(monkeypatch) -> None:
    """Regression: the sentinel string made the truthiness test always pass, so
    every payload carried a useless "No sibling insights available yet." section."""
    swarm, solver, _rec = _restart_swarm(monkeypatch, rounds=2, fresh=5, cap=DEFAULT_CAP)

    await swarm.run()

    for payload in solver.bumps:
        assert "No sibling insights" not in payload, payload[:300]
        assert "全队进展" not in payload, "empty team section was injected"


@pytest.mark.asyncio
async def test_sibling_findings_do_reach_the_restart_payload(monkeypatch) -> None:
    swarm, solver, _rec = _restart_swarm(monkeypatch, rounds=2, fresh=5, cap=DEFAULT_CAP)
    # A sibling has confirmed something; the restarting solver must see it.
    swarm.findings["m2"] = "[fact] the archive password is hunter2"

    await swarm.run()

    assert any("hunter2" in b for b in solver.bumps), solver.bumps
    assert any("全队进展" in b for b in solver.bumps)


@pytest.mark.asyncio
async def test_sibling_infra_noise_is_not_forwarded(monkeypatch) -> None:
    swarm, solver, _rec = _restart_swarm(monkeypatch, rounds=2, fresh=5, cap=DEFAULT_CAP)
    swarm.findings["m2"] = "API Error: Connection refused (ECONNREFUSED)"

    await swarm.run()

    for payload in solver.bumps:
        assert "Connection refused" not in payload


# ---------------------------------------------------------------------------
# The restart must be a real restart
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_restart_forgets_the_session_not_just_resets_the_client(monkeypatch) -> None:
    """"重新开始" must actually drop the conversation.

    reset_session() on the Claude backend keeps the session id and resumes the
    same history — the opposite of what a fresh restart means.
    """
    order: list[str] = []

    class Tracking(ScriptedSolver):
        async def forget_session(self) -> None:
            order.append("forget")
            await super().forget_session()

        async def reset_session(self) -> None:
            order.append("reset")
            await super().reset_session()

    solver = Tracking("m1", [_gave_up(), _flag()])
    RecordingSummarizer().install(monkeypatch)
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)

    await swarm.run()

    assert order == ["forget"], f"expected forget_session, got {order}"


@pytest.mark.asyncio
async def test_session_is_forgotten_once_per_restart(monkeypatch) -> None:
    class Counting(ScriptedSolver):
        async def forget_session(self) -> None:
            self.forgets += 1

    solver = Counting("m1", [_gave_up()] * 3 + [_flag()])
    RecordingSummarizer().install(monkeypatch)
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)

    await swarm.run()

    assert solver.forgets == 3, solver.forgets


# ---------------------------------------------------------------------------
# Cost accounting for the extra model calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarizer_calls_are_attributed_to_the_challenge(monkeypatch) -> None:
    """The design calls the model again on purpose; that spend must be visible."""
    seen: list[str] = []

    async def spy_round(text, settings, cost_tracker=None, agent_name=""):
        seen.append(agent_name)
        return "distilled"

    async def spy_meta(journal, settings, cost_tracker=None, agent_name=""):
        seen.append(agent_name)
        return SummaryResult(text="merged", condensed=True)

    monkeypatch.setattr(rs, "summarize_round", spy_round)
    monkeypatch.setattr(rs, "meta_summarize", spy_meta)

    tracker = SimpleNamespace(by_agent={}, total_cost_usd=0.0, record_tokens=lambda *a, **k: None)
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=1, max_attempts_per_challenge=DEFAULT_CAP)
    swarm.cost_tracker = tracker

    await swarm.run()

    assert seen and all(a == "chal/summarizer" for a in seen), seen


# ---------------------------------------------------------------------------
# Team digest — siblings get the same distilled treatment as the own journal
# ---------------------------------------------------------------------------

class RecordingSiblingSummarizer(RecordingSummarizer):
    """Also records calls to summarize_siblings."""

    def __init__(self, meta_condensed: bool = True, sibling_condensed: bool = True):
        super().__init__(meta_condensed=meta_condensed)
        self.sibling_calls: list[list[str]] = []
        self.sibling_condensed = sibling_condensed

    async def summarize_siblings(self, reports, settings, cost_tracker=None, agent_name=""):
        self.sibling_calls.append(list(reports))
        if self.sibling_condensed:
            return SummaryResult(text="<TEAM-STATE>", condensed=True)
        return SummaryResult(text="\n".join(reports), condensed=False)

    def install(self, monkeypatch):
        super().install(monkeypatch)
        monkeypatch.setattr(rs, "summarize_siblings", self.summarize_siblings)
        return self


@pytest.mark.asyncio
async def test_restart_payload_carries_the_distilled_team_state(monkeypatch) -> None:
    """Siblings used to reach a restarting solver as "latest raw finding per
    model, clipped to 400 chars" — early discoveries dropped out and nothing was
    merged. They now get the same distillation the own journal gets."""
    rec = RecordingSiblingSummarizer().install(monkeypatch)
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)
    await swarm.message_bus.post("m2", "[fact] archive password is hunter2", kind="fact")
    await swarm.message_bus.post("m3", "[excluded] brute force is rate limited", kind="excluded")

    await swarm.run()

    assert rec.sibling_calls, "the team digest was never computed"
    joined = "\n".join(rec.sibling_calls[0])
    assert "hunter2" in joined, "an early sibling finding was dropped"
    assert "brute force" in joined
    assert any("全队进展" in b for b in solver.bumps), solver.bumps


@pytest.mark.asyncio
async def test_team_digest_reads_the_whole_blackboard_not_just_the_latest(monkeypatch) -> None:
    """`self.findings` only ever holds each model's LATEST finding; the digest
    must be built from the bus, which retains the history."""
    rec = RecordingSiblingSummarizer().install(monkeypatch)
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)
    await swarm.message_bus.post("m2", "[fact] EARLY discovery: the CRC is faked", kind="fact")
    await swarm.message_bus.post("m2", "[fact] LATE discovery: offset 72 overflows", kind="fact")
    # only the latest is in the per-model map
    swarm.findings["m2"] = "[fact] LATE discovery: offset 72 overflows"

    await swarm.run()

    joined = "\n".join(rec.sibling_calls[0])
    assert "EARLY discovery" in joined, "the early finding was lost"
    assert "LATE discovery" in joined


@pytest.mark.asyncio
async def test_team_digest_is_computed_once_per_bump_generation(monkeypatch) -> None:
    """Four solvers restarting in the same round must cost ONE call, not four."""
    rec = RecordingSiblingSummarizer().install(monkeypatch)
    solvers = {f"m{i}": ScriptedSolver(f"m{i}", [_gave_up(), _flag()]) for i in range(1, 5)}
    swarm = _swarm(solvers, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)
    await swarm.message_bus.post("m1", "[fact] something confirmed", kind="fact")
    await swarm.message_bus.post("m2", "[excluded] something ruled out", kind="excluded")

    await swarm.run()

    # one call per bump generation, not one per solver
    assert len(rec.sibling_calls) <= 2, f"digest recomputed {len(rec.sibling_calls)}x"


@pytest.mark.asyncio
async def test_team_digest_falls_back_to_raw_when_the_model_is_unavailable(monkeypatch) -> None:
    RecordingSiblingSummarizer(sibling_condensed=False).install(monkeypatch)
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)
    swarm.findings["m2"] = "[fact] raw fallback content"

    await swarm.run()

    assert any("raw fallback content" in b for b in solver.bumps), solver.bumps
    assert not any("<TEAM-STATE>" in b for b in solver.bumps)


@pytest.mark.asyncio
async def test_team_digest_needs_a_minimum_number_of_reports(monkeypatch) -> None:
    """One solver's own findings do not need merging with anything — do not spend
    a model call on it."""
    rec = RecordingSiblingSummarizer().install(monkeypatch)
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5,
                   max_attempts_per_challenge=DEFAULT_CAP, sibling_digest_min_reports=2)
    await swarm.message_bus.post("m1", "[fact] only my own finding", kind="fact")

    await swarm.run()

    assert rec.sibling_calls == [], "spent a digest call on a single report"


@pytest.mark.asyncio
async def test_team_digest_excludes_infra_noise(monkeypatch) -> None:
    rec = RecordingSiblingSummarizer().install(monkeypatch)
    solver = ScriptedSolver("m1", [_gave_up(), _flag()])
    swarm = _swarm({"m1": solver}, fresh_restart_rounds=5, max_attempts_per_challenge=DEFAULT_CAP)
    await swarm.message_bus.post("m2", "API Error: Connection refused (ECONNREFUSED)", kind="fact")
    await swarm.message_bus.post("m3", "[fact] real finding", kind="fact")

    await swarm.run()

    joined = "\n".join(rec.sibling_calls[0]) if rec.sibling_calls else ""
    assert "Connection refused" not in joined


# ---------------------------------------------------------------------------
# Interaction with the round cap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_cap_allows_exactly_one_merge_cycle(monkeypatch) -> None:
    """Documents the coupling: cap=6 with fresh=5 merges once, at bump 5, and the
    merged state drives round 6 before the solver retires. Raise the cap to get
    more cycles (see test_merge_cycle_can_repeat)."""
    swarm, solver, rec = _restart_swarm(monkeypatch, rounds=20, fresh=5, cap=6)

    await swarm.run()

    assert solver.rounds == 6
    assert len(rec.meta_calls) == 1
    assert "round cap" in swarm.retire_reason


# ---------------------------------------------------------------------------
# Summarizer budget — measured on a live run, not chosen by taste
# ---------------------------------------------------------------------------
# A 13-minute run on a 500-point crypto challenge logged
# `summarizer truncated at max_tokens=900 ... discarding the reply` five times.
# The journal merge the tests above depend on is only as good as the distillation
# feeding it, and the distillation was running on a budget it did not fit.


def _gw_settings(**over):
    base = {"gateway_base_url": "http://gw.test/v1", "gateway_api_key": "k",
            "summarizer_model": "sum-model",
            "summarizer_max_tokens": 2500, "summarizer_timeout_s": 60.0,
            "verifier_triage_max_tokens": 4000, "verifier_triage_timeout_s": 60.0}
    base.update(over)
    return SimpleNamespace(**base)


class _Capture:
    """MockTransport patch that records the request bodies and client kwargs."""

    def __init__(self, monkeypatch, reply: dict):
        self.bodies: list[dict] = []
        self.client_kwargs: list[dict] = []
        outer = self

        def handler(request):
            import json as _json
            outer.bodies.append(_json.loads(request.content))
            return httpx.Response(200, json=reply)

        transport = httpx.MockTransport(handler)
        real = httpx.AsyncClient

        class PatchedClient(real):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                outer.client_kwargs.append(kwargs)
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)


def _reply(content: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}}


@pytest.mark.asyncio
async def test_the_summarizer_budget_comes_from_settings_not_a_literal(monkeypatch) -> None:
    cap = _Capture(monkeypatch, _reply("distilled"))
    await _call_gateway("p", _gw_settings(summarizer_max_tokens=2500), "sum-model")
    assert cap.bodies[0]["max_tokens"] == 2500
    assert cap.client_kwargs[0]["timeout"] == 60.0


@pytest.mark.asyncio
async def test_an_explicit_budget_still_wins_over_the_summarizer_default(monkeypatch) -> None:
    """audit / strategist / briefing / triage pass their own measured budget."""
    cap = _Capture(monkeypatch, _reply("x"))
    await _call_gateway("p", _gw_settings(), "m", timeout_s=300.0, max_tokens=8000)
    assert cap.bodies[0]["max_tokens"] == 8000
    assert cap.client_kwargs[0]["timeout"] == 300.0


@pytest.mark.asyncio
async def test_a_truncated_distillation_puts_raw_text_in_the_journal(monkeypatch) -> None:
    """Pins the degraded-but-lossless behaviour, and why the budget matters.

    `summarize_round` returns `out or text`, so a discarded reply does NOT drop the
    round — it puts the UNDISTILLED round output in the journal instead. The state
    stays bounded (meta_summarize falls back to `unique[-N:]`), but five raw blobs
    replace one merged state card, which is what inflates the restart payload.
    """
    _Capture(monkeypatch, _reply("partial reasoning that never finished", finish="length"))
    raw = "x" * 1200
    out = await rs.summarize_round(raw, _gw_settings())
    assert out == raw, "the raw text must survive, not a half-distilled one"
    assert len(out) > 1000, "and it is the undistilled blob that lands in the journal"


@pytest.mark.asyncio
async def test_a_truncated_finding_summary_returns_the_full_text(monkeypatch) -> None:
    """The board must never carry half a finding labelled as the whole thing.

    `summarize_for_siblings` used to return `choices[0].message.content` with no
    finish_reason check, so a 700-token cap silently published a truncated summary
    — and `_summarize_for_board` accepts anything that differs from the input, so
    the loss was invisible downstream.
    """
    _Capture(monkeypatch, _reply("the first half of a long finding", finish="length"))
    text = "a long finding " * 200
    assert await summarize_mod.summarize_for_siblings(text, _gw_settings()) == text.strip()


@pytest.mark.asyncio
async def test_finding_summary_uses_the_configured_budget(monkeypatch) -> None:
    cap = _Capture(monkeypatch, _reply("short"))
    await summarize_mod.summarize_for_siblings("t" * 3000, _gw_settings(
        summarizer_max_tokens=2500))
    assert cap.bodies[0]["max_tokens"] == 2500, "the hardcoded 700 is back"


@pytest.mark.asyncio
async def test_triage_uses_the_configured_budget(monkeypatch) -> None:
    """A truncated triage means the verifier probes blind, spending the decisive
    probes VERIFIER_MAX_PROBES exists to ration."""
    cap = _Capture(monkeypatch, _reply("TESTABLE: yes\nLOAD_BEARING: yes\nWHY: w"))
    await verifier_mod.triage_claim(
        claim="q is 286 bits", name="c", category="crypto", description="d",
        settings=_gw_settings(verifier_triage_max_tokens=4000))
    assert cap.bodies[0]["max_tokens"] == 4000, "the hardcoded 1500 is back"
    assert cap.client_kwargs[0]["timeout"] == 60.0, "the hardcoded 25s is back"


class _ScriptedCapture:
    """MockTransport that returns a DIFFERENT reply per attempt, so a retry can
    be distinguished from a first-try success."""

    def __init__(self, monkeypatch, replies: list[dict]):
        self.bodies: list[dict] = []
        self.timeouts: list[float] = []
        self.replies = replies
        outer = self

        def handler(request):
            import json as _json
            outer.bodies.append(_json.loads(request.content))
            i = min(len(outer.bodies) - 1, len(outer.replies) - 1)
            return httpx.Response(200, json=outer.replies[i])

        transport = httpx.MockTransport(handler)
        real = httpx.AsyncClient

        class PatchedClient(real):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                outer.timeouts.append(kwargs.get("timeout"))
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)


@pytest.mark.asyncio
async def test_a_truncated_reply_is_retried_with_a_grown_budget(monkeypatch) -> None:
    """No fixed cap is provably enough for the two prompts that demand every
    technical detail survive, so the budget grows instead of the reply being lost."""
    cap = _ScriptedCapture(monkeypatch, [
        _reply("cut off mid-", finish="length"),
        _reply("the complete distillation"),
    ])
    out = await _call_gateway("p", _gw_settings(summarizer_max_tokens=2500), "m")
    assert out == "the complete distillation"
    assert [b["max_tokens"] for b in cap.bodies] == [2500, 2500 * rs.TRUNCATION_GROWTH]


@pytest.mark.asyncio
async def test_the_retry_scales_the_timeout_too(monkeypatch) -> None:
    """Retrying at the original timeout would just convert truncation into timeout."""
    cap = _ScriptedCapture(monkeypatch, [
        _reply("x", finish="length"), _reply("y"),
    ])
    await _call_gateway("p", _gw_settings(summarizer_timeout_s=60.0), "m")
    assert cap.timeouts == [60.0, 60.0 * rs.TRUNCATION_GROWTH]


@pytest.mark.asyncio
async def test_exhausted_retries_discard_the_reply_rather_than_parse_a_fragment(monkeypatch) -> None:
    cap = _ScriptedCapture(monkeypatch, [_reply("never finishes", finish="length")])
    out = await _call_gateway("p", _gw_settings(summarizer_truncation_retries=1), "m")
    assert out == ""
    assert len(cap.bodies) == 2, "one attempt plus one retry"


@pytest.mark.asyncio
async def test_retries_can_be_switched_off(monkeypatch) -> None:
    cap = _ScriptedCapture(monkeypatch, [_reply("never finishes", finish="length")])
    assert await _call_gateway("p", _gw_settings(summarizer_truncation_retries=0), "m") == ""
    assert len(cap.bodies) == 1


@pytest.mark.asyncio
async def test_the_growth_is_capped_so_an_unsatisfiable_prompt_cannot_run_away(monkeypatch) -> None:
    cap = _ScriptedCapture(monkeypatch, [_reply("x", finish="length")])
    await _call_gateway("p", _gw_settings(
        summarizer_max_tokens=rs.TRUNCATION_CEILING, summarizer_truncation_retries=3), "m")
    assert len(cap.bodies) == 1, "no headroom to grow into, so no point retrying"


@pytest.mark.asyncio
async def test_a_truncated_finding_summary_retries_before_falling_back(monkeypatch) -> None:
    """`summarize_for_siblings` delegates now, so it inherits the retry instead of
    keeping a second copy of the request that can drift from the first."""
    cap = _ScriptedCapture(monkeypatch, [
        _reply("half a finding", finish="length"), _reply("still half", finish="length"),
    ])
    text = "a long finding " * 200
    assert await summarize_mod.summarize_for_siblings(text, _gw_settings()) == text.strip()
    assert len(cap.bodies) == 2, "it must retry before giving up and keeping the full text"
