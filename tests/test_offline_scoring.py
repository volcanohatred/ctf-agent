"""Offline benchmark scoring — answer keys and what happens without one.

A challenge with no `flag:` in metadata.yml cannot be scored offline at all:
`try_submit_flag` answers DRY RUN, so no solver can ever reach FLAG_FOUND and the
challenge is unsolvable by construction. That used to surface as a bare
"unsolved", which is indistinguishable from a model failing — and cost a real
600s run before the cause was found. Worse, the report's `else: result.solved =
True` branch was unreachable dead code that implied keyless challenges could
pass.

The trap is easy to fall into because a challenge's NAME and its directory name
are not always the same: `challenges/regression/misc-encoding/` contains
`name: reg-misc-encoding`, and a keyless flat duplicate sits at
`challenges/reg-misc-encoding/`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.offline import BenchResult, discover_challenges, run_benchmark

KEYED = 'name: keyed-chal\ncategory: misc\nvalue: 10\nflag: "flag{real_answer}"\n'
KEYLESS = "name: keyless-chal\ncategory: misc\nvalue: 10\n"


def _write_challenge(root: Path, dirname: str, yaml_text: str) -> Path:
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.yml").write_text(yaml_text, encoding="utf-8")
    return d


def _settings(**over) -> SimpleNamespace:
    base = dict(
        anthropic_base_url="", anthropic_auth_token="", gateway_base_url="",
        gateway_api_key="", summarizer_model="", startup_health_check=False,
        health_check_timeout_s=1.0, health_check_retries=1,
        cleanup_orphans_on_start=False, max_concurrent_challenges=1,
        solver_time_budget_s=1500, conclude_timeout_s=90,
        max_attempts_per_challenge=12, max_infra_failures=3,
        fresh_restart_rounds=5, skill_after_bumps=1, escalate_after_bumps=0,
        summarize_threshold_chars=900, sandbox_image="ctf-sandbox",
        container_memory_limit="1g", container_cpu_limit=1.0,
        sandbox_command_timeout_s=60, host_project_dir="", workspace_root="",
        skills_dir="", skills_host_dir="", sandbox_proxy_url="",
        bump_cooldown_base_s=0, bump_cooldown_step_s=0, bump_cooldown_max_s=0,
        quota_backoff_base_s=0, quota_backoff_max_s=0,
        coordinator_model="", patrol_interval_s=0, msg_port=0,
        sibling_digest_min_reports=2, sibling_digest_refresh_bumps=1,
        coordinator_turn_timeout_s=300, solver_teardown_grace_s=60,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovery_reports_which_challenges_have_keys(tmp_path) -> None:
    _write_challenge(tmp_path, "keyed", KEYED)
    _write_challenge(tmp_path, "keyless", KEYLESS)

    found = {c.name: c for c in discover_challenges(str(tmp_path))}

    assert found["keyed-chal"].meta.flags == ["flag{real_answer}"]
    assert found["keyless-chal"].meta.flags == []


def test_directory_name_and_challenge_name_can_differ(tmp_path) -> None:
    """The trap that cost a real run: the keyed copy lives in a directory whose
    name is NOT the challenge's `name:` field."""
    _write_challenge(tmp_path, "misc-encoding",
                     'name: reg-misc-encoding\ncategory: misc\nflag: "flag{x}"\n')

    found = discover_challenges(str(tmp_path))

    assert len(found) == 1
    assert found[0].name == "reg-misc-encoding"
    assert Path(found[0].directory).name == "misc-encoding"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_keyless_challenge_can_never_be_marked_solved() -> None:
    """FLAG_FOUND is unreachable without a key, so `solved = True` was dead code."""
    r = BenchResult(name="keyless", category="misc", has_answer_key=False)
    assert r.solved is False
    assert r.verified is False


def test_scoreable_denominator_excludes_keyless() -> None:
    """Including unscoreable challenges understates the solve rate and hides why."""
    results = [
        BenchResult(name="a", category="m", solved=True, verified=True, has_answer_key=True),
        BenchResult(name="b", category="m", solved=False, has_answer_key=True),
        BenchResult(name="c", category="m", solved=False, has_answer_key=False),
    ]
    scoreable = sum(1 for r in results if r.has_answer_key)
    solved = sum(1 for r in results if r.solved)
    assert scoreable == 2
    assert round(solved / scoreable, 2) == 0.5      # honest
    assert round(solved / len(results), 2) == 0.33  # what the old code reported


# ---------------------------------------------------------------------------
# run_benchmark refuses to produce a meaningless report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_keyless_selection_errors_out_before_spending_anything(tmp_path) -> None:
    _write_challenge(tmp_path, "a", KEYLESS)
    _write_challenge(tmp_path, "b", KEYLESS)

    report = await run_benchmark(
        settings=_settings(), challenges_root=str(tmp_path), concurrency=1,
        time_limit_s=60, output_path=str(tmp_path / "report.json"),
    )

    assert "error" in report, report
    assert "answer key" in report["error"]
    # the message must point at the likely cause, not just state the symptom
    assert "directory name" in report["error"]
    # it must not have written a report full of meaningless zeros
    assert report["results"] == []
    assert not (tmp_path / "report.json").exists()


@pytest.mark.asyncio
async def test_no_challenges_still_errors_cleanly(tmp_path) -> None:
    report = await run_benchmark(
        settings=_settings(), challenges_root=str(tmp_path), concurrency=1,
        time_limit_s=60, output_path=str(tmp_path / "report.json"),
    )
    assert "error" in report and "no challenges found" in report["error"]


@pytest.mark.asyncio
async def test_report_flags_an_unusable_run(tmp_path, monkeypatch) -> None:
    """A mixed selection must still run, but the report has to say which
    challenges could not be scored."""
    _write_challenge(tmp_path, "keyed", KEYED)
    _write_challenge(tmp_path, "keyless", KEYLESS)

    # Do not actually launch solvers — no Docker, no LLM in this suite.
    async def fake_solve_one(ch, settings, model_specs, cost_tracker, time_limit_s):
        return BenchResult(name=ch.name, category=ch.meta.category,
                           has_answer_key=bool(ch.meta.flags),
                           error="" if ch.meta.flags else "no answer key in metadata.yml")

    monkeypatch.setattr("backend.offline._solve_one", fake_solve_one)

    out = tmp_path / "report.json"
    report = await run_benchmark(
        settings=_settings(), challenges_root=str(tmp_path), concurrency=1,
        time_limit_s=60, output_path=str(out),
    )

    summary = report["benchmark"]
    assert summary["unscoreable_no_answer_key"] == 1
    assert summary["total_challenges"] == 2
    # one of two could be scored, neither solved -> rate over the scoreable set
    assert summary["solve_rate"] == 0.0
    assert out.exists()
    written = yaml.safe_load(out.read_text(encoding="utf-8"))  # valid YAML/JSON
    assert written["benchmark"]["unscoreable_no_answer_key"] == 1


@pytest.mark.asyncio
async def test_infra_failures_mark_the_run_unusable(tmp_path, monkeypatch) -> None:
    _write_challenge(tmp_path, "keyed", KEYED)

    async def fake_solve_one(ch, settings, model_specs, cost_tracker, time_limit_s):
        return BenchResult(name=ch.name, category="misc", has_answer_key=True,
                           infra_failure=True,
                           retire_reason="3 consecutive infrastructure failures")

    monkeypatch.setattr("backend.offline._solve_one", fake_solve_one)

    report = await run_benchmark(
        settings=_settings(), challenges_root=str(tmp_path), concurrency=1,
        time_limit_s=60, output_path=str(tmp_path / "r.json"),
    )

    assert report["benchmark"]["infra_failures"] == 1
    assert report["benchmark"]["usable_run"] is False


@pytest.mark.asyncio
async def test_healthy_run_is_marked_usable(tmp_path, monkeypatch) -> None:
    _write_challenge(tmp_path, "keyed", KEYED)

    async def fake_solve_one(ch, settings, model_specs, cost_tracker, time_limit_s):
        return BenchResult(name=ch.name, category="misc", solved=True, verified=True,
                           flag="flag{real_answer}", has_answer_key=True)

    monkeypatch.setattr("backend.offline._solve_one", fake_solve_one)

    report = await run_benchmark(
        settings=_settings(), challenges_root=str(tmp_path), concurrency=1,
        time_limit_s=60, output_path=str(tmp_path / "r.json"),
    )

    summary = report["benchmark"]
    assert summary["usable_run"] is True
    assert summary["solved"] == 1 and summary["solve_rate"] == 1.0
    assert summary["unscoreable_no_answer_key"] == 0


# ---------------------------------------------------------------------------
# The trust/stagnation audit trail in the report
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_report_carries_the_trust_audit_per_challenge(tmp_path, monkeypatch) -> None:
    """A solve rate cannot distinguish "converged on verified facts" from "each
    model asserted things nobody checked until the clock ran out". These counters
    can, and a post-mortem that lacks them blames the models for a coordination
    failure."""
    # Distinct `name:` per challenge — discovery keys off metadata, not the
    # directory, so two copies of KEYED would both be "keyed-chal".
    _write_challenge(tmp_path, "grinder",
                     'name: grinder\ncategory: web\nvalue: 10\nflag: "flag{real_answer}"\n')
    _write_challenge(tmp_path, "healthy",
                     'name: healthy\ncategory: web\nvalue: 10\nflag: "flag{real_answer}"\n')

    async def fake_solve_one(ch, settings, model_specs, cost_tracker, time_limit_s):
        if ch.name == "grinder":
            return BenchResult(name=ch.name, category="web", has_answer_key=True,
                               bumps=24, stale_breaks=5, strategist_pivots=2,
                               corroborated_facts=1, unverified_claims=40,
                               retire_reason="round cap reached (12)")
        return BenchResult(name=ch.name, category="web", solved=True, verified=True,
                           flag="flag{real_answer}", has_answer_key=True, bumps=2,
                           corroborated_facts=6, unverified_claims=2)

    monkeypatch.setattr("backend.offline._solve_one", fake_solve_one)

    report = await run_benchmark(
        settings=_settings(), challenges_root=str(tmp_path), concurrency=1,
        time_limit_s=60, output_path=str(tmp_path / "r.json"),
    )

    by_name = {r["name"]: r for r in report["results"]}
    assert set(by_name) == {"grinder", "healthy"}
    grinder = by_name["grinder"]
    assert grinder["stale_breaks"] == 5
    assert grinder["strategist_pivots"] == 2
    assert grinder["unverified_claims"] == 40
    assert grinder["corroborated_facts"] == 1
    # The healthy run is distinguishable from it on more than just `solved`.
    assert by_name["healthy"]["stale_breaks"] == 0
    assert by_name["healthy"]["corroborated_facts"] == 6


def test_bench_result_audit_counters_default_to_zero() -> None:
    """Every field must have a default: `_solve_one` writes them in a `finally`
    block, so a swarm that crashed before publishing must still yield a row."""
    r = BenchResult(name="x", category="web")
    assert (r.stale_breaks, r.strategist_pivots) == (0, 0)
    assert (r.corroborated_facts, r.unverified_claims) == (0, 0)


@pytest.mark.asyncio
async def test_solve_one_records_the_audit_even_when_the_swarm_crashes(tmp_path, monkeypatch) -> None:
    """The `finally` path is the one that matters most: a crashed or timed-out
    swarm is exactly the run somebody needs to post-mortem."""
    import asyncio

    import backend.offline as offline
    from backend.cost_tracker import CostTracker

    _write_challenge(tmp_path, "boom", KEYED)
    ch = offline.discover_challenges(str(tmp_path))[0]

    class ExplodingSwarm:
        retire_reason = "round cap reached (12)"

        def __init__(self, *args, **kwargs):
            self.cancel_event = asyncio.Event()
            self.winner = None
            self.winner_spec = ""

        async def run(self):
            raise RuntimeError("sandbox exploded")

        def kill(self, reason=""):
            self.cancel_event.set()

        def get_status(self):
            return {"bumps": 7, "stale_breaks": 3, "strategist_pivots": 1,
                    "corroborated_facts": 2, "unverified_claims": 9,
                    "probe_verified": 4, "probe_refuted": 2, "contradictions": 1}

    monkeypatch.setattr("backend.agents.swarm.ChallengeSwarm", ExplodingSwarm)

    result = await offline._solve_one(ch, _settings(), ["m1"], CostTracker(), 0)

    assert "RuntimeError" in result.error
    assert result.bumps == 7
    assert result.stale_breaks == 3
    assert result.strategist_pivots == 1
    assert result.corroborated_facts == 2
    assert result.unverified_claims == 9
    assert result.probe_verified == 4
    assert result.probe_refuted == 2
    assert result.contradictions == 1
    assert result.infra_failure is False, "a round cap is not an infrastructure failure"


@pytest.mark.asyncio
async def test_a_partial_status_snapshot_cannot_mask_the_real_error(tmp_path, monkeypatch) -> None:
    """The audit read happens in a `finally`, i.e. usually while handling a crash.
    A KeyError there would replace the real exception and destroy the only
    diagnostic anybody was going to get — so every counter has to tolerate a
    swarm that reports none of them."""
    import asyncio

    import backend.offline as offline
    from backend.cost_tracker import CostTracker

    _write_challenge(tmp_path, "sparse", KEYED)
    ch = offline.discover_challenges(str(tmp_path))[0]

    class SparseSwarm:
        retire_reason = ""

        def __init__(self, *args, **kwargs):
            self.cancel_event = asyncio.Event()
            self.winner = None
            self.winner_spec = ""

        async def run(self):
            raise RuntimeError("the actual problem")

        def kill(self, reason=""):
            self.cancel_event.set()

        def get_status(self):
            return {}          # no counters at all

    monkeypatch.setattr("backend.agents.swarm.ChallengeSwarm", SparseSwarm)

    result = await offline._solve_one(ch, _settings(), ["m1"], CostTracker(), 0)

    assert "the actual problem" in result.error, "the real error must survive"
    assert (result.bumps, result.probe_refuted, result.contradictions) == (0, 0, 0)


# ---------------------------------------------------------------------------
# The shipped challenge set
# ---------------------------------------------------------------------------

def test_bundled_regression_set_is_fully_keyed() -> None:
    """Guard the set the README's verification numbers are based on."""
    root = Path(__file__).resolve().parent.parent / "challenges" / "regression"
    if not root.is_dir():
        pytest.skip("challenges/regression not present in this checkout")

    keyless = []
    for meta_path in sorted(root.rglob("metadata.yml")):
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        if not (data.get("flag") or data.get("flags")):
            keyless.append(str(meta_path.parent.relative_to(root)))
    assert not keyless, f"regression challenges without an answer key: {keyless}"


# ---------------------------------------------------------------------------
# ctf-solve must warn, not silently produce an unscorable run
# ---------------------------------------------------------------------------

def _run_single_kwargs(tmp_path: Path, yaml_text: str) -> dict:
    d = _write_challenge(tmp_path, "chal", yaml_text)
    return {"challenge_dir": str(d)}


def test_single_challenge_mode_warns_when_unscoreable(tmp_path) -> None:
    """The trap that cost a real 600s run: a keyless challenge under --no-submit
    can never confirm a flag, and it used to end in a bare "No flag found" that
    looked exactly like a model failure.

    Synchronous on purpose — CliRunner drives the command, which calls
    asyncio.run() itself, and that cannot be nested inside a running loop.
    """
    from click.testing import CliRunner

    from backend.cli import main

    d = _write_challenge(tmp_path, "keyless", KEYLESS)
    runner = CliRunner()
    # --skip-health-check and a tiny time limit keep this fast and offline; the
    # warning is printed before any solver work starts.
    result = runner.invoke(
        main,
        ["--challenge", str(d), "--no-submit", "--time-limit", "1",
         "--skip-health-check", "--models", "claude-sdk/fake"],
        catch_exceptions=False,
    )

    out = result.output
    assert "No answer key" in out, out[:1200]
    # and it must say what to do about it, not just that something is wrong
    assert "flag:" in out or "--no-submit" in out


def test_keyed_challenge_does_not_warn(tmp_path) -> None:
    from click.testing import CliRunner

    from backend.cli import main

    d = _write_challenge(tmp_path, "keyed", KEYED)
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--challenge", str(d), "--no-submit", "--time-limit", "1",
         "--skip-health-check", "--models", "claude-sdk/fake"],
        catch_exceptions=False,
    )
    assert "No answer key" not in result.output


def test_metadata_flags_reach_the_swarm_expected_flags(tmp_path) -> None:
    """The warning must key off the same field the verifier uses."""
    from backend.prompts import ChallengeMeta

    d = _write_challenge(tmp_path, "keyed", KEYED)
    meta = ChallengeMeta.from_yaml(d / "metadata.yml")
    assert meta.flags == ["flag{real_answer}"]

    d2 = _write_challenge(tmp_path, "keyless", KEYLESS)
    assert ChallengeMeta.from_yaml(d2 / "metadata.yml").flags == []
