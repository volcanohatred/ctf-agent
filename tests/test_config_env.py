"""Settings must survive .env files written with trailing comments.

Docker's `--env-file` and compose's `env_file:` both take everything after the
first `=` verbatim, so `SOLVER_TIME_BUDGET_S=1500   # per-attempt wall clock`
reaches the process as that whole string. python-dotenv strips it, Docker does
not — and the shipped .env.example used to contain exactly such lines, so the
agent crashed at startup on any launch path that was not `docker compose`
(compose masked it by re-declaring those two keys in its environment: block).
"""

from __future__ import annotations

import pytest

from backend.config import Settings, _strip_inline_comment


def test_int_fields_survive_trailing_comments() -> None:
    s = Settings(
        SOLVER_TIME_BUDGET_S="1500     # per-attempt wall clock; 0 disables",
        SKILL_AFTER_BUMPS="1           # inject skill hints after N failed attempts",
        MAX_CONCURRENT_CHALLENGES="10  # parallel swarms",
        PATROL_INTERVAL_S="120          # scan every N seconds (0 disables)",
    )
    assert s.solver_time_budget_s == 1500
    assert s.skill_after_bumps == 1
    assert s.max_concurrent_challenges == 10
    assert s.patrol_interval_s == 120


def test_string_fields_are_trimmed_not_mangled() -> None:
    s = Settings(CONTAINER_MEMORY_LIMIT="16g   # per-sandbox memory")
    assert s.container_memory_limit == "16g"


def test_quoted_values_keep_their_hash() -> None:
    assert _strip_inline_comment('"keep # this"') == '"keep # this"'
    assert _strip_inline_comment("'keep # this'") == "'keep # this'"


def test_hash_without_preceding_space_is_preserved() -> None:
    """Passwords, tokens and flag-ish strings must not be truncated."""
    for value in ("abc#def", "flag{a#b}", "p@ss#1", "https://x/y#frag"):
        assert _strip_inline_comment(value) == value, value


def test_hash_after_whitespace_is_a_comment() -> None:
    assert _strip_inline_comment("value # note") == "value"
    assert _strip_inline_comment("value\t# note") == "value"
    assert _strip_inline_comment("value   # a # b") == "value"


def test_empty_and_blank_values() -> None:
    assert _strip_inline_comment("") == ""
    assert _strip_inline_comment("   ") == ""
    assert _strip_inline_comment("   # only a comment") == ""


def test_value_that_is_only_a_comment_becomes_empty() -> None:
    s = Settings(CTFD_TOKEN="   # filled in later")
    assert s.ctfd_token == ""


def test_real_dotenv_file_with_comments_parses(tmp_path, monkeypatch) -> None:
    """End to end through the actual .env reader."""
    env = tmp_path / ".env"
    env.write_text(
        "SOLVER_TIME_BUDGET_S=1500     # per-attempt wall clock; 0 disables\n"
        "SKILL_AFTER_BUMPS=1           # inject skill hints after N failed attempts\n"
        "CONTAINER_MEMORY_LIMIT=16g\n"
        "MAX_CONCURRENT_CHALLENGES=10\n"
        "CTFD_URL=https://ctf.example.com\n"
        'CTFD_TOKEN="tok#with-hash"\n',
        encoding="utf-8",
    )

    class Probed(Settings):
        model_config = {"env_file": str(env), "env_file_encoding": "utf-8",
                        "extra": "ignore"}

    s = Probed()
    assert s.solver_time_budget_s == 1500
    assert s.skill_after_bumps == 1
    assert s.container_memory_limit == "16g"
    assert s.ctfd_url == "https://ctf.example.com"


def test_env_var_with_comment_also_survives(monkeypatch) -> None:
    """The same value arriving as a real environment variable (the Docker case)."""
    monkeypatch.setenv("SOLVER_TIME_BUDGET_S", "900   # overridden")
    s = Settings(_env_file=None)
    assert s.solver_time_budget_s == 900


def test_new_env_example_has_no_inline_comments() -> None:
    """Guard the shipped template: comments belong on their own line.

    This is the root cause — a template that teaches an unsafe form guarantees
    every user's .env inherits it.
    """
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / ".env.example"
    if not example.exists():
        pytest.skip("no .env.example in this checkout")

    offenders = []
    for lineno, raw in enumerate(example.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        value = line.split("=", 1)[1].strip()
        if value[:1] in ("'", '"'):
            continue
        if " #" in value or "\t#" in value:
            offenders.append(f"{example.name}:{lineno}: {line.split('=')[0]}")
    assert not offenders, ".env.example teaches inline comments:\n" + "\n".join(offenders)
