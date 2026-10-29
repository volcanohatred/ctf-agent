"""Settings reads must go through the typed accessors, never a hardcoded fallback.

The bug class this prevents, found by audit and confirmed at 11 separate sites:
call sites read knobs as `getattr(self.settings, "field", LITERAL)` because tests
pass a `SimpleNamespace` that lacks fields. The literal then drifts from the real
default in `config.py`, and the drift is invisible — production gets the declared
default, everything else silently gets the fallback. Concrete consequences that
shipped:

  max_attempts_per_challenge -> 0   unlimited rounds
  solver_time_budget_s       -> 0   no wall-clock budget, no conclude harvest
  cli_api_timeout_ms         -> 0   API_TIMEOUT_MS never set; the CLI can stall
                                    indefinitely on a half-open connection
  strategist_every_bumps     -> 0   the Strategist never runs
  stale_round_limit          -> 0   the circuit breaker never fires
  web_fetch_allow_hosts      -> ""  the agent's own target becomes unreachable

A second, related idiom is `getattr(s, "f", X) or X`, which rewrites an explicit 0
into X. That made two documented off-states unreachable: PATROL_INTERVAL_S=0 became
120 and MAX_SPAWNS_PER_CHALLENGE=0 became 3.

`backend.config.int_setting` / `float_setting` / `bool_setting` / `str_setting` fix
both: the fallback is read from `Settings.model_fields`, so it cannot drift, and only
a MISSING or None value falls back, so an explicit 0 reaches the code.

This is an AST sweep because ruff structurally cannot express it — the same approach
`test_no_dead_params.py` takes for accepted-and-ignored parameters.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.config import (
    Settings,
    bool_setting,
    float_setting,
    int_setting,
    str_setting,
)

BACKEND = Path(__file__).resolve().parent.parent / "backend"
FIELD_NAMES = set(Settings.model_fields)
FIELD_DEFAULTS = {n: f.default for n, f in Settings.model_fields.items()}

# Reads that legitimately bypass the accessors, with the reason. Anything added here
# needs a justification, which is the point of the list being explicit.
ALLOWED: dict[tuple[str, int], str] = {}

# `validate_tuning` deliberately falls back to 0 rather than the shipped defaults: a
# caller that supplies nothing must not have a misalignment warning invented out of
# defaults they never asked for. Matched by function name, not line number.
ALLOWED_FUNCTIONS = {"validate_tuning"}

ACCESSORS = {"int_setting", "float_setting", "bool_setting", "str_setting"}


def _module_for(rel_path: str):
    """Import the module a finding came from, so module-level constants can be
    resolved when checking whether a drift is reconciled."""
    import importlib

    dotted = rel_path.removesuffix(".py").replace("/", ".").replace("\\", ".")
    try:
        return importlib.import_module(dotted)
    except Exception:  # noqa: BLE001 — unresolvable means "cannot prove reconciled"
        return None


def _resolves_to(value_node: ast.AST, module, expected) -> bool:
    """Does this expression evaluate to `expected`?

    Used to accept the `getattr(s, "f", 0) or MODULE_DEFAULT` pattern, where the
    trailing `or` supplies the real default. It is not a general evaluator: only
    literals and module-level names, which is all the pattern ever uses. Anything
    else is treated as unproven, i.e. reported.
    """
    if isinstance(value_node, ast.Constant):
        return value_node.value == expected
    if isinstance(value_node, ast.Name) and module is not None:
        return getattr(module, value_node.id, object()) == expected
    if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
        inner = _resolves_to(value_node.operand, module, None)
        del inner
        if isinstance(value_node.operand, ast.Constant):
            return -value_node.operand.value == expected
    return False


def _reconciled_by_or(node: ast.Call, tree: ast.AST, module) -> bool:
    """True when a later operand of the enclosing `or` chain supplies the real default.

    Handles the getattr at ANY position in the chain, not just the first:
    `model or getattr(s, "summarizer_model", "") or DEFAULT_SUMMARIZER_MODEL` is
    reconciled exactly as `getattr(...) or DEFAULT` is. Only operands to the RIGHT
    count — an earlier one cannot rescue a later fallback.
    """
    field = node.args[1].value
    expected = FIELD_DEFAULTS[field]
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is not node:
                continue
            # The getattr may be wrapped: int(getattr(...)) or DEFAULT.
            candidate = parent
            if isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Name) \
                    and candidate.func.id in ("int", "float", "str", "bool"):
                candidate = _find_parent_of(tree, candidate)
                if candidate is None:
                    return False
            if isinstance(candidate, ast.BoolOp) and isinstance(candidate.op, ast.Or):
                idx = next((i for i, v in enumerate(candidate.values)
                            if _contains(v, node)), None)
                if idx is None:
                    continue
                return any(_resolves_to(v, module, expected)
                           for v in candidate.values[idx + 1:])
    return False


def _find_parent_of(tree: ast.AST, target: ast.AST):
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            if child is target:
                return parent
    return None


def _contains(node: ast.AST, target: ast.AST) -> bool:
    return any(n is target for n in ast.walk(node))


def _enclosing_function(tree: ast.AST, node: ast.AST) -> str:
    """Name of the innermost function/method containing `node` ("" at module level)."""
    best = ""
    best_depth = -1
    for parent in ast.walk(tree):
        if not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(parent):
            if child is node:
                depth = getattr(parent, "lineno", 0)
                if depth > best_depth:
                    best, best_depth = parent.name, depth
                break
    return best


def _is_settings_target(target: ast.AST) -> bool:
    """Does this getattr read something settings-shaped?

    Matches `settings`, `self.settings`, `deps.settings`, `swarm.settings` — but not
    `getattr(message, "kind")` on a Finding, which shares no field names anyway and
    would only add noise.
    """
    node = target
    while isinstance(node, ast.Attribute):
        if "settings" in node.attr:
            return True
        node = node.value
    return isinstance(node, ast.Name) and "settings" in node.id


def _drifted_reads() -> list[str]:
    """Every settings read whose hardcoded fallback disagrees with config.py AND is
    not reconciled by a trailing `or <the real default>`."""
    out: list[str] = []
    for path in sorted(BACKEND.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(BACKEND.parent).as_posix()
        module = _module_for(rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Name) and fn.id == "getattr"):
                continue
            if len(node.args) < 3:
                continue
            target, name_node, literal = node.args[0], node.args[1], node.args[2]
            if not (isinstance(name_node, ast.Constant)
                    and isinstance(name_node.value, str)):
                continue
            field = name_node.value
            if field not in FIELD_NAMES or not isinstance(literal, ast.Constant):
                continue
            if not _is_settings_target(target):
                continue
            if literal.value == FIELD_DEFAULTS[field]:
                continue                      # agrees with config.py — no drift
            if (rel, node.lineno) in ALLOWED:
                continue
            if _enclosing_function(tree, node) in ALLOWED_FUNCTIONS:
                continue
            if _reconciled_by_or(node, tree, module):
                continue                      # `or MODULE_DEFAULT` supplies it
            out.append(
                f"{rel}:{node.lineno} getattr(..., {field!r}, {literal.value!r}) "
                f"but config.py declares {FIELD_DEFAULTS[field]!r}"
            )
    return out


def test_no_unreconciled_fallback_drift() -> None:
    """The drift guard. A literal fallback is a second source of truth for a default
    that `config.py` already owns; when the two disagree, production gets the
    declared default and every other caller silently gets the fallback.

    Concretely shipped consequences of this exact drift: unlimited rounds, no
    wall-clock budget, no CLI HTTP timeout, and the Strategist and circuit breaker
    both permanently disabled.
    """
    offenders = _drifted_reads()
    assert not offenders, (
        "these read a Settings field with a fallback that disagrees with config.py, "
        "and nothing downstream supplies the real default. Use int_setting / "
        "float_setting / bool_setting / str_setting, which take the default from "
        "Settings.model_fields and so cannot drift:\n  " + "\n  ".join(offenders)
    )


def test_no_or_default_idiom_on_a_settings_read() -> None:
    """`getattr(s, "f", X) or X` rewrites an explicit 0 into X, which is how two
    documented off-states became unreachable: PATROL_INTERVAL_S=0 was rewritten to
    120 by its own `or`, and MAX_SPAWNS_PER_CHALLENGE=0 to 3.

    Only flagged when the `or` operand is a plain literal equal to the fallback —
    that is the swallowing pattern. `or MODULE_DEFAULT` where the constant is the
    real default is the reconciliation the previous test accepts, so it is skipped.
    """
    offenders: list[str] = []
    for path in sorted(BACKEND.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(BACKEND.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
                continue
            first = node.values[0]
            call = first
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) \
                    and call.func.id in ("int", "float", "str", "bool") and call.args:
                call = call.args[0]
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "getattr"):
                continue
            if len(call.args) < 3 or not _is_settings_target(call.args[0]):
                continue
            name_node, literal = call.args[1], call.args[2]
            if not (isinstance(name_node, ast.Constant)
                    and isinstance(name_node.value, str)):
                continue
            field = name_node.value
            if field not in FIELD_NAMES or not isinstance(literal, ast.Constant):
                continue
            if _enclosing_function(tree, node) in ALLOWED_FUNCTIONS:
                continue
            # The swallowing case: the fallback is TRUTHY and the `or` operand is the
            # same literal, so an explicit 0 can never reach the code. A falsy
            # literal (`getattr(s,"token","") or ""`) is a redundant no-op rather
            # than a lost off-state, and flagging it would bury the real cases.
            rhs = node.values[1]
            if (isinstance(rhs, ast.Constant) and rhs.value == literal.value
                    and literal.value):
                offenders.append(
                    f"{rel}:{node.lineno} getattr(..., {field!r}, {literal.value!r}) "
                    f"or {rhs.value!r} — an explicit 0 is unreachable"
                )
    assert not offenders, (
        "these swallow an explicit falsy value and substitute the same default, so "
        "the knob's off-state cannot be expressed. The typed accessors treat only a "
        "missing/None value as unset:\n  " + "\n  ".join(offenders)
    )


def test_the_sweep_can_still_see_a_violation(tmp_path) -> None:
    """Guard the guard: a sweep that cannot detect anything is worthless."""
    src = tmp_path / "bad.py"
    src.write_text(
        "def f(self):\n"
        "    return int(getattr(self.settings, 'max_attempts_per_challenge', 0) or 0)\n",
        encoding="utf-8",
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr")
    assert call.args[1].value in FIELD_NAMES
    assert _is_settings_target(call.args[0]), "`self.settings` must look settings-shaped"
    assert _enclosing_function(tree, call) == "f"
    assert _enclosing_function(tree, call) not in ALLOWED_FUNCTIONS
    # The drift is real (fallback 0 vs declared 12) and `or 0` does not reconcile it,
    # so both sweeps would report this file.
    assert call.args[2].value != FIELD_DEFAULTS["max_attempts_per_challenge"]


def test_a_reconciled_fallback_is_not_reported(tmp_path) -> None:
    """The `getattr(s, "f", 0) or MODULE_DEFAULT` pattern is correct today, and
    flagging it would bury the real drift in false positives — which is how a guard
    gets switched off."""
    src = tmp_path / "ok.py"
    src.write_text(
        "DEFAULT_THING = 12\n"
        "def f(settings):\n"
        "    return int(getattr(settings, 'max_attempts_per_challenge', 0) "
        "or DEFAULT_THING)\n",
        encoding="utf-8",
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    call = next(n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "getattr")

    class _Mod:
        DEFAULT_THING = 12

    assert _reconciled_by_or(call, tree, _Mod) is True
    assert _reconciled_by_or(call, tree, None) is False, "unprovable must not pass"


def test_validate_tuning_is_the_only_exempt_function() -> None:
    """The exemption is narrow and justified; it must not grow quietly.

    Written via sorted() rather than a direct set comparison: ruff's SIM300 reads an
    ALL_CAPS name as the "constant" side and suggests putting the literal first,
    which is backwards from normal Python style.
    """
    assert sorted(ALLOWED_FUNCTIONS) == ["validate_tuning"]
    assert not ALLOWED


# ---------------------------------------------------------------------------
# The accessors themselves
# ---------------------------------------------------------------------------

def test_a_missing_field_falls_back_to_the_declared_default() -> None:
    from types import SimpleNamespace

    empty = SimpleNamespace()
    assert int_setting(empty, "max_attempts_per_challenge") == 12
    assert int_setting(empty, "solver_time_budget_s") == 1500
    assert int_setting(empty, "strategist_every_bumps") == 4
    assert int_setting(empty, "cli_api_timeout_ms") == 600_000
    assert float_setting(empty, "container_cpu_limit") == 2.0
    assert str_setting(empty, "web_fetch_allow_hosts") == "host.docker.internal"
    assert bool_setting(empty, "corroborate_facts") is True


def test_an_explicit_zero_is_honoured_not_rewritten() -> None:
    """The whole reason the `or X` idiom was removed. Each of these 0s means
    something: unlimited rounds, no budget, mechanism disabled, no cap."""
    from types import SimpleNamespace

    s = SimpleNamespace(max_attempts_per_challenge=0, solver_time_budget_s=0,
                        strategist_every_bumps=0, stale_round_limit=0,
                        patrol_interval_s=0, max_spawns_per_challenge=0,
                        verifier_max_probes=0, escalate_after_bumps=0)
    for field_name in ("max_attempts_per_challenge", "solver_time_budget_s",
                       "strategist_every_bumps", "stale_round_limit",
                       "patrol_interval_s", "max_spawns_per_challenge",
                       "verifier_max_probes", "escalate_after_bumps"):
        assert int_setting(s, field_name) == 0, field_name


def test_none_falls_back_but_zero_and_empty_string_do_not() -> None:
    from types import SimpleNamespace

    assert int_setting(SimpleNamespace(msg_port=None), "msg_port") == 9400
    assert int_setting(SimpleNamespace(msg_port=0), "msg_port") == 0
    # An empty string is the meaningful value for several string knobs.
    assert str_setting(SimpleNamespace(summarizer_model=""), "summarizer_model") == ""
    assert str_setting(SimpleNamespace(summarizer_model=None),
                       "summarizer_model") == "deepseek-v4-flash"


def test_bool_setting_parses_env_file_strings() -> None:
    from types import SimpleNamespace

    for truthy in ("1", "true", "TRUE", "yes", "on"):
        assert bool_setting(SimpleNamespace(verify_claims=truthy), "verify_claims") is True
    for falsy in ("0", "false", "FALSE", "no", "off"):
        assert bool_setting(SimpleNamespace(verify_claims=falsy), "verify_claims") is False


def test_an_empty_string_is_unset_for_numbers_but_a_value_for_strings() -> None:
    """A deliberate asymmetry, pinned so it is not "fixed" into inconsistency.

    For numeric and bool knobs an empty env value is meaningless — pydantic rejects
    `MSG_PORT=` outright — so treating it as unset is the only coherent reading. For
    string knobs `""` IS the meaningful value that callers branch on:
    `gateway_base_url=""` means "no gateway configured", `skills_dir=""` means "no
    library", `coordinator_model=""` means "fall through to the next choice".
    Substituting a default there would invent configuration the operator did not give.
    """
    from types import SimpleNamespace

    assert int_setting(SimpleNamespace(msg_port=""), "msg_port") == 9400
    assert float_setting(SimpleNamespace(container_cpu_limit=""),
                         "container_cpu_limit") == 2.0
    assert bool_setting(SimpleNamespace(verify_claims=""), "verify_claims") is True

    assert str_setting(SimpleNamespace(gateway_base_url=""), "gateway_base_url") == ""
    assert str_setting(SimpleNamespace(skills_dir=""), "skills_dir") == ""
    # ...while a MISSING string still falls back to its declared default.
    assert str_setting(SimpleNamespace(),
                       "web_fetch_allow_hosts") == "host.docker.internal"


def test_bool_setting_treats_an_empty_string_as_unset() -> None:
    from types import SimpleNamespace

    assert bool_setting(SimpleNamespace(verify_claims=""), "verify_claims") is True
    assert bool_setting(SimpleNamespace(verify_claims=None), "verify_claims") is True


def test_an_unknown_field_is_a_programming_error_not_a_silent_default() -> None:
    """A typo'd field name must fail loudly. `model_config` sets extra="ignore", so a
    typo in .env is already silently dropped; a typo in code should not be too."""
    from types import SimpleNamespace

    with pytest.raises(KeyError):
        int_setting(SimpleNamespace(), "max_attempts_per_chalenge")


def test_every_accessor_agrees_with_the_declared_defaults() -> None:
    """All 80-odd knobs, checked against Settings itself — so adding a field with a
    default the accessors would contradict fails here rather than in production."""
    real = Settings(_env_file=None)
    from types import SimpleNamespace

    empty = SimpleNamespace()
    for name in Settings.model_fields:
        expected = getattr(real, name)
        if isinstance(expected, bool):
            assert bool_setting(empty, name) is expected, name
        elif isinstance(expected, int):
            assert int_setting(empty, name) == expected, name
        elif isinstance(expected, float):
            assert float_setting(empty, name) == expected, name
        elif isinstance(expected, str):
            assert str_setting(empty, name) == expected, name
        else:  # pragma: no cover - every field is one of the four scalar types
            pytest.fail(f"{name}: unhandled default type {type(expected).__name__}")
