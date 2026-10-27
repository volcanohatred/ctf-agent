"""No function parameter may be accepted and then ignored.

ruff's ARG002/ARG005 are in this repo's ignore list because SDK hook callbacks
must accept `(input_data, tool_use_id, context)` whether or not they use all
three. That exemption is correct for hooks — but it also hid a real defect:
`_build_restart_payload` accepted `reason` and never used it, so a solver
restarted after a wall-clock interrupt was never told it had been interrupted and
promptly did the same thing again.

This test does the sweep ruff cannot, with the legitimate cases filtered:
  - hook/protocol parameter names imposed by an external signature
  - Protocol and abstract declarations (docstring + `...` is a declaration)
  - `_`-prefixed parameters, the conventional "intentionally unused" marker
  - parameters explicitly consumed with `del x`, which is how a tool marks a
    framework-required-but-unused argument without weakening its type contract

A new finding means either a real bug or a parameter that needs one of the
markers above to show the omission is deliberate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent / "backend"

# Names imposed by an external calling convention, not chosen by us.
FRAMEWORK_PARAMS = {
    "self", "cls",
    "input_data", "tool_use_id", "context",   # Claude Agent SDK hooks
    "call_id",
}


def _is_declaration_only(node: ast.AST) -> bool:
    """True for Protocol methods and abstract stubs.

    A body of just `...`, just a docstring, or docstring-then-`...` declares an
    interface; its parameter names exist for documentation and must be kept even
    though nothing reads them.
    """
    body = list(getattr(node, "body", []))
    meaningful = []
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            value = stmt.value.value
            # `...` parses as Constant(value=Ellipsis); ast.Ellipsis was removed
            # in Python 3.12, so it must not be referenced here.
            if isinstance(value, str) or value is Ellipsis or value is None:
                continue
        if isinstance(stmt, ast.Pass):
            continue
        meaningful.append(stmt)
    return not meaningful


def _consumed_names(node: ast.AST) -> set[str]:
    """Names the body actually uses, including `del x` as an explicit ack."""
    used: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            used.add(sub.id)
        elif isinstance(sub, ast.Delete):
            for target in sub.targets:
                if isinstance(target, ast.Name):
                    used.add(target.id)
    return used


def _unused_params() -> list[str]:
    findings: list[str] = []
    for path in sorted(BACKEND.rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        rel = path.relative_to(BACKEND.parent).as_posix()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if _is_declaration_only(node):
                continue

            args = node.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if args.vararg:
                params.append(args.vararg)
            if args.kwarg:
                params.append(args.kwarg)

            used = _consumed_names(node)
            for arg in params:
                name = arg.arg
                if name in FRAMEWORK_PARAMS or name.startswith("_"):
                    continue
                if name in used:
                    continue
                findings.append(f"{rel}:{node.lineno} {node.name}() -> '{name}'")
    return findings


def test_no_silently_ignored_parameters() -> None:
    unused = _unused_params()
    assert not unused, (
        "these parameters are accepted but never read:\n  "
        + "\n  ".join(unused)
        + "\nEither use them, drop them, prefix with `_`, or `del` them to mark "
          "the omission as deliberate."
    )


def test_sweep_actually_detects_a_violation(tmp_path) -> None:
    """Guard the guard: a sweep that can no longer see anything is worthless."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def takes_reason(reason: str, bump_count: int) -> str:\n"
        "    return f'round {bump_count}'\n",
        encoding="utf-8",
    )
    tree = ast.parse(bad.read_text(encoding="utf-8"))
    node = tree.body[0]
    assert not _is_declaration_only(node)
    used = _consumed_names(node)
    assert "bump_count" in used
    assert "reason" not in used, "the sweep can no longer see an unused parameter"


def test_del_counts_as_an_explicit_acknowledgement(tmp_path) -> None:
    """How the webhook tools keep a framework-required RunContext they do not use,
    without weakening the type contract that ty enforces."""
    src = tmp_path / "ok.py"
    src.write_text(
        "async def webhook_create(ctx):\n"
        "    del ctx\n"
        "    return 'x'\n",
        encoding="utf-8",
    )
    node = ast.parse(src.read_text(encoding="utf-8")).body[0]
    assert "ctx" in _consumed_names(node)


def test_protocol_declarations_are_exempt(tmp_path) -> None:
    src = tmp_path / "proto.py"
    src.write_text(
        "class P:\n"
        "    async def conclude(self, budget_s: int) -> str:\n"
        "        '''Docstring.'''\n"
        "        ...\n",
        encoding="utf-8",
    )
    tree = ast.parse(src.read_text(encoding="utf-8"))
    method = tree.body[0].body[0]
    assert _is_declaration_only(method), "a docstring + `...` stub must be exempt"


def test_known_real_case_is_covered() -> None:
    """The defect that motivated this test must stay fixed."""
    swarm = (BACKEND / "agents" / "swarm.py").read_text(encoding="utf-8")
    tree = ast.parse(swarm)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_build_restart_payload":
            names = [a.arg for a in node.args.args]
            assert "reason" in names
            assert "reason" in _consumed_names(node), (
                "_build_restart_payload accepts `reason` without using it again — "
                "a restarted solver would not be told why its round ended"
            )
            return
    pytest.fail("_build_restart_payload not found")
