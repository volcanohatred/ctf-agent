"""Import every backend module — a cheap smoke test that catches wiring breakage.

Run directly (`python tests/import_check.py`) or from CI. Importing is not the
same as working, but it does catch the failures that are most expensive to
discover late: a renamed symbol, a circular import introduced by a refactor, or
a dependency whose API moved (an unpinned `pydantic-ai` once dropped the
`openai`/`google` extras out from under this project).
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def iter_modules() -> list[str]:
    import backend

    names = ["backend"]
    for info in pkgutil.walk_packages(backend.__path__, prefix="backend."):
        if info.ispkg or not info.name.endswith("__"):
            names.append(info.name)
    return sorted(set(names))


def main() -> int:
    modules = iter_modules()
    failed: list[tuple[str, str]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 — report every failure, then exit non-zero
            failed.append((name, f"{type(e).__name__}: {e}"))

    print(f"imported {len(modules) - len(failed)}/{len(modules)} backend modules")
    for name, err in failed:
        print(f"  FAIL {name}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
