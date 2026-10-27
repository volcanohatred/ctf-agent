"""Shared pytest fixtures.

The important one is `_isolate_tracer_logs`: `SolverTracer` defaults to
`log_dir="logs"`, so any test that builds a solver writes a real JSONL file into
the repository's log directory. That silently accumulated ~200 junk traces next
to the user's actual run history. Redirecting the default for the whole session
fixes it once, including for tests added later.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend import tracing


@pytest.fixture(autouse=True)
def _isolate_tracer_logs(monkeypatch, tmp_path_factory):
    """Send every SolverTracer created during a test into a throwaway directory."""
    log_dir = str(tmp_path_factory.mktemp("tracer-logs"))
    real_init = tracing.SolverTracer.__init__

    def patched_init(self, challenge_name, model_id, log_dir=log_dir):  # noqa: ARG001
        real_init(self, challenge_name, model_id, log_dir)

    monkeypatch.setattr(tracing.SolverTracer, "__init__", patched_init)
    yield


@pytest.fixture
def scratch_dir():
    """A temp directory that is removed even if the test leaves files behind."""
    path = Path(tempfile.mkdtemp(prefix="ctf-test-"))
    try:
        yield path
    finally:
        import shutil

        shutil.rmtree(path, ignore_errors=True)
