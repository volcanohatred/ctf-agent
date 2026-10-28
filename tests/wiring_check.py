"""Ad-hoc verification of the wiring that unit tests do not cover directly."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"{'ok  ' if cond else 'FAIL'} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


# --- dashboard token injection -------------------------------------------------
from backend.dashboard_ui import TOKEN_PLACEHOLDER, render_dashboard

html = render_dashboard("abc123")
check("placeholder replaced", TOKEN_PLACEHOLDER not in html)
check("token injected as a JSON string", 'const TOKEN = "abc123"' in html)
check("jsq() helper present", "function jsq(" in html)
check("esc() escapes quotes", "&#39;" in html and "&quot;" in html)
check("export links carry the token", "withToken('/api/export" in html)
# The pending-challenge panel always told the operator to "点下方手动 spawn", but
# no such button existed — /api/spawn was server-side only. Lock the pairing down
# so a promised control cannot go missing again.
check("spawn button exists in the UI", "spawnSwarm()" in html)
check("spawn handler posts to /api/spawn", "'/api/spawn'" in html)
check("no unquoted onclick interpolation", "sel('${esc(name)}')" not in html)
check("tool_calls column present", "工具调用" in html)
check("retire_reason surfaced", "retire_reason" in html)

# a token containing a quote must not break out of the literal
quoted = render_dashboard('a"b\\c')
check("hostile token stays a literal", 'const TOKEN = "a\\"b\\\\c"' in quoted, quoted[:120])

# --- solver backend contracts --------------------------------------------------
from backend.agents.claude_solver import ClaudeSolver
from backend.agents.solver import Solver

for cls in (ClaudeSolver, Solver):
    name = cls.__name__
    check(f"{name}.conclude exists", callable(getattr(cls, "conclude", None)))
    check(f"{name}.bump exists", callable(getattr(cls, "bump", None)))
    check(f"{name}.forget_session exists", callable(getattr(cls, "forget_session", None)))
    sig = inspect.signature(cls.run_until_done_or_gave_up)
    check(f"{name}.run accepts budget_s", "budget_s" in sig.parameters, str(sig))

check("Solver.forget_session aliases reset_history",
      Solver.forget_session is Solver.reset_history)

# findings accumulate rather than being overwritten by the last block
from collections import deque

cs = ClaudeSolver.__new__(ClaudeSolver)
cs._finding_blocks = deque(maxlen=24)
cs._add_finding("first confirmed fact")
cs._add_finding("second confirmed fact")
check("ClaudeSolver accumulates findings",
      "first confirmed fact" in cs.findings and "second confirmed fact" in cs.findings)

# bump is a FIFO queue, not a single slot
import tempfile as _tempfile

from backend.loop_detect import LoopDetector
from backend.tracing import SolverTracer

# SolverTracer defaults to log_dir="logs" — pointing it at the repo would drop a
# junk trace next to the user's real run history.
_TRACER_DIR = _tempfile.mkdtemp(prefix="ctf-wiring-tracer-")

cs._bump_queue = deque(maxlen=8)
cs.loop_detector = LoopDetector()
cs.tracer = SolverTracer("wiring-check", "fake", log_dir=_TRACER_DIR)
cs.agent_name = "wiring-check/fake"
cs._session_id = None
cs.bump("coordinator guidance")
cs.bump("swarm rotation payload")
check("both bumps retained", cs.pending_bumps == 2, str(cs.pending_bumps))
prompt = cs._next_prompt()
check("first bump is consumed first", "coordinator guidance" in prompt)
check("second bump still queued", cs.pending_bumps == 1)
cs.tracer.close()

# --- swarm reset dispatch ------------------------------------------------------
from backend.agents.swarm import ChallengeSwarm

src = inspect.getsource(ChallengeSwarm._reset_solver_session)
check("swarm tries forget_session first", src.index("forget_session") < src.index("reset_history"))

# --- sandbox command wrapping --------------------------------------------------
from backend.sandbox import DockerSandbox

sb = DockerSandbox(image="x", challenge_dir=".", command_timeout_s=900)
sb._container_id = "abc"
wrapped = ClaudeSolver._wrap_for_sandbox(cs_with_container := type("S", (), {
    "settings": type("C", (), {"sandbox_command_timeout_s": 900})(),
    "_container_id": "abc123",
})(), "sleep 10000")
check("sandbox commands are timeout-wrapped", "timeout -k 5 900" in wrapped, wrapped)
check("wrapped for docker exec", wrapped.startswith("docker exec -i abc123"), wrapped)

unwrapped = ClaudeSolver._wrap_for_sandbox(type("S", (), {
    "settings": type("C", (), {"sandbox_command_timeout_s": 0})(),
    "_container_id": "abc123",
})(), "ls")
check("timeout wrapper is optional", "timeout" not in unwrapped, unwrapped)

check("DockerSandbox.is_running property exists",
      isinstance(getattr(DockerSandbox, "is_running", None), property))
check("DockerSandbox has cpu_limit", "cpu_limit" in {f.name for f in sb.__dataclass_fields__.values()})

# --- health check wiring -------------------------------------------------------
from backend.cli import _preflight
from backend.config import Settings
from backend.health import ensure_healthy

settings = Settings()
check("health check on by default", settings.startup_health_check is True)
check("msg_port default matches ctf-msg", settings.msg_port == 9400, str(settings.msg_port))
check("dashboard defaults to loopback", settings.dashboard_host == "127.0.0.1")
check("round cap is non-zero by default", settings.max_attempts_per_challenge > 0)
check("conclude budget configured", settings.conclude_timeout_s > 0)
check("escalate_after_bumps present", hasattr(settings, "escalate_after_bumps"))
check("_preflight calls ensure_healthy", "ensure_healthy" in inspect.getsource(_preflight))
check("ensure_healthy is awaitable", inspect.iscoroutinefunction(ensure_healthy))

# --- coordinator turn timeout + cost accounting --------------------------------
from backend.agents import coordinator_loop as cl

loop_src = inspect.getsource(cl.run_event_loop)
check("coordinator turn is time-bounded", "wait_for(turn_fn(msg)" in loop_src)
check("coordinator cost is accounted", "record_cost_only" in loop_src)
check("swarms are reaped, not popped", "reap_finished_swarms" in loop_src)

from backend.agents import coordinator_core as cc

reap_src = inspect.getsource(cc.reap_finished_swarms)
check("reap only retires done tasks", "if not task.done()" in reap_src)
check("reap reads task.exception()", "task.exception()" in reap_src)

# --- patrol detects zero-tool-call spinning ------------------------------------
from backend.trace_read import TraceHealth

check("spinning detected with no tool calls",
      TraceHealth(turns=3, tool_calls=0).spinning is True)
check("healthy solver is not spinning",
      TraceHealth(turns=3, tool_calls=5).spinning is False)
check("looping needs variety loss",
      TraceHealth(tool_sigs=["a"] * 6).looping is True
      and TraceHealth(tool_sigs=["a", "b", "c", "d", "e", "f"]).looping is False)

patrol_src = inspect.getsource(cl._patrol_scan)
check("patrol uses the spinning signal", "health.spinning" in patrol_src)
check("patrol runs off the event loop", "asyncio.to_thread(_patrol_scan" in inspect.getsource(cl._patrol_loop))
check("patrol forgets dead solvers", "intersection_update" in patrol_src)

# --- dashboard auth -------------------------------------------------------------
from backend.web_dashboard import DashboardServer

srv = DashboardServer(deps=type("D", (), {"settings": Settings()})())
check("token auto-generated when unset", bool(srv.token) and len(srv.token) >= 20)
check("explicit token honoured",
      DashboardServer(deps=srv.deps, token="fixed").token == "fixed")

req = type("R", (), {"headers": {"Authorization": "Bearer fixed"}, "query": {}})()
srv2 = DashboardServer(deps=srv.deps, token="fixed")
check("bearer header authorizes", srv2._authorized(req) is True)
bad = type("R", (), {"headers": {}, "query": {"token": "wrong"}})()
check("wrong token rejected", srv2._authorized(bad) is False)
q = type("R", (), {"headers": {}, "query": {"token": "fixed"}})()
check("query token authorizes (needed for <a href>)", srv2._authorized(q) is True)

index_routes = inspect.getsource(srv2.start)
for route in ("/api/state", "/api/trace", "/api/export", "/api/advise", "/api/bump",
              "/api/broadcast", "/api/ask", "/api/spawn"):
    handler = route.split("/")[-1]
    body = inspect.getsource(getattr(type(srv2), f"_{handler}"))
    check(f"{route} is authenticated", "_authorized" in body)

# The briefing endpoint hands an LLM the whole challenge dossier and returns text
# that an operator will act on. Two things must not silently regress: the route has
# to be registered at all, and it must be a POST (a GET would put the question in the
# URL, landing it in proxies and access logs).
check("/api/ask is registered", '"/api/ask"' in index_routes)
check("/api/ask is POST", 'web.post("/api/ask"' in index_routes)

# --- output schema allows an honest "not yet" ----------------------------------
from backend.output_types import FlagFound, StillWorking, solver_output_json_schema

schema = solver_output_json_schema()
check("schema allows still_working", "still_working" in schema["properties"]["type"]["enum"])
check("only type is required", schema["required"] == ["type"])
check("FlagFound still validates", FlagFound(flag="f{x}", method="m").type == "flag_found")
check("StillWorking defaults are empty",
      StillWorking().confirmed == "" and StillWorking().next_step == "")

# --- skills hint renders real paths --------------------------------------------
import tempfile

from backend.skills import SkillRouter

with tempfile.TemporaryDirectory() as td:
    for pack in ("ctf-pwn", "ctf-web"):
        d = Path(td) / pack
        d.mkdir()
        (d / "INDEX.md").write_text(f"# {pack}\n", encoding="utf-8")
        (d / "a.md").write_text("# A\n", encoding="utf-8")
    r = SkillRouter(skills_dir=td)
    hint = r.render_hint(["ctf-pwn", "ctf-web"], level=1)
    check("every pack index path is complete",
          "/skills/ctf-pwn/INDEX.md" in hint and "/skills/ctf-web/INDEX.md" in hint, hint)
    check("no malformed joined path", "/skills/ctf-pwn/, " not in hint, hint)

# module-level cache is shared across routers
from backend import skills as skills_mod

with tempfile.TemporaryDirectory() as td:
    d = Path(td) / "ctf-pwn"
    d.mkdir()
    (d / "INDEX.md").write_text("# pwn\n", encoding="utf-8")
    skills_mod.clear_caches()
    a = SkillRouter(skills_dir=td)
    a._index_excerpt("ctf-pwn")
    b = SkillRouter(skills_dir=td)
    b._index_excerpt("ctf-pwn")
    check("index cache is process-wide", len(skills_mod._INDEX_CACHE) == 1,
          str(len(skills_mod._INDEX_CACHE)))

# --- tracer hygiene -------------------------------------------------------------
import atexit

from backend.tracing import SolverTracer, prune_old_traces

before = len(atexit._exithandlers) if hasattr(atexit, "_exithandlers") else 0
with tempfile.TemporaryDirectory() as td:
    tracers = [SolverTracer("c", "m", log_dir=td) for _ in range(25)]
    paths = {t.path for t in tracers}
    check("trace filenames are unique within a second", len(paths) == 25, str(len(paths)))
    for t in tracers:
        t.close()
    check("close is idempotent", tracers[0].closed is True)
    tracers[0].event("after_close")  # must not raise
    check("write after close is a no-op", True)

# Pruning gets its own directory so the count above does not skew it.
with tempfile.TemporaryDirectory() as td2:
    made = [SolverTracer("c", "m", log_dir=td2) for _ in range(6)]
    for t in made:
        t.close()
    removed = prune_old_traces(td2, keep=3)
    left = len(list(Path(td2).glob("trace-*.jsonl")))
    check("prune removes all but the newest N", removed == 3, str(removed))
    check("prune leaves exactly N files", left == 3, str(left))
    check("prune is a no-op under the keep count",
          prune_old_traces(td2, keep=50) == 0)

# bench-*.json reports must never be swept by trace pruning
with tempfile.TemporaryDirectory() as td3:
    for i in range(4):
        (Path(td3) / f"trace-c-m-2020010{i}-000000.jsonl").write_text("{}", encoding="utf-8")
    (Path(td3) / "bench-20200101-000000.json").write_text("{}", encoding="utf-8")
    prune_old_traces(td3, keep=1)
    check("benchmark reports survive pruning",
          (Path(td3) / "bench-20200101-000000.json").exists())

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("all wiring checks passed")
