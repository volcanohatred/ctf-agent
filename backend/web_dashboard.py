"""Web operations dashboard — live view of swarms/solvers + human guidance.

Runs an aiohttp server alongside the coordinator event loop. Lets the operator
see, per challenge and per solver: current status, last activity, findings —
and inject guidance (advise / bump / broadcast) without leaving the browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from aiohttp import web

from backend.trace_read import last_model_text, trace_activity

logger = logging.getLogger(__name__)


def _tail_events(trace_path: str, n: int = 40) -> list[dict]:
    """Parse the last N events of a solver trace into compact dicts."""
    try:
        with open(trace_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
        out = []
        for line in tail.strip().splitlines():
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get("type", "?")
            rec: dict[str, Any] = {"ts": ev.get("ts", 0), "type": t}
            if t == "tool_call":
                args = ev.get("args", {})
                rec["tool"] = ev.get("tool", "?")
                rec["args"] = (args.get("command") or json.dumps(args))[:400] if isinstance(args, dict) else str(args)[:400]
            elif t == "tool_result":
                rec["tool"] = ev.get("tool", "?")
                rec["result"] = str(ev.get("result", ""))[:400]
            elif t in ("bump", "error", "finish", "flag_confirmed", "model_response", "advice_picked", "session_reset", "start"):
                rec.update({k: str(v)[:300] for k, v in ev.items() if k not in ("ts", "type")})
            else:
                continue
            out.append(rec)
        return out[-n:]
    except Exception:
        return []


class DashboardServer:
    """aiohttp server bound to the coordinator's shared deps."""

    def __init__(self, deps: Any, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.deps = deps
        self.host = host
        self.port = port
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.add_routes([
            web.get("/", self._index),
            web.get("/api/state", self._state),
            web.get("/api/trace", self._trace),
            web.post("/api/advise", self._advise),
            web.post("/api/bump", self._bump),
            web.post("/api/broadcast", self._broadcast),
            web.post("/api/spawn", self._spawn),
        ])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Dashboard listening on http://{self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    # ----------------------------------------------------------- handlers ----

    async def _index(self, _req: web.Request) -> web.Response:
        from backend.dashboard_ui import DASHBOARD_HTML
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    def _swarm_summary(self, name: str, swarm: Any) -> dict:
        meta = swarm.meta
        agents = {}
        cost_tracker = getattr(self.deps, "cost_tracker", None)
        for spec in swarm.model_specs:
            solver = swarm.solvers.get(spec)
            agent: dict[str, Any] = {
                "status": "not started", "last": "", "last_ts": 0.0,
                "thinking": "", "quiet_s": None, "steps": 0, "cost_usd": 0.0,
                "approach": "",
            }
            # per-solver cost from the shared tracker (agent_name = "<ch>/<model>")
            if cost_tracker is not None:
                from backend.models import model_id_from_spec
                ag = cost_tracker.by_agent.get(f"{name}/{model_id_from_spec(spec)}")
                if ag:
                    agent["cost_usd"] = round(ag.cost_usd, 4)
                    agent["steps"] = 0  # filled below if available
            if solver is None:
                agents[spec] = agent
                continue
            agent["status"] = "running"
            agent["steps"] = getattr(solver, "_step_count", 0)
            # rotation assignment (which technique it's been told to try)
            assigned = getattr(swarm, "_assigned_approach", {}).get(spec, "")
            if assigned:
                agent["approach"] = assigned
            tracer = getattr(solver, "tracer", None)
            path = getattr(tracer, "path", None)
            if path and Path(path).exists():
                last_ts, sigs = trace_activity(path)
                agent["last_ts"] = last_ts
                agent["quiet_s"] = round(time.time() - last_ts) if last_ts else None
                if sigs:
                    agent["last"] = sigs[-1][:200]
                thinking = last_model_text(path)
                if thinking:
                    agent["thinking"] = thinking
            agents[spec] = agent
        # Shared blackboard feed — what solvers posted for siblings (facts,
        # excluded dead-ends, coordinator broadcasts). This is the
        # "shared experience" across models on this challenge.
        feed: list[dict] = []
        try:
            bus = getattr(swarm, "message_bus", None)
            if bus is not None:
                for f in bus.findings[-30:]:
                    feed.append({
                        "model": f.model,
                        "kind": f.kind,
                        "content": f.content,  # full text — the dashboard shows it whole
                        "ts": f.timestamp,
                    })
        except Exception:
            pass
        return {
            "challenge": name,
            "category": getattr(meta, "category", ""),
            "value": getattr(meta, "value", 0),
            "cancelled": swarm.cancel_event.is_set(),
            "winner": swarm.winner.flag if swarm.winner else None,
            "winner_spec": swarm.winner_spec,
            "bumps": getattr(swarm, "_total_bumps", 0),
            "failed_approaches": getattr(swarm, "_failed_approaches", []),
            "agents": agents,
            "findings": dict(swarm.findings),
            "shared_feed": feed,
        }

    async def _state(self, _req: web.Request) -> web.Response:
        deps = self.deps
        swarms = {name: self._swarm_summary(name, s) for name, s in deps.swarms.items()}
        # Archived (finished) swarms — keep them visible as completed.
        history = getattr(deps, "swarms_history", {})
        for name, hist in history.items():
            if name not in swarms:
                hist = dict(hist)
                hist.setdefault("cancelled", True)
                hist.setdefault("archived", True)
                swarms[name] = hist
        # Challenges known to CTFd but not yet spawned → shown as "待解题".
        for name, meta in getattr(deps, "challenge_metas", {}).items():
            if name not in swarms:
                swarms[name] = {
                    "challenge": name,
                    "category": getattr(meta, "category", ""),
                    "value": getattr(meta, "value", 0),
                    "cancelled": False,
                    "winner": None,
                    "winner_spec": "",
                    "bumps": 0,
                    "failed_approaches": [],
                    "agents": {},
                    "findings": {},
                    "pending": True,
                }
        data = {
            "ts": time.time(),
            "solved": list(getattr(deps, "results", {}).keys()),
            "swarms": swarms,
            "cost_usd": round(getattr(deps.cost_tracker, "total_cost_usd", 0.0), 4),
            "results": getattr(deps, "results", {}),
        }
        return web.json_response(data)

    async def _trace(self, req: web.Request) -> web.Response:
        name = req.query.get("name", "")
        spec = req.query.get("spec", "")
        n = int(req.query.get("n", "50"))
        swarm = self.deps.swarms.get(name)
        if not swarm:
            return web.json_response({"error": "no such swarm"}, status=404)
        solver = swarm.solvers.get(spec)
        tracer = getattr(solver, "tracer", None) if solver else None
        path = getattr(tracer, "path", None)
        if not path or not Path(path).exists():
            return web.json_response({"events": []})
        return web.json_response({"events": _tail_events(path, n)})

    async def _advise(self, req: web.Request) -> web.Response:
        body = await req.json()
        name, spec, advice = body.get("challenge", ""), body.get("model_spec", ""), body.get("advice", "")
        swarm = self.deps.swarms.get(name)
        if not swarm:
            return web.json_response({"error": "no such swarm"}, status=404)
        await swarm.message_bus.post_advice(spec, f"[operator] {advice}")
        return web.json_response({"ok": True})

    async def _bump(self, req: web.Request) -> web.Response:
        body = await req.json()
        name, spec = body.get("challenge", ""), body.get("model_spec", "")
        insights = body.get("insights", "")
        swarm = self.deps.swarms.get(name)
        solver = swarm.solvers.get(spec) if swarm else None
        if not solver:
            return web.json_response({"error": "no such solver"}, status=404)
        solver.bump(f"[operator guidance] {insights}" if insights else "[operator] retry with a fresh approach")
        return web.json_response({"ok": True})

    async def _broadcast(self, req: web.Request) -> web.Response:
        body = await req.json()
        name, message = body.get("challenge", ""), body.get("message", "")
        swarm = self.deps.swarms.get(name)
        if not swarm:
            return web.json_response({"error": "no such swarm"}, status=404)
        await swarm.message_bus.broadcast(f"[operator] {message}")
        return web.json_response({"ok": True})

    async def _spawn(self, req: web.Request) -> web.Response:
        body = await req.json()
        name = body.get("challenge", "")
        from backend.agents.coordinator_core import do_spawn_swarm
        result = await do_spawn_swarm(self.deps, name)
        return web.json_response({"ok": True, "result": result})
