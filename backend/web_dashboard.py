"""Web operations dashboard — live view of swarms/solvers + human guidance.

Runs an aiohttp server alongside the coordinator event loop. Lets the operator
see, per challenge and per solver: current status, last activity, findings —
and inject guidance (advise / bump / broadcast) without leaving the browser.

Security model: every route requires a bearer token. The dashboard can inject
arbitrary instructions into solvers (which execute arbitrary code inside
privileged sandboxes), spawn new containers, and export confirmed flags — so an
unauthenticated listener on 0.0.0.0 was effectively a remote code-execution and
flag-disclosure endpoint for anything that could reach the host, including the
solver sandboxes themselves via host.docker.internal.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import secrets
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aiohttp import web

from backend.briefing import Turn, ask
from backend.message_bus import KIND_CLAIM, KIND_FACT, KIND_VERIFIED
from backend.trace_read import last_model_text, trace_activity

logger = logging.getLogger(__name__)

# Trace tails are re-read on every dashboard poll (default 2.5s) for every
# solver. Caching by (path, size) keeps that off the event loop's critical path
# without ever showing stale data: the size changes as soon as a line is appended.
_TAIL_CACHE: dict[str, tuple[int, float, Any]] = {}
TAIL_CACHE_TTL_S = 1.0
STATE_FEED_LIMIT = 30


def _cache_get(key: str, size: int) -> Any | None:
    hit = _TAIL_CACHE.get(key)
    if not hit:
        return None
    cached_size, cached_at, value = hit
    if cached_size != size or (time.monotonic() - cached_at) > TAIL_CACHE_TTL_S:
        return None
    return value


def _cache_put(key: str, size: int, value: Any) -> None:
    # Bound the cache: one entry per live solver, plus a little slack.
    if len(_TAIL_CACHE) > 512:
        _TAIL_CACHE.clear()
    _TAIL_CACHE[key] = (size, time.monotonic(), value)


def _file_size(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return -1


def _tail_events_sync(trace_path: str, n: int = 40) -> list[dict]:
    """Parse the last N events of a solver trace into compact dicts."""
    try:
        with open(trace_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type", "?")
        rec: dict[str, Any] = {"ts": ev.get("ts", 0), "type": t}
        if t == "tool_call":
            args = ev.get("args", {})
            rec["tool"] = ev.get("tool", "?")
            rec["args"] = ((args.get("command") or json.dumps(args))[:400]
                           if isinstance(args, dict) else str(args)[:400])
        elif t == "tool_result":
            rec["tool"] = ev.get("tool", "?")
            rec["result"] = str(ev.get("result", ""))[:400]
        elif t in ("bump", "error", "finish", "flag_confirmed", "flag_rejected",
                   "model_response", "advice_picked", "session_reset",
                   "session_forgotten", "start", "budget_interrupt",
                   "conclude_ok", "conclude_failed", "still_working",
                   "structured_flag_verified", "loop_break"):
            rec.update({k: str(v)[:300] for k, v in ev.items() if k not in ("ts", "type")})
        else:
            continue
        out.append(rec)
    return out[-n:]


async def _tail_events(trace_path: str, n: int = 40) -> list[dict]:
    """Cached, off-loop variant of `_tail_events_sync`."""
    size = await asyncio.to_thread(_file_size, trace_path)
    if size < 0:
        return []
    key = f"{trace_path}#{n}"
    cached = _cache_get(key, size)
    if cached is not None:
        return cached
    events = await asyncio.to_thread(_tail_events_sync, trace_path, n)
    _cache_put(key, size, events)
    return events


async def _trace_activity_cached(trace_path: str) -> tuple[float, list[str]]:
    size = await asyncio.to_thread(_file_size, trace_path)
    if size < 0:
        return 0.0, []
    key = f"{trace_path}#act"
    cached = _cache_get(key, size)
    if cached is not None:
        return cached
    value = await asyncio.to_thread(trace_activity, trace_path)
    _cache_put(key, size, value)
    return value


async def _last_text_cached(trace_path: str) -> str:
    size = await asyncio.to_thread(_file_size, trace_path)
    if size < 0:
        return ""
    key = f"{trace_path}#txt"
    cached = _cache_get(key, size)
    if cached is not None:
        return cached
    value = await asyncio.to_thread(last_model_text, trace_path)
    _cache_put(key, size, value)
    return value


def _as_int(value: Any) -> int:
    """Normalise a solver step counter.

    The Pydantic backend keeps `_step_count` as a one-element list (a mutable
    reference shared with its toolset) while the Claude backend uses a plain
    int, so the JSON field used to be an array for half the lineup.
    """
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _clip(text: Any, limit: int) -> str:
    s = str(text or "")
    if limit > 0 and len(s) > limit:
        return s[:limit] + f"\n… [truncated for the live view — {len(s)} chars total; " \
                           "use the export endpoint for the full text]"
    return s


class DashboardServer:
    """aiohttp server bound to the coordinator's shared deps."""

    def __init__(self, deps: Any, host: str = "127.0.0.1", port: int = 8080,
                 token: str = "") -> None:
        self.deps = deps
        self.host = host
        self.port = port
        self._runner: web.AppRunner | None = None
        # Never run silently unauthenticated: with no configured token, mint one
        # and print it so the operator can still reach the UI.
        self.token = token or secrets.token_urlsafe(24)
        self._generated_token = not token
        self.feed_chars = int(getattr(getattr(deps, "settings", None),
                                      "dashboard_feed_chars", 4000) or 4000)

    async def start(self) -> None:
        app = web.Application()
        app.add_routes([
            web.get("/", self._index),
            web.get("/api/state", self._state),
            web.get("/api/trace", self._trace),
            web.get("/api/export", self._export),
            web.post("/api/advise", self._advise),
            web.post("/api/bump", self._bump),
            web.post("/api/broadcast", self._broadcast),
            web.post("/api/ask", self._ask),
            web.post("/api/spawn", self._spawn),
            web.get("/healthz", self._healthz),
        ])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("Dashboard listening on http://%s:%d (token required)", self.host, self.port)
        if self._generated_token:
            # Printed once, at INFO, because without it the UI is unreachable.
            logger.info("Dashboard access token (no DASHBOARD_TOKEN was set): %s", self.token)
            logger.info("  → open http://%s:%d/?token=%s", self.host, self.port, self.token)
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "Dashboard is bound to %s. It can inject instructions into solvers "
                "and export flags — put a reverse proxy with auth in front of it.",
                self.host,
            )

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    # ------------------------------------------------------------- auth ----

    def _authorized(self, req: web.Request) -> bool:
        supplied = ""
        header = req.headers.get("Authorization", "")
        if header.lower().startswith("bearer "):
            supplied = header[7:].strip()
        if not supplied:
            supplied = req.query.get("token", "")
        return bool(supplied) and hmac.compare_digest(supplied, self.token)

    def _unauthorized(self) -> web.Response:
        return web.json_response(
            {"error": "unauthorized",
             "hint": "pass ?token=<DASHBOARD_TOKEN> or an Authorization: Bearer header"},
            status=401,
        )

    # ----------------------------------------------------------- handlers ----

    async def _healthz(self, _req: web.Request) -> web.Response:
        # Unauthenticated liveness probe — reveals nothing but "up".
        return web.json_response({"ok": True})

    async def _index(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        from backend.dashboard_ui import render_dashboard

        return web.Response(text=render_dashboard(self.token), content_type="text/html")

    async def _swarm_summary(self, name: str, swarm: Any) -> dict:
        meta = swarm.meta
        agents = {}
        cost_tracker = getattr(self.deps, "cost_tracker", None)
        specs = list(getattr(swarm, "_all_specs", None) or swarm.model_specs)
        # Escalated solvers are not in model_specs, so iterating that alone hid
        # them from the operator entirely.
        for spec in dict.fromkeys([*specs, *swarm.solvers]):
            solver = swarm.solvers.get(spec)
            agent: dict[str, Any] = {
                "status": "not started", "last": "", "last_ts": 0.0,
                "thinking": "", "quiet_s": None, "steps": 0, "tool_calls": 0,
                "cost_usd": 0.0, "approach": "",
            }
            if cost_tracker is not None:
                from backend.models import model_id_from_spec
                ag = cost_tracker.by_agent.get(f"{name}/{model_id_from_spec(spec)}")
                if ag:
                    agent["cost_usd"] = round(ag.cost_usd, 4)
            if solver is None:
                agents[spec] = agent
                continue
            agent["status"] = "running"
            agent["steps"] = _as_int(getattr(solver, "_step_count", 0))
            agent["tool_calls"] = _as_int(getattr(solver, "_tool_calls", 0))
            assigned = getattr(swarm, "_assigned_approach", {}).get(spec, "")
            if assigned:
                agent["approach"] = assigned
            path = getattr(getattr(solver, "tracer", None), "path", None)
            if path:
                last_ts, sigs = await _trace_activity_cached(path)
                agent["last_ts"] = last_ts
                agent["quiet_s"] = round(time.time() - last_ts) if last_ts else None
                if sigs:
                    agent["last"] = sigs[-1][:200]
                agent["thinking"] = await _last_text_cached(path)
            agents[spec] = agent

        # Shared blackboard feed — what solvers posted for siblings (facts,
        # excluded dead-ends, coordinator broadcasts). Content is clipped for
        # the live view: findings are deliberately never truncated at rest, so
        # streaming dozens of them in full every 2.5s was megabytes per poll.
        feed: list[dict] = []
        help_requests: list[dict] = []
        bus = getattr(swarm, "message_bus", None)
        if bus is not None:
            for f in bus.findings[-STATE_FEED_LIMIT:]:
                feed.append({
                    "model": f.model,
                    "kind": f.kind,
                    "content": _clip(f.content, self.feed_chars),
                    "chars": len(f.content or ""),
                    "ts": f.timestamp,
                })
                if f.kind == "help":
                    help_requests.append({
                        "model": f.model,
                        "question": _clip(f.content, self.feed_chars),
                        "ts": f.timestamp,
                    })
        return {
            "challenge": name,
            "category": getattr(meta, "category", ""),
            "value": getattr(meta, "value", 0),
            "cancelled": swarm.cancel_event.is_set(),
            "winner": swarm.winner.flag if (swarm.winner and swarm.winner.flag) else None,
            "winner_spec": swarm.winner_spec,
            "bumps": getattr(swarm, "_total_bumps", 0),
            "failed_approaches": list(getattr(swarm, "_failed_approaches", [])),
            "retire_reason": getattr(swarm, "retire_reason", ""),
            "infra_failures": getattr(swarm, "_infra_failures", 0),
            # Live audit trail: how much of what the board "knows" was ever
            # confirmed by a second solver, and how often the swarm had to be
            # broken out of a direction that was producing nothing new.
            "stale_breaks": getattr(swarm, "_stale_breaks", 0),
            "strategist_pivots": getattr(swarm, "_strategist_pivots", 0),
            "corroborated_facts": sum(
                1 for f in swarm.message_bus.findings if f.kind == KIND_FACT
            ),
            "probe_verified": sum(
                1 for f in swarm.message_bus.findings if f.kind == KIND_VERIFIED
            ),
            "unverified_claims": sum(
                1 for f in swarm.message_bus.findings if f.kind == KIND_CLAIM
            ),
            "probe_refuted": getattr(swarm, "_probe_refuted", 0),
            "contradictions": getattr(swarm, "_contradictions", 0),
            "claims_queued": len(getattr(swarm, "_claim_queue", []) or []),
            "agents": agents,
            "findings": {k: _clip(v, self.feed_chars) for k, v in dict(swarm.findings).items()},
            "shared_feed": feed,
            "help_requests": help_requests,
        }

    async def _state(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        deps = self.deps
        swarms: dict[str, Any] = {}
        for name, s in list(deps.swarms.items()):
            swarms[name] = await self._swarm_summary(name, s)
        # Archived (finished) swarms — keep them visible as completed.
        for name, hist in list(getattr(deps, "swarms_history", {}).items()):
            if name in swarms:
                continue
            hist = dict(hist)
            hist.setdefault("cancelled", True)
            hist.setdefault("archived", True)
            hist["shared_feed"] = [
                {**f, "content": _clip(f.get("content"), self.feed_chars)}
                for f in (hist.get("shared_feed") or [])[-STATE_FEED_LIMIT:]
            ]
            swarms[name] = hist
        # Challenges known locally but with no swarm. Two distinct cases, which
        # used to look identical:
        #   - on CTFd and unsolved → genuinely waiting for the coordinator
        #   - not on CTFd at all   → a local set that was never seeded; showing
        #     it as "待解题" implied the coordinator was neglecting it
        # An EMPTY ctfd_challenges means the poller has not populated it yet,
        # i.e. "unknown" — not "nothing is on CTFd". Treating it as the latter
        # would flag the entire board as local-only on a fresh start.
        ctfd_names = set(getattr(deps, "ctfd_challenges", None) or ())
        ctfd_known = bool(ctfd_names)
        solved_names = set(getattr(deps, "results", {}).keys())
        solved_names |= set(getattr(deps, "ctfd_solved", None) or ())
        for name, meta in list(getattr(deps, "challenge_metas", {}).items()):
            if name in swarms:
                continue
            on_ctfd = (name in ctfd_names) if ctfd_known else True
            # A solved challenge with no live swarm is NOT pending. Marking it so
            # left the badge correct (the UI tests `solved` first) but made the
            # detail panel read "待解题（尚未 spawn）" and sorted it below the
            # running ones.
            is_solved = name in solved_names
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
                "pending": on_ctfd and not is_solved,
                "local_only": not on_ctfd,
                "solved": is_solved,
            }

        # CTFd is the authority on what is solved. `results` only accumulates
        # during the current session, so relying on it alone showed 0 solved
        # after every restart even though the coordinator's own STATUS line —
        # which reads the poller — reported the true count. (`solved_names`
        # above is this same union, computed before the placeholder loop.)
        data = {
            "ts": time.time(),
            "solved": sorted(solved_names),
            "swarms": swarms,
            "cost_usd": round(getattr(deps.cost_tracker, "total_cost_usd", 0.0), 4),
            "results": getattr(deps, "results", {}),
            "ctfd_solved": sorted(getattr(deps, "ctfd_solved", None) or ()),
            "ctfd_challenges": sorted(ctfd_names),
        }
        return web.json_response(data)

    async def _trace(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        name = req.query.get("name", "")
        spec = req.query.get("spec", "")
        try:
            n = max(1, min(int(req.query.get("n", "50")), 500))
        except ValueError:
            n = 50
        swarm = self.deps.swarms.get(name)
        if not swarm:
            return web.json_response({"error": "no such swarm"}, status=404)
        solver = swarm.solvers.get(spec)
        path = getattr(getattr(solver, "tracer", None), "path", None) if solver else None
        if not path:
            return web.json_response({"events": []})
        return web.json_response({"events": await _tail_events(path, n)})

    async def _json_body(self, req: web.Request) -> dict | None:
        try:
            body = await req.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return body if isinstance(body, dict) else None

    async def _advise(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        body = await self._json_body(req)
        if body is None:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        name = body.get("challenge", "")
        spec = body.get("model_spec", "")
        advice = str(body.get("advice", "")).strip()
        if not advice:
            return web.json_response({"error": "advice is empty"}, status=400)
        swarm = self.deps.swarms.get(name)
        if not swarm:
            return web.json_response({"error": "no such swarm"}, status=404)
        await swarm.message_bus.post_advice(spec, f"[operator] {advice}")
        logger.info("Operator advised %s/%s (%d chars)", name, spec, len(advice))
        return web.json_response({"ok": True})

    async def _bump(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        body = await self._json_body(req)
        if body is None:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        name = body.get("challenge", "")
        spec = body.get("model_spec", "")
        insights = str(body.get("insights", "")).strip()
        swarm = self.deps.swarms.get(name)
        solver = swarm.solvers.get(spec) if swarm else None
        if not solver:
            return web.json_response({"error": "no such solver"}, status=404)
        solver.bump(f"[operator guidance] {insights}" if insights
                    else "[operator] retry with a fresh approach")
        logger.info("Operator bumped %s/%s", name, spec)
        return web.json_response({"ok": True})

    async def _broadcast(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        body = await self._json_body(req)
        if body is None:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        name = body.get("challenge", "")
        message = str(body.get("message", "")).strip()
        if not message:
            return web.json_response({"error": "message is empty"}, status=400)
        swarm = self.deps.swarms.get(name)
        if not swarm:
            return web.json_response({"error": "no such swarm"}, status=404)
        await swarm.message_bus.broadcast(f"[operator] {message}")
        logger.info("Operator broadcast to %s (%d chars)", name, len(message))
        return web.json_response({"ok": True})

    async def _ask(self, req: web.Request) -> web.Response:
        """Command-centre chat: answer an operator question about one challenge.

        Works for a LIVE swarm and for an archived one — "what happened on that
        challenge we lost four hours on?" is asked most often after the swarm is
        gone, and the archive is the only record left.

        The dossier is rebuilt on every request rather than cached: the swarm is
        live, and an answer about the current state built on a ten-minute-old
        snapshot is confidently wrong in exactly the way that gets an operator to
        issue a stale instruction.
        """
        if not self._authorized(req):
            return self._unauthorized()
        body = await self._json_body(req)
        if body is None:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        name = str(body.get("challenge", "")).strip()
        question = str(body.get("question", "")).strip()
        if not name:
            return web.json_response({"error": "challenge is required"}, status=400)
        if not question:
            return web.json_response({"error": "question is empty"}, status=400)
        if len(question) > 4000:
            return web.json_response({"error": "question too long (max 4000 chars)"}, status=400)

        swarm = self.deps.swarms.get(name)
        archived = (getattr(self.deps, "swarms_history", {}) or {}).get(name)
        if swarm is None and not archived:
            return web.json_response(
                {"error": f"no live or archived swarm for '{name}'"}, status=404)

        history = [
            Turn(role=str(t.get("role", "")), content=str(t.get("content", "")))
            for t in (body.get("history") or [])
            if isinstance(t, dict) and str(t.get("role", "")) in ("user", "assistant")
        ][-8:]

        try:
            # Trace reads are blocking file I/O over potentially large JSONL; keep
            # them off the event loop so the poller and patrol keep running.
            summary = (await self._swarm_summary(name, swarm)) if swarm is not None else None
            dossier, trimmed = await asyncio.to_thread(
                self._build_dossier, name, summary, archived, swarm)
            result = await ask(
                question=question,
                dossier=dossier,
                history=history,
                settings=self.deps.settings,
                cost_tracker=self.deps.cost_tracker,
                agent_name=f"{name}/briefing",
            )
        except Exception as e:  # noqa: BLE001 — the operator must see why, not a 500
            logger.warning("briefing for %s failed: %s", name, e)
            return web.json_response({"error": f"briefing failed: {type(e).__name__}: {e}"},
                                     status=502)

        logger.info("Operator asked %s (%d chars) → %s in %.1fs",
                    name, len(question), result.model, result.seconds)
        return web.json_response({
            "ok": result.ok,
            "answer": result.answer,
            "error": result.error,
            "model": result.model,
            "seconds": round(result.seconds, 1),
            "dossier_chars": result.dossier_chars,
            "trimmed": trimmed,
            "challenge": name,
        })

    def _build_dossier(self, name: str, summary: dict | None,
                       archived: dict | None, swarm: Any = None) -> tuple[str, list[str]]:
        """Assemble the briefing dossier. Synchronous — runs in a worker thread.

        Reuses `_swarm_summary` for the counters and blackboard instead of
        re-reading the swarm here: that method is already defensive about missing
        attributes, so a second reader would be a second place for the two views to
        drift. The live trace detail is added on top, because "is this solver
        spinning with zero tool calls?" changes the answer to "what is the progress"
        more than any counter does.
        """
        from backend.briefing import build_dossier, format_solver_notes

        if summary is not None:
            meta = (getattr(self.deps, "challenge_metas", {}) or {}).get(name)
            if meta is None:
                meta = SimpleNamespace(
                    category=summary.get("category", ""), value=summary.get("value", 0),
                    tags=[], description="", connection_info="",
                )
            notes = format_solver_notes(swarm) if swarm is not None else []
            if not notes:
                notes = [
                    f"### {spec}\n- 状态：{a.get('status', '?')}　工具调用：{a.get('tool_calls', 0)}"
                    for spec, a in (summary.get("agents", {}) or {}).items()
                ]
            return build_dossier(
                name=name, status=summary, meta=meta,
                feed=summary.get("shared_feed", []) or [],
                failed_approaches=summary.get("failed_approaches", []) or [],
                solver_notes=notes,
            )

        archived = archived or {}
        status = {
            "winner": archived.get("winner"),
            "winner_spec": archived.get("winner_spec", ""),
            "bumps": archived.get("bumps", 0),
            "cancelled": True,
            "retire_reason": archived.get("retire_reason", ""),
            "infra_failures": archived.get("infra_failures", 0),
            "probe_verified": archived.get("probe_verified", 0),
            "corroborated_facts": archived.get("corroborated_facts", 0),
            "unverified_claims": archived.get("unverified_claims", 0),
            "probe_refuted": archived.get("probe_refuted", 0),
            "contradictions": archived.get("contradictions", 0),
            "stale_breaks": archived.get("stale_breaks", 0),
            "strategist_pivots": archived.get("strategist_pivots", 0),
            "strategist_failures": archived.get("strategist_failures", 0),
        }
        meta = SimpleNamespace(category=archived.get("category", ""),
                               value=archived.get("value", 0),
                               tags=archived.get("tags", []),
                               description=archived.get("description", ""),
                               connection_info="")
        agents = archived.get("agents", {}) or {}
        notes = [
            f"### {spec}\n- 状态：{a.get('status', '?')}　工具调用：{a.get('tool_calls', 0)}\n"
            f"- 它的发现：\n{str(a.get('findings', ''))[:2000]}"
            for spec, a in agents.items()
        ]
        return build_dossier(
            name=name, status=status, meta=meta,
            feed=archived.get("shared_feed", []) or [],
            failed_approaches=archived.get("failed_approaches", []) or [],
            solver_notes=notes,
        )

    async def _spawn(self, req: web.Request) -> web.Response:
        if not self._authorized(req):
            return self._unauthorized()
        body = await self._json_body(req)
        if body is None:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        name = str(body.get("challenge", "")).strip()
        if not name:
            return web.json_response({"error": "challenge is required"}, status=400)
        from backend.agents.coordinator_core import do_spawn_swarm
        result = await do_spawn_swarm(self.deps, name)
        logger.info("Operator spawned swarm for %s: %s", name, result[:120])
        return web.json_response({"ok": True, "result": result})

    async def _export(self, req: web.Request) -> web.Response:
        """Download a full challenge report (JSON or Markdown)."""
        if not self._authorized(req):
            return self._unauthorized()
        from backend.export import build_report, render_json, render_markdown

        name = req.query.get("name", "")
        fmt = req.query.get("fmt", "md")
        swarm = self.deps.swarms.get(name)
        archived = getattr(self.deps, "swarms_history", {}).get(name)
        if swarm is None and archived is None:
            return web.json_response({"error": "no such challenge"}, status=404)
        report = await asyncio.to_thread(build_report, name, swarm, archived)
        if fmt == "json":
            body, ctype, ext = render_json(report), "application/json", "json"
        else:
            body, ctype, ext = render_markdown(report), "text/markdown", "md"
        # ASCII-only filename: challenge names come from CTFd and routinely
        # contain non-ASCII text, quotes and slashes. `str.isalnum()` is True for
        # 'ä' and for CJK, so it must be paired with isascii() or the value ends
        # up in a header that cannot represent it.
        safe = "".join(
            c if (c.isascii() and (c.isalnum() or c in "-_")) else "_" for c in name
        ).strip("_")[:80] or "challenge"
        fname = f"ctf-report-{safe}.{ext}"
        return web.Response(
            text=body, content_type=ctype, charset="utf-8",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )
