"""Per-challenge report export — everything we know about how a challenge went.

Aggregates metadata, outcome, per-model attempts/findings/cost, the shared
blackboard feed (facts + excluded dead-ends + broadcasts), the failed-approach
ledger, and trace highlights. Rendered as JSON (machine) or Markdown (human /
writeup-friendly).
"""

from __future__ import annotations

import json
import time
from typing import Any


def collect_challenge_data(
    name: str,
    swarm: Any | None,
    archived: dict | None,
    deps: Any,
) -> dict[str, Any]:
    """Normalize a challenge's data from a LIVE swarm or an archived summary."""
    if swarm is not None:
        meta = swarm.meta
        agents = {}
        for spec in swarm.model_specs:
            solver = swarm.solvers.get(spec)
            tracer = getattr(solver, "tracer", None)
            path = getattr(tracer, "path", None)
            agents[spec] = {
                "status": "running" if solver else "not started",
                "steps": getattr(solver, "_step_count", 0) if solver else 0,
                "findings": swarm.findings.get(spec, ""),
                "thinking": "",
                "trace_path": path,
            }
            if solver is not None and path:
                try:
                    from backend.trace_read import last_model_text
                    agents[spec]["thinking"] = last_model_text(path)
                except Exception:
                    pass
        feed = []
        bus = getattr(swarm, "message_bus", None)
        if bus is not None:
            for f in bus.findings:
                feed.append({"model": f.model, "kind": f.kind, "content": f.content, "ts": f.timestamp})
        return {
            "name": name,
            "category": getattr(meta, "category", ""),
            "value": getattr(meta, "value", 0),
            "description": getattr(meta, "description", ""),
            "tags": getattr(meta, "tags", []),
            "solved": bool(swarm.winner),
            "winner": swarm.winner.flag if swarm.winner else None,
            "winner_spec": swarm.winner_spec,
            "bumps": getattr(swarm, "_total_bumps", 0),
            "failed_approaches": list(getattr(swarm, "_failed_approaches", [])),
            "agents": agents,
            "shared_feed": feed,
            "archived": False,
        }

    # Archived path — everything comes from the summary dict
    archived = dict(archived or {})
    return {
        "name": name,
        "category": archived.get("category", ""),
        "value": archived.get("value", 0),
        "description": archived.get("description", ""),
        "tags": archived.get("tags", []),
        "solved": bool(archived.get("winner")),
        "winner": archived.get("winner"),
        "winner_spec": archived.get("winner_spec", ""),
        "bumps": archived.get("bumps", 0),
        "failed_approaches": archived.get("failed_approaches", []),
        "agents": archived.get("agents", {}),
        "shared_feed": archived.get("shared_feed", []),
        "archived": True,
        "finished_at": archived.get("finished_at"),
    }


def build_report(name: str, swarm: Any | None, archived: dict | None, deps: Any) -> dict[str, Any]:
    """Build the complete exportable report for one challenge."""
    data = collect_challenge_data(name, swarm, archived, deps)
    data["exported_at"] = time.time()
    data["exported_at_human"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return data


def render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a readable Markdown document (for review/writeup)."""
    lines: list[str] = []
    add = lines.append
    add(f"# CTF 解题复盘报告：{report['name']}")
    add("")
    add(f"- **类别**: {report.get('category', '?')}　**分值**: {report.get('value', '?')}")
    add(f"- **结果**: {'✅ 已解出' if report.get('solved') else '❌ 未解出'}" +
        (f"（由 `{report.get('winner_spec')}`）" if report.get("winner_spec") else ""))
    if report.get("winner"):
        add(f"- **Flag**: `{report['winner']}`")
    add(f"- **bump 次数**: {report.get('bumps', 0)}　**导出于**: {report.get('exported_at_human', '')}")
    add("")

    desc = (report.get("description") or "").strip()
    if desc:
        add("## 题目描述")
        add("")
        add(desc)
        add("")

    fa = report.get("failed_approaches") or []
    if fa:
        add("## 🚫 死路账本（已排除方向）")
        add("")
        for t in fa:
            add(f"- {t}")
        add("")

    feed = report.get("shared_feed") or []
    if feed:
        add("## 📋 模型间共享黑板（事实 / 排除 / 广播）")
        add("")
        for f in feed:
            ts = time.strftime("%H:%M:%S", time.localtime(f.get("ts", 0)))
            mark = {"fact": "✓", "excluded": "✗", "broadcast": "📢"}.get(f.get("kind"), "•")
            add(f"- `{ts}` {mark} **[{f.get('model', '?')}]** {f.get('content', '')}")
        add("")

    agents = report.get("agents") or {}
    if agents:
        add("## 🤖 各模型尝试情况")
        add("")
        for spec, a in agents.items():
            add(f"### {spec}")
            add(f"- 状态: {a.get('status', '?')}　步数: {a.get('steps', 0)}")
            th = (a.get("thinking") or "").strip()
            if th:
                add(f"- 最后思路: {th}")
            find = (a.get("findings") or "").strip()
            if find:
                add("- 它的发现/尝试:")
                add("```")
                add(find)
                add("```")
            add("")

    return "\n".join(lines)


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
