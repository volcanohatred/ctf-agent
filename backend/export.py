"""Per-challenge report export — everything we know about how a challenge went.

Aggregates metadata, outcome, per-model attempts/findings/cost, the shared
blackboard feed (corroborated facts + unverified claims + excluded dead-ends +
broadcasts), the failed-approach ledger, the trust/stagnation audit counters,
and trace highlights. Rendered as JSON (machine) or Markdown (human /
writeup-friendly).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.message_bus import KIND_CLAIM, KIND_FACT, KIND_TAGS, KIND_VERIFIED

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> int:
    """Normalise a solver step/tool counter (list on one backend, int on the other)."""
    if isinstance(value, (list, tuple)):
        return int(value[0]) if value else 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def collect_challenge_data(
    name: str,
    swarm: Any | None,
    archived: dict | None,
) -> dict[str, Any]:
    """Normalize a challenge's data from a LIVE swarm or an archived summary."""
    if swarm is not None:
        meta = swarm.meta
        agents = {}
        # _all_specs includes escalation solvers, which model_specs does not.
        specs = list(getattr(swarm, "_all_specs", None) or swarm.model_specs)
        for spec in dict.fromkeys([*specs, *swarm.solvers]):
            solver = swarm.solvers.get(spec)
            tracer = getattr(solver, "tracer", None)
            path = getattr(tracer, "path", None)
            agents[spec] = {
                "status": "running" if solver else "not started",
                "steps": _as_int(getattr(solver, "_step_count", 0)) if solver else 0,
                "tool_calls": _as_int(getattr(solver, "_tool_calls", 0)) if solver else 0,
                "findings": swarm.findings.get(spec, ""),
                "thinking": "",
                "trace_path": path,
            }
            if solver is not None and path:
                try:
                    from backend.trace_read import last_model_text
                    agents[spec]["thinking"] = last_model_text(path)
                except Exception as e:  # noqa: BLE001 — the report must still render
                    logger.debug("could not read thinking for %s: %s", spec, e)
        feed = []
        bus = getattr(swarm, "message_bus", None)
        if bus is not None:
            for f in bus.findings:
                feed.append({"model": f.model, "kind": f.kind, "content": f.content,
                             "summary": f.summary, "ts": f.timestamp})
        return {
            "name": name,
            "category": getattr(meta, "category", ""),
            "value": getattr(meta, "value", 0),
            "description": getattr(meta, "description", ""),
            "tags": getattr(meta, "tags", []),
            "solved": bool(swarm.winner and swarm.winner.flag),
            "winner": swarm.winner.flag if (swarm.winner and swarm.winner.flag) else None,
            "winner_spec": swarm.winner_spec,
            "bumps": getattr(swarm, "_total_bumps", 0),
            "failed_approaches": list(getattr(swarm, "_failed_approaches", [])),
            "retire_reason": getattr(swarm, "retire_reason", ""),
            "stale_breaks": getattr(swarm, "_stale_breaks", 0),
            "strategist_pivots": getattr(swarm, "_strategist_pivots", 0),
            "corroborated_facts": sum(1 for f in feed if f["kind"] == KIND_FACT),
            "probe_verified": sum(1 for f in feed if f["kind"] == KIND_VERIFIED),
            "unverified_claims": sum(1 for f in feed if f["kind"] == KIND_CLAIM),
            "probe_refuted": getattr(swarm, "_probe_refuted", 0),
            "contradictions": getattr(swarm, "_contradictions", 0),
            "audit_failures": getattr(swarm, "_audit_failures", 0),
            "strategist_failures": getattr(swarm, "_strategist_failures", 0),
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
        "retire_reason": archived.get("retire_reason", ""),
        "stale_breaks": archived.get("stale_breaks", 0),
        "strategist_pivots": archived.get("strategist_pivots", 0),
        "corroborated_facts": archived.get("corroborated_facts", 0),
        "probe_verified": archived.get("probe_verified", 0),
        "unverified_claims": archived.get("unverified_claims", 0),
        "probe_refuted": archived.get("probe_refuted", 0),
        "contradictions": archived.get("contradictions", 0),
        "audit_failures": archived.get("audit_failures", 0),
        "strategist_failures": archived.get("strategist_failures", 0),
        "agents": archived.get("agents", {}),
        "shared_feed": archived.get("shared_feed", []),
        "archived": True,
        "finished_at": archived.get("finished_at"),
    }


def build_report(name: str, swarm: Any | None, archived: dict | None) -> dict[str, Any]:
    """Build the complete exportable report for one challenge."""
    data = collect_challenge_data(name, swarm, archived)
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
    if report.get("retire_reason"):
        add(f"- **swarm 退役原因**: `{report['retire_reason']}`")
    # A solve rate cannot tell "converged on verified facts" from "each model
    # asserted things nobody checked until the clock ran out". These can.
    facts_n = report.get("corroborated_facts", 0)
    verified_n = report.get("probe_verified", 0)
    claims_n = report.get("unverified_claims", 0)
    if facts_n or verified_n or claims_n or report.get("stale_breaks") or report.get("strategist_pivots"):
        add(
            f"- **信任审计**: ✅ 探针实证 {verified_n} 条　✓ 双模型证实 {facts_n} 条　"
            f"⚠ 仅单一模型声称 {claims_n} 条　⛔ 被证伪 {report.get('probe_refuted', 0)} 条"
            f"　⚡ 自相矛盾 {report.get('contradictions', 0)} 处"
            f"　⛔ 熔断 {report.get('stale_breaks', 0)} 次"
            f"　🧭 战略家 PIVOT {report.get('strategist_pivots', 0)} 次"
        )
        # A zero here is ambiguous unless the failure counts are shown next to it:
        # "no contradictions found" and "the audit never ran" lead an operator to
        # opposite conclusions about whether the team's beliefs were ever checked.
        strat_failed = report.get("strategist_failures", 0)
        audit_failed = report.get("audit_failures", 0)
        if strat_failed or audit_failed:
            add(
                f"- ⚠ **机制未生效**: 战略家调用失败 {strat_failed} 次、矛盾审计失败 "
                f"{audit_failed} 次。上面的 0 值可能意味着「检查过且没问题」，也可能意味着"
                "「根本没检查」——复盘时不要当成前者。"
            )
        if claims_n > (facts_n + verified_n) * 3 and claims_n >= 6:
            add(
                "- ⚠ **本题大部分\"已知信息\"从未被第二个模型证实、也从未被探针实测**。"
                "下面的复盘结论若建立在 ⚠ 条目上，请先自行复验再采信。"
            )
        if report.get("probe_refuted"):
            add(
                f"- ⛔ **有 {report['probe_refuted']} 个前提被实测证伪**。复盘时注意："
                "这些前提曾经出现在黑板上并被当作事实使用过，任何基于它们推导出的"
                "中间结论同样作废。"
            )
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
        add("## 📋 模型间共享黑板（实证 / 证实 / 未证实声称 / 排除 / 广播）")
        add("")
        add("> ✅ = Verifier 在真实目标上跑探针实测确认　✓ = 被两个以上模型独立证实"
             "　⚠ = **仅单一模型声称、未经证实**　✗ = 已排除死路（含被证伪的前提）"
             "　📢 = 协调者/战略家指示")
        add("")
        for f in feed:
            ts = time.strftime("%H:%M:%S", time.localtime(f.get("ts", 0)))
            # Shared with the live blackboard on purpose: a post-mortem that
            # renders an unverified claim with the same bullet as a corroborated
            # fact reproduces the exact mistake the trust gate exists to prevent.
            mark = KIND_TAGS.get(f.get("kind", ""), "•")
            add(f"- `{ts}` {mark} **[{f.get('model', '?')}]** {f.get('content', '')}")
        add("")

    agents = report.get("agents") or {}
    if agents:
        add("## 🤖 各模型尝试情况")
        add("")
        for spec, a in agents.items():
            add(f"### {spec}")
            add(f"- 状态: {a.get('status', '?')}　步数: {a.get('steps', 0)}"
                f"　工具调用: {a.get('tool_calls', 0)}")
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
