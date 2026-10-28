"""Command-centre briefing — "what is the progress on <challenge>?"

Four hours into a hard challenge there is more state than a human can read: four
solvers, dozens of bumps, a blackboard of findings and dead ends, per-solver traces,
and a pile of audit counters. The operator's options today are to read the raw
export or to guess. This gives them a third option: ask.

The design constraint that matters most is the same one the rest of the trust system
exists to enforce, and it is *more* important here, not less:

    The briefing must never upgrade a ⚠ single-source claim into a fact.

An operator who is told "the team has established that the JAR gets loaded" will
type "go with the JAR-loading approach" into the advice box — and operator advice
outranks anything an agent says to itself. Laundering a hallucination through a
summary does not dilute it, it authorises it. So the dossier is handed over with its
trust tiers intact and the model is forbidden from collapsing them.

The second constraint: lead with what is DEAD. An operator skimming a progress
report needs the refuted premises and the excluded directions first, because the
most expensive thing they can do is suggest something already disproved.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from backend.message_bus import (
    KIND_BROADCAST,
    KIND_CLAIM,
    KIND_EXCLUDED,
    KIND_FACT,
    KIND_HELP,
    KIND_VERIFIED,
)
from backend.models import resolve_reasoning_model

logger = logging.getLogger(__name__)

# A four-hour grind can produce more state than fits comfortably. The dossier is
# assembled in trust order and trimmed from the LEAST authoritative end, and the
# model is told what was cut — a summary that silently dropped the refuted premises
# would be worse than no summary.
MAX_DOSSIER_CHARS = 60_000
MAX_FEED_PER_TIER = 120
MAX_TRACE_EVENTS = 25

SYSTEM_PROMPT = """\
你是 CTF 战队的**指挥参谋**。操作者（人类）正在盯一道已经打了很久的题，看不完全部日志，
所以向你提问。你的输入是这道题此刻的**完整状态档案**。

你的职责是让操作者在 30 秒内明白：现在到底什么状况、什么已经死了、下一步该让队伍干什么。

## 铁律：信任等级绝不允许改动

档案里的每条发现都带标记，这是整个系统防止幻觉传染的核心，你在总结时**必须原样保留**：

- ✅ **探针实证**：有 agent 在真实目标上跑过测试并观察到 —— 可以放心当作事实
- ✓ **双模型证实**：两个 agent 独立报告过同一件事 —— 是共识，但**没人实测过**
- ⚠ **单一声称**：只有一个 agent 说过，**未经任何验证**
- ✗ **已排除/已被证伪**：死路，不要再提
- 📢 **指示**：协调者或战略家说的话，不是事实

**绝对禁止**把 ⚠ 说成"已确认""团队发现""事实证明"。提到 ⚠ 内容时必须显式点明
"这只是某某模型单方面的说法，没有验证过"。这条比什么都重要：操作者会照你说的下指令，
而操作者的指令权威性高于任何 agent —— 你把一句幻觉洗成事实，等于给全队下达了一个
错误命令，而且没人会再质疑它。

同理，如果 ✅/✓ 的内容和 ⚠ 的内容冲突，以前者为准，并明确指出冲突。

## 回答方式

1. **先说结论**，再给依据。操作者时间有限。
2. **先说死的，再说活的**：被证伪的前提、已排除的方向要放在前面，因为操作者最
    expensive 的失误是建议一个已经被打假的东西。
3. 区分"**队伍知道的**"和"**队伍以为知道的**"。如果大量信息停在 ⚠ 层级，直接说
   这道题的根本问题是没有任何东西被验证过。
4. 如果档案显示这是**基础设施问题**（infra_failures、retire_reason 提到网关/超时、
   战略家持续失败），要明确告诉操作者：这不是解题方向问题，别再喂技术提示了。
5. **不知道就说不知道**。档案里没有的信息不要推测，直接讲"档案中没有这项"。
6. 用中文，简洁，用要点。不要复述整个档案，要提炼。

## 每次回答的最后

给出 1-3 条**操作者可以立刻执行的指令**，格式如下，让他能直接复制进指导信箱：

【建议指令】
- （发给某个 solver 或广播）具体、可执行、不超过三句话的指令

指令必须基于 ✅/✓ 层级的信息和已排除清单，绝不能建立在 ⚠ 之上；如果目前唯一合理的
动作是"先验证某个 ⚠ 声称"，那就把这条写成指令。
"""


@dataclass
class Briefing:
    """One answered question."""

    answer: str
    model: str = ""
    seconds: float = 0.0
    dossier_chars: int = 0
    trimmed: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.answer) and not self.error


@dataclass
class Turn:
    role: str    # "user" | "assistant"
    content: str


def _tier_header(kind: str) -> str:
    return {
        KIND_VERIFIED: "✅ 探针实证（最强证据，可当作事实）",
        KIND_FACT: "✓ 双模型独立证实（是共识，但没人实测）",
        KIND_CLAIM: "⚠ 仅单一模型声称，未经验证（引用时必须说明这一点）",
        KIND_EXCLUDED: "✗ 已排除 / 已被证伪（死路，不要建议）",
        KIND_BROADCAST: "📢 协调者 / 战略家指示（不是事实）",
        KIND_HELP: "? 待人工回答的知识缺口",
    }.get(kind, "• 其它")


# Strongest first: when the dossier has to be trimmed, the least authoritative tier
# is what goes, and the refuted/dead material is never cut.
_TIER_ORDER = (KIND_VERIFIED, KIND_EXCLUDED, KIND_FACT, KIND_BROADCAST, KIND_CLAIM, KIND_HELP)

# Trim priority, lowest number dropped FIRST. Verified facts, excluded/refuted
# material, the audit counters and the题面 are never dropped at any budget: they are
# exactly what an operator cannot reconstruct from memory, and losing the refuted
# premises is the one truncation that makes a briefing actively dangerous — the
# operator would then suggest something already disproved.
NEVER_TRIM = 0
_TRIM_CLAIMS = 1
_TRIM_HELP = 2
_TRIM_BROADCASTS = 3
_TRIM_SOLVER_DETAIL = 4
_TRIM_FAILED_APPROACHES = 5


def build_dossier(
    *,
    name: str,
    status: dict | None = None,
    meta: object | None = None,
    feed: list | None = None,
    failed_approaches: list | None = None,
    solver_notes: list | None = None,
    max_chars: int = MAX_DOSSIER_CHARS,
) -> tuple[str, list[str]]:
    """Assemble the full state dossier for one challenge.

    Returns `(dossier, trimmed_notes)`. The dossier is built as priority-ordered
    blocks and trimmed block by block from the weakest end, so a long run loses
    claim volume before it loses a single refuted premise or probe-verified fact.

    Trimming whole *sections* looks equivalent and is not: the blackboard tiers are
    subsections of one block, so a section-level trim discarded ✅ and ✗ alongside
    the ⚠ pile it was trying to shed.
    """
    status = status or {}
    trimmed: list[str] = []
    blocks: list[tuple[int, str, list[str]]] = []   # (trim_priority, label, lines)

    head: list[str] = [f"# 题目：{name}"]
    if meta is not None:
        head += [
            f"- 分类：{getattr(meta, 'category', '?')}　分值：{getattr(meta, 'value', '?')}"
            f"　标签：{', '.join(getattr(meta, 'tags', []) or []) or '(无)'}",
            f"- 连接方式：{getattr(meta, 'connection_info', '') or '(离线/仅附件)'}",
            f"- 题目描述原文：\n{(getattr(meta, 'description', '') or '(无)').strip()}",
        ]
    head += [
        "",
        "## 运行状态",
        f"- 已解出：{'是，flag=' + str(status.get('winner')) if status.get('winner') else '否'}"
        + (f"（by {status.get('winner_spec')}）" if status.get("winner_spec") else ""),
        f"- bump 轮数：{status.get('bumps', 0)}　是否已停止：{status.get('cancelled', False)}",
        f"- 退役原因：{status.get('retire_reason') or '(仍在运行)'}",
        f"- 连续基础设施失败：{status.get('infra_failures', 0)}"
        "　← 非 0 说明是网关/网络问题，不是解题方向问题",
    ]
    blocks.append((NEVER_TRIM, "题目与运行状态", head))

    blocks.append((NEVER_TRIM, "信任审计", [
        "## 信任审计（判断这道题的信息到底有多可靠）",
        f"- ✅ 探针实证：{status.get('probe_verified', 0)} 条",
        f"- ✓ 双模型证实：{status.get('corroborated_facts', 0)} 条",
        f"- ⚠ 仅单一声称、未验证：{status.get('unverified_claims', 0)} 条",
        f"- ⛔ 被探针证伪的前提：{status.get('probe_refuted', 0)} 条",
        f"- ⚡ 检测到的自相矛盾：{status.get('contradictions', 0)} 处",
        f"- ⛔ 熔断触发（有 solver 连续多轮零新信息）：{status.get('stale_breaks', 0)} 次",
        f"- 🧭 战略家 PIVOT：{status.get('strategist_pivots', 0)} 次"
        f"　战略家调用失败：{status.get('strategist_failures', 0)} 次"
        "　← 失败数高说明战略复盘一直没真正跑起来",
        f"- 🔬 待检验声称：{status.get('claims_queued', 0)} 条",
    ]))

    # The blackboard, one block per trust tier. Grouping is the point: a flat
    # chronological feed is what makes ⚠ and ✅ indistinguishable to a reader, and
    # separate blocks are also what makes tier-by-tier trimming possible.
    by_kind: dict[str, list] = {}
    for f in feed or []:
        kind = getattr(f, "kind", "") or (f.get("kind", "") if isinstance(f, dict) else "")
        content = getattr(f, "content", None)
        if content is None and isinstance(f, dict):
            content = f.get("content", "")
        model = getattr(f, "model", None)
        if model is None and isinstance(f, dict):
            model = f.get("model", "?")
        if not content:
            continue
        by_kind.setdefault(kind, []).append((model, str(content)))

    tier_priority = {
        KIND_VERIFIED: NEVER_TRIM,
        KIND_EXCLUDED: NEVER_TRIM,
        KIND_FACT: NEVER_TRIM,
        KIND_BROADCAST: _TRIM_BROADCASTS,
        KIND_CLAIM: _TRIM_CLAIMS,
        KIND_HELP: _TRIM_HELP,
    }
    blocks.append((NEVER_TRIM, "黑板标题",
                   ["## 共享黑板（按信任等级分组，标记必须原样保留）"]))
    for kind in _TIER_ORDER:
        items = by_kind.get(kind) or []
        if not items:
            continue
        lines = [f"### {_tier_header(kind)}"]
        shown = items[-MAX_FEED_PER_TIER:]
        if len(items) > len(shown):
            note = f"{_tier_header(kind)}：仅保留最近 {len(shown)}/{len(items)} 条"
            trimmed.append(note)
            lines.append(f"（{note}）")
        lines += [f"- [{model}] {content.strip()}" for model, content in shown]
        blocks.append((tier_priority.get(kind, _TRIM_CLAIMS), _tier_header(kind), lines))
    for kind, items in by_kind.items():
        if kind in _TIER_ORDER:
            continue
        blocks.append((_TRIM_CLAIMS, str(kind or "未标记"),
                       [f"### {kind or '(未标记)'}"]
                       + [f"- [{m}] {c.strip()}" for m, c in items[-MAX_FEED_PER_TIER:]]))

    if failed_approaches:
        blocks.append((_TRIM_FAILED_APPROACHES, "已失败的技术手册",
                       ["## 已试且失败的技术手册（不要建议这些）"]
                       + [f"- {a}" for a in failed_approaches]))
    if solver_notes:
        blocks.append((_TRIM_SOLVER_DETAIL, "各 solver 当前状态",
                       ["## 各 solver 当前状态", *solver_notes]))

    def _render(kept: list[tuple[int, str, list[str]]]) -> str:
        return "\n".join("\n".join(b[2]) for b in kept if b[2])

    kept = list(blocks)
    dossier = _render(kept)
    if len(dossier) > max_chars:
        # Drop whole blocks, weakest tier first, until it fits. A block is the
        # smallest safe unit: cutting inside one would leave the model quoting half
        # a finding as though it were complete.
        droppable = sorted(
            (i for i, b in enumerate(kept) if b[0] > NEVER_TRIM),
            key=lambda i: (kept[i][0], -len("\n".join(kept[i][2]))),
        )
        for i in droppable:
            if len(dossier) <= max_chars:
                break
            label = kept[i][1]
            size = len("\n".join(kept[i][2]))
            kept[i] = (kept[i][0], label, [])
            trimmed.append(f"为控制长度已整块省略「{label}」（{size} 字符）")
            dossier = _render(kept)

    if trimmed:
        dossier += (
            "\n\n## ⚠ 本档案已被截断\n"
            "以下为控制长度而省略的内容；如果操作者的问题正好落在这些内容上，"
            "必须明确说明你没有看到，不要猜：\n"
            + "\n".join(f"- {t}" for t in trimmed)
        )
    return dossier, trimmed


def format_solver_notes(swarm, max_events: int = MAX_TRACE_EVENTS) -> list[str]:
    """Per-solver live state: what it is doing right now, not its whole transcript.

    Reads traces off the event loop's usual path, so this is synchronous file I/O —
    callers on the dashboard run it through `asyncio.to_thread`.
    """
    from backend.trace_read import last_model_text, trace_health

    notes: list[str] = []
    status = {}
    try:
        status = (swarm.get_status() or {}).get("agents", {}) or {}
    except Exception:  # noqa: BLE001 — a partial briefing beats none
        logger.debug("could not read swarm status for briefing")

    for spec in (getattr(swarm, "_all_specs", None) or list(getattr(swarm, "solvers", {}))):
        info = status.get(spec, {}) or {}
        solver = (getattr(swarm, "solvers", {}) or {}).get(spec)
        path = getattr(getattr(solver, "tracer", None), "path", "") or ""
        line = [
            f"### {spec}",
            f"- 状态：{info.get('status', '?')}　工具调用：{info.get('tool_calls', 0)}"
            f"　连续零新信息轮数：{info.get('stale_rounds', 0)}",
        ]
        if path:
            try:
                health = trace_health(path)
                thinking = last_model_text(path)
                line.append(f"- 最近活动：{health.tool_calls} 次工具调用，{health.turns} 轮，"
                            f"{health.errors} 次错误")
                if health.spinning:
                    line.append("- ⚠ 轮次在完成但一次工具都没跑 —— 端点或工具接线坏了")
                elif health.looping:
                    line.append(f"- ⚠ 在重复同一条命令：{health.tool_sigs[-1][:150]}")
                if health.tool_sigs:
                    line.append("- 最近几条命令：")
                    line += [f"    - {s[:150]}" for s in health.tool_sigs[-max_events:]]
                if thinking:
                    line.append(f"- 它最后的思路：{thinking[:600]}")
            except OSError as e:
                line.append(f"- (trace 不可读: {e})")
        findings = (info.get("findings") or "").strip()
        if findings:
            line.append(f"- 它自报的最新发现（注意：这只是它自己的说法）：\n{findings[:2000]}")
        notes.append("\n".join(line))
    return notes


async def ask(
    *,
    question: str,
    dossier: str,
    history: list[Turn] | None = None,
    settings: object,
    cost_tracker: object | None = None,
    agent_name: str = "briefing",
    timeout_s: float | None = None,
    max_tokens: int | None = None,
) -> Briefing:
    """Answer one operator question against a freshly assembled dossier.

    The dossier is re-supplied every turn rather than cached in the conversation:
    the swarm is live, and an answer about "the current state" built on a snapshot
    from ten minutes ago is confidently wrong in exactly the way that gets an
    operator to issue a stale instruction.
    """
    base_url = getattr(settings, "gateway_base_url", "") or ""
    api_key = getattr(settings, "gateway_api_key", "") or ""
    if not base_url or not api_key:
        return Briefing(answer="", error="网关未配置（GATEWAY_BASE_URL / GATEWAY_API_KEY）",
                        dossier_chars=len(dossier))

    model = resolve_reasoning_model(settings, getattr(settings, "briefing_model", ""))
    if timeout_s is None:
        timeout_s = float(getattr(settings, "briefing_timeout_s", 0) or 300.0)
    if max_tokens is None:
        max_tokens = int(getattr(settings, "briefing_max_tokens", 0) or 8000)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-8:]:
        if turn.role in ("user", "assistant") and turn.content.strip():
            messages.append({"role": turn.role, "content": turn.content})
    messages.append({
        "role": "user",
        "content": f"【当前状态档案】\n{dossier}\n\n【操作者的问题】\n{question.strip()}",
    })

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={"model": model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": 0.3},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
    except (TimeoutError, httpx.TimeoutException):
        # httpx.TimeoutException is NOT a subclass of the builtin TimeoutError, so
        # catching only the latter silently routed real read timeouts into the
        # generic branch and lost the one piece of advice the operator can act on.
        logger.warning("briefing timed out after %.0fs on %s", timeout_s, model)
        return Briefing(answer="", model=model, seconds=time.monotonic() - t0,
                        dossier_chars=len(dossier),
                        error=f"简报超时（{timeout_s:.0f}s）——可提高 BRIEFING_TIMEOUT_S")
    except Exception as e:  # noqa: BLE001 — the operator must see why, not a blank pane
        logger.warning("briefing call failed on %s: %s", model, e)
        return Briefing(answer="", model=model, seconds=time.monotonic() - t0,
                        dossier_chars=len(dossier), error=f"{type(e).__name__}: {e}")

    elapsed = time.monotonic() - t0
    _record(cost_tracker, agent_name, model, data, elapsed)
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message", {}).get("content", "") or "").strip()
    if str(choice.get("finish_reason", "") or "") == "length":
        # A cut-off briefing may have dropped the 【建议指令】 block, which is the
        # part the operator actually acts on. Say so rather than showing a reply
        # that looks complete.
        return Briefing(answer=text, model=model, seconds=elapsed,
                        dossier_chars=len(dossier),
                        error="回答被 max_tokens 截断，建议指令可能不完整——可提高 BRIEFING_MAX_TOKENS")
    if not text:
        return Briefing(answer="", model=model, seconds=elapsed,
                        dossier_chars=len(dossier), error="模型返回空内容")
    return Briefing(answer=text, model=model, seconds=elapsed, dossier_chars=len(dossier))


def _record(cost_tracker, agent_name: str, model: str, data: dict, seconds: float) -> None:
    """Operator-facing calls are real billed traffic on a strong model."""
    if cost_tracker is None:
        return
    try:
        usage = data.get("usage") or {}
        cost_tracker.record_tokens(
            agent_name or "briefing", model,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            provider_spec="gateway",
            duration_seconds=seconds,
        )
    except Exception as e:  # noqa: BLE001 — accounting must never break the dashboard
        logger.debug("cost accounting for briefing failed: %s", e)
