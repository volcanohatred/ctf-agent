"""Round-level summaries for the fresh-restart retry loop.

After each GAVE_UP the round's outcome is distilled into a short "journal
entry" (what was tried, what worked, what died). Every N rounds the journal
is meta-summarized by an LLM — condensing and DE-DUPLICATING it — so the
carried-forward state stays tiny forever instead of exploding context.

This replaces "continue the same bloated session with a bump" with
"restart a clean session carrying only the distilled state".

Every gateway call here is accounted for through CostTracker: these are real
billed requests, and an earlier revision issued them without recording a cent,
so the totals the operator saw silently excluded them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SUMMARIZER_MODEL = "deepseek-v4-flash"

_ROUND_PROMPT = (
    "你是 CTF 复盘助手。把一个 solver 本轮的产出蒸馏成给「下一轮干净会话」用的状态卡，"
    "用中文要点列出，必须保留所有关键技术细节：\n"
    "- 试过的方向/技术（哪些有效、哪些失败）\n"
    "- 发现的具体产物：文件路径、地址、偏移、密钥、可能的 flag、凭据\n"
    "- 当前最优假设和下一步建议\n"
    "去掉过程性的啰嗦输出。直接给要点，不要解释。\n\n【本轮输出】\n{text}"
)

_META_PROMPT = (
    "你是 CTF 状态汇总助手。下面是某题连续多轮的探索摘要，把它们**合并去重**成一份"
    "极简的集体状态（不超过 12 行），格式固定为：\n"
    "【已排除的死路】…（合并所有失败方向，重复的只写一次）\n"
    "【已确认的事实】…（地址/偏移/文件/凭据等硬信息）\n"
    "【下一步主攻】…（综合判断当前最值得试的方向）\n"
    "直接输出这三段，不要解释。\n\n【各轮摘要】\n{text}"
)


@dataclass(frozen=True)
class SummaryResult:
    """Outcome of a summarization attempt.

    `condensed` distinguishes "the LLM actually condensed this" from "we handed
    the input back unchanged". Callers MUST check it: treating a fallback join
    as a successful condensation is what made the journal grow without bound
    whenever the gateway was unavailable.
    """

    text: str
    condensed: bool
    model: str = ""


async def _call_gateway(
    prompt: str,
    settings: object,
    model: str,
    timeout_s: float = 25.0,
    cost_tracker: object | None = None,
    agent_name: str = "",
) -> str:
    """One OpenAI-compatible chat completion. Returns "" on any failure."""
    base_url = getattr(settings, "gateway_base_url", "") or ""
    api_key = getattr(settings, "gateway_api_key", "") or ""
    if not base_url or not api_key:
        return ""
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 900, "temperature": 0.2},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            _record(cost_tracker, agent_name, model, data, time.monotonic() - t0)
            return ((data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    or "").strip()
    except Exception as e:  # noqa: BLE001 — summarizing is always best-effort
        logger.debug("round_summary gateway call failed: %s", e)
        return ""


def _record(cost_tracker, agent_name: str, model: str, data: dict, seconds: float) -> None:
    """Fold a gateway completion into the shared cost tracker."""
    if cost_tracker is None:
        return
    try:
        usage = data.get("usage") or {}
        cost_tracker.record_tokens(
            agent_name or "summarizer", model,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            provider_spec="gateway",
            duration_seconds=seconds,
        )
    except Exception as e:  # noqa: BLE001 — accounting must never break solving
        logger.debug("cost accounting for summarizer failed: %s", e)


def _model(settings: object) -> str:
    return getattr(settings, "summarizer_model", "") or DEFAULT_SUMMARIZER_MODEL


async def summarize_round(
    finding_text: str,
    settings: object,
    cost_tracker: object | None = None,
    agent_name: str = "",
) -> str:
    """Distill one round's output into a compact journal entry.

    On any failure returns the (trimmed) original so nothing is lost.
    """
    text = (finding_text or "").strip()
    if not text:
        return ""
    if len(text) < 700:
        return text  # short enough already
    out = await _call_gateway(
        _ROUND_PROMPT.format(text=text[:18000]), settings, _model(settings),
        cost_tracker=cost_tracker, agent_name=agent_name,
    )
    return out or text


async def meta_summarize(
    journal: list[str],
    settings: object,
    cost_tracker: object | None = None,
    agent_name: str = "",
) -> SummaryResult:
    """Condense + dedup N journal entries into a tight collective state.

    Returns `condensed=False` when the LLM was unavailable, so the caller keeps
    its bounded list of entries instead of replacing them with one ever-growing
    joined string.
    """
    entries = [j.strip() for j in journal if j and j.strip()]
    if not entries:
        return SummaryResult(text="", condensed=False)
    if len(entries) == 1:
        return SummaryResult(text=entries[0], condensed=True)
    joined = "\n---\n".join(f"[第{i+1}轮]\n{e}" for i, e in enumerate(entries))
    out = await _call_gateway(
        _META_PROMPT.format(text=joined[:20000]), settings, _model(settings),
        cost_tracker=cost_tracker, agent_name=agent_name,
    )
    if out:
        return SummaryResult(text=out, condensed=True, model=_model(settings))
    return SummaryResult(text=joined, condensed=False)
