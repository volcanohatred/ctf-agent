"""Round-level summaries for the fresh-restart retry loop.

After each GAVE_UP the round's outcome is distilled into a short "journal
entry" (what was tried, what worked, what died). Every N rounds the journal
is meta-summarized by an LLM — condensing and DE-DUPLICATING it — so the
carried-forward state stays tiny forever instead of exploding context.

This replaces "continue the same bloated session with a bump" with
"restart a clean session carrying only the distilled state".
"""

from __future__ import annotations

import logging

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


async def _call_gateway(prompt: str, settings: object, model: str, timeout_s: float = 25.0) -> str:
    base_url = getattr(settings, "gateway_base_url", "") or ""
    api_key = getattr(settings, "gateway_api_key", "") or ""
    if not base_url or not api_key:
        return ""
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
            return (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.debug(f"round_summary gateway call failed: {e}")
        return ""


def _model(settings: object) -> str:
    return getattr(settings, "summarizer_model", "") or DEFAULT_SUMMARIZER_MODEL


async def summarize_round(finding_text: str, settings: object) -> str:
    """Distill one round's output into a compact journal entry.

    On any failure returns the (trimmed) original so nothing is lost.
    """
    text = (finding_text or "").strip()
    if not text:
        return ""
    if len(text) < 700:
        return text  # short enough already
    out = await _call_gateway(_ROUND_PROMPT.format(text=text[:18000]), settings, _model(settings))
    return out or text


async def meta_summarize(journal: list[str], settings: object) -> str:
    """Condense + dedup N journal entries into a tight collective state."""
    entries = [j.strip() for j in journal if j and j.strip()]
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0]
    joined = "\n---\n".join(f"[第{i+1}轮]\n{e}" for i, e in enumerate(entries))
    out = await _call_gateway(_META_PROMPT.format(text=joined[:20000]), settings, _model(settings))
    return out or "\n---\n".join(entries)
