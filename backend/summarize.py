"""LLM-based key-info extraction for shared findings — replaces truncation.

When a solver's finding is long, we ask a fast gateway model to pull out the
load-bearing facts (flags, keys, addresses, payloads, techniques, dead-ends,
conclusions) instead of cutting the text. NOTHING is truncated — the full text
is always kept; the summary is only what siblings get injected with.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

# Keep it cheap+fast: summarize with a small/fast gateway model by default.
DEFAULT_SUMMARIZER_MODEL = "deepseek-v4-flash"

_SUMMARIZE_PROMPT = (
    "你是一个 CTF 解题助手。把下面这段 solver 输出提炼成给队友看的关键信息摘要，"
    "必须完整保留：flag、密钥、地址、payload、用了什么技术、排除了哪些方向、最终结论。"
    "去掉过程性的啰嗦输出。直接输出摘要，不要解释。\n\n"
    "【solver 输出】\n{text}"
)


async def summarize_for_siblings(
    text: str,
    settings: object,
    model: str | None = None,
    max_chars_in: int = 20000,
    timeout_s: float = 25.0,
) -> str:
    """Return a concise key-info summary of a (possibly long) finding.

    Uses the OpenAI-compatible gateway. On ANY failure returns the original
    text unchanged (never truncates — a summary failure must not lose info).
    """
    text = text.strip()
    if not text:
        return text
    base_url = getattr(settings, "gateway_base_url", "") or ""
    api_key = getattr(settings, "gateway_api_key", "") or ""
    if not base_url or not api_key:
        return text  # no summarizer configured — keep the full text

    model = model or getattr(settings, "summarizer_model", "") or DEFAULT_SUMMARIZER_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _SUMMARIZE_PROMPT.format(text=text[:max_chars_in])}],
        "max_tokens": 700,
        "temperature": 0.2,
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            summary = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            return summary or text
    except Exception as e:
        logger.debug(f"summarize_for_siblings failed ({e}) — keeping full text")
        return text
