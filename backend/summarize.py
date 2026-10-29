"""LLM-based key-info extraction for shared findings — replaces truncation.

When a solver's finding is long, we ask a fast gateway model to pull out the
load-bearing facts (flags, keys, addresses, payloads, techniques, dead-ends,
conclusions) instead of cutting the text. NOTHING is truncated — the full text
is always kept; the summary is only what siblings get injected with.

The gateway call itself is delegated to `round_summary._call_gateway`, which every
other summarizer-class call already uses. This module used to carry its own copy
of the request, the cost accounting and the truncation check — and that copy was
missing the truncation check entirely, so it published half a finding as though it
were whole while the identical call path elsewhere discarded the same reply. One
implementation is the only way that cannot happen again.
"""

from __future__ import annotations

import logging

from backend.round_summary import _call_gateway

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
    timeout_s: float | None = None,
    cost_tracker: object | None = None,
    agent_name: str = "",
) -> str:
    """Return a concise key-info summary of a (possibly long) finding.

    On ANY failure returns the original text unchanged — a summary failure must
    not lose info. Callers that need to tell "summarized" from "fell back" should
    compare the result against the input.

    A reply cut off by `max_tokens` counts as a failure and returns the original
    text too, but only after `_call_gateway` has retried with a grown budget. This
    prompt demands that every flag, key, address and payload survive, with no
    length bound, so its natural output grows with the finding and a fixed cap
    would otherwise clip it silently.
    """
    text = text.strip()
    if not text:
        return text
    out = await _call_gateway(
        _SUMMARIZE_PROMPT.format(text=text[:max_chars_in]),
        settings,
        model or getattr(settings, "summarizer_model", "") or DEFAULT_SUMMARIZER_MODEL,
        timeout_s=timeout_s,
        cost_tracker=cost_tracker,
        agent_name=agent_name,
    )
    return out or text
