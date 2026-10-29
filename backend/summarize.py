"""LLM-based key-info extraction for shared findings — replaces truncation.

When a solver's finding is long, we ask a fast gateway model to pull out the
load-bearing facts (flags, keys, addresses, payloads, techniques, dead-ends,
conclusions) instead of cutting the text. NOTHING is truncated — the full text
is always kept; the summary is only what siblings get injected with.
"""

from __future__ import annotations

import logging
import time

import httpx

from backend.config import float_setting, int_setting

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

    Uses the OpenAI-compatible gateway. On ANY failure returns the original
    text unchanged (never truncates — a summary failure must not lose info).
    Callers that need to tell "summarized" from "fell back" should compare the
    result against the input.

    A reply cut off by `max_tokens` counts as a failure and returns the original
    text too. It used to be returned as though complete, so the blackboard could
    carry the first 700 tokens of a long finding while every reader — including
    the verifier deciding what to probe — believed it held the whole thing.
    """
    text = text.strip()
    if not text:
        return text
    if timeout_s is None:
        timeout_s = float_setting(settings, "summarizer_timeout_s")
    max_tokens = int_setting(settings, "summarizer_max_tokens")
    base_url = getattr(settings, "gateway_base_url", "") or ""
    api_key = getattr(settings, "gateway_api_key", "") or ""
    if not base_url or not api_key:
        return text  # no summarizer configured — keep the full text

    model = model or getattr(settings, "summarizer_model", "") or DEFAULT_SUMMARIZER_MODEL
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _SUMMARIZE_PROMPT.format(text=text[:max_chars_in])}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            _record(cost_tracker, agent_name, model, data, time.monotonic() - t0)
            choice = (data.get("choices") or [{}])[0]
            if str(choice.get("finish_reason", "") or "") == "length":
                logger.warning(
                    "%s truncated at max_tokens=%s on %s — keeping the full text "
                    "rather than publishing half a finding as though it were whole",
                    agent_name or "summarizer", max_tokens, model,
                )
                return text
            summary = (choice.get("message", {}).get("content", "") or "").strip()
            return summary or text
    except Exception as e:  # noqa: BLE001 — summarizing is always best-effort
        logger.debug("summarize_for_siblings failed (%s) — keeping full text", e)
        return text


def _record(cost_tracker, agent_name: str, model: str, data: dict, seconds: float) -> None:
    """Fold a gateway completion into the shared cost tracker.

    These calls are real billed traffic; an earlier revision issued them without
    recording anything, so operator-visible totals silently excluded them.
    """
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
