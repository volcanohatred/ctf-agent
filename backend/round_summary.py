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

from backend.config import float_setting, int_setting, str_setting

logger = logging.getLogger(__name__)

DEFAULT_SUMMARIZER_MODEL = "deepseek-v4-flash"

# Truncation retry: how much to grow max_tokens, and where to stop. The ceiling
# exists because growth is multiplicative and a prompt that cannot be satisfied
# must not end up requesting an absurd completion.
TRUNCATION_GROWTH = 4
TRUNCATION_CEILING = 32_000
TRUNCATION_TIMEOUT_CEILING = 600.0

# Set once if the gateway answers HTTP 400 to `reasoning_effort`, so an
# unsupported field cannot silently disable every summarizer call in the run.
_REASONING_REJECTED = False

_ROUND_PROMPT = (
    "你是 CTF 复盘助手。把一个 solver 本轮的产出蒸馏成给「下一轮干净会话」用的状态卡，"
    "用中文要点列出，必须保留所有关键技术细节：\n"
    "- 试过的方向/技术（哪些有效、哪些失败）\n"
    "- 发现的具体产物：文件路径、地址、偏移、密钥、可能的 flag、凭据\n"
    "- 当前最优假设和下一步建议\n"
    "去掉过程性的啰嗦输出。直接给要点，不要解释。\n"
    "**全文控制在 20 行以内。** 硬信息（路径、地址、偏移、密钥、flag、凭据）必须逐条列出，"
    "一行放不下的宁可压缩描述文字也不要省略这些值；已经失败的方向每条只写一行结论，"
    "不要复述失败过程。\n\n【本轮输出】\n{text}"
)

_META_PROMPT = (
    "你是 CTF 状态汇总助手。下面是某题连续多轮的探索摘要，把它们**合并去重**成一份"
    "极简的集体状态（不超过 12 行），格式固定为：\n"
    "【已排除的死路】…（合并所有失败方向，重复的只写一次）\n"
    "【已确认的事实】…（地址/偏移/文件/凭据等硬信息）\n"
    "【下一步主攻】…（综合判断当前最值得试的方向）\n"
    "直接输出这三段，不要解释。\n\n【各轮摘要】\n{text}"
)

_SIBLING_PROMPT = (
    "你是 CTF 团队协作助手。下面是同一道题上**其他模型**已经报告的发现。"
    "把它们合并成一份给队友看的集体状态，必须去重、必须保留所有硬信息"
    "（文件路径、地址、偏移、密钥、凭据、payload、已排除的方向）。\n\n"
    "**每行开头的标记是信任等级，必须严格保留，绝对不许升级或降级：**\n"
    "✅ = 已被 Verifier 在真实目标上跑探针实测确认（最强证据）\n"
    "✓ = 已被两个以上模型独立报告（是共识，但没人实测过）\n"
    "⚠ = **仅单一模型声称、未经证实**——不许写进【已确认】\n"
    "✗ = 已排除的死路（含被探针证伪的前提）\n"
    "📢 = 协调者/战略家的指示，不是事实\n\n"
    "这条规则是本系统的核心：把 ⚠ 混进已确认区，会让整支队伍基于一句幻觉"
    "连续工作数小时。宁可少写，也不许改动信任等级。\n\n"
    "不超过 16 行，格式固定为：\n"
    "【已实测确认】…（只允许 ✅ 的内容）\n"
    "【队友已确认】…（只允许 ✓ 的内容）\n"
    "【未经证实的声称】…（⚠ 的内容，注明是谁声称的；用它之前先自己复验）\n"
    "【队友已排除】…（不要再试这些）\n"
    "【值得接力】…（队友做到一半、最有希望接着做的点）\n"
    "直接输出这五段，没有内容的段落写「无」。不要解释。\n\n【其他模型的报告】\n{text}"
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
    timeout_s: float | None = None,
    cost_tracker: object | None = None,
    agent_name: str = "",
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """One OpenAI-compatible chat completion. Returns "" on any failure.

    `reasoning_effort` is sent as a top-level field when non-empty. Summarizer-class
    callers pass it; the reasoning tasks (audit, strategist, briefing, triage) do
    not, because for those the thinking IS the product.

    A reply truncated by `max_tokens` also returns "". That is not the same as a
    transport failure but it must be handled the same way, because the callers ask
    for a strict trailing marker (PIVOT:, CONTRADICTION:, TESTABLE:) and a model cut
    off mid-reasoning never emits it. Returning the truncated tail instead would let
    `parse_verdict` read deliberation as a conclusion — observed live as a strategist
    reply that was heading for PIVOT being reported as CONTINUE, which is worse than
    no answer because the swarm then keeps doing the thing that was failing.
    """
    # None means "this is a summarizer-class call" — resolve the shared budget from
    # settings. Callers with their own measured budget (audit, strategist, briefing,
    # triage) pass an explicit value and are unaffected. The previous literals were
    # 900 tokens / 25s, which a live run showed the summarizer hitting repeatedly;
    # because a truncated reply is discarded below, every hit silently cost that
    # round its summary.
    if max_tokens is None:
        max_tokens = int_setting(settings, "summarizer_max_tokens")
    if timeout_s is None:
        timeout_s = float_setting(settings, "summarizer_timeout_s")
    retries = int_setting(settings, "summarizer_truncation_retries")
    base_url = getattr(settings, "gateway_base_url", "") or ""
    api_key = getattr(settings, "gateway_api_key", "") or ""
    if not base_url or not api_key:
        return ""

    budget, wait = max_tokens, timeout_s
    effort = (reasoning_effort or "").strip()
    global _REASONING_REJECTED

    async def _post(client, with_effort: bool):
        body: dict = {"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": budget, "temperature": 0.2}
        if with_effort:
            body["reasoning_effort"] = effort
        return await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=body, headers={"Authorization": f"Bearer {api_key}"})

    for attempt in range(retries + 1):
        t0 = time.monotonic()
        try:
            sent_effort = bool(effort) and not _REASONING_REJECTED
            async with httpx.AsyncClient(timeout=wait) as client:
                resp = await _post(client, sent_effort)
                if resp.status_code == 400 and sent_effort:
                    # A gateway that does not know the field would 400 on EVERY
                    # summarizer call, and the generic handler below turns that into
                    # "" — silently disabling summarization for the whole run. Pay
                    # the 400 once, remember it, and stop sending the field.
                    _REASONING_REJECTED = True
                    logger.warning(
                        "%s: gateway rejected reasoning_effort=%r (HTTP 400) — not "
                        "sending it again this process", agent_name or "gateway call",
                        effort,
                    )
                    resp = await _post(client, False)
                resp.raise_for_status()
                data = resp.json()
                _record(cost_tracker, agent_name, model, data, time.monotonic() - t0)
                choice = (data.get("choices") or [{}])[0]
                if str(choice.get("finish_reason", "") or "") != "length":
                    return (choice.get("message", {}).get("content", "") or "").strip()
        except (TimeoutError, httpx.TimeoutException):
            # httpx.TimeoutException is not a subclass of the builtin TimeoutError, so
            # catching only the latter let real read timeouts fall into the generic
            # branch below and be logged at debug level — the strategist measured
            # 100-150s per call against a live gateway, i.e. it timed out on every
            # single request while looking configured and healthy.
            logger.warning(
                "%s timed out after %.0fs on %s — no answer obtained (raise the timeout "
                "for this call if it keeps happening)",
                agent_name or "gateway call", wait, model,
            )
            return ""
        except Exception as e:  # noqa: BLE001 — summarizing is always best-effort
            logger.debug("round_summary gateway call failed: %s", e)
            return ""

        # Truncated. Two of the four prompts using this call (_ROUND_PROMPT and
        # _SUMMARIZE_PROMPT) demand that EVERY technical detail survive — paths,
        # offsets, keys, credentials — with no length bound, so their natural
        # output grows with the round and no fixed cap is provably safe. Rather
        # than guess a bigger number, grow it and try again: a distillation that
        # silently degrades to the raw round output is what inflates the journal
        # the FRESH_RESTART_ROUNDS merge exists to keep small.
        grown = min(budget * TRUNCATION_GROWTH, TRUNCATION_CEILING)
        if attempt >= retries or grown <= budget:
            logger.warning(
                "%s truncated at max_tokens=%s on %s — discarding the reply rather "
                "than parsing a partial one (raise the budget for this call)",
                agent_name or "gateway call", budget, model,
            )
            return ""
        logger.info(
            "%s truncated at max_tokens=%s on %s — retrying with %s",
            agent_name or "gateway call", budget, model, grown,
        )
        budget = grown
        # A longer completion takes proportionally longer to generate; retrying at
        # the original timeout would convert a truncation into a timeout.
        wait = min(wait * TRUNCATION_GROWTH, TRUNCATION_TIMEOUT_CEILING)
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


async def summarize_siblings(
    reports: list[str],
    settings: object,
    cost_tracker: object | None = None,
    agent_name: str = "",
) -> SummaryResult:
    """Condense what the OTHER solvers on this challenge have established.

    Sibling findings previously reached a restarting solver as "the latest raw
    finding per model, clipped to 400 chars" — so early discoveries dropped out,
    and nothing was merged or de-duplicated. This gives them the same treatment
    the solver's own journal gets.

    Returns `condensed=False` when the model was unavailable, so the caller can
    fall back to the raw (bounded) form instead of presenting a join as a summary.
    """
    from backend.dedup import dedup_entries

    entries = dedup_entries([r for r in reports if r and r.strip()])
    if not entries:
        return SummaryResult(text="", condensed=False)
    if len(entries) == 1 and len(entries[0]) < 700:
        return SummaryResult(text=entries[0], condensed=True)
    joined = "\n---\n".join(f"[{i+1}]\n{e}" for i, e in enumerate(entries))
    out = await _call_gateway(
        _SIBLING_PROMPT.format(text=joined[:20000]), settings, _model(settings),
        cost_tracker=cost_tracker, agent_name=agent_name,
        reasoning_effort=str_setting(settings, "summarizer_reasoning_effort"),
    )
    if out:
        return SummaryResult(text=out, condensed=True, model=_model(settings))
    return SummaryResult(text=joined, condensed=False)


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
        reasoning_effort=str_setting(settings, "summarizer_reasoning_effort"),
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
        reasoning_effort=str_setting(settings, "summarizer_reasoning_effort"),
    )
    if out:
        return SummaryResult(text=out, condensed=True, model=_model(settings))
    return SummaryResult(text=joined, condensed=False)
