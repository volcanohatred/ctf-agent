"""Contradiction audit — the swarm's statements checked against each other.

The Verifier tests a claim against the *target*. This tests claims against *each
other*, which catches a different and equally expensive failure: two agents holding
mutually exclusive beliefs while both look productive.

Observed live as the thing that should have ended a four-hour run in four minutes —
one agent had recorded that the allowlist rejects a gadget class, another was
building a payload out of that same class, and nothing in the system noticed. The
board held both statements side by side the entire time.

A contradiction is worth more than either claim on its own, because it is
self-localising: two statements cannot both be true, so at least one is wrong, and
the pair tells the Verifier exactly which two probes to run. That is why detection
here feeds straight into the verification queue at raised priority rather than just
being logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.config import float_setting, int_setting

logger = logging.getLogger(__name__)

# Reasoning models spend most of their completion budget thinking before they
# answer. At the 900 tokens `_call_gateway` used for summarization a reasoning
# model can be cut off before it emits a single CONTRADICTION line, which reads
# exactly like "no conflicts found".
DEFAULT_AUDIT_MAX_TOKENS = 4000
DEFAULT_AUDIT_TIMEOUT_S = 120.0

_AUDIT_PROMPT = (
    "你在审计一个 CTF 多智能体队伍的共享黑板。下面是**新写入的陈述**和**板上已有的陈述**。\n"
    "找出其中**互相矛盾**的对——即两者不可能同时为真的情况。\n\n"
    "判定标准要严格。只有下面这些才算矛盾：\n"
    "- 一个说某操作成功/被允许，另一个说同一操作失败/被拒绝\n"
    "- 一个说某物存在/可加载/可达，另一个说它不存在/加载失败/不可达\n"
    "- 两个给出互斥的具体值（不同的偏移、不同的类名、不同的状态码）指向同一个东西\n"
    "- 一个把某方向列为可行，另一个已把它列为死路\n\n"
    "**不算**矛盾：详略不同、语气不同、针对不同对象、一个是推测另一个是观察、"
    "仅仅是没有提到对方。宁可漏报也不要误报——误报会把队伍从正确方向上拽走。\n\n"
    "【新写入的陈述】\n{new}\n\n"
    "【板上已有的陈述】\n{existing}\n\n"
    "每个矛盾输出一行，严格用这个格式（三部分用 || 分隔）：\n"
    "CONTRADICTION: <陈述A原文> || <陈述B原文> || <为什么两者不可能同时为真，一句话>\n"
    "没有矛盾就只输出一行：\n"
    "NONE"
)

_PREFIX = "CONTRADICTION:"


@dataclass(frozen=True)
class Contradiction:
    """Two board statements that cannot both be true."""

    a: str
    b: str
    why: str

    @property
    def broadcast_text(self) -> str:
        """What goes on the board.

        Deliberately does NOT say which side is wrong — the audit has no evidence
        for that, and picking one would be a new unverified claim wearing an
        authoritative label. It names the conflict and says a probe is coming.
        """
        return (
            "⚡ **检测到矛盾**：板上同时存在两条不可能都为真的陈述。\n"
            f"  A：{self.a}\n"
            f"  B：{self.b}\n"
            f"  冲突点：{self.why}\n"
            "**在探针判定之前，不要基于 A 或 B 中的任何一条继续推导。**"
            "如果你已经基于其中一条构造了 payload，那部分工作暂时挂起。"
            "已派 Verifier 实测，结果会广播。"
        )


def parse_contradictions(raw: str) -> list[Contradiction]:
    """Read the strict `CONTRADICTION: a || b || why` lines out of a reply.

    Malformed lines are dropped rather than guessed at. A contradiction is an
    accusation that two agents are wrong about something, so a mis-parse is worse
    than a miss — the miss just leaves the board as it was.
    """
    out: list[Contradiction] = []
    for line in (raw or "").splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith(_PREFIX):
            continue
        parts = [p.strip() for p in stripped[len(_PREFIX):].split("||")]
        if len(parts) < 3 or not all(parts[:3]):
            logger.debug("dropping malformed contradiction line: %.120s", stripped)
            continue
        out.append(Contradiction(a=parts[0], b=parts[1], why=parts[2]))
    return out


def _render(items: list[str], limit: int, per_item: int = 300) -> str:
    return "\n".join(f"- {i[:per_item]}" for i in items[-limit:]) or "(none)"


async def find_contradictions(
    *,
    new_statements: list[str],
    existing: list[tuple[str, str]],
    settings: object,
    limit_existing: int = 80,
    cost_tracker: object | None = None,
    agent_name: str = "",
    model: str = "",
    max_tokens: int | None = None,
) -> list[Contradiction] | None:
    """One audit pass. `existing` is a list of (kind, content) board entries.

    Returns **None when no answer was obtained** — gateway unreachable, timeout,
    truncation, empty reply — as distinct from `[]` for "the model looked and found
    no conflict". Conflating the two is how the Strategist shipped dead once already:
    `_call_gateway` swallows every failure and returns "", so a broken audit read
    exactly like a perfectly consistent team, at debug level, with no counter. The
    caller tracks None in `_audit_failures` and surfaces it next to
    `strategist_failures`, because an operator reading `contradictions: 0` deserves to
    know whether that means "no conflicts" or "nobody checked".

    Never raises: this sits on the publish path, and an audit that can take down a
    round would be a far worse defect than the contradictions it fails to find.
    """
    new = [s.strip() for s in new_statements if s and s.strip()]
    if not new or not existing:
        return []

    from backend.models import resolve_reasoning_model
    from backend.round_summary import _call_gateway

    if max_tokens is None:
        max_tokens = int_setting(settings, "audit_max_tokens")
    prompt = _AUDIT_PROMPT.format(
        new=_render(new, 20),
        existing=_render([f"[{kind}] {content}" for kind, content in existing],
                         limit_existing),
    )
    resolved = model or resolve_reasoning_model(settings, getattr(settings, "audit_model", ""))
    raw = await _call_gateway(
        prompt, settings, resolved,
        timeout_s=float_setting(settings, "audit_timeout_s"),
        cost_tracker=cost_tracker, agent_name=agent_name, max_tokens=max_tokens,
    )
    if not raw:
        return None
    found = parse_contradictions(raw)
    if found:
        logger.info("contradiction audit found %d conflict(s)", len(found))
    return found
