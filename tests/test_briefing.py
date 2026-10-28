"""Command-centre briefing — the operator's view of a challenge.

The failure this guards against is specific and worse than the one the rest of the
trust system guards against. An operator who is told "the team has established that
the JAR gets loaded" will type "go with the JAR-loading approach" into the advice
box — and operator advice outranks anything an agent says to itself. Laundering a
⚠ claim through a summary does not dilute it, it AUTHORISES it.

So the load-bearing tests here are not "does it produce text" but:
  * the dossier reaches the model with its trust tiers intact and labelled,
  * the system prompt forbids collapsing them,
  * the dead material is presented first and is never trimmed away,
  * an infrastructure problem is not summarised as a strategy problem.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from backend.briefing import (
    MAX_DOSSIER_CHARS,
    SYSTEM_PROMPT,
    Briefing,
    Turn,
    ask,
    build_dossier,
)
from backend.message_bus import (
    KIND_BROADCAST,
    KIND_CLAIM,
    KIND_EXCLUDED,
    KIND_FACT,
    KIND_VERIFIED,
    Finding,
)


def _status(**over) -> dict:
    base = dict(
        winner=None, winner_spec="", bumps=24, cancelled=False,
        retire_reason="round cap reached (12)", infra_failures=0,
        probe_verified=1, corroborated_facts=3, unverified_claims=40,
        probe_refuted=2, contradictions=1, stale_breaks=5,
        strategist_pivots=2, strategist_failures=0, claims_queued=4,
    )
    base.update(over)
    return base


def _meta() -> SimpleNamespace:
    return SimpleNamespace(
        name="fatjar-relic", category="web", value=500, tags=["java", "fastjson"],
        connection_info="http://172.21.254.205:32002",
        description="离线连接器审核平台保留了旧版扩展配置的兼容迁移。Worker 显式保留 FD 12-48。",
    )


def _feed() -> list[Finding]:
    return [
        Finding(model="verifier", content="migrate accepts a JAR (HTTP 200 observed)",
                kind=KIND_VERIFIED),
        Finding(model="m1", content="the uploaded JAR gets loaded by the migrator",
                kind=KIND_CLAIM),
        Finding(model="m2", content="422 on every blocked class name", kind=KIND_FACT),
        Finding(model="m3", content="$ref chain is blocked by the allowlist", kind=KIND_EXCLUDED),
        Finding(model="strategist", content="PIVOT: attack the FD retention", kind=KIND_BROADCAST),
    ]


# ---------------------------------------------------------------------------
# The dossier
# ---------------------------------------------------------------------------

def test_the_dossier_labels_every_trust_tier_explicitly() -> None:
    """The model cannot preserve a distinction it was never shown. Each tier has to
    arrive with its marker AND a plain-language statement of what that marker means."""
    dossier, _ = build_dossier(name="fatjar-relic", status=_status(), meta=_meta(),
                               feed=_feed())

    assert "✅ 探针实证" in dossier
    assert "✓ 双模型独立证实" in dossier
    assert "仅单一模型声称，未经验证" in dossier
    assert "已排除" in dossier
    # And the actual findings, attributed to who said them.
    assert "[m1] the uploaded JAR gets loaded by the migrator" in dossier
    assert "[verifier] migrate accepts a JAR" in dossier


def test_a_claim_is_never_presented_bare() -> None:
    """The ⚠ line and its finding must be adjacent — a reader who skips the header
    still has to see the warning attached to the claim itself."""
    dossier, _ = build_dossier(name="c", status=_status(), meta=_meta(), feed=_feed())
    claim_section = dossier.split("仅单一模型声称")[1].split("###")[0]
    assert "the uploaded JAR gets loaded" in claim_section


def test_verified_outranks_everything_and_dead_material_comes_early() -> None:
    """Tier order is the reading order. Probe-verified first (what is solid), then
    the dead ends (what must not be suggested), then mere agreement, then claims."""
    dossier, _ = build_dossier(name="c", status=_status(), meta=_meta(), feed=_feed())

    i_verified = dossier.index("✅ 探针实证")
    i_excluded = dossier.index("✗ 已排除")
    i_fact = dossier.index("✓ 双模型独立证实")
    i_claim = dossier.index("仅单一模型声称")
    assert i_verified < i_excluded < i_fact < i_claim


def test_the_audit_counters_reach_the_model() -> None:
    """"40 unverified claims and 1 corroborated fact" IS the progress report on a
    run like this. Without the counters the model can only summarise the content and
    will report a busy-looking team that established nothing."""
    dossier, _ = build_dossier(name="c", status=_status(), meta=_meta(), feed=_feed())

    assert "40" in dossier and "被探针证伪的前提：2" in dossier
    assert "自相矛盾：1" in dossier
    assert "熔断触发" in dossier and "5" in dossier
    assert "战略家调用失败" in dossier


def test_an_infrastructure_problem_is_flagged_as_such() -> None:
    """The most expensive misreading: an operator told "the team is stuck on the
    deserialization angle" will send technical hints at a dead gateway."""
    dossier, _ = build_dossier(
        name="c",
        status=_status(infra_failures=3, retire_reason="3 consecutive infrastructure failures"),
        meta=_meta(), feed=_feed())

    assert "不是解题方向问题" in dossier
    assert "3 consecutive infrastructure failures" in dossier


def test_a_persistent_strategist_failure_is_surfaced() -> None:
    """A strategist that times out on every call leaves the run looking like the
    strategy was reviewed repeatedly. The counter is what says otherwise."""
    dossier, _ = build_dossier(name="c", status=_status(strategist_failures=7),
                               meta=_meta(), feed=_feed())
    assert "战略家调用失败：7" in dossier
    assert "一直没真正跑起来" in dossier


def test_the_challenge_description_is_included_verbatim() -> None:
    """The operator's usual question is "are they even attacking the right thing?",
    which cannot be answered without the题面."""
    dossier, _ = build_dossier(name="c", status=_status(), meta=_meta(), feed=_feed())
    assert "Worker 显式保留 FD 12-48" in dossier


def test_trimming_drops_the_weakest_tier_and_says_so() -> None:
    """Over budget, the least authoritative material goes. Silently dropping the
    refuted premises would be the one truncation that makes the briefing actively
    dangerous — an operator would happily suggest something already disproved."""
    feed = [Finding(model="m1", content=f"claim number {i} " + "x" * 200, kind=KIND_CLAIM)
            for i in range(400)]
    feed.append(Finding(model="v", content="the one probe-verified fact", kind=KIND_VERIFIED))
    feed.append(Finding(model="m9", content="the one refuted premise", kind=KIND_EXCLUDED))

    dossier, trimmed = build_dossier(name="c", status=_status(), meta=_meta(), feed=feed,
                                     max_chars=20000)

    assert trimmed, "the operator must be told the dossier was cut"
    assert "已被截断" in dossier
    # The strongest and the dead material survive; the claim pile is what got cut.
    assert "the one probe-verified fact" in dossier
    assert "the one refuted premise" in dossier


def test_a_small_dossier_is_not_trimmed() -> None:
    dossier, trimmed = build_dossier(name="c", status=_status(), meta=_meta(), feed=_feed())
    assert trimmed == []
    assert "已被截断" not in dossier
    assert len(dossier) < MAX_DOSSIER_CHARS


def test_an_empty_challenge_still_produces_a_usable_dossier() -> None:
    """A challenge that just started has no findings at all. The briefing must still
    answer — "nothing has been established yet" is a real and useful answer."""
    dossier, trimmed = build_dossier(
        name="c", status=_status(bumps=0, unverified_claims=0, probe_refuted=0,
                                 corroborated_facts=0, probe_verified=0),
        meta=_meta(), feed=[])

    assert "bump 轮数：0" in dossier
    assert "题目描述原文" in dossier, "the题面 is present even with no findings"
    assert trimmed == []


def test_dead_ends_and_failed_approaches_are_listed() -> None:
    dossier, _ = build_dossier(
        name="c", status=_status(), meta=_meta(), feed=_feed(),
        failed_approaches=["fastjson-gadgets.md", "java-deserialization.md"])
    assert "fastjson-gadgets.md" in dossier
    assert "不要建议这些" in dossier


def test_dict_shaped_feed_entries_work_too() -> None:
    """The archived path hands over plain dicts, not Finding objects. Both shapes
    must render identically or a finished challenge briefs worse than a live one —
    and finished challenges are exactly what gets post-mortemed."""
    as_dicts = [{"model": f.model, "kind": f.kind, "content": f.content} for f in _feed()]
    from_dicts, _ = build_dossier(name="c", status=_status(), meta=_meta(), feed=as_dicts)
    from_objects, _ = build_dossier(name="c", status=_status(), meta=_meta(), feed=_feed())

    for needle in ("the uploaded JAR gets loaded", "$ref chain is blocked",
                   "PIVOT: attack the FD retention"):
        assert needle in from_dicts
    assert len(from_dicts) == len(from_objects)


# ---------------------------------------------------------------------------
# The system prompt
# ---------------------------------------------------------------------------

def test_the_prompt_forbids_upgrading_a_claim() -> None:
    """This is the whole feature's reason to exist. If the prompt does not forbid it
    explicitly, the model will write "团队发现 X" for a ⚠ line every time, because
    that is what a helpful summary sounds like."""
    assert "绝对禁止" in SYSTEM_PROMPT
    assert "⚠" in SYSTEM_PROMPT
    assert "没有验证过" in SYSTEM_PROMPT or "未经任何验证" in SYSTEM_PROMPT


def test_the_prompt_explains_why_it_must_not_upgrade() -> None:
    """A rule with no rationale gets dropped by a model under length pressure. The
    reason — operator instructions outrank agent self-correction — is what makes it
    stick."""
    assert "操作者" in SYSTEM_PROMPT
    assert "指令" in SYSTEM_PROMPT


def test_the_prompt_puts_dead_material_first() -> None:
    assert SYSTEM_PROMPT.index("先说死的") < SYSTEM_PROMPT.index("【建议指令】")


def test_the_prompt_demands_actionable_output() -> None:
    """A summary the operator still has to translate into an instruction has only
    done half the job — and the dashboard has an advice box waiting for it."""
    assert "【建议指令】" in SYSTEM_PROMPT
    assert "绝不能建立在 ⚠ 之上" in SYSTEM_PROMPT


def test_the_prompt_requires_admitting_ignorance() -> None:
    assert "不知道就说不知道" in SYSTEM_PROMPT


def test_the_prompt_covers_every_tier_marker() -> None:
    for marker in ("✅", "✓", "⚠", "✗", "📢"):
        assert marker in SYSTEM_PROMPT, f"the prompt does not explain {marker}"


# ---------------------------------------------------------------------------
# The gateway call
# ---------------------------------------------------------------------------

def _settings(**over) -> SimpleNamespace:
    base = dict(gateway_base_url="http://gw/v1", gateway_api_key="k",
                briefing_model="", reasoning_model="", coordinator_model="",
                summarizer_model="deepseek-v4-flash",
                briefing_timeout_s=300.0, briefing_max_tokens=8000)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def patch_client(monkeypatch):
    def _apply(handler):
        transport = httpx.MockTransport(handler)
        real = httpx.AsyncClient

        class PatchedClient(real):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = transport
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", PatchedClient)

    return _apply


def _reply(content: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}


async def test_ask_returns_the_answer(patch_client) -> None:
    patch_client(lambda request: httpx.Response(200, json=_reply("队伍卡在反序列化范式。")))

    result = await ask(question="进展如何？", dossier="dossier", settings=_settings())

    assert result.ok
    assert "反序列化" in result.answer
    assert result.model


async def test_the_dossier_and_question_both_reach_the_model(patch_client) -> None:
    import json as _json

    seen: list[dict] = []

    def _capture(request):
        seen.append(_json.loads(request.content))
        return httpx.Response(200, json=_reply("ok"))

    patch_client(_capture)

    await ask(question="还有哪些说法没验证？", dossier="MARKER_DOSSIER_TEXT",
              settings=_settings())

    messages = seen[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "MARKER_DOSSIER_TEXT" in messages[-1]["content"]
    assert "还有哪些说法没验证" in messages[-1]["content"]


async def test_history_is_forwarded_but_capped(patch_client) -> None:
    import json as _json

    seen: list[dict] = []

    def _capture(request):
        seen.append(_json.loads(request.content))
        return httpx.Response(200, json=_reply("ok"))

    patch_client(_capture)

    history = [Turn(role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
               for i in range(30)]
    await ask(question="now?", dossier="d", history=history, settings=_settings())

    messages = seen[0]["messages"]
    # system + at most 8 history turns + the current question
    assert len(messages) <= 10
    assert messages[-1]["content"].endswith("now?")


async def test_a_truncated_answer_says_so(patch_client) -> None:
    """The 【建议指令】 block is at the END of the reply, so truncation removes
    precisely the part the operator acts on. A reply that looks complete but is not
    is worse than an error."""
    patch_client(lambda request: httpx.Response(
        200, json=_reply("队伍卡在……", finish="length")))

    result = await ask(question="q", dossier="d", settings=_settings())

    assert result.error and "截断" in result.error
    assert result.answer, "the partial text is still worth showing"


async def test_no_gateway_is_reported_not_silently_empty(patch_client) -> None:
    result = await ask(question="q", dossier="d",
                       settings=_settings(gateway_base_url="", gateway_api_key=""))
    assert not result.ok
    assert "网关未配置" in result.error


async def test_a_gateway_failure_surfaces_the_reason(patch_client) -> None:
    """The operator is staring at a pane waiting for an answer. A blank pane with no
    reason sends them to check the wrong thing."""
    def _boom(request):
        raise httpx.ConnectError("refused")

    patch_client(_boom)

    result = await ask(question="q", dossier="d", settings=_settings())
    assert not result.ok and result.error


async def test_a_timeout_names_the_knob_to_raise(patch_client) -> None:
    """Measured live: reasoning models take 100-150s on a big dossier. A timeout here
    is a configuration fact the operator can act on, not a mystery."""
    def _slow(request):
        raise httpx.ReadTimeout("timed out")

    patch_client(_slow)

    result = await ask(question="q", dossier="d", settings=_settings(briefing_timeout_s=60.0))
    assert not result.ok
    assert "BRIEFING_TIMEOUT_S" in result.error


async def test_an_empty_reply_is_an_error_not_a_blank_answer(patch_client) -> None:
    patch_client(lambda request: httpx.Response(200, json=_reply("")))

    result = await ask(question="q", dossier="d", settings=_settings())
    assert not result.ok and result.error


async def test_the_briefing_runs_on_a_reasoning_model_not_the_summarizer(patch_client) -> None:
    """Judgement over a large dossier, on the same reasoning/summarization split the
    strategist and audit use. A flash model produces a fluent summary that quietly
    flattens the trust tiers — the exact failure the prompt exists to prevent."""
    import json as _json

    seen: list[str] = []

    def _capture(request):
        seen.append(_json.loads(request.content)["model"])
        return httpx.Response(200, json=_reply("ok"))

    patch_client(_capture)

    await ask(question="q", dossier="d", settings=_settings())
    assert seen[0] != "deepseek-v4-flash"


async def test_the_call_is_cost_tracked(patch_client) -> None:
    patch_client(lambda request: httpx.Response(200, json=_reply("ok")))
    recorded: list[tuple] = []
    tracker = SimpleNamespace(record_tokens=lambda *a, **k: recorded.append((a, k)))

    await ask(question="q", dossier="d", settings=_settings(),
              cost_tracker=tracker, agent_name="chal/briefing")

    assert recorded
    args, kwargs = recorded[0]
    assert args[0] == "chal/briefing"
    assert kwargs["provider_spec"] == "gateway"


def test_briefing_ok_requires_an_answer_and_no_error() -> None:
    assert Briefing(answer="x").ok
    assert not Briefing(answer="").ok
    assert not Briefing(answer="x", error="truncated").ok
