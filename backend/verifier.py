"""The Verifier — an active falsification agent, not a second opinion.

`corroborate_facts` withholds trust from a single solver's assertion. That is
necessary but it is not sufficient, and it has two holes that cost solve rate:

  * **It can never refute anything.** A false premise is not labelled trusted,
    but it stays on the board, still readable, still tempting. On the run this was
    built after, "the uploaded JAR does get loaded" was never corroborated and
    never contradicted — it just sat there while four solvers derived payloads
    from it for hours.
  * **It cannot confirm anything either.** A true finding that is hard to
    rediscover stays ⚠ forever and gets under-weighted, which is a false negative
    on the one thing the swarm most needs: a solid fact to build on.

Both holes close the same way — somebody has to actually *run a test*. This module
does that. It owns one sandbox for the whole swarm, reused across claims, and it
runs concurrently with the solvers so verification costs no wall-clock.

The verdict has three values, and the asymmetry is deliberate:

  VERIFIED      → ✅ probe-verified, the strongest tier on the board
  REFUTED       → moved to ✗ excluded AND broadcast, which prunes every dependent
                  line of attack instead of merely declining to trust it
  INCONCLUSIVE  → left as ⚠. This is the default for anything unparseable.

INCONCLUSIVE is free. A wrong VERIFIED makes four agents build on sand; a wrong
REFUTED kills the one line of attack that might have worked. So an unparsable,
crashed, timed-out or ambiguous verification always resolves to INCONCLUSIVE and
never to a verdict with consequences.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"

# A wrong verdict is worse than no verdict, so every failure mode funnels here.
_FALLBACK_EVIDENCE = "verifier produced no parsable verdict"

ROLE_PROMPT = """\
## YOUR ROLE: VERIFIER (blue team) — you do NOT solve this challenge

Other agents are solving it. Your ONLY job is to decide whether specific claims
they made are true, by running the smallest possible test against the real target.
You are handed one claim at a time.

You are this team's immune system against its own hallucinations. A false premise
nobody tests costs every other agent the rest of their budget, because they
inherit it from the shared board and treat it as ground truth.

### Rules

1. Write the MINIMAL test that could FALSIFY the claim. Not the most thorough —
   the minimal. If one HTTP request answers it, send one request.
2. Prefer a **discriminator** over a direct test: arrange two cases that MUST
   differ if the claim is true and MUST look identical if it is false. A test that
   produces the same output either way proves nothing, no matter how elaborate it
   looks. Say explicitly what the two cases were and how their outputs differed.
3. Report only what you **OBSERVED**: a status code, an exception class name, a
   timing delta, a byte count, a diff. Never what you expected, inferred, or what
   "should" happen. If you catch yourself writing "this suggests" or "this
   probably means", stop — that is INCONCLUSIVE.
4. **Absence of an error is not confirmation.** A missing `ClassNotFoundException`
   may mean the class loaded, OR that the code path never ran, OR that the
   exception was swallowed. Prove the path actually executed (a side effect, a
   marker, a positive response) before calling anything VERIFIED.
5. Do NOT solve the challenge. Do NOT hunt for the flag. If you stumble into real
   progress, note it in one line and get back to the claim.
6. Save your probes under `/challenge/workspace/verify/` so the solvers can reuse
   them instead of re-deriving the same measurement.
7. If a claim is not testable by any probe you can run (it is an interpretation, a
   guess about intent, or about something you cannot reach), say so and return
   INCONCLUSIVE immediately. Do not spend the budget dressing up a non-test.

### Output

Work in the sandbox, then END every turn with exactly one line in this form, and
nothing after it:

VERDICT: VERIFIED | <the observable evidence, quoting the actual output>
VERDICT: REFUTED | <the observable evidence that contradicts the claim>
VERDICT: INCONCLUSIVE | <what specifically blocked the test>

When in doubt choose INCONCLUSIVE. It costs nothing. A wrong VERIFIED makes four
agents build on sand; a wrong REFUTED kills the one line of attack that might
have worked.
"""

# Delivered per claim through bump(), not baked into the system prompt: one
# verifier sandbox serves the whole swarm, so the constant part (the rules above)
# belongs in the prompt and the variable part belongs in the turn.
CLAIM_TASK = """\
🔬 **待检验的声称**（由 `{source}` 断言，目前仅有它一家之言）：

> {claim}

{probe_line}

现在去 sandbox 里跑最小证伪探针。记住：
- 只报告你**实际观察到**的输出，不要报告你预期会发生什么。
- 「没有出现报错」不等于「声称成立」——先证明代码路径真的被执行了。
- 结束时只输出一行 `VERDICT: ...`。
"""

# Observed live: a verifier ended its turn mid-exploration ("Let me look at main's
# disassembly") after four tool calls, never emitting a verdict — 74s into a 240s
# budget. Ending a turn with narration instead of a tool call is normal model
# behaviour, and unlike a solver the verifier had no second phase to force it to
# commit. Without this prompt most probes would come back INCONCLUSIVE having done
# the work and then thrown the answer away.
VERDICT_FORCE = """\
⏹ **停止探索。** 不要再调用任何工具，不要再开新的调查方向。

你上一轮在还没有给出裁决时就结束了回合。现在**只基于你已经实际观察到的东西**，
立刻输出裁决，格式严格为一行、后面不要有任何内容：

VERDICT: VERIFIED | <你观察到的证据，引用真实输出>
VERDICT: REFUTED | <与声称矛盾的观察证据>
VERDICT: INCONCLUSIVE | <具体是什么阻碍了判定>

规则：
- 如果你的探针确实跑出了可区分的结果，就据实判 VERIFIED 或 REFUTED，并把关键输出引在证据里。
- 如果你还什么都没真正测到，就判 INCONCLUSIVE 并说明缺什么。**不要为了给出结论而猜。**
- 只输出这一行。
"""


_TRIAGE_PROMPT = (
    "你在为一个 CTF 多智能体系统做**可验证性分诊**。另一个模型对题目声称了下面这句话，"
    "我们要决定是否值得派一个 agent 去真实环境里跑探针检验它。\n\n"
    "【题目】{name}（{category}）\n【题目描述】\n{description}\n\n"
    "【待检验的声称】\n{claim}\n\n"
    "只输出三行，不要解释：\n"
    "TESTABLE: yes 或 no（能否用一个探针在真实目标上观察到真假？纯粹的主观推断/"
    "对出题人意图的猜测/无法触达的东西 = no）\n"
    "LOAD_BEARING: high 或 low（如果它是假的，会不会导致别的 agent 后续大量工作全部作废？"
    "只是某次尝试的中间观察 = low）\n"
    "PROBE: 一句话给出**最小的证伪性测试**——要设计成「声称为真时结果必然不同于声称为假时」，"
    "而不是简单重放一次。无法给出就写 none"
)


@dataclass(frozen=True)
class Verdict:
    """One verification outcome.

    `verdict` is always one of VERIFIED / REFUTED / INCONCLUSIVE. Anything the
    parser could not confidently read becomes INCONCLUSIVE — see the module
    docstring for why that asymmetry is the whole design.

    `explicit` distinguishes "the model chose INCONCLUSIVE" from "the model never
    answered". Only the second is worth a forced-commit retry: re-asking a model
    that deliberately said INCONCLUSIVE just pressures it into guessing, which is
    the one outcome worse than no verdict.
    """

    verdict: str
    evidence: str
    claim: str
    source: str = ""
    explicit: bool = False

    @property
    def is_verdict(self) -> bool:
        """True only for the two outcomes that change the board."""
        return self.verdict in (VERIFIED, REFUTED)

    @property
    def board_text(self) -> str:
        """What goes on the blackboard.

        The original claim is always quoted alongside the evidence. A bare
        "REFUTED" tells a solver nothing about what to stop doing, and a bare
        evidence string gets re-interpreted by whoever reads it.
        """
        marker = {VERIFIED: "✅ 已实证", REFUTED: "⛔ 已证伪"}.get(self.verdict, "❓ 未能判定")
        return f"{marker}：「{self.claim}」— 探针观察：{self.evidence}"


@dataclass
class PendingClaim:
    """A claim waiting for a probe."""

    content: str
    source: str
    key: str
    priority: int = 0        # higher is checked first
    probe_hint: str = ""
    enqueued_at: float = field(default=0.0)


_VERDICT_RE = re.compile(
    r"VERDICT\s*[:：]\s*(VERIFIED|REFUTED|INCONCLUSIVE)\s*[|\-–—:]?\s*(.*)",
    re.IGNORECASE,
)
# The bare form REQUIRES a separator. An optional one let ordinary prose through:
# measured on the live regex, "Verified that the uploaded JAR is loaded by the
# worker." parsed as VERIFIED and "refuted by nothing in particular" parsed as
# REFUTED — both explicit, so both suppressed the forced-commit retry and both went
# straight onto the board, the first at the strongest trust tier. Narration is not a
# verdict, and a separator the prompt always asks for costs nothing to require.
_BARE_VERDICT_RE = re.compile(r"^\s*(VERIFIED|REFUTED|INCONCLUSIVE)\s*[|:]\s*(\S.*)$",
                              re.IGNORECASE)

# Evidence that is really the prompt's own placeholder echoed back. Both ROLE_PROMPT
# and VERDICT_FORCE spell the required format out verbatim, and a model that repeats
# it produced `VERDICT: VERIFIED | <the observable evidence, quoting the actual
# output>` — which parsed as a real verdict with the template as its evidence.
_PLACEHOLDER_MARKERS = (
    "the observable evidence", "what specifically blocked", "quoting the actual output",
    "你观察到的证据", "与声称矛盾的观察证据", "具体是什么阻碍", "一句话",
)
# A verdict whose evidence is shorter than this carries no observable content.
MIN_EVIDENCE_CHARS = 8


def _looks_like_placeholder(evidence: str) -> bool:
    """True when the "evidence" is template text rather than an observation."""
    text = (evidence or "").strip()
    if not text:
        return True
    if text.startswith("<") and text.endswith(">"):
        return True
    low = text.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def parse_verdict(raw: str, claim: str = "", source: str = "") -> Verdict:
    """Extract the verdict line from a verifier's output.

    Scans from the END: the role prompt asks for the verdict as the final line, and
    a verifier that reasons aloud first will have discussed all three options before
    choosing one. Matching the first occurrence would read its deliberation as its
    conclusion.

    Anything ambiguous resolves to INCONCLUSIVE with `explicit=False`, so the caller
    can force a commit. That asymmetry is the whole design: a wrong VERIFIED makes
    four agents build on sand and a wrong REFUTED kills the one line of attack that
    might have worked, while INCONCLUSIVE costs one probe.
    """
    text = (raw or "").strip()
    if not text:
        return Verdict(INCONCLUSIVE, _FALLBACK_EVIDENCE, claim, source)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    window = lines[-15:]
    for offset, line in enumerate(reversed(window)):
        m = _VERDICT_RE.search(line) or _BARE_VERDICT_RE.match(line)
        if not m:
            continue
        verdict = m.group(1).upper()
        evidence = (m.group(2) or "").strip().strip("|-–—: ").strip()
        if not evidence:
            # The evidence may be on the following line — `(.*)` cannot span lines,
            # and a model that puts the marker and its observation on separate lines
            # is still answering. Losing that verdict permanently (the previous
            # behaviour, which also marked it explicit and so suppressed the retry)
            # is worse than looking one line further.
            idx = len(window) - 1 - offset
            following = window[idx + 1:]
            if following and not _VERDICT_RE.search(following[0]):
                evidence = following[0].strip()
        if verdict in (VERIFIED, REFUTED) and (
            len(evidence) < MIN_EVIDENCE_CHARS or _looks_like_placeholder(evidence)
        ):
            # A verdict with no observable evidence is exactly the failure mode this
            # module exists to catch. Do not launder it onto the board — and leave
            # explicit=False so the forced-commit pass gets a chance to supply the
            # observation instead of the claim being lost outright.
            logger.warning(
                "verifier returned %s with unusable evidence (%r) — downgrading",
                verdict, evidence[:80],
            )
            return Verdict(INCONCLUSIVE,
                           f"{verdict} claimed with no observable evidence",
                           claim, source)
        return Verdict(verdict, evidence or _FALLBACK_EVIDENCE, claim, source,
                       explicit=True)

    return Verdict(INCONCLUSIVE, _FALLBACK_EVIDENCE, claim, source, explicit=False)


@dataclass(frozen=True)
class Triage:
    """Whether a claim deserves a probe, and how to test it."""

    testable: bool
    load_bearing: bool
    probe_hint: str

    @property
    def should_verify(self) -> bool:
        return self.testable and self.load_bearing


def parse_triage(raw: str) -> Triage:
    """Read the three-line triage reply. Unparseable → do not spend a probe."""
    text = (raw or "").strip()
    if not text:
        return Triage(testable=False, load_bearing=False, probe_hint="")
    testable = load_bearing = False
    probe = ""
    for line in text.splitlines():
        low = line.lower()
        if "testable" in low:
            testable = "yes" in low.split("testable", 1)[1]
        elif "load_bearing" in low:
            load_bearing = "high" in low.split("load_bearing", 1)[1]
        elif low.startswith("probe"):
            probe = line.split(":", 1)[1].strip() if ":" in line else ""
    if probe.lower() in ("none", "n/a", ""):
        probe = ""
    return Triage(testable=testable, load_bearing=load_bearing, probe_hint=probe)


async def triage_claim(
    *, claim: str, name: str, category: str, description: str,
    settings: object, cost_tracker: object | None = None, agent_name: str = "",
) -> Triage | None:
    """One cheap call deciding whether a probe is worth a sandbox round.

    Returns **None when no answer was obtained** — gateway unreachable, timeout,
    truncation, or an empty reply — as distinct from a Triage that says "not
    testable". That distinction is the entire point: `_call_gateway` swallows every
    failure and returns "", and `parse_triage("")` reads as testable=False, so a
    dead gateway used to look exactly like "the triage model examined every claim
    and declined all of them". Verification was silently off for the rest of the run
    with nothing above debug level in the log.

    Money is not the constraint here — wall-clock and sandbox contention are. A probe
    that cannot falsify anything still occupies the verifier for a whole round while
    solvers wait on the answer, so filtering happens before the expensive part.
    """
    from backend.round_summary import _call_gateway, _model

    prompt = _TRIAGE_PROMPT.format(
        name=name or "?", category=category or "?",
        description=(description or "(none)").strip()[:2000],
        claim=(claim or "").strip()[:1500],
    )
    raw = await _call_gateway(
        prompt, settings, _model(settings), timeout_s=25.0,
        cost_tracker=cost_tracker, agent_name=agent_name,
        # Above the 900 default: a truncated reply is discarded by _call_gateway, so
        # a tight cap here would turn into "no answer" and then into "verify
        # everything" — the safe direction, but wasteful and worth not hitting.
        max_tokens=1500,
    )
    if not raw:
        return None
    return parse_triage(raw)


class ClaimQueue:
    """Bounded, priority-ordered claim queue.

    Bounded on purpose: an unsolvable challenge produces claims faster than one
    verifier can probe them, and an unbounded queue would keep the verifier busy
    long after the swarm retired. Newest claims are more relevant than old ones —
    the old ones have usually already been superseded or excluded.
    """

    def __init__(self, maxlen: int = 40) -> None:
        self._items: deque[PendingClaim] = deque()
        self._keys: set[str] = set()
        self._maxlen = max(1, maxlen)
        self._event = asyncio.Event()

    def __len__(self) -> int:
        return len(self._items)

    def push(self, item: PendingClaim) -> bool:
        """Enqueue. Returns False when the same key is already queued or done."""
        if item.key in self._keys:
            return False
        self._keys.add(item.key)
        self._items.append(item)
        while len(self._items) > self._maxlen:
            dropped = self._items.popleft()
            self._keys.discard(dropped.key)
        self._event.set()
        return True

    def bump_priority(self, key: str) -> None:
        """Raise a claim's priority — used when a contradiction is detected, since
        two agents disagreeing is the strongest possible signal that a probe is
        worth running."""
        for item in self._items:
            if item.key == key:
                item.priority += 10

    async def pop(self, timeout: float | None = None) -> PendingClaim | None:
        """Take the highest-priority claim, waiting for one if the queue is empty."""
        if timeout is not None:
            try:
                await asyncio.wait_for(self._event.wait(), timeout=timeout)
            except TimeoutError:
                return None
        if not self._items:
            self._event.clear()
            return None
        best = max(range(len(self._items)), key=lambda i: self._items[i].priority)
        item = self._items[best]
        del self._items[best]
        if not self._items:
            self._event.clear()
        return item

    def release(self, key: str) -> None:
        """Forget a key so the same claim can be re-queued later."""
        self._keys.discard(key)

    def clear(self) -> None:
        self._items.clear()
        self._keys.clear()
        self._event.clear()


def build_claim_task(claim: str, source: str, probe_hint: str = "") -> str:
    """Render one claim into the turn text handed to the verifier."""
    hint = (probe_hint or "").strip()
    probe_line = (
        f"💡 分诊建议的最小探针：{hint}\n（这只是建议——如果它不能真正区分真假，就自己设计一个能区分的。）"
        if hint and hint.lower() != "none"
        else "（没有现成探针建议，请自己设计一个能区分真假的判别器。）"
    )
    return CLAIM_TASK.format(
        source=source or "unknown",
        claim=(claim or "").strip(),
        probe_line=probe_line,
    )
