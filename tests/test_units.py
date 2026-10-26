"""Stdlib-only unit tests for the new core modules.

Runs with plain `python tests/test_units.py` (no pytest required) and is also
pytest-compatible. Heavy deps (pydantic-ai, aiodocker) are stubbed out.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import types
from pathlib import Path

# Stub aiodocker before importing backend.sandbox (not installed locally).
sys.modules.setdefault("aiodocker", types.ModuleType("aiodocker"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.message_bus import ChallengeMessageBus  # noqa: E402
from backend.sandbox import to_host_path  # noqa: E402
from backend.skills import SkillRouter  # noqa: E402


# ---------------------------------------------------------------------------
# SkillRouter
# ---------------------------------------------------------------------------

def _make_fake_skills(root: Path) -> Path:
    pwn = root / "ctf-pwn"
    pwn.mkdir(parents=True)
    (pwn / "INDEX.md").write_text(
        "---\nname: ctf-pwn\ndescription: heap, stack, rop\n---\n"
        "# ctf-pwn\n\n## Additional Resources\n"
        "- `heap-techniques.md` — House of Orange, tcache poisoning\n"
        "- `stack-basics.md` — ret2libc, canary\n",
        encoding="utf-8",
    )
    (pwn / "heap-techniques.md").write_text("# heap\n", encoding="utf-8")
    (pwn / "stack-basics.md").write_text("# stack\n", encoding="utf-8")

    web = root / "ctf-web"
    web.mkdir(parents=True)
    (web / "INDEX.md").write_text("# ctf-web\n\n- `sqli.md` — union, blind\n", encoding="utf-8")
    (web / "sqli.md").write_text("# sqli\n", encoding="utf-8")

    wp = root / "zh-knowledge" / "CTF大赛WP集合" / "articles"
    wp.mkdir(parents=True)
    (wp / "heap-tcache-baby.meta.md").write_text(
        "title: baby tcache\nvuln_type: heap\n"
        "tags: [tcache, heap]\none_liner: tcache dup 劫持 free hook\n",
        encoding="utf-8",
    )
    (wp / "heap-tcache-baby.md").write_text("# writeup\n", encoding="utf-8")
    return root


def test_triage_pwn() -> None:
    with tempfile.TemporaryDirectory() as td:
        router = SkillRouter(skills_dir=str(_make_fake_skills(Path(td))))
        assert router.enabled
        packs = router.triage(
            category="pwn", description="heap exploit with tcache", tags=[],
            distfiles=["chal"],
        )
        assert packs and packs[0] == "ctf-pwn", packs


def test_triage_by_extension() -> None:
    with tempfile.TemporaryDirectory() as td:
        router = SkillRouter(skills_dir=str(_make_fake_skills(Path(td))))
        packs = router.triage(category="", description="", distfiles=["capture.pcapng"])
        assert "ctf-forensics" in packs, packs


def test_triage_disabled_without_packs() -> None:
    with tempfile.TemporaryDirectory() as td:
        router = SkillRouter(skills_dir=td)
        assert not router.enabled
        assert router.triage(category="pwn") == []
        assert router.render_hint(["ctf-pwn"]) == ""


def test_render_hint_levels() -> None:
    with tempfile.TemporaryDirectory() as td:
        router = SkillRouter(skills_dir=str(_make_fake_skills(Path(td))))
        h1 = router.render_hint(["ctf-pwn"], level=1)
        # L1 is a SHORT pointer (no full content) — progressive disclosure
        assert "/skills/ctf-pwn" in h1
        assert "heap-techniques.md" not in h1  # content is withheld until L2+

        h2 = router.render_hint(["ctf-pwn"], level=2)
        assert "Full file list" in h2

        h3 = router.render_hint(["ctf-pwn"], level=3, keywords=["tcache", "heap"])
        assert "heap-tcache-baby.md" in h3, h3  # writeup match points at article
        assert ".meta.md" not in h3


def test_render_strips_frontmatter() -> None:
    with tempfile.TemporaryDirectory() as td:
        router = SkillRouter(skills_dir=str(_make_fake_skills(Path(td))))
        hint = router.render_hint(["ctf-pwn"], level=1)
        assert "description: heap, stack, rop" not in hint


# ---------------------------------------------------------------------------
# Message bus (structured findings)
# ---------------------------------------------------------------------------

def test_bus_grouping() -> None:
    async def _run() -> None:
        bus = ChallengeMessageBus()
        await bus.post("model-a", "ruled out format string — no write primitive", kind="excluded")
        await bus.post("model-a", "libc base at 0x7f...", kind="fact")
        await bus.broadcast("flag format is flag{...}")

        unread = await bus.check("model-b")
        assert len(unread) == 3  # model-b sees everything from model-a + broadcast
        text = bus.format_unread(unread)
        assert "Dead ends" in text and "Confirmed facts" in text and "Coordinator" in text
        # cursor advanced
        assert await bus.check("model-b") == []
        # model-a does not see its own posts, only the broadcast
        own = await bus.check("model-a")
        assert len(own) == 1 and own[0].kind == "broadcast"

    asyncio.run(_run())


def test_patrol_trace_activity() -> None:
    """trace_activity parses a solver trace for last-ts + repeated signatures."""
    import json
    import time as _time

    from backend.trace_read import trace_activity

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.jsonl"
        events = []
        for i in range(8):
            events.append({"ts": _time.time() - 100 + i, "type": "tool_call",
                           "tool": "bash", "args": {"command": "id"}})
        events.append({"ts": _time.time(), "type": "tool_call",
                       "tool": "bash", "args": {"command": "ls"}})
        p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

        last_ts, sigs = trace_activity(str(p))
        assert last_ts > 0
        assert len(sigs) == 9
        # 8 identical 'bash:{"command": "id"}' signatures → looping detected shape
        assert sigs.count(sigs[0]) == 8
        # empty / missing file → safe defaults
        assert trace_activity(str(Path(td) / "nope.jsonl")) == (0.0, [])


def test_approach_queue() -> None:
    """build_approach_queue turns skill packs into a deterministic technique queue."""
    import tempfile

    from backend.approaches import build_approach_queue

    with tempfile.TemporaryDirectory() as td:
        pack = Path(td) / "ctf-pwn"
        pack.mkdir(parents=True)
        (pack / "INDEX.md").write_text("# index", encoding="utf-8")
        (pack / "heap-techniques.md").write_text("# Heap Techniques\n...", encoding="utf-8")
        (pack / "format-string.md").write_text("# Format String\n...", encoding="utf-8")
        (pack / "rop.md").write_text("# ROP\n...", encoding="utf-8")

        q = build_approach_queue(["ctf-pwn"], td)
        assert len(q) == 3
        assert q[0].path == "/skills/ctf-pwn/format-string.md"  # sorted
        assert q[0].title == "Format String"
        paths = [a.path for a in q]
        assert len(set(paths)) == 3  # unique


def test_swarm_summary_shared_feed() -> None:
    """_swarm_summary exposes the message_bus findings as shared_feed entries."""
    import asyncio
    from types import SimpleNamespace

    from backend.message_bus import ChallengeMessageBus
    from backend.web_dashboard import DashboardServer

    bus = ChallengeMessageBus()
    asyncio.run(bus.post("m1", "found weak CRC32", kind="fact"))
    asyncio.run(bus.post("m2", "format string ruled out", kind="excluded"))
    asyncio.run(bus.broadcast("coordinator hint"))

    swarm = SimpleNamespace(
        meta=SimpleNamespace(category="crypto", value=100),
        model_specs=["m1"],
        solvers={},
        cancel_event=asyncio.Event(),
        winner=None, winner_spec="",
        _total_bumps=0, _failed_approaches=[],
        findings={},
        message_bus=bus,
    )
    deps = SimpleNamespace(cost_tracker=None)
    srv = DashboardServer(deps)
    out = srv._swarm_summary("chal", swarm)
    feed = out["shared_feed"]
    assert len(feed) == 3, feed
    assert feed[0]["kind"] == "fact" and "CRC32" in feed[0]["content"]
    assert feed[1]["kind"] == "excluded"
    assert feed[2]["kind"] == "broadcast"
    assert feed[0]["model"] == "m1"


def test_advice_mailbox() -> None:
    """Coordinator→solver advice is queued per-solver and drained once."""
    async def _run() -> None:
        bus = ChallengeMessageBus()
        await bus.post_advice("model-a", "try the tcache poison")
        await bus.post_advice("model-a", "and check the GOT")
        await bus.post_advice("model-b", "unrelated")

        # model-a drains both, once
        got = await bus.check_advice("model-a")
        assert got == ["try the tcache poison", "and check the GOT"], got
        # drained — empty on second check
        assert await bus.check_advice("model-a") == []
        # model-b unaffected
        assert await bus.check_advice("model-b") == ["unrelated"]

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# DooD path translation
# ---------------------------------------------------------------------------

def test_host_path_translation() -> None:
    mapping = ("/app", "D:/code/ctf/proj")
    assert to_host_path("/app/challenges/x/distfiles", mapping) == \
        "D:/code/ctf/proj/challenges/x/distfiles"
    assert to_host_path("/etc/passwd", mapping) == "/etc/passwd"
    assert to_host_path("/app/x", None) == "/app/x"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
