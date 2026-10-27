"""Challenge metadata parsing — encoding and field handling.

Regression coverage for a bug that silently lost challenges: `from_yaml` opened
metadata.yml in text mode without an explicit encoding, so it fell back to the
platform locale. On a Windows agent that is cp936/GBK, and every challenge whose
description contained UTF-8 multibyte text failed to parse. Discovery logged a
"bad metadata.yml" warning and moved on, so the run simply covered fewer
challenges than the directory held — the benchmark then reported a solve rate
computed over the wrong denominator.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from backend.offline import discover_challenges
from backend.prompts import ChallengeMeta, _rewrite_connection_info, list_distfiles


def _write(path: Path, text: str) -> Path:
    """Write UTF-8 text to `path`, creating parents. Returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


UTF8_METADATA = """\
name: reg-forensics-magic
category: forensics
value: 100
description: "一张损坏的 PNG —— 修复魔数与 IHDR 后即可读到 flag。"
tags: [png, stego]
connection_info: "nc localhost 31337"
flag: "flag{真实中文_flag}"
"""


def test_utf8_metadata_parses_regardless_of_platform_locale() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td) / "metadata.yml", UTF8_METADATA)
        meta = ChallengeMeta.from_yaml(p)

    assert meta.name == "reg-forensics-magic"
    assert meta.category == "forensics"
    assert "损坏的 PNG" in meta.description
    assert meta.tags == ["png", "stego"]
    # the answer key must survive too, or offline verification silently degrades
    # to "any claimed flag counts"
    assert meta.flags == ["flag{真实中文_flag}"]


def test_utf8_metadata_parses_under_a_forced_non_utf8_locale(monkeypatch) -> None:
    """The explicit encoding must win over whatever the locale says."""
    import builtins

    real_open = builtins.open
    seen: list[str] = []

    def spy(file, mode="r", *args, **kwargs):
        if isinstance(file, (str, Path)) and str(file).endswith("metadata.yml"):
            seen.append(kwargs.get("encoding") or (args[1] if len(args) > 1 else ""))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    # If from_yaml ever drops back to the locale default, this makes it fail
    # loudly instead of depending on the machine it happens to run on.
    monkeypatch.setattr("locale.getpreferredencoding", lambda *_a: "cp936")

    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td) / "metadata.yml", UTF8_METADATA)
        meta = ChallengeMeta.from_yaml(p)

    assert "损坏的 PNG" in meta.description
    assert any(enc == "utf-8" for enc in seen), f"from_yaml did not pass encoding: {seen}"


def test_discovery_does_not_silently_drop_utf8_challenges() -> None:
    """The failure mode that mattered: a whole challenge vanishing from the set."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write(root / "ascii-chal" / "metadata.yml",
               "name: ascii-chal\ncategory: misc\nvalue: 50\ndescription: plain\n")
        _write(root / "utf8-chal" / "metadata.yml", UTF8_METADATA)
        _write(root / "nested" / "deep" / "metadata.yml",
               "name: deep-chal\ncategory: web\nvalue: 200\ndescription: 中文描述\n")

        found = discover_challenges(str(root))

    names = sorted(c.name for c in found)
    assert names == ["ascii-chal", "deep-chal", "reg-forensics-magic"], names


def test_discovery_skips_genuinely_broken_yaml_but_keeps_the_rest() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write(root / "good" / "metadata.yml", "name: good\ncategory: misc\n")
        _write(root / "bad" / "metadata.yml", "name: [unclosed\n  : :\n")

        found = discover_challenges(str(root))

    assert [c.name for c in found] == ["good"]


def test_flag_and_flags_both_accepted() -> None:
    with tempfile.TemporaryDirectory() as td:
        single = _write(Path(td) / "single.yml", 'name: a\nflag: "flag{one}"\n')
        multi = _write(Path(td) / "multi.yml", 'name: b\nflags: ["flag{x}", "flag{y}"]\n')
        none = _write(Path(td) / "none.yml", "name: c\n")

        assert ChallengeMeta.from_yaml(single).flags == ["flag{one}"]
        assert ChallengeMeta.from_yaml(multi).flags == ["flag{x}", "flag{y}"]
        assert ChallengeMeta.from_yaml(none).flags == []


def test_missing_metadata_fields_get_defaults() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td) / "metadata.yml", "name: bare\n")
        meta = ChallengeMeta.from_yaml(p)

    assert meta.category == "" and meta.value == 0
    assert meta.description == "" and meta.tags == []
    assert meta.connection_info == "" and meta.hints == []


def test_empty_file_does_not_crash() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = _write(Path(td) / "metadata.yml", "")
        meta = ChallengeMeta.from_yaml(p)
    assert meta.name == "Unknown"


def test_nonexistent_file_raises_not_silently_passes() -> None:
    with tempfile.TemporaryDirectory() as td, pytest.raises(FileNotFoundError):
        ChallengeMeta.from_yaml(Path(td) / "nope.yml")


# ---------------------------------------------------------------------------
# Prompt-building helpers
# ---------------------------------------------------------------------------

def test_connection_info_is_rewritten_for_the_sandbox() -> None:
    assert _rewrite_connection_info("nc localhost 31337") == "nc host.docker.internal 31337"
    assert _rewrite_connection_info("http://127.0.0.1:8080/") == "http://host.docker.internal:8080/"
    assert _rewrite_connection_info("") == ""
    # a real remote target must be left alone
    assert _rewrite_connection_info("nc ctf.example.com 1337") == "nc ctf.example.com 1337"


def test_list_distfiles_ignores_subdirs_and_missing_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        assert list_distfiles(str(root)) == []
        _write(root / "distfiles" / "b.txt", "x")
        _write(root / "distfiles" / "a.png", "x")
        (root / "distfiles" / "sub").mkdir()
        assert list_distfiles(str(root)) == ["a.png", "b.txt"]


def test_answer_key_is_stripped_from_the_sandbox_mount() -> None:
    """The sanitized metadata that gets bind-mounted must not leak the flag."""
    import yaml

    from backend.sandbox import DockerSandbox

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        meta = _write(root / "metadata.yml", UTF8_METADATA)
        ws = root / "ws"
        ws.mkdir()

        sb = DockerSandbox(image="x", challenge_dir=str(root), workspace_dir=str(ws))
        out = sb._sanitized_metadata(str(meta))

        data = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
        assert "flag" not in data and "flags" not in data
        assert data["name"] == "reg-forensics-magic"
        assert "损坏的 PNG" in data["description"]
