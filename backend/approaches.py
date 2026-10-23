"""Approach playbooks — per-category technique rotation for bump guidance.

"Try something different" is not actionable guidance, especially for weaker
models that otherwise loop on the same wrong idea. This module turns the
challenge's triaged skill packs into an ordered queue of concrete techniques
(one per skill file). The swarm assigns the NEXT untried technique on every
bump, so retries systematically cover the technique space instead of repeating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Approach:
    """One candidate technique, backed by a skill-library file."""
    path: str      # sandbox path, e.g. /skills/ctf-pwn/heap-techniques.md
    title: str     # human-readable technique name
    pack: str      # owning pack id, e.g. ctf-pwn


def _title_of(md_path: Path) -> str:
    """First markdown H1/H2 heading, else the filename."""
    try:
        for line in md_path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]:
            m = re.match(r"^#{1,2}\s+(.*)", line.strip())
            if m:
                return m.group(1).strip()[:90]
    except Exception:
        pass
    return md_path.stem


def build_approach_queue(
    pack_ids: list[str],
    skills_dir: str,
    sandbox_root: str = "/skills",
    max_per_pack: int = 12,
) -> list[Approach]:
    """Build an ordered queue of candidate techniques from the triaged packs.

    Each *.md theme file in a pack (other than INDEX.md) is one approach.
    Order: packs in triage order, files sorted (stable, deterministic).
    """
    queue: list[Approach] = []
    seen: set[str] = set()
    root = Path(skills_dir)
    for pack_id in pack_ids:
        pack_dir = root / pack_id
        try:
            files = sorted(f for f in pack_dir.glob("*.md") if f.name != "INDEX.md")
        except OSError:
            continue
        for f in files[:max_per_pack]:
            rel = f"{sandbox_root}/{pack_id}/{f.name}"
            if rel in seen:
                continue
            seen.add(rel)
            queue.append(Approach(path=rel, title=_title_of(f), pack=pack_id))
    return queue
