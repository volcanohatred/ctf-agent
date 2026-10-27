"""Approach playbooks — per-category technique rotation for bump guidance.

"Try something different" is not actionable guidance, especially for weaker
models that otherwise loop on the same wrong idea. This module turns the
challenge's triaged skill packs into an ordered queue of concrete techniques
(one per skill file). The swarm assigns the NEXT untried technique on every
bump, so retries systematically cover the technique space instead of repeating.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Titles are read out of every candidate handbook, and the queue is rebuilt for
# each swarm. Cache per resolved file path so N concurrent challenges do not
# each re-read the same library.
_TITLE_CACHE: dict[str, str] = {}


def clear_title_cache() -> None:
    """Drop cached handbook titles (tests / hot reload)."""
    _TITLE_CACHE.clear()


@dataclass(frozen=True)
class Approach:
    """One candidate technique, backed by a skill-library file."""

    path: str      # sandbox path, e.g. /skills/ctf-pwn/heap-techniques.md
    title: str     # human-readable technique name
    pack: str      # owning pack id, e.g. ctf-pwn


def _title_of(md_path: Path) -> str:
    """First markdown H1/H2 heading, else the filename."""
    key = str(md_path)
    cached = _TITLE_CACHE.get(key)
    if cached is not None:
        return cached
    title = md_path.stem
    try:
        with md_path.open(encoding="utf-8", errors="replace") as fh:
            for _ in range(40):
                line = fh.readline()
                if not line:
                    break
                m = re.match(r"^#{1,2}\s+(.*)", line.strip())
                if m:
                    title = m.group(1).strip()[:90]
                    break
    except OSError as e:
        logger.debug("Could not read title from %s: %s", md_path, e)
    _TITLE_CACHE[key] = title
    return title


def build_approach_queue(
    pack_ids: list[str],
    skills_dir: str,
    sandbox_root: str = "/skills",
    max_per_pack: int = 12,
) -> list[Approach]:
    """Build an ordered queue of candidate techniques from the triaged packs.

    Each *.md theme file in a pack (other than INDEX.md) is one approach.
    Order: packs in triage order, files sorted (stable, deterministic).

    Returns [] when no agent-visible skills_dir is configured — the rotation
    then degrades to plain bump text instead of pointing at files that do not
    exist. (In docker-outside-of-docker setups SKILLS_DIR must be the path as
    seen by the AGENT container; SKILLS_HOST_DIR alone is not enough.)
    """
    queue: list[Approach] = []
    if not skills_dir or not pack_ids:
        return queue
    seen: set[str] = set()
    root = Path(skills_dir)
    if not root.is_dir():
        logger.warning("Approach queue: skills_dir %s is not a directory", skills_dir)
        return queue
    for pack_id in pack_ids:
        pack_dir = root / pack_id
        try:
            files = sorted(f for f in pack_dir.glob("*.md") if f.name != "INDEX.md")
        except OSError as e:
            logger.debug("Could not list pack %s: %s", pack_dir, e)
            continue
        for f in files[:max_per_pack]:
            rel = f"{sandbox_root}/{pack_id}/{f.name}"
            if rel in seen:
                continue
            seen.add(rel)
            queue.append(Approach(path=rel, title=_title_of(f), pack=pack_id))
    return queue
