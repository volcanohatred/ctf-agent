"""Skill library router — triage a challenge to knowledge packs and render hints.

Pure stdlib (no pydantic/httpx) so it is cheap to import and easy to unit-test.

Design (see project docs):
- The skill library lives OUTSIDE the prompt. It is bind-mounted read-only at
  /skills/ inside every solver sandbox, and the agent reads it lazily.
- Level 0 (default): solvers work with raw model capability only.
- Level 1: inject the matching pack's INDEX.md excerpt (file menu + one-line
  summaries). The solver then reads the specific file itself.
- Level 2: additionally list every theme file in the pack (full menu).
- Level 3: additionally grep the Chinese writeup metadata (zh-knowledge
  .meta.md files) for similar historical challenges and point at them.

Escalation is driven by ChallengeSwarm: bump_count maps to levels.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Where the library is mounted inside solver sandboxes (read-only).
SANDBOX_SKILLS_ROOT = "/skills"

MAX_INDEX_CHARS = 2600
MAX_WP_RESULTS = 3

# ---------------------------------------------------------------------------
# Pack registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pack:
    pack_id: str            # directory name under the skills root
    categories: tuple[str, ...]   # CTFd category names (lowercased) this pack serves
    keywords: tuple[str, ...]     # description/tag keyword triggers (lowercased)


PACKS: tuple[Pack, ...] = (
    Pack("ctf-web", ("web", "websec"), (
        "sqli", "sql injection", "ssti", "ssrf", "xss", "xxe", "jwt", "oauth",
        "csrf", "upload", "lfi", "rfi", "deserialization", "pickle", "prototype pollution",
        "request smuggling", "saml", "solidity", "web3", "cookie", "session", "flask",
        "django", "php", "java", "nodejs", "express", "http",
    )),
    Pack("ctf-pwn", ("pwn", "binary", "pwnable", "exploitation"), (
        "heap", "stack", "buffer overflow", "bof", "tcache", "fastbin", "unsorted",
        "house of", "rop", "ret2libc", "srop", "format string", "fmtstr", "shellcode",
        "seccomp", "kernel", "kaslr", "smep", "kpti", "pivot", "one_gadget", "libc",
        "got", "plt", "canary", "pie", "nx", "aslr", "sandbox escape", "vm escape",
        "use after free", "uaf", "double free", "oob",
    )),
    Pack("ctf-reverse", ("reverse", "reversing", "re", "rev"), (
        "ghidra", "ida", "decompile", "disassemble", "crackme", "keygen", "obfuscat",
        "anti-debug", "anti debug", "mba", "vmprotect", "apk", "android", "wasm",
        "firmware", "frida", "unicorn", "angr", "dotnet", ".net", "golang", "rust",
        "packed", "upx", "hook",
    )),
    Pack("ctf-crypto", ("crypto", "cryptography"), (
        "rsa", "aes", "ecc", "elliptic", "dh", "diffie", "lattice", "lwe", "coppersmith",
        "padding oracle", "cbc", "ecb", "nonce", "prng", "random", "xor", "cipher",
        "encrypt", "decrypt", "hash", "md5", "sha", "fermat", "wiener", "hastad",
        "crt", "discrete log", "post-quantum", "zkp", "zero knowledge",
    )),
    Pack("ctf-forensics", ("forensics", "forensic", "dfir", "stego", "steganography"), (
        "pcap", "pcapng", "capture", "wireshark", "memory dump", "memdump", "volatility",
        "disk image", "filesystem", "deleted", "recover", "carve", "steg", "lsb",
        "exif", "metadata", "hidden", "audio", "spectrogram", "signal", "rf", "sdr",
        "windows registry", "evtx", "logs", "usb",
    )),
    Pack("ctf-misc", ("misc", "miscellaneous", "jail"), (
        "pyjail", "python jail", "jail", "sandbox", "bashjail", "encoding", "base64",
        "qr code", "esolang", "brainfuck", "game", "vm", "dns", "privesc", "linux",
        "osint",
    )),
    Pack("ctf-osint", ("osint",), (
        "geolocation", "social media", "username", "email", "doxx", "satellite",
        "maps", "image search",
    )),
    Pack("ctf-malware", ("malware",), (
        "malware", "ransomware", "c2", "beacon", "rat", "trojan", "pe ", "shellcode",
        "cobalt", "emotet", "packed",
    )),
    Pack("ctf-ai-ml", ("ai", "ml", "ai/ml", "llm"), (
        "model", "pytorch", "tensorflow", "safetensors", "lora", "adversarial",
        "fgsm", "pgd", "prompt injection", "jailbreak", "neural", "mnist", "onnx",
    )),
)

# File-extension → pack signals (weight 2 each)
_EXT_SIGNALS: dict[str, str] = {
    ".pcap": "ctf-forensics", ".pcapng": "ctf-forensics",
    ".img": "ctf-forensics", ".dd": "ctf-forensics", ".e01": "ctf-forensics",
    ".dmp": "ctf-forensics", ".vmem": "ctf-forensics", ".raw": "ctf-forensics",
    ".wav": "ctf-forensics", ".mp3": "ctf-forensics", ".flac": "ctf-forensics",
    ".apk": "ctf-reverse", ".dex": "ctf-reverse",
    ".wasm": "ctf-reverse", ".pyc": "ctf-reverse",
    ".exe": "ctf-reverse", ".dll": "ctf-reverse", ".so": "ctf-reverse",
    ".sol": "ctf-web",
    ".sage": "ctf-crypto", ".ipynb": "ctf-crypto",
    ".pem": "ctf-crypto", ".pub": "ctf-crypto",
    ".safetensors": "ctf-ai-ml", ".pt": "ctf-ai-ml", ".onnx": "ctf-ai-ml",
    ".h5": "ctf-ai-ml", ".pkl": "ctf-ai-ml",
}

# Image files → steganography is very common (forensics pack covers it)
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1)


@dataclass
class SkillRouter:
    """Triage challenges to skill packs and render escalating hints."""

    skills_dir: str = ""
    max_packs: int = 2
    _enabled: bool = field(init=False, default=False)
    _root: Path | None = field(init=False, default=None)
    _index_cache: dict[str, str] = field(init=False, default_factory=dict)
    # Lazy: list of (relpath, one_liner, searchable_text) for zh-knowledge WPs
    _wp_meta_cache: list[tuple[str, str, str]] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.skills_dir:
            root = Path(self.skills_dir)
            if root.is_dir() and any(root.glob("ctf-*/INDEX.md")):
                self._root = root
                self._enabled = True
            else:
                logger.warning("Skills dir %s has no ctf-*/INDEX.md packs — disabled", self.skills_dir)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Triage
    # ------------------------------------------------------------------

    def triage(
        self,
        category: str = "",
        description: str = "",
        tags: list[str] | None = None,
        distfiles: list[str] | None = None,
    ) -> list[str]:
        """Return pack ids ordered by relevance (best first)."""
        if not self._enabled:
            return []

        scores: dict[str, int] = {}
        cat = (category or "").strip().lower()

        for pack in PACKS:
            score = 0
            if cat and any(c in cat or cat in c for c in pack.categories if c):
                score += 3
            haystack = " ".join([description or "", *(tags or [])]).lower()
            if haystack:
                hits = sum(1 for kw in pack.keywords if kw in haystack)
                score += min(hits, 4)  # cap keyword contribution
            if score:
                scores[pack.pack_id] = score

        for fname in distfiles or []:
            ext = Path(fname).suffix.lower()
            pack_id = _EXT_SIGNALS.get(ext)
            if pack_id:
                scores[pack_id] = scores.get(pack_id, 0) + 2
            elif ext in _IMAGE_EXTS:
                scores["ctf-forensics"] = scores.get("ctf-forensics", 0) + 1

        if not scores:
            # Unknown challenge — default to misc so the solver has *some* map.
            return ["ctf-misc"]

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [pid for pid, _ in ranked[: self.max_packs]]

    # ------------------------------------------------------------------
    # Hint rendering
    # ------------------------------------------------------------------

    def render_hint(
        self,
        pack_ids: list[str],
        level: int = 1,
        keywords: list[str] | None = None,
    ) -> str:
        """Render an escalating skill hint for injection into a solver bump."""
        if not self._enabled or not pack_ids:
            return ""

        parts: list[str] = [
            f"KNOWLEDGE BASE (escalation level {level}):",
            f"A CTF technique library is mounted read-only at `{SANDBOX_SKILLS_ROOT}/` "
            "inside your container. Your current direction has not worked — "
            "pick ONE file below that matches your situation, read it with bash "
            f"(e.g. `sed -n '1,160p' {SANDBOX_SKILLS_ROOT}/<pack>/<file>`), and apply a "
            "CONCRETE technique from it. Do not skim multiple files; commit to one.",
            "",
        ]

        for pid in pack_ids:
            excerpt = self._index_excerpt(pid)
            if excerpt:
                parts.append(f"=== Pack index: {SANDBOX_SKILLS_ROOT}/{pid}/INDEX.md ===")
                parts.append(excerpt)
                parts.append("")

        if level >= 2:
            for pid in pack_ids:
                files = self._pack_files(pid)
                if files:
                    parts.append(
                        f"Full file list for {pid}: "
                        + ", ".join(f"{SANDBOX_SKILLS_ROOT}/{pid}/{f}" for f in files)
                    )
            parts.append("")

        if level >= 3:
            wps = self._search_writeups(keywords or [])
            if wps:
                parts.append("Similar historical writeups (read these article files):")
                parts.extend(f"- {SANDBOX_SKILLS_ROOT}/{p} — {one_liner}" for p, one_liner in wps)
                parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _index_excerpt(self, pack_id: str) -> str:
        if pack_id in self._index_cache:
            return self._index_cache[pack_id]
        assert self._root is not None
        path = self._root / pack_id / "INDEX.md"
        text = ""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            text = _strip_frontmatter(raw).strip()
            if len(text) > MAX_INDEX_CHARS:
                text = text[:MAX_INDEX_CHARS] + "\n... [truncated — read the full INDEX.md in the sandbox]"
        except FileNotFoundError:
            logger.warning("Pack index missing: %s", path)
        self._index_cache[pack_id] = text
        return text

    def _pack_files(self, pack_id: str) -> list[str]:
        assert self._root is not None
        pack_dir = self._root / pack_id
        try:
            return sorted(
                f.name for f in pack_dir.glob("*.md") if f.name != "INDEX.md"
            )
        except OSError:
            return []

    def _load_wp_meta(self) -> list[tuple[str, str, str]]:
        """Lazily scan zh-knowledge writeup .meta.md files.

        Returns list of (relpath, one_liner, searchable_text).
        """
        if self._wp_meta_cache is not None:
            return self._wp_meta_cache
        assert self._root is not None
        results: list[tuple[str, str, str]] = []
        wp_root = self._root / "zh-knowledge"
        try:
            metas = list(wp_root.rglob("*.meta.md"))
        except OSError:
            metas = []
        for meta_path in metas:
            try:
                text = meta_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            one_liner = ""
            m = re.search(r"^one_liner:\s*(.+)$", text, re.MULTILINE)
            if m:
                one_liner = m.group(1).strip().strip("\"'")
            if not one_liner:
                m = re.search(r"^title:\s*(.+)$", text, re.MULTILINE)
                one_liner = m.group(1).strip().strip("\"'") if m else ""
            rel = meta_path.relative_to(self._root).as_posix()
            results.append((rel, one_liner, text))
        self._wp_meta_cache = results
        return results

    def _search_writeups(self, keywords: list[str]) -> list[tuple[str, str]]:
        """Keyword-match Chinese writeup metadata. Returns (article_relpath, one_liner)."""
        kws = [k.lower() for k in keywords if len(k) >= 3][:8]
        if not kws:
            return []
        hits: list[tuple[int, str, str]] = []
        for rel, one_liner, text in self._load_wp_meta():
            hay = text.lower()
            score = sum(1 for k in kws if k in hay)
            if score >= 2:
                hits.append((score, rel, one_liner))
        hits.sort(key=lambda h: h[0], reverse=True)
        # Point at the .md (article), not the .meta.md
        return [
            (rel[: -len(".meta.md")] + ".md", one_liner)
            for _, rel, one_liner in hits[:MAX_WP_RESULTS]
        ]
