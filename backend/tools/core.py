"""SDK-agnostic tool logic — pure async functions, no Pydantic AI types."""

from __future__ import annotations

import ipaddress
import json
import shlex
from collections.abc import Sequence
from pathlib import Path

import httpx

MAX_OUTPUT = 24_000


def _truncate(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    lines = text.split("\n")
    head = "\n".join(lines[:200])
    return head[:limit] + f"\n... [truncated — {len(text)} total chars, {len(lines)} lines]"


async def do_bash(sandbox, command: str, timeout_seconds: int = 60) -> str:
    result = await sandbox.exec(command, timeout_s=timeout_seconds)
    parts: list[str] = []
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr}")
    if result.exit_code != 0:
        parts.append(f"[exit {result.exit_code}]")
    out = "\n".join(parts).strip() or "(no output)"
    return _truncate(out)


async def do_read_file(sandbox, path: str) -> str:
    try:
        data = await sandbox.read_file(path)
    except Exception as e:
        return f"Error reading file: {e}"

    if isinstance(data, bytes):
        sample = data[:4096]
        non_text = sum(
            1
            for b in sample
            if b == 0 or (b < 9 and b not in (7, 8)) or (9 < b < 13) or (13 < b < 32 and b != 27)
        )
        if len(sample) > 0 and non_text / len(sample) > 0.05:
            return (
                f"Binary file ({len(data)} bytes) — use bash to inspect it:\n"
                f"  file {path}\n"
                f"  xxd {path} | head -40\n"
                f"  strings {path}\n"
                f"  exiftool {path}\n"
                f"  binwalk {path}"
            )
        return _truncate(data.decode("utf-8", errors="replace"))

    return _truncate(data) if isinstance(data, str) else data


async def do_write_file(sandbox, path: str, content: str) -> str:
    try:
        await sandbox.write_file(path, content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


async def do_list_files(sandbox, path: str = "/challenge/distfiles") -> str:
    result = await sandbox.exec(f"ls -la {shlex.quote(path)}")
    out = result.stdout.strip()
    if result.exit_code != 0:
        return result.stderr.strip() or f"Error listing {path}"
    return out or f"{path} is empty."


class FlagSubmissionError(RuntimeError):
    """The submission never reached CTFd, so the flag was never judged.

    Distinct from an incorrect flag on purpose. `do_submit_flag` used to return
    `("submit_flag error: ...", False)` for both, and `try_submit_flag` could only
    see the False — so a CTFd outage incremented the solver's wrong-submission count
    and walked it up SUBMISSION_COOLDOWNS to a 10-minute lockout. The model was
    punished for the network, and the operator saw nothing because `_submit_count`
    is not in get_status().
    """


async def do_submit_flag(ctfd, challenge_name: str, flag: str) -> tuple[str, bool]:
    """Submit a flag. Returns (display_message, is_confirmed).

    Raises FlagSubmissionError when the submission could not be delivered, so callers
    can avoid penalising the solver for an outage. An incorrect flag is a normal
    (message, False) return.
    """
    flag = flag.strip()
    if not flag:
        return "Empty flag — nothing to submit.", False

    try:
        result = await ctfd.submit_flag(challenge_name, flag)
    except Exception as e:  # noqa: BLE001 — surfaced as a distinct outcome, not a wrong answer
        raise FlagSubmissionError(f"{type(e).__name__}: {e}") from e
    is_confirmed = result.status in ("correct", "already_solved")
    return result.display, is_confirmed


def _ip_is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        # Docker's bridge gateway and the metadata endpoints solvers should not
        # be able to reach from the agent process (which holds the docker socket).
        or (ip.version == 4 and ip.packed[0] == 169 and ip.packed[1] == 254)
    )


def resolve_target(url: str, allow_hosts: str | Sequence[str] = ()) -> tuple[bool, str, list[str]]:
    """Decide whether `url` may be fetched from the AGENT process.

    Returns (allowed, reason, resolved_ips).

    An earlier revision matched the hostname against a list of string prefixes,
    which `http://2130706433:2375/` (decimal IP), `http://0x7f.1/`, IPv6
    literals and any DNS name pointing at the host all walked straight past.
    Since the agent container holds /var/run/docker.sock, that was a path from a
    solver to the Docker API. Resolution is done here and EVERY returned address
    must be public — a name that resolves to both a public and a private IP is
    rejected, so a multi-record answer cannot smuggle one through.

    `allow_hosts` is an explicit operator-controlled exception list. Challenge
    services are normally reached from the agent container as
    `host.docker.internal`, which resolves to a private gateway address, so
    without a narrow allowlist this tool would block its own primary use case.
    The allowlist is matched on the HOSTNAME, not the resolved IP, so it cannot
    be widened by DNS.

    Residual risk, stated plainly: this is still a check-then-connect, so a
    deliberately rebinding DNS server can change its answer in between. Closing
    that needs connection pinning, which is out of scope for a CTF utility.
    Redirects are NOT followed, so a public URL cannot bounce to an internal one.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme or '(none)'}' is not allowed", []
    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        return False, "no host in URL", []

    allowed_names = {str(h).strip().rstrip(".").lower() for h in _as_seq(allow_hosts) if str(h).strip()}
    host_allowed = host in allowed_names

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return False, f"DNS resolution failed for {host}: {e}", []

    ips: list[str] = []
    for info in infos:
        # getaddrinfo's sockaddr[0] is typed str|int by typeshed; coerce so the
        # IPv6 zone-id strip ("fe80::1%eth0") is always valid.
        raw = str(info[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(raw)
        except ValueError:
            return False, f"unparsable address {raw!r} for {host}", []
        ips.append(str(ip))
        if host_allowed:
            continue
        if _ip_is_forbidden(ip):
            return False, f"{host} resolves to a non-public address ({ip})", ips

    if not ips:
        return False, f"{host} resolved to no addresses", []
    return True, "", ips


def _as_seq(value) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [p for p in value.split(",") if p.strip()]
    return list(value)


def _is_internal_url(url: str, allow_hosts=()) -> bool:
    """Back-compat predicate: True when the URL must not be fetched."""
    allowed, _reason, _ips = resolve_target(url, allow_hosts)
    return not allowed


async def do_web_fetch(url: str, method: str = "GET", body: str = "",
                       allow_hosts=()) -> str:
    allowed, reason, _ips = resolve_target(url, allow_hosts)
    if not allowed:
        return f"Fetch error: blocked — {reason}"
    try:
        # verify=False: CTF challenge services often use self-signed certs.
        # follow_redirects stays False so an allowed public URL cannot bounce the
        # request to an internal address after the check above.
        async with httpx.AsyncClient(verify=False, timeout=30.0,
                                     follow_redirects=False) as client:
            resp = await client.request(
                method,
                url,
                content=body or None,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            text = resp.text
            prefix = f"HTTP {resp.status_code} {resp.reason_phrase}\n{'─' * 40}\n"
            if resp.is_redirect:
                return (prefix + "[redirect not followed — re-issue the request with "
                        f"the Location URL: {resp.headers.get('location', '?')}]")
            if len(text) > 20_000:
                text = text[:20_000] + f"\n... [truncated, total {len(resp.text)} bytes]"
            return prefix + text
    except Exception as e:  # noqa: BLE001 — every failure is reported to the model
        return f"Fetch error: {e}"


async def do_webhook_create() -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post("https://webhook.site/token")
            if resp.status_code != 200:
                return f"webhook.site error: HTTP {resp.status_code}"
            data = resp.json()
            return json.dumps({"uuid": data["uuid"], "url": f"https://webhook.site/{data['uuid']}"})
    except Exception as e:
        return f"webhook_create error: {e}"


async def do_webhook_get_requests(uuid: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"https://webhook.site/token/{uuid}/requests")
            if resp.status_code != 200:
                return f"webhook.site error: HTTP {resp.status_code}"
            data = resp.json()
            if not data.get("data"):
                return "No requests received yet."
            out = json.dumps(data["data"], indent=2)
            return out[:8000] if len(out) > 8000 else out
    except Exception as e:
        return f"webhook_get_requests error: {e}"


async def do_check_findings(message_bus, model_spec: str) -> str:
    """Get unread findings from sibling solvers."""
    if not message_bus:
        return "No message bus available."
    findings = await message_bus.check(model_spec)
    if not findings:
        return "No new findings from other agents."
    return message_bus.format_unread(findings)


async def do_request_help(message_bus, model_spec: str, question: str,
                          notify_coordinator=None) -> str:
    """A solver raises a knowledge-gap help request for the human operator."""
    if not message_bus:
        return "No help channel available."
    await message_bus.post_help(model_spec, question)
    if notify_coordinator:
        try:
            await notify_coordinator(f"🆘 HELP REQUEST: {question[:300]}")
        except Exception:
            pass
    return "Help request recorded — a human operator will see it and may reply via the mailbox. Keep working other angles meanwhile."


# Image constants (shared with vision wrapper)
IMAGE_EXTS_FOR_VISION: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".webp": "image/webp",
}

IMAGE_MAGIC: dict[str, list[int]] = {
    "image/png": [0x89, 0x50, 0x4E, 0x47],
    "image/jpeg": [0xFF, 0xD8, 0xFF],
    "image/gif": [0x47, 0x49, 0x46],
    "image/bmp": [0x42, 0x4D],
    "image/webp": [0x52, 0x49, 0x46, 0x46],
}

MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB


def _has_valid_magic(data: bytes, mime_type: str) -> bool:
    magic = IMAGE_MAGIC.get(mime_type)
    if not magic:
        return True
    return all(i < len(data) and data[i] == b for i, b in enumerate(magic))


async def do_view_image(sandbox, filename: str, use_vision: bool) -> tuple[bytes, str] | str:
    """Returns (image_bytes, media_type) on success, or error string."""
    # Strip leading path if model passes full container path
    basename = Path(filename).name
    ext = Path(basename).suffix.lower()
    mime_type = IMAGE_EXTS_FOR_VISION.get(ext)
    if not mime_type:
        return f"Not a supported image type: {filename}"

    if not use_vision:
        return "Vision not available for this model. Use bash tools (steghide, zsteg, exiftool, strings) instead."

    # Try the filename as-is first (if it's an absolute path), then search standard dirs
    search_paths = []
    if filename.startswith("/"):
        search_paths.append(filename)
    search_paths.extend([f"/challenge/distfiles/{basename}", f"/challenge/workspace/{basename}"])

    for path in search_paths:
        try:
            data = await sandbox.read_file_bytes(path)
            if not _has_valid_magic(data, mime_type):
                return (
                    "Cannot load image: file appears invalid or corrupted. "
                    "Fix the magic bytes in the sandbox first, save to /challenge/workspace/, "
                    "then call view_image again."
                )
            if len(data) > MAX_IMAGE_BYTES:
                return (
                    f"Image too large for vision ({len(data) / 1024 / 1024:.1f} MB > 4 MB limit). "
                    "Use bash tools (steghide, zsteg, binwalk, exiftool, strings, xxd) instead."
                )
            return (data, mime_type)
        except Exception:
            continue

    return f"File not found: {filename} (searched: {', '.join(search_paths)})"
