#!/usr/bin/env python3
"""Seed a local CTFd instance with the regression challenge sets.

Uses only the standard library (urllib) plus pyyaml — run inside the agent
container:

  docker run --rm -v <repo>:/work -w /work --entrypoint /app/.venv/bin/python \
      ctf-agent scripts/seed_ctfd.py --url http://host.docker.internal:8000 \
      --roots challenges/regression challenges/hard challenges/hard2 \
      challenges/example-static

Idempotent: reuses existing challenges by name.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def http_req(opener, method, url, data=None, headers=None, json_body=None, multipart=None):
    if multipart:
        boundary = "----pyform"
        body = b""
        for name, value in multipart.get("fields", []):
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, (filename, content, ctype) in multipart.get("files", []):
            body += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\nContent-Type: {ctype}\r\n\r\n'
            ).encode() + content + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", **(headers or {})},
        )
    elif json_body is not None:
        req = urllib.request.Request(
            url, data=json.dumps(json_body).encode(), method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
    elif data is not None:
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    else:
        req = urllib.request.Request(url, method=method, headers=headers or {})
    try:
        with opener.open(req, timeout=30) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raw = e.read()
    try:
        return json.loads(raw), {}
    except Exception:
        return {"_raw": raw.decode("utf-8", errors="replace")}, {}


def _fetch_html(opener, url):
    with opener.open(url, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--admin-user", default="admin")
    ap.add_argument("--admin-pass", default="admin123")
    ap.add_argument("--admin-email", default="admin@ctf.local")
    ap.add_argument("--roots", nargs="+", required=True)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    import time
    for _ in range(60):
        try:
            _fetch_html(opener, base + "/setup")
            break
        except Exception:
            time.sleep(2)
    print("[seed] CTFd is up")

    setup_html = _fetch_html(opener, base + "/setup")
    if 'name="nonce"' in setup_html:
        nonce = re.search(r'name="nonce"[^>]*value="([^"]+)"', setup_html).group(1)
        form = urllib.parse.urlencode({
            "ctf_name": "Local CTF",
            "name": args.admin_user,
            "email": args.admin_email,
            "password": args.admin_pass,
            "nonce": nonce,
        }).encode()
        resp, _ = http_req(opener, "POST", base + "/setup", data=form,
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        print("[seed] setup:", resp.get("success", str(resp)[:120]))
    else:
        login_html = _fetch_html(opener, base + "/login")
        m = re.search(r'name="nonce"[^>]*value="([^"]+)"', login_html)
        form = urllib.parse.urlencode({
            "name": args.admin_user, "password": args.admin_pass,
            "nonce": m.group(1) if m else "", "_submit": "Submit",
        }).encode()
        resp, _ = http_req(opener, "POST", base + "/login", data=form,
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
        print("[seed] login:", resp.get("success", str(resp)[:120]))

    resp, _ = http_req(opener, "POST", base + "/api/v1/tokens", json_body={"name": "agent"})
    token = (resp.get("data") or {}).get("value") if isinstance(resp, dict) else None
    if not token:
        # CTFd wants a CSRF header for session-authenticated POSTs — fetch one.
        # The token appears in admin pages as  'csrfNonce': "<hex>".
        csrf = None
        for page_path in ("/admin", "/admin/challenges", "/settings"):
            try:
                page = _fetch_html(opener, base + page_path)
            except Exception:
                continue
            m = (re.search(r"""["']csrfNonce["']\s*:\s*"([0-9a-f]+)""", page)
                 or re.search(r"""csrfNonce\s*=\s*"([^"]+)""", page)
                 or re.search(r"""csrf-token["']?\s*content="([^"]+)""", page))
            if m:
                csrf = m.group(1)
                break
        if csrf:
            resp, _ = http_req(opener, "POST", base + "/api/v1/tokens",
                               json_body={"name": "agent"},
                               headers={"X-CSRFToken": csrf, "CSRF-Token": csrf})
            token = (resp.get("data") or {}).get("value") if isinstance(resp, dict) else None
    if not token:
        print("[seed] token response:", resp)
        sys.exit("failed to create API token")
    print(f"[seed] CTFD_TOKEN={token}")
    auth = {"Authorization": f"Token {token}"}

    created = 0
    for root in args.roots:
        for meta_path in sorted(Path(ROOT / root).rglob("metadata.yml")):
            ch_dir = meta_path.parent
            data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            name = data.get("name", ch_dir.name)
            flag = data.get("flag") or (data.get("flags") or [None])[0]
            if not flag:
                print(f"[seed]   skip {name} (no answer key)")
                continue

            payload = {
                "name": name,
                "category": data.get("category", "misc"),
                "description": data.get("description", "") or name,
                "value": int(data.get("value", 100) or 100),
                "state": "visible",
                "type": "standard",
            }
            resp, _ = http_req(opener, "POST", base + "/api/v1/challenges", json_body=payload, headers=auth)
            ch = (resp.get("data") or {})
            ch_id = ch.get("id")
            if ch_id is None:
                resp2, _ = http_req(opener, "GET", base + "/api/v1/challenges", headers=auth)
                existing = {c["name"]: c["id"] for c in resp2.get("data", [])}
                ch_id = existing.get(name)
                if ch_id is None:
                    print(f"[seed]   FAILED to create {name}: {resp}")
                    continue
                print(f"[seed]   exists {name} (id={ch_id})")
            else:
                print(f"[seed]   created {name} (id={ch_id})")

            resp, _ = http_req(opener, "POST", base + "/api/v1/flags",
                               json_body={"content": str(flag), "type": "static", "challenge": ch_id, "data": ""},
                               headers=auth)
            if not resp.get("success", False):
                print(f"[seed]     flag issue: {resp}")

            dist = ch_dir / "distfiles"
            if dist.is_dir():
                files = [("file", (f.name, f.read_bytes(), "application/octet-stream"))
                         for f in sorted(dist.iterdir()) if f.is_file()]
                if files:
                    resp, _ = http_req(opener, "POST", base + "/api/v1/files",
                                       multipart={"fields": [("challenge", str(ch_id)), ("type", "challenge")],
                                                  "files": files},
                                       headers=auth)
                    if not resp.get("success", False):
                        print(f"[seed]     file upload issue: {resp}")
            created += 1

    print(f"[seed] done — {created} challenges ready")
    print(f"CTFD_URL={base}")
    print(f"CTFD_TOKEN={token}")


if __name__ == "__main__":
    main()
