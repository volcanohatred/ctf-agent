#!/usr/bin/env python3
"""Generate HARD regression challenges (designed to stress the skill system).

Runs INSIDE the ctf-sandbox container (needs gcc, python3, hashlib):

  docker run --rm -v <repo>:/work ctf-sandbox \
      python3 /work/scripts/gen_hard.py

Output: challenges/hard/<name>/{metadata.yml, distfiles/...}
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/work") if Path("/work").exists() else Path(__file__).resolve().parent.parent
OUT = ROOT / "challenges" / "hard"


def _w(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _meta(ch_dir: Path, name: str, category: str, value: int, desc: str, flag: str) -> None:
    _w(ch_dir / "metadata.yml", (
        f"name: {name}\n"
        f"category: {category}\n"
        f"value: {value}\n"
        f"description: |\n"
        + "".join(f"  {line}\n" for line in desc.strip().splitlines())
        + f"flag: {flag!r}\n"
    ))


def _clean(ch_dir: Path) -> None:
    if ch_dir.exists():
        shutil.rmtree(ch_dir)
    (ch_dir / "distfiles").mkdir(parents=True)


def _c_array(name: str, data: bytes) -> str:
    return f"static const unsigned char {name}[] = {{" + ", ".join(str(b) for b in data) + "};"


# --------------------------------------------------------------------- pwn ----

def gen_pwn_heap_tcache() -> None:
    """Menu heap notes with UAF → tcache poisoning (glibc ≥2.32 safe-linking).

    Exploit: leak heap_shift via freed fd, forge fd to &is_admin, write 1,
    then trigger the admin-only flag printer.
    """
    ch = OUT / "pwn-heap-tcache"
    _clean(ch)
    flag = "flag{tc4ch3_p01s0n_s4f3l1nk}"
    src = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAXN 16
static char *notes[MAXN];
static size_t sizes[MAXN];
static volatile long is_admin;   /* volatile: keep the BSS symbol for the exploit */

{_c_array("enc", bytes(b ^ 0x42 for b in flag.encode()))}

static void win(void) {{
    char f[sizeof(enc) + 1];
    for (unsigned long i = 0; i < sizeof(enc); i++) f[i] = enc[i] ^ 0x42;
    f[sizeof(enc)] = 0;
    puts(f);
}}

static int read_int(void) {{
    char buf[32];
    if (!fgets(buf, sizeof buf, stdin)) exit(0);
    return atoi(buf);
}}

static void add(void) {{
    puts("index:"); int i = read_int();
    if (i < 0 || i >= MAXN || notes[i]) {{ puts("bad index"); return; }}
    notes[i] = malloc(0x20);
    sizes[i] = 0x20;
    puts("data:");
    ssize_t n = read(0, notes[i], 0x20);
    if (n > 0 && notes[i][n-1] == '\\n') notes[i][n-1] = 0;
}}

static void del(void) {{
    puts("index:"); int i = read_int();
    if (i < 0 || i >= MAXN || !notes[i]) {{ puts("bad index"); return; }}
    free(notes[i]);   /* UAF: pointer not cleared */
}}

static void edit(void) {{
    puts("index:"); int i = read_int();
    if (i < 0 || i >= MAXN || !notes[i]) {{ puts("bad index"); return; }}
    puts("data:");
    read(0, notes[i], sizes[i]);
}}

static void show(void) {{
    puts("index:"); int i = read_int();
    if (i < 0 || i >= MAXN || !notes[i]) {{ puts("bad index"); return; }}
    write(1, notes[i], sizes[i]);
    puts("");
}}

int main(void) {{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stdin, NULL, _IONBF, 0);
    puts("heap notes v1.0");
    for (;;) {{
        printf("1)add 2)delete 3)edit 4)show 5)flag 6)exit > ");
        int c = read_int();
        if (c == 1) add();
        else if (c == 2) del();
        else if (c == 3) edit();
        else if (c == 4) show();
        else if (c == 5) {{ if (is_admin) win(); else puts("not admin"); }}
        else return 0;
    }}
}}
"""
    _w(ch / "build" / "heap.c", src)
    subprocess.run(
        ["gcc", "-O1", "-fno-stack-protector", "-no-pie", "-o",
         str(ch / "distfiles" / "heap"), str(ch / "build" / "heap.c")],
        check=True,
    )
    _meta(ch, "reg-pwn-heap-tcache", "pwn", 350,
          "A heap note manager (glibc 2.39, safe-linking enabled). delete() leaves a\n"
          "dangling pointer. Get the program to print the flag.\n"
          "Binary at /challenge/distfiles/heap — run it locally in your sandbox.",
          flag)


# ------------------------------------------------------------------ crypto ----

def gen_crypto_coppersmith() -> None:
    """RSA with the high half of p leaked → Coppersmith small_roots recovery."""
    import sympy

    ch = OUT / "crypto-coppersmith"
    _clean(ch)
    flag = "flag{c0pp3rsm1th_h1gh_b1ts}"
    p = int(sympy.randprime(2**511, 2**512))
    q = int(sympy.randprime(2**511, 2**512))
    n = p * q
    e = 65537
    m = int.from_bytes(flag.encode(), "big")
    c = pow(m, e, n)
    p_high = p >> 212   # 212 unknown bits — comfortably inside Coppersmith bounds
    _w(ch / "distfiles" / "leak.txt",
       f"n = {n}\ne = {e}\np_high = {p_high}\n# p = (p_high << 212) + p_low, 0 <= p_low < 2^212\n"
       f"c = {c}\n")
    _meta(ch, "reg-crypto-coppersmith", "crypto", 350,
          "RSA-1024. The top 300 bits of prime p leaked.\n"
          "Recover the full factorization and decrypt the flag.",
          flag)


# ----------------------------------------------------------------- reverse ----

def gen_reverse_antidebug() -> None:
    """ptrace anti-debug + per-char transform table; the flag IS the input."""
    ch = OUT / "reverse-anti-debug"
    _clean(ch)
    flag = "flag{4nt1_d3bug_tr4bl3}"
    # Per-position transform: enc[i] = f_i(flag[i]), f alternates between
    # xor-with-key, add-constant, xor-with-index-mul — easy to invert, tedious
    # to hand-trace across 24 chars with an anti-debug wrapper.
    transforms = []
    enc = []
    for i, c in enumerate(flag.encode()):
        t = i % 3
        if t == 0:
            enc.append(c ^ 0x13)
        elif t == 1:
            enc.append((c + 7) & 0xFF)
        else:
            enc.append(c ^ ((i * 11) & 0xFF))
        transforms.append(t)
    src = f"""#include <stdio.h>
#include <string.h>
#include <sys/ptrace.h>

{_c_array("enc", bytes(enc))}

int main(int argc, char **argv) {{
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) < 0) {{
        puts("debugger detected — bye");
        return 1;
    }}
    if (argc < 2) {{ puts("usage: ./check <flag>"); return 1; }}
    const char *s = argv[1];
    if (strlen(s) != sizeof(enc)) {{ puts("wrong"); return 0; }}
    for (unsigned long i = 0; i < sizeof(enc); i++) {{
        unsigned char v;
        switch (i % 3) {{
            case 0: v = s[i] ^ 0x13; break;
            case 1: v = (s[i] + 7) & 0xff; break;
            default: v = s[i] ^ ((i * 11) & 0xff); break;
        }}
        if (v != enc[i]) {{ puts("wrong"); return 0; }}
    }}
    puts("correct!");
    return 0;
}}
"""
    _w(ch / "build" / "check.c", src)
    subprocess.run(
        ["gcc", "-O2", "-s", "-o", str(ch / "distfiles" / "check"), str(ch / "build" / "check.c")],
        check=True,
    )
    _meta(ch, "reg-reverse-anti-debug", "reverse", 300,
          "check verifies the flag passed as argv[1] — but it refuses to run under a\n"
          "debugger, and the per-character checks are all different.\n"
          "Recover the flag. (Stripping the binary for analysis is fair game.)",
          flag)


# --------------------------------------------------------------------- web ----

def gen_web_flask_cookie() -> None:
    """Flask session cookie with a weak SECRET_KEY (from rockyou).

    flag = flag{md5(secret)[:20]} — recoverable only by cracking the cookie.
    """
    ch = OUT / "web-flask-cookie"
    _clean(ch)
    secret = "q1w2e3r4"  # present in rockyou.txt
    flag = "flag{" + hashlib.md5(secret.encode()).hexdigest()[:20] + "}"

    # Sign a REAL flask session cookie via the flask framework itself
    # (guarantees flask-unsign compatibility — sha1/hmac, flask serializer).
    from flask import Flask, session

    app = Flask(__name__)
    app.secret_key = secret

    @app.route("/login")
    def _login():
        session["user"] = "admin"
        session["uid"] = 1337
        return "ok"

    client = app.test_client()
    resp = client.get("/login")
    cookie = resp.headers["Set-Cookie"].split(";", 1)[0].split("=", 1)[1]

    app_py = '''"""Item service — session signing key was picked from a common wordlist."""
import os

from flask import Flask, session

app = Flask(__name__)
app.secret_key = open("/etc/service_secret").read().strip()  # NOT this file :)


@app.route("/")
def index():
    if session.get("user") == "admin":
        return "Welcome admin. (The flag is derived from my signing secret.)"
    return "login required"
'''
    _w(ch / "distfiles" / "app.py", app_py)
    _w(ch / "distfiles" / "cookie.txt",
       f"# captured flask session cookie (itsdangerous, sha1)\n{cookie}\n")
    _w(ch / "distfiles" / "README.txt",
       "The flag is flag{md5(secret)[:20]} where secret is the flask signing key.\n"
       "The key was picked from a common password wordlist.\n")
    _meta(ch, "reg-web-flask-cookie", "web", 250,
          "A captured Flask session cookie (see cookie.txt). The signing secret is a\n"
          "common wordlist password. Crack it, then compute flag = flag{{md5(secret)[:20]}}.",
          flag)


# ------------------------------------------------------------------ driver ----

def main() -> None:
    print(f"[gen-hard] output → {OUT}")
    gens = [
        gen_pwn_heap_tcache,
        gen_crypto_coppersmith,
        gen_reverse_antidebug,
        gen_web_flask_cookie,
    ]
    for g in gens:
        g()
        print(f"[gen-hard]   ✓ {g.__name__}")
    print(f"[gen-hard] done: {len(gens)} challenges")


if __name__ == "__main__":
    main()
