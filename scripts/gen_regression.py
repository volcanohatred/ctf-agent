#!/usr/bin/env python3
"""Generate the CTF regression test set (10 challenges with answer keys).

Runs INSIDE the ctf-sandbox container (needs gcc, PIL, scapy, zip):

  docker run --rm -v <repo>:/work ctf-sandbox \
      python3 /work/scripts/gen_regression.py

Output: challenges/regression/<name>/{metadata.yml, distfiles/...}
Every flag is also written into metadata.yml (`flag:` answer key) — the
sandbox mounts a sanitized copy without it (see backend/sandbox.py).
"""

from __future__ import annotations

import base64
import codecs
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/work") if Path("/work").exists() else Path(__file__).resolve().parent.parent
OUT = ROOT / "challenges" / "regression"


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


# ---------------------------------------------------------------- crypto ----

def gen_crypto_wiener() -> None:
    import secrets

    import gmpy2
    import sympy

    ch = OUT / "crypto-wiener"
    _clean(ch)
    flag = "flag{w13n3r_4ttack_sm4ll_d}"
    p = sympy.randprime(2**511, 2**512)
    q = sympy.randprime(2**511, 2**512)
    n = int(p * q)
    phi = (p - 1) * (q - 1)
    # Wiener attack works when d < (1/3) * n^(1/4); n is 1024-bit → 256-bit bound.
    while True:
        d = secrets.randbits(200) | 1
        if gmpy2.gcd(d, phi) == 1:
            break
    e = int(gmpy2.invert(d, phi))
    m = int.from_bytes(flag.encode(), "big")
    c = pow(m, e, n)
    _w(ch / "distfiles" / "public_key.txt", f"n = {n}\ne = {e}\n")
    _w(ch / "distfiles" / "ciphertext.txt", f"c = {c}\n")
    _meta(ch, "reg-crypto-wiener", "crypto", 150,
          "We intercepted an RSA-encrypted message. The public exponent looks huge...\n"
          "Recover the plaintext and find the flag.",
          flag)


def gen_crypto_xor() -> None:
    ch = OUT / "crypto-xor-repeating"
    _clean(ch)
    flag = "flag{x0r_r3p34t_k3y_fr3q}"
    key = b"s3cr3t7"
    pt = (
        "This message was encrypted with a repeating-key XOR cipher.\n"
        f"Nothing to see here... except the flag: {flag}\n"
        "The quick brown fox jumps over the lazy dog. " * 8
    ).encode()
    ct = bytes(b ^ key[i % len(key)] for i, b in enumerate(pt))
    _w(ch / "distfiles" / "cipher.hex", ct.hex() + "\n")
    _meta(ch, "reg-crypto-xor-repeating", "crypto", 100,
          "cipher.hex contains a repeating-key XOR ciphertext (hex encoded).\n"
          "Break it and recover the flag.",
          flag)


# ------------------------------------------------------------- forensics ----

def gen_forensics_magic() -> None:
    from PIL import Image, ImageDraw, ImageFont

    ch = OUT / "forensics-magic"
    _clean(ch)
    flag = "flag{png_m4g1c_r3p41r}"
    img = Image.new("RGB", (1000, 260), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 90), flag, fill="black", font=font)
    png = ch / "distfiles" / "evidence.png"
    img.save(png, "PNG")
    data = bytearray(png.read_bytes())
    data[0:8] = b"\x00" * 8  # corrupt the PNG magic
    png.write_bytes(bytes(data))
    _meta(ch, "reg-forensics-magic", "forensics", 50,
          "evidence.png seems corrupted — no viewer can open it.\n"
          "Repair the file and read the flag.",
          flag)


def gen_forensics_pcap() -> None:
    import gzip

    from scapy.all import IP, TCP, Raw, wrpcap

    ch = OUT / "forensics-pcap"
    _clean(ch)
    flag = "flag{pcap_http_gz1p_0bj}"
    body = gzip.compress(f"Congrats! The flag is {flag}\n".encode())
    http = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
        b"Content-Encoding: gzip\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    ) + body
    sport, dport = 80, 44444
    pkts = [
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=sport, dport=dport, flags="S", seq=1000),
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=dport, dport=sport, flags="SA", seq=2000, ack=1001),
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=sport, dport=dport, flags="A", seq=1001, ack=2001),
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=sport, dport=dport, flags="PA", seq=1001, ack=2001)
        / Raw(load=b"GET /flag.txt HTTP/1.1\r\nHost: x\r\n\r\n"),
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=dport, dport=sport, flags="PA", seq=2001, ack=1182)
        / Raw(load=http),
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=sport, dport=dport, flags="FA", seq=1182, ack=2001 + len(http)),
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=dport, dport=sport, flags="FA", seq=2001 + len(http), ack=1183),
    ]
    wrpcap(str(ch / "distfiles" / "capture.pcap"), pkts)
    _meta(ch, "reg-forensics-pcap", "forensics", 100,
          "capture.pcap contains intercepted traffic of someone downloading a secret file.\n"
          "Recover the file contents and find the flag.",
          flag)


# --------------------------------------------------------------- reverse ----

def _c_array(name: str, data: bytes) -> str:
    return f"static const unsigned char {name}[] = {{" + ", ".join(str(b) for b in data) + "};"


def gen_reverse_xor() -> None:
    ch = OUT / "reverse-xor-binary"
    _clean(ch)
    flag = "flag{r3v3rs3_th3_x0r}"
    password = "M4sterK3y_2024"
    src = f"""#include <stdio.h>
#include <string.h>
{_c_array("pw_enc", bytes(b ^ 0x37 for b in password.encode()))}
{_c_array("fl_enc", bytes(b ^ 0xA5 for b in flag.encode()))}
int main(int argc, char **argv) {{
    if (argc < 2) {{ puts("usage: ./check <password>"); return 1; }}
    char pw[sizeof(pw_enc) + 1];
    for (unsigned long i = 0; i < sizeof(pw_enc); i++) pw[i] = pw_enc[i] ^ 0x37;
    pw[sizeof(pw_enc)] = 0;
    if (strcmp(argv[1], pw) != 0) {{ puts("wrong password"); return 0; }}
    char fl[sizeof(fl_enc) + 1];
    for (unsigned long i = 0; i < sizeof(fl_enc); i++) fl[i] = fl_enc[i] ^ 0xA5;
    fl[sizeof(fl_enc)] = 0;
    printf("Correct! The flag is: %s\\n", fl);
    return 0;
}}
"""
    _w(ch / "build" / "check.c", src)
    subprocess.run(
        ["gcc", "-O2", "-s", "-o", str(ch / "distfiles" / "check"), str(ch / "build" / "check.c")],
        check=True,
    )
    _meta(ch, "reg-reverse-xor-binary", "reverse", 150,
          "check asks for a password and rewards it with the flag.\n"
          "Analyze the binary and extract the flag.",
          flag)


def gen_reverse_pyc() -> None:
    ch = OUT / "reverse-pyc"
    _clean(ch)
    flag = "flag{pyc_d3c0mp1l3_m3}"
    enc = tuple(ord(c) ^ 0x5A for c in flag)  # tuple → stored as a co_consts entry
    src = f'''import sys

_K = {enc!r}

def check(s):
    return tuple(ord(c) ^ 0x5A for c in s) == _K

if __name__ == "__main__":
    if len(sys.argv) > 1 and check(sys.argv[1]):
        print("correct!")
    else:
        print("wrong")
'''
    build = ch / "build"
    _w(build / "checker.py", src)
    subprocess.run([sys.executable, "-m", "py_compile", str(build / "checker.py")], check=True)
    pyc = next((build / "__pycache__").glob("checker.cpython-*.pyc"))
    shutil.copy(pyc, ch / "distfiles" / "checker.pyc")
    _meta(ch, "reg-reverse-pyc", "reverse", 100,
          "checker.pyc validates the flag (compiled Python bytecode).\n"
          "Reverse it and recover the flag.",
          flag)


# ------------------------------------------------------------------- misc ----

def gen_misc_encoding() -> None:
    ch = OUT / "misc-encoding"
    _clean(ch)
    flag = "flag{l4y3r3d_3nc0d1ng}"
    t = codecs.encode(flag, "rot13")
    t = t.encode().hex()
    t = base64.b85encode(t.encode()).decode()
    t = base64.b64encode(t.encode()).decode()
    _w(ch / "distfiles" / "message.txt", f"Decode me:\n{t}\n")
    _meta(ch, "reg-misc-encoding", "misc", 25,
          "message.txt went through several layers of encoding (base64, base85, hex, rot13).\n"
          "Decode it to get the flag.",
          flag)


def gen_misc_zip() -> None:
    ch = OUT / "misc-zip-crack"
    _clean(ch)
    flag = "flag{z1p_cr4ck_r0cky0u}"
    _w(ch / "build" / "flag.txt", f"Well done! {flag}\n")
    subprocess.run(
        ["zip", "-j", "-P", "shadow", str(ch / "distfiles" / "flag.zip"), str(ch / "build" / "flag.txt")],
        check=True,
    )
    _meta(ch, "reg-misc-zip-crack", "misc", 75,
          "flag.zip is password-protected. The password is weak and common.\n"
          "Crack it and extract the flag.",
          flag)


# -------------------------------------------------------------------- web ----

def gen_web_sqli() -> None:
    import sqlite3

    ch = OUT / "web-sqli-local"
    _clean(ch)
    flag = "flag{sql1_un10n_3xtr4ct}"
    db = ch / "distfiles" / "app.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, info TEXT)")
    con.executemany("INSERT INTO items VALUES (?, ?, ?)", [
        (1, "apple", "a fruit"), (2, "banana", "yellow"), (3, "cherry", "red"),
    ])
    con.execute("CREATE TABLE flag (secret TEXT)")
    con.execute("INSERT INTO flag VALUES (?)", (flag,))
    con.commit()
    con.close()
    app_py = '''import os
import sqlite3

from flask import Flask, request

app = Flask(__name__)
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")


@app.route("/")
def index():
    return "Item lookup service. Try /item?id=1"


@app.route("/item")
def item():
    i = request.args.get("id", "1")
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(f"SELECT name, info FROM items WHERE id = {i}").fetchall()
        return "<br>".join(f"{a}: {b}" for a, b in rows) or "not found"
    except Exception as e:
        return f"error: {e}"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
'''
    _w(ch / "distfiles" / "app.py", app_py)
    _meta(ch, "reg-web-sqli-local", "web", 200,
          "A Flask item-lookup service ships as source (app.py) + database (app.db).\n"
          "Start it inside your sandbox (`cd /challenge/distfiles && nohup python3 app.py &`),\n"
          "then exploit the SQL injection at /item?id= and extract the flag from the database.",
          flag)


# -------------------------------------------------------------------- pwn ----

def gen_pwn_ret2win() -> None:
    ch = OUT / "pwn-ret2win"
    _clean(ch)
    flag = "flag{r3t_2_w1n_b00m}"
    src = f"""#include <stdio.h>
{_c_array("enc", bytes(b ^ 0x42 for b in flag.encode()))}
void win(void) {{
    char f[sizeof(enc) + 1];
    for (unsigned long i = 0; i < sizeof(enc); i++) f[i] = enc[i] ^ 0x42;
    f[sizeof(enc)] = 0;
    puts(f);
}}
void vuln(void) {{
    char buf[48];
    puts("Send your payload:");
    fflush(stdout);
    gets(buf);
}}
int main(void) {{
    setvbuf(stdout, NULL, _IONBF, 0);
    vuln();
    return 0;
}}
"""
    _w(ch / "build" / "vuln.c", src)
    subprocess.run(
        ["gcc", "-O0", "-fno-stack-protector", "-no-pie", "-z", "execstack", "-Wno-implicit-function-declaration",
         "-o", str(ch / "distfiles" / "vuln"), str(ch / "build" / "vuln.c")],
        check=True,
    )
    _meta(ch, "reg-pwn-ret2win", "pwn", 200,
          "A 64-bit binary with a classic overflow. No PIE, no canary, execstack.\n"
          "The flag is only printed by a hidden function — exploit it (or reverse the encoding)\n"
          "and submit the flag.",
          flag)


# ------------------------------------------------------------------ driver ----

def _self_test(flag_dir: Path) -> list[str]:
    """Sanity checks on generated artifacts."""
    problems = []
    for meta in flag_dir.rglob("metadata.yml"):
        ch_dir = meta.parent
        if not (ch_dir / "distfiles").exists():
            problems.append(f"{ch_dir.name}: no distfiles")
    return problems


def main() -> None:
    print(f"[gen] output → {OUT}")
    gens = [
        gen_crypto_wiener, gen_crypto_xor,
        gen_forensics_magic, gen_forensics_pcap,
        gen_reverse_xor, gen_reverse_pyc,
        gen_misc_encoding, gen_misc_zip,
        gen_web_sqli, gen_pwn_ret2win,
    ]
    for g in gens:
        g()
        print(f"[gen]   ✓ {g.__name__}")
    problems = _self_test(OUT)
    for p in problems:
        print(f"[gen]   ⚠ {p}")
    print(f"[gen] done: {len(gens)} challenges")


if __name__ == "__main__":
    main()
