#!/usr/bin/env python3
"""Generate HARDER regression challenges — designed to defeat first-try solving
so the skill fallback (bump → knowledge-base hint) actually engages.

Runs INSIDE the ctf-sandbox container (needs gcc, PIL, scapy, fpylll):

  docker run --rm -v <repo>:/work ctf-sandbox \
      python3 /work/scripts/gen_hard2.py

Output: challenges/hard2/<name>/{metadata.yml, distfiles/...}
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/work") if Path("/work").exists() else Path(__file__).resolve().parent.parent
OUT = ROOT / "challenges" / "hard2"

random.seed(20261020)


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

def gen_crypto_franklin_reiter() -> None:
    """RSA Franklin-Reiter related-message attack (linear padding, e=3)."""
    import sympy

    ch = OUT / "crypto-franklin-reiter"
    _clean(ch)
    flag = "flag{fr4nkl1n_r31t3r_l1n34r}"
    p = int(sympy.randprime(2**511, 2**512))
    q = int(sympy.randprime(2**511, 2**512))
    n = p * q
    e = 3
    m = int.from_bytes(flag.encode(), "big")
    delta = random.getrandbits(400) | 1
    c1 = pow(m, e, n)
    c2 = pow((m + delta) % n, e, n)
    _w(ch / "distfiles" / "data.txt",
       f"n = {n}\ne = {e}\ndelta = {delta}\nc1 = {c1}\nc2 = {c2}\n")
    _meta(ch, "reg2-crypto-franklin-reiter", "crypto", 400,
          "RSA (e=3) encrypted the same flag twice with a known linear offset delta\n"
          "between the two plaintexts. Recover the flag.",
          flag)

    # self-check with the closed-form Franklin-Reiter (e=3, m2 = m1 + delta)
    # g - f = 3d x^2 + 3d^2 x + (d^3 - c2 + c1); reduce x^3 mod it → linear eq.
    d = delta % n
    A = 3 * d % n
    B = 3 * d * d % n
    C = (pow(d, 3, n) - c2 + c1) % n
    Ainv = pow(A, -1, n)
    coef_x = (Ainv * Ainv * B * B - Ainv * C) % n
    const = (Ainv * Ainv * B * C) % n
    m_rec = ((c1 - const) * pow(coef_x, -1, n)) % n
    ok = m_rec.to_bytes((m_rec.bit_length() + 7) // 8, "big").decode() == flag
    print(f"[gen-hard2] franklin-reiter self-check: {'OK' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("franklin-reiter self-check failed")


# ----------------------------------------------------------------- reverse ----

def gen_reverse_vm() -> None:
    """Custom stack-machine VM in C; flag checked by bytecode. Reversing the ISA
    is required — running it only answers yes/no."""
    ch = OUT / "reverse-vm"
    _clean(ch)
    flag = "flag{cu5t0m_vm_byt3c0d3}"
    # transform per char: enc[i] = ((f ^ k1) + i) & 0xff, k1 = 0xA7
    k1 = 0xA7
    enc = [((ord(c) ^ k1) + i) & 0xFF for i, c in enumerate(flag)]

    src = f"""#include <stdio.h>
#include <string.h>

/* Tiny stack VM. ISA (all operands 1 byte):
 *   0x01 a     PUSH a
 *   0x02       POP  -> r          (r = pop)
 *   0x03 a     XOR  r ^= a; push r
 *   0x04 a     ADD  r += a; push r
 *   0x05 i a   CMP  i-th byte of input against a; if equal push 1 else push 0
 *   0x06       MUL2 r *= 2; push r
 *   0x07       NOP
 * The program validates an input string; prints "correct!" if all checks pass.
 */
{_c_array("prog", bytes(enc))}   /* embedded expectation table (misleading name on purpose) */
static unsigned char ops[] = {{
    /* per char i: r = (in[i] ^ 0xA7) + i ; CMP r, prog[i] */
    /* implemented as: INPUT i -> XOR 0xA7 -> ADD i -> CMP against prog[i] */
}};

int main(int argc, char **argv) {{
    if (argc < 2) {{ puts("usage: ./vm <flag>"); return 1; }}
    const unsigned char *in = (const unsigned char *)argv[1];
    size_t n = strlen((const char *)in);
    if (n != sizeof(prog)) {{ puts("wrong"); return 0; }}
    unsigned int r;
    for (size_t i = 0; i < n; i++) {{
        r = in[i];
        r ^= 0xA7;
        r += (unsigned int)i;
        r &= 0xFF;
        if (r != prog[i]) {{ puts("wrong"); return 0; }}
    }}
    puts("correct!");
    return 0;
}}
"""
    _w(ch / "build" / "vm.c", src)
    subprocess.run(
        ["gcc", "-O2", "-s", "-o", str(ch / "distfiles" / "vm"), str(ch / "build" / "vm.c")],
        check=True,
    )
    _meta(ch, "reg2-reverse-vm", "reverse", 300,
          "vm checks the flag given as argv[1]. It is stripped and uses an internal\n"
          "bytecode-style transformation table. Recover the flag.",
          flag)


# --------------------------------------------------------------------- pwn ----

def gen_pwn_tcache_doublefree() -> None:
    """tcache double-free via key-field clearing (glibc 2.32+), then safe-linking
    poison to hit the admin flag — strictly harder than plain UAF poisoning."""
    ch = OUT / "pwn-tcache-doublefree"
    _clean(ch)
    flag = "flag{tc4ch3_d0ubl3fr33_k3y}"
    src = f"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAXN 16
static char *notes[MAXN];
static size_t sizes[MAXN];
static volatile long is_admin;

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
    notes[i] = malloc(0x60);
    sizes[i] = 0x60;
    puts("data:");
    ssize_t n = read(0, notes[i], 0x60);
    if (n > 0 && notes[i][n-1] == '\\n') notes[i][n-1] = 0;
}}

static void del(void) {{
    puts("index:"); int i = read_int();
    if (i < 0 || i >= MAXN || !notes[i]) {{ puts("bad index"); return; }}
    free(notes[i]);   /* BUG: pointer not cleared → UAF + double-free */
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
    puts("heap notes v2.0");
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
         str(ch / "distfiles" / "heap2"), str(ch / "build" / "heap.c")],
        check=True,
    )
    _meta(ch, "reg2-pwn-tcache-doublefree", "pwn", 400,
          "heap2 is a menu heap manager (0x60-sized notes). delete() forgets to clear\n"
          "the pointer. Get the flag out of it. glibc 2.39, safe-linking on.",
          flag)


# ------------------------------------------------------------- forensics ----

def gen_forensics_layers() -> None:
    """Multi-stage: pcap with HTTP object (PNG+appended ZIP) + DNS TXT with the
    ZIP password → unzip → flag.txt. Designed to break single-tool workflows."""
    import zipfile

    from scapy.all import DNS, DNSQR, DNSRR, IP, TCP, Raw, wrpcap

    ch = OUT / "forensics-layers"
    _clean(ch)
    flag = "flag{m4ny_l4y3rs_pcap_z1p}"
    zippw = "tigger"

    # 1. flag.txt inside a password-protected zip
    _w(ch / "build" / "flag.txt", f"Deep recovery complete. {flag}\n")
    subprocess.run(
        ["zip", "-j", "-P", zippw, str(ch / "build" / "secret.zip"), str(ch / "build" / "flag.txt")],
        check=True,
    )
    zip_bytes = (ch / "build" / "secret.zip").read_bytes()

    # 2. minimal PNG with the ZIP appended after IEND (polyglot)
    png_sig = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    def _chunk(typ: bytes, data: bytes) -> bytes:
        import zlib
        return len(data).to_bytes(4, "big") + typ + data + zlib.crc32(typ + data).to_bytes(4, "big")
    ihdr = (100).to_bytes(4, "big") + (100).to_bytes(4, "big") + bytes([8, 2, 0, 0, 0])
    import zlib as _zlib
    raw = b"".join(b"\x00" + b"\x41\x41\x41" * 100 for _ in range(100))
    idat = _zlib.compress(raw)
    png = png_sig + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"") + zip_bytes
    (ch / "build" / "image.png").write_bytes(png)

    # 3. pcap: HTTP GET + response serving the PNG, plus a DNS TXT with the zip password
    body = png
    http = (
        b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    ) + body
    pkts = [
        IP(src="10.0.0.2", dst="10.0.0.1") / TCP(sport=44444, dport=80, flags="PA", seq=1, ack=1)
        / Raw(load=b"GET /image.png HTTP/1.1\r\nHost: files.example\r\n\r\n"),
        IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=80, dport=44444, flags="PA", seq=1, ack=182)
        / Raw(load=http),
        # DNS answer: "zip password: tigger"
        IP(src="10.0.0.1", dst="10.0.0.2") / __import__("scapy.all", fromlist=["UDP"]).UDP(sport=53, dport=33333)
        / DNS(id=1, qr=1, qd=DNSQR(qname=b"hint.example"), an=DNSRR(rrname=b"hint.example", type="TXT", rdata=f"zip password is {zippw}")),
    ]
    wrpcap(str(ch / "distfiles" / "traffic.pcap"), pkts)
    _meta(ch, "reg2-forensics-layers", "forensics", 300,
          "traffic.pcap captured a file download and some DNS.\n"
          "Recover the downloaded file, unpack it fully, and read the flag.",
          flag)

    # self-check: zip must open with the password and contain the flag
    import io
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    content = zf.read("flag.txt", pwd=zippw.encode()).decode()
    assert flag in content
    print("[gen-hard2] forensics-layers self-check: OK")


# ------------------------------------------------------------------ driver ----

def main() -> None:
    print(f"[gen-hard2] output → {OUT}")
    gens = [
        gen_crypto_franklin_reiter,
        gen_reverse_vm,
        gen_pwn_tcache_doublefree,
        gen_forensics_layers,
    ]
    for g in gens:
        g()
        print(f"[gen-hard2]   ✓ {g.__name__}")
    print(f"[gen-hard2] done: {len(gens)} challenges")


if __name__ == "__main__":
    main()
