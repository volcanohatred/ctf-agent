#!/bin/bash
# Sanity-check the generated regression set: verify answer keys are actually
# recoverable from the distfiles (runs inside ctf-sandbox).
set -u
cd /work/challenges/regression
ok=0; fail=0
check() {  # check <name> <expected> <actual>
  if [ "$2" = "$3" ]; then echo "PASS  $1"; ok=$((ok+1)); else echo "FAIL  $1: want [$2] got [$3]"; fail=$((fail+1)); fi
}

echo "== crypto-wiener (sage small-d) =="
python3 - <<'PY'
import sympy
n = int(open('crypto-wiener/distfiles/public_key.txt').read().splitlines()[0].split('=')[1])
e = int(open('crypto-wiener/distfiles/public_key.txt').read().splitlines()[1].split('=')[1])
c = int(open('crypto-wiener/distfiles/ciphertext.txt').read().split('=')[1])
# Wiener attack via continued fractions
def wiener(e, n):
    conv = sympy.continued_fraction(sympy.Rational(e, n))
    for i in range(1, len(conv)):
        k = sympy.continued_fraction_reduce(conv[:i]).q
        d = sympy.continued_fraction_reduce(conv[:i]).p
        if k == 0: continue
        phi = (e*d - 1) // k
        s = n - phi + 1
        disc = s*s - 4*n
        if disc > 0 and sympy.integer_nthroot(disc, 2)[1]:
            return d
    return None
d = wiener(e, n)
m = pow(c, d, n) if d else 0
print("wiener recovered:", m.to_bytes((m.bit_length()+7)//8, 'big'))
PY

echo "== reverse-xor-binary =="
./reverse-xor-binary/distfiles/check M4sterK3y_2024

echo "== reverse-pyc =="
python3 -c "
import dis, marshal, importlib.util
f = open('reverse-pyc/distfiles/checker.pyc','rb'); f.read(16)
code = marshal.load(f)
K = None
for const in code.co_consts:
    if isinstance(const, list) and const and isinstance(const[0], int): K = const
print('pyc flag:', ''.join(chr(b ^ 0x5A) for b in K))
"

echo "== misc-encoding =="
python3 -c "
import base64, codecs
t = open('misc-encoding/distfiles/message.txt').read().split()[-1]
print(codecs.encode(bytes.fromhex(base64.b85decode(base64.b64decode(t)).decode()).decode(), 'rot13'))
"

echo "== misc-zip-crack =="
cd /tmp && rm -f flag.txt && unzip -P shadow -o /work/challenges/regression/misc-zip-crack/distfiles/flag.zip && cat flag.txt; cd /work/challenges/regression

echo "== web-sqli =="
python3 -c "import sqlite3; print(sqlite3.connect('web-sqli-local/distfiles/app.db').execute('select * from flag').fetchall())"

echo "== pwn binary info =="
file pwn-ret2win/distfiles/vuln
python3 -c "
data = open('pwn-ret2win/distfiles/vuln','rb').read()
import re
m = re.search(rb'[\x00-\xff]{20,}?\x42', data)  # rough: flag enc bytes end with }^0x42
print('binary size', len(data))
"
echo "== pyc decompile via pycdc =="
pycdc reverse-pyc/distfiles/checker.pyc | head -5
