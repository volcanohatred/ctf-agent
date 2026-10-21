# CTF Sandbox Environment

You are solving a CTF challenge inside this container.

## Layout
- `/challenge/distfiles/` (read-only) — original challenge files
- `/challenge/workspace/` (read-write) — YOUR working area: scripts, exploits, extracted data. Files persist across commands.
- `/challenge/metadata.yml` (read-only) — challenge description
- `/tools.txt` — quick reference of installed tools (pwntools, gdb, radare2, sage, z3, volatility3, steg/forensics tools, ...)

## Skills library (read-only, mounted at `/skills/` when enabled)
Curated CTF technique packs: `ctf-web`, `ctf-pwn`, `ctf-reverse`, `ctf-crypto`,
`ctf-forensics`, `ctf-misc`, `ctf-osint`, `ctf-malware`, `ctf-ai-ml`, plus
`zh-knowledge/` (Chinese writeups with `.meta.md` metadata).
- Do NOT browse it upfront — rely on your own capability first.
- If genuinely stuck: read the relevant pack's `INDEX.md`, pick ONE matching
  file, read it, and apply a concrete technique.

## Services
- Connection info is in the prompt. `host.docker.internal` reaches the host.
- Each bash call is a fresh process — use heredocs or scripts for stateful work.

## Flag submission
- `submit_flag 'flag{...}'` (bash) or the `submit_flag` tool.
