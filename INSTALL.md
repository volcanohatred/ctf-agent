# 安装指南（INSTALL）

本指南从零开始把 CTF Agent 跑起来。全程只需要 Docker Desktop + 一份配置。

---

## 一、前置要求

| 依赖 | 说明 |
|---|---|
| **Docker Desktop** | Windows / macOS / Linux 均可（Windows 需 WSL2，安装器会自动引导） |
| **统一网关** | 一个同时兼容 Anthropic 协议和 OpenAI 协议的模型网关地址 + 密钥 |
| **技能知识库**（可选） | 一个包含 `ctf-web/`、`ctf-pwn/` 等技能包的目录（没有也能跑，只是不用 skill 兜底） |

> 不需要 Python / uv / Node —— 全部封装在镜像里。

---

## 二、获取镜像（国内镜像仓库，拉取飞快）

两个镜像都已推送到阿里云容器镜像服务（国内直连）：

```bash
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest

# 打上本地短标签，后续命令用短名
docker tag crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest   ctf-agent:latest
docker tag crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest ctf-sandbox:latest
```

| 镜像 | 大小 | 作用 |
|---|---|---|
| `ctf-agent` | ~1.3 GB | 控制面：协调器 + 各模型 solver 驱动 |
| `ctf-sandbox` | ~17 GB | 解题沙箱：全套 CTF 工具链（pwn/rev/crypto/forensics/web/steg） |

> 若 registry 设为私有，先 `docker login crpi-...`；公开访问则直接 pull。

**不想拉镜像也可以本地构建**（慢，约 1 小时）：
```bash
git clone https://github.com/volcanohatred/ctf-agent.git
cd ctf-agent
docker build -t ctf-agent .
docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .
```

---

## 三、配置

```bash
git clone https://github.com/volcanohatred/ctf-agent.git
cd ctf-agent
cp .env.example .env
```

编辑 `.env`，必填这几项：

```ini
# —— 统一网关（必填）——
# Anthropic 协议端点（Claude Code CLI 走这里，驱动 kimi-k3 / deepseek-v4-flash / qwen3.8-max / glm-5.3）
ANTHROPIC_BASE_URL=http://host.docker.internal:8090/anthropic
ANTHROPIC_AUTH_TOKEN=你的网关密钥

# OpenAI 兼容端点（同一模型的降级通道，配额问题时自动切换）
GATEWAY_BASE_URL=https://你的网关/v1
GATEWAY_API_KEY=你的网关密钥

# —— Docker 部署（必填）——
# 本仓库在 Docker 宿主机上的绝对路径（正斜杠）。
# 因为 agent 容器要通过宿主机的 docker.sock 启动沙箱容器。
HOST_PROJECT_DIR=D:/code/ctf/ctf-agent          # ← 改成你的实际路径
```

可选（技能知识库）：如果有 CTF 技能库目录（含 `ctf-web/INDEX.md`、`ctf-pwn/INDEX.md` 等），填：

```ini
SKILLS_HOST_DIR=D:/code/ctf/skill/ctf            # 宿主机上的技能库路径
```

比赛模式才需要填的 CTFd 连接：

```ini
CTFD_URL=https://ctf.example.com
CTFD_TOKEN=ctfd_xxx          # 或用 CTFD_USER / CTFD_PASS
```

---

## 四、运行（三种模式）

### ① 冒烟测试（1 分钟，验证链路）

仓库自带一道 base64 签到题，先确认整条链通了：

```bash
docker compose run --rm agent ctf-solve --challenge challenges/example-static --no-submit -v
```

看到 `FLAG FOUND: flag{sm0k3_t3st_ok}` + `Answer-key check: MATCH` 即成功。

### ② 正式比赛（coordinator 模式，对着 CTFd 全自动打）

```bash
# .env 里填好 CTFD_URL / CTFD_TOKEN，然后：
docker compose run --rm --service-ports agent
```

它会自动轮询新题、为每题拉起 4 模型竞速、提交 flag、解题后自动回收。运营中可随时插话指导：

```bash
docker compose exec agent uv run --no-sync ctf-msg "这题是 JWT，试 kid 注入"
```

### ③ 离线评测（本地题目集 + 答案自动校验）

```bash
docker compose run --rm agent ctf-bench --challenges-dir challenges --concurrency 2 -v
```

报告写在 `logs/bench-*.json`（解题率 / 每题耗时 / 成本）。题目格式见下方"自带题目"。

---

## 五、自带题目（回归测试集）

仓库内置一道冒烟题。完整的 21 道自研回归题（3 个难度梯度，含堆 poisoning、HNP、MBA 混淆等）用生成器一键产出（在沙箱里跑）：

```bash
docker run --rm -v ${PWD}:/work ctf-sandbox python3 /work/scripts/gen_regression.py
docker run --rm -v ${PWD}:/work ctf-sandbox python3 /work/scripts/gen_hard.py
docker run --rm -v ${PWD}:/work ctf-sandbox python3 /work/scripts/gen_hard2.py
docker run --rm -v ${PWD}:/work ctf-sandbox python3 /work/scripts/gen_hard3.py
```

想用真实 CTFd 环境验证在线链路，本地起一个并自动灌题：

```bash
docker run -d -p 8000:8000 --name ctfd ctfd/ctfd
docker run --rm -v ${PWD}:/work -w /work --entrypoint /app/.venv/bin/python ctf-agent \
  scripts/seed_ctfd.py --url http://host.docker.internal:8000 \
  --roots challenges/regression challenges/hard challenges/hard2 challenges/hard3 challenges/example-static
```

---

## 六、常见问题（FAQ）

**Q：容器里 `127.0.0.1` 访问不到宿主机服务？**
A：容器内的 `127.0.0.1` 指容器自己。访问宿主机一律用 `host.docker.internal`（`.env` 里的 `ANTHROPIC_BASE_URL`、`CTFD_URL` 同理）。

**Q：报 `Recvmmsg: connection reset` 或沙箱起不来？**
A：确认 `HOST_PROJECT_DIR` 是宿主机上本仓库的**绝对路径**（正斜杠，Windows 盘符如 `D:/code/...`），且 Docker Desktop 正在运行。

**Q：模型报 quota / rate limit？**
A：正常现象——swarm 会自动把该模型切换到 OpenAI 协议同名模型（`gateway/X`）继续，复用同一个沙箱，不用管。

**Q：17GB 的 ctf-sandbox 拉不动？**
A：确认你拉的是国内 ACR 地址（`crpi-...aliyuncs.com`）。如果还是慢，说明当前网络对阿里云也不快，那就只能本地 build（`docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .`）。

---

## 七、目录结构速览

```
ctf-agent/
├── Dockerfile              # ctf-agent 镜像（控制面）
├── docker-compose.yml      # 一键编排
├── .env.example            # 全部配置项模板
├── backend/                # 控制面源码
│   ├── agents/             # solver / swarm / coordinator
│   ├── skills.py           # 技能库路由（triage + 升级注入）
│   └── ...
├── sandbox/
│   ├── Dockerfile.sandbox  # 17GB 沙箱镜像（全工具链）
│   └── ...
├── scripts/                # 题目生成器、CTFd 灌题、自验脚本
├── challenges/             # 冒烟题（其余回归题用生成器产出）
└── tests/                  # 单元测试（python tests/test_units.py）
```
