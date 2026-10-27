# 安装指南（INSTALL）

本指南从零开始把 CTF Agent 跑起来。全程只需要 Docker Desktop + 一份配置。

---

## 一、前置要求

| 依赖 | 说明 |
|---|---|
| **Docker Desktop** | Windows / macOS / Linux 均可（Windows 需 WSL2，安装器会自动引导） |
| **统一网关** | 一个同时兼容 Anthropic 协议和 OpenAI 协议的模型网关地址 + 密钥 |
| **技能知识库**（可选） | 一个包含 `ctf-web/`、`ctf-pwn/` 等技能包的目录（没有也能跑，只是不用 skill 兜底） |

> **跑 agent 本身**不需要 Python / uv / Node —— 全部封装在镜像里，只需要 Docker 和 git。
> 只有在你想跑测试套件（见第七节）时才需要本机的 `uv`。

---

## 二、获取镜像

> ⚠️ **先读这条**：ACR / ghcr 上已发布的 `ctf-agent:latest` 是**旧构建**，不包含
> `feature/coach-mode` 分支上的修复（网关健康检查、轮次上限、两阶段 conclude 收割、
> dashboard 鉴权、泄漏修复等）。判断方法：
> ```bash
> docker run --rm --entrypoint ls crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-agent:latest /app/backend/health.py
> # 报 No such file → 是旧镜像
> ```
> 在新镜像推上去之前，请**本地构建**（见本节末尾），否则你会跑到修复前的代码。
> `ctf-sandbox` 不受影响——它只是工具链镜像，没有本次改动，可以直接 pull。

**沙箱镜像**（17 GB，直接拉，国内 ACR 直连）：

```bash
docker pull crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest
docker tag  crpi-6el9nmdepucmhr6v.cn-hangzhou.personal.cr.aliyuncs.com/ctf-qq/ctf-sandbox:latest ctf-sandbox:latest
```

| 镜像 | 大小 | 作用 |
|---|---|---|
| `ctf-agent` | ~1.3 GB | 控制面：协调器 + 各模型 solver 驱动（**需本地构建**） |
| `ctf-sandbox` | ~17 GB | 解题沙箱：全套 CTF 工具链（pwn/rev/crypto/forensics/web/steg） |

> 若 registry 设为私有，先 `docker login crpi-...`；公开访问则直接 pull。

**构建 ctf-agent**（快，依赖层有缓存时代码改动只需几秒）：
```bash
git clone https://github.com/volcanohatred/ctf-agent.git
cd ctf-agent
git checkout feature/coach-mode
docker compose build agent          # 或 docker build -t ctf-agent .
```

沙箱镜像也可以本地构建（慢，约 1 小时），但通常没必要：
```bash
docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .
```

> 构建依赖已提交的 `uv.lock`（`uv sync --frozen`），所以同样代码在任何机器上装出同样的
> 依赖版本。Dockerfile 把 `COPY pyproject.toml uv.lock` 放在 `COPY backend` 之前，
> 因此改代码不会击穿依赖层缓存。

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

`ctf-msg` 默认打 9400 端口，与 `MSG_PORT` 和 compose 发布的端口一致；改过其中一个就要同步改另外两个。

### 运营面板（Dashboard）

启动后浏览器打开 `http://127.0.0.1:8080`，可以看到每题每个模型的实时思路、最近动作、攻坚方向、工具调用数、成本、静默时长，以及模型间的共享黑板和求助项；可以直接喂提示、广播、导出复盘报告。

面板能给 solver 注入指令（solver 在特权沙箱里执行任意代码）、拉起新容器、导出已确认的 flag，所以**必须带 token**。没设 `DASHBOARD_TOKEN` 时会在启动日志里随机生成一个并打印出可直接点开的 URL：

```bash
docker compose logs agent | grep -i "access token"
# Dashboard access token (no DASHBOARD_TOKEN was set): <token>
#   → open http://127.0.0.1:8080/?token=<token>
```

compose 里两个端口都只发布到宿主的 `127.0.0.1`，所以局域网和沙箱容器（走 `host.docker.internal`）都碰不到。要对外开放请自己在前面挂一层带鉴权的反代。

面板新增的两个排障信号值得留意：

- **工具调用 = 0 且状态"运行"**：显示为红色"空转"。这是网关挂了或工具链没接上的特征，不是题目难。
- **`swarm 自行退役：<原因>`**：可能是 `round cap reached`（轮次上限）、`quota retries exhausted`（配额耗尽）或 `N consecutive infrastructure failures`（基础设施连续失败）。第三种说明该去查网关，不是去换解题方向。

### ③ 离线评测（本地题目集 + 答案自动校验）

```bash
docker compose run --rm agent ctf-bench --challenges-dir challenges --concurrency 2 -v
```

报告写在 `logs/bench-*.json`（解题率 / 每题耗时 / 成本 / 每题 `retire_reason`）。题目格式见下方"自带题目"。

**先看 `benchmark.usable_run` 再看解题率。** 它为 `false` 时表示大部分题目是被基础设施问题（网关不可达、配额、鉴权）判死的，这次的数字不代表模型能力。每题的 `infra_failure` 和 `retire_reason` 会说明具体原因。

`--time-limit` 若小于 `SOLVER_TIME_BUDGET_S`，单轮预算和它的 conclude 收割永远不会触发，每题都只会以外层 timeout 收尾，报告也就无法区分"解得慢"和"网关死了"——启动时会直接告警。

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

**Q：不走 compose、直接在 Windows 宿主机上跑（`uv run ctf-solve`）报 `Cannot connect to Docker Engine via https://127.0.0.1:2376`？**
A：这是 aiodocker 在 `DOCKER_HOST` 未设时的默认地址，而 Docker Desktop **不暴露** 2376——引擎在命名管道上。代码已自动探测（`backend/sandbox.py` 的 `resolve_docker_host()`，枚举 `\\.\pipe` 找 `docker_engine`），正常情况无需干预。若你用的是远程引擎或非默认 context，显式设置即可，它的优先级最高：
```powershell
$env:DOCKER_HOST = "npipe:////./pipe/docker_engine"
```

**Q：某道题只跑了几轮就停了，日志说 `round cap reached`？**
A：这是 `MAX_ATTEMPTS_PER_CHALLENGE`（默认 12）在起作用，是有意的上限，不是故障——旧版本根本没有上限，`ctf-solve --challenge` 会无限重试。想给更多轮次就调大它。注意它与 `FRESH_RESTART_ROUNDS`（默认 5）耦合：一次"蒸馏→合并"循环要花掉 5 轮，所以上限至少要是 `2×5+1=11` 才能让合并后的状态多驱动几轮。配错了启动时会直接告警，不会静默。

**Q：同时跑两个 agent（比如一个协调器 + 一个 bench），其中一个的沙箱莫名被删？**
A：启动时的孤儿清理是**按 label 匹配**的，不区分属于哪个进程，所以会连带删掉另一个 agent 的活沙箱。给其中一个设：
```ini
CLEANUP_ORPHANS_ON_START=false
```

**Q：模型报 quota / rate limit？**
A：正常现象——swarm 会自动把该模型切换到 OpenAI 协议同名模型（`gateway/X`）继续，复用同一个沙箱，不用管。

**Q：启动时报 `Gateway health check FAILED` 然后退出？**
A：这是**有意的快速失败**，比让它跑几小时后报一堆"解不出"要好。分两种：
- `[FATAL] unreachable`（连接被拒 / DNS 失败）或 `[FATAL] auth`（401/403）——确定性问题，重试没用。检查 `ANTHROPIC_BASE_URL` 在容器内是否写成 `host.docker.internal`，以及密钥是否正确。
- `[warn] timeout` / `rate_limited` / 5xx——**只告警不阻断**，会继续跑。因为本地网关偶发挂起几分钟又恢复是常态，为一次慢探测否决整轮运行不值得；真持续不可用的话，swarm 会在 `MAX_INFRA_FAILURES`（默认 3）次连续基础设施失败后带着原因自行退役。
想完全跳过探测：`STARTUP_HEALTH_CHECK=false` 或 `--skip-health-check`。

**Q：17GB 的 ctf-sandbox 拉不动？**
A：确认你拉的是国内 ACR 地址（`crpi-...aliyuncs.com`）。如果还是慢，说明当前网络对阿里云也不快，那就只能本地 build（`docker build -f sandbox/Dockerfile.sandbox -t ctf-sandbox .`）。

---

## 七、目录结构速览

```
ctf-agent/
├── Dockerfile              # ctf-agent 镜像（控制面）
├── docker-compose.yml      # 一键编排
├── .env.example            # 全部配置项模板（每项都有注释）
├── uv.lock                 # 已提交的依赖锁 —— 让构建可复现
├── .github/workflows/      # CI：lint / typecheck / 单测 / Docker 集成 / lock 校验 / smoke
├── backend/                # 控制面源码
│   ├── agents/             # solver / swarm / coordinator
│   ├── health.py           # 启动前网关探测（快速失败）
│   ├── dedup.py            # 结构化去重（journal / 死路账本 / 团队蒸馏）
│   ├── skills.py           # 技能库路由（triage + 升级注入）
│   └── ...
├── sandbox/
│   ├── Dockerfile.sandbox  # 17GB 沙箱镜像（全工具链）
│   └── ...
├── scripts/                # 题目生成器、CTFd 灌题、自验脚本
├── challenges/             # 冒烟题（其余回归题用生成器产出）
└── tests/                  # 346 个测试，见下
```

### 跑测试

```bash
uv sync                      # 或 pip install -e ".[dev]"
uv run pytest -q             # 346 项，无需 Docker / LLM / 网络
uv run ruff check .
uv run ty check backend
uv run python tests/import_check.py     # 38 个模块全部可导入
uv run python tests/wiring_check.py     # ~80 条接线断言（鉴权、hook、包装、重置分派）
uv run python tests/e2e_dead_gateway.py # 死网关必须秒级失败，而不是磨满时限
uv run pytest tests/test_sandbox_integration.py -q -rs   # 需真实 Docker，否则自动跳过
```

测试刻意不依赖外部服务：swarm 用脚本化假 solver 驱动，`httpx` 走 `MockTransport`
子类，DNS 用例全用数字 IP。每个文件单独跑也全绿，所以不存在导入顺序依赖。
