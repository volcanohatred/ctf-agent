"""Self-contained single-page dashboard UI (served by web_dashboard.py)."""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTF Agent · 指挥台</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d; --fg: #c9d1d9;
    --dim: #8b949e; --green: #3fb950; --red: #f85149; --amber: #d29922;
    --blue: #58a6ff; --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font: 14px/1.5 "Consolas","SF Mono",monospace; }
  header { padding: 12px 18px; border-bottom: 1px solid var(--border); display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }
  header h1 { font-size: 18px; color: var(--blue); letter-spacing: 1px; }
  header .stat { color: var(--dim); }
  header .stat b { color: var(--green); }
  main { display: grid; grid-template-columns: 340px 1fr; height: calc(100vh - 55px); }
  #challenges { overflow-y: auto; border-right: 1px solid var(--border); padding: 10px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
          padding: 10px 12px; margin-bottom: 10px; cursor: pointer; }
  .card:hover { border-color: var(--blue); }
  .card.sel { border-color: var(--green); }
  .card .name { font-weight: 600; color: var(--fg); }
  .card .cat { color: var(--purple); font-size: 12px; }
  .card .status { float: right; font-size: 12px; }
  .st-running { color: var(--amber); } .st-solved { color: var(--green); }
  .st-stuck { color: var(--red); }
  .dots span { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
  .d-run { background: var(--amber); } .d-win { background: var(--green); } .d-dead { background: var(--border); }
  #detail { overflow-y: auto; padding: 14px 18px; }
  #detail h2 { color: var(--blue); margin-bottom: 6px; }
  .solver { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
            padding: 10px 12px; margin: 10px 0; }
  .solver .spec { color: var(--blue); font-weight: 600; }
  .solver .last { color: var(--dim); font-size: 12px; word-break: break-all; margin-top: 4px; }
  .trace { background: #010409; border: 1px solid var(--border); border-radius: 6px;
           padding: 10px; margin-top: 8px; max-height: 320px; overflow-y: auto; font-size: 12px; }
  .trace .tc { color: var(--blue); } .trace .tr { color: var(--dim); }
  .trace .ok { color: var(--green); } .trace .warn { color: var(--amber); } .trace .err { color: var(--red); }
  .guide { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 12px; margin-top: 14px; }
  .guide input, .guide select, .guide textarea { width: 100%; background: #010409; color: var(--fg);
    border: 1px solid var(--border); border-radius: 4px; padding: 7px; margin: 4px 0; font: inherit; font-size: 13px; }
  .guide textarea { min-height: 90px; resize: vertical; }
  .guide button { background: var(--blue); color: #04121f; border: 0; border-radius: 4px;
    padding: 8px 16px; cursor: pointer; font-weight: 600; margin-right: 8px; }
  .guide button.alt { background: var(--purple); } .guide button.bc { background: var(--amber); }
  .guide button:hover { filter: brightness(1.2); }
  .muted { color: var(--dim); font-size: 12px; }
  .findings { color: var(--green); font-size: 12px; white-space: pre-wrap; }
</style>
</head>
<body>
<header>
  <h1>⚔ CTF 指挥台</h1>
  <span class="stat">已解: <b id="nSolved">0</b></span>
  <span class="stat">活跃 swarm: <b id="nActive">0</b></span>
  <span class="stat">花费: <b id="cost">$0</b></span>
  <span class="stat muted" id="clock"></span>
</header>
<main>
  <div id="challenges"><div class="muted">加载中…</div></div>
  <div id="detail"><div class="muted">左侧点选一个题目查看详情</div></div>
</main>
<script>
let STATE = null, SEL = null, SELSPEC = null;

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }

function dotClass(a) {
  if (!a) return 'd-dead';
  return 'd-run';
}

async function refresh() {
  STATE = await api('/api/state');
  document.getElementById('nSolved').textContent = STATE.solved.length;
  document.getElementById('cost').textContent = '$' + STATE.cost_usd.toFixed(3);
  const active = Object.values(STATE.swarms).filter(s => !s.cancelled).length;
  document.getElementById('nActive').textContent = active;
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  renderList();
  if (SEL) renderDetail();
}

function renderList() {
  const el = document.getElementById('challenges');
  const names = Object.keys(STATE.swarms).sort();
  const solvedSet = new Set(STATE.solved);
  let html = '';
  for (const name of names) {
    const s = STATE.swarms[name];
    const solved = solvedSet.has(name) || s.winner;
    const dots = Object.entries(s.agents).map(([sp, a]) =>
      `<span class="${s.winner ? (s.winner_spec === sp ? 'd-win' : 'd-dead') : 'd-run'}" title="${esc(sp)}"></span>`).join('');
    html += `<div class="card ${SEL === name ? 'sel' : ''}" onclick="sel('${esc(name)}')">
      <span class="status ${solved ? 'st-solved' : 'st-running'}">${solved ? '✔ 已解' : '● 进行中'}</span>
      <div class="name">${esc(name)}</div>
      <div class="cat">${esc(s.category)} · ${s.value}分 · bump ${s.bumps} 次</div>
      <div class="dots">${dots}</div>
    </div>`;
  }
  el.innerHTML = html || '<div class="muted">还没有 swarm</div>';
}

function sel(name) { SEL = name; SELSPEC = null; renderDetail(); }

async function renderDetail() {
  const s = STATE.swarms[SEL];
  if (!s) return;
  const el = document.getElementById('detail');
  let html = `<h2>${esc(SEL)}</h2>
    <div class="muted">${esc(s.category)} · ${s.value} 分 · 状态: ${s.winner ? '已解出 (' + esc(s.winner_spec) + ')' : (s.cancelled ? '已取消' : '进行中')}</div>`;
  if (s.winner) html += `<div style="color:var(--green);margin:8px 0">🚩 ${esc(s.winner)}</div>`;
  if (s.failed_approaches && s.failed_approaches.length)
    html += `<div style="color:var(--red);font-size:12px;margin:6px 0">已排除方向: ${esc(s.failed_approaches.join('、'))}</div>`;
  for (const [spec, a] of Object.entries(s.agents)) {
    const win = s.winner_spec === spec;
    html += `<div class="solver">
      <div><span class="spec">${esc(spec)}</span>
      <span class="muted" style="float:right">${win ? '🏆' : ''} ${esc(a.status)}</span></div>
      ${a.thinking ? `<div style="color:var(--purple);font-size:12px;margin-top:5px;white-space:pre-wrap">💭 ${esc(a.thinking)}</div>` : ''}
      <div class="last">最近动作: ${esc(a.last || '(无)')}</div>
      <div style="margin-top:6px">
        <button class="guide" style="padding:3px 10px;font-size:12px" onclick="showTrace('${esc(spec)}')">看实时轨迹</button>
        <button class="guide" style="padding:3px 10px;font-size:12px;background:var(--purple)" onclick="prepAdvise('${esc(spec)}')">给他提示</button>
      </div>
      <div class="trace" id="trace-${esc(spec)}" style="display:none"></div>
    </div>`;
  }
  html += `<div class="guide">
      <div style="color:var(--blue);margin-bottom:6px">📨 给模型喂提示（指导信箱，模型执行中就能收到）</div>
      <div class="muted">目标 solver: <span id="gTarget">$${SELSPEC ? esc(SELSPEC) : '(点上方"给他提示"选择)'}</span></div>
      <textarea id="gText" placeholder="例如：这个协议是 XXX，去查 /skills/ctf-web/xx.md 里的 Y 手法；或者：你已经有了 A，下一步该用 B 解码"></textarea>
      <button onclick="sendAdvice()">发送给该 solver</button>
      <button class="bc" onclick="sendBroadcast()">广播给本题全部</button>
      <div class="muted" style="margin-top:6px">提示内容会直接插进模型的工具结果里（每 3 步拉取一次）</div>
    </div>`;
  el.innerHTML = html;
}

function prepAdvise(spec) { SELSPEC = spec; const t = document.getElementById('gTarget'); if (t) t.textContent = spec; }

async function showTrace(spec) {
  SELSPEC = spec;
  const box = document.getElementById('trace-' + CSS.escape(spec));
  if (!box) return;
  if (box.style.display === 'none') {
    box.style.display = 'block';
    await loadTrace(spec, box);
  } else { box.style.display = 'none'; }
}

async function loadTrace(spec, box) {
  const d = await api('/api/trace?name=' + encodeURIComponent(SEL) + '&spec=' + encodeURIComponent(spec) + '&n=60');
  let html = '';
  for (const e of d.events) {
    const ts = new Date(e.ts * 1000).toLocaleTimeString();
    if (e.type === 'tool_call') html += `<div class="tc">[${ts}] ▶ ${esc(e.tool)}: ${esc(e.args)}</div>`;
    else if (e.type === 'tool_result') html += `<div class="tr">[${ts}] ◀ ${esc((e.result || '').slice(0, 240))}</div>`;
    else if (e.type === 'model_response') html += `<div style="color:var(--purple)">[${ts}] 💭 思路: ${esc(e.text || '')}</div>`;
    else if (e.type === 'flag_confirmed') html += `<div class="ok">[${ts}] 🚩 ${esc(e.flag || '')}</div>`;
    else if (e.type === 'bump') html += `<div class="warn">[${ts}] ⤴ bump: ${esc((e.insights || '').slice(0, 160))}</div>`;
    else if (e.type === 'advice_picked') html += `<div class="ok">[${ts}] 📨 收到指导 x${e.count}</div>`;
    else if (e.type === 'error') html += `<div class="err">[${ts}] ✖ ${esc(e.error || '')}</div>`;
    else html += `<div class="tr">[${ts}] ${esc(e.type)}</div>`;
  }
  box.innerHTML = html || '<div class="muted">暂无轨迹</div>';
  box.scrollTop = box.scrollHeight;
}

async function sendAdvice() {
  const text = document.getElementById('gText').value.trim();
  if (!text || !SELSPEC) { alert('先点"给他提示"选一个 solver，再写内容'); return; }
  await api('/api/advise', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ challenge: SEL, model_spec: SELSPEC, advice: text }) });
  document.getElementById('gText').value = '';
  alert('已投进信箱，模型下一个动作就能读到');
}

async function sendBroadcast() {
  const text = document.getElementById('gText').value.trim();
  if (!text) { alert('先写内容'); return; }
  await api('/api/broadcast', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ challenge: SEL, message: text }) });
  document.getElementById('gText').value = '';
  alert('已广播给本题所有 solver');
}

refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>
"""
