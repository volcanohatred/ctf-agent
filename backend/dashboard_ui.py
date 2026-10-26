"""Self-contained single-page dashboard UI (served by web_dashboard.py)."""

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CTF Agent · 指挥台</title>
<style>
  :root {
    --bg: #eef1f5; --card: #ffffff; --border: #e2e8f0; --fg: #1e293b;
    --dim: #64748b; --green: #16a34a; --red: #dc2626; --amber: #d97706;
    --blue: #2563eb; --purple: #7c3aed; --teal: #0d9488;
    --shadow: 0 1px 3px rgba(15,23,42,.08), 0 1px 2px rgba(15,23,42,.04);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font: 14px/1.55 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif; }
  code, .mono { font-family: "JetBrains Mono","Consolas",monospace; }
  header { background: linear-gradient(120deg,#1e293b,#0f172a); color:#e2e8f0;
           padding: 14px 22px; display:flex; gap:26px; align-items:center; flex-wrap:wrap;
           box-shadow: var(--shadow); position: sticky; top:0; z-index: 9; }
  header h1 { font-size: 17px; color:#fff; letter-spacing:.5px; }
  .hstat { color:#94a3b8; font-size:13px; }
  .hstat b { color:#4ade80; font-size:15px; }
  .hstat b.blue { color:#60a5fa; } .hstat b.amber { color:#fbbf24; }
  main { display:grid; grid-template-columns: 360px 1fr; gap:14px; padding:14px; height: calc(100vh - 58px); }
  #challenges { overflow-y:auto; }
  .card { background: var(--card); border:1px solid var(--border); border-radius:10px;
          padding:12px 14px; margin-bottom:10px; cursor:pointer; box-shadow: var(--shadow);
          transition: transform .08s, border-color .15s; }
  .card:hover { transform: translateY(-1px); border-color: var(--blue); }
  .card.sel { border-color: var(--blue); border-width:2px; }
  .card .name { font-weight:600; }
  .card .cat { color: var(--purple); font-size:12px; }
  .badge { float:right; font-size:11px; padding:2px 9px; border-radius:99px; font-weight:600; }
  .b-solved { background:#dcfce7; color:var(--green); }
  .b-run { background:#fef3c7; color:var(--amber); }
  .b-stuck { background:#fee2e2; color:var(--red); }
  .meta { color: var(--dim); font-size:12px; margin-top:3px; }
  .dots span { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; }
  .d-run { background: var(--amber); } .d-win { background: var(--green); } .d-dead { background:#cbd5e1; }
  #detail { overflow-y:auto; }
  .panel { background: var(--card); border:1px solid var(--border); border-radius:10px;
           padding:16px 18px; box-shadow: var(--shadow); margin-bottom:14px; }
  .kv { display:inline-block; margin-right:18px; font-size:13px; color:var(--dim); }
  .kv b { color:var(--fg); font-size:14px; }
  table.solvers { width:100%; border-collapse: collapse; margin-top:10px; font-size:12.5px; }
  table.solvers th { text-align:left; color:var(--dim); font-weight:600; font-size:11px;
    text-transform:uppercase; letter-spacing:.4px; padding:6px 8px; border-bottom:1px solid var(--border); }
  table.solvers td { padding:8px; border-bottom:1px solid #f1f5f9; vertical-align:top; }
  .tag { font-size:11px; padding:1px 7px; border-radius:99px; }
  .t-run { background:#fef3c7; color:var(--amber); } .t-win { background:#dcfce7; color:var(--green); }
  .t-idle { background:#e2e8f0; color:var(--dim); } .t-stuck { background:#fee2e2; color:var(--red); }
  .thinking { color: var(--purple); font-size:12px; white-space:pre-wrap; max-width:420px; }
  .last { color: var(--dim); font-size:12px; word-break:break-all; max-width:280px; }
  .approach { color: var(--teal); font-size:12px; }
  button { background: var(--blue); color:#fff; border:0; border-radius:6px; padding:6px 12px;
           cursor:pointer; font-size:12px; font-weight:600; }
  button:hover { filter: brightness(1.1); }
  button.ghost { background:#e2e8f0; color:var(--fg); }
  button.warn { background: var(--amber); }
  button.purp { background: var(--purple); }
  .trace { background:#0f172a; color:#cbd5e1; border-radius:8px; padding:10px; margin-top:8px;
           max-height:340px; overflow-y:auto; font-size:12px; font-family:"JetBrains Mono",monospace; }
  .trace .tc { color:#7dd3fc; } .trace .tr { color:#94a3b8; }
  .trace .ok { color:#4ade80; } .trace .warn { color:#fbbf24; } .trace .err { color:#f87171; }
  .trace .think { color:#c4b5fd; }
  .guide textarea, .guide select, .guide input { width:100%; background:#fff; color:var(--fg);
    border:1px solid var(--border); border-radius:6px; padding:8px; margin:4px 0; font:inherit; font-size:13px; }
  .guide textarea { min-height:96px; resize:vertical; }
  .muted { color: var(--dim); font-size:12px; }
  .ledger { color: var(--red); font-size:12px; }
  .flagbox { background:#f0fdf4; border:1px solid #bbf7d0; color:var(--green); padding:8px 12px;
             border-radius:8px; margin:10px 0; font-weight:600; word-break:break-all; }
  #toast { position:fixed; bottom:20px; right:20px; background:#0f172a; color:#fff; padding:10px 16px;
           border-radius:8px; font-size:13px; opacity:0; transition:opacity .25s; z-index:99; }
</style>
</head>
<body>
<header>
  <h1>⚔ CTF 指挥台</h1>
  <span class="hstat">已解 <b id="nSolved">0</b></span>
  <span class="hstat">活跃 <b class="blue" id="nActive">0</b></span>
  <span class="hstat">总成本 <b class="amber" id="cost">$0</b></span>
  <span class="hstat" id="clock" style="margin-left:auto"></span>
</header>
<main>
  <div id="challenges"><div class="muted" style="padding:20px">加载中…</div></div>
  <div id="detail"><div class="muted" style="padding:20px">← 点选一个题目看详情与喂提示</div></div>
</main>
<div id="toast"></div>
<script>
let STATE = null, SEL = null, SELSPEC = null;
const openTraces = new Set();   // trace panels to keep open across refreshes
async function api(path, opts) { const r = await fetch(path, opts); return r.json(); }
function esc(s) { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; }
function toast(msg) { const t = document.getElementById('toast'); t.textContent = msg; t.style.opacity = 1; setTimeout(() => t.style.opacity = 0, 2200); }
function fmtAgo(ts) {
  if (!ts) return '—';
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  return s < 60 ? s + 's前' : s < 3600 ? Math.floor(s / 60) + 'm前' : Math.floor(s / 3600) + 'h前';
}

async function refresh() {
  try { STATE = await api('/api/state'); } catch (e) { return; }
  document.getElementById('nSolved').textContent = STATE.solved.length;
  document.getElementById('cost').textContent = '$' + STATE.cost_usd.toFixed(2);
  document.getElementById('nActive').textContent = Object.values(STATE.swarms).filter(s => !s.cancelled).length;
  document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  renderList();
  if (SEL) { renderDetail().then(restoreTraces); } else { restoreTraces(); }
}

function renderList() {
  const el = document.getElementById('challenges');
  const names = Object.keys(STATE.swarms).sort((a, b) => {
    const sa = STATE.swarms[a], sb = STATE.swarms[b];
    const ra = sa.pending ? 2 : (sa.winner ? 1 : 0);
    const rb = sb.pending ? 2 : (sb.winner ? 1 : 0);
    return ra - rb;  // running first, then solved, then pending
  });
  const solvedSet = new Set(STATE.solved);
  let html = '';
  for (const name of names) {
    const s = STATE.swarms[name];
    const solved = solvedSet.has(name) || s.winner;
    const pending = !!s.pending;
    const stuck = !solved && !pending && Object.values(s.agents).some(a => a.quiet_s && a.quiet_s > 240);
    const badge = pending ? '<span class="badge" style="background:#e2e8f0;color:var(--dim)">待解题</span>'
      : solved ? '<span class="badge b-solved">✔ 已解</span>'
      : stuck ? '<span class="badge b-stuck">⚠ 疑似卡死</span>'
      : '<span class="badge b-run">● 进行中</span>';
    const cost = Object.values(s.agents || {}).reduce((x, a) => x + (a.cost_usd || 0), 0);
    const dots = Object.entries(s.agents || {}).map(([sp, a]) =>
      `<span class="${s.winner ? (s.winner_spec === sp ? 'd-win' : 'd-dead') : 'd-run'}" title="${esc(sp)}"></span>`).join('');
    const dimStyle = pending ? 'opacity:.55' : (solved ? 'opacity:.8' : '');
    html += `<div class="card ${SEL === name ? 'sel' : ''}" style="${dimStyle}" onclick="sel('${esc(name)}')">
      ${badge}
      <div class="name">${esc(name)}</div>
      <div class="meta">${esc(s.category)} · ${s.value}分 · bump ${s.bumps || 0}${cost ? ' · $' + cost.toFixed(3) : ''}</div>
      <div class="dots">${dots}</div>
    </div>`;
  }
  el.innerHTML = html || '<div class="muted" style="padding:20px">还没有题目</div>';
}

function sel(name) { SEL = name; SELSPEC = null; renderDetail(); }

async function renderDetail() {
  const s = STATE.swarms[SEL];
  if (!s) return;
  const el = document.getElementById('detail');
  const cost = Object.values(s.agents || {}).reduce((x, a) => x + (a.cost_usd || 0), 0);
  const statusTxt = s.pending ? '⏸ 待解题（尚未 spawn）' : s.winner ? '✅ 已解出' : (s.cancelled ? '⏹ 已停止' : '▶ 进行中');
  let html = `<div class="panel">
    <h2 style="font-size:16px">${esc(SEL)}</h2>
    <div style="margin-top:6px">
      <span class="kv">类别 <b>${esc(s.category)}</b></span>
      <span class="kv">分值 <b>${s.value}</b></span>
      <span class="kv">bump <b>${s.bumps || 0}</b> 次</span>
      <span class="kv">本题成本 <b>$${cost.toFixed(3)}</b></span>
      <span class="kv">状态 <b>${statusTxt}</b></span>
    </div>`;
  if (s.winner) html += `<div class="flagbox">🚩 ${esc(s.winner)}<span class="muted" style="font-weight:400">　by ${esc(s.winner_spec || '')}</span></div>`;
  if (s.failed_approaches && s.failed_approaches.length)
    html += `<div class="ledger" style="margin-top:8px">🚫 已排除方向：${esc(s.failed_approaches.join('、'))}</div>`;
  html += `</div>`;

  if (s.pending) {
    html += `<div class="panel muted">这道题还没 spawn swarm。等协调器排上它，或点下方手动 spawn。</div>`;
    el.innerHTML = html;
    return;
  }
  const agentEntries = Object.entries(s.agents || {});
  if (agentEntries.length === 0) {
    html += `<div class="panel muted">此题已完成/归档，活跃 solver 数据已回收（历史 trace 在 logs/ 目录）。</div>`;
    el.innerHTML = html;
    return;
  }

  // per-solver table
  html += `<div class="panel"><table class="solvers"><thead><tr>
    <th>模型</th><th>状态</th><th>💭 当前思路</th><th>最近动作</th><th>攻坚方向</th><th>步数</th><th>成本</th><th>静默</th><th></th>
  </tr></thead><tbody>`;
  for (const [spec, a] of Object.entries(s.agents)) {
    const win = s.winner_spec === spec;
    const stuckTag = (a.quiet_s && a.quiet_s > 240) ? '<span class="tag t-stuck">卡死?</span>'
      : (a.status === 'running' ? '<span class="tag t-run">运行</span>' : '<span class="tag t-idle">—</span>');
    const short = spec.replace('claude-sdk/', '');
    html += `<tr>
      <td class="mono">${esc(short)}${win ? ' 🏆' : ''}</td>
      <td>${stuckTag}</td>
      <td class="thinking">${esc((a.thinking || '').slice(0, 140)) || '<span class="muted">—</span>'}</td>
      <td class="last mono">${esc((a.last || '').slice(0, 90)) || '—'}</td>
      <td class="approach">${esc(a.approach ? a.approach.split('/').pop() : '—')}</td>
      <td>${a.steps || 0}</td>
      <td>$${(a.cost_usd || 0).toFixed(3)}</td>
      <td class="muted">${a.quiet_s != null ? fmtAgo(a.last_ts) : '—'}</td>
      <td><button class="ghost" onclick="event.stopPropagation();showTrace(this.dataset.spec)" data-spec="${esc(spec)}">轨迹</button>
          <button class="purp" onclick="event.stopPropagation();prepAdvise(this.dataset.spec)" data-spec="${esc(spec)}">提示</button></td>
    </tr><tr style="display:none" id="tracerow-${esc(spec)}"><td colspan="9"><div class="trace" id="trace-${esc(spec)}"></div></td></tr>`;
  }
  html += `</tbody></table></div>`;

  // shared-experience feed (blackboard): facts + dead-ends + broadcasts
  if (s.shared_feed && s.shared_feed.length) {
    html += `<div class="panel"><div style="color:var(--blue);font-weight:600;margin-bottom:6px">📋 共享经验与踩坑记录（模型间的黑板）</div><div style="max-height:240px;overflow-y:auto">`;
    for (const f of [...s.shared_feed].reverse()) {
      const ts = new Date(f.ts * 1000).toLocaleTimeString();
      const mdl = (f.model || '').replace('claude-sdk/', '');
      if (f.kind === 'excluded') html += `<div style="color:var(--red);font-size:12px;margin:3px 0">✗ [${ts}] <b>${esc(mdl)}</b> 排除: ${esc(f.content)}</div>`;
      else if (f.kind === 'broadcast') html += `<div style="color:var(--amber);font-size:12px;margin:3px 0">📢 [${ts}] <b>${esc(mdl)}</b>: ${esc(f.content)}</div>`;
      else html += `<div style="color:var(--green);font-size:12px;margin:3px 0">✓ [${ts}] <b>${esc(mdl)}</b>: ${esc(f.content)}</div>`;
    }
    html += `</div></div>`;
  }

  // guidance panel
  html += `<div class="panel guide">
      <div style="color:var(--blue);font-weight:600;margin-bottom:6px">📨 喂提示 / 指导（指导信箱——模型执行中就能收到）</div>
      <div class="muted">目标 solver：<b id="gTarget">${SELSPEC ? esc(SELSPEC) : '(点上表"提示"选择)'}</b></div>
      <textarea id="gText" placeholder="例如：这个协议是 XX，格式是…，去读 /skills/ctf-web/xx.md 的 Y 手法；或者：你已经拿到 A，下一步用 B 解"></textarea>
      <button onclick="sendAdvice()">📨 发给该 solver</button>
      <button class="warn" onclick="sendBroadcast()">📢 广播给本题全部</button>
      <div class="muted" style="margin-top:6px">提示会插进模型的工具结果里（每 3 步拉取一次）；写入即达，不用等重开</div>
    </div>`;
  el.innerHTML = html;
}

function prepAdvise(spec) { SELSPEC = spec; const t = document.getElementById('gTarget'); if (t) t.textContent = spec; toast('已选定 ' + spec + '，写内容后点发送'); }

async function showTrace(spec) {
  SELSPEC = spec;
  const row = document.getElementById('tracerow-' + spec);
  const box = document.getElementById('trace-' + spec);
  if (!row || !box) return;
  if (row.style.display === 'none' || row.style.display === '') {
    openTraces.add(spec);
    row.style.display = '';
    await loadTrace(spec, box);
  } else {
    openTraces.delete(spec);
    row.style.display = 'none';
  }
}

// After each refresh re-render, restore any open trace panels (and reload content)
async function restoreTraces() {
  for (const spec of openTraces) {
    const row = document.getElementById('tracerow-' + spec);
    const box = document.getElementById('trace-' + spec);
    if (row && box) {
      row.style.display = '';
      await loadTrace(spec, box);
    }
  }
}

async function loadTrace(spec, box) {
  const d = await api('/api/trace?name=' + encodeURIComponent(SEL) + '&spec=' + encodeURIComponent(spec) + '&n=80');
  let html = '';
  for (const e of d.events) {
    const ts = new Date(e.ts * 1000).toLocaleTimeString();
    if (e.type === 'tool_call') html += `<div class="tc">[${ts}] ▶ ${esc(e.tool)}: ${esc(e.args)}</div>`;
    else if (e.type === 'tool_result') html += `<div class="tr">[${ts}] ◀ ${esc((e.result || '').slice(0, 240))}</div>`;
    else if (e.type === 'model_response') html += `<div class="think">[${ts}] 💭 ${esc(e.text || '')}</div>`;
    else if (e.type === 'flag_confirmed') html += `<div class="ok">[${ts}] 🚩 ${esc(e.flag || '')}</div>`;
    else if (e.type === 'bump') html += `<div class="warn">[${ts}] ⤴ bump</div>`;
    else if (e.type === 'advice_picked') html += `<div class="ok">[${ts}] 📨 收到指导 x${e.count}</div>`;
    else if (e.type === 'error') html += `<div class="err">[${ts}] ✖ ${esc(e.error || '')}</div>`;
    else html += `<div class="tr">[${ts}] ${esc(e.type)}</div>`;
  }
  box.innerHTML = html || '<div class="muted">暂无轨迹</div>';
  box.scrollTop = box.scrollHeight;
}

async function sendAdvice() {
  const text = document.getElementById('gText').value.trim();
  if (!text || !SELSPEC) { toast('先选 solver 再写内容'); return; }
  await api('/api/advise', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ challenge: SEL, model_spec: SELSPEC, advice: text }) });
  document.getElementById('gText').value = '';
  toast('📨 已投进信箱，模型下个动作就能读到');
}

async function sendBroadcast() {
  const text = document.getElementById('gText').value.trim();
  if (!text) { toast('先写内容'); return; }
  await api('/api/broadcast', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ challenge: SEL, message: text }) });
  document.getElementById('gText').value = '';
  toast('📢 已广播给本题所有 solver');
}

refresh();
setInterval(refresh, 2500);
</script>
</body>
</html>
"""
