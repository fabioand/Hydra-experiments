"""Registry de dashboards de métricas/auditorias por experimento.

Fase 1:
- mantém manifests para `dashboard_runs` e `dashboard_audits`;
- cria HTMLs básicos (se não existirem);
- oferece API de registro resiliente (não deve quebrar avaliadores em caso de falha).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal

DashboardKind = Literal["runs", "audits"]

_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dashboard_dir(experiment_root: Path, kind: DashboardKind) -> Path:
    if kind == "runs":
        return experiment_root / "dashboard_runs"
    if kind == "audits":
        return experiment_root / "dashboard_audits"
    raise ValueError(f"Dashboard kind inválido: {kind}")


def _manifest_path(experiment_root: Path, kind: DashboardKind) -> Path:
    return _dashboard_dir(experiment_root, kind) / "manifest.json"


def _json_atomic_write(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {"schema_version": _SCHEMA_VERSION, "updated_at_utc": _utc_now_iso(), "items": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        # Fallback seguro: preserva backup e reinicia manifest.
        backup = path.with_suffix(path.suffix + ".corrupt.bak")
        try:
            path.replace(backup)
        except Exception:
            pass
        return {"schema_version": _SCHEMA_VERSION, "updated_at_utc": _utc_now_iso(), "items": []}

    if not isinstance(data, dict):
        return {"schema_version": _SCHEMA_VERSION, "updated_at_utc": _utc_now_iso(), "items": []}
    if "items" not in data or not isinstance(data["items"], list):
        data["items"] = []
    data["schema_version"] = _SCHEMA_VERSION
    return data


def rel_to_experiment(path: Path, experiment_root: Path) -> str:
    p = Path(path).resolve()
    exp = Path(experiment_root).resolve()
    try:
        return p.relative_to(exp).as_posix()
    except Exception:
        # fallback absoluto para não perder referência.
        return p.as_posix()


def ensure_dashboards(experiment_root: Path) -> None:
    experiment_root = Path(experiment_root)
    for kind in ("runs", "audits"):
        d = _dashboard_dir(experiment_root, kind)  # type: ignore[arg-type]
        d.mkdir(parents=True, exist_ok=True)
        idx = d / "index.html"
        if not idx.exists():
            if kind == "runs":
                idx.write_text(_RUNS_INDEX_HTML, encoding="utf-8")
            else:
                idx.write_text(_AUDITS_INDEX_HTML, encoding="utf-8")
        m = _manifest_path(experiment_root, kind)  # type: ignore[arg-type]
        if not m.exists():
            _json_atomic_write(
                m,
                {"schema_version": _SCHEMA_VERSION, "updated_at_utc": _utc_now_iso(), "items": []},
            )


def register_record(experiment_root: Path, kind: DashboardKind, record: Dict) -> None:
    experiment_root = Path(experiment_root)
    ensure_dashboards(experiment_root)

    manifest_p = _manifest_path(experiment_root, kind)
    manifest = _load_manifest(manifest_p)
    items: List[Dict] = manifest.get("items", [])

    rec = dict(record)
    rec.setdefault("created_at_utc", _utc_now_iso())
    if "id" not in rec:
        rid = "__".join(
            [
                str(rec.get("kind", "unknown")),
                str(rec.get("run_name", "")),
                str(rec.get("split", "")),
                str(rec.get("created_at_utc", "")),
            ]
        )
        rec["id"] = rid

    # Upsert por id
    by_id = {str(x.get("id")): i for i, x in enumerate(items)}
    rid = str(rec["id"])
    if rid in by_id:
        items[by_id[rid]] = rec
    else:
        items.append(rec)

    # Ordena do mais novo para o mais antigo (quando timestamp disponível)
    items.sort(key=lambda x: str(x.get("created_at_utc", "")), reverse=True)

    manifest["items"] = items
    manifest["updated_at_utc"] = _utc_now_iso()
    _json_atomic_write(manifest_p, manifest)


_RUNS_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hydra Dashboard Runs</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background:#0f1217; color:#e7edf5; }
    .wrap { padding: 14px; max-width: 1400px; margin: 0 auto; }
    h1 { margin: 0 0 8px; font-size: 22px; }
    .meta { color:#9fb0c4; font-size: 12px; margin-bottom: 10px; }
    .controls { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
    select,input { background:#151b22; color:#e7edf5; border:1px solid #2a3442; border-radius:6px; padding:7px 8px; }
    .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:10px; margin-bottom:14px; }
    .card { background:#151b22; border:1px solid #273241; border-radius:8px; padding:10px; }
    .k { color:#8da3be; font-size:12px; }
    .v { font-size:20px; margin-top:4px; font-weight:700; }
    table { width:100%; border-collapse:collapse; background:#151b22; border:1px solid #273241; border-radius:8px; overflow:hidden; }
    th, td { font-size:12px; padding:8px; border-bottom:1px solid #273241; text-align:left; vertical-align:top; }
    th { color:#a9bad0; background:#141920; position:sticky; top:0; }
    a { color:#7db3ff; text-decoration:none; }
    .hist { margin-top:14px; background:#151b22; border:1px solid #273241; border-radius:8px; padding:10px; }
    .bar { display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }
    .bar .w { height:12px; background:#4f83ff; border-radius:3px; }
    .muted { color:#9fb0c4; font-size:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Dashboard Runs</h1>
    <div class="meta">Fonte: <code>manifest.json</code> nesta pasta</div>
    <div class="controls">
      <input id="runQ" placeholder="Filtrar run..." />
      <select id="splitQ"><option value="">Todos splits</option></select>
      <select id="metricQ"></select>
    </div>
    <div class="cards" id="cards"></div>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Split</th>
          <th>Kind</th>
          <th>Métricas-chave</th>
          <th>Artefatos</th>
          <th>Data UTC</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
    <div class="hist">
      <div style="font-weight:700; margin-bottom:8px;">Histograma (itens filtrados)</div>
      <div class="muted" id="histMeta"></div>
      <div id="histBars"></div>
    </div>
  </div>
  <script>
    const fmt = (v) => (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(4) : String(v ?? '');
    const byId = (id) => document.getElementById(id);
    const els = { runQ: byId('runQ'), splitQ: byId('splitQ'), metricQ: byId('metricQ'), cards: byId('cards'), tbody: byId('tbody'), histMeta: byId('histMeta'), histBars: byId('histBars') };
    let rows = [];
    const metricCandidates = ['presence_f1_macro','presence_auc_macro','point_error_median_px','point_within_5px_rate','false_point_rate_gt_absent_global'];

    function uniq(a){ return [...new Set(a)].sort((x,y)=>String(x).localeCompare(String(y), undefined, {numeric:true})); }
    function pickMetric(row){
      const s = row.summary || {};
      for (const k of metricCandidates){ if (typeof s[k] === 'number' && Number.isFinite(s[k])) return k; }
      return '';
    }
    function linkList(artifacts){
      if (!artifacts || typeof artifacts !== 'object') return '';
      return Object.entries(artifacts).map(([k,v]) => `<a href="../${String(v)}" target="_blank">${k}</a>`).join(' | ');
    }
    function filtered(){
      const rq = els.runQ.value.trim().toLowerCase();
      const sq = els.splitQ.value;
      return rows.filter(r => (!rq || String(r.run_name||'').toLowerCase().includes(rq)) && (!sq || String(r.split||'')===sq));
    }
    function renderCards(data){
      if (!data.length){ els.cards.innerHTML = '<div class="muted">Sem itens.</div>'; return; }
      const latestByRun = new Map();
      for (const r of data){ if (!latestByRun.has(r.run_name)) latestByRun.set(r.run_name, r); }
      let vals = [];
      for (const r of latestByRun.values()){
        const s = r.summary || {};
        vals.push(['Runs', 1]);
        for (const k of metricCandidates){ if (typeof s[k] === 'number' && Number.isFinite(s[k])) vals.push([k, s[k]]); }
      }
      const agg = {};
      for (const [k,v] of vals){ (agg[k] ||= []).push(v); }
      const cards = [];
      for (const [k,arr] of Object.entries(agg)){
        const mean = arr.reduce((a,b)=>a+b,0)/arr.length;
        cards.push(`<div class="card"><div class="k">${k} (média)</div><div class="v">${fmt(mean)}</div><div class="muted">n=${arr.length}</div></div>`);
      }
      els.cards.innerHTML = cards.join('');
    }
    function renderTable(data){
      els.tbody.innerHTML = data.map(r => {
        const s = r.summary || {};
        const metrics = metricCandidates.filter(k => typeof s[k] === 'number' && Number.isFinite(s[k])).map(k => `${k}: ${fmt(s[k])}`).join('<br/>');
        return `<tr>
          <td>${r.run_name||''}</td>
          <td>${r.split||''}</td>
          <td>${r.kind||''}</td>
          <td>${metrics||''}</td>
          <td>${linkList(r.artifacts)}</td>
          <td>${r.created_at_utc||''}</td>
        </tr>`;
      }).join('');
    }
    function renderHist(data){
      const metric = els.metricQ.value;
      const vals = data.map(r => Number((r.summary||{})[metric])).filter(v => Number.isFinite(v));
      if (!metric || !vals.length){
        els.histMeta.textContent = 'Sem dados para histograma.';
        els.histBars.innerHTML = '';
        return;
      }
      const mn = Math.min(...vals), mx = Math.max(...vals);
      const bins = 8;
      const eps = 1e-12;
      const step = (mx - mn + eps) / bins;
      const counts = Array.from({length: bins}, () => 0);
      for (const v of vals){
        const b = Math.min(bins - 1, Math.floor((v - mn) / step));
        counts[b] += 1;
      }
      const maxC = Math.max(...counts, 1);
      els.histMeta.textContent = `${metric} | n=${vals.length} | min=${fmt(mn)} | max=${fmt(mx)}`;
      els.histBars.innerHTML = counts.map((c,i) => {
        const a = mn + i*step, b = a + step;
        const w = Math.max(2, Math.round((c/maxC)*460));
        return `<div class="bar"><div style="width:140px;">[${fmt(a)}, ${fmt(b)})</div><div class="w" style="width:${w}px"></div><div>${c}</div></div>`;
      }).join('');
    }
    function render(){
      const data = filtered();
      renderCards(data);
      renderTable(data);
      renderHist(data);
    }
    async function load(){
      const resp = await fetch('manifest.json', {cache:'no-store'});
      const m = await resp.json();
      rows = (m.items||[]).filter(x => (x.kind||'').includes('eval') || x.kind==='hydra_train_metrics');
      const splits = uniq(rows.map(r => r.split || '').filter(Boolean));
      for (const s of splits){ const o=document.createElement('option'); o.value=s; o.textContent=s; els.splitQ.appendChild(o); }
      const metricsAvail = uniq(rows.map(pickMetric).filter(Boolean));
      const metrics = metricsAvail.length ? metricsAvail : metricCandidates;
      els.metricQ.innerHTML = metrics.map(k => `<option value="${k}">${k}</option>`).join('');
      render();
    }
    for (const ev of ['input','change']){ els.runQ.addEventListener(ev, render); els.splitQ.addEventListener(ev, render); els.metricQ.addEventListener(ev, render); }
    load().catch(err => {
      els.cards.innerHTML = `<div class="muted">Falha ao carregar manifest.json: ${String(err)}</div>`;
    });
  </script>
</body>
</html>
"""


_AUDITS_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hydra Dashboard Audits</title>
  <style>
    body { margin:0; font-family: Arial, sans-serif; background:#0f1217; color:#e7edf5; }
    .wrap { padding:14px; max-width:1400px; margin:0 auto; }
    h1 { margin:0 0 8px; font-size:22px; }
    .meta { color:#9fb0c4; font-size:12px; margin-bottom:10px; }
    .controls { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
    select,input { background:#151b22; color:#e7edf5; border:1px solid #2a3442; border-radius:6px; padding:7px 8px; }
    .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:10px; margin-bottom:14px; }
    .card { background:#151b22; border:1px solid #273241; border-radius:8px; padding:10px; }
    table { width:100%; border-collapse:collapse; background:#151b22; border:1px solid #273241; border-radius:8px; overflow:hidden; }
    th, td { font-size:12px; padding:8px; border-bottom:1px solid #273241; text-align:left; vertical-align:top; }
    th { color:#a9bad0; background:#141920; position:sticky; top:0; }
    a { color:#7db3ff; text-decoration:none; }
    .bar { display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }
    .w { height:12px; background:#4f83ff; border-radius:3px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Dashboard Auditorias</h1>
    <div class="meta">Fonte: <code>manifest.json</code> nesta pasta</div>
    <div class="controls">
      <input id="kindQ" placeholder="Filtrar kind..." />
      <select id="splitQ"><option value="">Todos splits</option></select>
    </div>
    <div class="cards" id="cards"></div>
    <table>
      <thead>
        <tr>
          <th>Kind</th>
          <th>Run</th>
          <th>Split</th>
          <th>Resumo</th>
          <th>Artefatos</th>
          <th>Data UTC</th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>
  <script>
    const byId = (id) => document.getElementById(id);
    const els = { kindQ: byId('kindQ'), splitQ: byId('splitQ'), cards: byId('cards'), tbody: byId('tbody') };
    let rows = [];
    function uniq(a){ return [...new Set(a)].sort((x,y)=>String(x).localeCompare(String(y), undefined, {numeric:true})); }
    function linkList(artifacts){
      if (!artifacts || typeof artifacts !== 'object') return '';
      return Object.entries(artifacts).map(([k,v]) => `<a href="../${String(v)}" target="_blank">${k}</a>`).join(' | ');
    }
    function filtered(){
      const kq = els.kindQ.value.trim().toLowerCase();
      const sq = els.splitQ.value;
      return rows.filter(r => (!kq || String(r.kind||'').toLowerCase().includes(kq)) && (!sq || String(r.split||'')===sq));
    }
    function renderCards(data){
      const byKind = {};
      for (const r of data){ const k = r.kind || 'unknown'; byKind[k] = (byKind[k] || 0) + 1; }
      const entries = Object.entries(byKind).sort((a,b)=>b[1]-a[1]);
      if (!entries.length){ els.cards.innerHTML = '<div>Sem itens.</div>'; return; }
      const maxV = Math.max(...entries.map(x=>x[1]), 1);
      els.cards.innerHTML = entries.map(([k,v]) => {
        const w = Math.max(2, Math.round((v/maxV)*180));
        return `<div class="card"><div style="font-weight:700">${k}</div><div class="bar"><div class="w" style="width:${w}px"></div><div>${v}</div></div></div>`;
      }).join('');
    }
    function renderTable(data){
      els.tbody.innerHTML = data.map(r => {
        const s = r.summary || {};
        const summaryTxt = Object.entries(s).slice(0,6).map(([k,v]) => `${k}: ${typeof v==='number' ? v.toFixed(4) : v}`).join('<br/>');
        return `<tr>
          <td>${r.kind||''}</td>
          <td>${r.run_name||''}</td>
          <td>${r.split||''}</td>
          <td>${summaryTxt}</td>
          <td>${linkList(r.artifacts)}</td>
          <td>${r.created_at_utc||''}</td>
        </tr>`;
      }).join('');
    }
    function render(){ const data = filtered(); renderCards(data); renderTable(data); }
    async function load(){
      const resp = await fetch('manifest.json', {cache:'no-store'});
      const m = await resp.json();
      rows = (m.items||[]).filter(x => !(String(x.kind||'').includes('eval') && String(x.kind||'')==='hydra_eval'));
      const splits = uniq(rows.map(r => r.split || '').filter(Boolean));
      for (const s of splits){ const o=document.createElement('option'); o.value=s; o.textContent=s; els.splitQ.appendChild(o); }
      render();
    }
    for (const ev of ['input','change']){ els.kindQ.addEventListener(ev, render); els.splitQ.addEventListener(ev, render); }
    load().catch(err => { els.cards.innerHTML = `<div>Falha ao carregar manifest: ${String(err)}</div>`; });
  </script>
</body>
</html>
"""

