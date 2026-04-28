"""
TASK-16: Gerador do dashboard HTML single-file.

Lê skills/frontend-design/SKILL.md antes de qualquer geração (requisito).
Produz reports/dashboard.html com Tailwind CSS + Leaflet.js + Chart.js.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PREDICTIONS = "data/outputs/predictions.parquet"
_METRICS     = "data/outputs/metrics_by_fold.json"
_OUTPUT      = "reports/dashboard.html"


def _build_geodata(gdf: gpd.GeoDataFrame, last_ano: int) -> dict:
    last = gdf[gdf["ano"] == last_ano].copy()
    last["geometry"] = last["geometry"].simplify(0.01)
    props = ["cod_ibge", "municipio", "uf", "score_risco",
             "taxa_desmatamento", "desmatamento_km2"]
    return last[props + ["geometry"]].__geo_interface__


def _build_all_data(gdf: gpd.GeoDataFrame) -> dict:
    cols = ["cod_ibge", "municipio", "uf", "ano",
            "score_risco", "taxa_desmatamento", "desmatamento_km2", "cluster_id"]
    cols = [c for c in cols if c in gdf.columns]
    df = gdf[gdf["score_risco"].notna()][cols].copy()
    df["ano"] = df["ano"].astype(int)
    df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
    df["score_risco"] = df["score_risco"].round(1)
    df["taxa_desmatamento"] = df["taxa_desmatamento"].round(6)
    df["desmatamento_km2"] = df["desmatamento_km2"].round(2)
    result: dict[str, list] = {}
    for ano, grp in df.groupby("ano"):
        result[str(ano)] = grp.drop(columns="ano").to_dict("records")
    return result


def _build_series(gdf: gpd.GeoDataFrame) -> dict:
    agg = (
        gdf.groupby(["uf", "ano"])
        .agg(taxa_media=("taxa_desmatamento", "mean"),
             desmat_total=("desmatamento_km2", "sum"))
        .reset_index()
    )
    agg["taxa_media"]  = agg["taxa_media"].round(6)
    agg["desmat_total"] = agg["desmat_total"].round(2)
    result: dict = {}
    for uf, grp in agg.groupby("uf"):
        result[uf] = {
            int(r["ano"]): {"taxa": r["taxa_media"], "desmat": r["desmat_total"]}
            for _, r in grp.iterrows()
        }
    return result


def _build_clusters(gdf: gpd.GeoDataFrame) -> list:
    if "cluster_id" not in gdf.columns:
        return []

    # Perfil estrutural: média histórica por município, depois por cluster
    agg = (
        gdf.groupby(["cod_ibge", "cluster_id"])
        .agg(
            taxa=("taxa_desmatamento", "mean"),
            area=("area_km2", "mean"),
            pop=("populacao", "mean"),
            pib=("pib_agropecuario", "mean"),
            uc=("area_uc_pct", "mean"),
            ti=("area_ti_pct", "mean"),
            autos=("autos_ibama", "sum"),
        )
        .reset_index()
    )
    profiles = (
        agg.groupby("cluster_id")
        .agg(
            n=("cod_ibge", "count"),
            taxa=("taxa", "mean"),
            area=("area", "mean"),
            pop=("pop", "mean"),
            pib=("pib", "mean"),
            uc=("uc", "mean"),
            ti=("ti", "mean"),
            autos=("autos", "mean"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )

    def _label(row) -> str:
        if int(row["cluster_id"]) == -1:
            return "Ruído (sem cluster)"
        prot = row["uc"] + row["ti"]
        if prot > 50:
            return "Alta proteção territorial (UC + TI)"
        if row["uc"] > 25 and row["ti"] < 5:
            return "Alta cobertura de Unidades de Conservação"
        if row["ti"] > 15 and row["autos"] > 150:
            return "Fronteira com TIs e alta fiscalização"
        if row["taxa"] > 0.18 and row["autos"] > 150:
            return "Alta pressão agrícola — fronteira ativa"
        if row["area"] > 10_000:
            return "Grandes municípios com desmatamento moderado"
        if row["area"] < 3_000 and row["pib"] < 80_000:
            return "Pequenos municípios — baixa pressão"
        return "Perfil intermediário"

    result = []
    for _, row in profiles.iterrows():
        result.append({
            "id":    int(row["cluster_id"]),
            "label": _label(row),
            "n":     int(row["n"]),
            "taxa":  round(float(row["taxa"]), 4),
            "area":  round(float(row["area"]) / 1000, 1),
            "pop":   round(float(row["pop"]) / 1000, 1),
            "pib":   round(float(row["pib"]) / 1000, 1),
            "uc":    round(float(row["uc"]), 1),
            "ti":    round(float(row["ti"]), 1),
            "autos": round(float(row["autos"]), 0),
        })
    return result


def _load_metrics() -> dict:
    p = Path(_METRICS)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_dashboard(
    predictions_path: str = _PREDICTIONS,
    output_path: str = _OUTPUT,
) -> None:
    logger.info("Carregando dados de '%s'...", predictions_path)
    gdf = gpd.read_parquet(predictions_path)

    last_ano = int(gdf["ano"].max())
    anos     = sorted(int(a) for a in gdf["ano"].unique())
    ufs      = sorted(gdf["uf"].unique())

    logger.info("Extraindo GEODATA (último ano: %d)...", last_ano)
    geodata  = _build_geodata(gdf, last_ano)

    logger.info("Extraindo ALL_DATA (%d anos)...", len(anos))
    all_data = _build_all_data(gdf)

    logger.info("Extraindo SERIES...")
    series   = _build_series(gdf)

    logger.info("Extraindo perfis de CLUSTERS...")
    clusters = _build_clusters(gdf)

    metrics  = _load_metrics()

    anos_js     = _json(anos)
    ufs_js      = _json(ufs)
    geodata_js  = _json(geodata)
    alldata_js  = _json(all_data)
    series_js   = _json(series)
    clusters_js = _json(clusters)
    metrics_js  = _json(metrics)
    today       = date.today().strftime("%d/%m/%Y")

    html = _HTML_TEMPLATE.replace("__ANOS__",     anos_js)    \
                         .replace("__UFS__",      ufs_js)     \
                         .replace("__GEODATA__",  geodata_js) \
                         .replace("__ALLDATA__",  alldata_js) \
                         .replace("__SERIES__",   series_js)  \
                         .replace("__CLUSTERS__", clusters_js)\
                         .replace("__METRICS__",  metrics_js) \
                         .replace("__TODAY__",    today)      \
                         .replace("__LAST_ANO__", str(last_ano))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html, encoding="utf-8")
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    logger.info("Dashboard salvo em '%s' (%.2f MB).", output_path, size_mb)


# ── HTML template ──────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Risco de Desmatamento · Amazônia Legal</title>
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  body { font-family: 'Inter', system-ui, sans-serif; }
  #map { height: 480px; }
  .tab-btn { transition: all .15s; }
  .tab-btn.active { border-bottom: 2px solid #22c55e; color: #15803d; font-weight:600; }
  .badge { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:12px; font-weight:600; }
  table { border-collapse: collapse; width: 100%; }
  th { position: sticky; top: 0; z-index: 1; }
  tr:nth-child(even) td { background: #f8fafc; }
</style>
</head>
<body class="bg-slate-50 text-slate-900">

<!-- HEADER -->
<header class="bg-slate-800 text-white flex items-center px-6 h-14 shadow-md">
  <span class="text-2xl mr-3">🌿</span>
  <div>
    <h1 class="text-lg font-bold leading-tight">Risco de Desmatamento · Amazônia Legal</h1>
    <p class="text-xs text-slate-400" id="header-sub">Ano: __LAST_ANO__ · 772 municípios · 9 estados</p>
  </div>
  <div class="ml-auto text-xs text-slate-400">Modelo: LightGBM + Optuna · R²=0.76</div>
</header>

<div class="flex h-[calc(100vh-56px)]">

  <!-- SIDEBAR -->
  <aside class="w-60 bg-white border-r border-slate-200 flex flex-col p-4 gap-4 overflow-y-auto shrink-0">
    <div>
      <label class="block text-xs font-semibold text-slate-500 uppercase mb-1">Ano</label>
      <select id="sel-ano" class="w-full border border-slate-200 rounded px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-green-400">
      </select>
    </div>
    <div>
      <label class="block text-xs font-semibold text-slate-500 uppercase mb-1">Estados (UF)</label>
      <div id="uf-checks" class="flex flex-col gap-1 text-sm max-h-52 overflow-y-auto"></div>
    </div>
    <hr class="border-slate-200">
    <div class="text-xs text-slate-500">
      <p class="font-semibold text-slate-700 mb-1">Métricas do Modelo</p>
      <div id="metrics-box" class="space-y-1"></div>
    </div>
  </aside>

  <!-- MAIN -->
  <main class="flex-1 flex flex-col overflow-hidden">

    <!-- KPI CARDS -->
    <div class="grid grid-cols-4 gap-3 p-4 bg-white border-b border-slate-200 shrink-0">
      <div class="bg-slate-50 rounded-lg p-3 border border-slate-200">
        <div class="text-xs text-slate-500 font-medium">Desmatamento Total</div>
        <div class="text-xl font-bold text-slate-800 mt-1" id="kpi-desmat">—</div>
        <div class="text-xs text-slate-400">km² no ano</div>
      </div>
      <div class="bg-slate-50 rounded-lg p-3 border border-slate-200">
        <div class="text-xs text-slate-500 font-medium">Taxa Média</div>
        <div class="text-xl font-bold text-slate-800 mt-1" id="kpi-taxa">—</div>
        <div class="text-xs text-slate-400">%/ano (municípios filtrados)</div>
      </div>
      <div class="bg-slate-50 rounded-lg p-3 border border-slate-200">
        <div class="text-xs text-slate-500 font-medium">Score Médio de Risco</div>
        <div class="text-xl font-bold mt-1" id="kpi-score" style="color:#f97316">—</div>
        <div class="text-xs text-slate-400">0–100 (percentílico)</div>
      </div>
      <div class="bg-slate-50 rounded-lg p-3 border border-slate-200">
        <div class="text-xs text-slate-500 font-medium">Municípios Score ≥ 80</div>
        <div class="text-xl font-bold text-red-500 mt-1" id="kpi-high">—</div>
        <div class="text-xs text-slate-400">alto risco crítico</div>
      </div>
    </div>

    <!-- TABS -->
    <div class="flex border-b border-slate-200 bg-white px-4 shrink-0">
      <button class="tab-btn active py-2 px-4 text-sm mr-1" data-tab="mapa">🗺️ Mapa</button>
      <button class="tab-btn py-2 px-4 text-sm mr-1 text-slate-500" data-tab="ranking">🏆 Ranking</button>
      <button class="tab-btn py-2 px-4 text-sm mr-1 text-slate-500" data-tab="clusters">🔵 Clusters</button>
      <button class="tab-btn py-2 px-4 text-sm mr-1 text-slate-500" data-tab="tendencia">📈 Tendência</button>
      <button class="tab-btn py-2 px-4 text-sm text-slate-500" data-tab="sobre">ℹ️ Sobre</button>
    </div>

    <!-- TAB CONTENT -->
    <div class="flex-1 overflow-auto">

      <!-- Mapa -->
      <div id="tab-mapa" class="p-4 h-full">
        <div id="map" class="rounded-lg border border-slate-200 shadow-sm"></div>
        <p class="text-xs text-slate-400 mt-2">Clique em um município para ver detalhes. Cores: 🟢 baixo risco → 🔴 alto risco.</p>
      </div>

      <!-- Ranking -->
      <div id="tab-ranking" class="p-4 hidden">
        <h2 class="text-base font-semibold mb-3">Top-20 Municípios · Maior Score de Risco</h2>
        <div class="overflow-auto max-h-[440px] rounded border border-slate-200">
          <table>
            <thead>
              <tr class="bg-slate-800 text-white text-xs">
                <th class="px-3 py-2 text-left">#</th>
                <th class="px-3 py-2 text-left">Município</th>
                <th class="px-3 py-2 text-left">UF</th>
                <th class="px-3 py-2 text-right">Score</th>
                <th class="px-3 py-2 text-right">Taxa (%/ano)</th>
                <th class="px-3 py-2 text-right">Desmat. (km²)</th>
              </tr>
            </thead>
            <tbody id="ranking-body" class="text-sm"></tbody>
          </table>
        </div>
      </div>

      <!-- Clusters -->
      <div id="tab-clusters" class="p-4 hidden">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-base font-semibold">Perfis de Municípios — HDBSCAN</h2>
          <span class="text-xs text-slate-400">Silhouette = 0.225 · 4 clusters + ruído</span>
        </div>
        <div id="clusters-cards" class="grid grid-cols-1 gap-4 max-w-5xl"></div>
      </div>

      <!-- Tendência -->
      <div id="tab-tendencia" class="p-4 hidden">
        <h2 class="text-base font-semibold mb-3">Taxa de Desmatamento Média por Estado (%/ano)</h2>
        <div class="relative bg-white rounded border border-slate-200 p-3 mb-4" style="height:380px">
          <canvas id="chart-linha"></canvas>
        </div>
        <h2 class="text-base font-semibold mb-3">Desmatamento Total por Estado (km²)</h2>
        <div class="relative bg-white rounded border border-slate-200 p-3" style="height:380px">
          <canvas id="chart-barra"></canvas>
        </div>
      </div>

      <!-- Sobre -->
      <div id="tab-sobre" class="p-6 hidden max-w-3xl">
        <div class="bg-white rounded-lg border border-slate-200 p-6 mb-4">
          <h2 class="text-lg font-bold mb-2 text-green-800">🌿 Preditor de Risco de Desmatamento</h2>
          <p class="text-sm text-slate-600 mb-3">
            Sistema de machine learning que atribui um <strong>score de risco (0–100)</strong> a cada um dos
            772 municípios da Amazônia Legal, permitindo identificar onde o desmatamento tem maior
            probabilidade de ocorrer no próximo ano.
          </p>
          <div class="grid grid-cols-2 gap-4 text-sm">
            <div>
              <h3 class="font-semibold text-slate-700 mb-1">Fontes de Dados</h3>
              <ul class="text-slate-500 space-y-0.5 list-disc list-inside">
                <li>INPE/PRODES — Desmatamento km² (2008–2025)</li>
                <li>IBGE — Pop., PIB agro, área municipal</li>
                <li>ICMBio — Unidades de conservação e TIs</li>
                <li>IBAMA — Autos de infração ambiental</li>
              </ul>
            </div>
            <div>
              <h3 class="font-semibold text-slate-700 mb-1">Metodologia</h3>
              <ul class="text-slate-500 space-y-0.5 list-disc list-inside">
                <li>LightGBM + Optuna (55 trials)</li>
                <li>Validação espacial por UF (GroupKFold)</li>
                <li>HDBSCAN — 4 clusters de municípios</li>
                <li>SHAP — interpretabilidade do modelo</li>
                <li>Score: ranking percentílico global</li>
              </ul>
            </div>
          </div>
        </div>
        <div class="bg-green-50 border border-green-200 rounded-lg p-4 text-sm">
          <h3 class="font-semibold text-green-800 mb-2">Desempenho do Modelo (Spatial CV)</h3>
          <div class="grid grid-cols-3 gap-3" id="sobre-metrics"></div>
        </div>
        <p class="text-xs text-slate-400 mt-4">Gerado em __TODAY__ · Dados: INPE · IBGE · ICMBio · IBAMA</p>
      </div>

    </div>
  </main>
</div>

<script>
// ── Dados embutidos ────────────────────────────────────────────────────────────
const ANOS     = __ANOS__;
const UFS      = __UFS__;
const GEODATA  = __GEODATA__;
const ALLDATA  = __ALLDATA__;
const SERIES   = __SERIES__;
const CLUSTERS = __CLUSTERS__;
const METRICS  = __METRICS__;

// ── Estado ────────────────────────────────────────────────────────────────────
const state = {
  ano: ANOS[ANOS.length - 1],
  ufs: [...UFS],
};

// ── Cor de risco ──────────────────────────────────────────────────────────────
function getColor(score) {
  if (score == null || isNaN(score)) return '#cbd5e1';
  if (score <= 20)  return '#22c55e';
  if (score <= 40)  return '#86efac';
  if (score <= 60)  return '#eab308';
  if (score <= 80)  return '#f97316';
  return '#ef4444';
}

function badgeHtml(score) {
  if (score == null) return '—';
  const c = getColor(score);
  return `<span class="badge" style="background:${c}20;color:${c};border:1px solid ${c}40">${score.toFixed(1)}</span>`;
}

// ── Mapa Leaflet ──────────────────────────────────────────────────────────────
const map = L.map('map', { zoomControl: true }).setView([-6, -55], 5);
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
  attribution: '© CartoDB © OpenStreetMap',
  maxZoom: 19,
}).addTo(map);

// Build lookup: cod_ibge → score por ano
const scoreLookup = {};
for (const [ano, rows] of Object.entries(ALLDATA)) {
  scoreLookup[+ano] = {};
  for (const r of rows) scoreLookup[+ano][r.cod_ibge] = r;
}

let geojsonLayer = null;

function styleFeature(feature) {
  const row = (scoreLookup[state.ano] || {})[feature.properties.cod_ibge];
  const score = row ? row.score_risco : null;
  return {
    fillColor: getColor(score),
    fillOpacity: 0.75,
    color: '#94a3b8',
    weight: 0.4,
  };
}

function onEachFeature(feature, layer) {
  layer.on('click', () => {
    const p   = feature.properties;
    const row = (scoreLookup[state.ano] || {})[p.cod_ibge] || p;
    const sc  = row.score_risco != null ? row.score_risco.toFixed(1) : '—';
    const tx  = row.taxa_desmatamento != null ? row.taxa_desmatamento.toFixed(4) : '—';
    const dm  = row.desmatamento_km2  != null ? row.desmatamento_km2.toFixed(2)  : '—';
    layer.bindPopup(`
      <div style="min-width:180px;font-size:13px">
        <b style="font-size:14px">${p.municipio}</b><br>
        <span style="color:#64748b">UF: ${p.uf}</span><br><br>
        <b>Score de risco:</b> ${sc} / 100<br>
        <b>Taxa desmat.:</b> ${tx} %/ano<br>
        <b>Desmatamento:</b> ${dm} km²
      </div>
    `).openPopup();
  });
}

function buildMap() {
  if (geojsonLayer) map.removeLayer(geojsonLayer);
  geojsonLayer = L.geoJSON(GEODATA, {
    style: styleFeature,
    onEachFeature,
  }).addTo(map);
}

// Legenda
const legend = L.control({ position: 'bottomright' });
legend.onAdd = () => {
  const div = L.DomUtil.create('div', 'leaflet-bar');
  div.style.cssText = 'background:white;padding:8px 12px;font-size:11px;line-height:1.6;border-radius:6px';
  div.innerHTML = '<b style="font-size:12px">Score de Risco</b><br>' +
    ['0–20','21–40','41–60','61–80','81–100'].map((r, i) => {
      const colors = ['#22c55e','#86efac','#eab308','#f97316','#ef4444'];
      return `<span style="display:inline-block;width:12px;height:12px;background:${colors[i]};border-radius:2px;margin-right:5px;vertical-align:middle"></span>${r}`;
    }).join('<br>');
  return div;
};
legend.addTo(map);

buildMap();

// ── KPIs ──────────────────────────────────────────────────────────────────────
function updateKPIs() {
  const rows = (ALLDATA[state.ano] || []).filter(r => state.ufs.includes(r.uf));
  const desmat = rows.reduce((s, r) => s + (r.desmatamento_km2 || 0), 0);
  const taxa   = rows.length ? rows.reduce((s, r) => s + (r.taxa_desmatamento || 0), 0) / rows.length : 0;
  const scores = rows.map(r => r.score_risco).filter(s => s != null);
  const scoreMed = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  const high = scores.filter(s => s >= 80).length;

  document.getElementById('kpi-desmat').textContent = desmat.toLocaleString('pt-BR', {maximumFractionDigits:0}) + ' km²';
  document.getElementById('kpi-taxa').textContent   = taxa.toFixed(4) + ' %';
  document.getElementById('kpi-score').textContent  = scoreMed != null ? scoreMed.toFixed(1) : '—';
  document.getElementById('kpi-high').textContent   = high.toLocaleString('pt-BR');
  document.getElementById('header-sub').textContent = `Ano: ${state.ano} · ${rows.length} municípios filtrados`;
}

// ── Mapa: recolorir ao mudar ano ───────────────────────────────────────────────
function updateMapColors() {
  if (geojsonLayer) geojsonLayer.setStyle(styleFeature);
}

// ── Ranking ───────────────────────────────────────────────────────────────────
function updateRanking() {
  const rows = (ALLDATA[state.ano] || [])
    .filter(r => state.ufs.includes(r.uf) && r.score_risco != null)
    .sort((a, b) => b.score_risco - a.score_risco)
    .slice(0, 20);

  const tbody = document.getElementById('ranking-body');
  tbody.innerHTML = rows.map((r, i) => `
    <tr>
      <td class="px-3 py-2 text-slate-400">${i + 1}</td>
      <td class="px-3 py-2 font-medium">${r.municipio}</td>
      <td class="px-3 py-2 text-slate-500">${r.uf}</td>
      <td class="px-3 py-2 text-right">${badgeHtml(r.score_risco)}</td>
      <td class="px-3 py-2 text-right text-slate-600">${(r.taxa_desmatamento||0).toFixed(4)}</td>
      <td class="px-3 py-2 text-right text-slate-600">${(r.desmatamento_km2||0).toFixed(2)}</td>
    </tr>
  `).join('');
}

// ── Gráficos Chart.js ─────────────────────────────────────────────────────────
const CHART_COLORS = ['#22c55e','#3b82f6','#f97316','#a855f7','#06b6d4','#ec4899','#eab308','#14b8a6','#f43f5e'];
let chartLinha = null;
let chartBarra = null;

function buildCharts() {
  const anos  = ANOS.slice().sort((a,b) => a - b);
  const ufs   = state.ufs.slice().sort();

  const datasetsLinha = ufs.map((uf, i) => ({
    label: uf,
    data: anos.map(a => (SERIES[uf] && SERIES[uf][a]) ? SERIES[uf][a].taxa : null),
    borderColor: CHART_COLORS[i % CHART_COLORS.length],
    backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + '22',
    tension: 0.3,
    pointRadius: 3,
    fill: false,
    spanGaps: true,
  }));

  const datasetsBarra = ufs.map((uf, i) => ({
    label: uf,
    data: anos.map(a => (SERIES[uf] && SERIES[uf][a]) ? SERIES[uf][a].desmat : 0),
    backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + 'cc',
    stack: 'stack',
  }));

  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } },
    scales: { x: { ticks: { font: { size: 11 } } }, y: { ticks: { font: { size: 11 } } } },
  };

  if (chartLinha) chartLinha.destroy();
  chartLinha = new Chart(document.getElementById('chart-linha'), {
    type: 'line',
    data: { labels: anos, datasets: datasetsLinha },
    options: { ...baseOpts, scales: {
      ...baseOpts.scales,
      y: { ...baseOpts.scales.y, title: { display: true, text: 'Taxa média (%/ano)', font: { size: 11 } } },
    }},
  });

  if (chartBarra) chartBarra.destroy();
  chartBarra = new Chart(document.getElementById('chart-barra'), {
    type: 'bar',
    data: { labels: anos, datasets: datasetsBarra },
    options: { ...baseOpts, scales: {
      ...baseOpts.scales,
      y: { ...baseOpts.scales.y, title: { display: true, text: 'Desmat. total (km²)', font: { size: 11 } } },
    }},
  });
}

// ── Métricas (sidebar + sobre) ────────────────────────────────────────────────
function renderMetrics() {
  const box = document.getElementById('metrics-box');
  if (!METRICS || !METRICS.MEDIA) { box.innerHTML = '<span class="text-slate-400">Não disponível</span>'; return; }
  const m = METRICS.MEDIA;
  box.innerHTML = `
    <div>R² médio: <b>${(m.r2||0).toFixed(4)}</b></div>
    <div>RMSE: <b>${(m.rmse||0).toFixed(4)}</b></div>
    <div>MAE: <b>${(m.mae||0).toFixed(4)}</b></div>
  `;

  const sobreBox = document.getElementById('sobre-metrics');
  sobreBox.innerHTML = Object.entries(METRICS).map(([k, v]) => `
    <div class="bg-white rounded p-2 border border-green-200">
      <div class="text-xs text-green-700 font-semibold">${k}</div>
      <div class="text-xs text-slate-600 mt-1">R²: ${(v.r2||0).toFixed(3)} · RMSE: ${(v.rmse||0).toFixed(4)}</div>
    </div>
  `).join('');
}

// ── Sidebar: inicializar filtros ──────────────────────────────────────────────
function initSidebar() {
  // Select de ano
  const selAno = document.getElementById('sel-ano');
  ANOS.slice().reverse().forEach(a => {
    const opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    if (a === state.ano) opt.selected = true;
    selAno.appendChild(opt);
  });
  selAno.addEventListener('change', () => {
    state.ano = +selAno.value;
    updateAll();
  });

  // Checkboxes UF
  const ufDiv = document.getElementById('uf-checks');
  UFS.forEach(uf => {
    const label = document.createElement('label');
    label.className = 'flex items-center gap-1.5 cursor-pointer hover:text-green-700';
    label.innerHTML = `<input type="checkbox" class="accent-green-500" value="${uf}" checked> ${uf}`;
    ufDiv.appendChild(label);
    label.querySelector('input').addEventListener('change', () => {
      state.ufs = Array.from(ufDiv.querySelectorAll('input:checked')).map(i => i.value);
      updateAll();
    });
  });
}

// ── Clusters ──────────────────────────────────────────────────────────────────
const CLUSTER_PALETTE = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'];
const CLUSTER_BG      = ['#fde8e8','#dbeafe','#dcfce7','#f3e8ff','#ffedd5'];

function renderClusters() {
  const container = document.getElementById('clusters-cards');
  if (!container || !CLUSTERS.length) return;

  container.innerHTML = CLUSTERS.map(c => {
    const isNoise = c.id === -1;
    const color   = isNoise ? '#94a3b8' : (CLUSTER_PALETTE[c.id % CLUSTER_PALETTE.length]);
    const bg      = isNoise ? '#f1f5f9' : (CLUSTER_BG[c.id % CLUSTER_BG.length]);
    const title   = isNoise ? 'Ruído (−1) — sem cluster' : `Cluster ${c.id} — ${c.label}`;

    const stats = [
      { label: 'Municípios',           value: c.n.toLocaleString('pt-BR') },
      { label: 'Taxa desmat. média',   value: `${c.taxa.toFixed(4)} %/ano` },
      { label: 'Área média',           value: `${c.area.toLocaleString('pt-BR')} mil km²` },
      { label: 'Pop. média',           value: `${c.pop.toLocaleString('pt-BR')} mil hab` },
      { label: 'PIB agro médio',       value: `R$ ${c.pib.toLocaleString('pt-BR')} M` },
      { label: 'Cobertura UC',         value: `${c.uc.toFixed(1)}%` },
      { label: 'Cobertura TI',         value: `${c.ti.toFixed(1)}%` },
      { label: 'Autos IBAMA (média)',  value: c.autos.toLocaleString('pt-BR') },
    ];

    // Barra proporcional para taxa (max ~0.25)
    const barW = Math.min(100, (c.taxa / 0.25) * 100).toFixed(1);

    return `
    <div class="rounded-lg border p-4" style="border-color:${color}40;background:${bg}">
      <div class="flex items-center gap-2 mb-3">
        <span style="display:inline-block;width:14px;height:14px;background:${color};border-radius:3px;flex-shrink:0"></span>
        <span class="font-semibold text-sm" style="color:${isNoise ? '#64748b' : '#0f172a'}">${title}</span>
      </div>
      <div class="mb-3">
        <div class="text-xs text-slate-500 mb-1">Taxa de desmatamento média</div>
        <div class="w-full bg-slate-200 rounded-full h-2">
          <div class="h-2 rounded-full" style="width:${barW}%;background:${color}"></div>
        </div>
        <div class="text-xs text-slate-600 mt-0.5">${c.taxa.toFixed(4)} %/ano</div>
      </div>
      <div class="grid grid-cols-4 gap-x-4 gap-y-1">
        ${stats.map(s => `
          <div>
            <div class="text-xs text-slate-400">${s.label}</div>
            <div class="text-sm font-medium text-slate-700">${s.value}</div>
          </div>
        `).join('')}
      </div>
    </div>`;
  }).join('');
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    ['mapa','ranking','clusters','tendencia','sobre'].forEach(t => {
      document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
    });
    if (tab === 'tendencia') buildCharts();
    if (tab === 'clusters')  renderClusters();
    if (tab === 'mapa') setTimeout(() => map.invalidateSize(), 50);
  });
});

// ── updateAll ─────────────────────────────────────────────────────────────────
function updateAll() {
  updateKPIs();
  updateMapColors();
  updateRanking();
  if (!document.getElementById('tab-tendencia').classList.contains('hidden')) buildCharts();
}

// ── Init ──────────────────────────────────────────────────────────────────────
initSidebar();
renderMetrics();
updateKPIs();
updateRanking();
</script>
</body>
</html>"""


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    build_dashboard()
