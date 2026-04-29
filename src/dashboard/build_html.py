"""
TASK-16: Gerador do dashboard HTML single-file.

Lê skills/frontend-design/SKILL.md antes de qualquer geração (requisito).
Produz reports/dashboard.html com Tailwind CSS + Leaflet.js + Chart.js.
"""

from __future__ import annotations

import base64
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
    if "cluster_id" in last.columns:
        last["cluster_id"] = last["cluster_id"].fillna(-1).astype(int)
    props = ["cod_ibge", "municipio", "uf", "score_risco",
             "taxa_desmatamento", "taxa_desmatamento_prevista",
             "desmatamento_km2", "ano_tipo", "cluster_id"]
    props = [p for p in props if p in last.columns]
    return last[props + ["geometry"]].__geo_interface__


def _build_all_data(gdf: gpd.GeoDataFrame) -> dict:
    want = ["cod_ibge", "municipio", "uf", "ano", "ano_tipo",
            "score_risco", "taxa_desmatamento", "taxa_desmatamento_prevista",
            "desmatamento_km2", "cluster_id"]
    cols = [c for c in want if c in gdf.columns]
    base = gdf[gdf["score_risco"].notna()]
    df = base[cols].copy()
    df["ano"] = df["ano"].astype(int)
    df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
    df["score_risco"] = df["score_risco"].round(1)
    df["taxa_desmatamento"] = df["taxa_desmatamento"].round(6)
    if "taxa_desmatamento_prevista" in df.columns:
        df["taxa_desmatamento_prevista"] = df["taxa_desmatamento_prevista"].round(6)
    df["desmatamento_km2"] = df["desmatamento_km2"].round(2)

    # Desmatamento estimado para anos de previsão: taxa_prevista / 100 * area_km2
    if "taxa_desmatamento_prevista" in df.columns and "area_km2" in base.columns:
        area = base["area_km2"].values
        mask = df["taxa_desmatamento"].isna() & df["taxa_desmatamento_prevista"].notna()
        df["desmatamento_km2_previsto"] = np.nan
        df.loc[mask, "desmatamento_km2_previsto"] = (
            df.loc[mask, "taxa_desmatamento_prevista"] / 100 * area[mask]
        ).round(1)

    # JSON não serializa NaN — substituir por None (null em JS)
    df = df.where(df.notna(), other=None)
    result: dict[str, list] = {}
    for ano, grp in df.groupby("ano"):
        result[str(ano)] = grp.drop(columns="ano").to_dict("records")
    return result


def _build_series(gdf: gpd.GeoDataFrame) -> dict:
    # Para 2026 (previsão), taxa_desmatamento é NaN — usar taxa_desmatamento_prevista
    df = gdf.copy()
    if "taxa_desmatamento_prevista" in df.columns:
        mask_prev = df["taxa_desmatamento"].isna() & df["taxa_desmatamento_prevista"].notna()
        df.loc[mask_prev, "taxa_desmatamento"] = df.loc[mask_prev, "taxa_desmatamento_prevista"]

    agg = (
        df.groupby(["uf", "ano"])
        .agg(taxa_media=("taxa_desmatamento", "mean"),
             desmat_total=("desmatamento_km2", "sum"))
        .reset_index()
    )
    agg["taxa_media"]   = agg["taxa_media"].round(6)
    agg["desmat_total"] = agg["desmat_total"].round(2)
    result: dict = {}
    for uf, grp in agg.groupby("uf"):
        result[uf] = {
            int(r["ano"]): {"taxa": r["taxa_media"], "desmat": r["desmat_total"]}
            for _, r in grp.iterrows()
        }
    return result


def _cluster_label_html(row) -> str:
    """Rótulo descritivo para clusters — espelha a lógica de hdbscan_model.py."""
    if int(row["cluster_id"]) == -1:
        return "Perfil transicional (não classificado)"
    taxa  = row["taxa"]
    uc    = row["uc"]
    ti    = row["ti"]
    autos = row["autos"]
    area  = row["area"]
    pib   = row["pib"]
    trend = row.get("trend", 1.0)
    prot  = uc + ti

    if taxa > 0.35 and prot < 10:
        return "Arco do desmatamento — alta pressão sem áreas protegidas"
    if area > 10_000 and prot > 35 and trend > 1.5:
        return "Grandes municípios — proteção formal com pressão crescente"
    if ti > 15 and taxa > 0.12:
        return "Fronteira agrícola com presença de Terras Indígenas"
    if uc > 25 and ti < 10 and taxa < 0.15:
        return "Proteção por Unidades de Conservação reduz pressão"
    if area > 10_000 and prot < 25:
        return "Grandes municípios com expansão agrícola moderada"
    if area < 2_500 and pib < 80 and taxa < 0.12:
        return "Municípios periféricos com baixa pressão de desmatamento"
    return "Perfil intermediário de expansão agrícola"


_RISK_COLORS = ["#22c55e", "#86efac", "#eab308", "#f97316", "#ef4444"]
_RISK_BG     = ["#dcfce7", "#d1fae5", "#fef9c3", "#ffedd5", "#fee2e2"]


def _build_clusters(gdf: gpd.GeoDataFrame) -> list:
    if "cluster_id" not in gdf.columns:
        return []

    # Usar apenas dados históricos para calcular perfil estrutural
    hist = gdf[gdf["ano_tipo"] == "histórico"] if "ano_tipo" in gdf.columns else gdf

    # Perfil estrutural: média histórica por município, depois por cluster
    agg = (
        hist.groupby(["cod_ibge", "cluster_id"])
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

    # Tendência recente (2020-2025 vs 2008-2015)
    recente  = hist[hist["ano"] >= 2020].groupby("cod_ibge")["taxa_desmatamento"].mean()
    historico = hist[hist["ano"] <= 2015].groupby("cod_ibge")["taxa_desmatamento"].mean()
    trend_map = (recente / (historico + 1e-6)).clip(0, 10)
    agg["trend"] = agg["cod_ibge"].map(trend_map).fillna(1.0)

    # Score médio do modelo por município (usa todo o gdf, não só histórico)
    score_map = (
        gdf[gdf["score_risco"].notna()]
        .groupby("cod_ibge")["score_risco"]
        .mean()
    )
    agg["score"] = agg["cod_ibge"].map(score_map).fillna(0.0)

    profiles = (
        agg.groupby("cluster_id")
        .agg(
            n=("cod_ibge", "count"),
            taxa=("taxa", "mean"),
            score=("score", "mean"),
            area=("area", "mean"),
            pop=("pop", "mean"),
            pib=("pib", "mean"),
            uc=("uc", "mean"),
            ti=("ti", "mean"),
            autos=("autos", "mean"),
            trend=("trend", "mean"),
        )
        .reset_index()
        .sort_values("cluster_id")
    )

    result = []
    for _, row in profiles.iterrows():
        result.append({
            "id":    int(row["cluster_id"]),
            "label": _cluster_label_html(row),
            "n":     int(row["n"]),
            "taxa":  round(float(row["taxa"]), 4),
            "score": round(float(row["score"]), 1),
            "area":  round(float(row["area"]) / 1000, 1),
            "pop":   round(float(row["pop"]) / 1000, 1),
            "pib":   round(float(row["pib"]) / 1000, 1),
            "uc":    round(float(row["uc"]), 1),
            "ti":    round(float(row["ti"]), 1),
            "autos": round(float(row["autos"]), 0),
        })

    # Cores baseadas em rank de score_risco: menor score → verde, maior score → vermelho
    valid = sorted([r for r in result if r["id"] >= 0], key=lambda x: x["score"])
    n = len(valid)
    for rank, cluster in enumerate(valid):
        idx = round(rank * (len(_RISK_COLORS) - 1) / max(n - 1, 1))
        cluster["color"] = _RISK_COLORS[idx]
        cluster["bg"]    = _RISK_BG[idx]
    for r in result:
        if r["id"] == -1:
            r["color"] = "#94a3b8"
            r["bg"]    = "#f1f5f9"
    return result


def _load_metrics() -> dict:
    p = Path(_METRICS)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _json(obj) -> str:
    # json.dumps serializa float NaN como 'NaN' (inválido em JSON) — substituir por null
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.replace(":NaN", ":null").replace(",NaN", ",null").replace("[NaN", "[null")


def _img_to_b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode()


_SHAP_FILES = {
    "beeswarm": "data/outputs/shap_beeswarm.png",
    "cluster":  "data/outputs/shap_por_cluster.png",
    "wf_nova_santa_helena":        "data/outputs/shap_waterfall_nova_santa_helena.png",
    "wf_marcelandia":              "data/outputs/shap_waterfall_marcelândia.png",
    "wf_mojui":                    "data/outputs/shap_waterfall_mojuí_dos_campos.png",
    "wf_nova_esperanca":           "data/outputs/shap_waterfall_nova_esperança_do_piriá.png",
    "wf_uniao_sul":                "data/outputs/shap_waterfall_união_do_sul.png",
}

_WF_LABELS = {
    "wf_nova_santa_helena":  "Nova Santa Helena (MT)",
    "wf_marcelandia":        "Marcelândia (MT)",
    "wf_mojui":              "Mojuí dos Campos (PA)",
    "wf_nova_esperanca":     "Nova Esperança do Piriá (PA)",
    "wf_uniao_sul":          "União do Sul (MT)",
}


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

    logger.info("Carregando imagens SHAP...")
    shap_imgs = {k: _img_to_b64(v) for k, v in _SHAP_FILES.items()}
    loaded = sum(1 for v in shap_imgs.values() if v)
    logger.info("  %d/%d imagens SHAP encontradas.", loaded, len(_SHAP_FILES))

    anos_js     = _json(anos)
    ufs_js      = _json(ufs)
    geodata_js  = _json(geodata)
    alldata_js  = _json(all_data)
    series_js   = _json(series)
    clusters_js = _json(clusters)
    metrics_js  = _json(metrics)
    shap_js     = _json(shap_imgs)
    today       = date.today().strftime("%d/%m/%Y")

    html = _HTML_TEMPLATE.replace("__ANOS__",      anos_js)    \
                         .replace("__UFS__",       ufs_js)     \
                         .replace("__GEODATA__",   geodata_js) \
                         .replace("__ALLDATA__",   alldata_js) \
                         .replace("__SERIES__",    series_js)  \
                         .replace("__CLUSTERS__",  clusters_js)\
                         .replace("__METRICS__",   metrics_js) \
                         .replace("__SHAP_IMGS__", shap_js)    \
                         .replace("__TODAY__",     today)      \
                         .replace("__LAST_ANO__",  str(last_ano))

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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=Quantico:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
/* ── Variáveis de tema ─────────────────────────────────────────────────── */
:root {
  --bg-app:       #f5f1eb;
  --bg-surface:   #ffffff;
  --bg-card:      #fdf9f4;
  --bg-header:    #e8e2d4;
  --bg-sidebar:   #fdfaf7;
  --bg-tabs:      #f7f3ed;
  --text-heading: #4a7c5e;
  --text-primary: #2c3830;
  --text-secondary:#6b7870;
  --text-muted:   #9aa89f;
  --text-hsub:    #7a9485;
  --accent:       #4a7c5e;
  --accent-soft:  #d4e9dc;
  --border:       #ddd8ce;
  --shadow:       rgba(44,56,48,.08);
  --tab-active:   #4a7c5e;
  --tab-inactive: #8a9e94;
  --row-even:     #f5f0ea;
  --th-bg:        #e0dbd2;
  --th-color:     #3a5044;
  --bg-chart:     #ede8dc;
  --chart-grid:   #d5cfc5;
}
html.dark {
  --bg-app:       #1b2130;
  --bg-surface:   #242c3d;
  --bg-card:      #2a3347;
  --bg-header:    #141929;
  --bg-sidebar:   #1f2638;
  --bg-tabs:      #1f2638;
  --text-heading: #85c4a0;
  --text-primary: #c8d5ce;
  --text-secondary:#7d9488;
  --text-muted:   #4e6459;
  --text-hsub:    #5a8070;
  --accent:       #85c4a0;
  --accent-soft:  #243d30;
  --border:       #2e3d4e;
  --shadow:       rgba(0,0,0,.3);
  --tab-active:   #85c4a0;
  --tab-inactive: #4e6659;
  --row-even:     #222d3e;
  --th-bg:        #161e2b;
  --th-color:     #85c4a0;
  --bg-chart:     #1e2636;
  --chart-grid:   #2a3347;
}

/* ── Base ──────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: 'Quantico', 'Inter', system-ui, sans-serif;
  background-color: var(--bg-app);
  color: var(--text-primary);
  margin: 0;
}
h1, h2, h3 {
  font-family: 'Silkscreen', monospace;
  color: var(--text-heading);
  font-weight: 700;
  margin: 0 0 4px;
}
h2 { font-size: 15px; }
h3 { font-size: 13px; }

/* ── Header ────────────────────────────────────────────────────────────── */
#app-header {
  background-color: var(--bg-header);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 1px 4px var(--shadow);
}
#app-header h1 { font-size: 15px; letter-spacing: .4px; }
#header-sub { color: var(--text-hsub); font-size: 11px; margin-top: 1px; }
#header-model { color: var(--text-hsub); font-size: 11px; }

/* ── Dark mode toggle ──────────────────────────────────────────────────── */
#dark-toggle {
  background: none;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 11px;
  cursor: pointer;
  color: var(--text-secondary);
  font-family: 'Quantico', sans-serif;
  transition: all .2s;
  white-space: nowrap;
}
#dark-toggle:hover { background-color: var(--accent-soft); }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
aside {
  background-color: var(--bg-sidebar) !important;
  border-color: var(--border) !important;
}
aside label, aside p, aside div { color: var(--text-secondary) !important; }
aside select, aside input {
  background-color: var(--bg-card) !important;
  color: var(--text-primary) !important;
  border-color: var(--border) !important;
}
#metrics-box div { color: var(--text-secondary) !important; }
#metrics-box b   { color: var(--text-primary) !important; }

/* ── KPI bar ───────────────────────────────────────────────────────────── */
#kpi-bar {
  background-color: var(--bg-surface);
  border-bottom: 1px solid var(--border);
}
.kpi-card {
  background-color: var(--bg-card) !important;
  border-color: var(--border) !important;
  border-radius: 8px;
  border: 1px solid var(--border);
  padding: 12px;
}
.kpi-label { font-size: 11px; color: var(--text-secondary); font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }
.kpi-value { font-size: 20px; font-weight: 700; color: var(--text-primary); margin: 4px 0 2px; font-family: 'Quantico', sans-serif; }
.kpi-sub   { font-size: 11px; color: var(--text-muted); }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
#tabs-bar {
  background-color: var(--bg-tabs);
  border-bottom: 1px solid var(--border);
}
.tab-btn {
  font-family: 'Quantico', sans-serif;
  font-size: 13px;
  color: var(--tab-inactive);
  border: none;
  border-bottom: 2px solid transparent;
  background: none;
  padding: 9px 16px;
  white-space: nowrap;
  cursor: pointer;
  transition: color .15s, border-color .15s;
  letter-spacing: .2px;
}
.tab-btn:hover  { color: var(--tab-active); }
.tab-btn.active { border-bottom: 2px solid var(--tab-active); color: var(--tab-active); font-weight: 700; }

/* ── Main content ──────────────────────────────────────────────────────── */
main { background-color: var(--bg-app); }
.content-pad { background-color: var(--bg-app); }

/* ── Cards/surfaces ────────────────────────────────────────────────────── */
.bg-white         { background-color: var(--bg-surface) !important; }
.bg-slate-50      { background-color: var(--bg-card) !important; }
.border-slate-200 { border-color: var(--border) !important; }
.text-slate-900, .text-slate-800 { color: var(--text-primary) !important; }
.text-slate-700, .text-slate-600 { color: var(--text-secondary) !important; }
.text-slate-500, .text-slate-400 { color: var(--text-muted) !important; }
.text-green-800   { color: var(--text-heading) !important; }
.text-green-700   { color: var(--accent) !important; }
.bg-green-50      { background-color: var(--accent-soft) !important; }
.border-green-200 { border-color: var(--accent) !important; }

/* ── Table ─────────────────────────────────────────────────────────────── */
table { border-collapse: collapse; width: 100%; }
th {
  position: sticky; top: 0; z-index: 1;
  background-color: var(--th-bg) !important;
  color: var(--th-color) !important;
  font-family: 'Quantico', sans-serif;
  font-size: 11px;
}
td { color: var(--text-primary) !important; border-top: 1px solid var(--border); }
tr:nth-child(even) td { background-color: var(--row-even) !important; }

/* ── Chart containers ──────────────────────────────────────────────────── */
.chart-wrap { background-color: var(--bg-chart) !important; }

/* ── Map ───────────────────────────────────────────────────────────────── */
#map { height: 480px; border-radius: 8px; }

/* ── Badge ─────────────────────────────────────────────────────────────── */
.badge { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:12px; font-weight:600; }

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: var(--bg-app); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }
</style>
</head>
<body>

<!-- HEADER -->
<header id="app-header" class="flex items-center px-6 h-14 shrink-0">
  <div>
    <h1>Risco de Desmatamento · Amazônia Legal</h1>
    <p id="header-sub">Ano: __LAST_ANO__ · 808 municípios · 9 estados</p>
  </div>
  <div class="ml-auto flex items-center gap-3">
    <span id="header-model">Modelo: LightGBM + Optuna · R²=0.76</span>
    <button id="dark-toggle" onclick="toggleDark()">◑ Tema</button>
  </div>
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
  </aside>

  <!-- MAIN -->
  <main class="flex-1 flex flex-col overflow-hidden">

    <!-- KPI CARDS -->
    <div id="kpi-bar" class="grid grid-cols-4 gap-3 p-4 shrink-0">
      <div class="kpi-card">
        <div class="kpi-label" id="kpi-desmat-label">Desmatamento Total</div>
        <div class="kpi-value" id="kpi-desmat">—</div>
        <div class="kpi-sub">km² no ano</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Taxa Média</div>
        <div class="kpi-value" id="kpi-taxa">—</div>
        <div class="kpi-sub">%/ano (municípios filtrados)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Score Médio de Risco</div>
        <div class="kpi-value" id="kpi-score" style="color:#f97316">—</div>
        <div class="kpi-sub">0–100 (percentílico)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Municípios Score ≥ 80</div>
        <div class="kpi-value" id="kpi-high" style="color:#ef4444">—</div>
        <div class="kpi-sub">alto risco crítico</div>
      </div>
    </div>

    <!-- TABS -->
    <div id="tabs-bar" class="flex px-4 shrink-0 overflow-x-auto">
      <button class="tab-btn active" data-tab="mapa">⊕ Mapa</button>
      <button class="tab-btn" data-tab="ranking">≡ Ranking</button>
      <button class="tab-btn" data-tab="clusters">◉ Clusters</button>
      <button class="tab-btn" data-tab="tendencia">↗ Tendência</button>
      <button class="tab-btn" data-tab="shap">∑ SHAP</button>
      <button class="tab-btn" data-tab="dicionario">∷ Dicionário</button>
      <button class="tab-btn" data-tab="sobre">◦ Sobre</button>
    </div>

    <!-- TAB CONTENT -->
    <div class="flex-1 overflow-auto">

      <!-- Mapa -->
      <div id="tab-mapa" class="p-4 h-full">
        <div id="map" class="rounded-lg border border-slate-200 shadow-sm"></div>
        <p class="text-xs text-slate-400 mt-2">Clique em um município para ver detalhes. Cores: verde = baixo risco · vermelho = alto risco.</p>
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
                <th class="px-3 py-2 text-right" id="ranking-th-desmat">Desmat. (km²)</th>
              </tr>
            </thead>
            <tbody id="ranking-body" class="text-sm"></tbody>
          </table>
        </div>
      </div>

      <!-- Clusters -->
      <div id="tab-clusters" class="p-4 hidden">
        <div class="flex items-center justify-between mb-3">
          <h2 class="text-base font-semibold">Distribuição Espacial dos Clusters — HDBSCAN</h2>
          <span class="text-xs" id="cluster-stats-text" style="color:var(--text-muted)">—</span>
        </div>
        <div id="cluster-map" style="height:420px;border-radius:8px;border:1px solid var(--border);margin-bottom:20px"></div>
        <h2 class="text-base font-semibold mb-3">Perfis por Cluster</h2>
        <div id="clusters-cards" class="grid grid-cols-1 gap-4 max-w-5xl"></div>
      </div>

      <!-- Tendência -->
      <div id="tab-tendencia" class="p-4 hidden">
        <h2 class="text-base font-semibold mb-3">Taxa de Desmatamento Média por Estado (%/ano)</h2>
        <div class="relative chart-wrap rounded border border-slate-200 p-3 mb-4" style="height:380px">
          <canvas id="chart-linha"></canvas>
        </div>
        <h2 class="text-base font-semibold mb-3">Desmatamento Total por Estado (km²)</h2>
        <div class="relative chart-wrap rounded border border-slate-200 p-3" style="height:380px">
          <canvas id="chart-barra"></canvas>
        </div>
      </div>

      <!-- SHAP -->
      <div id="tab-shap" class="p-4 hidden">
        <div class="max-w-4xl">
          <h2 class="text-base font-semibold mb-1">Importância Global das Features (SHAP Beeswarm)</h2>
          <p class="text-xs text-slate-500 mb-3">Cada ponto representa um município. Cor vermelha = valor alto da feature; azul = valor baixo. Posição no eixo X = impacto no score previsto.</p>
          <div id="shap-beeswarm-wrap" class="chart-wrap rounded border border-slate-200 p-3 mb-6 text-center text-slate-400 text-sm"></div>

          <h2 class="text-base font-semibold mb-1">Importância SHAP por Cluster HDBSCAN</h2>
          <p class="text-xs text-slate-500 mb-3">Comparação da importância relativa de cada feature nos diferentes perfis de municípios.</p>
          <div id="shap-cluster-wrap" class="chart-wrap rounded border border-slate-200 p-3 mb-6 text-center text-slate-400 text-sm"></div>

          <h2 class="text-base font-semibold mb-1">Waterfall — Top-5 Municípios de Maior Risco (2026)</h2>
          <p class="text-xs text-slate-500 mb-3">Cada gráfico mostra a contribuição individual de cada feature para o score do município (vermelho = aumenta o risco, azul = reduz).</p>
          <div id="shap-waterfall-grid" class="grid grid-cols-2 gap-4"></div>
        </div>
      </div>

      <!-- Dicionário de Features -->
      <div id="tab-dicionario" class="p-4 hidden">
        <div class="max-w-4xl">
          <h2 class="text-base font-semibold mb-1">Indicadores do Dashboard</h2>
          <p class="text-xs mb-3" style="color:var(--text-muted)">O que cada número no topo da página significa.</p>
          <div class="overflow-auto rounded border border-slate-200 mb-6">
            <table class="text-sm">
              <thead>
                <tr class="bg-slate-800 text-white text-xs">
                  <th class="px-3 py-2 text-left font-semibold w-48">Indicador</th>
                  <th class="px-3 py-2 text-left font-semibold">Definição</th>
                </tr>
              </thead>
              <tbody>
                <tr><td class="px-3 py-2 font-semibold" style="color:var(--text-primary)">Desmatamento total (km²)</td><td class="px-3 py-2" style="color:var(--text-secondary)">Soma da área desmatada em todos os municípios filtrados no ano selecionado. Para 2026 é uma estimativa: taxa prevista pelo modelo × área do município (dados INPE/PRODES ainda não disponíveis).</td></tr>
                <tr><td class="px-3 py-2 font-semibold" style="color:var(--text-primary)">Taxa média (%/ano)</td><td class="px-3 py-2" style="color:var(--text-secondary)">Média da taxa de desmatamento entre os municípios filtrados. Taxa = área desmatada ÷ área total × 100. Mede a <em>intensidade relativa</em> do desmatamento, independente do tamanho do município — um município pequeno com taxa alta é tão preocupante quanto um grande com muito desmatamento absoluto. Para 2026 usa a previsão do modelo.</td></tr>
                <tr><td class="px-3 py-2 font-semibold" style="color:var(--text-primary)">Score de risco médio (0–100)</td><td class="px-3 py-2" style="color:var(--text-secondary)">Média do ranking percentílico entre os municípios filtrados. O score é recalculado a cada ano: score 80 = o município desmatou mais do que 80% dos 808 municípios da Amazônia Legal naquele mesmo ano. Permite comparar anos distintos mesmo que o nível absoluto de desmatamento varie.</td></tr>
              </tbody>
            </table>
          </div>
          <h2 class="text-base font-semibold mb-1">Dicionário de Features do Modelo</h2>
          <p class="text-xs text-slate-500 mb-4">Descrição de todas as variáveis utilizadas no modelo e no dashboard.</p>
          <div class="overflow-auto rounded border border-slate-200">
            <table class="text-sm">
              <thead>
                <tr class="bg-slate-800 text-white text-xs">
                  <th class="px-3 py-2 text-left font-semibold">Feature</th>
                  <th class="px-3 py-2 text-left font-semibold">Unidade</th>
                  <th class="px-3 py-2 text-left font-semibold">Descrição</th>
                  <th class="px-3 py-2 text-left font-semibold">Fonte</th>
                </tr>
              </thead>
              <tbody class="text-slate-700">
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmatamento</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Taxa de perda de cobertura vegetal nativa em relação à área total do município</td><td class="px-3 py-2 text-slate-400">INPE/PRODES</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmatamento_prevista</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Previsão do modelo LightGBM — disponível apenas para 2026 (dados INPE ainda não medidos)</td><td class="px-3 py-2 text-slate-400">Modelo</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">desmatamento_km2</td><td class="px-3 py-2 text-slate-500">km²</td><td class="px-3 py-2">Área absoluta desmatada no ano</td><td class="px-3 py-2 text-slate-400">INPE/PRODES</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">score_risco</td><td class="px-3 py-2 text-slate-500">0–100</td><td class="px-3 py-2">Ranking percentílico da taxa de desmatamento dentro do mesmo ano — score 80 = município com maior desmatamento que 80% dos municípios naquele ano</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">area_km2</td><td class="px-3 py-2 text-slate-500">km²</td><td class="px-3 py-2">Área total do município</td><td class="px-3 py-2 text-slate-400">IBGE 2022</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">populacao</td><td class="px-3 py-2 text-slate-500">habitantes</td><td class="px-3 py-2">População estimada do município no ano</td><td class="px-3 py-2 text-slate-400">IBGE</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">pib_agropecuario</td><td class="px-3 py-2 text-slate-500">R$ milhões</td><td class="px-3 py-2">Produto Interno Bruto do setor agropecuário municipal — proxy de pressão econômica sobre a terra</td><td class="px-3 py-2 text-slate-400">IBGE</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">area_uc_pct</td><td class="px-3 py-2 text-slate-500">%</td><td class="px-3 py-2">Percentual do território municipal coberto por Unidades de Conservação (parques, reservas, APAs) — barreira formal ao desmatamento</td><td class="px-3 py-2 text-slate-400">ICMBio</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">area_ti_pct</td><td class="px-3 py-2 text-slate-500">%</td><td class="px-3 py-2">Percentual do território coberto por Terras Indígenas demarcadas — barreira legal reconhecida ao desmatamento</td><td class="px-3 py-2 text-slate-400">FUNAI / ICMBio</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">autos_ibama</td><td class="px-3 py-2 text-slate-500">contagem</td><td class="px-3 py-2">Número de autos de infração ambiental emitidos pelo IBAMA no município — indica tanto pressão quanto presença de fiscalização</td><td class="px-3 py-2 text-slate-400">IBAMA</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">local_moran_i</td><td class="px-3 py-2 text-slate-500">índice</td><td class="px-3 py-2">Índice de autocorrelação espacial local (LISA) — valor positivo indica que o município tem desmatamento semelhante ao de seus vizinhos; captura efeitos de contágio territorial</td><td class="px-3 py-2 text-slate-400">Calculado (esda)</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">cluster_id</td><td class="px-3 py-2 text-slate-500">inteiro (0–4)</td><td class="px-3 py-2">Grupo HDBSCAN ao qual o município pertence — agrupa municípios com perfil estrutural similar de pressão de desmatamento</td><td class="px-3 py-2 text-slate-400">HDBSCAN</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmat_lag1</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Taxa de desmatamento do ano anterior (t−1) — feature temporal de primeira ordem</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmat_lag2</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Taxa de desmatamento de dois anos atrás (t−2)</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmat_lag3</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Taxa de desmatamento de três anos atrás (t−3)</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmat_roll3</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Média móvel da taxa de desmatamento nos últimos 3 anos — suaviza ruído anual</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">taxa_desmat_roll5</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Média móvel da taxa de desmatamento nos últimos 5 anos — tendência de médio prazo</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
                <tr><td class="px-3 py-2 font-mono text-xs font-semibold text-slate-800">std_desmat_roll3</td><td class="px-3 py-2 text-slate-500">% / ano</td><td class="px-3 py-2">Desvio padrão da taxa de desmatamento nos últimos 3 anos — captura volatilidade ou instabilidade recente</td><td class="px-3 py-2 text-slate-400">Calculado</td></tr>
              </tbody>
            </table>
          </div>
          <p class="text-xs text-slate-400 mt-3">* Features de lag e rolling window são construídas com dados reais INPE/PRODES para anos históricos; para 2026, usam os dados de 2025 como âncora mais recente.</p>
        </div>
      </div>

      <!-- Sobre -->
      <div id="tab-sobre" class="p-6 hidden max-w-3xl">
        <div class="bg-white rounded-lg border border-slate-200 p-6 mb-4">
          <h2 class="text-lg font-bold mb-2 text-green-800">Preditor de Risco de Desmatamento</h2>
          <p class="text-sm text-slate-600 mb-3">
            Sistema de machine learning que atribui um <strong>score de risco (0–100)</strong> a cada um dos
            808 municípios da Amazônia Legal, permitindo identificar onde o desmatamento tem maior
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
                <li>Score: ranking percentílico por ano (0–100)</li>
              </ul>
            </div>
          </div>
        </div>
        <div class="bg-green-50 border border-green-200 rounded-lg p-4 text-sm">
          <h3 class="font-semibold text-green-800 mb-2">Desempenho do Modelo (Spatial CV)</h3>
          <div class="grid grid-cols-4 gap-3" id="sobre-metrics"></div>
          <div style="margin-top:12px;padding:10px 14px;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;font-size:12px;color:#78350f;line-height:1.6">
            <b>Nota sobre variabilidade do WMAPE entre folds (Spatial CV):</b>
            O fold <b>RR+TO</b> apresenta WMAPE de ~56%, bem acima dos demais (MA: 30%, PA: 25%, MT: 32%, AC/AM/AP/RO: 29%).
            A causa é estrutural: esses estados têm taxa de desmatamento muito baixa (~0,038%/ano), e a fórmula
            <code>WMAPE = Σ|erro| / Σy_real</code> amplifica qualquer resíduo quando o denominador (volume total de desmatamento) é pequeno.
            Para estados de alta pressão — onde o modelo é operacionalmente mais relevante — o desempenho converge entre 25%–32%.
          </div>
        </div>
        <p class="text-xs text-slate-400 mt-4">Gerado em __TODAY__ · Dados: INPE · IBGE · ICMBio · IBAMA</p>
      </div>

    </div>
  </main>
</div>

<script>
// ── Dados embutidos ────────────────────────────────────────────────────────────
const ANOS      = __ANOS__;
const UFS       = __UFS__;
const GEODATA   = __GEODATA__;
const ALLDATA   = __ALLDATA__;
const SERIES    = __SERIES__;
const CLUSTERS  = __CLUSTERS__;
const METRICS   = __METRICS__;
const SHAP_IMGS = __SHAP_IMGS__;

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
  if (!state.ufs.includes(feature.properties.uf)) {
    return { fillOpacity: 0, color: '#cbd5e1', weight: 0.3, fillColor: 'transparent' };
  }
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
    const p    = feature.properties;
    const row  = (scoreLookup[state.ano] || {})[p.cod_ibge] || p;
    const sc   = row.score_risco != null ? row.score_risco.toFixed(1) : '—';
    const prev = row.ano_tipo === 'previsão';
    // Para 2026 (previsão), mostrar taxa prevista; para histórico, taxa real
    const taxaVal = prev
      ? row.taxa_desmatamento_prevista
      : row.taxa_desmatamento;
    const tx  = taxaVal != null ? taxaVal.toFixed(4) : '—';
    const dm  = row.desmatamento_km2 != null ? row.desmatamento_km2.toFixed(2) : '—';
    const taxaLabel = prev ? 'Taxa prevista (modelo)' : 'Taxa desmat. (real)';
    const prevBadge = prev
      ? '<span style="background:#fef3c7;color:#92400e;padding:1px 6px;border-radius:9999px;font-size:11px;font-weight:600;">Previsão 2026</span><br>'
      : '';
    layer.bindPopup(`
      <div style="min-width:180px;font-size:13px">
        <b style="font-size:14px">${p.municipio}</b><br>
        <span style="color:#64748b">UF: ${p.uf}</span><br>
        ${prevBadge}<br>
        <b>Score de risco:</b> ${sc} / 100<br>
        <b>${taxaLabel}:</b> ${tx} %/ano<br>
        ${!prev ? `<b>Desmatamento:</b> ${dm} km²` : ''}
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
function taxaEfetiva(r) {
  // Para ano de previsão, taxa_desmatamento é null — usar taxa prevista
  return r.taxa_desmatamento != null ? r.taxa_desmatamento : (r.taxa_desmatamento_prevista || 0);
}

function updateKPIs() {
  const rows = (ALLDATA[state.ano] || []).filter(r => state.ufs.includes(r.uf));
  const isPrevisao = rows.length > 0 && rows[0].ano_tipo === 'previsão';
  const desmat = rows.reduce((s, r) => s + (r.desmatamento_km2 || 0), 0);
  const taxa   = rows.length ? rows.reduce((s, r) => s + taxaEfetiva(r), 0) / rows.length : 0;
  const scores = rows.map(r => r.score_risco).filter(s => s != null);
  const scoreMed = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
  const high = scores.filter(s => s >= 80).length;

  // Desmatamento total: real para histórico, estimado para previsão
  let desmatVal, desmatLabel;
  if (isPrevisao) {
    const desmatEst = rows.reduce((s, r) => s + (r.desmatamento_km2_previsto || 0), 0);
    desmatVal   = desmatEst.toLocaleString('pt-BR', {maximumFractionDigits: 0}) + ' km²';
    desmatLabel = 'Desmatamento Total (est.)';
  } else {
    desmatVal   = desmat.toLocaleString('pt-BR', {maximumFractionDigits: 0}) + ' km²';
    desmatLabel = 'Desmatamento Total';
  }
  document.getElementById('kpi-desmat-label').textContent = desmatLabel;
  document.getElementById('kpi-desmat').textContent = desmatVal;
  document.getElementById('kpi-taxa').textContent   = taxa.toFixed(4) + ' %' + (isPrevisao ? ' (prev.)' : '');
  document.getElementById('kpi-score').textContent  = scoreMed != null ? scoreMed.toFixed(1) : '—';
  document.getElementById('kpi-high').textContent   = high.toLocaleString('pt-BR');
  const anoLabel = isPrevisao ? `${state.ano} (previsão)` : String(state.ano);
  document.getElementById('header-sub').textContent = `Ano: ${anoLabel} · ${rows.length} municípios filtrados`;
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
  const isPrevisao = rows.length > 0 && rows[0].ano_tipo === 'previsão';
  const thDesmat = document.getElementById('ranking-th-desmat');
  if (thDesmat) thDesmat.textContent = isPrevisao ? 'Desmat. est. (km²)' : 'Desmat. (km²)';
  tbody.innerHTML = rows.map((r, i) => `
    <tr>
      <td class="px-3 py-2 text-slate-400">${i + 1}</td>
      <td class="px-3 py-2 font-medium">${r.municipio}</td>
      <td class="px-3 py-2 text-slate-500">${r.uf}</td>
      <td class="px-3 py-2 text-right">${badgeHtml(r.score_risco)}</td>
      <td class="px-3 py-2 text-right text-slate-600">${taxaEfetiva(r).toFixed(4)}${isPrevisao ? ' *' : ''}</td>
      <td class="px-3 py-2 text-right text-slate-600">${isPrevisao
        ? (r.desmatamento_km2_previsto != null ? r.desmatamento_km2_previsto.toFixed(1) + ' *' : '—')
        : (r.desmatamento_km2||0).toFixed(2)}</td>
    </tr>
  `).join('') + (isPrevisao ? '<tr><td colspan="6" class="px-3 py-1 text-xs text-amber-600 italic">* Valores estimados: taxa prevista × área do município (dados INPE/PRODES para 2026 ainda não disponíveis)</td></tr>' : '');
}

// ── Gráficos Chart.js ─────────────────────────────────────────────────────────
// Fonte global — herda Quantico carregada no <head>
Chart.defaults.font.family = "'Quantico', sans-serif";
Chart.defaults.font.size   = 13;
const CHART_COLORS = ['#8dbfa8','#89b8cc','#d4956a','#b09ac8','#c8a85e','#d48898','#6cbab0','#98bc70','#c4906a'];
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

  const gridColor = getComputedStyle(document.documentElement).getPropertyValue('--chart-grid').trim() || '#d5cfc5';
  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'bottom', labels: { font: { size: 13 } } } },
    scales: {
      x: { ticks: { font: { size: 13 } }, grid: { color: gridColor } },
      y: { ticks: { font: { size: 13 } }, grid: { color: gridColor } },
    },
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
  const m = (METRICS || {}).mean || (METRICS || {}).MEDIA;
  const sobreBox = document.getElementById('sobre-metrics');
  if (!m || !sobreBox) return;
  const wmapeStr = m.wmape != null ? `${m.wmape.toFixed(1)}%` : '—';
  const metrics4 = [
    { label: 'R²',    value: (m.r2  ||0).toFixed(4), hint: 'variância explicada' },
    { label: 'RMSE',  value: (m.rmse||0).toFixed(4), hint: '%/ano' },
    { label: 'MAE',   value: (m.mae ||0).toFixed(4), hint: '%/ano' },
    { label: 'WMAPE', value: wmapeStr,                hint: 'erro percentual ponderado' },
  ];
  sobreBox.innerHTML = metrics4.map(({ label, value, hint }) => `
    <div style="background:var(--accent-soft);border:1px solid var(--accent);border-radius:6px;padding:10px 14px">
      <div style="font-size:11px;color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:.4px">${label}</div>
      <div style="font-size:20px;font-weight:700;color:var(--text-primary);margin:4px 0 2px;font-family:'Quantico',sans-serif">${value}</div>
      <div style="font-size:11px;color:var(--text-muted)">${hint}</div>
    </div>
  `).join('');
}

// ── Sidebar: inicializar filtros ──────────────────────────────────────────────
function initSidebar() {
  // Select de ano
  const selAno = document.getElementById('sel-ano');
  const PREV_ANO = __LAST_ANO__;
  ANOS.slice().reverse().forEach(a => {
    const opt = document.createElement('option');
    opt.value = a;
    opt.textContent = a === PREV_ANO ? `${a} (previsão)` : String(a);
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
// Mapa de cor por cluster_id (cores atribuídas por rank de risco em Python)
const clusterColorMap = {};
const clusterBgMap = {};
CLUSTERS.forEach(c => { clusterColorMap[c.id] = c.color; clusterBgMap[c.id] = c.bg ?? '#f8fafc'; });

function renderClusters() {
  const container = document.getElementById('clusters-cards');
  if (!container || !CLUSTERS.length) return;

  const statsEl = document.getElementById('cluster-stats-text');
  if (statsEl) {
    const n = CLUSTERS.filter(c => c.id >= 0).length;
    const noiseN = CLUSTERS.filter(c => c.id === -1).reduce((s, c) => s + c.n, 0);
    statsEl.textContent = `${n} clusters · ${noiseN > 0 ? noiseN + ' municípios ruído' : 'sem ruído'}`;
  }

  const sortedClusters = [...CLUSTERS].sort((a, b) =>
    a.id === -1 ? 1 : b.id === -1 ? -1 : a.score - b.score
  );
  container.innerHTML = sortedClusters.map(c => {
    const isNoise = c.id === -1;
    const color   = clusterColorMap[c.id] ?? '#94a3b8';
    const bg      = clusterBgMap[c.id]    ?? '#f1f5f9';
    const title   = isNoise ? 'Ruído — sem cluster' : c.label;

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

// ── Mapa de Clusters ──────────────────────────────────────────────────────────
let clusterLeafletMap = null;
let clusterMapBuilt  = false;

function buildClusterMap() {
  if (clusterMapBuilt) return;
  clusterMapBuilt = true;

  clusterLeafletMap = L.map('cluster-map', { zoomControl: true }).setView([-6, -55], 5);
  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© CartoDB © OpenStreetMap', maxZoom: 19,
  }).addTo(clusterLeafletMap);

  const lastYear = ANOS[ANOS.length - 1];
  const clusterLookup = {};
  for (const r of (ALLDATA[lastYear] || [])) {
    clusterLookup[r.cod_ibge] = r.cluster_id != null ? r.cluster_id : -1;
  }

  L.geoJSON(GEODATA, {
    style(feature) {
      const cid = clusterLookup[feature.properties.cod_ibge] ?? -1;
      const color = clusterColorMap[cid] ?? '#94a3b8';
      return { fillColor: color, fillOpacity: 0.75, color: '#94a3b8', weight: 0.4 };
    },
    onEachFeature(feature, layer) {
      const p   = feature.properties;
      const cid = clusterLookup[p.cod_ibge] ?? -1;
      layer.bindTooltip(
        `<div style="font-size:12px"><b>${p.municipio}</b> (${p.uf})<br>` +
        `${cid === -1 ? 'Não classificado' : (CLUSTERS.find(c => c.id === cid)?.label ?? 'Cluster ' + cid)}</div>`
      );
    },
  }).addTo(clusterLeafletMap);

  // Legenda
  const legendCtrl = L.control({ position: 'bottomright' });
  legendCtrl.onAdd = () => {
    const div = L.DomUtil.create('div', 'leaflet-bar');
    div.style.cssText = 'background:white;padding:8px 12px;font-size:11px;line-height:1.7;border-radius:6px;max-width:240px';
    const legendClusters = [...CLUSTERS].sort((a, b) =>
      a.id === -1 ? 1 : b.id === -1 ? -1 : a.score - b.score
    );
    div.innerHTML = '<b style="font-size:12px">Clusters HDBSCAN</b><br>' +
      legendClusters.map(c => {
        const color = clusterColorMap[c.id] ?? '#94a3b8';
        const lbl   = c.id === -1 ? 'Ruído' : (c.label.length > 32 ? c.label.slice(0, 30) + '…' : c.label);
        return `<span style="display:inline-block;width:10px;height:10px;background:${color};border-radius:2px;margin-right:4px;vertical-align:middle"></span>${lbl}`;
      }).join('<br>');
    return div;
  };
  legendCtrl.addTo(clusterLeafletMap);

  setTimeout(() => clusterLeafletMap.invalidateSize(), 100);
}

// ── Renderizar imagens SHAP ───────────────────────────────────────────────────
let shapRendered = false;
function renderShap() {
  if (shapRendered) return;
  shapRendered = true;

  function imgOrMsg(b64, alt) {
    if (!b64) return `<p class="text-slate-400 italic text-sm">Imagem não encontrada — execute <code>python -m src.model.evaluate</code></p>`;
    return `<img src="data:image/png;base64,${b64}" alt="${alt}" style="max-width:100%;height:auto">`;
  }

  document.getElementById('shap-beeswarm-wrap').innerHTML = imgOrMsg(SHAP_IMGS.beeswarm, 'SHAP Beeswarm');
  document.getElementById('shap-cluster-wrap').innerHTML  = imgOrMsg(SHAP_IMGS.cluster,  'SHAP por Cluster');

  const wfKeys = [
    { key: 'wf_nova_santa_helena', label: 'Nova Santa Helena (MT)' },
    { key: 'wf_marcelandia',       label: 'Marcelândia (MT)' },
    { key: 'wf_mojui',             label: 'Mojuí dos Campos (PA)' },
    { key: 'wf_nova_esperanca',    label: 'Nova Esperança do Piriá (PA)' },
    { key: 'wf_uniao_sul',         label: 'União do Sul (MT)' },
  ];
  const grid = document.getElementById('shap-waterfall-grid');
  grid.innerHTML = wfKeys.map(({ key, label }) => `
    <div class="chart-wrap rounded border border-slate-200 p-3">
      <div class="text-xs font-semibold text-slate-600 mb-2">${label}</div>
      ${imgOrMsg(SHAP_IMGS[key], label)}
    </div>
  `).join('');
}

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    ['mapa','ranking','clusters','tendencia','shap','dicionario','sobre'].forEach(t => {
      document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
    });
    if (tab === 'tendencia')  buildCharts();
    if (tab === 'clusters') { buildClusterMap(); renderClusters(); }
    if (tab === 'shap')       renderShap();
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

// ── Modo escuro ───────────────────────────────────────────────────────────────
function toggleDark() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('dark-mode', isDark ? '1' : '0');
  document.getElementById('dark-toggle').textContent = isDark ? '○ Tema' : '◑ Tema';
}
(function initTheme() {
  const saved = localStorage.getItem('dark-mode');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (saved === '1' || (saved === null && prefersDark)) {
    document.documentElement.classList.add('dark');
    const btn = document.getElementById('dark-toggle');
    if (btn) btn.textContent = '○ Tema';
  }
})();

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
