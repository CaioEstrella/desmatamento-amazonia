"""
Dashboard Streamlit — Preditor de Risco de Desmatamento (Amazônia Legal) · TASK-15.

Execução:
    streamlit run src/dashboard/app.py

Abas:
    1. Mapa       — coroplético de score_risco (Folium)
    2. Ranking    — top-20 municípios de maior risco
    3. Clusters   — perfis HDBSCAN dos municípios
    4. Série      — evolução histórica por estado
    5. SHAP       — importância de features
    6. Dicionário — descrição de cada feature
    7. Sobre      — metodologia, fontes e métricas do modelo
"""

from __future__ import annotations

import json
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Risco de Desmatamento — Amazônia Legal",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Silkscreen:wght@400;700&family=Quantico:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Quantico', 'Inter', system-ui, sans-serif !important;
}
h1 {
    font-family: 'Silkscreen', monospace !important;
    color: #4a7c5e !important;
    font-size: 1.3rem !important;
}
h2, h3 {
    font-family: 'Silkscreen', monospace !important;
    color: #4a7c5e !important;
}
/* Dark mode override — Silkscreen uses lighter green */
@media (prefers-color-scheme: dark) {
    h1, h2, h3 { color: #85c4a0 !important; }
}
section[data-testid="stSidebar"] {
    background-color: #fdfaf7 !important;
}
.stApp {
    background-color: #f5f1eb !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #f7f3ed;
    border-bottom: 1px solid #ddd8ce;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Quantico', sans-serif !important;
    color: #8a9e94 !important;
    font-size: 13px !important;
}
.stTabs [aria-selected="true"] {
    color: #4a7c5e !important;
    border-bottom: 2px solid #4a7c5e !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

_PREDICTIONS_PATH = "data/outputs/predictions.parquet"
_METRICS_PATH     = "data/outputs/metrics_by_fold.json"
_SHAP_BEESWARM    = "data/outputs/shap_beeswarm.png"
_SHAP_CLUSTER     = "data/outputs/shap_por_cluster.png"

_CLUSTER_NAMES = {
    0: "Proteção por UCs reduz pressão",
    1: "Fronteira agrícola com Terras Indígenas",
    2: "Grandes municípios — proteção formal com pressão crescente",
    3: "Arco do desmatamento — alta pressão sem áreas protegidas",
    4: "Municípios periféricos — baixa pressão de desmatamento",
}

_RISK_COLORS = ["#22c55e", "#86efac", "#eab308", "#f97316", "#ef4444"]
_RISK_BG     = ["#dcfce7", "#d1fae5", "#fef9c3", "#ffedd5", "#fee2e2"]

_SCORE_BANDS = [(20, "#22c55e"), (40, "#86efac"), (60, "#eab308"), (80, "#f97316"), (101, "#ef4444")]

def _score_color(score) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "#cbd5e1"
    for threshold, color in _SCORE_BANDS:
        if score <= threshold:
            return color
    return "#ef4444"

_FEATURE_DICT = [
    {"Feature": "taxa_desmatamento",          "Unidade": "% / ano",      "Descrição": "Taxa de perda de cobertura vegetal nativa em relação à área total do município",                                                        "Fonte": "INPE/PRODES"},
    {"Feature": "taxa_desmatamento_prevista",  "Unidade": "% / ano",      "Descrição": "Previsão do modelo LightGBM — disponível apenas para 2026 (dados INPE ainda não medidos)",                                              "Fonte": "Modelo"},
    {"Feature": "desmatamento_km2",            "Unidade": "km²",          "Descrição": "Área absoluta desmatada no ano",                                                                                                          "Fonte": "INPE/PRODES"},
    {"Feature": "score_risco",                 "Unidade": "0–100",        "Descrição": "Ranking percentílico da taxa de desmatamento dentro do mesmo ano — score 80 = maior desmatamento que 80% dos municípios naquele ano",    "Fonte": "Calculado"},
    {"Feature": "area_km2",                    "Unidade": "km²",          "Descrição": "Área total do município",                                                                                                                  "Fonte": "IBGE 2022"},
    {"Feature": "populacao",                   "Unidade": "habitantes",   "Descrição": "População estimada do município no ano",                                                                                                   "Fonte": "IBGE"},
    {"Feature": "pib_agropecuario",            "Unidade": "R$ milhões",   "Descrição": "PIB do setor agropecuário municipal — proxy de pressão econômica sobre a terra",                                                          "Fonte": "IBGE"},
    {"Feature": "area_uc_pct",                 "Unidade": "%",            "Descrição": "Percentual do território coberto por Unidades de Conservação (parques, reservas, APAs) — barreira formal ao desmatamento",               "Fonte": "ICMBio"},
    {"Feature": "area_ti_pct",                 "Unidade": "%",            "Descrição": "Percentual do território coberto por Terras Indígenas demarcadas — barreira legal reconhecida ao desmatamento",                          "Fonte": "FUNAI / ICMBio"},
    {"Feature": "autos_ibama",                 "Unidade": "contagem",     "Descrição": "Número de autos de infração ambiental emitidos pelo IBAMA — indica pressão e presença de fiscalização",                                  "Fonte": "IBAMA"},
    {"Feature": "local_moran_i",               "Unidade": "índice",       "Descrição": "Autocorrelação espacial local (LISA) — valor positivo indica desmatamento semelhante ao dos municípios vizinhos (efeito contágio)",      "Fonte": "Calculado (esda)"},
    {"Feature": "cluster_id",                  "Unidade": "inteiro (0–4)","Descrição": "Grupo HDBSCAN — agrupa municípios com perfil estrutural similar de pressão de desmatamento",                                             "Fonte": "HDBSCAN"},
    {"Feature": "taxa_desmat_lag1",            "Unidade": "% / ano",      "Descrição": "Taxa de desmatamento do ano anterior (t−1) — feature temporal de primeira ordem",                                                        "Fonte": "Calculado"},
    {"Feature": "taxa_desmat_lag2",            "Unidade": "% / ano",      "Descrição": "Taxa de desmatamento de dois anos atrás (t−2)",                                                                                           "Fonte": "Calculado"},
    {"Feature": "taxa_desmat_lag3",            "Unidade": "% / ano",      "Descrição": "Taxa de desmatamento de três anos atrás (t−3)",                                                                                           "Fonte": "Calculado"},
    {"Feature": "taxa_desmat_roll3",           "Unidade": "% / ano",      "Descrição": "Média móvel da taxa de desmatamento nos últimos 3 anos — suaviza ruído anual",                                                            "Fonte": "Calculado"},
    {"Feature": "taxa_desmat_roll5",           "Unidade": "% / ano",      "Descrição": "Média móvel da taxa de desmatamento nos últimos 5 anos — tendência de médio prazo",                                                       "Fonte": "Calculado"},
    {"Feature": "std_desmat_roll3",            "Unidade": "% / ano",      "Descrição": "Desvio padrão da taxa nos últimos 3 anos — captura volatilidade ou instabilidade recente",                                               "Fonte": "Calculado"},
]


@st.cache_data(show_spinner="Carregando dados...")
def load_data() -> gpd.GeoDataFrame:
    return gpd.read_parquet(_PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    p = Path(_METRICS_PATH)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_geodata_simplified() -> gpd.GeoDataFrame:
    """Geometrias simplificadas para o mapa Folium — evita MessageSizeError."""
    gdf = gpd.read_parquet(_PREDICTIONS_PATH)
    last_ano = int(gdf["ano"].max())
    geo = gdf[gdf["ano"] == last_ano][["cod_ibge", "geometry"]].copy()
    geo["geometry"] = geo["geometry"].simplify(0.01)
    return geo


@st.cache_data(show_spinner=False)
def build_cluster_profiles(_gdf: gpd.GeoDataFrame) -> list[dict]:
    """Replica a lógica de _build_clusters() do build_html.py."""
    if "cluster_id" not in _gdf.columns:
        return []

    hist = _gdf[_gdf["ano_tipo"] == "histórico"] if "ano_tipo" in _gdf.columns else _gdf

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

    recente   = hist[hist["ano"] >= 2020].groupby("cod_ibge")["taxa_desmatamento"].mean()
    historico = hist[hist["ano"] <= 2015].groupby("cod_ibge")["taxa_desmatamento"].mean()
    trend_map = (recente / (historico + 1e-6)).clip(0, 10)
    agg["trend"] = agg["cod_ibge"].map(trend_map).fillna(1.0)

    # Score médio do modelo por município (usa todo o gdf, não só histórico)
    score_map = (
        _gdf[_gdf["score_risco"].notna()]
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

    def _label(row) -> str:
        cid   = int(row["cluster_id"])
        taxa  = row["taxa"]
        uc    = row["uc"]
        ti    = row["ti"]
        area  = row["area"]
        pib   = row["pib"]
        trend = row["trend"]
        prot  = uc + ti
        if taxa > 0.35 and prot < 10:
            return "Arco do desmatamento — alta pressão sem áreas protegidas"
        if area > 10_000 and prot > 35 and trend > 1.5:
            return "Grandes municípios — proteção formal com pressão crescente"
        if ti > 15 and taxa > 0.12:
            return "Fronteira agrícola com presença de Terras Indígenas"
        if uc > 25 and ti < 10 and taxa < 0.15:
            return "Proteção por Unidades de Conservação reduz pressão"
        if area < 2_500 and pib < 80 and taxa < 0.12:
            return "Municípios periféricos com baixa pressão de desmatamento"
        return _CLUSTER_NAMES.get(cid, f"Cluster {cid}")

    result = []
    for _, row in profiles.iterrows():
        result.append({
            "id":    int(row["cluster_id"]),
            "label": _label(row),
            "n":     int(row["n"]),
            "taxa":  float(row["taxa"]),
            "score": round(float(row["score"]), 1),
            "area":  float(row["area"]) / 1000,
            "pop":   float(row["pop"]) / 1000,
            "pib":   float(row["pib"]) / 1000,
            "uc":    float(row["uc"]),
            "ti":    float(row["ti"]),
            "autos": float(row["autos"]),
            "trend": float(row["trend"]),
        })

    # Retorna sem cores — cores aplicadas fora do cache em _apply_risk_colors()
    return result


def _apply_risk_colors(profiles: list[dict]) -> list[dict]:
    """Atribui cores verde→amarelo→laranja→roxo→vermelho por rank de score_risco.

    Separado do cache para que mudanças em _RISK_COLORS sejam sempre refletidas.
    """
    profiles = [dict(r) for r in profiles]  # cópia rasa — não modifica o objeto cacheado
    valid = sorted([r for r in profiles if r["id"] >= 0], key=lambda x: x["score"])
    n = len(valid)
    for rank, cluster in enumerate(valid):
        idx = round(rank * (len(_RISK_COLORS) - 1) / max(n - 1, 1))
        cluster["color"] = _RISK_COLORS[idx]
        cluster["bg"]    = _RISK_BG[idx]
    for r in profiles:
        if r["id"] == -1:
            r["color"] = "#94a3b8"
            r["bg"]    = "#f1f5f9"
    return profiles


@st.cache_data(show_spinner=False)
def load_cluster_geodata(_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Geometrias simplificadas com cluster_id para o mapa de clusters."""
    hist = _gdf[_gdf["ano_tipo"] == "histórico"].copy() if "ano_tipo" in _gdf.columns else _gdf.copy()
    mun_info = (
        hist.groupby("cod_ibge")
        .agg(cluster_id=("cluster_id", "first"),
             municipio=("municipio", "first"),
             uf=("uf", "first"))
        .reset_index()
    )
    last_ano = int(_gdf["ano"].max())
    geo = _gdf[_gdf["ano"] == last_ano][["cod_ibge", "geometry"]].copy()
    geo["geometry"] = geo["geometry"].simplify(0.01)
    gdf_cl = geo.merge(mun_info, on="cod_ibge", how="inner")
    gdf_cl["cluster_id"] = gdf_cl["cluster_id"].fillna(-1).astype(int)
    return gpd.GeoDataFrame(gdf_cl, geometry="geometry", crs="EPSG:4326")


# ── Carregar dados ────────────────────────────────────────────────────────────
gdf = load_data()
metrics = load_metrics()
geo_simplified = load_geodata_simplified()

anos     = sorted(gdf["ano"].unique())
ufs      = sorted(gdf["uf"].unique())
clusters = sorted(gdf["cluster_id"].dropna().unique()) if "cluster_id" in gdf.columns else []

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Filtros")

ano_labels = {a: f"{a} (previsão)" if a == 2026 else str(a) for a in anos}
ano_sel = st.sidebar.selectbox(
    "Ano", anos, index=len(anos) - 1,
    format_func=lambda a: ano_labels[a],
)
uf_sel = st.sidebar.multiselect("Estado (UF)", ufs, default=ufs)

cluster_labels = {
    int(c): _CLUSTER_NAMES.get(int(c), f"Cluster {int(c)}")
    for c in clusters
}
cluster_sel = st.sidebar.multiselect(
    "Cluster HDBSCAN",
    options=list(cluster_labels.keys()),
    default=list(cluster_labels.keys()),
    format_func=lambda c: cluster_labels[c],
) if clusters else None


# ── Filtro de dados ───────────────────────────────────────────────────────────
mask = (gdf["ano"] == ano_sel) & (gdf["uf"].isin(uf_sel))
if cluster_sel is not None and "cluster_id" in gdf.columns:
    mask &= gdf["cluster_id"].isin(cluster_sel)

gdf_view = gdf[mask].copy()
has_score = gdf_view["score_risco"].notna().any()

is_prev = (
    "ano_tipo" in gdf_view.columns
    and (gdf_view["ano_tipo"] == "previsão").any()
)

# Desmatamento estimado para 2026
if is_prev and "taxa_desmatamento_prevista" in gdf_view.columns and "area_km2" in gdf_view.columns:
    gdf_view["desmatamento_km2_previsto"] = (
        gdf_view["taxa_desmatamento_prevista"] / 100 * gdf_view["area_km2"]
    ).round(1)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("Preditor de Risco de Desmatamento — Amazônia Legal")
ano_header = ano_labels[ano_sel]
st.markdown(
    f"**Ano:** {ano_header}  ·  **Estados:** {', '.join(uf_sel)}  ·  "
    f"**Municípios:** {len(gdf_view):,}"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    if is_prev and "desmatamento_km2_previsto" in gdf_view.columns:
        desmat_prev = gdf_view["desmatamento_km2_previsto"].sum()
        st.metric("Desmatamento total (est.)", f"{desmat_prev:,.0f} km²",
                  help="Estimativa: taxa prevista × área do município")
    else:
        total_desmat = gdf_view["desmatamento_km2"].sum()
        st.metric("Desmatamento total", f"{total_desmat:,.0f} km²")

with col2:
    if is_prev and "taxa_desmatamento_prevista" in gdf_view.columns:
        media_taxa = gdf_view["taxa_desmatamento_prevista"].mean()
        st.metric("Taxa prevista (modelo)", f"{media_taxa:.4f} %/ano")
    else:
        media_taxa = gdf_view["taxa_desmatamento"].mean()
        st.metric("Taxa média", f"{media_taxa:.4f} %/ano")

with col3:
    if has_score:
        score_med = gdf_view["score_risco"].mean()
        st.metric("Score risco médio", f"{score_med:.1f} / 100")
    else:
        st.metric("Score risco médio", "—")

with col4:
    mun_alto_risco = (gdf_view["score_risco"] >= 80).sum() if has_score else 0
    st.metric("Municípios score ≥ 80", f"{mun_alto_risco:,}")

st.markdown("---")

# ── Layout principal ──────────────────────────────────────────────────────────
(tab_mapa, tab_ranking, tab_clusters,
 tab_serie, tab_shap, tab_dicionario, tab_sobre) = st.tabs([
    "⊕ Mapa de risco", "≡ Ranking", "◉ Clusters",
    "↗ Série temporal", "∑ SHAP", "∷ Dicionário", "◦ Sobre",
])

# ── Tab 1: Mapa Folium ────────────────────────────────────────────────────────
with tab_mapa:
    st.subheader(f"Score de risco de desmatamento — {ano_header}")

    if not has_score:
        st.info(f"Score não disponível para {ano_sel} (lag features exigem ano ≥ 2011).")
    else:
        df_map = gdf_view[gdf_view["score_risco"].notna()].drop(columns="geometry")
        gdf_map = geo_simplified.merge(df_map, on="cod_ibge", how="inner")
        gdf_map = gpd.GeoDataFrame(gdf_map, geometry="geometry", crs="EPSG:4326")

        m = folium.Map(location=[-6.0, -55.0], zoom_start=5, tiles="CartoDB positron")

        score_lookup = dict(zip(gdf_map["cod_ibge"], gdf_map["score_risco"]))

        def _map_style(feature):
            sc = score_lookup.get(feature["properties"]["cod_ibge"])
            return {
                "fillColor": _score_color(sc),
                "color": "#94a3b8",
                "weight": 0.4,
                "fillOpacity": 0.75,
            }

        if is_prev and "taxa_desmatamento_prevista" in gdf_map.columns:
            tooltip_cols = ["municipio", "uf", "score_risco", "taxa_desmatamento_prevista"]
            aliases = ["Município", "UF", "Score de risco", "Taxa prevista (%/ano)"]
        else:
            tooltip_cols = ["municipio", "uf", "score_risco", "taxa_desmatamento", "desmatamento_km2"]
            aliases = ["Município", "UF", "Score de risco", "Taxa desmat. (%/ano)", "Desmat. (km²)"]

        tooltip_cols = [c for c in tooltip_cols if c in gdf_map.columns]
        folium.GeoJson(
            gdf_map[tooltip_cols + ["geometry", "cod_ibge"]],
            style_function=_map_style,
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_cols,
                aliases=aliases[:len(tooltip_cols)],
                localize=True,
            ),
        ).add_to(m)

        _legend_html = """
<div style="position:fixed;bottom:30px;right:10px;z-index:9999;background:white;padding:10px 14px;
     border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);font-size:12px;line-height:1.8;font-family:sans-serif">
  <b style="font-size:13px">Score de Risco</b><br>
  <span style="display:inline-block;width:12px;height:12px;background:#22c55e;border-radius:2px;margin-right:5px;vertical-align:middle"></span>0–20<br>
  <span style="display:inline-block;width:12px;height:12px;background:#86efac;border-radius:2px;margin-right:5px;vertical-align:middle"></span>21–40<br>
  <span style="display:inline-block;width:12px;height:12px;background:#eab308;border-radius:2px;margin-right:5px;vertical-align:middle"></span>41–60<br>
  <span style="display:inline-block;width:12px;height:12px;background:#f97316;border-radius:2px;margin-right:5px;vertical-align:middle"></span>61–80<br>
  <span style="display:inline-block;width:12px;height:12px;background:#ef4444;border-radius:2px;margin-right:5px;vertical-align:middle"></span>81–100<br>
</div>"""
        m.get_root().html.add_child(folium.Element(_legend_html))

        components.html(m._repr_html_(), height=520, scrolling=False)

# ── Tab 2: Ranking ────────────────────────────────────────────────────────────
with tab_ranking:
    st.subheader(f"Top-20 municípios por score de risco — {ano_header}")

    if not has_score:
        st.info(f"Score não disponível para {ano_sel}.")
    else:
        if is_prev and "taxa_desmatamento_prevista" in gdf_view.columns:
            cols_show = ["municipio", "uf", "score_risco",
                         "taxa_desmatamento_prevista", "desmatamento_km2_previsto", "cluster_id"]
            rename = {
                "municipio": "Município", "uf": "UF",
                "score_risco": "Score (0–100)",
                "taxa_desmatamento_prevista": "Taxa prevista (%/ano)",
                "desmatamento_km2_previsto": "Desmat. est. (km²)",
                "cluster_id": "Cluster",
            }
            fmt = {"Score (0–100)": "{:.1f}", "Taxa prevista (%/ano)": "{:.4f}", "Desmat. est. (km²)": "{:.1f}"}
            st.caption("▲ Taxa e desmatamento estimados pelo modelo LightGBM — dados INPE/PRODES para 2026 ainda não disponíveis.")
        else:
            cols_show = ["municipio", "uf", "score_risco",
                         "taxa_desmatamento", "desmatamento_km2", "cluster_id"]
            rename = {
                "municipio": "Município", "uf": "UF",
                "score_risco": "Score (0–100)", "taxa_desmatamento": "Taxa (%/ano)",
                "desmatamento_km2": "Desmat. (km²)", "cluster_id": "Cluster",
            }
            fmt = {"Score (0–100)": "{:.1f}", "Taxa (%/ano)": "{:.4f}", "Desmat. (km²)": "{:.2f}"}

        cols_show = [c for c in cols_show if c in gdf_view.columns]
        top20 = (
            gdf_view[gdf_view["score_risco"].notna()]
            .nlargest(20, "score_risco")[cols_show]
            .reset_index(drop=True)
        )
        top20.index += 1

        st.dataframe(
            top20.rename(columns=rename).style
            .background_gradient(subset=["Score (0–100)"], cmap="RdYlGn_r")
            .format(fmt),
            use_container_width=True,
        )

# ── Tab 3: Clusters ───────────────────────────────────────────────────────────
with tab_clusters:
    st.subheader("Distribuição Espacial dos Clusters — HDBSCAN")

    gdf_cl = load_cluster_geodata(gdf)
    cluster_profiles = _apply_risk_colors(build_cluster_profiles(gdf))
    color_map = {c["id"]: c["color"] for c in cluster_profiles}

    m_cl = folium.Map(location=[-6.0, -55.0], zoom_start=5, tiles="CartoDB positron")

    cl_lookup = dict(zip(gdf_cl["cod_ibge"], gdf_cl["cluster_id"]))

    def _cl_style(feature):
        raw = cl_lookup.get(feature["properties"]["cod_ibge"])
        cid = -1 if raw is None else int(raw)
        color = color_map.get(cid, "#94a3b8")
        return {"fillColor": color, "color": "#94a3b8", "weight": 0.4, "fillOpacity": 0.75}

    folium.GeoJson(
        gdf_cl[["geometry", "cod_ibge", "municipio", "uf", "cluster_id"]],
        style_function=_cl_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["municipio", "uf", "cluster_id"],
            aliases=["Município", "UF", "Cluster"],
            localize=True,
        ),
    ).add_to(m_cl)

    _cl_legend = (
        "<div style='position:fixed;bottom:30px;right:10px;z-index:9999;background:white;"
        "padding:10px 14px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);"
        "font-size:12px;line-height:1.8;font-family:sans-serif'><b>Clusters HDBSCAN</b><br>"
        + "".join([
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{c['color']};border-radius:2px;margin-right:5px;"
            f"vertical-align:middle'></span>{c['label'][:30]}{'…' if len(c['label']) > 30 else ''}<br>"
            for c in sorted(cluster_profiles, key=lambda x: x.get("score", 0)) if c["id"] >= 0
        ])
        + "</div>"
    )
    m_cl.get_root().html.add_child(folium.Element(_cl_legend))

    components.html(m_cl._repr_html_(), height=440, scrolling=False)

    st.subheader("Perfis por Cluster")

    if not cluster_profiles:
        st.info("Dados de clustering não disponíveis.")
    else:
        for c in sorted(cluster_profiles, key=lambda x: (x["id"] == -1, x.get("score", 0))):
            color = c["color"]
            bg    = c["bg"]
            bar_w = min(100.0, (c["taxa"] / 0.25) * 100)

            stats_items = [
                ("Municípios",      f"{c['n']:,}"),
                ("Taxa média",      f"{c['taxa']:.4f} %/ano"),
                ("Cobertura UC",    f"{c['uc']:.1f}%"),
                ("Cobertura TI",    f"{c['ti']:.1f}%"),
                ("Área média",      f"{c['area']:.1f} mil km²"),
                ("Pop. média",      f"{c['pop']:.1f} mil hab"),
                ("PIB agro médio",  f"R$ {c['pib']:.1f} M"),
                ("Autos IBAMA",     f"{c['autos']:,.0f}"),
            ]
            stats_html = "".join(
                f"<div style='min-width:130px'>"
                f"<div style='font-size:11px;color:#64748b'>{lbl}</div>"
                f"<div style='font-size:14px;font-weight:600;color:#1e293b'>{val}</div>"
                f"</div>"
                for lbl, val in stats_items
            )

            st.markdown(f"""
<div style="background:{bg};border:1px solid {color}50;border-radius:8px;padding:16px;margin-bottom:12px">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
    <span style="display:inline-block;width:14px;height:14px;background:{color};border-radius:3px;flex-shrink:0"></span>
    <span style="font-weight:600;font-size:14px;color:#0f172a">{c['label']}</span>
  </div>
  <div style="margin-bottom:10px">
    <div style="font-size:11px;color:#64748b;margin-bottom:3px">Taxa de desmatamento média</div>
    <div style="background:#e2e8f0;border-radius:4px;height:6px">
      <div style="background:{color};height:6px;border-radius:4px;width:{bar_w:.1f}%"></div>
    </div>
    <div style="font-size:11px;color:#475569;margin-top:2px">{c['taxa']:.4f} %/ano</div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:16px 24px">{stats_html}</div>
</div>""", unsafe_allow_html=True)

# ── Tab 4: Série temporal ─────────────────────────────────────────────────────
with tab_serie:
    st.subheader("Evolução da taxa de desmatamento por estado")

    import plotly.express as px

    df_serie = gdf[gdf["uf"].isin(uf_sel)].copy()
    if "taxa_desmatamento_prevista" in df_serie.columns:
        mask_fill = df_serie["taxa_desmatamento"].isna() & df_serie["taxa_desmatamento_prevista"].notna()
        df_serie.loc[mask_fill, "taxa_desmatamento"] = df_serie.loc[mask_fill, "taxa_desmatamento_prevista"]

    df_uf_ano = (
        df_serie.groupby(["uf", "ano"])
        .agg(taxa_media=("taxa_desmatamento", "mean"),
             desmat_total=("desmatamento_km2", "sum"))
        .reset_index()
    )

    _PASTEL_COLORS = ['#8dbfa8','#89b8cc','#d4956a','#b09ac8','#c8a85e','#d48898','#6cbab0','#98bc70','#c4906a']
    fig = px.line(
        df_uf_ano, x="ano", y="taxa_media", color="uf", markers=True,
        color_discrete_sequence=_PASTEL_COLORS,
        title="Taxa de desmatamento média por estado (% /ano) — 2026 baseado em previsão do modelo",
        labels={"taxa_media": "Taxa média (%/ano)", "ano": "Ano", "uf": "UF"},
    )
    _CHART_BG     = "#ede8dc"
    _CHART_GRID   = "#d5cfc5"
    _PASTEL_COLORS = ['#8dbfa8','#89b8cc','#d4956a','#b09ac8','#c8a85e','#d48898','#6cbab0','#98bc70','#c4906a']
    _AXIS_FONT    = dict(size=13)

    _FONT = dict(family="Quantico, sans-serif", size=13)
    fig.update_layout(
        height=420, legend_title="Estado",
        plot_bgcolor=_CHART_BG, paper_bgcolor=_CHART_BG,
        font=_FONT,
        yaxis=dict(showgrid=True, gridcolor=_CHART_GRID, gridwidth=0.7, tickfont=_AXIS_FONT),
        xaxis=dict(showgrid=False, tickfont=_AXIS_FONT),
        legend=dict(font=_FONT),
    )
    fig.update_traces(marker=dict(size=5))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Desmatamento total acumulado por estado (km²)")
    fig2 = px.area(
        df_uf_ano, x="ano", y="desmat_total", color="uf",
        color_discrete_sequence=_PASTEL_COLORS,
        title="Desmatamento acumulado (km²) por estado",
        labels={"desmat_total": "Desmatamento (km²)", "ano": "Ano", "uf": "UF"},
    )
    fig2.update_layout(
        height=380, legend_title="Estado",
        plot_bgcolor=_CHART_BG, paper_bgcolor=_CHART_BG,
        font=_FONT,
        yaxis=dict(showgrid=True, gridcolor=_CHART_GRID, gridwidth=0.7, tickfont=_AXIS_FONT),
        xaxis=dict(showgrid=False, tickfont=_AXIS_FONT),
        legend=dict(font=_FONT),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 5: SHAP ───────────────────────────────────────────────────────────────
with tab_shap:
    st.subheader("Importância global das features (SHAP beeswarm)")
    st.caption("Cada ponto = um município. Vermelho = valor alto da feature, azul = valor baixo. Posição no eixo X = impacto no score previsto.")
    if Path(_SHAP_BEESWARM).exists():
        st.image(_SHAP_BEESWARM, use_container_width=True)
    else:
        st.warning("Execute `python -m src.model.evaluate` para gerar os plots SHAP.")

    st.subheader("Importância SHAP por cluster HDBSCAN")
    st.caption("Comparação da importância relativa de cada feature nos diferentes perfis de municípios.")
    if Path(_SHAP_CLUSTER).exists():
        st.image(_SHAP_CLUSTER, use_container_width=True)
    else:
        st.warning("Execute `python -m src.model.evaluate` para gerar os plots SHAP.")

    st.subheader("Waterfall — top-5 municípios de maior risco (2026)")
    st.caption("Cada gráfico mostra a contribuição de cada feature para o score do município.")
    waterfall_files = sorted(Path("data/outputs").glob("shap_waterfall_*.png"))
    if waterfall_files:
        cols = st.columns(2)
        for i, wf in enumerate(waterfall_files[:5]):
            with cols[i % 2]:
                mun_name = wf.stem.replace("shap_waterfall_", "").replace("_", " ").title()
                st.image(str(wf), caption=mun_name, use_container_width=True)
    else:
        st.warning("Execute `python -m src.model.evaluate` para gerar os waterfall plots.")

# ── Tab 6: Dicionário ─────────────────────────────────────────────────────────
with tab_dicionario:
    st.subheader("Indicadores do Dashboard")
    st.caption("O que cada número no topo da página significa.")
    _KPI_DICT = [
        {
            "Indicador": "Desmatamento total (km²)",
            "Definição": (
                "Soma da área desmatada em todos os municípios filtrados no ano selecionado. "
                "Para 2026 é uma estimativa: taxa prevista pelo modelo × área do município "
                "(dados INPE/PRODES ainda não disponíveis)."
            ),
        },
        {
            "Indicador": "Taxa média (%/ano)",
            "Definição": (
                "Média da taxa de desmatamento entre os municípios filtrados. "
                "Taxa = área desmatada ÷ área total do município × 100. "
                "Mede a intensidade relativa do desmatamento, independente do tamanho do município — "
                "um município pequeno com taxa alta é tão preocupante quanto um grande com muito "
                "desmatamento absoluto. Para 2026 usa a previsão do modelo."
            ),
        },
        {
            "Indicador": "Score de risco médio (0–100)",
            "Definição": (
                "Média do ranking percentílico entre os municípios filtrados. "
                "O score é recalculado a cada ano: score 80 = o município desmatou mais do que "
                "80% dos 808 municípios da Amazônia Legal naquele mesmo ano. "
                "Permite comparar anos distintos mesmo que o nível absoluto de desmatamento varie."
            ),
        },
    ]
    st.dataframe(pd.DataFrame(_KPI_DICT), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Dicionário de Features do Modelo")
    st.caption("Descrição de todas as variáveis utilizadas no modelo e no dashboard.")
    df_dict = pd.DataFrame(_FEATURE_DICT)
    st.dataframe(df_dict, use_container_width=True, hide_index=True)
    st.caption(
        "* Features de lag e rolling window são construídas com dados reais INPE/PRODES para anos "
        "históricos; para 2026, usam os dados de 2025 como âncora mais recente."
    )

# ── Tab 7: Sobre ──────────────────────────────────────────────────────────────
with tab_sobre:
    st.subheader("Preditor de Risco de Desmatamento — Amazônia Legal")
    st.markdown(
        "Sistema de machine learning que atribui um **score de risco (0–100)** a cada um dos "
        "**808 municípios** da Amazônia Legal, permitindo identificar onde o desmatamento tem maior "
        "probabilidade de ocorrer no próximo ano."
    )

    col_fontes, col_metodo = st.columns(2)
    with col_fontes:
        st.markdown("**Fontes de Dados**")
        st.markdown(
            "- INPE/PRODES — Desmatamento km² (2008–2025)\n"
            "- IBGE — Área municipal, população, PIB agropecuário\n"
            "- ICMBio — Unidades de Conservação e Terras Indígenas\n"
            "- IBAMA — Autos de infração ambiental"
        )
    with col_metodo:
        st.markdown("**Metodologia**")
        st.markdown(
            "- LightGBM + Optuna (55 trials)\n"
            "- Validação espacial por UF (GroupKFold)\n"
            "- HDBSCAN — 5 clusters de municípios\n"
            "- SHAP — interpretabilidade do modelo\n"
            "- Score histórico: ranking percentílico por ano (dados reais INPE)\n"
            "- Score 2026: ranking percentílico da previsão LightGBM"
        )

    if metrics:
        st.subheader("Desempenho do Modelo (Spatial CV — média dos 5 folds)")
        m = metrics.get("mean") or metrics.get("MEDIA") or next(iter(metrics.values()), {})
        cols_m = st.columns(4)
        cols_m[0].metric("R²",    f"{m.get('r2',    0):.4f}", help="Variância explicada (1 = perfeito)")
        cols_m[1].metric("RMSE",  f"{m.get('rmse',  0):.4f}", help="Erro quadrático médio (%/ano)")
        cols_m[2].metric("MAE",   f"{m.get('mae',   0):.4f}", help="Erro absoluto médio (%/ano)")
        cols_m[3].metric("WMAPE", f"{m.get('wmape', 0):.1f}%",
                         help="Erro % ponderado: Σ|erro|/Σy_true — pondera municípios de alta taxa")
        st.markdown(
            "<div style='margin-top:10px;padding:10px 14px;background:#fffbeb;border:1px solid #fde68a;"
            "border-radius:6px;font-size:12px;color:#78350f;line-height:1.6'>"
            "<b>Nota sobre variabilidade do WMAPE entre folds (Spatial CV):</b> "
            "O fold <b>RR+TO</b> apresenta WMAPE de ~56%, bem acima dos demais (MA: 30%, PA: 25%, "
            "MT: 32%, AC/AM/AP/RO: 29%). A causa é estrutural: esses estados têm taxa de "
            "desmatamento muito baixa (~0,038%/ano), e a fórmula "
            "<code>WMAPE = Σ|erro| / Σy_real</code> amplifica qualquer resíduo quando o denominador "
            "(volume total de desmatamento) é pequeno. Para estados de alta pressão — onde o modelo "
            "é operacionalmente mais relevante — o desempenho converge entre 25%–32%."
            "</div>",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Dados: INPE/PRODES · IBGE · ICMBio · IBAMA  |  "
    "Modelo: LightGBM + Optuna (spatial CV por UF)  |  "
    "Score: ranking percentílico por ano (0–100) — 2026 baseado em previsão do modelo"
)
