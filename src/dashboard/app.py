"""
Dashboard Streamlit — Preditor de Risco de Desmatamento (Amazônia Legal) · TASK-15.

Execução:
    streamlit run src/dashboard/app.py

Seções:
    1. Sidebar   — filtros de ano, UF e cluster
    2. KPIs      — métricas globais do ano selecionado
    3. Mapa      — coroplético de score_risco (Folium embarcado via st.components)
    4. Ranking   — top-20 municípios de maior risco com SHAP values
    5. Série     — evolução histórica de taxa_desmatamento por estado
    6. Correlações — heatmap de features vs taxa_desmatamento
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
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

_PREDICTIONS_PATH = "data/outputs/predictions.parquet"
_METRICS_PATH     = "data/outputs/metrics_by_fold.json"
_SHAP_BEESWARM    = "reports/shap_beeswarm.png"
_SHAP_CLUSTER     = "reports/shap_por_cluster.png"


@st.cache_data(show_spinner="Carregando dados...")
def load_data() -> gpd.GeoDataFrame:
    return gpd.read_parquet(_PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    p = Path(_METRICS_PATH)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ── Carregar dados ────────────────────────────────────────────────────────────
gdf = load_data()
metrics = load_metrics()

anos        = sorted(gdf["ano"].unique())
ufs         = sorted(gdf["uf"].unique())
clusters    = sorted(gdf["cluster_id"].dropna().unique()) if "cluster_id" in gdf.columns else []

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🌿 Filtros")
ano_sel = st.sidebar.selectbox("Ano", anos, index=len(anos) - 1)
uf_sel  = st.sidebar.multiselect("Estado (UF)", ufs, default=ufs)
cluster_labels = {int(c): f"Cluster {int(c)}" if int(c) >= 0 else "Ruído" for c in clusters}
cluster_sel = st.sidebar.multiselect(
    "Cluster HDBSCAN",
    options=list(cluster_labels.keys()),
    default=list(cluster_labels.keys()),
    format_func=lambda c: cluster_labels[c],
) if clusters else None

st.sidebar.markdown("---")
st.sidebar.markdown("**Métricas do modelo (CV espacial)**")
if metrics:
    media = metrics.get("MEDIA", {})
    st.sidebar.metric("RMSE médio", f"{media.get('rmse', 0):.4f}")
    st.sidebar.metric("R² médio",   f"{media.get('r2', 0):.4f}")
    st.sidebar.metric("MAE médio",  f"{media.get('mae', 0):.4f}")

# ── Filtro de dados ───────────────────────────────────────────────────────────
mask = (gdf["ano"] == ano_sel) & (gdf["uf"].isin(uf_sel))
if cluster_sel is not None and "cluster_id" in gdf.columns:
    mask &= gdf["cluster_id"].isin(cluster_sel)

gdf_view = gdf[mask].copy()
has_score = gdf_view["score_risco"].notna().any()

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🌿 Preditor de Risco de Desmatamento — Amazônia Legal")
st.markdown(
    f"**Ano:** {ano_sel}  ·  **Estados:** {', '.join(uf_sel)}  ·  "
    f"**Municípios:** {len(gdf_view):,}"
)

# ── KPIs ──────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_desmat = gdf_view["desmatamento_km2"].sum()
    st.metric("Desmatamento total", f"{total_desmat:,.0f} km²")

with col2:
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
tab_mapa, tab_ranking, tab_serie, tab_shap = st.tabs(
    ["🗺️ Mapa de risco", "🏆 Ranking", "📈 Série temporal", "🔍 SHAP"]
)

# ── Tab 1: Mapa Folium ────────────────────────────────────────────────────────
with tab_mapa:
    st.subheader(f"Score de risco de desmatamento — {ano_sel}")

    if not has_score:
        st.info(f"Score não disponível para {ano_sel} (lag features exigem ano ≥ 2011).")
    else:
        gdf_map = gdf_view[gdf_view["score_risco"].notna()].copy()

        m = folium.Map(
            location=[-6.0, -55.0],
            zoom_start=5,
            tiles="CartoDB positron",
        )

        choropleth = folium.Choropleth(
            geo_data=gdf_map.__geo_interface__,
            data=gdf_map[["cod_ibge", "score_risco"]],
            columns=["cod_ibge", "score_risco"],
            key_on="feature.properties.cod_ibge",
            fill_color="RdYlGn_r",
            fill_opacity=0.75,
            line_opacity=0.2,
            legend_name="Score de risco (0–100)",
            nan_fill_color="lightgray",
        ).add_to(m)

        # Tooltip interativo
        tooltip_cols = ["municipio", "uf", "score_risco", "taxa_desmatamento", "desmatamento_km2"]
        tooltip_cols = [c for c in tooltip_cols if c in gdf_map.columns]
        folium.GeoJson(
            gdf_map[tooltip_cols + ["geometry"]],
            style_function=lambda f: {
                "fillOpacity": 0,
                "weight": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_cols,
                aliases=[
                    "Município", "UF", "Score de risco",
                    "Taxa desmat. (%/ano)", "Desmat. (km²)",
                ],
                localize=True,
            ),
        ).add_to(m)

        html_str = m._repr_html_()
        components.html(html_str, height=520, scrolling=False)

# ── Tab 2: Ranking ────────────────────────────────────────────────────────────
with tab_ranking:
    st.subheader(f"Top-20 municípios por score de risco — {ano_sel}")

    if not has_score:
        st.info(f"Score não disponível para {ano_sel}.")
    else:
        cols_show = [
            "municipio", "uf", "score_risco", "taxa_desmatamento",
            "desmatamento_km2", "cluster_id",
        ]
        cols_show = [c for c in cols_show if c in gdf_view.columns]
        top20 = (
            gdf_view[gdf_view["score_risco"].notna()]
            .nlargest(20, "score_risco")[cols_show]
            .reset_index(drop=True)
        )
        top20.index += 1

        rename = {
            "municipio": "Município", "uf": "UF",
            "score_risco": "Score (0–100)", "taxa_desmatamento": "Taxa (%/ano)",
            "desmatamento_km2": "Desmat. (km²)", "cluster_id": "Cluster",
        }
        st.dataframe(
            top20.rename(columns=rename).style
            .background_gradient(subset=["Score (0–100)"], cmap="RdYlGn_r")
            .format({"Score (0–100)": "{:.1f}", "Taxa (%/ano)": "{:.4f}", "Desmat. (km²)": "{:.2f}"}),
            use_container_width=True,
        )

# ── Tab 3: Série temporal ─────────────────────────────────────────────────────
with tab_serie:
    st.subheader("Evolução da taxa de desmatamento por estado")

    import plotly.express as px

    df_serie = gdf[gdf["uf"].isin(uf_sel)].copy()
    df_uf_ano = (
        df_serie.groupby(["uf", "ano"])
        .agg(taxa_media=("taxa_desmatamento", "mean"),
             desmat_total=("desmatamento_km2", "sum"))
        .reset_index()
    )

    fig = px.line(
        df_uf_ano,
        x="ano",
        y="taxa_media",
        color="uf",
        markers=True,
        title="Taxa de desmatamento média por estado (% /ano)",
        labels={"taxa_media": "Taxa média (%/ano)", "ano": "Ano", "uf": "UF"},
    )
    fig.update_layout(height=420, legend_title="Estado")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Desmatamento total acumulado por estado (km²)")
    fig2 = px.area(
        df_uf_ano,
        x="ano",
        y="desmat_total",
        color="uf",
        title="Desmatamento acumulado (km²) por estado",
        labels={"desmat_total": "Desmatamento (km²)", "ano": "Ano", "uf": "UF"},
    )
    fig2.update_layout(height=380, legend_title="Estado")
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 4: SHAP ───────────────────────────────────────────────────────────────
with tab_shap:
    st.subheader("Importância global das features (SHAP beeswarm)")
    if Path(_SHAP_BEESWARM).exists():
        st.image(_SHAP_BEESWARM, use_container_width=True)
    else:
        st.warning("Execute `python -m src.model.evaluate` para gerar os plots SHAP.")

    st.subheader("Importância SHAP por cluster HDBSCAN")
    if Path(_SHAP_CLUSTER).exists():
        st.image(_SHAP_CLUSTER, use_container_width=True)
    else:
        st.warning("Execute `python -m src.model.evaluate` para gerar os plots SHAP.")

    st.subheader("Waterfall plots — top-5 municípios de maior risco")
    waterfall_files = sorted(Path("reports").glob("shap_waterfall_*.png"))
    if waterfall_files:
        cols = st.columns(min(len(waterfall_files), 3))
        for i, wf in enumerate(waterfall_files[:5]):
            with cols[i % 3]:
                mun_name = wf.stem.replace("shap_waterfall_", "").replace("_", " ").title()
                st.image(str(wf), caption=mun_name, use_container_width=True)
    else:
        st.warning("Execute `python -m src.model.evaluate` para gerar os waterfall plots.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Dados: INPE/PRODES · IBGE · ICMBio · IBAMA  |  "
    "Modelo: LightGBM + Optuna (spatial CV por UF)  |  "
    "Score: ranking percentílico global (0–100)"
)
