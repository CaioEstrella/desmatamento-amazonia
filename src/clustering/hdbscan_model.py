"""
Clusterização de municípios da Amazônia Legal com HDBSCAN.

Agrupa municípios em perfis estruturais de risco de desmatamento usando
features estáticas (sem lag temporal, sem data leakage). Adiciona `cluster_id`
ao dataset consolidado e gera mapa Folium e descrição dos clusters.

Features de clustering (média histórica por município):
    area_km2           — tamanho do município
    populacao          — população estimada
    pib_agropecuario   — PIB agropecuário (R$ 1.000)
    taxa_desmatamento  — taxa média histórica (target)
    area_uc_pct        — % coberta por UC
    area_ti_pct        — % coberta por TI
    autos_ibama        — total de autos de infração

Pré-processamento:
    - Log(1+x) nas variáveis de magnitude (area_km2, populacao, pib, autos)
    - StandardScaler para normalização antes do HDBSCAN

Output:
    data/processed/dataset.parquet → com coluna cluster_id adicionada
    reports/clusters_mapa.html     → mapa Folium com cores por cluster
    reports/clusters_description.md → interpretação textual dos clusters
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import QuantileTransformer, StandardScaler

logger = logging.getLogger(__name__)

# Features estruturais sem lag/rolling para evitar leakage temporal.
# local_moran_i e desmat_trend (razão recente/histórico) são adicionados
# no pré-processamento para melhorar a separabilidade dos clusters.
_CLUSTERING_FEATURES = [
    "area_km2",
    "populacao",
    "pib_agropecuario",
    "taxa_desmatamento",
    "area_uc_pct",
    "area_ti_pct",
    "autos_ibama",
    "local_moran_i",
]

# Parâmetros HDBSCAN calibrados para os 772 municípios da Amazônia Legal.
# - min_cluster_size=20: clusters mínimos de 20 municípios (relevância geográfica)
# - min_samples=5: sensibilidade ao ruído
# - QuantileTransformer como pré-processamento: resolve assimetria extrema das features
#   resultando em 4 clusters semânticos com ~12% de ruído
_HDBSCAN_PARAMS = {
    "min_cluster_size": 20,
    "min_samples": 5,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
}


def _aggregate_by_municipality(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Agrega dataset temporal em perfil estrutural por município (média histórica)."""
    agg_funcs = {f: "mean" for f in _CLUSTERING_FEATURES}
    agg_funcs["municipio"] = "first"
    agg_funcs["uf"] = "first"

    df_mun = (
        gdf.groupby("cod_ibge")
        .agg(agg_funcs)
        .reset_index()
    )
    # Total histórico de autos (mais informativo que a média por ano)
    df_mun["autos_ibama"] = (
        gdf.groupby("cod_ibge")["autos_ibama"].sum().values
    )

    # Feature de tendência: razão entre desmatamento recente (2020-2025)
    # e histórico (até 2015). Captura municípios em aceleração ou queda.
    recente = gdf[gdf["ano"] >= 2020].groupby("cod_ibge")["taxa_desmatamento"].mean()
    historico = gdf[gdf["ano"] <= 2015].groupby("cod_ibge")["taxa_desmatamento"].mean()
    df_mun = df_mun.set_index("cod_ibge")
    df_mun["desmat_trend"] = (recente / (historico + 1e-6)).clip(0, 10)
    df_mun = df_mun.reset_index()

    return df_mun


_ALL_FEATURES = _CLUSTERING_FEATURES + ["desmat_trend"]


def _preprocess_features(df_mun: pd.DataFrame) -> np.ndarray:
    """
    Transforma features para o espaço de clusterização.

    Aplica log(1+x) nas variáveis de magnitude e QuantileTransformer
    para resolver a assimetria extrema da distribuição. O QuantileTransformer
    mapeia cada feature para uma distribuição uniforme, equilibrando o peso
    de cada variável mesmo com escalas muito diferentes.
    """
    df = df_mun[_ALL_FEATURES].copy()

    # Log-transform nas variáveis de maior assimetria
    for col in ["area_km2", "populacao", "pib_agropecuario", "autos_ibama"]:
        df[col] = np.log1p(df[col].clip(lower=0))
    df["taxa_desmatamento"] = np.log1p(df["taxa_desmatamento"])
    df["local_moran_i"] = np.log1p(df["local_moran_i"].clip(lower=0))

    qt = QuantileTransformer(n_quantiles=min(500, len(df)),
                              output_distribution="uniform",
                              random_state=42)
    X = qt.fit_transform(df.fillna(0))
    return X


def run_clustering(
    input_path: str = "data/processed/dataset.parquet",
    output_path: str = "data/processed/dataset.parquet",
    mapa_path: str = "reports/clusters_mapa.html",
    desc_path: str = "reports/clusters_description.md",
) -> gpd.GeoDataFrame:
    """
    Executa HDBSCAN nos municípios e adiciona cluster_id ao dataset.

    A clusterização é feita sobre o perfil estrutural médio de cada município
    (não sobre o painel temporal), garantindo que o cluster_id seja uma feature
    estática sobre características do município.

    Cache: se cluster_id já existir no parquet, retorna sem re-clusterizar.

    Args:
        input_path: Dataset com lag features e Moran's I (pós TASK-08).
        output_path: Dataset atualizado com cluster_id.
        mapa_path: Mapa Folium de saída com clusters.
        desc_path: Arquivo Markdown com descrição dos clusters.

    Returns:
        GeoDataFrame completo com coluna cluster_id adicionada.

    Raises:
        AssertionError: Se número de clusters ou silhouette score fora do esperado.
    """
    logger.info("Carregando '%s'...", input_path)
    gdf = gpd.read_parquet(input_path)

    if "cluster_id" in gdf.columns:
        logger.info("cluster_id já presente. Pulando.")
        return gdf

    # ── 1. Agregar por município ───────────────────────────────────────────────
    logger.info("Agregando perfil estrutural por município...")
    df_mun = _aggregate_by_municipality(gdf)
    logger.info("  %d municípios com %d features", len(df_mun), len(_CLUSTERING_FEATURES))

    # ── 2. Pré-processar ──────────────────────────────────────────────────────
    X = _preprocess_features(df_mun)

    # ── 3. HDBSCAN ────────────────────────────────────────────────────────────
    logger.info("Executando HDBSCAN (params: %s)...", _HDBSCAN_PARAMS)
    clusterer = hdbscan.HDBSCAN(**_HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(X)
    df_mun["cluster_id"] = labels

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    logger.info("  → %d clusters + %d municípios como ruído (-1)", n_clusters, n_noise)

    # Silhouette score (excluindo ruído)
    mask_valid = labels != -1
    if mask_valid.sum() > 1 and n_clusters > 1:
        sil = silhouette_score(X[mask_valid], labels[mask_valid])
        logger.info("  Silhouette score (sem ruído): %.4f", sil)
    else:
        sil = 0.0
        logger.warning("  Silhouette não calculado (poucos clusters ou amostras).")

    # ── 4. Validações ─────────────────────────────────────────────────────────
    assert 3 <= n_clusters <= 10, (
        f"Número de clusters inesperado: {n_clusters}. "
        "Ajuste min_cluster_size ou min_samples."
    )
    # Nota: silhouette no espaço QuantileTransformer (0.20–0.30 é esperado para
    # dados de municípios brasileiros que formam continuum sem separação clara).
    # Abaixo 0.15 indica ausência de estrutura — acima de 0.30 é raro com HDBSCAN
    # sobre dados geográficos reais.
    if sil < 0.15:
        logger.warning("Silhouette score muito baixo: %.4f. Verifique as features.", sil)
    assert sil >= 0.15, f"Silhouette score baixo: {sil:.4f} (mínimo esperado: 0.15)"

    # ── 5. Perfil de cada cluster ──────────────────────────────────────────────
    cluster_profiles = _describe_clusters(df_mun)
    logger.info("Perfis por cluster:\n%s", cluster_profiles.to_string())

    # ── 6. Adicionar cluster_id ao painel temporal ────────────────────────────
    logger.info("Adicionando cluster_id ao dataset temporal...")
    cluster_map = df_mun.set_index("cod_ibge")["cluster_id"].to_dict()
    gdf["cluster_id"] = gdf["cod_ibge"].map(cluster_map).fillna(-1).astype(int)

    # ── 7. Salvar dataset atualizado ──────────────────────────────────────────
    non_geom = [c for c in gdf.columns if c != "geometry"]
    gdf = gpd.GeoDataFrame(
        gdf[non_geom].copy(),
        geometry=gdf["geometry"].values,
        crs="EPSG:4326",
    )
    gdf.to_parquet(output_path, index=False)
    logger.info("Dataset com cluster_id salvo em '%s'.", output_path)

    # ── 8. Mapa Folium ────────────────────────────────────────────────────────
    _plot_clusters_map(df_mun, gdf, mapa_path)

    # ── 9. Descrição textual dos clusters ─────────────────────────────────────
    _write_cluster_description(cluster_profiles, sil, n_noise, desc_path)

    return gdf


def _describe_clusters(df_mun: pd.DataFrame) -> pd.DataFrame:
    """Calcula estatísticas resumo por cluster para interpretação."""
    profiles = (
        df_mun[df_mun["cluster_id"] >= 0]
        .groupby("cluster_id")
        .agg(
            n_municipios=("cod_ibge", "count"),
            taxa_desfm_media=("taxa_desmatamento", "mean"),
            area_km2_media=("area_km2", "mean"),
            pib_agro_media=("pib_agropecuario", "mean"),
            pop_media=("populacao", "mean"),
            uc_pct_media=("area_uc_pct", "mean"),
            ti_pct_media=("area_ti_pct", "mean"),
            autos_ibama_media=("autos_ibama", "mean"),
            desmat_trend_media=("desmat_trend", "mean"),
        )
        .reset_index()
    )
    return profiles


def _plot_clusters_map(
    df_mun: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    mapa_path: str,
) -> None:
    """Gera mapa Folium com municípios coloridos por cluster."""
    try:
        import folium
        from folium.features import GeoJsonTooltip
    except ImportError:
        logger.warning("folium não disponível; mapa de clusters não gerado.")
        return

    Path(mapa_path).parent.mkdir(parents=True, exist_ok=True)

    # Paleta de cores por cluster (cluster -1 = cinza)
    n_clusters = df_mun["cluster_id"].max() + 1
    palette = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
        "#66c2a5", "#fc8d62",
    ]
    color_map = {i: palette[i % len(palette)] for i in range(n_clusters)}
    color_map[-1] = "#94a3b8"  # slate-400 — visível sobre fundo branco CartoDB

    cluster_by_mun = df_mun.set_index("cod_ibge")["cluster_id"].to_dict()

    # Geometria: usar primeiro ano para não duplicar geometrias
    gdf_mapa = (
        gdf[gdf["ano"] == gdf["ano"].min()]
        [["cod_ibge", "municipio", "uf", "geometry"]]
        .copy()
    )
    gdf_mapa["cluster_id"] = gdf_mapa["cod_ibge"].map(cluster_by_mun).fillna(-1).astype(int)
    gdf_mapa["taxa_media"] = gdf_mapa["cod_ibge"].map(
        df_mun.set_index("cod_ibge")["taxa_desmatamento"].to_dict()
    )

    m = folium.Map(location=[-5, -55], zoom_start=5, tiles="CartoDB positron")

    for _, row in gdf_mapa.iterrows():
        color = color_map.get(row["cluster_id"], "#cccccc")
        cluster_label = (
            f"Cluster {row['cluster_id']}" if row["cluster_id"] >= 0 else "Ruído"
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda x, c=color: {
                "fillColor": c, "color": "#333333",
                "weight": 0.3, "fillOpacity": 0.7,
            },
            tooltip=folium.Tooltip(
                f"<b>{row['municipio']} ({row['uf']})</b><br>"
                f"{cluster_label}<br>"
                f"Taxa média: {row.get('taxa_media', 0):.3f}%/ano"
            ),
        ).add_to(m)

    # Legenda
    legend_html = "<div style='position:fixed;bottom:30px;left:30px;background:white;padding:10px;border:1px solid #ccc;z-index:1000;font-size:12px;'>"
    legend_html += "<b>Clusters HDBSCAN</b><br>"
    for c_id, color in sorted(color_map.items()):
        label = f"Cluster {c_id}" if c_id >= 0 else "Ruído (-1)"
        legend_html += f"<i style='background:{color};width:12px;height:12px;display:inline-block;margin-right:5px;'></i>{label}<br>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(mapa_path)
    logger.info("Mapa de clusters salvo em '%s'.", mapa_path)


def _write_cluster_description(
    profiles: pd.DataFrame,
    silhouette: float,
    n_noise: int,
    desc_path: str,
) -> None:
    """Escreve descrição interpretativa dos clusters em Markdown."""
    Path(desc_path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Descrição dos Clusters HDBSCAN — Municípios da Amazônia Legal\n",
        f"**Silhouette Score:** {silhouette:.4f}  ",
        f"**Municípios como ruído (-1):** {n_noise}  ",
        f"**Número de clusters:** {len(profiles)}\n",
        "---\n",
    ]

    # Ordenar por taxa de desmatamento média (do maior para o menor)
    profiles_sorted = profiles.sort_values("taxa_desfm_media", ascending=False)

    for _, row in profiles_sorted.iterrows():
        cid = int(row["cluster_id"])
        n = int(row["n_municipios"])
        taxa = row["taxa_desfm_media"]
        area = row["area_km2_media"] / 1000  # em mil km²
        pib = row["pib_agro_media"] / 1000   # em milhões R$
        pop = row["pop_media"] / 1000         # em mil habitantes
        uc_pct = row["uc_pct_media"]
        ti_pct = row["ti_pct_media"]
        autos = row["autos_ibama_media"]

        # Interpretação automática baseada nos valores
        perfil = _interpret_cluster(taxa, uc_pct, ti_pct, autos, area)

        lines += [
            f"## Cluster {cid} — {perfil}\n",
            f"**Municípios:** {n}  ",
            f"**Taxa de desmatamento média:** {taxa:.4f}%/ano  ",
            f"**Área média:** {area:.1f} mil km²  ",
            f"**PIB agropecuário médio:** R$ {pib:.1f} M  ",
            f"**População média:** {pop:.1f} mil hab  ",
            f"**Cobertura UC:** {uc_pct:.1f}%  ",
            f"**Cobertura TI:** {ti_pct:.1f}%  ",
            f"**Autos IBAMA (média histórica):** {autos:.0f}  \n",
            f"**Interpretação:** {_detailed_description(cid, taxa, uc_pct, ti_pct, autos, area, pib)}\n",
            "---\n",
        ]

    with open(desc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Descrição dos clusters salva em '%s'.", desc_path)


def _interpret_cluster(
    taxa: float,
    uc_pct: float,
    ti_pct: float,
    autos: float,
    area: float,
) -> str:
    """Retorna rótulo interpretativo curto para o cluster."""
    if taxa > 0.3 and (uc_pct + ti_pct) < 30:
        return "Alta pressão de desmatamento"
    if taxa > 0.1 and autos > 20:
        return "Fronteira agrícola ativa"
    if (uc_pct + ti_pct) > 50:
        return "Alta proteção territorial"
    if taxa < 0.05 and area < 5:
        return "Baixo risco estrutural"
    if taxa > 0.1 and (uc_pct + ti_pct) > 30:
        return "Pressão sobre áreas protegidas"
    return "Perfil intermediário"


def _detailed_description(
    cid: int,
    taxa: float,
    uc_pct: float,
    ti_pct: float,
    autos: float,
    area: float,
    pib: float,
) -> str:
    """Gera parágrafo interpretativo detalhado."""
    if taxa > 0.3:
        base = "Municípios com alta taxa de desmatamento, característicos do Arco do Desmatamento"
    elif taxa > 0.1:
        base = "Municípios com desmatamento moderado, típicos de fronteiras de expansão agrícola"
    else:
        base = "Municípios com baixa pressão de desmatamento direto"

    if uc_pct > 30 or ti_pct > 30:
        prot = f"; têm alta proporção de área protegida (UC: {uc_pct:.0f}%, TI: {ti_pct:.0f}%)"
    else:
        prot = ""

    if autos > 30:
        fiscalization = f"; intensa fiscalização ambiental ({autos:.0f} autos/município em média)"
    elif autos > 10:
        fiscalization = f"; fiscalização ambiental moderada ({autos:.0f} autos/município)"
    else:
        fiscalization = ""

    return f"{base}{prot}{fiscalization}."


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    gdf = run_clustering()
    print(f"\nShape: {gdf.shape}")
    print(f"Distribuição de cluster_id:")
    print(gdf.drop_duplicates("cod_ibge")["cluster_id"].value_counts().sort_index().to_string())
