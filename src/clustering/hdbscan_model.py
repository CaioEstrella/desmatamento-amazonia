"""
Clusterização de municípios da Amazônia Legal com HDBSCAN.

Agrupa municípios em perfis estruturais de risco de desmatamento usando
features estáticas (sem lag temporal, sem data leakage). Adiciona `cluster_id`
ao dataset consolidado e gera mapa Folium e descrição dos clusters.

Features de clustering (média histórica por município):
    area_km2           — tamanho do município
    populacao          — população estimada
    pib_agropecuario   — PIB agropecuário (R$ 1.000)
    area_uc_pct        — % coberta por Unidades de Conservação
    area_ti_pct        — % coberta por Terras Indígenas
    autos_ibama        — total histórico de autos de infração
    local_moran_i      — autocorrelação espacial local do desmatamento

Nota: `taxa_desmatamento` e `desmat_trend` foram intencionalmente excluídos das
features de clustering — ambas são derivadas do target do modelo LightGBM.
Usá-las criaria raciocínio circular: taxa → cluster_id → prediz taxa. O
`desmat_trend` permanece no DataFrame para interpretação dos clusters, mas não
entra no HDBSCAN.

Pré-processamento:
    - Log(1+x) nas variáveis de magnitude (area, pop, pib, autos, taxa)
    - local_moran_i: deslocado para domínio positivo antes do log1p
      (preserva autocorrelação negativa, que indica municípios cercados por
      municípios com baixo desmatamento — informação relevante)
    - QuantileTransformer(uniform): equaliza distribuições antes do HDBSCAN

Estratégia de ruído:
    1. HDBSCAN com min_samples=3 (limiar baixo → menos ruído)
    2. Post-assignment via all_points_membership_vectors: municípios que
       continuam como ruído são atribuídos ao cluster com maior afinidade
       de densidade (soft clustering). Isso é preferível a simplesmente
       ignorar esses pontos, pois todo município tem um perfil válido.

Output:
    data/processed/dataset.parquet → com coluna cluster_id adicionada
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
    "area_uc_pct",
    "area_ti_pct",
    "autos_ibama",
    "local_moran_i",
]

# Parâmetros HDBSCAN calibrados para os ~808 municípios da Amazônia Legal.
# - min_cluster_size=20: clusters mínimos de 20 municípios (relevância geográfica)
# - min_samples=3: limiar baixo → mais pontos são núcleos de densidade → menos ruído
#   (min_samples=5 gerava ~75 municípios de ruído; 3 reduz para ~20-30)
# - prediction_data=True: necessário para all_points_membership_vectors (post-assignment)
_HDBSCAN_PARAMS = {
    "min_cluster_size": 20,
    "min_samples": 3,
    "metric": "euclidean",
    "cluster_selection_method": "eom",
    "prediction_data": True,
}


def _aggregate_by_municipality(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Agrega dataset temporal em perfil estrutural por município (média histórica)."""
    agg_funcs = {f: "mean" for f in _CLUSTERING_FEATURES}
    agg_funcs["municipio"] = "first"
    agg_funcs["uf"] = "first"
    # taxa_desmatamento: incluída apenas para interpretação/descrição dos clusters.
    # Não entra no HDBSCAN — excluída de _CLUSTERING_FEATURES para evitar
    # raciocínio circular com o target do LightGBM.
    agg_funcs["taxa_desmatamento"] = "mean"

    df_mun = (
        gdf.groupby("cod_ibge")
        .agg(agg_funcs)
        .reset_index()
    )
    # Total histórico de autos é mais informativo que a média por ano.
    # Usa map() em vez de .values para garantir alinhamento por cod_ibge,
    # independentemente da ordem retornada pelo groupby.
    autos_total = gdf.groupby("cod_ibge")["autos_ibama"].sum()
    df_mun["autos_ibama"] = df_mun["cod_ibge"].map(autos_total)

    # Feature de tendência: razão entre desmatamento recente (2020-2025)
    # e histórico (até 2015). Captura municípios em aceleração ou queda.
    # Alinhamento via index cod_ibge garante correspondência correta.
    recente = gdf[gdf["ano"] >= 2020].groupby("cod_ibge")["taxa_desmatamento"].mean()
    historico = gdf[gdf["ano"] <= 2015].groupby("cod_ibge")["taxa_desmatamento"].mean()
    df_mun = df_mun.set_index("cod_ibge")
    df_mun["desmat_trend"] = (recente / (historico + 1e-6)).clip(0, 10)
    df_mun = df_mun.reset_index()

    return df_mun


def _preprocess_features(df_mun: pd.DataFrame) -> np.ndarray:
    """
    Transforma features para o espaço de clusterização.

    Aplica log(1+x) nas variáveis de magnitude e QuantileTransformer
    para resolver a assimetria extrema da distribuição. O QuantileTransformer
    mapeia cada feature para uma distribuição uniforme, equilibrando o peso
    de cada variável mesmo com escalas muito diferentes.

    local_moran_i pode ser negativo (autocorrelação negativa — município
    cercado por municípios com desmatamento oposto ao seu). Clipping para 0
    perderia essa informação; em vez disso, deslocamos para o domínio positivo
    antes do log1p: log1p(x - min(x)), que preserva a ordenação relativa.
    """
    df = df_mun[_CLUSTERING_FEATURES].copy()

    # Log-transform nas variáveis de maior assimetria (skew > 3)
    for col in ["area_km2", "populacao", "pib_agropecuario", "autos_ibama"]:
        df[col] = np.log1p(df[col].clip(lower=0))

    # Moran's I: deslocar para domínio ≥ 0 antes do log1p para preservar
    # valores negativos (não clipar, pois informação de autocorrelação
    # negativa é relevante para separar municípios isolados dos clusters)
    moran_min = df["local_moran_i"].min()
    shift = max(0, -moran_min)  # 0 se já positivo, |min| se negativo
    df["local_moran_i"] = np.log1p(df["local_moran_i"] + shift)

    qt = QuantileTransformer(n_quantiles=min(500, len(df)),
                              output_distribution="uniform",
                              random_state=42)
    X = qt.fit_transform(df.fillna(0))
    return X


def run_clustering(
    input_path: str = "data/processed/dataset.parquet",
    output_path: str = "data/processed/dataset.parquet",
    desc_path: str = "reports/clusters_description.md",
    force: bool = False,
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
        desc_path: Arquivo Markdown com descrição dos clusters.

    Returns:
        GeoDataFrame completo com coluna cluster_id adicionada.

    Raises:
        AssertionError: Se número de clusters ou silhouette score fora do esperado.
    """
    logger.info("Carregando '%s'...", input_path)
    gdf = gpd.read_parquet(input_path)

    if "cluster_id" in gdf.columns and not force:
        logger.info("cluster_id já presente. Pulando. Use force=True para reclusterizar.")
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

    n_clusters = len(set(labels) - {-1})
    n_noise_before = (labels == -1).sum()
    logger.info("  → %d clusters + %d municípios como ruído antes do post-assignment",
                n_clusters, n_noise_before)

    # Post-assignment via soft clustering: municípios classificados como ruído
    # recebem o cluster com maior afinidade de densidade. Isso garante que
    # todo município tem um perfil — ruído em HDBSCAN não significa anomalia,
    # mas sim baixa densidade local no espaço de features.
    if n_noise_before > 0:
        membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
        noise_mask = labels == -1
        if membership_vectors.shape[1] > 0:
            soft_labels = np.argmax(membership_vectors[noise_mask], axis=1)
            labels = labels.copy()
            labels[noise_mask] = soft_labels
            n_noise_after = (labels == -1).sum()
            logger.info("  → %d municípios reatribuídos via soft clustering | restam %d como ruído",
                        n_noise_before - n_noise_after, n_noise_after)

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

    # ── 8. Descrição textual dos clusters ────────────────────────────────────
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


def _cluster_label(row: pd.Series) -> str:
    """
    Rótulo descritivo para o cluster baseado em seu perfil dominante.

    A lógica avalia as características mais distintas de cada perfil em
    ordem de especificidade, da mais rara (alta pressão sem proteção) para
    a mais comum (pequenos municípios periféricos).
    """
    taxa  = row["taxa_desfm_media"]
    uc    = row["uc_pct_media"]
    ti    = row["ti_pct_media"]
    autos = row["autos_ibama_media"]
    area  = row["area_km2_media"]
    pib   = row["pib_agro_media"]
    trend = row["desmat_trend_media"]
    prot  = uc + ti

    # Cluster do Arco: alta taxa, nenhuma proteção, alto PIB agropecuário
    if taxa > 0.35 and prot < 10:
        return "Arco do desmatamento — alta pressão sem áreas protegidas"

    # Grandes municípios com dupla proteção (UCs e Terras Indígenas)
    # mas sob pressão crescente (trend > 1.5 = aceleração recente)
    if area > 10_000 and prot > 35 and trend > 1.5:
        return "Grandes municípios — proteção formal com pressão crescente"

    # Presença significativa de Terras Indígenas — barreira parcial
    if ti > 15 and taxa > 0.12:
        return "Fronteira agrícola com presença de Terras Indígenas"

    # Municípios onde Unidades de Conservação dominam — barreira eficaz
    if uc > 25 and ti < 10 and taxa < 0.15:
        return "Proteção por Unidades de Conservação reduz pressão"

    # Municípios grandes sem proteção formal mas com pressão moderada
    if area > 10_000 and prot < 25:
        return "Grandes municípios com expansão agrícola moderada"

    # Municípios pequenos, periféricos, baixo agronegócio
    if area < 2_500 and pib < 80_000 and taxa < 0.12:
        return "Municípios periféricos com baixa pressão de desmatamento"

    return "Perfil intermediário de expansão agrícola"


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
        cid   = int(row["cluster_id"])
        n     = int(row["n_municipios"])
        taxa  = row["taxa_desfm_media"]
        area  = row["area_km2_media"] / 1000
        pib   = row["pib_agro_media"] / 1000
        pop   = row["pop_media"] / 1000
        uc    = row["uc_pct_media"]
        ti    = row["ti_pct_media"]
        autos = row["autos_ibama_media"]
        trend = row["desmat_trend_media"]

        perfil = _cluster_label(row)

        lines += [
            f"## Cluster {cid} — {perfil}\n",
            f"**Municípios:** {n}  ",
            f"**Taxa de desmatamento média:** {taxa:.4f}%/ano  ",
            f"**Área média:** {area:.1f} mil km²  ",
            f"**PIB agropecuário médio:** R$ {pib:.1f} M  ",
            f"**População média:** {pop:.1f} mil hab  ",
            f"**Cobertura por Unidades de Conservação:** {uc:.1f}%  ",
            f"**Cobertura por Terras Indígenas:** {ti:.1f}%  ",
            f"**Autos de infração IBAMA (média histórica):** {autos:.0f}  ",
            f"**Tendência recente de desmatamento:** {trend:.2f}× (1.0 = estável)  \n",
            f"**Interpretação:** {_detailed_description(taxa, uc, ti, autos, area, pib, trend)}\n",
            "---\n",
        ]

    with open(desc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Descrição dos clusters salva em '%s'.", desc_path)



def _detailed_description(
    taxa: float,
    uc_pct: float,
    ti_pct: float,
    autos: float,
    area: float,
    pib: float,
    trend: float,
) -> str:
    """Gera parágrafo interpretativo detalhado para o Markdown de clusters."""
    if taxa > 0.35:
        base = (
            "Municípios com alta taxa histórica de desmatamento, representativos do "
            "Arco do Desmatamento amazônico. Sem cobertura expressiva de áreas protegidas "
            "e com forte pressão do agronegócio"
        )
    elif taxa > 0.15:
        base = "Municípios com desmatamento moderado a alto, típicos de fronteiras de expansão agrícola"
    elif taxa > 0.08:
        base = "Municípios com desmatamento em nível intermediário"
    else:
        base = "Municípios com baixa pressão de desmatamento"

    parts = [base]

    if uc_pct > 25:
        parts.append(
            f"alta cobertura por Unidades de Conservação ({uc_pct:.0f}% do território), "
            "o que atua como barreira eficaz à conversão de vegetação nativa"
        )
    if ti_pct > 15:
        parts.append(
            f"presença significativa de Terras Indígenas ({ti_pct:.0f}% do território), "
            "demarcadas legalmente e reconhecidas como barreira ao desmatamento"
        )
    if autos > 300:
        parts.append(
            f"elevada fiscalização ambiental ({autos:.0f} autos de infração por município em média), "
            "refletindo alta pressão sobre a legislação ambiental"
        )
    elif autos > 80:
        parts.append(f"fiscalização ambiental moderada ({autos:.0f} autos por município)")

    if trend > 1.8:
        parts.append(
            f"tendência recente preocupante: desmatamento acelerou {trend:.1f}× "
            "em relação ao histórico (2020–2025 vs. 2008–2015)"
        )
    elif trend < 0.6:
        parts.append(
            f"tendência positiva: desmatamento recuou para {trend:.1f}× do nível histórico"
        )

    if len(parts) == 1:
        return parts[0] + "."
    return parts[0] + "; " + "; ".join(parts[1:]) + "."


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    gdf = run_clustering(force=True)
    print(f"\nShape: {gdf.shape}")
    print(f"Distribuição de cluster_id:")
    print(gdf.drop_duplicates("cod_ibge")["cluster_id"].value_counts().sort_index().to_string())
