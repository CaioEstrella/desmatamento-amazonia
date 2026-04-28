"""
Geração de scores de risco de desmatamento (0–100) — TASK-14.

Estratégia:
    1. Carrega o modelo LightGBM treinado e o dataset completo.
    2. Calcula predições brutas (taxa_desmatamento prevista) para TODOS os
       pares (município, ano) com features disponíveis (ano >= 2011).
    3. Normaliza para 0–100 via ranking percentílico:
           score = percentile_rank(pred) * 100
       Isso garante distribuição uniforme e comparabilidade entre anos.
    4. Adiciona score_risco ao GeoDataFrame e salva como GeoParquet em
       data/outputs/predictions.parquet.

O ranking é calculado globalmente (todos os anos juntos), preservando a
comparabilidade inter-temporal: um score de 80 em 2015 significa o mesmo
que um score de 80 em 2023 — o município está no 80º percentil do histórico.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

_EXCLUDE_COLS = {
    "cod_ibge", "municipio", "taxa_desmatamento",
    "desmatamento_km2", "score_risco", "geometry",
}
_MODEL_PATH      = "data/outputs/lgbm_model.pkl"
_DATASET_PATH    = "data/processed/dataset.parquet"
_OUTPUT_PATH     = "data/outputs/predictions.parquet"


def generate_scores(
    model_path: str = _MODEL_PATH,
    dataset_path: str = _DATASET_PATH,
    output_path: str = _OUTPUT_PATH,
) -> gpd.GeoDataFrame:
    """
    Gera score_risco (0–100) para todos os municípios e salva predictions.parquet.

    Args:
        model_path: Caminho do modelo LightGBM serializado (.pkl).
        dataset_path: Caminho do dataset com features completas.
        output_path: Caminho do GeoParquet de saída.

    Returns:
        GeoDataFrame com coluna score_risco adicionada.
    """
    out = Path(output_path)

    # ── 1. Carregar modelo ──────────────────────────────────────────────────────
    logger.info("Carregando modelo de '%s'...", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── 2. Carregar e preparar dataset ──────────────────────────────────────────
    logger.info("Carregando dataset de '%s'...", dataset_path)
    gdf = gpd.read_parquet(dataset_path)

    # Subconjunto com features disponíveis (lag3 exige ano >= 2011)
    df_feat = gdf[gdf["ano"] >= 2011].dropna(subset=["taxa_desmat_lag3"]).copy()
    feature_cols = [c for c in df_feat.columns if c not in _EXCLUDE_COLS]

    for col in ["uf", "cluster_id"]:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].astype("category").cat.codes

    X = df_feat[feature_cols]
    logger.info("  %d linhas com features completas | %d features", len(X), len(feature_cols))

    # ── 3. Predições brutas ─────────────────────────────────────────────────────
    logger.info("Gerando predições brutas...")
    preds_raw = np.clip(model.predict(X), 0, None)
    logger.info(
        "  Predições — min: %.4f | max: %.4f | média: %.4f",
        preds_raw.min(), preds_raw.max(), preds_raw.mean(),
    )

    # ── 4. Normalização percentílica → 0–100 ────────────────────────────────────
    logger.info("Normalizando scores via ranking percentílico...")
    n = len(preds_raw)
    ranks = rankdata(preds_raw, method="average")
    scores = (ranks - 1) / (n - 1) * 100  # [0, 100] com distribuição uniforme

    df_feat = df_feat.copy()
    df_feat["score_risco"] = np.round(scores, 2)

    logger.info(
        "  score_risco — min: %.2f | max: %.2f | mediana: %.2f",
        scores.min(), scores.max(), np.median(scores),
    )

    # ── 5. Merge de volta no GeoDataFrame completo ──────────────────────────────
    logger.info("Incorporando scores ao dataset completo...")
    score_map = df_feat.set_index(["cod_ibge", "ano"])["score_risco"].to_dict()

    gdf["score_risco"] = gdf.apply(
        lambda r: score_map.get((r["cod_ibge"], r["ano"]), np.nan), axis=1
    )

    pct_sem_score = gdf["score_risco"].isna().mean() * 100
    logger.info(
        "  Municípios sem score (anos < 2011 ou sem lag): %.1f%% das linhas",
        pct_sem_score,
    )

    # ── 6. Reordenar colunas ────────────────────────────────────────────────────
    non_geom = [c for c in gdf.columns if c != "geometry"]
    gdf = gpd.GeoDataFrame(
        gdf[non_geom].copy(),
        geometry=gdf["geometry"].values,
        crs="EPSG:4326",
    )

    # ── 7. Salvar ───────────────────────────────────────────────────────────────
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(out, index=False)
    logger.info("Predictions salvas em '%s'.", output_path)

    # ── 8. Top-10 municípios de maior risco (último ano) ────────────────────────
    last_ano = gdf["ano"].max()
    top10 = (
        gdf[gdf["ano"] == last_ano]
        .nlargest(10, "score_risco")[
            ["municipio", "uf", "ano", "taxa_desmatamento", "score_risco"]
        ]
    )
    logger.info("\nTop-10 municípios de maior risco (%d):\n%s", last_ano, top10.to_string(index=False))

    return gdf


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    gdf = generate_scores()
    print(f"\nShape: {gdf.shape}")
    print(f"Linhas com score: {gdf['score_risco'].notna().sum()}")
    print(f"\nDistribuição score_risco:\n{gdf['score_risco'].describe().to_string()}")
