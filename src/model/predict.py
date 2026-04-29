"""
Geração de scores de risco de desmatamento (0–100).

Estratégia em duas etapas:

    Dados históricos (2008–2025):
        score_risco = rank percentílico da taxa_desmatamento REAL (INPE/PRODES)
        dentro de cada ano. O modelo LightGBM não é usado — o score reflete
        o que de fato ocorreu.

    Previsão 2026:
        O LightGBM prevê a taxa_desmatamento esperada para 2026 usando os dados
        de 2025 como janela de lags. O score é o rank percentílico da previsão
        entre os 808 municípios de 2026. A coluna taxa_desmatamento permanece
        NaN (ainda não medida); a estimativa fica em taxa_desmatamento_prevista.

Ranking por ano:
    Um score de 80 sempre significa "80º percentil dos municípios naquele ano",
    tornando scores comparáveis entre anos. O ranking anual preserva a pergunta
    operacional: "quais municípios são os mais preocupantes ESTE ano?"

Output (data/outputs/predictions.parquet):
    Colunas novas vs. dataset base:
        score_risco                — rank percentílico 0–100 (por ano)
        taxa_desmatamento_prevista — saída do modelo para 2026, NaN para histórico
        ano_tipo                   — "histórico" | "previsão"
    Shape: 15.352 linhas (14.544 histórico + 808 de 2026)
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
    "cod_ibge", "municipio",
    "taxa_desmatamento", "desmatamento_km2",
    "score_risco", "geometry",
    "taxa_desmatamento_prevista", "ano_tipo",
}
_MODEL_PATH   = "data/outputs/lgbm_model.pkl"
_DATASET_PATH = "data/processed/dataset.parquet"
_OUTPUT_PATH  = "data/outputs/predictions.parquet"

# Ano de previsão: dados INPE disponíveis até 2025, logo 2026 é o próximo.
_ANO_PREVISAO = 2026


def _percentile_rank_by_year(taxa: pd.Series, anos: pd.Series) -> pd.Series:
    """
    Rank percentílico da taxa dentro de cada ano (0–100, distribuição uniforme).

    Para cada ano calcula rankdata / (n-1) * 100 de forma independente.
    Municípios com taxa NaN recebem NaN.
    """
    result = pd.Series(np.nan, index=taxa.index, dtype=float)
    for ano, idx in taxa.groupby(anos).groups.items():
        vals = taxa.loc[idx]
        valid_mask = vals.notna()
        valid_idx = idx[valid_mask.values]
        if valid_idx.empty:
            continue
        v = vals.loc[valid_idx].values.astype(float)
        n = len(v)
        if n == 1:
            result.loc[valid_idx] = 50.0
        else:
            ranks = rankdata(v, method="average")
            result.loc[valid_idx] = np.round((ranks - 1) / (n - 1) * 100, 2)
    return result


def _build_2026_features(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Constrói o DataFrame de features para o ano 2026 a partir do histórico real.

    Cada município recebe uma linha com ano=2026. Os campos temporais são
    preenchidos com os valores reais dos anos anteriores (já disponíveis),
    e os campos estruturais são copiados de 2025 (proxy para o ano corrente).
    """
    # Base: linha de 2025 de cada município (estrutura + geometria + features estáticas)
    df_base = gdf[gdf["ano"] == 2025].copy()
    df_2026 = df_base.copy()
    df_2026["ano"] = _ANO_PREVISAO
    df_2026["taxa_desmatamento"] = np.nan
    df_2026["desmatamento_km2"] = np.nan
    df_2026["score_risco"] = np.nan
    df_2026["taxa_desmatamento_prevista"] = np.nan
    df_2026["ano_tipo"] = "previsão"

    mun_idx = df_2026.set_index("cod_ibge").index  # para alinhamento

    # ── Lag features ──────────────────────────────────────────────────────────
    for ano_src, col_lag in [
        (2025, "taxa_desmat_lag1"),
        (2024, "taxa_desmat_lag2"),
        (2023, "taxa_desmat_lag3"),
    ]:
        series = gdf[gdf["ano"] == ano_src].set_index("cod_ibge")["taxa_desmatamento"]
        df_2026[col_lag] = df_2026["cod_ibge"].map(series)

    # ── Rolling window ─────────────────────────────────────────────────────────
    for anos_win, col in [
        ([2023, 2024, 2025], "taxa_desmat_roll3"),
        ([2021, 2022, 2023, 2024, 2025], "taxa_desmat_roll5"),
    ]:
        series = (
            gdf[gdf["ano"].isin(anos_win)]
            .groupby("cod_ibge")["taxa_desmatamento"].mean()
        )
        df_2026[col] = df_2026["cod_ibge"].map(series)

    std_series = (
        gdf[gdf["ano"].isin([2023, 2024, 2025])]
        .groupby("cod_ibge")["taxa_desmatamento"].std()
        .fillna(0.0)
    )
    df_2026["std_desmat_roll3"] = df_2026["cod_ibge"].map(std_series)

    # Variáveis estruturais (autos_ibama, populacao, pib_agropecuario, area_uc_pct,
    # area_ti_pct, local_moran_i) são copiadas de 2025 como proxy para 2026.
    # Justificativa: dados reais de 2026 ainda não estão disponíveis (INPE/IBGE
    # publicam com defasagem de ~12 meses). A premissa de que estas variáveis variam
    # lentamente ano a ano é razoável para features estruturais. Deve ser revisada
    # quando os dados de 2026 forem publicados e o pipeline for reexecutado.

    return df_2026


def generate_scores(
    model_path: str = _MODEL_PATH,
    dataset_path: str = _DATASET_PATH,
    output_path: str = _OUTPUT_PATH,
) -> gpd.GeoDataFrame:
    """
    Gera scores de risco e salva predictions.parquet.

    Histórico (2008–2025): score baseado em dados reais INPE por ano.
    Previsão (2026): score baseado na predição LightGBM por ano.

    Returns:
        GeoDataFrame com 15.352 linhas (14.544 histórico + 808 previsão 2026).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Carregar dataset histórico ──────────────────────────────────────────
    logger.info("Carregando dataset de '%s'...", dataset_path)
    gdf = gpd.read_parquet(dataset_path)
    logger.info("  %d linhas | %d municípios | anos %d–%d",
                len(gdf), gdf["cod_ibge"].nunique(),
                gdf["ano"].min(), gdf["ano"].max())

    # ── 2. Score histórico: rank percentílico da taxa REAL por ano ─────────────
    logger.info("Calculando score histórico (dados reais, por ano)...")
    gdf["score_risco"] = _percentile_rank_by_year(gdf["taxa_desmatamento"], gdf["ano"])
    gdf["taxa_desmatamento_prevista"] = np.nan
    gdf["ano_tipo"] = "histórico"

    anos_sem_score = gdf[gdf["score_risco"].isna()]["ano"].unique()
    if len(anos_sem_score):
        logger.info("  Anos sem score (taxa NaN): %s", sorted(anos_sem_score))
    logger.info("  Score calculado para %d linhas", gdf["score_risco"].notna().sum())

    # ── 3. Carregar modelo LightGBM ────────────────────────────────────────────
    logger.info("Carregando modelo de '%s'...", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── 4. Construir features de 2026 ──────────────────────────────────────────
    logger.info("Construindo features para %d...", _ANO_PREVISAO)
    df_2026 = _build_2026_features(gdf)
    logger.info("  %d municípios prontos para previsão", len(df_2026))

    # ── 5. Prever taxa_desmatamento de 2026 ────────────────────────────────────
    logger.info("Gerando previsões para %d...", _ANO_PREVISAO)
    feature_cols = [c for c in df_2026.columns if c not in _EXCLUDE_COLS
                    and c != "geometry"]
    df_2026_feat = df_2026[feature_cols].copy()
    for col in ["uf", "cluster_id"]:
        if col in df_2026_feat.columns:
            df_2026_feat[col] = df_2026_feat[col].astype("category").cat.codes

    taxa_prevista = np.clip(model.predict(df_2026_feat), 0, None)
    logger.info("  Previsão — min: %.4f | max: %.4f | média: %.4f",
                taxa_prevista.min(), taxa_prevista.max(), taxa_prevista.mean())

    df_2026 = df_2026.copy()
    df_2026["taxa_desmatamento_prevista"] = np.round(taxa_prevista, 6)

    # ── 6. Score 2026: rank percentílico da previsão entre municípios ──────────
    logger.info("Calculando score de previsão %d (por município)...", _ANO_PREVISAO)
    df_2026["score_risco"] = _percentile_rank_by_year(
        pd.Series(taxa_prevista, index=df_2026.index),
        df_2026["ano"],
    )
    logger.info("  score_risco 2026 — min: %.2f | max: %.2f | mediana: %.2f",
                df_2026["score_risco"].min(),
                df_2026["score_risco"].max(),
                df_2026["score_risco"].median())

    # ── 7. Concatenar histórico + previsão 2026 ────────────────────────────────
    logger.info("Consolidando histórico + previsão %d...", _ANO_PREVISAO)

    # Garantir colunas alinhadas entre os dois DataFrames
    for col in ["taxa_desmatamento_prevista", "ano_tipo", "score_risco"]:
        if col not in gdf.columns:
            gdf[col] = np.nan

    gdf_final = gpd.GeoDataFrame(
        pd.concat([gdf, df_2026], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )

    # Ordenar colunas: manter schema legível
    col_order = [
        "cod_ibge", "municipio", "uf", "ano", "ano_tipo",
        "area_km2", "populacao", "pib_agropecuario",
        "desmatamento_km2", "taxa_desmatamento", "taxa_desmatamento_prevista",
        "taxa_desmatamento_prevista",
        "score_risco",
        "area_uc_pct", "area_ti_pct", "autos_ibama",
        "taxa_desmat_lag1", "taxa_desmat_lag2", "taxa_desmat_lag3",
        "taxa_desmat_roll3", "taxa_desmat_roll5", "std_desmat_roll3",
        "local_moran_i", "cluster_id",
        "geometry",
    ]
    # Manter apenas colunas que existem, sem duplicatas
    seen = set()
    col_order_clean = []
    for c in col_order:
        if c in gdf_final.columns and c not in seen:
            col_order_clean.append(c)
            seen.add(c)
    remaining = [c for c in gdf_final.columns if c not in seen]
    gdf_final = gdf_final[col_order_clean + remaining]

    gdf_final = gdf_final.sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    # ── 8. Salvar ──────────────────────────────────────────────────────────────
    gdf_final.to_parquet(out, index=False)
    logger.info("Predictions salvas em '%s'. Shape: %s", output_path, gdf_final.shape)

    # ── 9. Sumário final ───────────────────────────────────────────────────────
    hist = gdf_final[gdf_final["ano_tipo"] == "histórico"]
    prev = gdf_final[gdf_final["ano_tipo"] == "previsão"]

    top10_2026 = (
        prev.nlargest(10, "score_risco")
        [["municipio", "uf", "taxa_desmatamento_prevista", "score_risco"]]
    )
    logger.info(
        "\nTop-10 municípios de maior risco previsto (%d):\n%s",
        _ANO_PREVISAO,
        top10_2026.to_string(index=False),
    )

    return gdf_final


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    gdf = generate_scores()

    hist = gdf[gdf["ano_tipo"] == "histórico"]
    prev = gdf[gdf["ano_tipo"] == "previsão"]

    print(f"\nShape total: {gdf.shape}")
    print(f"Histórico (2008–2025): {len(hist)} linhas | {hist['ano'].nunique()} anos")
    print(f"Previsão 2026:         {len(prev)} linhas | {prev['score_risco'].notna().sum()} com score")
    print(f"\nScore histórico (por ano, todos os anos):")
    print(f"  min: {hist['score_risco'].min():.2f} | max: {hist['score_risco'].max():.2f} | mediana: {hist['score_risco'].median():.2f}")
    print(f"\nScore previsão 2026:")
    print(f"  min: {prev['score_risco'].min():.2f} | max: {prev['score_risco'].max():.2f} | mediana: {prev['score_risco'].median():.2f}")
    print(f"\nTop-10 risco 2026:")
    print(prev.nlargest(10, "score_risco")[
        ["municipio", "uf", "taxa_desmatamento_prevista", "score_risco"]
    ].to_string(index=False))
    print(f"\nExemplo — Altamira (1500602), 2023–2026:")
    print(gdf[gdf["cod_ibge"] == "1500602"].tail(5)[
        ["ano", "ano_tipo", "taxa_desmatamento", "taxa_desmatamento_prevista", "score_risco"]
    ].to_string(index=False))
