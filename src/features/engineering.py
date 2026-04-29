"""
Feature engineering temporal e espacial para o dataset de desmatamento.

Módulo com duas funções principais:

    add_temporal_features(df)  → lag features e rolling window (TASK-07)
    add_local_moran(df, gdf)   → Local Moran's I por ano (TASK-08)

Ambas recebem o dataset de data/processed/dataset.parquet e retornam uma
cópia enriquecida, sobreescrevendo o arquivo de saída.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def _normalize_geom_for_queen(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Garante que o GeoDataFrame contém apenas Polygon/MultiPolygon.

    `Queen.from_dataframe` levanta TypeError para GeometryCollection.
    Esta função extrai apenas as partes poligonais de geometrias mistas.
    Municípios que resultam em geometria nula/vazia são removidos.
    """
    def extract_poly(g):
        if g is None or g.is_empty:
            return None
        if isinstance(g, (Polygon, MultiPolygon)):
            return g if g.is_valid else g.buffer(0)
        polys = [
            part for part in getattr(g, "geoms", [])
            if isinstance(part, (Polygon, MultiPolygon))
        ]
        if not polys:
            return None
        result = unary_union(polys)
        return result if not result.is_empty else None

    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(extract_poly)
    removed = gdf["geometry"].isna().sum()
    if removed > 0:
        logger.info("  %d municípios removidos (geometria não poligonal).", removed)
    gdf = gdf[gdf["geometry"].notna()].reset_index(drop=True)
    return gdf


# ── TASK-07: Lag features e rolling window ───────────────────────────────────

def add_temporal_features(
    input_path: str = "data/processed/dataset.parquet",
    output_path: str = "data/processed/dataset.parquet",
) -> gpd.GeoDataFrame:
    """
    Adiciona lag features e rolling window de taxa_desmatamento ao dataset.

    Calcula, para cada município ordenado por (cod_ibge, ano):
        - taxa_desmat_lag1/2/3: valor defasado 1, 2 e 3 anos atrás.
        - taxa_desmat_roll3: média móvel de 3 anos anteriores.
        - taxa_desmat_roll5: média móvel de 5 anos anteriores.
        - std_desmat_roll3:  desvio padrão móvel de 3 anos anteriores.

    O `.shift(1)` antes do rolling evita data leakage: a janela nunca usa
    o valor do próprio ano. Primeiros anos de cada município terão NaN
    nos lags (comportamento correto).

    Args:
        input_path: Caminho do dataset de entrada (GeoParquet).
        output_path: Caminho para sobrescrever com dataset atualizado.

    Returns:
        GeoDataFrame com as 6 colunas de features adicionadas.
    """
    logger.info("Carregando '%s'...", input_path)
    gdf = gpd.read_parquet(input_path)

    if "taxa_desmat_lag1" in gdf.columns:
        logger.info("Lag features já presentes. Pulando.")
        return gdf

    logger.info("Calculando lag features e rolling window...")
    gdf = gdf.sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    grp = gdf.groupby("cod_ibge")["taxa_desmatamento"]

    # Lag features (shift sem janela)
    gdf["taxa_desmat_lag1"] = grp.shift(1)
    gdf["taxa_desmat_lag2"] = grp.shift(2)
    gdf["taxa_desmat_lag3"] = grp.shift(3)

    # Rolling window sobre valores JÁ defasados (shift(1) antes do rolling)
    gdf["taxa_desmat_roll3"] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    gdf["taxa_desmat_roll5"] = grp.transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    gdf["std_desmat_roll3"] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    )

    # Validação de data leakage: lag1 em t = taxa em t-1
    sample_mun = gdf["cod_ibge"].iloc[18]  # pegar um município com dados suficientes
    df_mun = gdf[gdf["cod_ibge"] == sample_mun].sort_values("ano")
    if len(df_mun) >= 3:
        ano_t = df_mun.iloc[2]["ano"]
        lag1_t = df_mun.iloc[2]["taxa_desmat_lag1"]
        taxa_t_minus_1 = df_mun.iloc[1]["taxa_desmatamento"]
        if not np.isnan(lag1_t) and not np.isnan(taxa_t_minus_1):
            assert abs(lag1_t - taxa_t_minus_1) < 1e-9, (
                f"Data leakage! lag1({ano_t}) = {lag1_t:.6f} ≠ taxa({ano_t - 1}) = {taxa_t_minus_1:.6f}"
            )
            logger.info(
                "  Verificação anti-leakage OK: lag1(%d) = %.6f = taxa(%d)",
                ano_t, lag1_t, ano_t - 1,
            )

    n_lag_nulos = gdf["taxa_desmat_lag1"].isna().sum()
    logger.info(
        "  taxa_desmat_lag1: %d nulos (%.1f%%) — esperado para 1º ano de cada município",
        n_lag_nulos, n_lag_nulos / len(gdf) * 100,
    )

    # Reordenar colunas para incluir as novas antes de geometry
    base_cols = [c for c in gdf.columns if c not in (
        "taxa_desmat_lag1", "taxa_desmat_lag2", "taxa_desmat_lag3",
        "taxa_desmat_roll3", "taxa_desmat_roll5", "std_desmat_roll3",
        "geometry",
    )]
    feature_cols = [
        "taxa_desmat_lag1", "taxa_desmat_lag2", "taxa_desmat_lag3",
        "taxa_desmat_roll3", "taxa_desmat_roll5", "std_desmat_roll3",
    ]
    gdf = gdf[base_cols + feature_cols + ["geometry"]].copy()

    gdf.to_parquet(output_path, index=False)
    logger.info("Salvo em '%s'.", output_path)
    return gdf


# ── TASK-08: Local Moran's I ─────────────────────────────────────────────────

def add_local_moran(
    input_path: str = "data/processed/dataset.parquet",
    output_path: str = "data/processed/dataset.parquet",
    moran_plot_path: str = "data/outputs/moran_scatterplot.png",
) -> gpd.GeoDataFrame:
    """
    Adiciona o Local Moran's I da taxa de desmatamento ao dataset.

    Para cada ano disponível, calcula o I local usando pesos Queen
    (vizinhança por aresta/vértice) row-standardizados. Municípios sem
    vizinhos (ilhas geográficas) recebem valor 0.

    O Moran's I local captura o efeito de clustering espacial: valor positivo
    alto indica município com alto desmatamento rodeado de vizinhos com alto
    desmatamento (hotspot); valor negativo indica outlier espacial.

    Salva um scatterplot de Moran (taxa_desmatamento médio × I médio) em
    data/outputs/moran_scatterplot.png.

    Args:
        input_path: Caminho do dataset com taxa_desmat_lag* (pós TASK-07).
        output_path: Caminho para sobrescrever com local_moran_i adicionado.
        moran_plot_path: Caminho do scatterplot de saída.

    Returns:
        GeoDataFrame com coluna local_moran_i (float) adicionada.
    """
    try:
        from esda.moran import Moran_Local
        from libpysal.weights import Queen
    except ImportError as exc:
        raise ImportError(
            "Pacotes 'esda' e 'libpysal' são necessários para o Moran's I. "
            "Execute: pip install esda libpysal"
        ) from exc

    logger.info("Carregando '%s'...", input_path)
    gdf = gpd.read_parquet(input_path)

    if "local_moran_i" in gdf.columns:
        logger.info("local_moran_i já presente. Pulando.")
        return gdf

    anos = sorted(gdf["ano"].unique())
    logger.info("Calculando Local Moran's I para %d anos...", len(anos))

    moran_vals: dict[tuple, float] = {}

    # Calcular pesos de vizinhança uma vez por geometria (não por ano)
    # Usar apenas os municípios do último ano como referência espacial
    gdf_geom = (
        gdf[gdf["ano"] == anos[-1]][["cod_ibge", "geometry"]]
        .copy()
        .reset_index(drop=True)
    )

    # Queen.from_dataframe não aceita GeometryCollection — extrair só partes poligonais
    gdf_geom = _normalize_geom_for_queen(gdf_geom)
    logger.info("  Construindo pesos Queen (%d municípios)...", len(gdf_geom))
    w = Queen.from_dataframe(gdf_geom, silence_warnings=True)
    w.transform = "r"  # row-standardize

    for ano in anos:
        subset = (
            gdf[gdf["ano"] == ano]
            .merge(gdf_geom[["cod_ibge"]], on="cod_ibge", how="inner")
            .set_index("cod_ibge")
        )
        # Alinhar com a ordem dos pesos
        y = np.array([
            subset.loc[cod, "taxa_desmatamento"] if cod in subset.index else 0.0
            for cod in gdf_geom["cod_ibge"]
        ])
        y = np.nan_to_num(y, nan=0.0)

        moran_loc = Moran_Local(y, w, seed=42)
        for cod, moran_i in zip(gdf_geom["cod_ibge"], moran_loc.Is):
            moran_vals[(cod, ano)] = float(moran_i)

    logger.info("  Atribuindo valores ao dataset...")
    gdf["local_moran_i"] = gdf.apply(
        lambda r: moran_vals.get((r["cod_ibge"], r["ano"]), 0.0), axis=1
    )

    # Reordenar colunas (local_moran_i antes de geometry)
    non_geom = [c for c in gdf.columns if c != "geometry"]
    gdf = gpd.GeoDataFrame(
        gdf[non_geom].copy(),
        geometry=gdf["geometry"].values,
        crs="EPSG:4326",
    )

    # Scatterplot de Moran
    _plot_moran_scatter(gdf, moran_plot_path)

    gdf.to_parquet(output_path, index=False)
    logger.info("Salvo em '%s'.", output_path)
    return gdf


def _plot_moran_scatter(gdf: gpd.GeoDataFrame, plot_path: str) -> None:
    """Salva scatterplot de Moran (taxa_desmatamento média vs I médio) por município."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib não disponível; scatterplot de Moran não gerado.")
        return

    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    by_mun = gdf.groupby("cod_ibge").agg(
        taxa_media=("taxa_desmatamento", "mean"),
        moran_medio=("local_moran_i", "mean"),
        uf=("uf", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        by_mun["taxa_media"],
        by_mun["moran_medio"],
        c=by_mun["taxa_media"],
        cmap="RdYlGn_r",
        alpha=0.5,
        s=12,
        linewidths=0,
    )
    plt.colorbar(scatter, ax=ax, label="Taxa de desmatamento média (%)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(by_mun["taxa_media"].mean(), color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Taxa de desmatamento média (%/ano)")
    ax.set_ylabel("Local Moran's I médio")
    ax.set_title("Moran Scatterplot — Municípios da Amazônia Legal")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("  Scatterplot salvo em '%s'.", plot_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    task = sys.argv[1] if len(sys.argv) > 1 else "07"

    if task in ("07", "all"):
        gdf = add_temporal_features()
        print(f"\nShape após TASK-07: {gdf.shape}")
        print(f"Colunas lag/roll: {[c for c in gdf.columns if 'lag' in c or 'roll' in c or 'std' in c]}")
        # Exemplo de verificação anti-leakage
        mun = gdf["cod_ibge"].iloc[18]
        df_m = gdf[gdf["cod_ibge"] == mun].sort_values("ano")
        print(f"\nVerificação de lag para município {mun}:")
        print(df_m[["ano", "taxa_desmatamento", "taxa_desmat_lag1", "taxa_desmat_roll3"]].head(5).to_string())

    if task in ("08", "all"):
        gdf = add_local_moran()
        print(f"\nShape após TASK-08: {gdf.shape}")
        print(f"local_moran_i — min: {gdf['local_moran_i'].min():.4f} | max: {gdf['local_moran_i'].max():.4f}")
        print(f"Zeros: {(gdf['local_moran_i'] == 0).sum()}")
