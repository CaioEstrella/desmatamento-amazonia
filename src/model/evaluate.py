"""
SHAP analysis do modelo LightGBM de risco de desmatamento — TASK-13.

Gera três tipos de visualização:
    1. Beeswarm plot global (top-20 features por |SHAP|)
    2. Waterfall plots para os 5 municípios de maior risco previsto
    3. Bar plot de importância SHAP média por cluster HDBSCAN

Outputs em reports/:
    shap_beeswarm.png
    shap_waterfall_<municipio>.png  (5 arquivos)
    shap_por_cluster.png
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EXCLUDE_COLS = {
    "cod_ibge", "municipio", "taxa_desmatamento",
    "desmatamento_km2", "score_risco", "geometry",
}
_MODEL_PATH   = "data/outputs/lgbm_model.pkl"
_DATASET_PATH = "data/processed/dataset.parquet"
_REPORTS_DIR  = Path("reports")


def _load_model_and_data(
    model_path: str = _MODEL_PATH,
    dataset_path: str = _DATASET_PATH,
) -> tuple:
    logger.info("Carregando modelo de '%s'...", model_path)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    logger.info("Carregando dataset de '%s'...", dataset_path)
    gdf = gpd.read_parquet(dataset_path)

    df = gdf[gdf["ano"] >= 2011].dropna(subset=["taxa_desmat_lag3"]).copy()
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

    for col in ["uf", "cluster_id"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    X = df[feature_cols]
    y = df["taxa_desmatamento"]
    logger.info("  Dataset: %d linhas | %d features", len(X), len(feature_cols))
    return model, X, y, df, feature_cols


def _compute_shap_values(model, X: pd.DataFrame):
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Instale shap: pip install shap") from exc

    logger.info("Calculando SHAP values (%d amostras)...", len(X))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    logger.info("  SHAP values calculados.")
    return shap_values, explainer


def plot_beeswarm(
    shap_values,
    X: pd.DataFrame,
    output_path: str = "reports/shap_beeswarm.png",
    max_display: int = 20,
) -> None:
    import shap

    logger.info("Gerando beeswarm plot...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # beeswarm não aceita ax; usar plot_size para controlar dimensões
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False,
        plot_size=(10, 8),
    )
    plt.title("Importância SHAP — Top features (taxa de desmatamento)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("  Beeswarm salvo em '%s'.", output_path)


def plot_waterfall_top5(
    shap_values,
    X: pd.DataFrame,
    df: pd.DataFrame,
    output_dir: str = "reports",
) -> None:
    import shap

    logger.info("Gerando waterfall plots para top-5 municípios...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Usar o valor previsto médio por município (último ano disponível)
    last_ano = df["ano"].max()
    top5_idx = (
        df[df["ano"] == last_ano]
        .nlargest(5, "taxa_desmatamento")
        .index
    )

    # Mapear para posição no array X (X pode ser subset do df)
    for rank, idx in enumerate(top5_idx, start=1):
        if idx not in X.index:
            continue
        pos = X.index.get_loc(idx)

        mun_name = df.loc[idx, "municipio"] if "municipio" in df.columns else str(idx)
        uf_code  = df.loc[idx, "uf"] if "uf" in df.columns else ""
        ano_val  = df.loc[idx, "ano"]

        shap.plots.waterfall(shap_values[pos], show=False)
        plt.gcf().set_size_inches(10, 6)
        plt.title(
            f"SHAP Waterfall — {mun_name} ({uf_code}) · {ano_val}",
            fontsize=12,
        )
        plt.tight_layout()

        safe_name = mun_name.lower().replace(" ", "_").replace("/", "-")
        out = Path(output_dir) / f"shap_waterfall_{safe_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Waterfall %d/%d salvo em '%s'.", rank, 5, out)


def plot_shap_by_cluster(
    shap_values,
    X: pd.DataFrame,
    df: pd.DataFrame,
    output_path: str = "reports/shap_por_cluster.png",
) -> None:
    logger.info("Gerando importância SHAP por cluster...")

    if "cluster_id" not in df.columns:
        logger.warning("  cluster_id ausente; pulando plot por cluster.")
        return

    shap_df = pd.DataFrame(
        np.abs(shap_values.values),
        index=X.index,
        columns=X.columns,
    )
    shap_df["cluster_id"] = df.loc[X.index, "cluster_id"].values

    cluster_importance = (
        shap_df.groupby("cluster_id")
        .mean()
        .T
        .sort_values(by=shap_df["cluster_id"].mode()[0], ascending=False)
        .head(15)
    )

    n_clusters = cluster_importance.shape[1]
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(cluster_importance))
    width = 0.8 / max(n_clusters, 1)
    cmap = plt.get_cmap("tab10")

    for i, col in enumerate(cluster_importance.columns):
        label = f"Cluster {int(col)}" if col != -1 else "Ruído"
        ax.bar(
            x + i * width,
            cluster_importance[col],
            width,
            label=label,
            color=cmap(i),
            alpha=0.85,
        )

    ax.set_xticks(x + width * (n_clusters - 1) / 2)
    ax.set_xticklabels(cluster_importance.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("|SHAP| médio")
    ax.set_title("Importância SHAP média por cluster HDBSCAN (top-15 features)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  SHAP por cluster salvo em '%s'.", output_path)


def run_shap_analysis(
    model_path: str = _MODEL_PATH,
    dataset_path: str = _DATASET_PATH,
    reports_dir: str = "reports",
) -> None:
    """Executa análise SHAP completa e salva todos os plots."""
    model, X, y, df, feature_cols = _load_model_and_data(model_path, dataset_path)
    shap_values, _ = _compute_shap_values(model, X)

    plot_beeswarm(shap_values, X, f"{reports_dir}/shap_beeswarm.png")
    plot_waterfall_top5(shap_values, X, df, reports_dir)
    plot_shap_by_cluster(shap_values, X, df, f"{reports_dir}/shap_por_cluster.png")

    logger.info("Análise SHAP concluída. Plots em '%s/'.", reports_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    run_shap_analysis()
