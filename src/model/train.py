"""
Treinamento do modelo LightGBM para predição de risco de desmatamento.

TASK-11: Tunagem de hiperparâmetros com Optuna (>= 50 trials)
    - Função objetivo: RMSE médio em spatial CV (UF como grupo)
    - Saída: data/outputs/optuna_study.pkl, data/outputs/best_params.json

TASK-12: Treinamento final com melhores hiperparâmetros
    - Treina no dataset completo (sem filtro temporal)
    - Avalia métricas por fold espacial
    - Saída: data/outputs/lgbm_model.pkl, data/outputs/metrics_by_fold.json

Spatial Cross-Validation:
    Usa sklearn GroupKFold com groups=UF, garantindo que todos os municípios
    de uma UF fiquem sempre no mesmo fold. Com 9 UFs e 5 folds, cada fold
    de validação contém 1-2 UFs inteiras, evitando vazamento espacial entre
    municípios vizinhos de estados diferentes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

# Colunas a excluir do conjunto de features
_EXCLUDE_COLS = {
    "cod_ibge", "municipio",
    "taxa_desmatamento", "desmatamento_km2",
    "score_risco", "geometry",
}

# Colunas categóricas para o LightGBM
_CATEGORICAL_COLS = ["uf", "cluster_id"]

# Anos mínimos para ter lags completos (lag3 disponível)
_ANO_MIN_TREINO = 2011

# Número de trials Optuna (mínimo = 50 conforme spec)
_N_TRIALS = 55

# Espaço de hiperparâmetros (9 parâmetros conforme spec)
_PARAM_SPACE = {
    "num_leaves":        ("int",   20, 300),
    "max_depth":         ("int",   3, 12),
    "learning_rate":     ("float", 0.01, 0.3, True),  # log=True
    "n_estimators":      ("int",   100, 1000),
    "min_child_samples": ("int",   5, 100),
    "subsample":         ("float", 0.5, 1.0),
    "colsample_bytree":  ("float", 0.5, 1.0),
    "reg_alpha":         ("float", 1e-8, 10.0, True),
    "reg_lambda":        ("float", 1e-8, 10.0, True),
}


def _load_dataset(dataset_path: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Carrega e prepara o dataset para treinamento.

    Filtra anos com lags completos, codifica categorias e retorna
    X, y e groups (array de grupo UF para spatial CV).

    Returns:
        (X, y, groups) onde groups é array de inteiros por UF.
    """
    logger.info("Carregando '%s'...", dataset_path)
    gdf = gpd.read_parquet(dataset_path)

    # Filtrar para anos com lag3 disponível
    df = pd.DataFrame(gdf.drop(columns=["geometry"]))
    df = df[df["ano"] >= _ANO_MIN_TREINO].copy()
    df = df.dropna(subset=["taxa_desmat_lag3"]).copy()
    logger.info("  Dataset após filtro temporal: %d linhas (anos >= %d)",
                len(df), _ANO_MIN_TREINO)

    # Target
    y = df["taxa_desmatamento"].copy()

    # Features: todas as colunas exceto as excluídas
    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]
    X = df[feature_cols].copy()

    # Codificar colunas categóricas como inteiros para o LightGBM
    for col in _CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes

    logger.info("  Features: %d | Target: taxa_desmatamento", len(feature_cols))
    logger.info("  UFs presentes: %s", sorted(df["uf"].unique()))

    # Groups para spatial CV: mesmo inteiro para todos os registros da mesma UF
    uf_codes = df["uf"].astype("category").cat.codes.values

    return X, y, uf_codes


def _make_folds(X: pd.DataFrame, y: pd.Series, groups: np.ndarray, n_splits: int = 5):
    """Cria folds de validação cruzada espacial (UF como grupo)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y, groups=groups))


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
) -> float:
    """Função objetivo do Optuna: RMSE médio em spatial CV."""
    params = {
        "num_leaves":        trial.suggest_int("num_leaves", *_PARAM_SPACE["num_leaves"][1:]),
        "max_depth":         trial.suggest_int("max_depth", *_PARAM_SPACE["max_depth"][1:]),
        "learning_rate":     trial.suggest_float("learning_rate",
                                                   _PARAM_SPACE["learning_rate"][1],
                                                   _PARAM_SPACE["learning_rate"][2],
                                                   log=True),
        "n_estimators":      trial.suggest_int("n_estimators", *_PARAM_SPACE["n_estimators"][1:]),
        "min_child_samples": trial.suggest_int("min_child_samples",
                                                *_PARAM_SPACE["min_child_samples"][1:]),
        "subsample":         trial.suggest_float("subsample", *_PARAM_SPACE["subsample"][1:]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",
                                                   *_PARAM_SPACE["colsample_bytree"][1:]),
        "reg_alpha":         trial.suggest_float("reg_alpha",
                                                   _PARAM_SPACE["reg_alpha"][1],
                                                   _PARAM_SPACE["reg_alpha"][2],
                                                   log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda",
                                                   _PARAM_SPACE["reg_lambda"][1],
                                                   _PARAM_SPACE["reg_lambda"][2],
                                                   log=True),
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    rmse_scores = []
    for train_idx, val_idx in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        # Clip negatives — taxa de desmatamento não pode ser negativa
        preds = np.clip(preds, 0, None)
        rmse_scores.append(_rmse(y.iloc[val_idx], preds))

    return float(np.mean(rmse_scores))


def run_optuna(
    dataset_path: str = "data/processed/dataset.parquet",
    study_path: str = "data/outputs/optuna_study.pkl",
    params_path: str = "data/outputs/best_params.json",
    n_trials: int = _N_TRIALS,
) -> dict:
    """
    Executa o estudo Optuna para tunar os hiperparâmetros do LightGBM.

    Realiza busca com >= 50 trials usando spatial cross-validation (UF como
    grupo). Salva o estudo completo e os melhores hiperparâmetros encontrados.

    Cache: se study_path já existir, carrega sem re-executar a tunagem.

    Args:
        dataset_path: Dataset com features completas (pós TASK-10).
        study_path: Caminho para salvar o estudo Optuna (joblib.pkl).
        params_path: Caminho para salvar best_params como JSON.
        n_trials: Número de trials (mínimo 50 conforme spec).

    Returns:
        Dicionário com os melhores hiperparâmetros.

    Raises:
        AssertionError: Se best RMSE não melhorar baseline com params padrão.
    """
    study_file = Path(study_path)
    params_file = Path(params_path)

    if study_file.exists() and params_file.exists():
        logger.info("Estudo Optuna encontrado em '%s'. Carregando.", study_path)
        study = joblib.load(study_file)
        best_params = json.loads(params_file.read_text(encoding="utf-8"))
        logger.info("  Melhor RMSE: %.6f | Trials: %d",
                    study.best_trial.value, len(study.trials))
        return best_params

    Path(study_path).parent.mkdir(parents=True, exist_ok=True)

    X, y, groups = _load_dataset(dataset_path)
    folds = _make_folds(X, y, groups, n_splits=5)

    logger.info("Calculando baseline RMSE (parâmetros padrão LightGBM)...")
    baseline_rmse_scores = []
    for train_idx, val_idx in folds:
        m = lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = np.clip(m.predict(X.iloc[val_idx]), 0, None)
        baseline_rmse_scores.append(_rmse(y.iloc[val_idx], preds))
    baseline_rmse = float(np.mean(baseline_rmse_scores))
    logger.info("  Baseline RMSE (params padrão): %.6f", baseline_rmse)

    logger.info("Iniciando estudo Optuna (%d trials)...", n_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm_desmatamento",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective_wrapper(trial):
        return _objective(trial, X, y, folds)

    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True)

    best_rmse = study.best_trial.value
    logger.info("  Melhor RMSE (Optuna): %.6f | Melhoria: %.2f%%",
                best_rmse,
                (baseline_rmse - best_rmse) / baseline_rmse * 100)

    assert best_rmse < baseline_rmse, (
        f"Optuna não melhorou o baseline! Best={best_rmse:.6f} >= Baseline={baseline_rmse:.6f}"
    )

    # Salvar estudo
    joblib.dump(study, study_path)
    logger.info("Estudo salvo em '%s'.", study_path)

    # Salvar best_params como JSON (sem random_state e verbosity)
    best_params = dict(study.best_params)
    best_params["random_state"] = 42
    best_params["verbosity"] = -1
    best_params["n_jobs"] = -1

    params_file.write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Melhores hiperparâmetros salvos em '%s'.", params_path)
    logger.info("  %s", json.dumps({k: v for k, v in best_params.items()
                                     if k not in ("random_state", "verbosity", "n_jobs")},
                                    indent=2))

    return best_params


def train_final_model(
    dataset_path: str = "data/processed/dataset.parquet",
    params_path: str = "data/outputs/best_params.json",
    model_path: str = "data/outputs/lgbm_model.pkl",
    metrics_path: str = "data/outputs/metrics_by_fold.json",
) -> dict:
    """
    Treina o modelo LightGBM final com os melhores hiperparâmetros.

    Realiza spatial CV para reportar métricas por fold (UF), depois treina
    o modelo final em todo o dataset de treino.

    Cache: se model_path já existir, carrega sem re-treinar.

    Args:
        dataset_path: Dataset com features completas.
        params_path: JSON com melhores hiperparâmetros do Optuna.
        model_path: Caminho para salvar o modelo joblib.pkl.
        metrics_path: Caminho para salvar métricas por fold JSON.

    Returns:
        Dicionário de métricas por fold.

    Raises:
        FileNotFoundError: Se best_params.json não existir.
        AssertionError: Se R² médio < 0.65.
    """
    model_file = Path(model_path)

    if model_file.exists():
        logger.info("Modelo encontrado em '%s'. Pulando treinamento.", model_path)
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        return metrics

    params_file = Path(params_path)
    if not params_file.exists():
        raise FileNotFoundError(
            f"'{params_path}' não encontrado. Execute run_optuna() primeiro."
        )

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    best_params = json.loads(params_file.read_text(encoding="utf-8"))
    logger.info("Carregando best_params de '%s'...", params_path)

    X, y, groups = _load_dataset(dataset_path)
    folds = _make_folds(X, y, groups, n_splits=5)

    # Derivar os nomes das UFs por fold (para reportar métricas)
    gdf = gpd.read_parquet(dataset_path)
    df_base = gdf[gdf["ano"] >= _ANO_MIN_TREINO].dropna(subset=["taxa_desmat_lag3"])
    uf_by_idx = df_base["uf"].values

    # ── Spatial CV para métricas ───────────────────────────────────────────────
    logger.info("Avaliando com spatial CV (%d folds)...", len(folds))
    metrics_by_fold = {}

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        ufs_val = sorted(set(uf_by_idx[val_idx]))
        fold_name = "+".join(ufs_val) if ufs_val else f"fold_{fold_i}"

        model_fold = lgb.LGBMRegressor(**best_params)
        model_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = np.clip(model_fold.predict(X.iloc[val_idx]), 0, None)

        y_val = y.iloc[val_idx]
        fold_metrics = {
            "uf": fold_name,
            "n_samples": len(val_idx),
            "rmse": float(_rmse(y_val, preds)),
            "mae": float(mean_absolute_error(y_val, preds)),
            "r2": float(r2_score(y_val, preds)),
        }
        metrics_by_fold[f"fold_{fold_i}"] = fold_metrics
        logger.info("  Fold %d (%s): RMSE=%.4f | MAE=%.4f | R²=%.4f",
                    fold_i, fold_name,
                    fold_metrics["rmse"], fold_metrics["mae"], fold_metrics["r2"])

    # Média dos folds
    rmse_mean = np.mean([m["rmse"] for m in metrics_by_fold.values()])
    mae_mean = np.mean([m["mae"] for m in metrics_by_fold.values()])
    r2_mean = np.mean([m["r2"] for m in metrics_by_fold.values()])
    metrics_by_fold["mean"] = {
        "uf": "MÉDIA",
        "n_samples": sum(m["n_samples"] for m in metrics_by_fold.values()),
        "rmse": float(rmse_mean),
        "mae": float(mae_mean),
        "r2": float(r2_mean),
    }
    logger.info("  Média: RMSE=%.4f | MAE=%.4f | R²=%.4f", rmse_mean, mae_mean, r2_mean)

    # Tabela de métricas
    rows = [(f["uf"], f["n_samples"], f["rmse"], f["mae"], f["r2"])
            for f in metrics_by_fold.values()]
    header = f"\n{'UF(s)':30s} {'N':>6} {'RMSE':>8} {'MAE':>8} {'R²':>8}"
    logger.info("Tabela de métricas por fold:%s", header)
    for fold_name, n, rmse, mae, r2 in rows:
        logger.info("  %-30s %6d %8.4f %8.4f %8.4f", fold_name, n, rmse, mae, r2)

    assert r2_mean >= 0.65, (
        f"R² médio abaixo do esperado: {r2_mean:.4f} (mínimo: 0.65). "
        "Verifique as features e o dataset."
    )

    # ── Treinar modelo final no dataset completo ───────────────────────────────
    logger.info("Treinando modelo final em todo o dataset...")
    model_final = lgb.LGBMRegressor(**best_params)
    model_final.fit(X, y)

    joblib.dump(model_final, model_path)
    logger.info("Modelo final salvo em '%s'.", model_path)

    Path(metrics_path).write_text(
        json.dumps(metrics_by_fold, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Métricas por fold salvas em '%s'.", metrics_path)

    return metrics_by_fold


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    task = sys.argv[1] if len(sys.argv) > 1 else "all"

    if task in ("11", "all"):
        print("\n=== TASK-11: Optuna tuning ===")
        best_params = run_optuna()
        print(f"\nMelhores hiperparâmetros encontrados:")
        for k, v in best_params.items():
            if k not in ("random_state", "verbosity", "n_jobs"):
                print(f"  {k}: {v}")

    if task in ("12", "all"):
        print("\n=== TASK-12: Treinamento final ===")
        metrics = train_final_model()
        print("\nMétricas por fold:")
        for fold, m in metrics.items():
            print(f"  {m['uf']:30s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}")
