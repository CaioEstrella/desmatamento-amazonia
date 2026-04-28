"""
ETL — Consolida as cinco fontes de dados brutas em dataset.parquet.

Fontes de entrada (data/raw/):
    inpe_raw.parquet   → desmatamento_km2 por (cod_ibge, ano)
    ibge_raw.parquet   → populacao, pib_agropecuario, area_km2
    icmbio_raw.parquet → area_uc_km2, area_ti_km2 (estático, sem ano)
    ibama_raw.parquet  → autos_ibama por (cod_ibge, ano)
    municipios.gpkg    → geometria canonical dos 805 municípios

Estratégia de join:
    1. Painel base: 805 municípios (municipios.gpkg) × anos do INPE (2008–2025)
    2. Left join de todas as fontes sobre esse painel.
    3. Colunas derivadas: taxa_desmatamento, area_uc_pct, area_ti_pct.
    4. Imputação conservadora para missing values (zeros para áreas, ffill para séries).

Output:
    data/processed/dataset.parquet (GeoParquet)
    Colunas: cod_ibge, municipio, uf, ano, area_km2, populacao,
             pib_agropecuario, desmatamento_km2, taxa_desmatamento,
             area_uc_pct, area_ti_pct, autos_ibama, geometry
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_AMAZONIA_LEGAL_UFS = {"AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"}


def _load_raw(raw_dir: Path) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """Carrega todas as fontes brutas. Levanta FileNotFoundError se alguma faltar."""
    paths = {
        "inpe": raw_dir / "inpe_raw.parquet",
        "ibge": raw_dir / "ibge_raw.parquet",
        "icmbio": raw_dir / "icmbio_raw.parquet",
        "ibama": raw_dir / "ibama_raw.parquet",
        "municipios": raw_dir / "municipios.gpkg",
    }
    sources: dict[str, pd.DataFrame | gpd.GeoDataFrame] = {}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo '{path}' não encontrado. Execute a TASK correspondente primeiro."
            )
        if path.suffix == ".gpkg":
            sources[name] = gpd.read_file(path)
        else:
            sources[name] = pd.read_parquet(path)
        logger.info("  Carregado %-12s → %d linhas", name, len(sources[name]))

    return sources


def _validate_cod_ibge(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Garante que cod_ibge tem 7 dígitos numéricos."""
    df["cod_ibge"] = (
        df["cod_ibge"].astype(str).str.strip()
        .str.replace(r"\D", "", regex=True)
        .str.zfill(7)
    )
    invalidos = ~df["cod_ibge"].str.match(r"^\d{7}$")
    if invalidos.any():
        n = invalidos.sum()
        logger.warning("  %s: %d cod_ibge inválidos removidos.", source, n)
        df = df[~invalidos].copy()
    return df


def _interpolate_series(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Preenche NaN em uma série temporal por município com interpolação linear.

    Ordena por (cod_ibge, ano) e aplica interpolação dentro de cada grupo.
    Valores ainda ausentes nas bordas (extrapolação) são preenchidos com
    o primeiro/último valor disponível (ffill/bfill).
    """
    df = df.sort_values(["cod_ibge", "ano"])
    df[col] = (
        df.groupby("cod_ibge")[col]
        .transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
    )
    return df


def build_dataset(
    raw_dir: str = "data/raw",
    output_path: str = "data/processed/dataset.parquet",
) -> gpd.GeoDataFrame:
    """
    Consolida as cinco fontes de dados em um único GeoDataFrame municipio×ano.

    Carrega os arquivos de data/raw/, faz left join a partir do painel canônico
    (municipios × anos do INPE), calcula colunas derivadas e salva como GeoParquet.

    Cache: se output_path já existir, retorna sem reprocessar.

    Args:
        raw_dir: Diretório com os arquivos de dados brutos.
        output_path: Caminho do GeoParquet de saída.

    Returns:
        GeoDataFrame com schema completo (exceto features de lag, Moran's I e cluster).

    Raises:
        FileNotFoundError: Se algum arquivo de entrada estiver ausente.
        AssertionError: Se as validações de qualidade falharem.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem reprocessar.", output_path)
        return gpd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. Carregar fontes ────────────────────────────────────────────────────
    logger.info("Carregando fontes brutas de '%s'...", raw_dir)
    raw = _load_raw(Path(raw_dir))

    df_inpe: pd.DataFrame = raw["inpe"]
    df_ibge: pd.DataFrame = raw["ibge"]
    df_icmbio: pd.DataFrame = raw["icmbio"]
    df_ibama: pd.DataFrame = raw["ibama"]
    gdf_mun: gpd.GeoDataFrame = raw["municipios"]

    # ── 2. Normalizar cod_ibge ────────────────────────────────────────────────
    logger.info("Normalizando chaves...")
    for name, df in [("INPE", df_inpe), ("IBGE", df_ibge),
                     ("ICMBio", df_icmbio), ("IBAMA", df_ibama), ("municípios", gdf_mun)]:
        _validate_cod_ibge(df, name)

    # ── 3. Painel base: 805 municípios × anos do INPE ────────────────────────
    anos_inpe = sorted(df_inpe["ano"].unique())
    logger.info("Anos INPE: %d–%d (%d anos)", min(anos_inpe), max(anos_inpe), len(anos_inpe))

    # Municípios canônicos da Amazônia Legal (somente as 9 UFs)
    mun_canonical = gdf_mun[gdf_mun["uf"].isin(_AMAZONIA_LEGAL_UFS)][
        ["cod_ibge", "municipio", "uf"]
    ].drop_duplicates("cod_ibge").copy()
    logger.info("Municípios canônicos (9 UFs): %d", len(mun_canonical))

    df_panel = mun_canonical.assign(key=1).merge(
        pd.DataFrame({"ano": anos_inpe, "key": 1}), on="key"
    ).drop(columns="key")
    logger.info("Painel base: %d linhas (%d × %d)", len(df_panel),
                len(mun_canonical), len(anos_inpe))

    # ── 4. Join INPE (desmatamento_km2) ──────────────────────────────────────
    logger.info("Juntando dados INPE...")
    # Filtrar INPE para só os municípios canônicos
    df_inpe_amz = df_inpe[df_inpe["cod_ibge"].isin(mun_canonical["cod_ibge"])].copy()
    df_panel = df_panel.merge(
        df_inpe_amz[["cod_ibge", "ano", "desmatamento_km2"]],
        on=["cod_ibge", "ano"],
        how="left",
    )
    df_panel["desmatamento_km2"] = df_panel["desmatamento_km2"].fillna(0.0)

    # ── 5. Join IBGE (populacao, pib_agropecuario, area_km2) ─────────────────
    logger.info("Juntando dados IBGE...")
    df_ibge_amz = df_ibge[df_ibge["uf"].isin(_AMAZONIA_LEGAL_UFS)].copy()
    df_panel = df_panel.merge(
        df_ibge_amz[["cod_ibge", "ano", "populacao", "pib_agropecuario", "area_km2"]],
        on=["cod_ibge", "ano"],
        how="left",
    )

    # Imputar séries temporais de IBGE por interpolação
    for col in ["populacao", "pib_agropecuario", "area_km2"]:
        n_null = df_panel[col].isna().sum()
        if n_null > 0:
            logger.info("  Interpolando %d nulos em '%s'...", n_null, col)
            df_panel = _interpolate_series(df_panel, col)

    # area_km2 restante nula: usar mediana do estado (fallback)
    if df_panel["area_km2"].isna().any():
        mediana_uf = df_panel.groupby("uf")["area_km2"].median()
        df_panel["area_km2"] = df_panel.apply(
            lambda r: mediana_uf[r["uf"]] if pd.isna(r["area_km2"]) else r["area_km2"],
            axis=1,
        )

    # ── 6. Join ICMBio (area_uc_km2, area_ti_km2 — estático) ─────────────────
    logger.info("Juntando dados ICMBio...")
    df_panel = df_panel.merge(
        df_icmbio[["cod_ibge", "area_uc_km2", "area_ti_km2"]],
        on="cod_ibge",
        how="left",
    )
    df_panel["area_uc_km2"] = df_panel["area_uc_km2"].fillna(0.0)
    df_panel["area_ti_km2"] = df_panel["area_ti_km2"].fillna(0.0)

    # ── 7. Join IBAMA (autos_ibama) ───────────────────────────────────────────
    logger.info("Juntando dados IBAMA...")
    df_ibama_amz = df_ibama[df_ibama["cod_ibge"].isin(mun_canonical["cod_ibge"])].copy()
    df_panel = df_panel.merge(
        df_ibama_amz[["cod_ibge", "ano", "autos_ibama"]],
        on=["cod_ibge", "ano"],
        how="left",
    )
    df_panel["autos_ibama"] = df_panel["autos_ibama"].fillna(0).astype(int)

    # ── 8. Colunas derivadas ──────────────────────────────────────────────────
    logger.info("Calculando colunas derivadas...")
    # Evitar divisão por zero (area_km2 = 0 é improvável mas possível para municípios novos)
    area_safe = df_panel["area_km2"].replace(0, np.nan)
    df_panel["taxa_desmatamento"] = (df_panel["desmatamento_km2"] / area_safe * 100).fillna(0.0)
    df_panel["area_uc_pct"] = (df_panel["area_uc_km2"] / area_safe * 100).fillna(0.0)
    df_panel["area_ti_pct"] = (df_panel["area_ti_km2"] / area_safe * 100).fillna(0.0)

    # ── 9. Adicionar geometria ────────────────────────────────────────────────
    logger.info("Adicionando geometria dos municípios...")
    gdf_result = gpd.GeoDataFrame(
        df_panel.merge(
            gdf_mun[["cod_ibge", "geometry"]],
            on="cod_ibge",
            how="left",
        ),
        geometry="geometry",
        crs="EPSG:4326",
    )

    # ── 10. Ordenar colunas ────────────────────────────────────────────────────
    cols_ordered = [
        "cod_ibge", "municipio", "uf", "ano",
        "area_km2", "populacao", "pib_agropecuario",
        "desmatamento_km2", "taxa_desmatamento",
        "area_uc_pct", "area_ti_pct",
        "autos_ibama",
        "geometry",
    ]
    gdf_result = gdf_result[cols_ordered].sort_values(
        ["cod_ibge", "ano"]
    ).reset_index(drop=True)

    # ── 11. Validações ─────────────────────────────────────────────────────────
    logger.info("Validando dataset...")

    # cod_ibge com 7 dígitos
    cod_invalidos = ~gdf_result["cod_ibge"].str.match(r"^\d{7}$")
    assert not cod_invalidos.any(), f"{cod_invalidos.sum()} cod_ibge inválidos!"

    # ano dentro do range esperado
    ano_min, ano_max = gdf_result["ano"].min(), gdf_result["ano"].max()
    assert 2000 <= ano_min, f"Ano mínimo inesperado: {ano_min}"
    assert ano_max <= 2030, f"Ano máximo inesperado: {ano_max}"

    # taxa_desmatamento ≥ 0
    assert (gdf_result["taxa_desmatamento"] >= 0).all(), "taxa_desmatamento negativa!"

    # Missing values < 5%
    for col in ["area_km2", "populacao", "pib_agropecuario", "desmatamento_km2",
                "taxa_desmatamento", "area_uc_pct", "area_ti_pct"]:
        pct_null = gdf_result[col].isna().mean() * 100
        if pct_null > 5:
            logger.warning("  ATENÇÃO: '%s' tem %.1f%% de nulos!", col, pct_null)
        assert pct_null <= 5, f"'{col}' tem {pct_null:.1f}% de missing (> 5%)"

    # Sanity: municípios conhecidos presentes
    municipios_esperados = {
        "1500602": "Altamira",
        "1507300": "São Félix do Xingu",
        "1505031": "Novo Progresso",
    }
    for cod, nome in municipios_esperados.items():
        assert cod in gdf_result["cod_ibge"].values, f"Município {nome} ({cod}) ausente!"

    # Tamanho mínimo
    assert len(gdf_result) > 5000, f"Dataset muito pequeno: {len(gdf_result)} linhas"

    n_munic = gdf_result["cod_ibge"].nunique()
    n_anos = gdf_result["ano"].nunique()
    logger.info(
        "Dataset final: %d linhas | %d municípios | %d anos | anos: %d–%d",
        len(gdf_result), n_munic, n_anos, ano_min, ano_max,
    )

    # Missing por coluna (resumo)
    nulls = gdf_result.drop(columns="geometry").isnull().mean() * 100
    for col, pct in nulls[nulls > 0].items():
        logger.info("  Missing: %s = %.2f%%", col, pct)

    # ── 12. Salvar como GeoParquet ─────────────────────────────────────────────
    gdf_result.to_parquet(path, index=False)
    logger.info("Salvo em '%s'.", output_path)

    return gdf_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    gdf = build_dataset()
    print(f"\nShape: {gdf.shape}")
    print(f"Municípios: {gdf['cod_ibge'].nunique()}")
    print(f"Anos: {gdf['ano'].min()}–{gdf['ano'].max()}")
    print(f"CRS: {gdf.crs}")
    print(f"\nMissing por coluna:\n{gdf.drop(columns='geometry').isnull().mean().to_string()}")
    print(f"\nEstatísticas básicas:")
    print(gdf[["taxa_desmatamento", "area_uc_pct", "area_ti_pct", "autos_ibama"]].describe())
    print(f"\nAltamira-PA:")
    print(gdf[gdf["cod_ibge"] == "1500602"][
        ["ano", "desmatamento_km2", "taxa_desmatamento", "area_uc_pct"]
    ].tail(5).to_string())
