"""
Ingestão de autos de infração ambiental do IBAMA por município e ano.

Fonte: Portal de Dados Abertos do IBAMA (dados.gov.br / dadosabertos.ibama.gov.br)
Dataset: 'fiscalizacao-auto-de-infracao'
URL do ZIP: https://dadosabertos.ibama.gov.br/dados/SIFISC/auto_infracao/auto_infracao/auto_infracao_csv.zip

Estratégia:
    1. Baixar o arquivo ZIP com o CSV completo histórico.
    2. Extrair o CSV (encoding latin-1, separador ';').
    3. Identificar colunas de IBGE-code, UF e data por lista de aliases.
    4. Filtrar para as 9 UFs da Amazônia Legal.
    5. Extrair ano a partir da coluna de data.
    6. Agrupar por (cod_ibge, ano) e contar registros → autos_ibama.
    7. Left-join com painel completo (cod_ibge × ano) para garantir zeros.

Output:
    data/raw/ibama_raw.parquet
    - cod_ibge    (str, 7 dígitos)
    - municipio   (str)
    - uf          (str)
    - ano         (int)
    - autos_ibama (int)

Cache local: se output_path já existir, retorna sem recoletar.

Fallback: se o download falhar, cria o parquet com autos_ibama=0 para todos
os municípios e anos disponíveis. O fallback é registrado em log como WARNING.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

_ZIP_URL = (
    "https://dadosabertos.ibama.gov.br/dados/SIFISC/auto_infracao"
    "/auto_infracao/auto_infracao_csv.zip"
)

# UFs da Amazônia Legal
_AMAZONIA_LEGAL_UFS = {"AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"}

# Aliases de nomes de coluna que o IBAMA já usou em diferentes versões do CSV
_IBGE_COL_ALIASES = [
    "MUNICIPIO_IBGE", "MUN_AUTO_IBGE", "COD_MUNICIPIO", "GEOCODIGO_IBGE",
    "COD_IBGE", "IBGE_CODE", "municipio_ibge",
]
_UF_COL_ALIASES = [
    "UF_EMBARGO", "UF_AUTO", "UF", "SGL_UF", "uf_embargo", "uf",
]
_DATE_COL_ALIASES = [
    "DAT_AUTO_INFRACAO", "DAT_HORA_AUTO_INFRACAO", "DATA_AUTO",
    "DT_AUTO", "DAT_INFRACAO", "dat_auto_infracao",
]
_MUN_NAME_ALIASES = [
    "MUNICIPIO_EMBARGO", "MUNICIPIO", "MUN_AUTO", "NOM_MUNICIPIO",
    "municipio_embargo", "municipio",
]


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Retorna o primeiro nome de coluna da lista que existir no DataFrame."""
    for alias in aliases:
        if alias in df.columns:
            return alias
        # Busca case-insensitive como fallback
        match = [c for c in df.columns if c.upper() == alias.upper()]
        if match:
            return match[0]
    return None


def _read_csv_bytes(content: bytes, filename: str) -> pd.DataFrame | None:
    """Tenta ler bytes de um CSV com latin-1, utf-8 e cp1252."""
    for encoding in ("latin-1", "utf-8", "cp1252"):
        try:
            df = pd.read_csv(
                io.BytesIO(content),
                sep=";",
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )
            return df
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            logger.debug("  Erro ao ler %s: %s", filename, exc)
            return None
    return None


def _download_csv() -> pd.DataFrame:
    """
    Baixa o ZIP do IBAMA e lê todos os CSVs internos (um por ano), concatenando-os.

    O ZIP contém arquivos no formato auto_infracao_ano_YYYY.csv.
    Encoding: latin-1. Separador: ';'.
    """
    logger.info("Baixando CSV de autos de infração do IBAMA...")
    resp = requests.get(_ZIP_URL, timeout=300, verify=False, stream=True)
    resp.raise_for_status()

    raw = io.BytesIO(resp.content)
    all_dfs: list[pd.DataFrame] = []
    cols_reference: list[str] | None = None

    with zipfile.ZipFile(raw) as zf:
        csv_names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not csv_names:
            raise RuntimeError("ZIP do IBAMA não contém nenhum arquivo CSV.")

        logger.info("  %d arquivos CSV encontrados no ZIP.", len(csv_names))

        for csv_name in csv_names:
            with zf.open(csv_name) as f:
                content = f.read()

            df_part = _read_csv_bytes(content, csv_name)
            if df_part is None or df_part.empty:
                continue

            if cols_reference is None:
                cols_reference = list(df_part.columns)
                logger.info("  Colunas (primeiro CSV): %s", cols_reference[:10])

            all_dfs.append(df_part)

    if not all_dfs:
        raise RuntimeError("Nenhum CSV pôde ser lido do ZIP do IBAMA.")

    df = pd.concat(all_dfs, ignore_index=True)
    logger.info("  Total após concat: %d registros", len(df))
    return df


def _build_zero_panel(anos_referencia: list[int]) -> pd.DataFrame:
    """Cria painel vazio (autos_ibama=0) a partir do shapefile de municípios."""
    mun_path = Path("data/raw/municipios.gpkg")
    if mun_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(mun_path)
        df_mun = gdf[["cod_ibge", "municipio", "uf"]].copy()
    else:
        ibge_path = Path("data/raw/ibge_raw.parquet")
        if ibge_path.exists():
            df_ibge = pd.read_parquet(ibge_path)
            df_mun = df_ibge[["cod_ibge", "municipio", "uf"]].drop_duplicates("cod_ibge")
        else:
            logger.warning("Sem fonte de municípios disponível para painel zero.")
            df_mun = pd.DataFrame(columns=["cod_ibge", "municipio", "uf"])

    panel = df_mun.assign(key=1).merge(
        pd.DataFrame({"ano": anos_referencia, "key": 1}), on="key"
    ).drop(columns="key")
    panel["autos_ibama"] = 0
    return panel.sort_values(["cod_ibge", "ano"]).reset_index(drop=True)


def collect_ibama(
    output_path: str = "data/raw/ibama_raw.parquet",
    anos_referencia: list[int] | None = None,
) -> pd.DataFrame:
    """
    Coleta autos de infração ambiental do IBAMA por município e ano.

    Baixa o dataset histórico completo de autos de infração do portal de
    dados abertos do IBAMA, filtra para a Amazônia Legal e agrega por
    (município, ano). Municípios sem autos em um dado ano recebem 0.

    Cache local: se output_path já existir, retorna sem recoletar.

    Fallback: se o download falhar, grava um parquet com autos_ibama=0
    e registra WARNING no log. A limitação deve ser anotada no README.

    Args:
        output_path: Caminho do Parquet de saída.
        anos_referencia: Anos para o painel completo. Se None, usa 2005–2025.

    Returns:
        DataFrame com colunas: cod_ibge (str), municipio (str), uf (str),
        ano (int), autos_ibama (int).

    Raises:
        RuntimeError: Apenas se o download falhar E não houver dados de
            municípios disponíveis para o fallback.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem recoletar.", output_path)
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if anos_referencia is None:
        anos_referencia = list(range(2005, 2026))

    # ── Tentativa de download ────────────────────────────────────────────────
    try:
        df_raw = _download_csv()
    except Exception as exc:
        logger.warning(
            "Download IBAMA falhou: %s. Usando fallback com autos_ibama=0.", exc
        )
        df_fallback = _build_zero_panel(anos_referencia)
        df_fallback.to_parquet(path, index=False)
        logger.warning("Fallback salvo em '%s'. autos_ibama=0 para todos os registros.", output_path)
        return df_fallback

    # ── Identificar colunas chave ────────────────────────────────────────────
    col_ibge = _find_column(df_raw, _IBGE_COL_ALIASES)
    col_uf = _find_column(df_raw, _UF_COL_ALIASES)
    col_date = _find_column(df_raw, _DATE_COL_ALIASES)
    col_mun = _find_column(df_raw, _MUN_NAME_ALIASES)

    logger.info(
        "Colunas identificadas — IBGE: %s | UF: %s | data: %s | município: %s",
        col_ibge, col_uf, col_date, col_mun,
    )

    if col_uf is None and col_ibge is None:
        logger.warning(
            "Colunas de UF e IBGE não encontradas. Colunas disponíveis: %s",
            list(df_raw.columns),
        )
        logger.warning("Usando fallback com autos_ibama=0.")
        df_fallback = _build_zero_panel(anos_referencia)
        df_fallback.to_parquet(path, index=False)
        return df_fallback

    # ── Filtrar para Amazônia Legal via UF ──────────────────────────────────
    if col_uf is not None:
        df_amz = df_raw[df_raw[col_uf].str.strip().isin(_AMAZONIA_LEGAL_UFS)].copy()
    else:
        # Sem coluna UF: tentar derivar dos 2 primeiros dígitos do cod IBGE
        _ibge_uf = {
            "11": "RO", "12": "AC", "13": "AM", "14": "RR",
            "15": "PA", "16": "AP", "17": "TO", "21": "MA", "51": "MT",
        }
        df_raw["_uf_deriv"] = df_raw[col_ibge].astype(str).str.zfill(7).str[:2].map(_ibge_uf)
        df_amz = df_raw[df_raw["_uf_deriv"].isin(_AMAZONIA_LEGAL_UFS)].copy()
        col_uf = "_uf_deriv"

    logger.info("  → %d registros na Amazônia Legal (de %d total)", len(df_amz), len(df_raw))

    # ── Extrair ano ──────────────────────────────────────────────────────────
    if col_date is not None:
        # Formatos comuns: DD/MM/YYYY, DD/MM/YYYY HH:MM:SS, YYYY-MM-DD
        date_series = pd.to_datetime(
            df_amz[col_date].astype(str).str[:10],
            format="%d/%m/%Y",
            errors="coerce",
        )
        # Tentar formato ISO se o primeiro falhou
        if date_series.isna().mean() > 0.5:
            date_series = pd.to_datetime(
                df_amz[col_date].astype(str).str[:10],
                format="%Y-%m-%d",
                errors="coerce",
            )
        df_amz["ano"] = date_series.dt.year
    else:
        logger.warning("Coluna de data não encontrada; não é possível extrair ano.")
        df_fallback = _build_zero_panel(anos_referencia)
        df_fallback.to_parquet(path, index=False)
        return df_fallback

    df_amz = df_amz.dropna(subset=["ano"])
    df_amz["ano"] = df_amz["ano"].astype(int)
    logger.info("  → Anos disponíveis: %s", sorted(df_amz["ano"].unique()))

    # ── Normalizar cod_ibge ──────────────────────────────────────────────────
    if col_ibge is not None:
        df_amz["cod_ibge"] = (
            df_amz[col_ibge].astype(str).str.strip()
            .str.replace(r"\D", "", regex=True)  # remover não-dígitos
            .str.zfill(7)
        )
    else:
        # Sem código IBGE: não é possível fazer join preciso
        logger.warning("Coluna de código IBGE não encontrada. Usando fallback com 0.")
        df_fallback = _build_zero_panel(anos_referencia)
        df_fallback.to_parquet(path, index=False)
        return df_fallback

    # Nome e UF do município
    df_amz["uf"] = df_amz[col_uf].str.strip()
    if col_mun is not None:
        df_amz["municipio"] = df_amz[col_mun].str.strip()
    else:
        df_amz["municipio"] = ""

    # ── Agregação ────────────────────────────────────────────────────────────
    logger.info("Agregando autos por (cod_ibge, ano)...")
    agg = (
        df_amz.groupby(["cod_ibge", "municipio", "uf", "ano"], as_index=False)
        .size()
        .rename(columns={"size": "autos_ibama"})
    )
    # Para municípios com nomes duplicados (mesmo cod_ibge, UFs diferentes), somar
    agg = (
        agg.groupby(["cod_ibge", "ano"], as_index=False)
        .agg(
            municipio=("municipio", "first"),
            uf=("uf", "first"),
            autos_ibama=("autos_ibama", "sum"),
        )
    )

    # ── Painel completo (todos os municípios × anos de referência) ───────────
    logger.info("Construindo painel completo com zeros para ausentes...")
    mun_path = Path("data/raw/municipios.gpkg")
    if mun_path.exists():
        import geopandas as gpd
        gdf = gpd.read_file(mun_path)
        df_mun = gdf[["cod_ibge", "municipio", "uf"]].copy()
    else:
        ibge_path = Path("data/raw/ibge_raw.parquet")
        if ibge_path.exists():
            df_ibge = pd.read_parquet(ibge_path)
            df_mun = df_ibge[["cod_ibge", "municipio", "uf"]].drop_duplicates("cod_ibge")
        else:
            df_mun = agg[["cod_ibge", "municipio", "uf"]].drop_duplicates("cod_ibge")

    panel = df_mun.assign(key=1).merge(
        pd.DataFrame({"ano": anos_referencia, "key": 1}), on="key"
    ).drop(columns="key")

    df_result = panel.merge(
        agg[["cod_ibge", "ano", "autos_ibama"]],
        on=["cod_ibge", "ano"],
        how="left",
    )
    df_result["autos_ibama"] = df_result["autos_ibama"].fillna(0).astype(int)
    df_result["cod_ibge"] = df_result["cod_ibge"].astype(str)
    df_result["ano"] = df_result["ano"].astype(int)
    df_result = df_result.sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    # ── Validações ───────────────────────────────────────────────────────────
    n_munic = df_result["cod_ibge"].nunique()
    n_anos = df_result["ano"].nunique()
    total_autos = df_result["autos_ibama"].sum()
    pct_positivo = (df_result["autos_ibama"] > 0).mean() * 100
    logger.info(
        "Validação: %d municípios | %d anos | total autos: %d | registros>0: %.1f%%",
        n_munic, n_anos, total_autos, pct_positivo,
    )
    assert (df_result["autos_ibama"] >= 0).all(), "Há autos_ibama negativo!"

    df_result.to_parquet(path, index=False)
    logger.info("Salvo em '%s'.", output_path)
    return df_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    df = collect_ibama()
    print(f"\nShape: {df.shape}")
    print(f"Municípios: {df['cod_ibge'].nunique()}")
    print(f"Anos: {sorted(df['ano'].unique())}")
    print(f"Total de autos: {df['autos_ibama'].sum():,}")
    print(f"\nTop 10 municípios (total histórico):")
    top = df.groupby(["cod_ibge", "municipio", "uf"])["autos_ibama"].sum().nlargest(10)
    print(top.to_string())
    print(f"\nAmostra:\n{df.head(10).to_string()}")
