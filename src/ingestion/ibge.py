"""
Ingestão de dados socioeconômicos municipais via API IBGE Servicodados.

Fontes:
    - Localidades: https://servicodados.ibge.gov.br/api/v1/localidades/
    - Agregados:   https://servicodados.ibge.gov.br/api/v3/agregados/

Variáveis coletadas:
    - populacao       : Agregado 6579, Variável 9324 (estimativas populacionais)
    - pib_agropecuario: Agregado 5938, Variável 513  (VAB agropecuária, R$ 1.000)
    - area_km2        : Agregado 1301, Variável 615  (área territorial, km²)

Cobertura temporal:
    - populacao:        2001–2025 (interpolação linear para anos ausentes)
    - pib_agropecuario: 2002–2023
    - area_km2:         valor único (2010), propagado para todos os anos

Cache local: se `output_path` já existir, retorna sem chamar a API.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
_IBGE_BASE = "https://servicodados.ibge.gov.br/api"
_LOCALIDADES_URL = f"{_IBGE_BASE}/v1/localidades"
_AGREGADOS_URL = f"{_IBGE_BASE}/v3/agregados"

# Amazônia Legal: 9 UFs e seus códigos IBGE de estado
_AMAZONIA_LEGAL = {
    "RO": "11", "AC": "12", "AM": "13", "RR": "14",
    "PA": "15", "AP": "16", "TO": "17", "MA": "21", "MT": "51",
}

# Agregados IBGE
_AGG_PIB = 5938        # PIB Municipal
_VAR_PIB_AGRO = 513    # Valor adicionado bruto da agropecuária (R$ 1.000)
_AGG_POP = 6579        # Estimativas populacionais
_VAR_POP = 9324        # População residente estimada
_AGG_AREA = 1301       # Área territorial
_VAR_AREA = 615        # Área (km²)

_MAX_RETRIES = 3


# ── Utilitários HTTP ─────────────────────────────────────────────────────────

def _get_json(url: str, retries: int = _MAX_RETRIES) -> list | dict:
    """GET com retry/backoff. Retorna JSON ou levanta RuntimeError."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120, verify=False)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"IBGE API falhou ({url}): {exc}") from exc
            wait = 2 ** attempt
            logger.warning("Tentativa %d/%d falhou. Aguardando %ds...", attempt + 1, retries, wait)
            time.sleep(wait)


# ── Coleta de listas base ────────────────────────────────────────────────────

def _get_municipios() -> pd.DataFrame:
    """Retorna DataFrame com todos os municípios da Amazônia Legal."""
    rows = []
    for uf, cod_estado in _AMAZONIA_LEGAL.items():
        url = f"{_LOCALIDADES_URL}/estados/{cod_estado}/municipios"
        municipios = _get_json(url)
        for m in municipios:
            rows.append({
                "cod_ibge": str(m["id"]),
                "municipio": m["nome"],
                "uf": uf,
            })
    df = pd.DataFrame(rows)
    logger.info("  Municípios encontrados: %d", len(df))
    return df


# ── Coleta de agregados por estado ───────────────────────────────────────────

def _parse_aggregate(data: list) -> pd.DataFrame:
    """
    Converte resposta da API de agregados em DataFrame tidy.

    Returns:
        DataFrame com colunas: cod_ibge, ano, valor (float)
    """
    rows = []
    for variavel in data:
        for resultado in variavel.get("resultados", []):
            for serie in resultado.get("series", []):
                loc_id = str(serie["localidade"]["id"])
                for ano_str, valor_str in serie.get("serie", {}).items():
                    try:
                        valor = float(str(valor_str).replace(",", "."))
                    except (ValueError, TypeError):
                        valor = np.nan
                    rows.append({"cod_ibge": loc_id, "ano": int(ano_str), "valor": valor})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["cod_ibge", "ano", "valor"])


def _fetch_aggregate_for_amazon(
    agregado_id: int,
    variavel_id: int,
    periodos: str,
    nome: str,
) -> pd.DataFrame:
    """
    Baixa um agregado IBGE para todos os municípios da Amazônia Legal.

    Faz uma requisição por estado para evitar timeouts (9 chamadas).
    """
    logger.info("  Baixando %s (agg=%d, var=%d)...", nome, agregado_id, variavel_id)
    all_dfs = []
    for uf, cod_estado in _AMAZONIA_LEGAL.items():
        url = (
            f"{_AGREGADOS_URL}/{agregado_id}"
            f"/periodos/{periodos}"
            f"/variaveis/{variavel_id}"
            f"?localidades=N6[N3[{cod_estado}]]"
        )
        data = _get_json(url)
        df_uf = _parse_aggregate(data)
        all_dfs.append(df_uf)

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    logger.info("    → %d registros (cod_ibge×ano)", len(df))
    return df


# ── Tratamento de série temporal ─────────────────────────────────────────────

def _align_to_inpe_years(
    df_wide: pd.DataFrame,
    anos_inpe: list[int],
    value_col: str,
    method: str = "linear",
) -> pd.DataFrame:
    """
    Expande/interpola os dados para cobrir os anos do dataset INPE (2008–2025).

    Para anos ausentes na série IBGE, usa interpolação linear por município.
    Para projeção além do último ano disponível (extrapolação), repete o último valor.

    Args:
        df_wide: DataFrame com cod_ibge como index e anos como colunas.
        anos_inpe: Lista de anos alvo.
        value_col: Nome da coluna de valor.
        method: Método de interpolação pandas.

    Returns:
        DataFrame tidy com colunas: cod_ibge, ano, {value_col}
    """
    # Garantir que os anos alvo existam como colunas (com NaN se ausentes)
    for ano in anos_inpe:
        if ano not in df_wide.columns:
            df_wide[ano] = np.nan

    df_sorted = df_wide[sorted(df_wide.columns)]

    # Interpolação linear ao longo dos anos (axis=1)
    df_interp = df_sorted.interpolate(method=method, axis=1, limit_direction="both")

    # Empilhar (wide → tidy)
    df_tidy = (
        df_interp[anos_inpe]
        .stack()
        .reset_index()
    )
    df_tidy.columns = ["cod_ibge", "ano", value_col]
    df_tidy["ano"] = df_tidy["ano"].astype(int)
    return df_tidy


# ── Ponto de entrada ─────────────────────────────────────────────────────────

def collect_ibge(
    output_path: str = "data/raw/ibge_raw.parquet",
    anos_referencia: list[int] | None = None,
) -> pd.DataFrame:
    """
    Coleta dados socioeconômicos municipais (PIB agropecuário, população, área).

    Consulta a API IBGE Servicodados para os municípios da Amazônia Legal.
    Realiza interpolação linear para preencher anos ausentes na série histórica.

    Cache local: se `output_path` já existir, retorna sem chamar a API.

    Args:
        output_path: Caminho do arquivo Parquet de cache.
        anos_referencia: Lista de anos para alinhar os dados. Se None, usa
            os mesmos anos do dataset INPE (2008–2025).

    Returns:
        DataFrame com colunas: cod_ibge (str), municipio (str), uf (str),
        ano (int), populacao (float), pib_agropecuario (float), area_km2 (float).

    Raises:
        RuntimeError: Se a API IBGE falhar após todas as tentativas de retry.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem chamar a API.", output_path)
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if anos_referencia is None:
        anos_referencia = list(range(2008, 2026))

    # 1. Base de municípios
    logger.info("Buscando municípios da Amazônia Legal...")
    df_munic = _get_municipios()

    # 2. PIB agropecuário (anos 2002–2023)
    logger.info("Coletando PIB agropecuário...")
    periodos_pib = "|".join(str(y) for y in range(2002, 2024))
    df_pib_raw = _fetch_aggregate_for_amazon(
        _AGG_PIB, _VAR_PIB_AGRO, periodos_pib, "PIB agropecuário"
    )
    df_pib_wide = (
        df_pib_raw.pivot_table(index="cod_ibge", columns="ano", values="valor", aggfunc="first")
    )
    df_pib = _align_to_inpe_years(df_pib_wide, anos_referencia, "pib_agropecuario")

    # 3. População (anos 2001–2025)
    logger.info("Coletando população...")
    periodos_pop = "|".join(str(y) for y in range(2001, 2026))
    df_pop_raw = _fetch_aggregate_for_amazon(
        _AGG_POP, _VAR_POP, periodos_pop, "população"
    )
    df_pop_wide = (
        df_pop_raw.pivot_table(index="cod_ibge", columns="ano", values="valor", aggfunc="first")
    )
    df_pop = _align_to_inpe_years(df_pop_wide, anos_referencia, "populacao")

    # 4. Área territorial (valor único de 2010, propagado)
    logger.info("Coletando área territorial...")
    df_area_raw = _fetch_aggregate_for_amazon(
        _AGG_AREA, _VAR_AREA, "2010", "área"
    )
    # Área é estática: pegar o valor de 2010 e replicar para todos os anos
    df_area_static = (
        df_area_raw.groupby("cod_ibge")["valor"]
        .first()
        .reset_index()
        .rename(columns={"valor": "area_km2"})
    )

    # 5. Montar painel base (todos os municípios × todos os anos)
    logger.info("Construindo painel base...")
    df_panel = df_munic.assign(key=1).merge(
        pd.DataFrame({"ano": anos_referencia, "key": 1}), on="key"
    ).drop(columns="key")

    # 6. Juntar todas as variáveis
    df_result = (
        df_panel
        .merge(df_pop[["cod_ibge", "ano", "populacao"]], on=["cod_ibge", "ano"], how="left")
        .merge(df_pib[["cod_ibge", "ano", "pib_agropecuario"]], on=["cod_ibge", "ano"], how="left")
        .merge(df_area_static[["cod_ibge", "area_km2"]], on="cod_ibge", how="left")
    )

    df_result["cod_ibge"] = df_result["cod_ibge"].astype(str)
    df_result["ano"] = df_result["ano"].astype(int)
    df_result = df_result.sort_values(["cod_ibge", "ano"]).reset_index(drop=True)

    # 7. Validações
    n_munic = df_result["cod_ibge"].nunique()
    pct_pop_ok = df_result["populacao"].notna().mean() * 100
    pct_area_ok = df_result["area_km2"].notna().mean() * 100
    n_area_zero = (df_result["area_km2"] == 0).sum()
    logger.info(
        "Validação: %d municípios | pop preench.: %.1f%% | area preench.: %.1f%% | area_zero: %d",
        n_munic, pct_pop_ok, pct_area_ok, n_area_zero,
    )

    # 8. Salvar
    df_result.to_parquet(path, index=False)
    logger.info("Salvo em '%s'.", output_path)
    return df_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    df = collect_ibge()
    print(f"\nShape: {df.shape}")
    print(f"Municípios: {df['cod_ibge'].nunique()}")
    print(f"Anos: {sorted(df['ano'].unique())}")
    print(f"\nMissing por coluna:\n{df.isnull().mean().to_string()}")
    print(f"\nAmostra:\n{df.head(5).to_string()}")
