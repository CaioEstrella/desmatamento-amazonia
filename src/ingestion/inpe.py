"""
Ingestão de dados PRODES/INPE via TerraBrasilis WFS (OGC Web Feature Service).

Fonte: TerraBrasilis — https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/
Cobertura: Amazônia Legal, anos 2008–atual (resolução anual, corte raso).

Estratégia de coleta:
    1. Baixa polígonos de desmatamento (`yearly_deforestation`) sem geometria — apenas
       propriedades + bbox — usando paginação de 10.000 registros por requisição.
    2. Baixa geometrias dos 805 municípios da Amazônia Legal (`municipalities_legal_amazon`).
    3. Calcula centroide de cada polígono de desmatamento a partir do bbox.
    4. Faz join espacial (centroide → município) via GeoPandas.sjoin.
    5. Agrega SUM(area_km) por (geocodigo, year) → `desmatamento_km2`.
    6. Salva resultado em `data/raw/inpe_raw.parquet`.

Cache local: se `output_path` já existir, retorna os dados sem chamar o WFS.
Retry: até 3 tentativas com backoff exponencial em caso de falha HTTP.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import urllib3
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point

# O GeoServer do TerraBrasilis usa certificado auto-assinado em cadeia;
# desabilitar verificação para evitar SSLError intermitente.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# ── Configurações da API ────────────────────────────────────────────────────
_WFS_BASE = (
    "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/ows"
    "?service=WFS&version=2.0.0"
)
_DEFORESTATION_LAYER = "prodes-legal-amz:yearly_deforestation"
_MUNICIPALITIES_LAYER = "prodes-legal-amz:municipalities_legal_amazon"

_DEFORESTATION_PROPS = "uid,state,year,area_km"  # sem geometria → bbox apenas
_PAGE_SIZE = 10_000
_MAX_RETRIES = 3

# Siglas das UFs da Amazônia Legal
_AMAZONIA_LEGAL_UFS = ["AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"]

# Mapeamento sigla → nome completo (usado como filtro CQL no WFS)
_UF_NAMES = {
    "AC": "Acre",
    "AM": "Amazonas",
    "AP": "Amapá",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "PA": "Pará",
    "RO": "Rondônia",
    "RR": "Roraima",
    "TO": "Tocantins",
}


# ── Utilitários de HTTP ─────────────────────────────────────────────────────

def _get_json(url: str, retries: int = _MAX_RETRIES) -> Any:
    """GET com retry/backoff exponencial. Retorna JSON ou levanta RuntimeError."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120, verify=False)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"WFS falhou após {retries} tentativas: {exc}") from exc
            wait = 2 ** attempt
            logger.warning("Tentativa %d/%d falhou (%s). Aguardando %ds...",
                           attempt + 1, retries, exc, wait)
            time.sleep(wait)


# ── Coleta de dados ─────────────────────────────────────────────────────────

def _download_municipalities() -> gpd.GeoDataFrame:
    """Baixa geometrias dos municípios da Amazônia Legal com geocodigo (cod_ibge)."""
    logger.info("Baixando municípios da Amazônia Legal...")
    # Não especificar propertyName → WFS retorna todas as colunas, incluindo geom
    url = (
        f"{_WFS_BASE}"
        f"&request=GetFeature"
        f"&typeName={_MUNICIPALITIES_LAYER}"
        f"&outputFormat=application/json"
    )
    data = _get_json(url)
    features = data.get("features", [])
    logger.info("  → %d municípios recebidos", len(features))

    # gpd.GeoDataFrame.from_features aceita lista de GeoJSON features diretamente
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf = gdf.rename(columns={"nome": "municipio"})
    gdf["geocodigo"] = gdf["geocodigo"].astype(str)

    # Derivar UF a partir dos 2 primeiros dígitos do código IBGE
    _ibge_uf = {
        "11": "RO", "12": "AC", "13": "AM", "14": "RR",
        "15": "PA", "16": "AP", "17": "TO", "21": "MA", "51": "MT",
    }
    gdf["uf"] = gdf["geocodigo"].str[:2].map(_ibge_uf).fillna("??")

    # Corrigir geometrias inválidas (self-intersections etc.)
    from shapely.validation import make_valid
    gdf["geometry"] = gdf["geometry"].apply(
        lambda g: make_valid(g) if g is not None and not g.is_valid else g
    )
    return gdf[["geocodigo", "municipio", "uf", "geometry"]].copy()


def _download_deforestation_page(startindex: int) -> list[dict]:
    """Baixa uma página de polígonos de desmatamento sem geometria."""
    url = (
        f"{_WFS_BASE}"
        f"&request=GetFeature"
        f"&typeName={_DEFORESTATION_LAYER}"
        f"&outputFormat=application/json"
        f"&propertyName={_DEFORESTATION_PROPS}"
        f"&count={_PAGE_SIZE}"
        f"&STARTINDEX={startindex}"
    )
    data = _get_json(url)
    return data.get("features", [])


def _download_all_deforestation() -> pd.DataFrame:
    """
    Baixa todos os polígonos de desmatamento com paginação.

    Cada registro tem: uid, state, year, area_km + bbox.
    O bbox (formato [minlat, minlon, maxlat, maxlon]) é usado para calcular
    o centroide de cada polígono, que serve como representante espacial
    no join com os municípios.

    Returns:
        DataFrame com colunas: uid, state, year, area_km, lat, lon
    """
    logger.info("Baixando polígonos de desmatamento (paginação de %d)...", _PAGE_SIZE)
    all_records: list[dict] = []
    startindex = 0

    while True:
        logger.info("  Página %d (offset=%d)...", startindex // _PAGE_SIZE + 1, startindex)
        features = _download_deforestation_page(startindex)
        if not features:
            break

        for f in features:
            props = f.get("properties", {})
            bbox = f.get("bbox")
            if bbox is None or len(bbox) < 4:
                continue
            # Formato do bbox neste servidor: [minlat, minlon, maxlat, maxlon]
            minlat, minlon, maxlat, maxlon = bbox
            all_records.append({
                "uid": props.get("uid"),
                "state": props.get("state"),
                "year": props.get("year"),
                "area_km": props.get("area_km"),
                # Centroide do bbox
                "lat": (minlat + maxlat) / 2.0,
                "lon": (minlon + maxlon) / 2.0,
            })

        if len(features) < _PAGE_SIZE:
            break  # última página
        startindex += _PAGE_SIZE

    logger.info("  → %d polígonos baixados no total", len(all_records))
    return pd.DataFrame(all_records)


# ── Join espacial e agregação ───────────────────────────────────────────────

def _assign_municipalities(
    df_deforestation: pd.DataFrame,
    gdf_municipios: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Associa cada polígono de desmatamento a um município via join espacial.

    Usa o centroide do bbox (lat/lon) como proxy do polígono de desmatamento.
    Faz point-in-polygon com as geometrias reais dos municípios (sjoin nearest).

    Returns:
        DataFrame com coluna 'geocodigo' adicionada.
    """
    logger.info("Criando GeoDataFrame de centroides (%d pontos)...", len(df_deforestation))
    geom_series = [
        Point(row.lon, row.lat)  # Shapely: Point(x=lon, y=lat)
        for row in df_deforestation.itertuples(index=False)
    ]
    gdf_centroids = gpd.GeoDataFrame(
        df_deforestation.copy(),
        geometry=geom_series,
        crs="EPSG:4326",
    )

    logger.info("Executando join espacial centroide → município...")
    # predicate='within': cada centroide cai dentro de um município
    # Polígonos na fronteira de dois municípios são ignorados; sjoin nearest os pega
    joined = gpd.sjoin(
        gdf_centroids,
        gdf_municipios[["geocodigo", "municipio", "uf", "geometry"]],
        how="left",
        predicate="within",
    )

    # Para centroides que não caíram dentro de nenhum município (fronteiras/ilhas),
    # usar o município mais próximo via nearest join
    orphans_mask = joined["geocodigo"].isna()
    n_orphans = orphans_mask.sum()
    if n_orphans > 0:
        logger.info("  %d centroides sem município; aplicando nearest join...", n_orphans)
        orphans = gdf_centroids[orphans_mask].copy()
        nearest = gpd.sjoin_nearest(
            orphans,
            gdf_municipios[["geocodigo", "municipio", "uf", "geometry"]],
            how="left",
        )
        joined.loc[orphans_mask, ["geocodigo", "municipio", "uf"]] = (
            nearest[["geocodigo", "municipio", "uf"]].values
        )

    logger.info(
        "  → Cobertura de municípios: %.1f%%",
        (1 - joined["geocodigo"].isna().mean()) * 100,
    )
    return joined[["geocodigo", "municipio", "uf", "year", "area_km"]].copy()


def _aggregate_by_municipality(
    df: pd.DataFrame,
    gdf_municipios: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Agrega área desmatada por município e ano, incluindo todos os municípios.

    Municípios sem eventos de desmatamento em um dado ano recebem desmatamento_km2=0.
    Isso garante que o painel seja completo (todos os 805 municípios × todos os anos).
    """
    logger.info("Agregando por (geocodigo, ano)...")
    agg = (
        df.dropna(subset=["geocodigo"])
        .groupby(["geocodigo", "municipio", "uf", "year"], as_index=False)
        ["area_km"]
        .sum()
        .rename(columns={"year": "ano", "area_km": "desmatamento_km2"})
    )

    # Expandir para painel completo: todos os municípios × todos os anos
    all_years = sorted(agg["ano"].unique())
    all_munic = gdf_municipios[["geocodigo", "municipio", "uf"]].copy()
    panel = all_munic.assign(key=1).merge(
        pd.DataFrame({"ano": all_years, "key": 1}),
        on="key",
    ).drop(columns="key")

    result = panel.merge(
        agg[["geocodigo", "ano", "desmatamento_km2"]],
        on=["geocodigo", "ano"],
        how="left",
    )
    result["desmatamento_km2"] = result["desmatamento_km2"].fillna(0.0)
    result["geocodigo"] = result["geocodigo"].astype(str)
    result["ano"] = result["ano"].astype(int)
    result = result.sort_values(["geocodigo", "ano"]).reset_index(drop=True)
    logger.info("  → %d registros (municípios × anos)", len(result))
    return result


# ── Ponto de entrada ────────────────────────────────────────────────────────

def collect_inpe(output_path: str = "data/raw/inpe_raw.parquet") -> pd.DataFrame:
    """
    Coleta dados históricos de desmatamento PRODES por município e ano.

    Utiliza o WFS do TerraBrasilis (INPE) para obter dados anuais consolidados
    de desmatamento por corte raso (PRODES) para os municípios da Amazônia Legal.
    Cobertura temporal: 2008–ano corrente (limite da série anual disponível via WFS).

    Cache local: se `output_path` já existir, retorna os dados sem realizar novas
    chamadas à API.

    Args:
        output_path: Caminho para salvar/ler o arquivo Parquet de cache.

    Returns:
        DataFrame com colunas: cod_ibge (str), municipio (str), uf (str),
        ano (int), desmatamento_km2 (float). Uma linha por (município, ano).

    Raises:
        RuntimeError: Se a API WFS falhar após todas as tentativas de retry.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem chamar a API.", output_path)
        df = pd.read_parquet(path)
        return df

    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Geometrias dos municípios
    gdf_municipios = _download_municipalities()

    # 2. Polígonos de desmatamento (sem geometria, apenas bbox)
    df_deforestation = _download_all_deforestation()

    # 3. Join espacial
    df_joined = _assign_municipalities(df_deforestation, gdf_municipios)

    # 4. Agregação por município × ano (painel completo com zeros)
    df_result = _aggregate_by_municipality(df_joined, gdf_municipios)
    df_result = df_result.rename(columns={"geocodigo": "cod_ibge"})

    # 5. Validações básicas
    assert (df_result["desmatamento_km2"] >= 0).all(), "Há valores negativos em desmatamento_km2!"
    n_municipios = df_result["cod_ibge"].nunique()
    n_anos = df_result["ano"].nunique()
    logger.info(
        "Validação: %d municípios, %d anos distintos (mín: %d, máx: %d)",
        n_municipios, n_anos,
        df_result["ano"].min(), df_result["ano"].max(),
    )

    # 6. Salvar como Parquet
    df_result.to_parquet(path, index=False)
    logger.info("Salvo em '%s'.", output_path)

    return df_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    df = collect_inpe()
    print(f"\nResultado: {len(df):,} registros")
    print(f"Municípios únicos: {df['cod_ibge'].nunique()}")
    print(f"Anos disponíveis: {sorted(df['ano'].unique())}")
    print(f"\nAmostra:\n{df.head(10).to_string()}")
