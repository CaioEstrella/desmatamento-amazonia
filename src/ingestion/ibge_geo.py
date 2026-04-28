"""
Ingestão do shapefile de municípios da Amazônia Legal via TerraBrasilis WFS.

Fonte: INPE/TerraBrasilis — camada `prodes-legal-amz:municipalities_legal_amazon`,
que utiliza os limites administrativos oficiais do IBGE (Censo 2010/2022).

Alternativa (não usada): download direto do ZIP IBGE (~200MB) via
`geopandas.read_file('https://geoftp.ibge.gov.br/.../BR_Municipios_2022.zip')`.
A camada WFS já é pré-filtrada para a Amazônia Legal, economizando largura de banda.

Output:
    data/raw/municipios.gpkg
    - cod_ibge  (str, 7 dígitos)
    - municipio (str)
    - uf        (str, 2 letras)
    - geometry  (MultiPolygon, EPSG:4326 / SIRGAS 2000)

Cache local: se `output_path` já existir, retorna sem recoletar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import requests
import urllib3
from shapely.validation import make_valid

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

_WFS_BASE = (
    "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/ows"
    "?service=WFS&version=2.0.0"
)
_MUNICIPALITIES_LAYER = "prodes-legal-amz:municipalities_legal_amazon"

# Mapeamento 2 primeiros dígitos do geocodigo → UF
_IBGE_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR",
    "15": "PA", "16": "AP", "17": "TO", "21": "MA", "51": "MT",
}


def collect_shapefile(output_path: str = "data/raw/municipios.gpkg") -> gpd.GeoDataFrame:
    """
    Coleta o shapefile de municípios da Amazônia Legal e salva como GeoPackage.

    Baixa as geometrias poligonais dos 805 municípios da Amazônia Legal a partir
    da camada WFS do TerraBrasilis (limites IBGE). O CRS é EPSG:4326 (SIRGAS 2000).
    Geometrias inválidas são corrigidas com `shapely.validation.make_valid`.

    Cache local: se `output_path` já existir, retorna sem recoletar.

    Args:
        output_path: Caminho do GeoPackage de saída.

    Returns:
        GeoDataFrame com colunas: cod_ibge (str), municipio (str),
        uf (str), geometry (MultiPolygon).

    Raises:
        RuntimeError: Se o download WFS falhar ou não retornar geometrias.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem recoletar.", output_path)
        return gpd.read_file(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Baixando municípios da Amazônia Legal (WFS TerraBrasilis)...")
    url = (
        f"{_WFS_BASE}"
        f"&request=GetFeature"
        f"&typeName={_MUNICIPALITIES_LAYER}"
        f"&outputFormat=application/json"
    )
    resp = requests.get(url, timeout=120, verify=False)
    resp.raise_for_status()
    data = resp.json()
    features = data.get("features", [])

    if not features:
        raise RuntimeError("WFS retornou 0 features para municipalities_legal_amazon.")

    logger.info("  → %d features recebidas", len(features))

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf = gdf.rename(columns={"nome": "municipio"})
    gdf["cod_ibge"] = gdf["geocodigo"].astype(str)
    gdf["uf"] = gdf["cod_ibge"].str[:2].map(_IBGE_UF).fillna("??")

    # Corrigir geometrias inválidas (self-intersections, etc.)
    n_invalidas = (~gdf.geometry.is_valid).sum()
    if n_invalidas > 0:
        logger.info("  Corrigindo %d geometrias inválidas com make_valid...", n_invalidas)
        gdf["geometry"] = gdf["geometry"].apply(
            lambda g: make_valid(g) if g is not None and not g.is_valid else g
        )

    gdf = gdf[["cod_ibge", "municipio", "uf", "geometry"]].copy()
    gdf = gdf.sort_values("cod_ibge").reset_index(drop=True)

    # Validações
    n_nulos_geom = gdf.geometry.isna().sum()
    n_invalidas_final = (~gdf.geometry.is_valid).sum()
    logger.info(
        "Validação: %d municípios | geom nulas: %d | geom inválidas: %d | CRS: %s",
        len(gdf), n_nulos_geom, n_invalidas_final, gdf.crs,
    )
    assert n_nulos_geom == 0, "Há municípios sem geometria!"

    # Salvar como GeoPackage
    gdf.to_file(path, driver="GPKG")
    logger.info("Salvo em '%s'.", output_path)

    return gdf


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    gdf = collect_shapefile()
    print(f"\nShape: {gdf.shape}")
    print(f"CRS: {gdf.crs}")
    print(f"UFs: {sorted(gdf['uf'].unique())}")
    print(f"\nAmostra:\n{gdf.head(5)[['cod_ibge','municipio','uf']].to_string()}")
