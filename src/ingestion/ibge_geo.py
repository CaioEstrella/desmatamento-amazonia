"""
Ingestão do shapefile de municípios da Amazônia Legal via IBGE FTP.

Fonte: IBGE — `BR_Municipios_2022.zip` (malhas municipais nacionais 2022).
O arquivo nacional é baixado e filtrado pelas 9 UFs da Amazônia Legal,
garantindo geometrias completas mesmo para municípios que cruzam o limite
da Amazônia Legal (ex.: municípios do MA).

Nota: a camada WFS `prodes-legal-amz:municipalities_legal_amazon` do
TerraBrasilis foi descartada pois realiza clipping server-side nas
geometrias, cortando municípios na fronteira -44°.

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
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

_IBGE_MUNICIPIOS_URL = (
    "https://geoftp.ibge.gov.br/organizacao_do_territorio"
    "/malhas_territoriais/malhas_municipais/municipio_2022"
    "/Brasil/BR/BR_Municipios_2022.zip"
)

_AMAZONIA_LEGAL_UFS = {"AC", "AM", "AP", "MA", "MT", "PA", "RO", "RR", "TO"}

# Mapeamento 2 primeiros dígitos do geocódigo → UF (fallback se coluna UF ausente)
_IBGE_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR",
    "15": "PA", "16": "AP", "17": "TO", "21": "MA", "51": "MT",
}


def collect_shapefile(output_path: str = "data/raw/municipios.gpkg") -> gpd.GeoDataFrame:
    """
    Coleta o shapefile de municípios da Amazônia Legal e salva como GeoPackage.

    Baixa o shapefile nacional do IBGE (BR_Municipios_2022.zip) e filtra
    pelas 9 UFs da Amazônia Legal. Geometrias inválidas são corrigidas com
    `shapely.validation.make_valid`. O CRS é EPSG:4326 (SIRGAS 2000).

    Cache local: se `output_path` já existir, retorna sem recoletar.

    Args:
        output_path: Caminho do GeoPackage de saída.

    Returns:
        GeoDataFrame com colunas: cod_ibge (str), municipio (str),
        uf (str), geometry (MultiPolygon).

    Raises:
        RuntimeError: Se o download falhar ou não retornar geometrias.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem recoletar.", output_path)
        return gpd.read_file(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Baixando BR_Municipios_2022.zip do IBGE FTP (~200 MB)...")
    gdf = gpd.read_file(_IBGE_MUNICIPIOS_URL)
    logger.info("  → %d municípios no shapefile nacional", len(gdf))

    # Normalizar nomes de colunas (IBGE 2022 usa CD_MUN, NM_MUN, SIGLA_UF)
    col_map: dict[str, str] = {}
    for c in gdf.columns:
        cu = c.upper()
        if cu in ("CD_MUN", "CD_GEOCMU", "GEOCODIGO", "COD_IBGE"):
            col_map[c] = "cod_ibge"
        elif cu in ("NM_MUN", "NOME", "MUNICIPIO", "NM_MUNICIP"):
            col_map[c] = "municipio"
        elif cu in ("SIGLA_UF", "UF", "SIGLA", "NM_UF"):
            col_map[c] = "uf"
    if col_map:
        gdf = gdf.rename(columns=col_map)
    logger.info("  Colunas mapeadas: %s", col_map if col_map else "(nenhuma renomeação)")

    # Derivar UF dos 2 primeiros dígitos do geocódigo caso coluna ausente
    if "uf" not in gdf.columns:
        logger.info("  Coluna 'uf' ausente; derivando dos 2 primeiros dígitos do geocódigo...")
        gdf["uf"] = gdf["cod_ibge"].astype(str).str[:2].map(_IBGE_UF).fillna("??")

    gdf["cod_ibge"] = gdf["cod_ibge"].astype(str).str.strip().str.zfill(7)

    # Reprojetar para EPSG:4326 se necessário
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        logger.info("  Reprojetando %s → EPSG:4326...", gdf.crs.to_string())
        gdf = gdf.to_crs("EPSG:4326")

    # Filtrar somente as 9 UFs da Amazônia Legal
    gdf = gdf[gdf["uf"].isin(_AMAZONIA_LEGAL_UFS)].copy()
    logger.info("  → %d municípios após filtro Amazônia Legal (9 UFs)", len(gdf))

    if len(gdf) == 0:
        raise RuntimeError(
            "Nenhum município encontrado para as UFs da Amazônia Legal. "
            "Verifique o mapeamento de colunas no shapefile IBGE."
        )

    # Corrigir geometrias inválidas
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
    print(f"\nMunicípios MA (bounds para verificar clipping):")
    ma = gdf[gdf["uf"] == "MA"]
    print(f"  Total MA: {len(ma)}")
    print(f"  maxx máximo: {ma.bounds['maxx'].max():.4f}°")
    print(f"  maxx mínimo: {ma.bounds['maxx'].min():.4f}°")
    print(f"\nAmostra:\n{gdf.head(5)[['cod_ibge','municipio','uf']].to_string()}")
