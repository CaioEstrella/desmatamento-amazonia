"""
Ingestão de Unidades de Conservação (UC) e Terras Indígenas (TI)
via TerraBrasilis WFS, com interseção espacial por município.

Fontes:
    - prodes-legal-amz:conservation_units_legal_amazon  (≈358 UCs)
    - prodes-legal-amz:indigenous_area_legal_amazon     (≈386 TIs)

Estratégia:
    1. Baixar polígonos de UC e TI via WFS (requisição única, sem paginação).
    2. Carregar limites municipais de data/raw/municipios.gpkg (TASK-03).
       Se o arquivo não existir, executa collect_shapefile() automaticamente.
    3. Reprojetar tudo para EPSG:5880 (SIRGAS 2000 / Brazil Polyconic, em metros).
    4. overlay(how='intersection') para obter fragmentos de sobreposição.
    5. Calcular área de cada fragmento em km².
    6. Agregar SUM(area_uc_km2) e SUM(area_ti_km2) por cod_ibge.
    7. Left-join com a lista completa de municípios para preencher zeros.

Output:
    data/raw/icmbio_raw.parquet
    - cod_ibge      (str, 7 dígitos)
    - area_uc_km2   (float) — área coberta por UCs no município, em km²
    - area_ti_km2   (float) — área coberta por TIs no município, em km²

Cache local: se output_path já existir, retorna sem recoletar.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import urllib3
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

_WFS_BASE = (
    "https://terrabrasilis.dpi.inpe.br/geoserver/prodes-legal-amz/ows"
    "?service=WFS&version=2.0.0"
)
_UC_LAYER = "prodes-legal-amz:conservation_units_legal_amazon"
_TI_LAYER = "prodes-legal-amz:indigenous_area_legal_amazon"

# CRS métrico para cálculo de áreas em todo o Brasil
_CRS_METRICO = "EPSG:5880"  # SIRGAS 2000 / Brazil Polyconic


def _download_layer(layer_name: str, label: str) -> gpd.GeoDataFrame:
    """Baixa uma camada WFS completa e retorna GeoDataFrame."""
    logger.info("Baixando camada %s (%s)...", label, layer_name)
    url = (
        f"{_WFS_BASE}"
        f"&request=GetFeature"
        f"&typeName={layer_name}"
        f"&outputFormat=application/json"
    )
    resp = requests.get(url, timeout=180, verify=False)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    if not features:
        raise RuntimeError(f"WFS retornou 0 features para '{layer_name}'.")

    logger.info("  → %d features recebidas", len(features))
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    n_invalidas = (~gdf.geometry.is_valid).sum()
    if n_invalidas > 0:
        logger.info("  Corrigindo %d geometrias inválidas...", n_invalidas)
        gdf["geometry"] = gdf["geometry"].apply(_fix_geometry)

    # Garantir que tenhamos apenas Polygon/MultiPolygon (make_valid pode gerar
    # GeometryCollection com pontos/linhas em bordas degeneradas)
    gdf["geometry"] = gdf["geometry"].apply(_extract_polygons)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    logger.info("  → %d features com geometria poligonal válida", len(gdf))
    return gdf


def _fix_geometry(g):
    """
    Corrige geometria inválida usando buffer(0), que sempre retorna
    Polygon/MultiPolygon — ao contrário de make_valid, que pode gerar
    GeometryCollection de dimensão mista.
    """
    if g is None or g.is_empty:
        return None
    if g.is_valid:
        return g
    try:
        fixed = g.buffer(0)
        if fixed.is_empty:
            return None
        return fixed
    except Exception:
        return None


def _extract_polygons(g):
    """Extrai apenas as partes Polygon/MultiPolygon de uma geometria qualquer."""
    if g is None or g.is_empty:
        return None
    if isinstance(g, (Polygon, MultiPolygon)):
        # Garantir que o polígono em si é válido
        if not g.is_valid:
            g = g.buffer(0)
        return g if not g.is_empty else None
    # GeometryCollection ou outra geometria mista
    polys = [
        part for part in getattr(g, "geoms", [g])
        if isinstance(part, (Polygon, MultiPolygon))
    ]
    if not polys:
        return None
    result = unary_union(polys)
    if not result.is_valid:
        result = result.buffer(0)
    return result if not result.is_empty else None


def _calc_overlap_km2(
    gdf_municipios: gpd.GeoDataFrame,
    gdf_layer: gpd.GeoDataFrame,
    col_name: str,
) -> pd.DataFrame:
    """
    Calcula a área de sobreposição (em km²) entre municípios e uma camada de polígonos.

    Ambos os GeoDataFrames devem estar no mesmo CRS métrico antes desta chamada.

    Returns:
        DataFrame com colunas: cod_ibge, {col_name}
    """
    logger.info("  Calculando interseção para '%s'...", col_name)

    # Interseção espacial: fragmentos onde município e polígono se sobrepõem
    # make_valid=False: geometrias já foram sanitizadas com buffer(0)
    intersection = gpd.overlay(
        gdf_municipios[["cod_ibge", "geometry"]],
        gdf_layer[["geometry"]],
        how="intersection",
        keep_geom_type=False,
        make_valid=False,
    )

    if intersection.empty:
        logger.warning("  Interseção vazia para '%s'.", col_name)
        return pd.DataFrame(columns=["cod_ibge", col_name])

    # Área em km² (CRS já está em metros → dividir por 1e6)
    intersection[col_name] = intersection.geometry.area / 1e6

    result = (
        intersection.groupby("cod_ibge")[col_name]
        .sum()
        .reset_index()
    )
    logger.info("  → %d municípios com %s > 0", len(result), col_name)
    return result


def collect_icmbio(
    output_path: str = "data/raw/icmbio_raw.parquet",
    municipios_path: str = "data/raw/municipios.gpkg",
) -> pd.DataFrame:
    """
    Coleta áreas de UCs e TIs por município via interseção espacial.

    Baixa os polígonos de Unidades de Conservação e Terras Indígenas do
    TerraBrasilis WFS e calcula a área de sobreposição com cada município
    da Amazônia Legal. Municípios sem cobertura recebem 0.

    Cache local: se output_path já existir, retorna sem recoletar.

    Args:
        output_path: Caminho do Parquet de saída.
        municipios_path: Caminho do GeoPackage de municípios (TASK-03).

    Returns:
        DataFrame com colunas: cod_ibge (str), area_uc_km2 (float),
        area_ti_km2 (float).

    Raises:
        RuntimeError: Se o WFS falhar ou o shapefile de municípios não puder
            ser gerado.
    """
    path = Path(output_path)

    if path.exists():
        logger.info("Cache encontrado em '%s'. Carregando sem recoletar.", output_path)
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Carregar municípios — executa collect_shapefile se necessário
    mun_path = Path(municipios_path)
    if not mun_path.exists():
        logger.info("Shapefile de municípios não encontrado. Executando TASK-03...")
        from src.ingestion.ibge_geo import collect_shapefile
        collect_shapefile(municipios_path)

    logger.info("Carregando municípios de '%s'...", municipios_path)
    gdf_municipios = gpd.read_file(mun_path)
    logger.info("  → %d municípios carregados", len(gdf_municipios))

    # Baixar camadas UC e TI
    gdf_uc = _download_layer(_UC_LAYER, "UCs")
    gdf_ti = _download_layer(_TI_LAYER, "TIs")

    # Reprojetar para CRS métrico
    logger.info("Reprojetando para %s...", _CRS_METRICO)
    gdf_municipios = gdf_municipios.to_crs(_CRS_METRICO)
    gdf_uc = gdf_uc.to_crs(_CRS_METRICO)
    gdf_ti = gdf_ti.to_crs(_CRS_METRICO)

    # gpd.overlay exige tipo homogêneo (somente Polygon/MultiPolygon)
    for _name, _gdf in [("municípios", gdf_municipios), ("UCs", gdf_uc), ("TIs", gdf_ti)]:
        tipos = _gdf.geometry.geom_type.unique()
        if len(tipos) > 1 or any(t not in ("Polygon", "MultiPolygon") for t in tipos):
            logger.info("  Normalizando tipos de geometria em %s: %s", _name, tipos)
    gdf_municipios["geometry"] = gdf_municipios["geometry"].apply(_extract_polygons)
    gdf_uc["geometry"] = gdf_uc["geometry"].apply(_extract_polygons)
    gdf_ti["geometry"] = gdf_ti["geometry"].apply(_extract_polygons)
    gdf_municipios = gdf_municipios[gdf_municipios.geometry.notna() & ~gdf_municipios.geometry.is_empty].copy()
    gdf_uc = gdf_uc[gdf_uc.geometry.notna() & ~gdf_uc.geometry.is_empty].copy()
    gdf_ti = gdf_ti[gdf_ti.geometry.notna() & ~gdf_ti.geometry.is_empty].copy()

    # Calcular sobreposições
    df_uc = _calc_overlap_km2(gdf_municipios, gdf_uc, "area_uc_km2")
    df_ti = _calc_overlap_km2(gdf_municipios, gdf_ti, "area_ti_km2")

    # Montar resultado completo (todos os 805 municípios)
    df_result = gdf_municipios[["cod_ibge"]].copy()
    df_result = (
        df_result
        .merge(df_uc, on="cod_ibge", how="left")
        .merge(df_ti, on="cod_ibge", how="left")
    )
    df_result["area_uc_km2"] = df_result["area_uc_km2"].fillna(0.0)
    df_result["area_ti_km2"] = df_result["area_ti_km2"].fillna(0.0)
    df_result["cod_ibge"] = df_result["cod_ibge"].astype(str)
    df_result = df_result.sort_values("cod_ibge").reset_index(drop=True)

    # Validações
    n_uc = (df_result["area_uc_km2"] > 0).sum()
    n_ti = (df_result["area_ti_km2"] > 0).sum()
    cobertura_pct = len(df_result) / len(gdf_municipios) * 100
    logger.info(
        "Validação: %d municípios | UC>0: %d | TI>0: %d | cobertura: %.1f%%",
        len(df_result), n_uc, n_ti, cobertura_pct,
    )
    assert (df_result["area_uc_km2"] >= 0).all(), "Há área_uc_km2 negativa!"
    assert (df_result["area_ti_km2"] >= 0).all(), "Há área_ti_km2 negativa!"
    assert cobertura_pct >= 80, f"Cobertura insuficiente: {cobertura_pct:.1f}%"

    df_result.to_parquet(path, index=False)
    logger.info("Salvo em '%s'.", output_path)
    return df_result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    df = collect_icmbio()
    print(f"\nShape: {df.shape}")
    print(f"Municípios com UC: {(df['area_uc_km2'] > 0).sum()}")
    print(f"Municípios com TI: {(df['area_ti_km2'] > 0).sum()}")
    print(f"\nTop 10 por área UC:\n{df.nlargest(10, 'area_uc_km2').to_string()}")
