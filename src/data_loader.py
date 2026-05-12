"""
Modulo para descarga de datos desde NYC Open Data y OpenStreetMap.
"""

import json
import logging
import time
import requests
import pandas as pd
from pathlib import Path
from shapely.geometry import box as shapely_box, shape

# ── Rutas ─────────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_RAW.mkdir(parents=True, exist_ok=True)

# ── NYC Open Data ─────────────────────────────────────────────────────────────
URL_NYC = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"
BATCH = 1_000


def descargar_accidentes_manhattan(
    anios: list[int] | None = None,
    limite_total: int = 10_000,
    forzar: bool = False,
) -> pd.DataFrame:
    """
    Descarga accidentes de transito en Manhattan para uno o varios años.
    Combina los CSV cacheados de cada año para no repetir descargas.

    Args:
        anios: lista de años a descargar. Por defecto [2024, 2023].
        limite_total: maximo de registros en total (sumados todos los años).
        forzar: si True, re-descarga aunque exista el cache.

    Returns:
        DataFrame combinado con las columnas seleccionadas.
    """
    if anios is None:
        anios = [2024, 2023]

    frames = []
    por_anio = (limite_total // len(anios)) + 1  # distribuir entre años

    for anio in anios:
        ruta_cache = DIR_RAW / f"accidentes_manhattan_{anio}.csv"

        if ruta_cache.exists() and not forzar:
            print(f"[cache] {anio}: cargando desde {ruta_cache.name}")
            frames.append(pd.read_csv(ruta_cache))
            continue

        print(f"[NYC API] Descargando hasta {por_anio:,} accidentes de Manhattan ({anio})...")
        registros = []
        offset = 0

        while len(registros) < por_anio:
            batch_size = min(BATCH, por_anio - len(registros))
            params = {
                "$where": (
                    f"borough='MANHATTAN' "
                    f"AND latitude IS NOT NULL "
                    f"AND date_extract_y(crash_date)={anio}"
                ),
                "$limit": batch_size,
                "$offset": offset,
                "$select": "crash_date,crash_time,borough,latitude,longitude,"
                            "number_of_persons_injured,number_of_persons_killed",
            }
            respuesta = requests.get(URL_NYC, params=params, timeout=30)
            respuesta.raise_for_status()
            lote = respuesta.json()

            if not lote:
                print(f"  [fin] No hay mas datos en offset={offset}")
                break

            registros.extend(lote)
            offset += len(lote)
            print(f"  >> {anio}: {len(registros):,} descargados", end="\r")
            time.sleep(0.1)

        df_anio = pd.DataFrame(registros)
        df_anio.to_csv(ruta_cache, index=False)
        print(f"\n[NYC API] Guardado {ruta_cache.name}  ({len(df_anio):,} filas)")
        frames.append(df_anio)

    df = pd.concat(frames, ignore_index=True)
    print(f"[datos] Total combinado: {len(df):,} filas de {anios}")
    return df


# ── Geometría de Manhattan (fuente OSM real) ──────────────────────────────────
# bbox real de la isla: lon [-74.0472, -73.9068], lat [40.6797, 40.8820]
# Derivado de ox.geocode_to_gdf("Manhattan, New York City, USA").bounds
# Los ríos (Hudson, East River, Harlem River) son el exterior de este bbox:
# no se modelan como zonas prohibidas internas sino como los 4 semiplanos
# que definen la region factible R (ver src/constraints.py).
BBOX_MANHATTAN_WGS84 = (-74.0472, 40.6797, -73.9068, 40.8820)  # (lon_min, lat_min, lon_max, lat_max)

# Zona prohibida interna: Central Park
# Su bbox proviene de OSM real (ox.geocode_to_gdf("Central Park, ...").bounds)
BBOX_CENTRAL_PARK_WGS84 = (-73.9814, 40.7647, -73.9496, 40.8003)


def obtener_zonas_osm(forzar: bool = False) -> dict:
    """
    Devuelve un dict con dos entradas:
      - "region_factible": Polygon (bbox de Manhattan) — donde el centro PUEDE estar.
      - "central_park":    Polygon (bbox del parque)   — zona prohibida interna.

    Intenta obtener ambas desde OSM real; si falla usa los valores derivados
    previamente de OSM y guardados en las constantes de este modulo.
    Guarda en data/raw/zonas_prohibidas.geojson.

    Returns:
        dict {nombre: shapely.Polygon} en coordenadas WGS84.
    """
    ruta_cache = DIR_RAW / "zonas_prohibidas.geojson"

    if ruta_cache.exists() and not forzar:
        print(f"[cache] Cargando zonas desde {ruta_cache.name}")
        with open(ruta_cache) as f:
            coleccion = json.load(f)
        return {
            feat["properties"]["nombre"]: shape(feat["geometry"])
            for feat in coleccion["features"]
        }

    print("[OSM] Obteniendo geometrias de zonas...")
    geometrias = {}
    features = []

    try:
        import osmnx as ox
        logging.getLogger("osmnx").setLevel(logging.ERROR)
        ox.settings.log_console = False
        _ox_disponible = True
    except ImportError:
        _ox_disponible = False

    consultas_osm = {
        "region_factible":    "Manhattan Island, New York City, USA",
        "central_park":       "Central Park, New York City, USA",
        "morningside_park":   "Morningside Park, Manhattan, New York City, USA",
        "marcus_garvey_park": "Marcus Garvey Park, New York City, USA",
        "inwood_hill_park":   "Inwood Hill Park, Manhattan, New York City, USA",
        "fort_tryon_park":    "Fort Tryon Park, Manhattan, New York City, USA",
        "battery_park":       "The Battery, Manhattan, New York City, USA",
    }

    for nombre, query in consultas_osm.items():
        try:
            if not _ox_disponible:
                raise ImportError("osmnx no disponible")
            gdf = ox.geocode_to_gdf(query)
            geom = gdf.geometry.iloc[0]
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda g: g.area)
            hull = geom.convex_hull
            n_sp = len(hull.exterior.coords) - 1
            geometrias[nombre] = hull
            features.append({
                "type": "Feature",
                "properties": {"nombre": nombre, "n_semiplanos": n_sp},
                "geometry": hull.__geo_interface__,
            })
            lo, la, hi, ha = hull.bounds
            print(f"  [OSM ok] {nombre}: {n_sp} semiplanos  lat [{la:.4f}, {ha:.4f}]")
        except Exception:
            # Fallback solo para las 2 zonas con constantes conocidas
            if nombre == "region_factible":
                bb = BBOX_MANHATTAN_WGS84
            elif nombre == "central_park":
                bb = BBOX_CENTRAL_PARK_WGS84
            else:
                print(f"  [skip] {nombre}: osmnx no disponible y sin fallback")
                continue
            lon_min, lat_min, lon_max, lat_max = bb
            bbox_geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
            geometrias[nombre] = bbox_geom
            features.append({
                "type": "Feature",
                "properties": {"nombre": nombre, "n_semiplanos": 4},
                "geometry": bbox_geom.__geo_interface__,
            })
            print(f"  [fallback] {nombre}: bbox hardcodeado (4 semiplanos)")

    coleccion = {"type": "FeatureCollection", "features": features}
    with open(ruta_cache, "w") as f:
        json.dump(coleccion, f, indent=2)

    print(f"[OSM] Guardado en {ruta_cache.name}")
    return geometrias
