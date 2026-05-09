"""
Modulo para descarga de datos desde NYC Open Data y OpenStreetMap.
"""

import json
import logging
import time
import requests
import pandas as pd
import osmnx as ox
from pathlib import Path
from shapely.geometry import box as shapely_box, shape

# Suprimir logs de osmnx (usa Unicode que rompe en terminales cp1252)
logging.getLogger("osmnx").setLevel(logging.ERROR)
ox.settings.log_console = False

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


# ── Bounding boxes de zonas prohibidas ────────────────────────────────────────
# Coordenadas WGS84 (lon_min, lat_min, lon_max, lat_max)
# Usamos bounding boxes axis-aligned para mantener la formulacion de semiplanos.
BBOXES_WGS84 = {
    "central_park":    (-73.9814, 40.7647, -73.9496, 40.8003),
    "hudson_river":    (-74.0300, 40.7000, -73.9900, 40.8800),  # franja oeste
    "east_river":      (-73.9700, 40.6900, -73.9400, 40.8000),  # franja este
    "harlem_river":    (-73.9400, 40.8000, -73.9200, 40.8700),  # franja noreste
}


def obtener_bboxes_osm(forzar: bool = False) -> dict:
    """
    Devuelve bboxes de zonas prohibidas como Shapely Polygons (WGS84).
    Intenta enriquecer con OSM real; si falla usa los valores hardcoded.
    Guarda en data/raw/zonas_prohibidas.geojson.

    Returns:
        dict {nombre: shapely.Polygon} con la bbox de cada zona.
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

    print("[OSM] Obteniendo bounding boxes de zonas prohibidas...")
    geometrias = {}
    features = []

    consultas_osm = {
        "central_park": "Central Park, New York City, USA",
        "hudson_river": "Hudson River, New York, USA",
    }

    for nombre, query in consultas_osm.items():
        try:
            gdf = ox.geocode_to_gdf(query)
            geom = gdf.geometry.iloc[0]
            # Convertir a bounding box (lo acordado: aproximacion con bbox)
            bb = geom.bounds  # (minx, miny, maxx, maxy)
            bbox_geom = shapely_box(*bb)
            geometrias[nombre] = bbox_geom
            features.append({
                "type": "Feature",
                "properties": {"nombre": nombre},
                "geometry": bbox_geom.__geo_interface__,
            })
            print(f"  [OSM ok] {nombre}: bbox {bb[0]:.4f},{bb[1]:.4f} -> {bb[2]:.4f},{bb[3]:.4f}")
        except Exception:
            # Fallback a bbox hardcodeada
            lon_min, lat_min, lon_max, lat_max = BBOXES_WGS84[nombre]
            bbox_geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
            geometrias[nombre] = bbox_geom
            features.append({
                "type": "Feature",
                "properties": {"nombre": nombre},
                "geometry": bbox_geom.__geo_interface__,
            })
            print(f"  [fallback] {nombre}: usando bbox hardcodeada")

    # East River y Harlem River: solo bbox hardcodeada (Nominatim no devuelve poligono)
    for nombre in ["east_river", "harlem_river"]:
        lon_min, lat_min, lon_max, lat_max = BBOXES_WGS84[nombre]
        bbox_geom = shapely_box(lon_min, lat_min, lon_max, lat_max)
        geometrias[nombre] = bbox_geom
        features.append({
            "type": "Feature",
            "properties": {"nombre": nombre},
            "geometry": bbox_geom.__geo_interface__,
        })
        print(f"  [bbox] {nombre}: definida manualmente")

    coleccion = {"type": "FeatureCollection", "features": features}
    with open(ruta_cache, "w") as f:
        json.dump(coleccion, f, indent=2)

    print(f"[OSM] Guardado en {ruta_cache.name}")
    return geometrias
