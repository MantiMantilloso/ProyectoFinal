"""
Backend FastAPI para Manhattan S.O.S.

Sirve el frontend estatico (index.html, app.js, styles.css, data/) y expone:

  GET  /api/accidents          -> lista de [lat, lng] de los accidentes
  POST /api/seb                -> SEB con restricciones (zonas LP) + trail

Para correr:
    python manhattan_sos/server.py
    -> http://127.0.0.1:8765
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pyproj import Transformer
from shapely.geometry import Polygon, MultiPolygon, Point, shape, box as shapely_box
from shapely.ops import unary_union

# --- imports de src/ (acceso al algoritmo) ----------------------------------

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aristas_completo import enumerar_aristas  # noqa: E402

# --- constantes -------------------------------------------------------------

WGS_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
UTM_TO_WGS = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)

# Manhattan bbox WGS84 (lon_min, lat_min, lon_max, lat_max), igual que en src/data_loader.py
BBOX_MANHATTAN = (-74.0472, 40.6797, -73.9068, 40.8820)

# Tamano de la muestra de accidentes que usaremos para el SEB (rapido en demo)
DEMO_SAMPLE = 500
SEED = 42
MANHATTAN_STRICT_MARGIN_M = 2.0

STATIC_DIR = Path(__file__).resolve().parent
ACCIDENTS_PARQUET = ROOT / "data" / "processed" / "accidentes_utm.parquet"
ZONES_GEOJSON = STATIC_DIR / "data" / "zones.geojson"

# --- estado de la app (precomputado en startup) -----------------------------

PUNTOS_UTM: np.ndarray         # (n, 2) en UTM
PUNTOS_LATLON: list[list[float]]  # [[lat, lng], ...] (mismo orden que PUNTOS_UTM)
REGION_R_UTM: Polygon          # bbox de Manhattan reproyectado a UTM
MANHATTAN_MASK_UTM: Polygon | MultiPolygon  # mascara real (sectores) para validacion final


def _reproyectar_poligono_a_utm(pol_wgs: Polygon) -> Polygon:
    coords = list(pol_wgs.exterior.coords)
    coords_utm = [WGS_TO_UTM.transform(lon, lat) for lon, lat in coords]
    return Polygon(coords_utm)


def _cargar_accidentes() -> tuple[np.ndarray, list[list[float]]]:
    if not ACCIDENTS_PARQUET.exists():
        raise RuntimeError(
            f"No existe {ACCIDENTS_PARQUET}. Ejecuta primero el preprocesado."
        )
    df = pd.read_parquet(ACCIDENTS_PARQUET)
    if DEMO_SAMPLE and len(df) > DEMO_SAMPLE:
        df = df.sample(n=DEMO_SAMPLE, random_state=SEED).reset_index(drop=True)
    xy = df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)
    lats = df["latitude"].astype(float).tolist()
    lons = df["longitude"].astype(float).tolist()
    latlon = [[la, lo] for la, lo in zip(lats, lons)]
    return xy, latlon


def _construir_region_R() -> Polygon:
    lon_min, lat_min, lon_max, lat_max = BBOX_MANHATTAN
    bbox = shapely_box(lon_min, lat_min, lon_max, lat_max)
    return _reproyectar_poligono_a_utm(bbox)


def _construir_mascara_manhattan_utm() -> Polygon | MultiPolygon:
    """
    Construye la mascara "real" de Manhattan uniendo todos los features
    categoria 'sector' del zones.geojson y reproyectando a UTM.
    """
    if not ZONES_GEOJSON.exists():
        raise RuntimeError(f"No existe {ZONES_GEOJSON}")

    import json
    fc = json.loads(ZONES_GEOJSON.read_text(encoding="utf-8"))

    geoms_wgs = []
    for f in fc.get("features", []):
        props = f.get("properties") or {}
        if props.get("category") != "sector":
            continue
        g = shape(f.get("geometry"))
        if g.is_empty:
            continue
        geoms_wgs.append(g)

    if not geoms_wgs:
        raise RuntimeError("No se encontraron zonas 'sector' para mascara Manhattan")

    union_wgs = unary_union(geoms_wgs)

    def _poly_to_utm(p: Polygon) -> Polygon:
        return _reproyectar_poligono_a_utm(p)

    if isinstance(union_wgs, Polygon):
        mask_utm: Polygon | MultiPolygon = _poly_to_utm(union_wgs)
    elif isinstance(union_wgs, MultiPolygon):
        mask_utm = MultiPolygon([_poly_to_utm(p) for p in union_wgs.geoms if not p.is_empty])
    else:
        raise RuntimeError("Union de sectores no produjo Polygon/MultiPolygon")

    if MANHATTAN_STRICT_MARGIN_M > 0:
        shrunk = mask_utm.buffer(-MANHATTAN_STRICT_MARGIN_M)
        if not shrunk.is_empty:
            mask_utm = shrunk

    return mask_utm


def _zona_a_poligono_utm(geom_dict: dict) -> Optional[Polygon]:
    """
    Convierte una geometria GeoJSON (Polygon o MultiPolygon) a un Polygon
    en UTM 18N. Para MultiPolygon usa la pieza de mayor area.

    Nota: NO usamos convex_hull. Ese paso distorsiona zonas no-convexas
    (p.ej. sectores/community districts) y puede introducir restricciones
    falsas al "rellenar" concavidades.
    """
    geom = shape(geom_dict)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if not isinstance(geom, Polygon):
        return None
    if geom.is_empty:
        return None
    return _reproyectar_poligono_a_utm(geom)


def _centro_utm_a_latlon(c: Any) -> Optional[list[float]]:
    if c is None:
        return None
    lon, lat = UTM_TO_WGS.transform(float(c[0]), float(c[1]))
    return [float(lat), float(lon)]


# --- modelos ----------------------------------------------------------------

class Zona(BaseModel):
    name: str = ""
    category: str = ""
    osm_id: Optional[str] = None
    geometry: dict = Field(..., description="GeoJSON Polygon o MultiPolygon")


class SEBRequest(BaseModel):
    zones: List[Zona] = Field(default_factory=list)
    include_trail: bool = True


class AristaActiva(BaseModel):
    source: str
    p1: list[float]
    p2: list[float]


class SEBResponse(BaseModel):
    status: str                      # 'libre' | 'restringido' | 'infactible'
    center: Optional[list[float]]    # [lat, lng] o null si infactible
    radius_m: Optional[float]
    free_center: list[float]
    free_radius_m: float
    n_candidates: int
    n_feasible: int
    active_edge: Optional[AristaActiva] = None
    trail: list[list[float]] = Field(default_factory=list)  # centros por prefijo


# --- FastAPI ----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global PUNTOS_UTM, PUNTOS_LATLON, REGION_R_UTM, MANHATTAN_MASK_UTM
    PUNTOS_UTM, PUNTOS_LATLON = _cargar_accidentes()
    REGION_R_UTM = _construir_region_R()
    MANHATTAN_MASK_UTM = _construir_mascara_manhattan_utm()
    print(f"[startup] accidentes en demo: {len(PUNTOS_UTM)} (de {ACCIDENTS_PARQUET.name})")
    print(f"[startup] region R: bbox UTM {REGION_R_UTM.bounds}")
    print(f"[startup] mascara Manhattan strict-margin={MANHATTAN_STRICT_MARGIN_M} m")
    yield


app = FastAPI(title="Manhattan S.O.S.", lifespan=lifespan)


@app.get("/api/accidents")
def api_accidents() -> dict:
    return {"points": PUNTOS_LATLON, "count": len(PUNTOS_LATLON)}


def _computar_seb(zonas_utm: list[tuple[str, Polygon]]) -> dict:
    res = enumerar_aristas(
        puntos=PUNTOS_UTM,
        poligono_R=REGION_R_UTM,
        zonas_prohibidas=zonas_utm,
        semilla=SEED,
        incluir_R=True,
    )

    estado_base = res["estado"]
    c_libre = res["centro_libre"]
    r_libre = res["radio_libre"]
    n_total = res["n_total"]

    def _en_mascara(c: np.ndarray) -> bool:
        pt = Point(float(c[0]), float(c[1]))
        return bool(MANHATTAN_MASK_UTM.contains(pt))

    # Construir opciones factibles del solver y luego filtrar por mascara.
    opciones: list[dict] = []

    # Si el problema base es libre, c_libre es candidato valido del solver.
    if estado_base == "libre":
        opciones.append({
            "tipo": "libre",
            "c": c_libre,
            "r": float(r_libre),
            "edge": None,
        })

    # Candidatos factibles del algoritmo por aristas.
    for cand in res["candidatos"]:
        if not bool(cand.get("factible", False)):
            continue
        opciones.append({
            "tipo": "arista",
            "c": cand["c"],
            "r": float(cand["r"]),
            "edge": {
                "source": str(cand["fuente"]),
                "p1": _centro_utm_a_latlon(cand["p1"]),
                "p2": _centro_utm_a_latlon(cand["p2"]),
            },
        })

    opciones_mascara = [o for o in opciones if _en_mascara(o["c"])]

    if not opciones_mascara:
        estado = "infactible"
        center_latlon = None
        radius_m = None
        active_edge = None
        n_factibles = 0
    else:
        mejor = min(opciones_mascara, key=lambda o: o["r"])
        center_latlon = _centro_utm_a_latlon(mejor["c"])
        radius_m = float(mejor["r"])
        active_edge = mejor["edge"]
        n_factibles = len(opciones_mascara)
        estado = "libre" if mejor["tipo"] == "libre" else "restringido"

    return {
        "status": estado,
        "center": center_latlon,
        "radius_m": radius_m,
        "free_center": _centro_utm_a_latlon(c_libre),
        "free_radius_m": float(r_libre),
        "n_candidates": int(n_total),
        "n_feasible": int(n_factibles),
        "active_edge": active_edge,
    }


@app.post("/api/seb", response_model=SEBResponse)
def api_seb(req: SEBRequest) -> SEBResponse:
    # Reproyectar y convex-hull cada zona LP
    zonas_utm: list[tuple[str, Polygon]] = []
    for i, z in enumerate(req.zones):
        try:
            poly = _zona_a_poligono_utm(z.geometry)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"zona {i} ({z.name}): geometria invalida ({exc})",
            )
        if poly is None or poly.is_empty:
            continue
        nombre = z.name or f"zona_{i}"
        zonas_utm.append((nombre, poly))

    # 1) SEB final con todas las zonas
    final = _computar_seb(zonas_utm)

    # 2) Trail: SEB tras cada prefijo de zonas (en orden cronologico)
    trail: list[list[float]] = []
    if req.include_trail and zonas_utm:
        for k in range(1, len(zonas_utm) + 1):
            paso = _computar_seb(zonas_utm[:k])
            # si un paso es infactible no hay centro -> lo saltamos en el trail
            if paso["center"] is not None:
                trail.append(paso["center"])

    return SEBResponse(
        status=final["status"],
        center=final["center"],
        radius_m=final["radius_m"],
        free_center=final["free_center"],
        free_radius_m=final["free_radius_m"],
        n_candidates=final["n_candidates"],
        n_feasible=final["n_feasible"],
        active_edge=(AristaActiva(**final["active_edge"])
                     if final["active_edge"] else None),
        trail=trail,
    )


# --- Estatico (modo dev: solo si los archivos existen junto al server) ------
# En produccion con docker, nginx sirve el frontend y este bloque queda inerte.

_INDEX = STATIC_DIR / "index.html"
_APPJS = STATIC_DIR / "app.js"
_CSS   = STATIC_DIR / "styles.css"
_DATA  = STATIC_DIR / "data"

if _INDEX.exists():
    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(_INDEX)

if _APPJS.exists():
    @app.get("/app.js")
    def serve_appjs() -> FileResponse:
        return FileResponse(_APPJS, media_type="application/javascript")

if _CSS.exists():
    @app.get("/styles.css")
    def serve_css() -> FileResponse:
        return FileResponse(_CSS, media_type="text/css")

if _DATA.is_dir():
    app.mount("/data", StaticFiles(directory=_DATA), name="data")


# --- entrypoint -------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
