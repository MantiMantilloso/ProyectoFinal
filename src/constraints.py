"""
Restricciones geometricas para el SEB con restricciones (Fase 3).

El conjunto factible es:

    F = R \\ (Z_1 union Z_2 union ... union Z_k)

donde R (region_factible) y cada Z_i (zona prohibida) son poligonos convexos.
F es no-convexo (en general): es R con k "huecos" convexos.

Representacion por semiplanos
-----------------------------
Cada poligono convexo P se representa como una lista de semiplanos:

    P = { x : n_j . x <= d_j para todo j }

donde n_j es la normal externa unitaria de la arista j y d_j = n_j . p_j
con p_j cualquier vertice de la arista j.

Asi:
  - "x esta en P"        <=>  todos los n_j . x <= d_j
  - "x NO esta en P"     <=>  existe j con n_j . x > d_j

Por tanto:
  - "x esta en F"  <=>  x esta en R  Y  x NO esta en ninguna Z_i.
"""

from __future__ import annotations
from typing import List, Tuple, Iterable
import numpy as np
from shapely.geometry import Polygon
from pyproj import Transformer

# Tolerancia numerica (en metros, dado que trabajamos en UTM ~1e6 m)
EPS = 1e-7

Semiplano = Tuple[np.ndarray, float]   # (normal externa unitaria, offset)
Arista = Tuple[np.ndarray, np.ndarray] # (p1, p2)


# --- reproyeccion WGS84 -> UTM ----------------------------------------------

_TRANSF_WGS_A_UTM18N = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)


def reproyectar_a_utm(poligono_wgs: Polygon) -> Polygon:
    """Reproyecta un shapely.Polygon de WGS84 (lon, lat) a UTM Zone 18N (m)."""
    coords = list(poligono_wgs.exterior.coords)
    coords_utm = [_TRANSF_WGS_A_UTM18N.transform(lon, lat) for lon, lat in coords]
    return Polygon(coords_utm)


# --- representacion de un poligono convexo ----------------------------------

def _coords_ccw(poligono: Polygon) -> List[Tuple[float, float]]:
    """
    Devuelve los vertices del exterior del poligono en orden CCW (sin
    duplicar el cierre). Detecta orientacion por area con signo.
    """
    coords = list(poligono.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]

    # signed area (formula del shoelace); >0 si CCW
    area2 = 0.0
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1

    if area2 < 0:                      # CW -> invertir a CCW
        coords = coords[::-1]
    return coords


def aristas(poligono: Polygon) -> List[Arista]:
    """Devuelve la lista de aristas (p1, p2) en orden CCW."""
    coords = _coords_ccw(poligono)
    n = len(coords)
    return [
        (np.array(coords[i], dtype=np.float64),
         np.array(coords[(i + 1) % n], dtype=np.float64))
        for i in range(n)
    ]


def semiplanos(poligono: Polygon) -> List[Semiplano]:
    """
    Devuelve los semiplanos (n_j, d_j) tales que x in P  <=>  n_j . x <= d_j
    para todo j. n_j es la normal EXTERNA unitaria de la arista j.

    Asume P convexo. Para CCW, la normal externa de una arista (p_i, p_{i+1})
    es (dy, -dx) normalizada (rotacion 90 grados en sentido horario).
    """
    sp: List[Semiplano] = []
    for p1, p2 in aristas(poligono):
        edge = p2 - p1
        n_ext = np.array([edge[1], -edge[0]], dtype=np.float64)
        norm = float(np.linalg.norm(n_ext))
        if norm < EPS:
            continue                   # arista degenerada
        n_ext /= norm
        d = float(n_ext @ p1)
        sp.append((n_ext, d))
    return sp


# --- tests de pertenencia ---------------------------------------------------

def punto_en_poligono(c: np.ndarray, sps: List[Semiplano], tol: float = EPS) -> bool:
    """True si c esta dentro o en la frontera del poligono (con tolerancia)."""
    for n, d in sps:
        if float(n @ c) > d + tol:
            return False
    return True


def punto_estrictamente_en_poligono(
    c: np.ndarray, sps: List[Semiplano], tol: float = EPS
) -> bool:
    """True si c esta estrictamente en el INTERIOR del poligono."""
    for n, d in sps:
        if float(n @ c) >= d - tol:
            return False
    return True


def punto_en_F(
    c: np.ndarray,
    sps_R: List[Semiplano],
    sps_zonas: List[List[Semiplano]],
    tol: float = EPS,
) -> bool:
    """
    True si c pertenece al conjunto factible F = R \\ union(Z_i).

    El centro debe estar en R (incluyendo borde) y NO debe estar en el
    INTERIOR estricto de ninguna zona prohibida (i.e. tocar el borde de un
    parque es factible).
    """
    if not punto_en_poligono(c, sps_R, tol=tol):
        return False
    for sps_z in sps_zonas:
        if punto_estrictamente_en_poligono(c, sps_z, tol=tol):
            return False
    return True


# --- helper para vectorizar el test sobre un array de candidatos ------------

def mascara_en_F(
    cs: np.ndarray,
    sps_R: List[Semiplano],
    sps_zonas: List[List[Semiplano]],
    tol: float = EPS,
) -> np.ndarray:
    """
    Vectorizado: devuelve mascara booleana (m,) con True donde el centro
    esta en F. cs tiene shape (m, 2).
    """
    m = len(cs)
    mask = np.ones(m, dtype=bool)

    # Debe estar en R: todos los semiplanos n . x <= d + tol
    for n, d in sps_R:
        mask &= (cs @ n) <= d + tol
        if not mask.any():
            return mask

    # No debe estar estrictamente dentro de ninguna zona
    for sps_z in sps_zonas:
        # "estrictamente dentro" <=> todos los n . x < d - tol
        adentro = np.ones(m, dtype=bool)
        for n, d in sps_z:
            adentro &= (cs @ n) < d - tol
            if not adentro.any():
                break
        mask &= ~adentro

    return mask


# --- utilidades de empaquetado ----------------------------------------------

def cargar_zonas_utm(zonas_wgs: dict) -> dict:
    """
    Reproyecta el dict {nombre: shapely.Polygon en WGS84} a UTM Zone 18N.
    Devuelve un dict equivalente en UTM.
    """
    return {nombre: reproyectar_a_utm(p) for nombre, p in zonas_wgs.items()}


def descomponer_zonas(
    zonas_utm: dict, nombre_R: str = "region_factible",
) -> Tuple[Polygon, List[Tuple[str, Polygon]]]:
    """
    Separa el dict de zonas UTM en (R, [(nombre, Z_i)]).
    """
    if nombre_R not in zonas_utm:
        raise KeyError(f"No se encontro la region factible '{nombre_R}'")
    R = zonas_utm[nombre_R]
    prohibidas = [(n, p) for n, p in zonas_utm.items() if n != nombre_R]
    return R, prohibidas
