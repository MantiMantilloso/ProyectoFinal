"""
SEB con restricciones (Fase 3) - F = R \\ union(Z_i), no-convexo.

Algoritmo: enumeracion de aristas activas
-----------------------------------------
Sea c* el centro optimo del SEB restringido. Por estructura del problema:

  - Si el centro libre c_libre cae en F  ->  c* = c_libre y r* = r_libre.
  - En otro caso, c* yace en la FRONTERA de F. La frontera de F esta
    formada por (subconjuntos de) las aristas de R y de las zonas Z_i.
  - Por convexidad local de cada arista, la restriccion activa puede ser:
       (a) una unica arista de la frontera  -> c* esta en el interior de
           ese segmento, y resuelve un sub-problema "SEB-en-segmento".
       (b) un vertice de la frontera         -> c* coincide con un endpoint;
           ese caso queda cubierto por las aristas adyacentes (t = 0 o t = 1).

Por tanto basta enumerar todas las aristas de R y de las Z_i, resolver el
sub-problema 1D en cada una, descartar los infactibles, y elegir el de
radio minimo.

Sub-problema "SEB-en-segmento"
------------------------------
Dado un segmento [a, b] y los puntos {p_1, ..., p_n}, queremos:

    minimizar  r
    sujeto a   ||c - p_i|| <= r  para todo i,
               c = a + t (b - a),  t in [0, 1].

Equivalente a min_{t in [0,1]}  max_i  || (a + t v) - p_i ||^2,  v = b - a.

Cada || (a + t v) - p_i ||^2 es una parabola en t con coeficiente lider
||v||^2 (igual para todos los puntos). Su maximo punto-a-punto es convexo
en t -> minimizable de forma robusta con bisseccion ('bounded').
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize_scalar
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from .seb_seidel import seb_seidel
from .seb_primitivas import Bola, EPS
from .constraints import (
    Semiplano,
    aristas,
    semiplanos,
    punto_en_F_union,
)


def _preparar_zonas(
    zonas_prohibidas: List[Tuple[str, Polygon]],
):
    """
    Devuelve (Z_union, union_pols):
      - Z_union    : geometria Shapely del union (Polygon | MultiPolygon | None)
      - union_pols : lista de Polygon componentes para enumerar aristas
    """
    if not zonas_prohibidas:
        return None, []
    Z_union = unary_union([pol for _, pol in zonas_prohibidas])
    if isinstance(Z_union, Polygon):
        return Z_union, [Z_union]
    if isinstance(Z_union, MultiPolygon):
        return Z_union, list(Z_union.geoms)
    return None, []


def _interseccion_t(
    p1: np.ndarray, p2: np.ndarray,
    q1: np.ndarray, q2: np.ndarray,
    eps: float = 1e-10,
) -> Optional[float]:
    v = p2 - p1
    w = q2 - q1
    d = q1 - p1
    det = w[0] * v[1] - v[0] * w[1]
    if abs(det) < eps:
        return None
    t = (w[0] * d[1] - d[0] * w[1]) / det
    s = (v[0] * d[1] - d[0] * v[1]) / det
    if -eps <= t <= 1 + eps and -eps <= s <= 1 + eps:
        return float(np.clip(t, 0.0, 1.0))
    return None


# --- sub-problema 1D --------------------------------------------------------

def seb_en_segmento(
    puntos: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    xatol: float = 1e-9,
) -> Tuple[Bola, float]:
    """
    SEB con centro restringido al segmento [a, b].

    Returns:
        ((centro, radio), t)  con t in [0, 1] tal que centro = a + t (b - a).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = b - a
    L2 = float(v @ v)

    if L2 < EPS * EPS:
        # Segmento degenerado: el centro debe estar en a.
        c = a.copy()
        r = float(np.max(np.linalg.norm(puntos - c, axis=1)))
        return ((c, r), 0.0)

    def max_dist2(t: float) -> float:
        c = a + t * v
        d2 = (puntos - c)
        return float(np.max(np.einsum("ij,ij->i", d2, d2)))

    res = minimize_scalar(
        max_dist2, bounds=(0.0, 1.0), method="bounded",
        options={"xatol": xatol, "maxiter": 200},
    )
    t_opt = float(res.x)
    c_opt = a + t_opt * v
    r_opt = float(np.sqrt(res.fun))
    return ((c_opt, r_opt), t_opt)


# --- algoritmo principal ----------------------------------------------------

def seb_restringido(
    puntos: np.ndarray,
    poligono_R: Polygon,
    zonas_prohibidas: List[Tuple[str, Polygon]],
    semilla: Optional[int] = None,
    incluir_R_en_aristas: bool = True,
) -> dict:
    """
    Calcula la SEB de los puntos con la restriccion: el centro debe estar en
    F = R \\ union(zonas prohibidas).

    Args:
        puntos: array (n, 2) en UTM.
        poligono_R: shapely Polygon convexo (region factible).
        zonas_prohibidas: lista [(nombre, Polygon)] de zonas prohibidas convexas.
        semilla: para Seidel del SEB libre (reproducibilidad).
        incluir_R_en_aristas: si True, considera las aristas de R como
            candidatas. Util si el centro libre puede caer fuera de R; en
            nuestro problema (centro libre dentro de Manhattan) solo agrega
            candidatos no competitivos pero es seguro.

    Returns:
        dict con campos:
            centro: np.ndarray (2,)
            radio: float
            estado: 'libre' | 'restringido' | 'infactible'
            arista_activa: dict | None  (descripcion de la arista ganadora)
            centro_libre: np.ndarray  (centro del SEB sin restricciones)
            radio_libre: float        (radio del SEB sin restricciones)
            n_candidatos_evaluados: int
            n_candidatos_factibles: int
            candidatos: lista de dicts (uno por arista evaluada y factible)
    """
    sps_R = semiplanos(poligono_R)

    # Z_union: geometria Shapely del union de zonas (puede ser no-convexa).
    # union_pols: componentes para enumerar aristas exteriores.
    Z_union, union_pols = _preparar_zonas(zonas_prohibidas)

    # 1) SEB sin restricciones
    c_libre, r_libre = seb_seidel(puntos, semilla=semilla)

    if punto_en_F_union(c_libre, sps_R, Z_union):
        return {
            "centro": c_libre,
            "radio": r_libre,
            "estado": "libre",
            "arista_activa": None,
            "centro_libre": c_libre,
            "radio_libre": r_libre,
            "n_candidatos_evaluados": 0,
            "n_candidatos_factibles": 0,
            "candidatos": [],
        }

    # 2) Enumerar aristas candidatas
    fuentes: List[Tuple[str, Polygon]] = []
    if incluir_R_en_aristas:
        fuentes.append(("region_factible", poligono_R))
    for i, p in enumerate(union_pols):
        fuentes.append((f"zona_union_{i}", p))

    todas_aristas_geom: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, pol in fuentes:
        todas_aristas_geom.extend(aristas(pol))

    candidatos: list[dict] = []
    n_evaluados = 0

    for nombre, poligono in fuentes:
        for j, (p1, p2) in enumerate(aristas(poligono)):
            n_evaluados += 1
            (c, r), t = seb_en_segmento(puntos, p1, p2)
            if punto_en_F_union(c, sps_R, Z_union):
                candidatos.append({
                    "centro": c,
                    "radio": r,
                    "fuente": nombre,
                    "arista_idx": j,
                    "p1": p1,
                    "p2": p2,
                    "t": t,
                })

            v = p2 - p1
            for q1, q2 in todas_aristas_geom:
                t_j = _interseccion_t(p1, p2, q1, q2)
                if t_j is None or abs(t_j - t) < 1e-9:
                    continue
                c_j = p1 + t_j * v
                r_j = float(np.max(np.linalg.norm(puntos - c_j, axis=1)))
                if punto_en_F_union(c_j, sps_R, Z_union):
                    n_evaluados += 1
                    candidatos.append({
                        "centro": c_j,
                        "radio": r_j,
                        "fuente": nombre,
                        "arista_idx": j,
                        "p1": p1,
                        "p2": p2,
                        "t": t_j,
                    })

    if not candidatos:
        return {
            "centro": None,
            "radio": float("inf"),
            "estado": "infactible",
            "arista_activa": None,
            "centro_libre": c_libre,
            "radio_libre": r_libre,
            "n_candidatos_evaluados": n_evaluados,
            "n_candidatos_factibles": 0,
            "candidatos": [],
        }

    # 3) Mejor candidato
    mejor = min(candidatos, key=lambda d: d["radio"])

    return {
        "centro": mejor["centro"],
        "radio": mejor["radio"],
        "estado": "restringido",
        "arista_activa": {
            "fuente": mejor["fuente"],
            "arista_idx": mejor["arista_idx"],
            "p1": mejor["p1"],
            "p2": mejor["p2"],
            "t": mejor["t"],
        },
        "centro_libre": c_libre,
        "radio_libre": r_libre,
        "n_candidatos_evaluados": n_evaluados,
        "n_candidatos_factibles": len(candidatos),
        "candidatos": candidatos,
    }


# --- helpers para diagnostico ------------------------------------------------

def zonas_que_contienen(
    c: np.ndarray, zonas_prohibidas: List[Tuple[str, Polygon]],
) -> List[str]:
    """
    Devuelve los nombres de las zonas prohibidas cuyo INTERIOR contiene a c.
    Util para entender por que el centro libre es infactible.
    """
    out = []
    for nombre, poligono in zonas_prohibidas:
        sps = semiplanos(poligono)
        # estrictamente dentro: todos n . x < d
        if all(float(n @ c) < d - EPS for n, d in sps):
            out.append(nombre)
    return out
