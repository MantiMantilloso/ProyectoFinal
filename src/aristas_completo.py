"""
Enumeracion completa de aristas activas para visualizacion.

A diferencia de `seb_restringido.seb_restringido`, esta version devuelve TODOS
los candidatos (factibles e infactibles) con su feasibilidad, posicion (c, r)
y la geometria de la arista. Util para la visualizacion del algoritmo en la
app: ver cuantos candidatos se descartan, donde caen los radios, etc.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from .seb_seidel import seb_seidel
from .constraints import semiplanos, aristas, punto_en_F_union
from .seb_restringido import seb_en_segmento


def _preparar_zonas(
    zonas_prohibidas: List[Tuple[str, Polygon]],
):
    """
    Calcula la union de todas las zonas prohibidas y devuelve:
      - Z_union : geometria Shapely (Polygon | MultiPolygon | None)
                  usada en el chequeo de factibilidad con contains().
                  Correcta para uniones no-convexas.
      - union_pols : lista de Polygon componentes del union,
                     usados para enumerar las aristas exteriores.

    Zonas adyacentes/superpuestas se fusionan: sus aristas internas desaparecen
    y los puntos sobre ellas quedan en el interior del union -> infactibles.
    Zonas separadas dan un MultiPolygon; cada componente se enumera por separado.
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
    """
    Retorna t en [0,1] tal que p1+t*(p2-p1) intersecta q1+s*(q2-q1) con s en [0,1].
    None si los segmentos son paralelos o no se cruzan en ese rango.
    """
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


def enumerar_aristas(
    puntos: np.ndarray,
    poligono_R: Polygon,
    zonas_prohibidas: List[Tuple[str, Polygon]],
    semilla: int | None = None,
    incluir_R: bool = True,
) -> dict:
    """
    Enumera TODAS las aristas (R + zonas) y para cada una calcula el
    SEB-en-segmento. No filtra por feasibilidad; reporta el flag `factible`.

    Returns:
        dict con:
            candidatos          : list[dict]  todas las aristas con:
                fuente, arista_idx, p1, p2, c, r, t, factible
            centro_libre, radio_libre
            estado              : 'libre' | 'restringido' | 'infactible'
            mejor               : dict del candidato ganador (None si infactible
                                 o libre)
            n_factibles, n_total
    """
    sps_R = semiplanos(poligono_R)

    # Z_union: geometria Shapely del union de zonas (puede ser no-convexa).
    # union_pols: componentes para enumerar aristas exteriores.
    # El chequeo de factibilidad usa Z_union.contains() — correcto para
    # formas no-convexas, a diferencia del enfoque de semiplanos anterior.
    Z_union, union_pols = _preparar_zonas(zonas_prohibidas)

    c_libre, r_libre = seb_seidel(puntos, semilla=semilla)

    fuentes: List[Tuple[str, Polygon]] = []
    if incluir_R:
        fuentes.append(("region_factible", poligono_R))
    for i, p in enumerate(union_pols):
        fuentes.append((f"zona_union_{i}", p))

    if punto_en_F_union(c_libre, sps_R, Z_union):
        estado = "libre"
    else:
        estado = "candidate"  # se decide despues si factible o infactible

    # Todas las aristas de todas las fuentes, para calcular cruces entre ellas.
    # Cuando el optimo en una arista cae dentro de otra zona (infactible), el
    # punto donde esa arista cruza la frontera de la otra zona es el candidato
    # factible mas cercano al optimo — y el algoritmo sin este paso lo omite.
    todas_aristas_geom: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, pol in fuentes:
        todas_aristas_geom.extend(aristas(pol))

    candidatos: list[dict] = []
    for nombre, poligono in fuentes:
        for j, (p1, p2) in enumerate(aristas(poligono)):
            (c, r), t = seb_en_segmento(puntos, p1, p2)
            factible = bool(punto_en_F_union(c, sps_R, Z_union))
            candidatos.append({
                "fuente":     nombre,
                "arista_idx": j,
                "p1":         p1,
                "p2":         p2,
                "c":          c,
                "r":          float(r),
                "t":          float(t),
                "factible":   factible,
            })

            # Puntos de cruce con otras aristas: extremos de sub-segmentos
            # factibles cuando el optimo no lo es.
            v = p2 - p1
            for q1, q2 in todas_aristas_geom:
                t_j = _interseccion_t(p1, p2, q1, q2)
                if t_j is None or abs(t_j - t) < 1e-9:
                    continue
                c_j = p1 + t_j * v
                r_j = float(np.max(np.linalg.norm(puntos - c_j, axis=1)))
                factible_j = bool(punto_en_F_union(c_j, sps_R, Z_union))
                candidatos.append({
                    "fuente":     nombre,
                    "arista_idx": j,
                    "p1":         p1,
                    "p2":         p2,
                    "c":          c_j,
                    "r":          r_j,
                    "t":          t_j,
                    "factible":   factible_j,
                })

    factibles = [d for d in candidatos if d["factible"]]
    n_factibles = len(factibles)

    if estado == "libre":
        mejor = None
    elif n_factibles == 0:
        estado = "infactible"
        mejor = None
    else:
        estado = "restringido"
        mejor = min(factibles, key=lambda d: d["r"])

    return {
        "candidatos":   candidatos,
        "centro_libre": c_libre,
        "radio_libre":  float(r_libre),
        "estado":       estado,
        "mejor":        mejor,
        "n_factibles":  n_factibles,
        "n_total":      len(candidatos),
    }


def sub_problema_1d(
    puntos: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    n_t: int = 200,
    top_k: int = 6,
) -> dict:
    """
    Evalua f(t) = max_i ||p_i - c(t)||^2 sobre una grilla de t in [0, 1] para
    la arista (p1, p2). Devuelve la envolvente superior y las top_k parabolas
    individuales (las que tocan la envolvente cerca del optimo o que estan en
    el top global).

    Cada ||p_i - c(t)||^2 es una parabola en t con coef lider ||v||^2 (igual
    para todos los puntos) -> su pointwise max es convexo en t.

    Returns:
        dict con:
            t          : np.ndarray (n_t,)
            f          : np.ndarray (n_t,)  envolvente superior (km^2)
            parabolas  : list de dict { 'i', 'curva' (n_t,), 'es_soporte' }
            t_estrella : float
            r_estrella : float (km)
            c_estrella : np.ndarray (2,) en UTM
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    v = p2 - p1

    t_grid = np.linspace(0.0, 1.0, n_t)
    # c(t) shape (n_t, 2)
    c_grid = p1[None, :] + t_grid[:, None] * v[None, :]
    # d2 shape (n_t, n_puntos)
    diff = puntos[None, :, :] - c_grid[:, None, :]
    d2 = np.einsum("tij,tij->ti", diff, diff)
    f = d2.max(axis=1)

    # t* = argmin de f sobre la grilla
    i_min = int(np.argmin(f))
    t_star = float(t_grid[i_min])
    r_star = float(np.sqrt(f[i_min]))
    c_star = p1 + t_star * v

    # Identificar los puntos que aportan al maximo cerca de t*
    # Top-K por valor maximo de su parabola en cualquier t
    max_por_pto = d2.max(axis=0)  # (n_puntos,)
    top_idx = np.argsort(max_por_pto)[-top_k:][::-1]

    # En el optimo, el(los) punto(s) soporte son los que alcanzan el max
    soporte_idx = np.where(d2[i_min] >= f[i_min] * (1 - 1e-6))[0]

    parabolas = []
    for i in top_idx:
        parabolas.append({
            "i":          int(i),
            "curva":      d2[:, i] / 1e6,  # km^2
            "es_soporte": bool(i in soporte_idx),
        })

    return {
        "t":          t_grid,
        "f":          f / 1e6,  # km^2
        "parabolas":  parabolas,
        "t_estrella": t_star,
        "r_estrella": r_star,
        "c_estrella": c_star,
        "soporte_idx": [int(i) for i in soporte_idx],
    }
