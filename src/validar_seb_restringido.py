"""
Validacion del SEB con restricciones (Fase 3).

Estrategia
----------
Para validar el resultado del enumerador de aristas activas usamos cvxpy
con SOCP per-edge:

    Para cada arista (a, b) candidata, resolvemos:

        minimizar  r
        s.a.       ||c - p_i|| <= r,   i = 1..n
                   c = a + t (b - a),  t in [0, 1]

    El minimo de r entre todas las aristas factibles es el radio optimo
    del SEB restringido (tal como lo calcula nuestro enumerador).

Usamos un subset de las aristas (las del parque que atrapa al centro libre)
porque cvxpy es lento (~0.5-2 s por SOCP); 37 aristas de Central Park es
un buen balance entre cobertura y tiempo.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from shapely.geometry import Polygon

from .constraints import aristas, punto_en_F, semiplanos
from .seb_restringido import seb_en_segmento, seb_restringido


def seb_en_segmento_socp(
    puntos: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Resuelve via SOCP el SEB con centro restringido al segmento [a, b].

    Returns:
        (centro_opt, radio_opt, t_opt)
    """
    import cvxpy as cp

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = b - a

    t = cp.Variable()
    r = cp.Variable()
    c = a + t * v   # afin en t (vector (2,))

    # ||c - p_i|| <= r para todos los puntos. Para mantener un solo SOC
    # vectorizado: ||A c + b|| <= r donde A apila las identidades y b
    # apila los -p_i. Equivalente a apilar n constraints individuales.
    restricciones = [t >= 0.0, t <= 1.0]
    for i in range(len(puntos)):
        restricciones.append(cp.norm(c - puntos[i], 2) <= r)

    prob = cp.Problem(cp.Minimize(r), restricciones)
    prob.solve()

    t_val = float(t.value)
    c_val = a + t_val * v
    r_val = float(r.value)
    return (c_val, r_val, t_val)


def validar_arista_optima(
    puntos: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    radio_nuestro: float,
) -> dict:
    """
    Validacion ligera (1 SOCP) para la arista binding del SEB restringido.

    Resuelve el SOCP en el segmento [p1, p2] con cvxpy y compara con el
    radio que devolvio nuestro algoritmo. Tipicamente 5-15 s para n=8000.
    """
    c_socp, r_socp, t_socp = seb_en_segmento_socp(puntos, p1, p2)
    diff = abs(radio_nuestro - r_socp)
    return {
        "centro_socp": c_socp,
        "radio_socp": r_socp,
        "t_socp": t_socp,
        "radio_nuestro": radio_nuestro,
        "diff_radio": diff,
        "ok": diff < 1e-2,   # 1 cm de tolerancia (puntos en metros)
    }


def validar_contra_socp(
    puntos: np.ndarray,
    poligono_prohibido: Polygon,
    sps_R,
    sps_zonas,
    tolerancia_radio: float = 1e-2,
) -> dict:
    """
    Valida el enumerador comparando arista a arista contra SOCP, usando solo
    las aristas de un poligono prohibido (tipicamente el que atrapa al centro
    libre).

    Args:
        puntos: array (n, 2) de puntos.
        poligono_prohibido: shapely.Polygon convexo (e.g. Central Park UTM).
        sps_R, sps_zonas: representacion semiplana del conjunto F (para test
            de feasibilidad).
        tolerancia_radio: |r_nuestro - r_socp| < tol para declarar 'OK'.

    Returns:
        dict con la comparacion arista-a-arista y el mejor de cada lado.
    """
    aris = aristas(poligono_prohibido)

    filas = []
    for j, (p1, p2) in enumerate(aris):
        # Nuestro (scipy)
        (c_n, r_n), t_n = seb_en_segmento(puntos, p1, p2)
        factible_n = punto_en_F(c_n, sps_R, sps_zonas)

        # SOCP (cvxpy)
        try:
            c_s, r_s, t_s = seb_en_segmento_socp(puntos, p1, p2)
            factible_s = punto_en_F(c_s, sps_R, sps_zonas)
            socp_ok = True
        except Exception as e:
            c_s, r_s, t_s, factible_s = None, float("nan"), float("nan"), False
            socp_ok = False

        filas.append({
            "arista": j,
            "r_nuestro": r_n,
            "r_socp": r_s,
            "factible_nuestro": bool(factible_n),
            "factible_socp": bool(factible_s),
            "diff_radio": (abs(r_n - r_s) if socp_ok else float("nan")),
            "diff_t": (abs(t_n - t_s) if socp_ok else float("nan")),
            "ok": socp_ok and abs(r_n - r_s) < tolerancia_radio,
        })

    # Mejor candidato factible de cada lado
    fact_n = [f for f in filas if f["factible_nuestro"]]
    fact_s = [f for f in filas if f["factible_socp"]]
    mejor_n = min(fact_n, key=lambda f: f["r_nuestro"]) if fact_n else None
    mejor_s = min(fact_s, key=lambda f: f["r_socp"]) if fact_s else None

    todos_ok = all(f["ok"] for f in filas)

    return {
        "aristas_totales": len(filas),
        "aristas_factibles_nuestro": len(fact_n),
        "aristas_factibles_socp": len(fact_s),
        "todos_ok": todos_ok,
        "mejor_nuestro": mejor_n,
        "mejor_socp": mejor_s,
        "diff_mejor_radio": (
            abs(mejor_n["r_nuestro"] - mejor_s["r_socp"])
            if (mejor_n and mejor_s) else float("nan")
        ),
        "filas": filas,
    }
