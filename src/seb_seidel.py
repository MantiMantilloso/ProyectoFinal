"""
Algoritmo de Seidel para Smallest Enclosing Ball en 2D.

Es la version iterativa "move-to-front" del algoritmo aleatorizado incremental
para problemas LP-type, aplicado al SEB. Complejidad O(n) esperada.

Idea:
  - Se procesan los puntos uno a uno tras una permutacion aleatoria.
  - Se mantiene la SEB de los puntos procesados.
  - Si el siguiente punto cae fuera, se sabe que ese punto debe estar en la
    frontera de la nueva SEB. Se reinicia el calculo con esa restriccion.
  - El truco es que en 2D la SEB se determina por <= 3 puntos en la frontera,
    asi que la recursion tiene profundidad acotada.

Referencia:
  Seidel, R. (1991). "Small-dimensional linear programming and convex hulls
  made easy." Discrete & Computational Geometry, 6(3), 423-434.
"""

from __future__ import annotations
import numpy as np
from .seb_primitivas import (
    Bola,
    bola_de_1_punto,
    bola_de_2_puntos,
    bola_de_3_puntos,
    bola_minima_3,
    punto_en_bola,
)


def seb_seidel(puntos: np.ndarray, semilla: int | None = None) -> Bola:
    """
    Calcula la Smallest Enclosing Ball de un conjunto de puntos en 2D.

    Args:
        puntos: array (n, 2) con los puntos.
        semilla: semilla para la permutacion aleatoria (reproducibilidad).

    Returns:
        (centro, radio) con centro como np.ndarray shape (2,) y radio float.
    """
    pts = np.asarray(puntos, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Se esperaba shape (n, 2), se recibio {pts.shape}")

    n = len(pts)
    if n == 0:
        raise ValueError("Se requiere al menos 1 punto.")
    if n == 1:
        return bola_de_1_punto(pts[0])

    # Permutacion aleatoria: clave para que el analisis O(n) esperado funcione
    rng = np.random.default_rng(semilla)
    P = pts[rng.permutation(n)].copy()

    # Inicializacion con los 2 primeros puntos
    bola = bola_de_2_puntos(P[0], P[1])

    # Procesar el resto: si P[i] no esta en la bola actual, debe estar en
    # la frontera de la nueva bola.
    for i in range(2, n):
        if not punto_en_bola(P[i], bola):
            bola = _seb_con_1_frontera(P[:i], P[i])

    return bola


def _seb_con_1_frontera(P: np.ndarray, q: np.ndarray) -> Bola:
    """
    SEB de los puntos P sabiendo que q debe estar en la frontera.

    Se recorren los puntos de P; cuando uno cae fuera de la bola actual,
    pasa a estar tambien en la frontera (junto con q).
    """
    bola = bola_de_2_puntos(P[0], q)
    for i in range(1, len(P)):
        if not punto_en_bola(P[i], bola):
            bola = _seb_con_2_frontera(P[:i], P[i], q)
    return bola


def _seb_con_2_frontera(P: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> Bola:
    """
    SEB de los puntos P sabiendo que q1 y q2 estan en la frontera.

    En 2D, anadir un tercer punto en la frontera determina el circulo
    de manera unica (circuncirculo), asi que no hay mas recursion.
    """
    bola = bola_de_2_puntos(q1, q2)
    for i in range(len(P)):
        if not punto_en_bola(P[i], bola):
            # 3 puntos en la frontera: q1, q2, P[i]
            bola_nueva = bola_de_3_puntos(q1, q2, P[i])
            if bola_nueva is not None:
                bola = bola_nueva
            else:
                # Caso degenerado: 3 puntos colineales -> SEB de los 3
                bola = bola_minima_3(q1, q2, P[i])
    return bola
