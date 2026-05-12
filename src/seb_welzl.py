"""
Algoritmo de Welzl (recursivo) para Smallest Enclosing Ball en 2D.

Es el algoritmo aleatorizado clasico, equivalente a Seidel para SEB pero
expresado recursivamente. Complejidad O(n) esperada.

Lo usamos como baseline para validar que nuestra implementacion de Seidel
da los mismos resultados.

Pseudocodigo (Welzl, 1991):
    welzl(P, R):
        si P vacio o |R| = 3:
            retornar bola_trivial(R)
        elegir p aleatorio de P
        D = welzl(P - {p}, R)
        si p en D:
            retornar D
        retornar welzl(P - {p}, R u {p})

Referencia:
  Welzl, E. (1991). "Smallest enclosing disks (balls and ellipsoids)."
  New Results and New Trends in Computer Science, LNCS 555, 359-370.
"""

from __future__ import annotations
import sys
import numpy as np
from .seb_primitivas import (
    Bola,
    bola_de_1_punto,
    bola_de_2_puntos,
    bola_de_3_puntos,
    bola_minima_3,
    punto_en_bola,
)


def seb_welzl(puntos: np.ndarray, semilla: int | None = None) -> Bola:
    """
    Calcula la SEB usando el algoritmo recursivo de Welzl.

    Args:
        puntos: array (n, 2).
        semilla: semilla para la permutacion aleatoria.

    Returns:
        (centro, radio).
    """
    pts = np.asarray(puntos, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Se esperaba shape (n, 2), se recibio {pts.shape}")

    n = len(pts)
    if n == 0:
        raise ValueError("Se requiere al menos 1 punto.")

    # Subir el limite de recursion para n grande (~n niveles de profundidad)
    sys.setrecursionlimit(max(10_000, n + 1_000))

    rng = np.random.default_rng(semilla)
    P = pts[rng.permutation(n)].tolist()

    return _welzl(P, [])


def _welzl(P: list, R: list) -> Bola:
    """Recursion principal: SEB(P u R) con todos los puntos de R en la frontera."""
    if not P or len(R) == 3:
        return _bola_trivial(R)

    p = P[-1]               # tomar el ultimo (P ya esta permutado)
    D = _welzl(P[:-1], R)
    if punto_en_bola(p, D):
        return D
    return _welzl(P[:-1], R + [p])


def _bola_trivial(R: list) -> Bola:
    """Resuelve los casos base: 0, 1, 2 o 3 puntos en la frontera."""
    k = len(R)
    if k == 0:
        # Convenio: bola "vacia" - en la practica el algoritmo nunca llega aqui
        # con R vacio si P no es vacio, pero por seguridad:
        return (np.zeros(2, dtype=np.float64), 0.0)
    if k == 1:
        return bola_de_1_punto(R[0])
    if k == 2:
        return bola_de_2_puntos(R[0], R[1])
    # k == 3: tres puntos en la frontera = circuncirculo (o SEB si colineales)
    bola = bola_de_3_puntos(R[0], R[1], R[2])
    if bola is None:
        return bola_minima_3(R[0], R[1], R[2])
    return bola
