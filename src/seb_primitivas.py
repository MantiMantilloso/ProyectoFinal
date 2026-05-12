"""
Primitivas geometricas para el problema Smallest Enclosing Ball (SEB) en 2D.

Una "bola" se representa como una tupla (centro: np.ndarray shape (2,), radio: float).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# Tolerancia numerica para comparaciones (en metros, dado que trabajamos en UTM).
# 1e-7 m = 0.1 micras: muy estricto pero seguro para coordenadas de ~10^6 m.
EPS = 1e-7

Bola = Tuple[np.ndarray, float]


def bola_de_1_punto(p: np.ndarray) -> Bola:
    """Bola degenerada: centro en p, radio 0."""
    return (np.asarray(p, dtype=np.float64).copy(), 0.0)


def bola_de_2_puntos(p1: np.ndarray, p2: np.ndarray) -> Bola:
    """Bola con p1 y p2 en la frontera diametral (la mas pequeña que contiene ambos)."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    centro = (p1 + p2) / 2.0
    radio = float(np.linalg.norm(p2 - p1) / 2.0)
    return (centro, radio)


def bola_de_3_puntos(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Bola | None:
    """
    Circuncirculo de 3 puntos (los 3 quedan en la frontera).
    Si los puntos son colineales (no existe tal circulo), retorna None.

    Formula derivada trasladando p1 al origen y resolviendo el sistema lineal
    para que el centro sea equidistante a los 3 puntos.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    a = p2 - p1
    b = p3 - p1
    det = 2.0 * (a[0] * b[1] - a[1] * b[0])

    if abs(det) < EPS:
        return None  # colineales

    a_sq = a[0] ** 2 + a[1] ** 2
    b_sq = b[0] ** 2 + b[1] ** 2
    ux = (b[1] * a_sq - a[1] * b_sq) / det
    uy = (a[0] * b_sq - b[0] * a_sq) / det

    centro = p1 + np.array([ux, uy])
    radio = float(np.linalg.norm(centro - p1))
    return (centro, radio)


def punto_en_bola(p: np.ndarray, bola: Bola) -> bool:
    """True si p esta dentro o en la frontera de la bola (con tolerancia EPS)."""
    centro, radio = bola
    return float(np.linalg.norm(np.asarray(p) - centro)) <= radio + EPS


def bola_minima_2(p1: np.ndarray, p2: np.ndarray) -> Bola:
    """SEB de exactamente 2 puntos (alias de bola_de_2_puntos)."""
    return bola_de_2_puntos(p1, p2)


def bola_minima_3(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Bola:
    """
    SEB de 3 puntos: probar primero las 3 bolas con 2 puntos como diametro
    (la SEB puede ser una de ellas si el triangulo es obtuso). Si ninguna
    contiene al tercero, devolver el circuncirculo.
    """
    pts = [p1, p2, p3]
    candidatos: list[Bola] = []
    for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
        bola = bola_de_2_puntos(pts[i], pts[j])
        if punto_en_bola(pts[k], bola):
            candidatos.append(bola)
    if candidatos:
        return min(candidatos, key=lambda b: b[1])
    bola_3 = bola_de_3_puntos(p1, p2, p3)
    if bola_3 is None:
        # Caso degenerado (colineales): bola con los dos extremos como diametro
        max_d = -1.0
        mejor: Bola | None = None
        for i, j, k in [(0, 1, 2), (0, 2, 1), (1, 2, 0)]:
            d = float(np.linalg.norm(np.asarray(pts[i]) - np.asarray(pts[j])))
            if d > max_d:
                max_d = d
                mejor = bola_de_2_puntos(pts[i], pts[j])
        return mejor  # type: ignore[return-value]
    return bola_3
