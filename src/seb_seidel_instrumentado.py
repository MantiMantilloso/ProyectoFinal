"""
Version instrumentada de Seidel para visualizacion paso a paso.

Mantiene equivalencia con seb_seidel.py pero registra cada estado de la bola,
el punto procesado, la base (1/2/3 puntos en la frontera) y el evento
("inicio" / "dentro" / "actualiza_*"). El resultado se consume desde la app
Streamlit para animar la construccion del SEB.

NO se usa en produccion (notebooks usan seb_seidel.py). Esta version conserva
correctitud pero NO esta optimizada (copia listas, registra historial).
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from .seb_primitivas import (
    Bola,
    bola_de_1_punto,
    bola_de_2_puntos,
    bola_de_3_puntos,
    bola_minima_3,
    punto_en_bola,
)


def seb_seidel_instrumentado(puntos: np.ndarray, semilla: int = 42) -> List[dict]:
    """
    Ejecuta Seidel y devuelve la lista de pasos.

    Cada paso es un dict:
        i              : indice del punto procesado (segun la permutacion)
        punto_actual   : np.ndarray (2,)
        bola_antes     : (centro, radio) o None
        bola_despues   : (centro, radio)
        base           : list de np.ndarray (2 o 3 puntos en la frontera)
        evento         : 'inicio' | 'dentro' | 'actualiza_2' | 'actualiza_3'
        n_actualiza    : contador acumulado de actualizaciones
        orden_perm     : indices originales segun la permutacion (para overlay)
    """
    pts = np.asarray(puntos, dtype=np.float64)
    n = len(pts)
    if n < 2:
        raise ValueError("Se requieren al menos 2 puntos.")

    rng = np.random.default_rng(semilla)
    perm = rng.permutation(n)
    P = pts[perm].copy()

    pasos: List[dict] = []
    n_act = 0

    # Paso 0: inicializacion con los 2 primeros puntos (mismo que seb_seidel.py)
    bola = bola_de_2_puntos(P[0], P[1])
    n_act += 1
    pasos.append({
        "i": 1,
        "punto_actual": P[1].copy(),
        "bola_antes": None,
        "bola_despues": (bola[0].copy(), float(bola[1])),
        "base": [P[0].copy(), P[1].copy()],
        "evento": "inicio",
        "n_actualiza": n_act,
        "orden_perm": [int(perm[0]), int(perm[1])],
    })

    for i in range(2, n):
        p = P[i]
        if punto_en_bola(p, bola):
            pasos.append({
                "i": i,
                "punto_actual": p.copy(),
                "bola_antes": (bola[0].copy(), float(bola[1])),
                "bola_despues": (bola[0].copy(), float(bola[1])),
                "base": [],
                "evento": "dentro",
                "n_actualiza": n_act,
                "orden_perm": [int(perm[k]) for k in range(i + 1)],
            })
        else:
            bola_antes = (bola[0].copy(), float(bola[1]))
            bola, base = _seb_con_1_frontera_inst(P[:i], p)
            n_act += 1
            pasos.append({
                "i": i,
                "punto_actual": p.copy(),
                "bola_antes": bola_antes,
                "bola_despues": (bola[0].copy(), float(bola[1])),
                "base": [b.copy() for b in base],
                "evento": f"actualiza_{len(base)}",
                "n_actualiza": n_act,
                "orden_perm": [int(perm[k]) for k in range(i + 1)],
            })

    return pasos


def _seb_con_1_frontera_inst(P: np.ndarray, q: np.ndarray) -> Tuple[Bola, list]:
    bola = bola_de_2_puntos(P[0], q)
    base = [P[0], q]
    for i in range(1, len(P)):
        if not punto_en_bola(P[i], bola):
            bola, base = _seb_con_2_frontera_inst(P[:i], P[i], q)
    return bola, base


def _seb_con_2_frontera_inst(
    P: np.ndarray, q1: np.ndarray, q2: np.ndarray,
) -> Tuple[Bola, list]:
    bola = bola_de_2_puntos(q1, q2)
    base = [q1, q2]
    for i in range(len(P)):
        if not punto_en_bola(P[i], bola):
            bola_nueva = bola_de_3_puntos(q1, q2, P[i])
            if bola_nueva is not None:
                bola = bola_nueva
            else:
                bola = bola_minima_3(q1, q2, P[i])
            base = [q1, q2, P[i]]
    return bola, base


def filtrar_pasos_violadores(pasos: List[dict]) -> List[dict]:
    """Devuelve solo los pasos donde la bola fue actualizada (evento != 'dentro')."""
    return [s for s in pasos if s["evento"] != "dentro"]
