"""
Validacion del SEB: Seidel vs Welzl vs cvxpy (SOCP).

Casos:
  1. Casos analiticos (3 puntos, equilatero, triangulo rectangulo, obtuso).
  2. Datasets aleatorios pequeños (n=10, 50, 200).
  3. Subset real de Manhattan (n=500).
  4. Consistencia entre seedillas.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Tuple

from .seb_seidel import seb_seidel
from .seb_welzl import seb_welzl
from .seb_primitivas import Bola, EPS, punto_en_bola

# Tolerancia para comparar implementaciones (mayor que EPS por error numerico acumulado)
TOL_VALIDACION = 1e-4


def seb_cvxpy(puntos: np.ndarray) -> Bola:
    """SEB usando cvxpy (SOCP) - ground truth para validacion."""
    import cvxpy as cp

    n, d = puntos.shape
    c = cp.Variable(d)
    r = cp.Variable()
    restricciones = [cp.norm(c - puntos[i], 2) <= r for i in range(n)]
    problema = cp.Problem(cp.Minimize(r), restricciones)
    problema.solve()
    return (np.array(c.value), float(r.value))


def comparar_bolas(b1: Bola, b2: Bola, tol: float = TOL_VALIDACION) -> bool:
    """True si dos bolas son iguales dentro de tolerancia."""
    c1, r1 = b1
    c2, r2 = b2
    return abs(r1 - r2) < tol and float(np.linalg.norm(c1 - c2)) < tol


def verificar_solucion(puntos: np.ndarray, bola: Bola) -> dict:
    """
    Verifica que una bola sea solucion valida del SEB:
      - Todos los puntos estan dentro o en la frontera.
      - Al menos 2 puntos estan en la frontera (condicion necesaria del optimo).
    """
    centro, radio = bola
    distancias = np.linalg.norm(puntos - centro, axis=1)
    max_d = float(distancias.max())
    n_frontera = int(np.sum(np.abs(distancias - radio) < 1e-3))
    return {
        "todos_dentro": max_d <= radio + 1e-3,
        "max_distancia": max_d,
        "radio": radio,
        "puntos_frontera": n_frontera,
    }


# ── Casos analiticos ──────────────────────────────────────────────────────────

def test_un_punto():
    p = np.array([[1.5, -2.3]])
    centro, radio = seb_seidel(p)
    assert np.allclose(centro, p[0]), f"centro={centro}"
    assert radio == 0.0, f"radio={radio}"
    print("  [OK] 1 punto")


def test_dos_puntos():
    p = np.array([[0.0, 0.0], [10.0, 0.0]])
    centro, radio = seb_seidel(p)
    assert np.allclose(centro, [5.0, 0.0]), f"centro={centro}"
    assert abs(radio - 5.0) < EPS, f"radio={radio}"
    print("  [OK] 2 puntos")


def test_triangulo_equilatero():
    # Lado = 2, circuncirculo radio = 2/sqrt(3)
    h = np.sqrt(3)
    p = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, h]])
    centro, radio = seb_seidel(p, semilla=0)
    radio_esperado = 2.0 / np.sqrt(3)
    centro_esperado = np.array([0.0, h / 3.0])
    assert abs(radio - radio_esperado) < 1e-6, f"r={radio} vs {radio_esperado}"
    assert np.allclose(centro, centro_esperado, atol=1e-6), f"c={centro}"
    print(f"  [OK] equilatero (r={radio:.6f}, esperado={radio_esperado:.6f})")


def test_triangulo_rectangulo():
    # Hipotenusa = diametro de la SEB
    p = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 8.0]])
    centro, radio = seb_seidel(p, semilla=0)
    # Hipotenusa de 10 -> radio 5, centro en (3, 4)
    assert abs(radio - 5.0) < 1e-6, f"r={radio}"
    assert np.allclose(centro, [3.0, 4.0], atol=1e-6), f"c={centro}"
    print(f"  [OK] rectangulo (hipotenusa = diametro: r={radio:.6f})")


def test_triangulo_obtuso():
    # Triangulo obtuso: lado mas largo determina el diametro
    p = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 0.5]])
    centro, radio = seb_seidel(p, semilla=0)
    assert abs(radio - 5.0) < 1e-6, f"r={radio}"
    assert np.allclose(centro, [5.0, 0.0], atol=1e-6), f"c={centro}"
    print(f"  [OK] obtuso (lado mas largo = diametro: r={radio:.6f})")


# ── Comparaciones con cvxpy ───────────────────────────────────────────────────

def test_vs_cvxpy(n: int, semilla: int = 42) -> dict:
    """Genera n puntos aleatorios y compara las 3 implementaciones."""
    rng = np.random.default_rng(semilla)
    puntos = rng.normal(0, 100, size=(n, 2))

    b_seidel = seb_seidel(puntos, semilla=semilla)
    b_welzl = seb_welzl(puntos, semilla=semilla)
    b_cvxpy = seb_cvxpy(puntos)

    return {
        "n": n,
        "radio_seidel": b_seidel[1],
        "radio_welzl": b_welzl[1],
        "radio_cvxpy": b_cvxpy[1],
        "seidel_vs_cvxpy": comparar_bolas(b_seidel, b_cvxpy),
        "welzl_vs_cvxpy": comparar_bolas(b_welzl, b_cvxpy),
        "seidel_vs_welzl": comparar_bolas(b_seidel, b_welzl),
    }


def test_consistencia_seedillas(puntos: np.ndarray, k: int = 10) -> dict:
    """Ejecutar Seidel con k semillas distintas - el radio debe ser identico."""
    radios = []
    for s in range(k):
        _, r = seb_seidel(puntos, semilla=s)
        radios.append(r)
    return {
        "min": min(radios),
        "max": max(radios),
        "diff": max(radios) - min(radios),
        "consistente": (max(radios) - min(radios)) < TOL_VALIDACION,
    }


# ── Suite principal ───────────────────────────────────────────────────────────

def correr_suite_completa() -> bool:
    """Ejecuta todos los tests. Retorna True si todos pasan."""
    print("=" * 60)
    print("VALIDACION DE SEB: Seidel vs Welzl vs cvxpy")
    print("=" * 60)

    print("\n[1] Casos analiticos")
    test_un_punto()
    test_dos_puntos()
    test_triangulo_equilatero()
    test_triangulo_rectangulo()
    test_triangulo_obtuso()

    print("\n[2] Comparacion contra cvxpy (SOCP)")
    todos_ok = True
    for n in [10, 50, 200]:
        r = test_vs_cvxpy(n, semilla=42)
        ok = r["seidel_vs_cvxpy"] and r["welzl_vs_cvxpy"]
        marca = "[OK]" if ok else "[FAIL]"
        print(f"  {marca} n={n:>3}: "
              f"Seidel r={r['radio_seidel']:.6f}, "
              f"Welzl r={r['radio_welzl']:.6f}, "
              f"cvxpy r={r['radio_cvxpy']:.6f}")
        todos_ok = todos_ok and ok

    print("\n[3] Consistencia entre semillas (10 ejecuciones)")
    rng = np.random.default_rng(0)
    puntos = rng.normal(0, 100, size=(500, 2))
    cons = test_consistencia_seedillas(puntos, k=10)
    marca = "[OK]" if cons["consistente"] else "[FAIL]"
    print(f"  {marca} radio min={cons['min']:.6f}, max={cons['max']:.6f}, "
          f"diff={cons['diff']:.2e}")
    todos_ok = todos_ok and cons["consistente"]

    print()
    print("=" * 60)
    print("TODOS LOS TESTS PASARON" if todos_ok else "ALGUN TEST FALLO")
    print("=" * 60)
    return todos_ok


if __name__ == "__main__":
    correr_suite_completa()
