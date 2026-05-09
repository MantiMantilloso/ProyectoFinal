"""
Limpieza y proyección de los datos de accidentes a coordenadas UTM Zone 18N.

Por qué UTM Zone 18N (EPSG:32618):
  - Unidad en metros → distancia euclidiana válida para el SEB.
  - Cubre toda NYC con distorsión mínima.
  - Sin proyección, 1° lon ≠ 1° lat en km (error ~26% en Manhattan).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_PROCESSED.mkdir(parents=True, exist_ok=True)

# Bounding box geográfico de Manhattan (WGS84)
LAT_MIN, LAT_MAX = 40.68, 40.88
LON_MIN, LON_MAX = -74.03, -73.90

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)


def limpiar_y_proyectar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame crudo de la API y proyecta lat/lon → UTM (metros).

    Pasos:
      1. Convertir columnas a tipos correctos.
      2. Eliminar filas con coordenadas fuera del bounding box de Manhattan.
      3. Proyectar a UTM Zone 18N (EPSG:32618) → columnas x_utm, y_utm.
      4. Eliminar duplicados exactos de posición.

    Returns:
        DataFrame limpio con columnas x_utm, y_utm en metros.
    """
    df = df.copy()

    # 1. Tipos
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce")
    for col in ["number_of_persons_injured", "number_of_persons_killed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    filas_inicio = len(df)

    # 2. Filtro geográfico (elimina coords (0,0) y puntos fuera de Manhattan)
    mascara = (
        df["latitude"].between(LAT_MIN, LAT_MAX)
        & df["longitude"].between(LON_MIN, LON_MAX)
    )
    df = df[mascara].dropna(subset=["latitude", "longitude"])

    # 3. Proyección UTM
    x, y = _transformer.transform(df["longitude"].values, df["latitude"].values)
    df = df.assign(x_utm=x, y_utm=y)

    # 4. Deduplicación
    df = df.drop_duplicates(subset=["x_utm", "y_utm"])

    filas_fin = len(df)
    print(
        f"[preprocesado] {filas_inicio:,} -> {filas_fin:,} filas "
        f"(eliminadas: {filas_inicio - filas_fin:,})"
    )
    return df.reset_index(drop=True)


def guardar_procesado(df: pd.DataFrame, nombre: str = "accidentes_utm.parquet") -> Path:
    """Guarda el DataFrame procesado en data/processed/."""
    ruta = DIR_PROCESSED / nombre
    df.to_parquet(ruta, index=False)
    print(f"[preprocesado] Guardado en {ruta}")
    return ruta


def cargar_procesado(nombre: str = "accidentes_utm.parquet") -> pd.DataFrame:
    """Carga el DataFrame procesado desde data/processed/."""
    ruta = DIR_PROCESSED / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta primero limpiar_y_proyectar()."
        )
    return pd.read_parquet(ruta)


def obtener_puntos(df: pd.DataFrame) -> np.ndarray:
    """
    Extrae las columnas x_utm, y_utm como array numpy de forma (n, 2).
    Listo para pasar al solver SEB.
    """
    return df[["x_utm", "y_utm"]].to_numpy(dtype=np.float64)


def muestra_estratificada(
    df: pd.DataFrame, n: int, semilla: int = 42
) -> np.ndarray:
    """
    Devuelve n puntos aleatorios del DataFrame para los experimentos de
    complejidad (Experimento 2). Estratificada por zip_code si está disponible.
    """
    if "zip_code" in df.columns:
        muestra = df.groupby("zip_code", group_keys=False).apply(
            lambda g: g.sample(frac=min(1.0, n / len(df)), random_state=semilla)
        )
        # Si la estratificación quedó corta, completar aleatoriamente
        if len(muestra) < n:
            muestra = df.sample(n=min(n, len(df)), random_state=semilla)
    else:
        muestra = df.sample(n=min(n, len(df)), random_state=semilla)

    return obtener_puntos(muestra.head(n))
