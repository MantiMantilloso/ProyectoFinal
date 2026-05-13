"""Helpers de UI: conversiones UTM/WGS84 y funciones folium reutilizables."""
from __future__ import annotations
import numpy as np
import folium
from pyproj import Transformer
from shapely.geometry import Polygon

_T = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)


def utm_a_latlon(x: float, y: float) -> tuple[float, float]:
    lon, lat = _T.transform(x, y)
    return float(lat), float(lon)


def puntos_utm_a_latlon(xy: np.ndarray) -> list[tuple[float, float]]:
    """Convierte array (n, 2) UTM a lista de (lat, lon) — usa transformacion vectorizada."""
    lons, lats = _T.transform(xy[:, 0], xy[:, 1])
    return list(zip(lats.tolist(), lons.tolist()))


def pol_utm_a_latlon(pol: Polygon) -> list[tuple[float, float]]:
    coords = np.array(pol.exterior.coords)
    lons, lats = _T.transform(coords[:, 0], coords[:, 1])
    return list(zip(lats.tolist(), lons.tolist()))


def crear_mapa(lat: float = 40.776, lon: float = -73.971, zoom: int = 12) -> folium.Map:
    return folium.Map(location=(lat, lon), zoom_start=zoom, tiles="CartoDB positron")


def dibujar_zona(m: folium.Map, pol_utm: Polygon, color: str = "gray",
                 nombre: str = "", fill_opacity: float = 0.25) -> None:
    folium.Polygon(
        locations=pol_utm_a_latlon(pol_utm),
        color=color, fill=True, fill_color=color,
        fill_opacity=fill_opacity, tooltip=nombre, weight=1.5,
    ).add_to(m)


def dibujar_seb(m: folium.Map, c: np.ndarray, r: float,
                color: str = "black", label: str = "SEB") -> None:
    lat, lon = utm_a_latlon(c[0], c[1])
    folium.Circle(
        location=(lat, lon), radius=r, color=color,
        fill=False, weight=2.5,
        tooltip=f"{label}: r={r/1000:.3f} km",
    ).add_to(m)
    folium.CircleMarker(
        location=(lat, lon), radius=5,
        color=color, fill=True, fill_color=color,
        tooltip=f"Centro {label}",
    ).add_to(m)


def agregar_leyenda(m: folium.Map, items: list[tuple[str, str]]) -> None:
    """Superpone una leyenda HTML en la esquina inferior izquierda del mapa folium."""
    filas = "".join(
        f'<div style="display:flex;align-items:center;margin:3px 0;">'
        f'<span style="display:inline-block;width:12px;height:12px;background:{color};'
        f'border-radius:50%;margin-right:8px;flex-shrink:0;border:1px solid rgba(0,0,0,0.15);"></span>'
        f'<span style="color:#1e293b;font-size:12px;font-family:sans-serif;">{label}</span></div>'
        for color, label in items
    )
    html = (
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
        'background:#ffffff;padding:8px 12px;border-radius:6px;'
        'border:1px solid #d1d5db;font-size:12px;line-height:1.5;'
        'max-width:240px;box-shadow:0 2px 6px rgba(0,0,0,0.18);color:#1e293b;">'
        + filas + "</div>"
    )
    m.get_root().html.add_child(folium.Element(html))


def puntos_soporte(puntos: np.ndarray, c: np.ndarray, r: float,
                   tol: float = 1e-6) -> np.ndarray:
    """
    Puntos del input que estan en la frontera del SEB: ||p - c|| / r >= 1 - tol.

    En 2D el SEB tiene 2 o 3 puntos soporte (LP-type, dim combinatoria 3).
    """
    if r <= 0:
        return np.empty((0, 2))
    d = np.linalg.norm(puntos - c, axis=1)
    mask = d / r >= 1.0 - tol
    return puntos[mask]


def dibujar_soporte(m: folium.Map, soporte_utm: np.ndarray,
                    color: str = "#10b981", label: str = "Soporte") -> None:
    """Resalta los puntos soporte con marcador grande + halo."""
    for i, p in enumerate(soporte_utm):
        lat, lon = utm_a_latlon(p[0], p[1])
        folium.CircleMarker(
            location=(lat, lon), radius=11,
            color=color, fill=False, weight=2.5, opacity=0.8,
            tooltip=f"{label} #{i+1} (en la frontera del SEB)",
        ).add_to(m)
        folium.CircleMarker(
            location=(lat, lon), radius=5,
            color=color, fill=True, fill_color=color, fill_opacity=1.0,
            weight=0,
        ).add_to(m)


def dibujar_arista(m: folium.Map, p1: np.ndarray, p2: np.ndarray,
                   color: str = "#a855f7", label: str = "Arista",
                   weight: float = 4.5) -> None:
    """Resalta un segmento (e.g. la arista vinculante del SEB restringido)."""
    lat1, lon1 = utm_a_latlon(p1[0], p1[1])
    lat2, lon2 = utm_a_latlon(p2[0], p2[1])
    folium.PolyLine(
        locations=[(lat1, lon1), (lat2, lon2)],
        color=color, weight=weight, opacity=0.95, tooltip=label,
    ).add_to(m)


def ajustar_a_seb(m: folium.Map, c: np.ndarray, r: float,
                  margen: float = 1.05) -> None:
    """Encuadra el mapa al bounding box del SEB con un margen."""
    if c is None or r is None or not np.isfinite(r) or r <= 0:
        return
    rr = r * margen
    corners = [(c[0] - rr, c[1] - rr), (c[0] - rr, c[1] + rr),
               (c[0] + rr, c[1] - rr), (c[0] + rr, c[1] + rr)]
    latlons = [utm_a_latlon(x, y) for x, y in corners]
    lats = [ll[0] for ll in latlons]
    lons = [ll[1] for ll in latlons]
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])


def fmt_km(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v / 1000:.3f} km"


def fmt_m(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.1f} m"
