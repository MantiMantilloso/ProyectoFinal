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


def fmt_km(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v / 1000:.3f} km"


def fmt_m(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{v:.1f} m"
