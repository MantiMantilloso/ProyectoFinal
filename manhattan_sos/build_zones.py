"""
Build manhattan_sos/data/zones.geojson from OpenStreetMap via Overpass API.

Each feature carries properties.category in:
  parque | hospital | escuela | iglesia | estadio | comercial | heliport | museo

Run once (or whenever you want to refresh data):
    python manhattan_sos/build_zones.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Source for Manhattan Community Districts (12 CDs + Central Park JIA).
# Authoritative NYC partition: every point of the island falls in exactly one.
CD_URL = "https://raw.githubusercontent.com/nycehs/NYC_geography/master/CD.geo.json"

# Manhattan bbox (S, W, N, E)
BBOX = (40.6995, -74.0250, 40.8820, -73.9070)

CATEGORIES = {
    "parque":    [("leisure", "park")],
    "hospital":  [("amenity", "hospital")],
    "escuela":   [("amenity", "school"), ("amenity", "university")],
    "iglesia":   [("amenity", "place_of_worship")],
    "comercial": [("shop", "mall")],
    "heliport":  [("aeroway", "heliport")],
    "museo":     [("tourism", "museum")],
}

OUT_PATH = Path(__file__).parent / "data" / "zones.geojson"


def overpass_query(tag_filters: list[tuple[str, str]]) -> dict:
    """Query Overpass for (way|relation) elements matching any of the given tag filters."""
    s, w, n, e = BBOX
    bbox_str = f"({s},{w},{n},{e})"

    blocks = []
    for key, value in tag_filters:
        blocks.append(f'  way["{key}"="{value}"]{bbox_str};')
        blocks.append(f'  relation["{key}"="{value}"]{bbox_str};')

    query = (
        "[out:json][timeout:90];\n"
        "(\n"
        + "\n".join(blocks)
        + "\n);\n"
        "out geom;"
    )

    body = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        OVERPASS_URL,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "manhattan-sos/0.1 (academic project)",
            "Accept": "application/json",
        },
    )

    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            print(f"  ! intento {attempt + 1} fallo: {exc}. reintentando en 5s...")
            time.sleep(5)
    raise RuntimeError(f"Overpass fallo tras 3 intentos: {last_err}")


def way_to_polygon(way: dict) -> list[list[float]] | None:
    """Build a GeoJSON-style ring [[lng, lat], ...] from an Overpass way with geometry."""
    geom = way.get("geometry") or []
    if len(geom) < 4:
        return None
    ring = [[pt["lon"], pt["lat"]] for pt in geom]
    if ring[0] != ring[-1]:
        ring.append(ring[0])  # close
    return ring


def relation_to_multipolygon(rel: dict) -> list[list[list[float]]]:
    """
    Build a MultiPolygon coordinates list from an Overpass relation with geometry.
    Each outer way becomes one polygon's outer ring (inner rings are ignored for simplicity).
    """
    polygons = []
    for member in rel.get("members", []):
        if member.get("type") != "way":
            continue
        if member.get("role") not in (None, "", "outer"):
            continue
        geom = member.get("geometry") or []
        if len(geom) < 4:
            continue
        ring = [[pt["lon"], pt["lat"]] for pt in geom]
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        polygons.append([ring])
    return polygons


def element_to_feature(el: dict, category: str) -> dict | None:
    tags = el.get("tags", {}) or {}
    name = tags.get("name") or tags.get("name:en") or category

    if el["type"] == "way":
        ring = way_to_polygon(el)
        if ring is None:
            return None
        geometry = {"type": "Polygon", "coordinates": [ring]}
    elif el["type"] == "relation":
        polys = relation_to_multipolygon(el)
        if not polys:
            return None
        geometry = {"type": "MultiPolygon", "coordinates": polys}
    else:
        return None

    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "category": category,
            "name": name,
            "osm_id": f'{el["type"][0]}/{el["id"]}',
        },
    }


def fetch_community_districts() -> list[dict]:
    """
    Download Manhattan Community Districts and Joint Interest Areas from nycehs/NYC_geography.
    Returns a list of GeoJSON Features with properties.category = "sector".
    Together these 13 polygons tile Manhattan with no gaps.
    """
    req = urllib.request.Request(
        CD_URL,
        headers={
            "User-Agent": "manhattan-sos/0.1 (academic project)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    out = []
    for f in raw.get("features", []):
        props = f.get("properties") or {}
        if props.get("BOROUGH") != "Manhattan":
            continue
        # Friendly name: strip the trailing " (CDx)" tag for cleaner UI labels.
        full_name = props.get("GEONAME") or "Sector"
        nice_name = full_name.split(" (CD")[0].strip()
        geocode = props.get("GEOCODE", 0)
        # CD64 is the Joint Interest Area covering Central Park (no CD owns it).
        if geocode == 0:
            nice_name = "Central Park"
        out.append({
            "type": "Feature",
            "geometry": f.get("geometry"),
            "properties": {
                "category": "sector",
                "name": nice_name,
                "osm_id": f"cd/{geocode}",
            },
        })
    return out


def build() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features: list[dict] = []
    counts: dict[str, int] = {}

    for category, filters in CATEGORIES.items():
        print(f"[fetch] {category} ({', '.join(f'{k}={v}' for k, v in filters)})...")
        result = overpass_query(filters)
        elements = result.get("elements", [])

        per_cat = 0
        for el in elements:
            feature = element_to_feature(el, category)
            if feature is None:
                continue
            features.append(feature)
            per_cat += 1

        counts[category] = per_cat
        print(f"  -> {per_cat} poligonos")
        time.sleep(1.0)  # be polite to overpass

    print("[fetch] sector (NYC community districts)...")
    sectors = fetch_community_districts()
    features.extend(sectors)
    counts["sector"] = len(sectors)
    print(f"  -> {len(sectors)} poligonos")

    fc = {"type": "FeatureCollection", "features": features}
    OUT_PATH.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")

    total = sum(counts.values())
    print(f"\n[ok] {total} poligonos -> {OUT_PATH}")
    print("     resumen por categoria:")
    for cat, n in counts.items():
        print(f"       {cat:10s} {n:4d}")


if __name__ == "__main__":
    build()
