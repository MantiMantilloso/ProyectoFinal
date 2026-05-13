from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium.plugins import FastMarkerCluster
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing import cargar_procesado, obtener_puntos
from src.data_loader import obtener_zonas_osm
from src.constraints import cargar_zonas_utm, descomponer_zonas, semiplanos as calc_sps
from src.seb_seidel import seb_seidel
from src.seb_welzl import seb_welzl
from src.seb_restringido import seb_restringido
from src.aristas_completo import enumerar_aristas, sub_problema_1d
from src.app_helpers import (
    utm_a_latlon, puntos_utm_a_latlon, pol_utm_a_latlon,
    crear_mapa, dibujar_zona, dibujar_seb,
    dibujar_soporte, dibujar_arista, ajustar_a_seb,
    puntos_soporte, agregar_leyenda, fmt_km, fmt_m,
)

# ── Curva precalculada (notebook 05) ─────────────────────────────────────────
_CURVA = pd.DataFrame({
    "d_m":          [0, 100, 250, 500, 1000, 1500, 2500, 4000, 6000, 8000],
    "cobertura_pct":[6.9, 9.8, 14.3, 22.3, 38.5, 55.8, 78.7, 99.1, 100.0, 100.0],
    "estado":       ["restringido"]*8 + ["infactible"]*2,
    "radio_m":      [10429.3, 10432.9, 10440.0, 10456.7, 10508.1, 10583.1,
                     14618.6, 16610.9, float("nan"), float("nan")],
})

# ── Carga global (una vez por sesion) ────────────────────────────────────────
@st.cache_resource
def cargar_todo() -> dict:
    df = cargar_procesado()
    puntos_full = obtener_puntos(df)
    zonas_wgs = obtener_zonas_osm()
    zonas_utm = cargar_zonas_utm(zonas_wgs)
    R, prohibidas = descomponer_zonas(zonas_utm)
    zonas_dict = dict(prohibidas)

    c_lib, r_lib = seb_seidel(puntos_full, semilla=42)

    niveles = [
        ("Libre",                []),
        ("Solo Central Park",    ["central_park"]),
        ("Barrera Norte",        ["central_park", "morningside_park", "marcus_garvey_park"]),
        ("Seis zonas",           list(zonas_dict.keys())),
    ]
    sens: list[dict] = []
    for nombre, nombres_z in niveles:
        if not nombres_z:
            sens.append(dict(nombre=nombre, n_zonas=0, nombres_z=nombres_z,
                             centro=c_lib, radio=r_lib, estado="libre",
                             delta_r=0.0, delta_c=0.0, arista="-"))
        else:
            zonas_lvl = [(n, zonas_dict[n]) for n in nombres_z]
            res = seb_restringido(puntos_full, R, zonas_lvl, semilla=42)
            dr = res["radio"] - r_lib
            dc = (float(np.linalg.norm(res["centro"] - c_lib))
                  if res["centro"] is not None else float("nan"))
            arista = "-"
            if res["arista_activa"]:
                a = res["arista_activa"]
                arista = f"{a['fuente']}#{a['arista_idx']}"
            sens.append(dict(nombre=nombre, n_zonas=len(nombres_z), nombres_z=nombres_z,
                             centro=res["centro"], radio=res["radio"], estado=res["estado"],
                             delta_r=dr, delta_c=dc, arista=arista))

    zonas_info = [
        {"nombre": n, "n_semiplanos": len(calc_sps(p)), "area_km2": p.area / 1e6}
        for n, p in prohibidas
    ]
    n_sp_R = len(calc_sps(R))
    n_sp_total = n_sp_R + sum(z["n_semiplanos"] for z in zonas_info)

    return dict(df=df, puntos_full=puntos_full, R=R, prohibidas=prohibidas,
                zonas_dict=zonas_dict, c_lib_full=c_lib, r_lib_full=r_lib, sens=sens,
                zonas_info=zonas_info, n_sp_R=n_sp_R, n_sp_total=n_sp_total)


@st.cache_data(show_spinner=False)
def seb_libre_cached(puntos_bytes: bytes, shape: tuple, algoritmo: str,
                     semilla: int = 42):
    pts = np.frombuffer(puntos_bytes, dtype=np.float64).reshape(shape)
    fn = seb_seidel if algoritmo == "Seidel" else seb_welzl
    return fn(pts, semilla=semilla)


def calcular_libre(puntos: np.ndarray, algoritmo: str, semilla: int = 42):
    pts = np.ascontiguousarray(puntos, dtype=np.float64)
    return seb_libre_cached(pts.tobytes(), pts.shape, algoritmo, semilla)


def aplicar_filtros(df: pd.DataFrame, anos: list, h_rango: tuple,
                    severidades: list) -> tuple[pd.DataFrame, np.ndarray]:
    mask = df["crash_date"].dt.year.isin(anos)
    horas = pd.to_numeric(
        df["crash_time"].astype(str).str.split(":").str[0], errors="coerce"
    ).fillna(0).astype(int)
    mask &= horas.between(h_rango[0], h_rango[1])
    killed  = df["number_of_persons_killed"]
    injured = df["number_of_persons_injured"]
    sev = pd.Series(False, index=df.index)
    if "Fatales"     in severidades: sev |= killed > 0
    if "Con heridos" in severidades: sev |= (killed == 0) & (injured > 0)
    if "Sin heridos" in severidades: sev |= (killed == 0) & (injured == 0)
    mask &= sev
    df_f = df[mask].reset_index(drop=True)
    return df_f, df_f[["x_utm", "y_utm"]].to_numpy(dtype=float)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEB Manhattan - Cobertura de Emergencias",
    layout="wide",
)

# ── Estilos didacticos ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Caja de concepto clave (azul) */
.concepto {
    background: #1e3a5f;
    border-left: 4px solid #60a5fa;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 0.95rem;
    color: #e2e8f0 !important;
}
.concepto b, .concepto strong, .concepto code {
    color: #93c5fd !important;
}
/* Caja de hallazgo / resultado destacado (ambar) */
.hallazgo {
    background: #3d2e00;
    border-left: 4px solid #f59e0b;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 0.95rem;
    color: #fef3c7 !important;
}
.hallazgo b, .hallazgo strong, .hallazgo code {
    color: #fcd34d !important;
}
/* Caja de algoritmo / paso a paso (violeta) */
.algoritmo {
    background: #2d1b4e;
    border-left: 4px solid #a78bfa;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 10px 0;
    font-size: 0.95rem;
    color: #ede9fe !important;
}
.algoritmo b, .algoritmo strong, .algoritmo code {
    color: #c4b5fd !important;
}
/* Paso numerado */
.paso {
    display: inline-block;
    background: #3b82f6;
    color: white;
    border-radius: 50%;
    width: 22px;
    height: 22px;
    text-align: center;
    line-height: 22px;
    font-size: 12px;
    font-weight: bold;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Carga inicial ─────────────────────────────────────────────────────────────
try:
    with st.spinner("Cargando datos y calculando linea base (~10 s la primera vez)..."):
        datos = cargar_todo()
except FileNotFoundError:
    st.error("No se encontro data/processed/accidentes_utm.parquet. Ejecuta el notebook 01 primero.")
    st.stop()

N_TOTAL = len(datos["df"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## SEB Manhattan")
    st.markdown(
        "Encuentra la **ubicacion optima** para una central de emergencias "
        "que minimice la distancia maxima a cualquier accidente de transito, "
        "respetando que el centro no puede ubicarse dentro de un parque."
    )
    st.caption("Geometria Computacional — Semestre 8 | Mantilla & Soto")
    st.divider()

    st.subheader("Filtros del dataset")
    anos = st.multiselect("Ano", [2023, 2024], default=[2023, 2024])
    h_rango = st.slider("Hora del dia", 0, 23, (0, 23))
    st.markdown("**Severidad del accidente:**")
    severidades = (
        (["Fatales"]     if st.checkbox("Fatales",     value=True) else [])
        + (["Con heridos"] if st.checkbox("Con heridos", value=True) else [])
        + (["Sin heridos"] if st.checkbox("Sin heridos", value=True) else [])
    )
    st.divider()

    st.subheader("Algoritmo SEB libre")
    algoritmo = st.radio(
        "Metodo",
        ["Seidel", "Welzl"],
        help="Ambos tienen complejidad O(n) esperada pero implementaciones distintas.",
    )
    with st.expander("Diferencias entre Seidel y Welzl"):
        st.markdown("""
**Seidel** (iterativo, move-to-front):
Recorre los puntos en orden aleatorio. Si un punto queda fuera del SEB actual,
lo mueve al frente y recalcula. Muy cache-friendly.

**Welzl** (recursivo):
Divide y conquista con backtracking. Identifica el subconjunto minimo de puntos
que definen el circulo (base, maximo 3 puntos en 2D).

Ambos son equivalentes en resultado; Seidel suele ser mas rapido en la practica.
""")

if not anos or not severidades:
    st.warning("Selecciona al menos un ano y una categoria de severidad.")
    st.stop()

df_filt, puntos_filt = aplicar_filtros(datos["df"], anos, h_rango, severidades)
n_filt = len(df_filt)

filtros_hash = (tuple(sorted(anos)), tuple(h_rango), tuple(sorted(severidades)))
st.session_state["filtros_hash"] = filtros_hash

with st.sidebar:
    st.metric("Puntos seleccionados", f"{n_filt:,}",
              delta=f"de {N_TOTAL:,} totales",
              help="Accidentes de transito que cumplen los filtros activos")

if n_filt < 3:
    st.warning(f"Solo {n_filt} puntos tras filtrar. Ajusta los filtros.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 - Problema y Datos",
    "2 - SEB: Libre y Restringido",
    "3 - Aristas activas y sub-problema 1D",
    "4 - Sensibilidad Geometrica",
    "5 - Frontera de Factibilidad",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Problema y Datos
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("El Problema de Cobertura de Emergencias")

    st.markdown("""
    Dado un conjunto de **n puntos de demanda** (accidentes de transito en Manhattan),
    buscamos el circulo de radio minimo que los cubre a todos.
    El centro de ese circulo es la ubicacion optima para una central de emergencias.
    """)

    st.markdown(
        '<div class="concepto">'
        '<b>Smallest Enclosing Ball (SEB):</b> dado un conjunto P de n puntos en R², '
        'encontrar el centro c y radio r minimo tal que todos los puntos esten dentro '
        'del circulo. Es un problema LP-type → solucion en <b>O(n) esperado</b> '
        'con el algoritmo de Seidel o Welzl.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hallazgo">'
        '<b>Restriccion real:</b> el centro no puede ubicarse dentro de ninguno de los '
        '6 parques principales de Manhattan (zonas no operativas). Esto hace que el '
        'problema sea un SEB con restricciones geometricas sobre un conjunto factible '
        '<b>F no convexo</b> (Manhattan menos los parques).'
        '</div>',
        unsafe_allow_html=True,
    )

    c_lib, r_lib = calcular_libre(puntos_filt, algoritmo)
    soporte_lib = puntos_soporte(puntos_filt, c_lib, r_lib, tol=1e-6)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Puntos de demanda",
        f"{n_filt:,}",
        help="Accidentes de transito en Manhattan segun los filtros activos",
    )
    k2.metric(
        "Radio minimo sin restricciones",
        fmt_km(r_lib),
        help="El menor radio posible si ignoramos las zonas prohibidas",
    )
    k3.metric(
        "Puntos soporte",
        f"{len(soporte_lib)}",
        help="Puntos en la frontera del SEB (||p - c|| = r). "
             "En 2D el SEB queda determinado por 2 o 3 puntos (dim combinatoria 3).",
    )
    k4.metric(
        "Zonas prohibidas",
        "6 parques",
        help="Central Park, Morningside, Marcus Garvey, Inwood Hill, Fort Tryon, Battery",
    )
    k5.metric(
        "Semiplanos totales",
        str(datos["n_sp_total"]),
        help=(
            f"Manhattan (R): {datos['n_sp_R']} semiplanos + "
            f"6 parques: {datos['n_sp_total'] - datos['n_sp_R']} semiplanos. "
            "Cada arista del convex hull OSM define un semiplano."
        ),
    )

    lat0, lon0 = utm_a_latlon(*c_lib)
    m1 = crear_mapa(lat0, lon0, zoom=12)
    folium.Polygon(
        pol_utm_a_latlon(datos["R"]),
        color="#1e40af", fill=False, weight=2,
        tooltip="Region factible R: Manhattan (convex hull OSM)",
        dash_array="6",
    ).add_to(m1)
    for nombre, pol in datos["prohibidas"]:
        dibujar_zona(m1, pol, color="#dc2626",
                     nombre=f"Zona prohibida: {nombre.replace('_', ' ').title()}",
                     fill_opacity=0.35)
    FastMarkerCluster(
        puntos_utm_a_latlon(df_filt[["x_utm", "y_utm"]].values),
    ).add_to(m1)
    dibujar_seb(m1, c_lib, r_lib, color="#1e40af", label="SEB libre")
    dibujar_soporte(m1, soporte_lib, color="#10b981", label="Soporte libre")
    agregar_leyenda(m1, [
        ("#1e40af", f"SEB libre — r = {fmt_km(r_lib)}"),
        ("#10b981", f"Puntos soporte ({len(soporte_lib)})"),
        ("#1e40af", "Region factible (Manhattan)"),
        ("#dc2626", "Zonas prohibidas (parques)"),
        ("#6baed6", "Accidentes de transito"),
    ])
    ajustar_a_seb(m1, c_lib, r_lib, margen=1.06)
    st_folium(m1, use_container_width=True, height=520, key="m1")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        with st.expander("Como se modela el conjunto factible F"):
            st.markdown("""
**F = Manhattan \\ (Central Park ∪ Morningside ∪ Marcus Garvey ∪ Inwood Hill ∪ Fort Tryon ∪ Battery)**

F es **no convexo** — tiene 6 "huecos". Cada parque se representa como su
convex hull real (poligono OSM), lo que genera un conjunto de semiplanos con
normales arbitrarias (no solo axis-aligned).

El centro optimo con restricciones debe estar en F, es decir:
- Dentro del convex hull de Manhattan
- Fuera de todos los parques
""")
    with col_exp2:
        with st.expander("Detalle de zonas prohibidas"):
            st.dataframe(
                pd.DataFrame([
                    {
                        "Zona": z["nombre"].replace("_", " ").title(),
                        "Semiplanos": z["n_semiplanos"],
                        "Area (km2)": f"{z['area_km2']:.2f}",
                    }
                    for z in datos["zonas_info"]
                ] + [
                    {"Zona": "Manhattan (R)", "Semiplanos": datos["n_sp_R"], "Area (km2)": "—"},
                    {"Zona": "TOTAL", "Semiplanos": datos["n_sp_total"], "Area (km2)": "—"},
                ]),
                hide_index=True,
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEB: Libre y Restringido
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Del SEB Libre al SEB Restringido")

    c_lib2, r_lib2 = calcular_libre(puntos_filt, algoritmo)
    lat_lib, lon_lib = utm_a_latlon(*c_lib2)

    # ── Paso 1: SEB libre ────────────────────────────────────────────────────
    st.markdown(
        '<span class="paso">1</span> <b>SEB sin restricciones</b> — '
        f'Algoritmo: <code>{algoritmo}</code>',
        unsafe_allow_html=True,
    )

    col_par_txt, col_par_met = st.columns([3, 2])
    with col_par_txt:
        st.markdown(
            '<div class="hallazgo">'
            '<b>Paradoja geometrica:</b> el centro optimo <em>sin restricciones</em> '
            'cae <b>dentro de Central Park</b> (lat 40.787, lon -73.964) — '
            'exactamente donde no puede ubicarse la central de emergencias. '
            'Esto justifica la necesidad del problema con restricciones.'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_par_met:
        st.metric("Centro libre (lat)", f"{lat_lib:.4f} N")
        st.metric("Centro libre (lon)", f"{lon_lib:.4f} E")
        st.metric("Radio libre", fmt_km(r_lib2))

    st.divider()

    # ── Paso 2: SEB restringido ───────────────────────────────────────────────
    st.markdown(
        '<span class="paso">2</span> <b>SEB con restricciones</b> — '
        'Enumeracion de aristas activas',
        unsafe_allow_html=True,
    )

    with st.expander("Como funciona el algoritmo de aristas activas"):
        st.markdown(
            '<div class="algoritmo">'
            '<b>Teorema:</b> el centro optimo restringido c* cae en una de dos posiciones:<br>'
            '&nbsp;&nbsp;(a) el mismo centro libre (si ya esta en F), o<br>'
            '&nbsp;&nbsp;(b) sobre exactamente <b>una arista</b> del borde de F.<br><br>'
            '<b>Algoritmo:</b><br>'
            '&nbsp;&nbsp;Para cada arista del borde de F (260 en total):<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;1. Parametrizar el segmento como c(t) = P + t·(Q-P), t ∈ [0,1]<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;2. El radio maximo al mover el centro a lo largo del segmento '
            'es una funcion convexa en t (maximo de cuadraticas)<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;3. Minimizar con scipy minimize_scalar (~30 iteraciones)<br>'
            '&nbsp;&nbsp;&nbsp;&nbsp;4. Verificar que c(t*) este en F<br>'
            '&nbsp;&nbsp;Devolver la arista con menor radio factible.<br><br>'
            '<b>Complejidad:</b> O(A · n) donde A = 260 aristas → ~466 ms para n = 7,947'
            '</div>',
            unsafe_allow_html=True,
        )

    zonas_dict    = datos["zonas_dict"]
    nombres_todos = list(zonas_dict.keys())

    st.markdown("**Zonas prohibidas activas:**")
    c3 = st.columns(6)
    zonas_activas = [
        n for i, n in enumerate(nombres_todos)
        if c3[i].checkbox(
            n.replace("_", " ").title(), value=True, key=f"z_{n}"
        )
    ]

    # Recalculo automatico al cambiar toggles
    if zonas_activas:
        zonas_sel = [(n, zonas_dict[n]) for n in zonas_activas]
        res2 = seb_restringido(puntos_filt, datos["R"], zonas_sel, semilla=42)
    else:
        res2 = {"estado": "libre", "centro": c_lib2, "radio": r_lib2,
                "arista_activa": None}

    estado2  = res2["estado"]
    c_restr2 = res2["centro"]
    r_restr2 = res2["radio"]
    arista2  = res2.get("arista_activa")
    z2_used  = set(zonas_activas)

    if estado2 == "infactible":
        st.error("F = vacio: las zonas activas cubren todo Manhattan. "
                 "No existe punto factible.")

    # Soporte de ambos SEBs
    soporte_lib2 = puntos_soporte(puntos_filt, c_lib2, r_lib2, tol=1e-6)
    soporte_rst2 = (
        puntos_soporte(puntos_filt, c_restr2, r_restr2, tol=1e-6)
        if estado2 != "infactible" and c_restr2 is not None
        else np.empty((0, 2))
    )

    kc = st.columns(5)
    kc[0].metric("Radio libre", fmt_km(r_lib2),
                 help="Minimo posible sin restricciones")
    kc[1].metric(
        "Radio restringido",
        fmt_km(r_restr2) if estado2 != "infactible" else "Infactible",
        help="Radio con las zonas activas",
    )
    if estado2 != "infactible" and np.isfinite(r_restr2):
        delta_r = r_restr2 - r_lib2
        kc[2].metric("Costo geometrico (Delta r)", fmt_m(delta_r),
                     help="Cuanto crece el radio por efecto de las restricciones")
        if c_restr2 is not None:
            delta_c = float(np.linalg.norm(c_restr2 - c_lib2))
            kc[3].metric("Desplazamiento del centro", fmt_m(delta_c),
                         help="Distancia entre el centro libre y el restringido")
    if arista2:
        kc[4].metric("Arista vinculante",
                     f"{arista2['fuente']}#{arista2['arista_idx']}",
                     help="La unica arista del borde de F que determina la solucion optima")

    m2 = crear_mapa(*utm_a_latlon(*c_lib2), zoom=12)
    for nombre, pol in datos["prohibidas"]:
        if nombre in z2_used:
            dibujar_zona(m2, pol, color="#dc2626",
                         nombre=f"Activa: {nombre.replace('_', ' ').title()}",
                         fill_opacity=0.40)
        else:
            dibujar_zona(m2, pol, color="#9ca3af",
                         nombre=f"Inactiva: {nombre.replace('_', ' ').title()}",
                         fill_opacity=0.10)

    dibujar_seb(m2, c_lib2, r_lib2, color="#1e40af", label="Libre")
    dibujar_soporte(m2, soporte_lib2, color="#1e40af", label="Soporte libre")

    if estado2 != "infactible" and c_restr2 is not None:
        dibujar_seb(m2, c_restr2, r_restr2, color="#dc2626", label="Restringido")
        dibujar_soporte(m2, soporte_rst2, color="#dc2626", label="Soporte restringido")

    if arista2 and "p1" in arista2 and "p2" in arista2:
        dibujar_arista(m2, arista2["p1"], arista2["p2"],
                       color="#a855f7",
                       label=f"Arista vinculante: {arista2['fuente']}#{arista2['arista_idx']}",
                       weight=5.0)

    leyenda = [("#1e40af", f"SEB libre (r = {fmt_km(r_lib2)})")]
    if estado2 != "infactible":
        leyenda.append(("#dc2626", f"SEB restringido (r = {fmt_km(r_restr2)})"))
    if arista2:
        leyenda.append(("#a855f7", "Arista vinculante"))
    leyenda.extend([
        ("#dc2626", "Zonas prohibidas activas"),
        ("#9ca3af", "Zonas inactivas"),
    ])
    agregar_leyenda(m2, leyenda)
    r_view = max(r_lib2, r_restr2 if np.isfinite(r_restr2) else 0)
    ajustar_a_seb(m2, c_lib2, r_view, margen=1.06)
    st_folium(m2, use_container_width=True, height=520, key="m2")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Aristas activas + sub-problema 1D convexo
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _enum_aristas_cached(puntos_bytes: bytes, shape: tuple,
                         zonas_key: tuple, semilla: int = 42):
    """Cache de enumerar_aristas (depende de puntos + zonas activas)."""
    pts = np.frombuffer(puntos_bytes, dtype=np.float64).reshape(shape)
    z_act = [n for n in zonas_key if n]
    z_full = datos["zonas_dict"]
    zonas_sel = [(n, z_full[n]) for n in z_act] if z_act else []
    return enumerar_aristas(pts, datos["R"], zonas_sel, semilla=semilla)


with tab3:
    st.header("Algoritmo de Aristas Activas: enumeracion + sub-problema 1D")

    st.markdown(
        '<div class="algoritmo">'
        'El SEB restringido se reduce a un problema discreto-continuo. '
        '<b>Discreto:</b> enumerar las <b>aristas del borde de F</b> (260 en este problema). '
        '<b>Continuo:</b> en cada arista resolver un sub-problema 1D convexo:<br>'
        '&nbsp;&nbsp;<code>min_t in [0,1]  max_i  ||p_i - (P + t(Q-P))||&sup2;</code><br>'
        'Cada termino <code>||p_i - c(t)||&sup2;</code> es una parabola en <code>t</code> '
        'con el mismo coeficiente lider <code>||v||&sup2;</code>, asi que su pointwise max '
        'es <b>convexo en t</b> -> bisseccion de <code>scipy.optimize.minimize_scalar</code> '
        'converge garantizada al optimo de la arista.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Toggles de zonas (independientes del Tab 2 para no acoplar)
    zonas_dict_t3 = datos["zonas_dict"]
    nombres_t3 = list(zonas_dict_t3.keys())
    st.markdown("**Zonas prohibidas en el problema:**")
    cz3 = st.columns(6)
    zonas_act_t3 = [
        n for i, n in enumerate(nombres_t3)
        if cz3[i].checkbox(n.replace("_", " ").title(), value=True, key=f"z3_{n}")
    ]

    # Enumeracion completa (cached)
    z3_key = tuple(sorted(zonas_act_t3)) if zonas_act_t3 else ()
    pts_t3 = np.ascontiguousarray(datos["puntos_full"], dtype=np.float64)
    enum = _enum_aristas_cached(pts_t3.tobytes(), pts_t3.shape, z3_key, 42)

    candidatos = enum["candidatos"]
    n_total = enum["n_total"]
    n_factibles = enum["n_factibles"]
    mejor = enum["mejor"]
    r_libre_t3 = enum["radio_libre"]
    estado_t3 = enum["estado"]

    # KPIs
    k = st.columns(5)
    k[0].metric("Aristas evaluadas", f"{n_total}",
                help="Suma de aristas de R + zonas prohibidas activas")
    k[1].metric("Candidatos factibles", f"{n_factibles}",
                help="Aristas cuyo (c(t*), r(t*)) cae en F")
    if mejor:
        k[2].metric("Mejor radio (r*)", fmt_km(mejor["r"]),
                    help="Minimo entre los factibles")
        k[3].metric("Arista vinculante",
                    f"{mejor['fuente']}#{mejor['arista_idx']}",
                    help=f"t* = {mejor['t']:.4f}")
        k[4].metric("Delta r vs libre", fmt_m(mejor["r"] - r_libre_t3),
                    help="Costo geometrico de las restricciones")
    elif estado_t3 == "libre":
        k[2].metric("Estado", "Libre", help="El centro libre esta en F")
        k[3].metric("Radio libre", fmt_km(r_libre_t3))
    else:
        k[2].metric("Estado", "Infactible")

    # ── Panel A: scatter de los 260 candidatos ──────────────────────────────
    col_bars, col_1d = st.columns([1, 1])

    with col_bars:
        st.markdown("**Distribucion del radio sobre las 260 candidatas**")
        st.caption(
            "Cada punto = una arista. Color = factibilidad. El optimo global es "
            "el minimo entre los factibles."
        )

        # Index global con grupos por fuente
        fuentes_ord = ["region_factible"] + [n for n in nombres_t3 if n in zonas_act_t3]
        # mapping fuente -> color
        col_fuente = {
            "region_factible":   "#94a3b8",  # gris
            "central_park":      "#dc2626",  # rojo principal
            "morningside_park":  "#f87171",
            "marcus_garvey_park":"#fca5a5",
            "inwood_hill_park":  "#fb923c",
            "fort_tryon_park":   "#f59e0b",
            "battery_park":      "#a3e635",
        }

        # Ordenar candidatos por fuente y por arista_idx (preserva la enumeracion natural)
        candidatos_ord = sorted(
            candidatos,
            key=lambda d: (fuentes_ord.index(d["fuente"]) if d["fuente"] in fuentes_ord else 99,
                           d["arista_idx"]),
        )

        fig3, ax3 = plt.subplots(figsize=(7, 4.5))
        for i, cand in enumerate(candidatos_ord):
            color = col_fuente.get(cand["fuente"], "#94a3b8")
            mk = "o" if cand["factible"] else "x"
            alpha = 0.85 if cand["factible"] else 0.45
            ax3.scatter(i, cand["r"] / 1000, c=color, s=20,
                        marker=mk, alpha=alpha, edgecolors="none", zorder=2)

        if mejor:
            # Encontrar la posicion del ganador en el orden
            for i, cand in enumerate(candidatos_ord):
                if (cand["fuente"] == mejor["fuente"]
                        and cand["arista_idx"] == mejor["arista_idx"]):
                    ax3.scatter(i, cand["r"] / 1000, s=140, facecolors="none",
                                edgecolors="#a855f7", linewidths=2.5, zorder=5,
                                label=f"Ganadora: {mejor['fuente']}#{mejor['arista_idx']}")
                    break
            ax3.axhline(mejor["r"] / 1000, color="#a855f7", linewidth=1.0,
                        linestyle="--", alpha=0.6,
                        label=f"r* = {mejor['r']/1000:.3f} km")

        ax3.axhline(r_libre_t3 / 1000, color="#1e40af", linewidth=1.0,
                    linestyle=":", alpha=0.7,
                    label=f"r libre = {r_libre_t3/1000:.3f} km (cota inf.)")

        # Separadores verticales entre fuentes
        idx_acum = 0
        for f in fuentes_ord:
            n_f = sum(1 for d in candidatos_ord if d["fuente"] == f)
            idx_acum += n_f
            ax3.axvline(idx_acum - 0.5, color="#e5e7eb", linewidth=0.8,
                        alpha=0.5, zorder=1)

        # Etiquetas de grupo en el eje X
        cumul = 0
        for f in fuentes_ord:
            n_f = sum(1 for d in candidatos_ord if d["fuente"] == f)
            mid = cumul + n_f / 2
            ax3.text(mid, ax3.get_ylim()[1], f.replace("_", " ").title(),
                     ha="center", va="bottom", fontsize=7, color="#475569",
                     rotation=0, alpha=0.85)
            cumul += n_f

        ax3.set_xlabel("Arista (agrupada por fuente)")
        ax3.set_ylabel("Radio r (km) del SEB-en-segmento")
        ax3.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax3.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown(
            '<div class="concepto">'
            '<b>Lectura:</b> de las <b>'
            f'{n_total}</b> aristas evaluadas, <b>{n_factibles}</b> dan un centro '
            'factible. El minimo entre ellos es la solucion global. Los puntos '
            'circulares (o) son factibles; las X son infactibles.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Panel B: sub-problema 1D para una arista a inspeccionar ─────────────
    with col_1d:
        st.markdown("**Sub-problema 1D convexo sobre una arista**")

        # Selector
        col_sel_f, col_sel_i = st.columns([2, 1])
        with col_sel_f:
            fuente_sel = st.selectbox(
                "Fuente (zona o R)", fuentes_ord,
                index=fuentes_ord.index(mejor["fuente"]) if mejor else 0,
                key="sel_f3",
            )
        with col_sel_i:
            n_aristas_fuente = sum(
                1 for d in candidatos if d["fuente"] == fuente_sel
            )
            default_idx = mejor["arista_idx"] if (mejor and mejor["fuente"] == fuente_sel) else 0
            arista_sel = st.number_input(
                "Indice arista",
                0, max(0, n_aristas_fuente - 1),
                value=default_idx, step=1, key="sel_i3",
            )

        # Recuperar la arista
        cand_sel = next(
            (d for d in candidatos
             if d["fuente"] == fuente_sel and d["arista_idx"] == int(arista_sel)),
            None,
        )
        if cand_sel is None:
            st.warning("Arista no encontrada.")
        else:
            sub = sub_problema_1d(
                datos["puntos_full"], cand_sel["p1"], cand_sel["p2"],
                n_t=200, top_k=6,
            )

            es_ganadora = (
                mejor is not None
                and cand_sel["fuente"] == mejor["fuente"]
                and cand_sel["arista_idx"] == mejor["arista_idx"]
            )

            # KPIs de la arista
            ksel = st.columns(4)
            ksel[0].metric("t*", f"{sub['t_estrella']:.4f}",
                           help="Posicion del minimo a lo largo del segmento")
            ksel[1].metric("r*", fmt_km(sub["r_estrella"]),
                           help="Radio optimo en este sub-problema 1D")
            ksel[2].metric("Factible?", "Si" if cand_sel["factible"] else "No",
                           help="c(t*) cae en F?")
            ksel[3].metric("Vinculante?", "Si" if es_ganadora else "No",
                           help="Es la arista que define el optimo global?")

            # Plot del sub-problema 1D
            fig1d, ax1d = plt.subplots(figsize=(7, 4.5))

            # Top-K parabolas individuales
            for par in sub["parabolas"]:
                color_p = "#16a34a" if par["es_soporte"] else "#cbd5e1"
                alpha_p = 0.85 if par["es_soporte"] else 0.40
                width_p = 1.5 if par["es_soporte"] else 0.9
                label_p = (f"|p_{par['i']} - c(t)|^2  (soporte)"
                           if par["es_soporte"] else None)
                ax1d.plot(sub["t"], np.sqrt(par["curva"]),
                          color=color_p, alpha=alpha_p, linewidth=width_p,
                          label=label_p, zorder=2)

            # Envolvente superior f(t) -> r(t) = sqrt(f)
            ax1d.plot(sub["t"], np.sqrt(sub["f"]),
                      color="#1e40af", linewidth=2.5, alpha=0.95,
                      label="max_i ||p_i - c(t)|| (envolvente, convexa)",
                      zorder=4)

            # t* vertical + r* horizontal
            ax1d.axvline(sub["t_estrella"], color="#a855f7", linewidth=1.4,
                         linestyle="--", alpha=0.8,
                         label=f"t* = {sub['t_estrella']:.4f}")
            ax1d.axhline(sub["r_estrella"] / 1000, color="#a855f7",
                         linewidth=1.0, linestyle=":", alpha=0.6)
            ax1d.scatter([sub["t_estrella"]], [sub["r_estrella"] / 1000],
                         s=120, color="#a855f7", edgecolors="#0f172a",
                         linewidths=1.0, zorder=10,
                         label=f"r* = {sub['r_estrella']/1000:.3f} km")

            ax1d.set_xlabel(r"t  (centro c(t) = P + t (Q - P), t in [0, 1])")
            ax1d.set_ylabel("Distancia (km)")
            ax1d.set_xlim(0, 1)
            ax1d.legend(loc="upper center", fontsize=7, framealpha=0.92,
                        ncol=2)
            ax1d.grid(alpha=0.25)
            plt.tight_layout()
            st.pyplot(fig1d)
            plt.close(fig1d)

            st.markdown(
                '<div class="hallazgo">'
                f'<b>Convexidad visible:</b> la envolvente azul (max de '
                f'{len(sub["parabolas"])} parabolas con el mismo <code>||v||&sup2;</code>) '
                'es convexa en t. El minimo en <b>t* = '
                f'{sub["t_estrella"]:.4f}</b> esta tocado por '
                f'{len(sub["soporte_idx"])} punto(s) soporte (en verde). '
                f'Para esta arista <code>r = {sub["r_estrella"]/1000:.3f} km</code>.'
                '</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Sensibilidad Geometrica
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Sensibilidad Geometrica: que zonas importan?")

    st.markdown("""
    Agregamos las zonas prohibidas una a una y medimos como cambia la solucion.
    El objetivo es identificar cuales zonas son **geometricamente vinculantes**
    (obligan al centro a moverse) y cuales son irrelevantes para el optimo.
    """)
    st.info(
        "Este experimento usa siempre los 7,947 puntos completos del dataset. "
        "Los filtros del sidebar no se aplican aqui."
    )

    st.markdown(
        '<div class="hallazgo">'
        '<b>Resultado:</b> solo <b>Central Park</b> es vinculante. Agregar Morningside, '
        'Marcus Garvey, Inwood Hill, Fort Tryon o Battery Park no desplaza el centro '
        'ni modifica el radio. El optimo ya esta sobre el borde sur de Central Park '
        '(arista #24) — las demas zonas no bloquean ninguna direccion de mejora.'
        '</div>',
        unsafe_allow_html=True,
    )

    sens = datos["sens"]

    st.dataframe(
        pd.DataFrame([{
            "Configuracion":       s["nombre"],
            "Zonas activas":       s["n_zonas"],
            "Estado":              s["estado"],
            "Radio (km)":          f"{s['radio']/1000:.4f}" if np.isfinite(s["radio"]) else "N/A",
            "Delta r (m)":         f"{s['delta_r']:+.2f}"   if np.isfinite(s["delta_r"]) else "N/A",
            "Despl. centro (m)":   f"{s['delta_c']:.1f}"    if np.isfinite(s["delta_c"]) else "N/A",
            "Arista vinculante":   s["arista"],
        } for s in sens]),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        '<div class="concepto">'
        '<b>Interpretacion:</b> "Barrera Norte" (Central Park + Morningside + Marcus Garvey) '
        'da exactamente el mismo resultado que "Solo Central Park". El centro no escapa '
        'hacia el norte porque moverlo al <em>sur</em> del parque minimiza el radio — '
        'la barrera norte no es geometricamente relevante.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Trayectoria del centro (zoom a Central Park) ─────────────────────────
    st.markdown("#### Trayectoria del centro entre configuraciones")
    centros_tr = [s["centro"] for s in sens if s["centro"] is not None]
    nombres_tr = [s["nombre"] for s in sens if s["centro"] is not None]
    radios_tr  = [s["radio"]  for s in sens if s["centro"] is not None]

    if len(centros_tr) >= 2:
        cs = np.array(centros_tr)

        fig_tr, ax_tr = plt.subplots(figsize=(8, 4))

        # Parques como fondo
        for nombre, pol in datos["prohibidas"]:
            xx, yy = pol.exterior.xy
            color_pol = "#fca5a5" if nombre == "central_park" else "#e5e7eb"
            ax_tr.fill(xx, yy, color=color_pol, alpha=0.55, zorder=1)
            ax_tr.plot(xx, yy, color="#9ca3af", linewidth=0.8, zorder=2)
            # etiqueta del parque
            cx, cy = pol.centroid.x, pol.centroid.y
            ax_tr.text(cx, cy, nombre.replace("_", " ").title(),
                       ha="center", va="center", fontsize=7,
                       color="#6b7280", alpha=0.85, zorder=3)

        # Trayectoria: flechas entre configuraciones consecutivas
        for k in range(len(cs) - 1):
            dx = cs[k + 1, 0] - cs[k, 0]
            dy = cs[k + 1, 1] - cs[k, 1]
            if dx == 0 and dy == 0:
                continue
            ax_tr.annotate(
                "", xy=(cs[k + 1, 0], cs[k + 1, 1]),
                xytext=(cs[k, 0], cs[k, 1]),
                arrowprops=dict(arrowstyle="->", color="#475569",
                                lw=1.4, alpha=0.85),
                zorder=4,
            )

        # Marcadores de cada configuracion
        for k, (c, nombre, r) in enumerate(zip(cs, nombres_tr, radios_tr)):
            color_m = "#1e40af" if k == 0 else "#dc2626"
            ax_tr.scatter(c[0], c[1], s=110, color=color_m,
                          edgecolors="white", linewidths=1.8, zorder=5)
            ax_tr.annotate(
                f"{nombre}\nr = {r/1000:.3f} km",
                xy=(c[0], c[1]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8.5, color="#1f2937",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor=color_m,
                          alpha=0.92, linewidth=0.8),
                zorder=6,
            )

        # Acotar el view alrededor de Central Park + centros
        cp_xy = np.array(datos["zonas_dict"]["central_park"].exterior.coords)
        all_x = np.concatenate([cs[:, 0], cp_xy[:, 0]])
        all_y = np.concatenate([cs[:, 1], cp_xy[:, 1]])
        pad = max((all_x.max() - all_x.min()),
                  (all_y.max() - all_y.min())) * 0.12
        ax_tr.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax_tr.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax_tr.set_aspect("equal")
        ax_tr.set_xlabel("X (UTM Zone 18N, metros)", fontsize=9)
        ax_tr.set_ylabel("Y (UTM Zone 18N, metros)", fontsize=9)
        ax_tr.tick_params(labelsize=7)
        ax_tr.grid(alpha=0.25)
        ax_tr.set_title(
            "Los 3 centros restringidos colapsan en el mismo punto: "
            "solo Central Park es vinculante",
            fontsize=9, color="#475569", pad=8,
        )
        plt.tight_layout()
        st.pyplot(fig_tr)
        plt.close(fig_tr)

    st.markdown("#### Comparacion visual — 4 configuraciones")
    cols_s = st.columns(2)
    for i, s in enumerate(sens):
        with cols_s[i % 2]:
            st.markdown(
                f"**{s['nombre']}** &nbsp;|&nbsp; {s['n_zonas']} zona(s) "
                f"&nbsp;|&nbsp; `{s['estado']}`"
            )
            if np.isfinite(s["radio"]):
                st.caption(
                    f"Radio: {fmt_km(s['radio'])}  |  "
                    f"Delta r: {s['delta_r']:+.1f} m  |  "
                    f"Arista: {s['arista']}"
                )
            c = s["centro"]
            r = s["radio"]
            lat0, lon0 = utm_a_latlon(*c) if c is not None else (40.776, -73.971)
            ms = crear_mapa(lat0, lon0, zoom=11)
            for nombre, pol in datos["prohibidas"]:
                if nombre in s["nombres_z"]:
                    dibujar_zona(
                        ms, pol, color="#dc2626",
                        nombre=nombre.replace("_", " ").title(),
                        fill_opacity=0.40,
                    )
                else:
                    dibujar_zona(
                        ms, pol, color="#9ca3af",
                        nombre=nombre.replace("_", " ").title(),
                        fill_opacity=0.10,
                    )
            if c is not None:
                color_seb = "#1e40af" if i == 0 else "#dc2626"
                dibujar_seb(ms, c, r, color=color_seb, label=s["nombre"])
            agregar_leyenda(ms, [
                ("#dc2626" if s["n_zonas"] > 0 else "#1e40af",
                 f"SEB {s['nombre']} — r = {fmt_km(s['radio'])}"),
                ("#dc2626", "Zonas activas"),
                ("#9ca3af", "Zonas inactivas"),
            ])
            st_folium(ms, use_container_width=True, height=330, key=f"ms{i}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Frontera de Factibilidad
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Frontera de Factibilidad: hasta donde puede crecer F?")

    st.markdown("""
    Si los parques "crecen" (buffer de distancia d), el conjunto factible F se encoge.
    Existe un **d critico** a partir del cual F queda completamente vacio y el problema
    se vuelve infactible — no hay ningun punto de Manhattan que este fuera de todos los parques.
    """)
    st.info(
        "Este experimento usa siempre los 7,947 puntos completos del dataset. "
        "Los filtros del sidebar no se aplican aqui."
    )

    st.markdown(
        '<div class="concepto">'
        '<b>F = vacio</b> significa que la union de los parques expandidos cubre todo el '
        'convex hull de Manhattan. El algoritmo lo detecta automaticamente y retorna '
        '<code>estado=infactible</code>.'
        '</div>',
        unsafe_allow_html=True,
    )

    buf4  = st.slider(
        "Buffer d (m)",
        0, 8000, 0, step=100, key="buf4",
        help="Distancia de expansion de cada zona prohibida",
    )
    calc4 = st.button("Calcular con este buffer", type="primary", key="calc4")

    if calc4:
        with st.spinner("Calculando SEB con zonas expandidas..."):
            zonas_inf4 = [
                (n, p.buffer(buf4) if buf4 > 0 else p)
                for n, p in datos["prohibidas"]
            ]
            res4 = seb_restringido(datos["puntos_full"], datos["R"], zonas_inf4, semilla=42)
            cob4 = (
                unary_union([p for _, p in zonas_inf4])
                .intersection(datos["R"]).area / datos["R"].area * 100
            )
            st.session_state.update(res4=res4, buf4_used=buf4, cob4=cob4)

    col_m4, col_cv = st.columns([3, 2])

    with col_cv:
        st.markdown("**Cobertura de parques y radio vs buffer (pre-calculado):**")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5.5), sharex=True)

        colores = {"restringido": "#f59e0b", "infactible": "#dc2626"}
        etiquetas = {"restringido": "F no vacio (factible)", "infactible": "F = vacio (infactible)"}
        for est in ["restringido", "infactible"]:
            sub = _CURVA[_CURVA["estado"] == est]
            ax1.scatter(
                sub["d_m"], sub["cobertura_pct"],
                c=colores[est], s=70, zorder=3, label=etiquetas[est],
            )
        ax1.plot(_CURVA["d_m"], _CURVA["cobertura_pct"], color="#d1d5db", alpha=0.7)
        ax1.axhline(100, color="#dc2626", linestyle="--", alpha=0.5, linewidth=1)
        ax1.axvline(4453, color="#7c3aed", linestyle=":", alpha=0.8, linewidth=1.5,
                    label="d critico ~4,453 m")
        ax1.set_ylabel("Cobertura de F por parques (%)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.25)

        fact4 = _CURVA[_CURVA["radio_m"].notna()]
        ax2.plot(fact4["d_m"], fact4["radio_m"] / 1000, "o-", color="#f59e0b", linewidth=1.5)
        ax2.axvline(4453, color="#7c3aed", linestyle=":", alpha=0.8, linewidth=1.5)
        ax2.set_xlabel("Buffer d (m)")
        ax2.set_ylabel("Radio SEB restringido (km)")
        ax2.grid(alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            '<div class="hallazgo">'
            '<b>d critico:</b> entre 4,438 y 4,469 m.<br>'
            'A d = 4,000 m el radio ya salta a 16.6 km (vs 10.4 km libre, +59%).<br>'
            'A d >= 6,000 m los parques cubren el 100% de Manhattan → infactible.'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_m4:
        res4 = st.session_state.get("res4")
        if res4 is None:
            st.info("Ajusta el slider y pulsa 'Calcular' para ver como cambia el SEB.")
        else:
            est4   = res4["estado"]
            buf4_u = st.session_state.get("buf4_used", 0)
            cob4_v = st.session_state.get("cob4", 0.0)

            msg = f"Parques cubren **{cob4_v:.1f}%** de Manhattan con d = {buf4_u} m"
            if est4 == "infactible":
                st.error(f"F = VACIO — {msg}")
            elif est4 == "restringido":
                st.warning(f"F no vacio — {msg}")
            else:
                st.success(f"Centro libre valido — {msg}")

            col_r1, col_r2 = st.columns(2)
            col_r1.metric(
                "Radio SEB",
                fmt_km(res4["radio"]) if est4 != "infactible" else "Infactible",
            )
            col_r2.metric("Cobertura parques", f"{cob4_v:.1f}%")

            m4 = crear_mapa(*utm_a_latlon(*datos["c_lib_full"]), zoom=11)
            folium.Polygon(
                pol_utm_a_latlon(datos["R"]),
                color="#1e40af", fill=False, weight=1.5,
                tooltip="Region factible Manhattan",
                dash_array="6",
            ).add_to(m4)
            for nombre, pol in datos["prohibidas"]:
                pol_d = pol.buffer(buf4_u) if buf4_u > 0 else pol
                dibujar_zona(
                    m4, pol_d, color="#16a34a",
                    nombre=f"{nombre.replace('_', ' ').title()} (buffer={buf4_u} m)",
                    fill_opacity=0.35,
                )
            if est4 != "infactible" and res4["centro"] is not None:
                dibujar_seb(m4, res4["centro"], res4["radio"],
                            color="#1e40af", label="SEB restringido")
            agregar_leyenda(m4, [
                ("#16a34a", f"Zonas prohibidas (buffer = {buf4_u} m)"),
                ("#1e40af", "Region factible Manhattan"),
                ("#1e40af", "SEB restringido"),
            ])
            st_folium(m4, use_container_width=True, height=450, key="m4")
