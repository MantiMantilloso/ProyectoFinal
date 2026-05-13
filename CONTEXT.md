# Contexto del Proyecto — Manhattan S.O.S. / SEB con Restricciones

> Documento de estado para onboarding de agentes. Última actualización: 2026-05-12.

---

## 1. Qué es el proyecto

Proyecto final de **Geometría Computacional** (Semestre 8, Universidad). Autores: Mauricio Mantilla y James Soto.

**Problema académico**: dado un conjunto de puntos de demanda (accidentes de tráfico en Manhattan), encontrar el centro `c` y radio `r` mínimo de la bola envolvente (SEB) tal que `c` esté dentro de la región factible `R` y fuera de todas las zonas prohibidas `Z_i`. Es un problema **LP-type** de tipo Location-Allocation.

El conjunto factible es `F = R \ (Z_1 ∪ Z_2 ∪ ... ∪ Z_k)`, que es **no-convexo** (R con k huecos).

---

## 2. Dos apps independientes

### 2a. App Streamlit (`app.py`)

Demo académica seria del algoritmo. Usa datos reales (7 947 accidentes, 6 parques de Manhattan como zonas prohibidas). 5 tabs:

- **Tab 1** — Problema y datos (mapa de accidentes, KPIs, puntos soporte)
- **Tab 2** — SEB Libre vs. Restringido (toggle de zonas, arista vinculante en morado)
- **Tab 3** — Aristas activas + sub-problema 1D (scatter de 260 candidatos, función `f(t)` con parábolas individuales y envolvente convexa)
- **Tab 4** — Sensibilidad (trayectoria del centro en 4 configuraciones de zonas)
- **Tab 5** — Frontera de factibilidad

**Estética**: default de Streamlit. No cambiar a fuentes personalizadas, Plotly neon, ni tabs underline — una iteración así fue rechazada por el usuario y revertida con `git checkout`.

### 2b. App Manhattan S.O.S. (`manhattan_sos/`)

App cartoon interactiva. El usuario arrastra zonas LP al mapa y el backend recalcula el SEB en tiempo real. Es la segunda entrega, más vistosa.

- **Frontend**: HTML/CSS/JS puro + Leaflet. Sin framework. Estética cartoon (Bangers font, bordes negros gruesos, sombras hard-offset).
- **Backend**: FastAPI en `manhattan_sos/server.py`.
  - `GET /api/accidents` → 500 accidentes muestreados del parquet
  - `POST /api/seb` → `{zones, include_trail}` → `{status, center, radius_m, free_center, free_radius_m, trail, active_edge, n_candidates, n_feasible}`
  - El backend toma cada zona como GeoJSON, calcula su convex hull, reproyecta WGS84 → UTM 18N, llama a `src.aristas_completo.enumerar_aristas`, y reproyecta el resultado a WGS84.
  - `status` puede ser: `"libre"` | `"restringido"` | `"infactible"`
- **Categorías de zonas**: parque, hospital, escuela, iglesia, comercial, museo, heliport, sector. Los polígonos vienen de `manhattan_sos/data/zones.geojson` (1 710 features de Overpass + NYC community districts).
- **Trail**: el backend calcula el SEB tras cada prefijo cronológico de zonas y lo devuelve; el frontend lo dibuja como polyline punteada.
- **Docker**: `docker compose up --build` desde la raíz levanta dos servicios:
  - `manhattan-sos-api` (python:3.12-slim, uvicorn en puerto 8000, solo red interna)
  - `manhattan-sos-web` (nginx:alpine, puerto 8080 → público, sirve estáticos + proxy `/api/` → api:8000)
  - URL: http://localhost:8080

---

## 3. Estructura de archivos relevante

```
ProyectoFinal/
├── src/
│   ├── seb_seidel.py          # Algoritmo Seidel iterativo move-to-front (SEB libre, O(n) esperado)
│   ├── seb_restringido.py     # seb_en_segmento + seb_restringido (enumeración de aristas)
│   ├── aristas_completo.py    # enumerar_aristas (devuelve TODOS los candidatos con flag factible)
│   │                          # + sub_problema_1d (para visualización en Streamlit)
│   ├── constraints.py         # semiplanos, punto_en_F, mascara_en_F, reproyección WGS84→UTM
│   ├── seb_primitivas.py      # bola_de_1/2/3_puntos, bola_minima_3
│   ├── data_loader.py         # carga de accidentes NYC + zonas OSM (con cache)
│   ├── preprocessing.py       # limpieza y reproyección a UTM
│   └── app_helpers.py         # helpers para la app Streamlit (puntos soporte, dibujar arista, etc.)
├── manhattan_sos/
│   ├── server.py              # Backend FastAPI
│   ├── index.html / app.js / styles.css  # Frontend cartoon
│   ├── nginx.conf             # Proxy /api/ → api:8000
│   ├── data/zones.geojson     # 1710 zonas LP para el drag-and-drop
│   ├── Dockerfile.api         # Build context: raíz del proyecto
│   └── Dockerfile.web         # Build context: manhattan_sos/
├── data/
│   └── processed/accidentes_utm.parquet  # 7 947 accidentes en UTM (REQUERIDO por el backend)
├── docker-compose.yml
├── .dockerignore
└── app.py                     # App Streamlit
```

---

## 4. El algoritmo central

### SEB libre
Seidel iterativo move-to-front. O(n) esperado. En `src/seb_seidel.py`.

### SEB con restricciones
**Enumeración de aristas activas**. En `src/seb_restringido.py` y `src/aristas_completo.py`.

**Teoría**: si `c_libre` (centro del SEB sin restricciones) está en `F` → es la solución. Si no, el centro óptimo `c*` yace sobre la frontera de `F`. La frontera de `F` está compuesta por aristas de `R` y aristas de las zonas `Z_i`. Para cada arista `(p1, p2)` se resuelve el **sub-problema 1D**:

```
minimizar  max_i ||c - p_i||
sujeto a   c = p1 + t*(p2-p1),  t ∈ [0, 1]
```

Cada `||c - p_i||²` es una parábola en `t` (convexa) → su máximo punto a punto es convexo → `minimize_scalar bounded` converge en ~30 iteraciones. El candidato con menor radio que además sea factible (en `F`) es el ganador.

**Parámetros clave**:
- `EPS = 1e-7` m para tolerancias numéricas
- UTM Zone 18N (EPSG:32618) para todo el cálculo interno
- `SEED = 42`, `DEMO_SAMPLE = 500` en el backend Docker

### Resultado con datos reales (app Streamlit, 7 947 puntos, 6 parques)
- Centro libre: lat 40.787, lon -73.964 (dentro de Central Park), radio 10 424 m
- Arista vinculante: `central_park` edge #24 (borde sur-suroeste, t=0.3458)
- Centro restringido: lat 40.788, lon -73.967, radio 10 429 m (+4.93 m, +0.047%)
- Los otros 5 parques no son vinculantes

---

## 5. Bug conocido en el algoritmo — pendiente de fix

### Bug: zonas adyacentes se tratan como polígonos independientes

El chequeo de factibilidad en `constraints.py::punto_en_F` es:

```python
for sps_z in sps_zonas:
    if punto_estrictamente_en_poligono(c, sps_z, tol=tol):
        return False  # infactible solo si está ESTRICTAMENTE ADENTRO de una zona individual
```

**Problema**: si dos zonas Z1 y Z2 son adyacentes (se tocan), un punto sobre su frontera compartida:
- No está *estrictamente* dentro de Z1 → pasa ✓
- No está *estrictamente* dentro de Z2 → pasa ✓
- → Se considera **factible** aunque esté en el interior de `Z1 ∪ Z2`

El resultado observable: el centro SEB queda sobre la arista compartida entre dos zonas adyacentes, que debería ser zona prohibida.

**Fix correcto** (aún no implementado): antes de enumerar aristas, calcular `Z_union = unary_union([Z1, Z2, ...])` con Shapely. Usar `Z_union` para el chequeo de factibilidad y enumerar las aristas del **exterior** de `Z_union` (eliminando automáticamente las aristas internas compartidas).

### Fix ya aplicado (sesión 2026-05-12): candidatos en puntos de cruce

Cuando el óptimo en la arista de Z2 caía dentro de Z1 (infactible), el algoritmo descartaba *toda* la arista de Z2 sin evaluar los sub-segmentos factibles fuera de Z1.

**Fix**: se añadió `_interseccion_t(p1, p2, q1, q2)` en ambos archivos del algoritmo. Por cada arista enumerada, se calculan sus cruces con todas las demás aristas de todas las fuentes. Esos puntos de cruce se agregan como candidatos adicionales con su propia evaluación de factibilidad.

---

### Fix ya aplicado (sesión 2026-05-12): union de zonas prohibidas

Se implementó `_poligonos_de_union()` usando `shapely.ops.unary_union`. Las zonas individuales se fusionan antes de calcular semiplanos y antes de enumerar aristas. El resultado es un `Polygon` (si las zonas se tocan) o un `MultiPolygon` (si están separadas). En el primer caso, las aristas internas compartidas desaparecen y los puntos sobre ellas quedan correctamente marcados como infactibles.

---

### Bug activo: semiplanos asumen convexidad, unary_union puede ser no-convexo

**Síntoma observado (2026-05-12)**: con ciertas combinaciones de zonas "Sector" (community districts de NYC) la estrella (centro SEB) aparece visualmente dentro de uno de los polígonos de zona. Ocurre con 2-3 zonas colocadas en el área de Hell's Kitchen / Hudson Yards.

**Causa raíz**: `semiplanos()` en `src/constraints.py` asume explícitamente que el polígono de entrada es **convexo**:
```python
def semiplanos(poligono: Polygon) -> List[Semiplano]:
    """Asume P convexo..."""
```

Para un polígono convexo, la intersección de todos sus semiplanos define el interior exactamente. Para un polígono **no-convexo**, los semiplanos definen el *convex hull* del polígono — el interior de esa envoltura convexa puede contener puntos que están fuera del polígono real.

`unary_union` de dos zonas adyacentes pero no alineadas (p.ej., dos community districts en diagonal uno respecto al otro) produce un polígono **no-convexo**. Los semiplanos de ese polígono no-convexo son incorrectos: algunos puntos que están en el interior del union no satisfacen todos los semiplanos → `punto_estrictamente_en_poligono` los clasifica como exteriores → `punto_en_F` los marca como **factibles** cuando no lo son → el centro cae dentro de una zona.

**El fix correcto (pendiente de implementación)**:

Reemplazar el chequeo de factibilidad basado en semiplanos para las zonas por un chequeo usando Shapely directamente. Shapely maneja correctamente polígonos no-convexos:

```python
from shapely.geometry import Point
from shapely.ops import unary_union

# En lugar de sps_zonas + punto_estrictamente_en_poligono:
Z_union = unary_union([pol for _, pol in zonas_prohibidas])  # Polygon o MultiPolygon

def es_factible(c, sps_R, Z_union):
    if not punto_en_poligono(c, sps_R):   # R sigue siendo convexo -> semiplanos OK
        return False
    if Z_union.contains(Point(c[0], c[1])):  # strictly inside union -> infactible
        return False
    return True
```

`Polygon.contains(Point)` en Shapely retorna True solo para puntos estrictamente en el interior (no en la frontera), lo cual es exactamente la semántica que necesita `punto_en_F`.

La enumeración de aristas **sí puede seguir usando** las coordenadas exteriores del `unary_union` (Shapely las da correctas incluso para formas no-convexas), pero el chequeo de factibilidad debe usar `contains`, no semiplanos.

**Archivos a modificar**:
- `src/aristas_completo.py`: reemplazar `sps_zonas` y `punto_en_F(c, sps_R, sps_zonas)` por el chequeo con `Z_union.contains()`
- `src/seb_restringido.py`: ídem
- `src/constraints.py`: podría agregar una función `punto_en_F_union(c, sps_R, Z_union)` para encapsular, o bien manejarlo directamente en los dos archivos anteriores

**Consideraciones de implementación**:
- Crear un objeto Shapely `Point` por cada evaluación de factibilidad puede ser lento si hay muchos candidatos. Alternativa: usar `Z_union.exterior.distance(Point(c)) > -EPS` o batch con `shapely.vectorized` si el rendimiento fuera un problema. En el contexto del demo (500 puntos, pocas zonas) no debería ser relevante.
- `Z_union` puede ser `MultiPolygon` (zonas separadas sin contacto). `MultiPolygon.contains(Point)` funciona correctamente — retorna True si el punto está estrictamente dentro de alguno de los componentes.
- R sigue siendo convexo (bbox de Manhattan), por lo que el chequeo de R con semiplanos no tiene problema.

---

## 6. Entorno técnico

- **OS**: Windows 11 Home, PowerShell
- **Python local**: 3.14 (para Streamlit y notebooks)
- **Python en Docker**: 3.12-slim (para el backend FastAPI)
- **Working directory**: `C:\Users\james\OneDrive\Documents\Universidad\Semestre_8\GeometriaComputacional\ProyectoFinal`
- **Codificación terminal**: cp1252 — usar solo ASCII en `print()` / logs Python. `→` rompe; usar `->`.
- **Docker**: Docker Desktop en Windows. `docker compose up --build` desde la raíz.

---

## 7. Lo que hay que implementar a continuación

**Tarea pendiente**: reemplazar el chequeo de factibilidad de zonas (semiplanos → Shapely `contains`).

**Plan**:
1. En `src/aristas_completo.py` y `src/seb_restringido.py`:
   - Ya se calcula `union_pols = _poligonos_de_union(zonas_prohibidas)` con `unary_union`.
   - Eliminar `sps_zonas` (semiplanos de la union, que pueden ser incorrectos si la union es no-convexa).
   - Guardar `Z_union` (la geometría Shapely del union) para usarla en el chequeo de factibilidad.
   - Reemplazar `punto_en_F(c, sps_R, sps_zonas)` por una función que use `sps_R` para R y `Z_union.contains(Point(c))` para las zonas.
2. La enumeración de aristas puede seguir igual: `aristas(pol)` sobre los componentes de `union_pols` da los vértices exteriores correctos.
3. La función `_interseccion_t` y el código de candidatos en puntos de cruce se mantienen sin cambio.
4. Validar con el caso sintético de dos zonas adyacentes y con combinaciones de 3+ zonas.
