/* ==========================================================================
   Manhattan S.O.S. - frontend (mock, sin backend todavia)
   Cada zona LP es un poligono real de OpenStreetMap (data/zones.geojson)
   ========================================================================== */

const MANHATTAN_CENTER = [40.7831, -73.9712];
const DEFAULT_ZOOM = 12;

const ZONE_META = {
  hospital:  { icon: "#i-hospital",   color: "#FF4D4D", label: "Hospital"  },
  heliport:  { icon: "#i-helicopter", color: "#3DA9FC", label: "Heliport"  },
  parque:    { icon: "#i-tree",       color: "#42C97A", label: "Parque"    },
  comercial: { icon: "#i-bag",        color: "#FF9F1C", label: "Comercial" },
  escuela:   { icon: "#i-school",     color: "#F2C94C", label: "Escuela"   },
  iglesia:   { icon: "#i-church",     color: "#C49A6C", label: "Iglesia"   },
  museo:     { icon: "#i-landmark",   color: "#F45B9E", label: "Museo"     },
  sector:    { icon: "#i-grid",       color: "#14B8A6", label: "Sector"    }
};

// --------------------------------------------------------------------------
// 1. Mapa
// --------------------------------------------------------------------------

const map = L.map("map", {
  zoomControl: true,
  attributionControl: true
}).setView(MANHATTAN_CENTER, DEFAULT_ZOOM);

L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png", {
  attribution: "&copy; OpenStreetMap, &copy; CARTO",
  subdomains: "abcd",
  maxZoom: 19
}).addTo(map);

// --------------------------------------------------------------------------
// 2. Capas
// --------------------------------------------------------------------------

const accidentLayer = L.layerGroup().addTo(map); // accidentes mock
const previewLayer  = L.layerGroup().addTo(map); // pool de candidatos durante drag
const hoverLayer    = L.layerGroup().addTo(map); // poligono bajo el cursor
const lpLayer       = L.layerGroup().addTo(map); // zonas LP fijadas (rojas)
const trailLayer    = L.layerGroup().addTo(map); // recorrido historico del centro SEB
const sebLayer      = L.layerGroup().addTo(map); // SEB

// Centro/radio iniciales: se sobreescriben con la respuesta del backend al
// primer refreshState(). Solo sirven como placeholder visual hasta que carguen.
let SEB_INITIAL_CENTER = [40.7831, -73.9712];
let SEB_INITIAL_RADIUS = 6000;

// --------------------------------------------------------------------------
// 3. Accidentes (desde backend)
// --------------------------------------------------------------------------

let accidentsReady = false;
let accidentsCount = 0;

async function loadAccidents() {
  try {
    const resp = await fetch("/api/accidents");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    accidentLayer.clearLayers();
    for (const [lat, lng] of data.points) {
      L.circleMarker([lat, lng], {
        radius: 3,
        color: "#0F0F12",
        weight: 1,
        fillColor: "#FF3B30",
        fillOpacity: 0.85
      }).addTo(accidentLayer);
    }
    accidentsCount = data.count;
    accidentsReady = true;
    document.getElementById("statPoints").textContent = data.count;
    console.log("[accidents] cargados:", data.count);
  } catch (err) {
    console.error("[accidents] error:", err);
    showToast("error cargando accidentes: " + err.message, "err");
  }
}

// --------------------------------------------------------------------------
// 4. SEB mock
// --------------------------------------------------------------------------

function drawMockSEB(center, radiusMeters) {
  sebLayer.clearLayers();
  L.circle(center, {
    radius: radiusMeters,
    color: "#0F0F12",
    weight: 3,
    dashArray: "8 6",
    fillColor: "#FFC93C",
    fillOpacity: 0.18
  }).addTo(sebLayer);

  L.marker(center, {
    icon: L.divIcon({
      className: "",
      html: `<div class="lp-marker seb-marker" style="width:38px;height:38px;">
               <svg viewBox="0 0 24 24"><use href="#i-star"/></svg>
             </div>`,
      iconSize: [38, 38],
      iconAnchor: [19, 19]
    }),
    interactive: false
  }).addTo(sebLayer);

  document.getElementById("statRadius").textContent = `${Math.round(radiusMeters)} m`;
}

// Trail: el backend devuelve los centros por cada prefijo de zonas. Aqui solo
// dibujamos los waypoints encadenados desde el SEB libre inicial.
function drawTrail(points) {
  trailLayer.clearLayers();
  if (!points || points.length < 2) return;

  L.polyline(points, {
    color: "#0F0F12",
    weight: 2.5,
    opacity: 0.85,
    dashArray: "6 5",
    interactive: false
  }).addTo(trailLayer);

  for (let i = 0; i < points.length - 1; i++) {
    L.circleMarker(points[i], {
      radius: 4,
      color: "#0F0F12",
      weight: 2,
      fillColor: "#FFC93C",
      fillOpacity: 1,
      interactive: false
    }).addTo(trailLayer);
  }
}

// --- llamada al backend --------------------------------------------------
// concurrencia: si el usuario suelta zonas rapido, descartamos respuestas
// de peticiones obsoletas.
let sebSeq = 0;

async function requestSEB() {
  const seq = ++sebSeq;
  const payload = {
    zones: lpZones.map(z => ({
      name: z.feature.properties.name || "",
      category: z.category,
      osm_id: z.feature.properties.osm_id || null,
      geometry: z.feature.geometry
    })),
    include_trail: true
  };

  const resp = await fetch("/api/seb", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error("backend " + resp.status + ": " + txt);
  }
  const data = await resp.json();
  return { seq, data };
}

// --------------------------------------------------------------------------
// 5. Cargar poligonos OSM (categorizados)
// --------------------------------------------------------------------------

const zonesByCategory = {};   // category -> [features]
let zonesReady = false;

async function loadZones() {
  try {
    const resp = await fetch("data/zones.geojson");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const fc = await resp.json();

    for (const f of fc.features) {
      const cat = f.properties && f.properties.category;
      if (!cat) continue;
      f._bbox = computeBbox(f.geometry);
      if (!zonesByCategory[cat]) zonesByCategory[cat] = [];
      zonesByCategory[cat].push(f);
    }

    zonesReady = true;
    console.log("[zones] cargados:",
      Object.fromEntries(Object.entries(zonesByCategory).map(([k, v]) => [k, v.length])));
  } catch (err) {
    console.error("[zones] error al cargar:", err);
    showToast("error cargando zonas: " + err.message, "err");
  }
}

// arranque: cargar accidentes + zonas en paralelo, luego SEB libre inicial.
(async function init() {
  await Promise.all([loadAccidents(), loadZones()]);
  await refreshState();
})();

function computeBbox(geom) {
  let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
  const upd = (lng, lat) => {
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
  };
  if (geom.type === "Polygon") {
    geom.coordinates[0].forEach(([lng, lat]) => upd(lng, lat));
  } else if (geom.type === "MultiPolygon") {
    geom.coordinates.forEach(poly => poly[0].forEach(([lng, lat]) => upd(lng, lat)));
  }
  return [minLat, minLng, maxLat, maxLng];
}

// --------------------------------------------------------------------------
// 6. Point-in-polygon (ray casting)
// --------------------------------------------------------------------------

function pointInRing(lat, lng, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    const intersect = ((yi > lat) !== (yj > lat))
      && (lng < (xj - xi) * (lat - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

function pointInFeature(lat, lng, feature) {
  const [minLat, minLng, maxLat, maxLng] = feature._bbox;
  if (lat < minLat || lat > maxLat || lng < minLng || lng > maxLng) return false;
  const g = feature.geometry;
  if (g.type === "Polygon")      return pointInRing(lat, lng, g.coordinates[0]);
  if (g.type === "MultiPolygon") return g.coordinates.some(p => pointInRing(lat, lng, p[0]));
  return false;
}

function findFeatureAt(lat, lng, category) {
  const features = zonesByCategory[category] || [];
  // smallest-first prefer: si hay solapamiento, devolver el de bbox menor (mas especifico)
  let best = null;
  let bestArea = Infinity;
  for (const f of features) {
    if (!pointInFeature(lat, lng, f)) continue;
    const [mnLat, mnLng, mxLat, mxLng] = f._bbox;
    const area = (mxLat - mnLat) * (mxLng - mnLng);
    if (area < bestArea) {
      bestArea = area;
      best = f;
    }
  }
  return best;
}

// --------------------------------------------------------------------------
// 7. Preview pool y hover durante drag
// --------------------------------------------------------------------------

function showPreviewPool(category) {
  previewLayer.clearLayers();
  const features = zonesByCategory[category] || [];
  if (!features.length) return;

  const bounds = map.getBounds().pad(0.1);
  for (const f of features) {
    const [mnLat, mnLng, mxLat, mxLng] = f._bbox;
    // filtrar al viewport (cull): si bbox no toca viewport, skip
    if (mxLat < bounds.getSouth() || mnLat > bounds.getNorth()) continue;
    if (mxLng < bounds.getWest()  || mnLng > bounds.getEast())  continue;

    const layer = L.geoJSON(f, {
      style: {
        color: "#3DA9FC",
        weight: 1.5,
        opacity: 0.7,
        fillColor: "#3DA9FC",
        fillOpacity: 0.10,
        interactive: false
      }
    });
    f._previewLayer = layer;
    previewLayer.addLayer(layer);
  }
}

function clearPreviewPool() {
  previewLayer.clearLayers();
  for (const cat of Object.keys(zonesByCategory)) {
    for (const f of zonesByCategory[cat]) delete f._previewLayer;
  }
}

let lastHoverFeature = null;
function setHoverFeature(feature) {
  if (feature === lastHoverFeature) return;
  hoverLayer.clearLayers();
  lastHoverFeature = feature;
  if (!feature) return;

  L.geoJSON(feature, {
    style: {
      color: "#0F0F12",
      weight: 3,
      fillColor: "#3DA9FC",
      fillOpacity: 0.55,
      interactive: false
    }
  }).addTo(hoverLayer);
}

// --------------------------------------------------------------------------
// 8. Drag & drop: cajas -> mapa (con poligonos reales)
// --------------------------------------------------------------------------

const boxes = document.querySelectorAll(".zone-box");
const mapEl = document.getElementById("map");
const body  = document.body;

let currentDragType = null;
const lpZones = []; // {feature, category, layer}

boxes.forEach(box => {
  box.addEventListener("dragstart", (e) => {
    if (!zonesReady) {
      e.preventDefault();
      showToast("zonas aun cargando, intenta de nuevo", "warn");
      return;
    }
    currentDragType = box.dataset.type;
    box.classList.add("dragging");
    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setData("text/plain", currentDragType);
    }
    showPreviewPool(currentDragType);
  });

  box.addEventListener("dragend", () => {
    box.classList.remove("dragging");
    body.classList.remove("map-drop-hover");
    clearPreviewPool();
    setHoverFeature(null);
    currentDragType = null;
  });
});

mapEl.addEventListener("dragenter", (e) => {
  e.preventDefault();
  body.classList.add("map-drop-hover");
});

mapEl.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (!currentDragType) return;
  if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";

  const rect = mapEl.getBoundingClientRect();
  const point = L.point(e.clientX - rect.left, e.clientY - rect.top);
  const ll = map.containerPointToLatLng(point);
  const f = findFeatureAt(ll.lat, ll.lng, currentDragType);
  setHoverFeature(f);
});

mapEl.addEventListener("dragleave", (e) => {
  if (e.target === mapEl) body.classList.remove("map-drop-hover");
});

mapEl.addEventListener("drop", (e) => {
  e.preventDefault();
  body.classList.remove("map-drop-hover");

  const type = currentDragType || (e.dataTransfer && e.dataTransfer.getData("text/plain"));
  if (!type || !ZONE_META[type]) return;

  const rect = mapEl.getBoundingClientRect();
  const point = L.point(e.clientX - rect.left, e.clientY - rect.top);
  const ll = map.containerPointToLatLng(point);
  const feature = findFeatureAt(ll.lat, ll.lng, type);

  if (!feature) {
    showToast(`no hay ${ZONE_META[type].label} aqui`, "warn");
    return;
  }
  // evitar duplicado
  if (lpZones.some(z => z.feature === feature)) {
    showToast("ya esta en zonas LP", "warn");
    return;
  }

  commitLPZone(feature, type);
});

// --------------------------------------------------------------------------
// 9. Commit / remove LP zones
// --------------------------------------------------------------------------

function commitLPZone(feature, category) {
  const meta = ZONE_META[category];

  const polyLayer = L.geoJSON(feature, {
    style: {
      color: "#0F0F12",
      weight: 3,
      fillColor: meta.color,
      fillOpacity: 0.42
    }
  }).addTo(lpLayer);

  // centroide aproximado (centro del bbox) para colocar el icono
  const [mnLat, mnLng, mxLat, mxLng] = feature._bbox;
  const center = [(mnLat + mxLat) / 2, (mnLng + mxLng) / 2];

  const marker = L.marker(center, {
    icon: L.divIcon({
      className: "",
      html: `<div class="lp-marker" style="width:38px;height:38px;background:${meta.color};">
               <svg viewBox="0 0 24 24"><use href="${meta.icon}"/></svg>
             </div>`,
      iconSize: [38, 38],
      iconAnchor: [19, 19]
    }),
    title: `${feature.properties.name || meta.label} - click para eliminar`
  }).addTo(lpLayer);

  const remove = () => {
    lpLayer.removeLayer(polyLayer);
    lpLayer.removeLayer(marker);
    const idx = lpZones.findIndex(z => z.feature === feature);
    if (idx >= 0) lpZones.splice(idx, 1);
    refreshState();
    showToast(`${feature.properties.name || meta.label} removido`, "warn");
  };

  polyLayer.on("click", remove);
  marker.on("click", remove);

  lpZones.push({ feature, category, polyLayer, marker });
  refreshState();
  showToast(`+ ${feature.properties.name || meta.label}`);
}

// --------------------------------------------------------------------------
// 10. Estado: resultado + contadores (mock)
// --------------------------------------------------------------------------

const resultText    = document.getElementById("resultText");
const resultJustify = document.getElementById("resultJustify");
const statLp        = document.getElementById("statLp");
const trayCounter   = document.getElementById("trayCounter");

async function refreshState() {
  statLp.textContent = lpZones.length;
  trayCounter.textContent = `${lpZones.length} activa${lpZones.length === 1 ? "" : "s"}`;

  resultJustify.textContent = "calculando SEB...";

  let result;
  try {
    result = await requestSEB();
  } catch (err) {
    console.error("[seb] error:", err);
    showToast("error backend: " + err.message, "err");
    resultJustify.textContent = "error consultando el backend";
    return;
  }
  // descartar respuestas obsoletas (otro drop ya disparo una nueva)
  if (result.seq !== sebSeq) return;

  const data = result.data;
  // memorizar el centro libre como "punto de partida" del trail
  if (data.free_center) {
    SEB_INITIAL_CENTER = data.free_center;
    SEB_INITIAL_RADIUS = data.free_radius_m;
  }

  if (data.status === "infactible") {
    sebLayer.clearLayers();
    trailLayer.clearLayers();
    document.getElementById("statRadius").textContent = "---";
    resultText.innerHTML = "<b>Infactible:</b> no existe punto valido";
    resultJustify.textContent =
      `con ${lpZones.length} zona(s) LP el conjunto factible queda vacio. Quita alguna zona.`;
    return;
  }

  // libre o restringido: hay centro y radio
  drawMockSEB(data.center, data.radius_m);

  // trail: [libre] + centros de cada prefijo (incluye el final)
  if (lpZones.length === 0) {
    trailLayer.clearLayers();
  } else {
    const trailPoints = [data.free_center, ...data.trail];
    drawTrail(trailPoints);
  }

  if (data.status === "libre") {
    resultText.innerHTML =
      `Coloca el servicio en <b>${data.center[0].toFixed(4)}, ${data.center[1].toFixed(4)}</b>`;
    resultJustify.textContent =
      `SEB libre: radio ${Math.round(data.radius_m)} m sobre ${accidentsCount} accidentes (sin zonas LP activas)`;
  } else {
    const labels = lpZones
      .map(z => z.feature.properties.name || ZONE_META[z.category].label)
      .join(", ");
    const dr = data.radius_m - data.free_radius_m;
    const drTxt = dr >= 0 ? `+${Math.round(dr)} m` : `${Math.round(dr)} m`;
    resultText.innerHTML =
      `Coloca el servicio en <b>${data.center[0].toFixed(4)}, ${data.center[1].toFixed(4)}</b>`;
    resultJustify.textContent =
      `SEB restringido por ${lpZones.length} zona(s) LP (${labels}). Radio ${Math.round(data.radius_m)} m (${drTxt} vs libre).`;
  }
}

// --------------------------------------------------------------------------
// 11. Reset
// --------------------------------------------------------------------------

document.getElementById("resetBtn").addEventListener("click", () => {
  if (lpZones.length === 0) {
    showToast("el mapa ya esta limpio");
    return;
  }
  lpLayer.clearLayers();
  lpZones.length = 0;
  refreshState();
  showToast("mapa limpio");
});

// --------------------------------------------------------------------------
// 12. Toast
// --------------------------------------------------------------------------

const toastStack = document.getElementById("toastStack");

function showToast(message, variant = "ok") {
  const colors = { ok: "#42C97A", warn: "#FF9F1C", err: "#FF3B30" };
  const chipColor = colors[variant] || colors.ok;

  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = `
    <span class="toast-chip" style="background:${chipColor}">
      <svg viewBox="0 0 24 24"><use href="#i-bolt"/></svg>
    </span>
    <span>${message}</span>
  `;
  toastStack.appendChild(el);

  setTimeout(() => {
    el.classList.add("toast-out");
    setTimeout(() => el.remove(), 240);
  }, 1800);
}
