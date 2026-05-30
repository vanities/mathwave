// flood.js — 大洪水 / The Flood. A procedural Earth whose ocean rises and drowns
// the continents until only the highest peaks remain, then abates and loops —
// "and the waters prevailed upon the earth a hundred and fifty days" (Gen 7).
//
// THE PLANET is a high-detail icosphere (THREE.IcosahedronGeometry, detail 6).
// Each vertex's RADIUS is displaced by fractal Brownian motion (fBm) of 3D
// value noise: we sum several octaves of a hash-lattice noise sampled at the
// unit-sphere direction, each octave at double the frequency and half the
// amplitude (lacunarity 2, gain ~0.5). A continent mask (a second low-frequency
// fBm, ridged + biased) decides land vs. ocean basin, so you get broad
// continents instead of uniform bumps. The summed elevation (stored per vertex)
// both pushes the surface out and drives a natural color ramp — deep ocean →
// shallow → sand → green lowland → rock → snow — written into a vertexColors
// BufferAttribute. The planet is built ONCE (only rebuilt when the seed re-rolls).
//
// THE OCEAN is a separate translucent sphere centered on the planet whose RADIUS
// is the current sea level. A driver oscillates sea level from the base ocean
// floor up past the mountains (40 days/nights of rising), holds at high water,
// then recedes and loops — so it's a continuous filmable cycle. RAIN is one
// THREE.Points field of streaks falling inward, dense while rising and tapering
// as the waters abate. Storm mood: near-black sky, fog, faint lightning flashes.
//
// ↑↓ = presets (the deluge / rising tide / the abyss / new world seed).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const SKY = 0x05060d;
renderer.setClearColor(SKY, 1);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(SKY, 0.018);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 9, 30);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 16; controls.maxDistance = 90;

// ---------- lights (storm-lit) ----------
const ambient = new THREE.AmbientLight(0x2a3550, 0.7);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xdfe6ff, 1.15);   // cold storm-light "sun"
sun.position.set(18, 14, 12); scene.add(sun);
const fill = new THREE.DirectionalLight(0x16243f, 0.45);
fill.position.set(-16, -6, -10); scene.add(fill);
const sunSprite = addSun(scene, { scale: 60, position: [-40, 26, -160] });
if (sunSprite) sunSprite.material.opacity = 0.35;   // distant, dim through the storm

// ---------- 3D value noise + fBm (hash lattice) ----------
// seedable integer hash → [0,1); trilinear-interpolated lattice = value noise.
let SEED = 1337;
function hash3(xi, yi, zi) {
  let h = (xi * 374761393 + yi * 668265263 + zi * 2147483647 + SEED * 2654435761) | 0;
  h = (h ^ (h >>> 13)) | 0;
  h = (h * 1274126177) | 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;            // [0,1)
}
const smooth = (t) => t * t * (3 - 2 * t);
function valueNoise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const u = smooth(xf), v = smooth(yf), w = smooth(zf);
  const c000 = hash3(xi, yi, zi),       c100 = hash3(xi + 1, yi, zi);
  const c010 = hash3(xi, yi + 1, zi),   c110 = hash3(xi + 1, yi + 1, zi);
  const c001 = hash3(xi, yi, zi + 1),   c101 = hash3(xi + 1, yi, zi + 1);
  const c011 = hash3(xi, yi + 1, zi + 1), c111 = hash3(xi + 1, yi + 1, zi + 1);
  const x00 = c000 + (c100 - c000) * u, x10 = c010 + (c110 - c010) * u;
  const x01 = c001 + (c101 - c001) * u, x11 = c011 + (c111 - c011) * u;
  const y0 = x00 + (x10 - x00) * v, y1 = x01 + (x11 - x01) * v;
  return (y0 + (y1 - y0) * w) * 2 - 1;       // [-1,1]
}
// fractal Brownian motion: sum octaves, lacunarity 2, gain 0.5.
function fbm(x, y, z, oct, freq, gain) {
  let amp = 1, f = freq, sum = 0, norm = 0;
  for (let i = 0; i < oct; i++) {
    sum += amp * valueNoise(x * f, y * f, z * f);
    norm += amp; amp *= gain; f *= 2;
  }
  return sum / norm;                          // ~[-1,1]
}

// ---------- the planet ----------
const R0 = 10;                  // mean planet radius (sphere baseline)
const RELIEF = 2.4;             // how far peaks rise above mean
const detail = 6;               // icosphere detail (≈40k tris — smooth + fast)
let baseGeo = null, planet = null;
let elev = null;                // per-vertex elevation in WORLD units (radius offset)
let dirX = null, dirY = null, dirZ = null;   // unit direction per vertex
let elevMin = 0, elevMax = 1;   // sea-level sweep bounds (in world radius units)

// elevation color ramp keyed by NORMALIZED elevation 0..1 (ocean→snow).
function landColor(t, out) {
  // stops: deep, shallow, sand, lowland green, hill, rock, snow
  const STOPS = [
    [0.00, 0.03, 0.09, 0.26],   // deep ocean basin (only seen if water gone)
    [0.30, 0.05, 0.16, 0.34],   // shallow shelf
    [0.40, 0.78, 0.70, 0.42],   // sand / beach
    [0.46, 0.20, 0.46, 0.18],   // green lowland
    [0.62, 0.16, 0.34, 0.13],   // forest hill
    [0.78, 0.40, 0.31, 0.22],   // brown rock
    [0.90, 0.55, 0.45, 0.36],   // bare highland
    [1.00, 0.95, 0.96, 1.00],   // snow peak
  ];
  let a = STOPS[0], b = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++) {
    if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { a = STOPS[i]; b = STOPS[i + 1]; break; }
  }
  const span = (b[0] - a[0]) || 1;
  const f = Math.min(Math.max((t - a[0]) / span, 0), 1);
  out[0] = a[1] + (b[1] - a[1]) * f;
  out[1] = a[2] + (b[2] - a[2]) * f;
  out[2] = a[3] + (b[3] - a[3]) * f;
}

function buildPlanet() {
  if (planet) { scene.remove(planet); baseGeo.dispose(); planet.material.dispose(); }
  baseGeo = new THREE.IcosahedronGeometry(R0, detail);
  const pos = baseGeo.attributes.position;
  const n = pos.count;
  elev = new Float32Array(n);
  dirX = new Float32Array(n); dirY = new Float32Array(n); dirZ = new Float32Array(n);
  const col = new Float32Array(n * 3);
  baseGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));

  // pass 1: compute raw elevation per vertex
  let rawMin = Infinity, rawMax = -Infinity;
  const raw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const inv = 1 / Math.hypot(x, y, z);
    const nx = x * inv, ny = y * inv, nz = z * inv;   // unit direction
    dirX[i] = nx; dirY[i] = ny; dirZ[i] = nz;
    // continent mask: low-freq fBm biased so ~40% is land
    const cont = fbm(nx, ny, nz, 4, 1.4, 0.55);       // ~[-1,1]
    const land = Math.max(0, cont + 0.12);            // >0 → land
    // mountains: higher-freq ridged fBm, only meaningful on land
    const detailN = fbm(nx + 7.3, ny - 2.1, nz + 4.7, 6, 3.0, 0.5);
    const ridge = 1 - Math.abs(detailN);              // ridged → mountain spines
    let h;
    if (land <= 0) {
      h = -0.55 + cont * 0.5;                          // ocean basin depth
    } else {
      h = land * 1.4 + Math.pow(land, 1.8) * ridge * 1.1 + detailN * 0.15;
    }
    raw[i] = h;
    if (h < rawMin) rawMin = h;
    if (h > rawMax) rawMax = h;
  }

  // pass 2: normalize, displace radius, color, find a sensible "ocean floor"
  const span = (rawMax - rawMin) || 1;
  const tmp = [0, 0, 0];
  elevMin = Infinity; elevMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const t = (raw[i] - rawMin) / span;               // 0..1 normalized
    elev[i] = t;
    // radius: sea-floor sits near baseline; land rises by RELIEF
    const r = R0 + (t - 0.40) * RELIEF;               // 0.40 ≈ default shoreline
    pos.setXYZ(i, dirX[i] * r, dirY[i] * r, dirZ[i] * r);
    landColor(t, tmp);
    const j = i * 3; col[j] = tmp[0]; col[j + 1] = tmp[1]; col[j + 2] = tmp[2];
    const worldR = r;                                  // sweep sea level over world radii
    if (worldR < elevMin) elevMin = worldR;
    if (worldR > elevMax) elevMax = worldR;
  }
  pos.needsUpdate = true;
  baseGeo.attributes.color.needsUpdate = true;
  baseGeo.computeVertexNormals();
  baseGeo.computeBoundingSphere();

  planet = new THREE.Mesh(
    baseGeo,
    new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.92, metalness: 0.04, flatShading: false })
  );
  scene.add(planet);
}

// ---------- the ocean (rising/receding sphere) ----------
const oceanGeo = new THREE.IcosahedronGeometry(1, 5);   // smooth unit sphere, scaled to sea level
const oceanMat = new THREE.MeshStandardMaterial({
  color: 0x1f6fb8, transparent: true, opacity: 0.62, roughness: 0.18, metalness: 0.5,
  emissive: 0x06243f, emissiveIntensity: 0.4, depthWrite: false,
});
const ocean = new THREE.Mesh(oceanGeo, oceanMat);
ocean.renderOrder = 2;                                   // composite over the land
scene.add(ocean);
const oceanBasePos = oceanGeo.attributes.position.array.slice();   // for wobble

// ---------- rain (one Points field of inward streaks) ----------
const RAIN_MAX = 6000;
let rainCount = 4200;
const rainGeo = new THREE.BufferGeometry();
const rainPos = new Float32Array(RAIN_MAX * 3);
const rainPrev = new Float32Array(RAIN_MAX * 3);         // radial fall bookkeeping
const RAIN_HI = 26, RAIN_LO = 11;                        // spawn shell → planet
function seedDrop(i) {
  // random direction on a sphere, at the outer shell
  const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2;
  const s = Math.sqrt(1 - u * u);
  const dx = s * Math.cos(a), dy = u, dz = s * Math.sin(a);
  const r = RAIN_LO + Math.random() * (RAIN_HI - RAIN_LO);
  const k = i * 3;
  rainPos[k] = dx * r; rainPos[k + 1] = dy * r; rainPos[k + 2] = dz * r;
  rainPrev[k] = dx; rainPrev[k + 1] = dy; rainPrev[k + 2] = dz;   // store unit dir
}
for (let i = 0; i < RAIN_MAX; i++) seedDrop(i);
rainGeo.setAttribute("position", new THREE.BufferAttribute(rainPos, 3).setUsage(THREE.DynamicDrawUsage));
rainGeo.setDrawRange(0, rainCount);
const rainTex = (() => {
  const c = document.createElement("canvas"); c.width = 8; c.height = 32;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 32);
  grad.addColorStop(0, "rgba(190,215,255,0)");
  grad.addColorStop(0.5, "rgba(200,225,255,0.85)");
  grad.addColorStop(1, "rgba(160,195,255,0)");
  g.fillStyle = grad; g.fillRect(2, 0, 4, 32);
  return new THREE.CanvasTexture(c);
})();
const rain = new THREE.Points(rainGeo, new THREE.PointsMaterial({
  size: 1.4, map: rainTex, transparent: true, opacity: 0.55,
  blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
}));
rain.frustumCulled = false;
scene.add(rain);

// ---------- flood driver ----------
// sea level sweeps elevMin..elevMax via a smooth triangle: rise (cover) → hold →
// recede → hold, scaled by `crest` (how high the high-water mark reaches).
let floodSpeed = 1.0;       // cycle speed multiplier
let crest = 1.06;           // high-water as fraction of elevMax (>1 = full submersion)
let rainScale = 1.0;        // user rain intensity
let phase = 0;              // 0..1 around the flood cycle
let seaLevel = R0;         // current ocean radius (world units)
let submergedPct = 0;      // % of land vertices underwater
let dayCount = 0;          // narrative day counter
const CYCLE_DAYS = 150;    // "a hundred and fifty days"

// raised-cosine ease for the rise/recede legs
function floodCurve(p) {
  // p in 0..1: rise 0..0.42, hold 0.42..0.5, recede 0.5..0.92, hold 0.92..1
  if (p < 0.42) return 0.5 - 0.5 * Math.cos((p / 0.42) * Math.PI);          // 0→1
  if (p < 0.50) return 1;
  if (p < 0.92) return 0.5 + 0.5 * Math.cos(((p - 0.50) / 0.42) * Math.PI); // 1→0
  return 0;
}

function applyPreset(p) {
  floodSpeed = p.speed; crest = p.crest;
  const sp = document.getElementById("speed"); if (sp) { sp.value = floodSpeed; sp.dispatchEvent(new Event("input")); }
  const cr = document.getElementById("crest"); if (cr) { cr.value = crest; cr.dispatchEvent(new Event("input")); }
}

// ---------- presets ----------
const PRESETS = [
  ["the deluge",  { speed: 1.0, crest: 1.08, reseed: false }],   // full submersion
  ["rising tide", { speed: 0.5, crest: 0.92, reseed: false }],   // slow, peaks survive
  ["the abyss",   { speed: 1.4, crest: 1.15, reseed: false }],   // high water, fast
  ["new world",   { speed: 1.0, crest: 1.06, reseed: true }],    // re-roll the continents
];
let presetIdx = 0;
function setPreset(i) {
  presetIdx = ((i % PRESETS.length) + PRESETS.length) % PRESETS.length;
  const [, cfg] = PRESETS[presetIdx];
  if (cfg.reseed) { SEED = (Math.random() * 1e9) | 0; buildPlanet(); }
  applyPreset(cfg);
  phase = 0; dayCount = 0;
  chips && chips.forEach((b, k) => b.classList.toggle("active", k === presetIdx));
  return PRESETS[presetIdx][0];
}

// ---------- panel ----------
const wrap = document.getElementById("presets");
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => setPreset(i));
  wrap.appendChild(b);
  return b;
});
bindRange("speed", (v) => { floodSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("crest", (v) => { crest = v; }, (v) => Math.round(v * 100) + "%");
bindRange("rain", (v) => {
  rainScale = v;
  rainCount = Math.round(800 + v * (RAIN_MAX - 800));
  rainCount = Math.min(rainCount, RAIN_MAX);
  rainGeo.setDrawRange(0, rainCount);
}, (v) => Math.round(v * 100) + "%");

const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => {
  camera.position.copy(home); controls.target.set(0, 0, 0);
  phase = 0; dayCount = 0;
});
setVariantCycler((d) => setPreset(presetIdx + d));

// ---------- boot ----------
buildPlanet();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const dayEl = document.getElementById("day");
const subEl = document.getElementById("sub");

// lightning bookkeeping
let nextBolt = 1.5, boltT = 0, flashEnergy = 0;

window.__diag = () => JSON.stringify({
  preset: PRESETS[presetIdx][0],
  seaLevel: +seaLevel.toFixed(3),
  submerged: +submergedPct.toFixed(1),
  day: dayCount,
});

loop((dt, elapsed) => {
  meter(dt);

  // advance the flood cycle
  if (!reducedMotion) phase = (phase + dt * 0.018 * floodSpeed) % 1;
  const cv = floodCurve(phase);                       // 0..1 water height factor
  // sweep from ocean floor (elevMin, just covering basins) up to crest·elevMax
  const lo = elevMin + (elevMax - elevMin) * 0.36;    // resting sea level (covers basins)
  const hi = elevMax * crest;
  seaLevel = lo + (hi - lo) * cv;
  ocean.scale.setScalar(seaLevel);

  // ocean surface wobble (subtle living sea) — perturb radius slightly
  const op = oceanGeo.attributes.position.array;
  const t = elapsed * 1.3;
  for (let i = 0; i < op.length; i += 3) {
    const bx = oceanBasePos[i], by = oceanBasePos[i + 1], bz = oceanBasePos[i + 2];
    const wob = 1 + 0.012 * Math.sin(bx * 6 + t) * Math.cos(bz * 6 - t * 0.7);
    op[i] = bx * wob; op[i + 1] = by * wob; op[i + 2] = bz * wob;
  }
  oceanGeo.attributes.position.needsUpdate = true;
  oceanGeo.computeVertexNormals();

  // % of land vertices currently underwater (sampled for speed)
  if (elev) {
    let under = 0, landN = 0;
    const step = 7;                                   // stride-sample the vertices
    for (let i = 0; i < elev.length; i += step) {
      const r = R0 + (elev[i] - 0.40) * RELIEF;
      if (elev[i] > 0.42) { landN++; if (r <= seaLevel) under++; }
    }
    submergedPct = landN ? (under / landN) * 100 : 0;
  }

  // narrative day counter tracks the cycle position
  dayCount = Math.round(phase * CYCLE_DAYS);

  // rain: rises during flood, tapers as waters recede.
  // intensity ~ how fast water is rising (derivative of the curve) + user scale.
  const rising = phase < 0.46 ? 1 : (phase < 0.92 ? Math.max(0, 1 - (phase - 0.46) / 0.46) : 0.05);
  const fall = (24 + 40 * floodSpeed) * dt;
  const liveRain = Math.round(rainCount * (0.25 + 0.75 * rising) * rainScale);
  rainGeo.setDrawRange(0, Math.min(liveRain, rainCount));
  for (let i = 0; i < rainCount; i++) {
    const k = i * 3;
    // move inward along stored unit direction (rainPrev holds the dir)
    const dx = rainPrev[k], dy = rainPrev[k + 1], dz = rainPrev[k + 2];
    let x = rainPos[k] - dx * fall, y = rainPos[k + 1] - dy * fall, z = rainPos[k + 2] - dz * fall;
    const rr = Math.hypot(x, y, z);
    if (rr < seaLevel + 0.3 || rr < RAIN_LO * 0.5) { seedDrop(i); continue; }
    rainPos[k] = x; rainPos[k + 1] = y; rainPos[k + 2] = z;
  }
  rainGeo.attributes.position.needsUpdate = true;
  rain.material.opacity = 0.18 + 0.45 * rising * Math.min(rainScale, 1.2);

  // lightning: brief ambient/bg flash at semi-random intervals
  boltT += dt;
  if (boltT > nextBolt) {
    boltT = 0; nextBolt = 0.8 + Math.random() * 3.5;
    flashEnergy = 0.6 + Math.random() * 0.7;
  }
  if (flashEnergy > 0) {
    flashEnergy = Math.max(0, flashEnergy - dt * 3.2);
    ambient.intensity = 0.7 + flashEnergy * 1.6;
    const g = 0x05 + Math.round(flashEnergy * 40);
    renderer.setClearColor((g << 16) | (g << 8) | (g + 12), 1);
  } else {
    ambient.intensity = 0.7;
    renderer.setClearColor(SKY, 1);
  }

  if (dayEl) dayEl.textContent = dayCount;
  if (subEl) subEl.textContent = Math.round(submergedPct) + "%";

  controls.update();
  renderer.render(scene, camera);
});
