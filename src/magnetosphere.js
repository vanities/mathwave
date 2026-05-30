// magnetosphere.js — the solar wind meeting a planet's DIPOLE magnetic field.
// Charged particles streaming from the Sun feel the Lorentz force F = q(v×B);
// the planet's field deflects them around a shield (magnetopause), funnels a
// few down the field lines into glowing AURORAL ovals at the poles, and traps
// a population bouncing pole-to-pole + drifting around the planet = the Van
// Allen radiation belts.
//
// PHYSICS
// -------
// Dipole field of moment m (along the magnetic axis), at position r:
//     B(r) = (μ0/4π) · [ 3 r̂ (m·r̂) − m ] / |r|³.
//   We fold the constant into M (the "field strength" slider) and soften the
//   denominator → /(|r|³ + ε) so B can't blow up as r→0.
// A dipole FIELD LINE has the closed form r = L·sin²θ (θ = magnetic colatitude,
//   L = the line's equatorial radius); we draw a handful as faint loops — they
//   are the exact integral curves of B, so the structure reads before particles
//   fill it in.
// Each particle obeys F = q(v×B) (no E). We integrate with the BORIS pusher
//   (Birdsall & Langdon) — the standard scheme for v×B: it rotates v by the
//   gyration angle each step and conserves |v| EXACTLY, so gyration/bounce/drift
//   stay stable for thousands of steps even at coarse dt (plain Euler explodes).
// q/mass sign varies per particle (ions vs electrons → opposite gyration and
//   opposite azimuthal drift = the ring current).
//
// WHAT YOU SEE: a broad solar wind blows in from the sunward (−x) side; the
// field bends it around the planet (bow-shock pileup glows amber); grazing
// particles wrap into a comet-like tail; head-on ones spiral down to the poles
// and light the aurora; a seeded inner/outer belt bounces + drifts as two
// glowing toroidal bands. Field → 0 ⇒ the shield vanishes and the wind slams in.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setClearColor(0x02030a, 1);                 // near-black, not purple
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x02030a, 0.006);

const camera = new THREE.PerspectiveCamera(56, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(42, 24, 72);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.32;
controls.minDistance = 14; controls.maxDistance = 400;

scene.add(new THREE.AmbientLight(0x16243a, 0.7));
const sunLight = new THREE.DirectionalLight(0xfff0d6, 1.1);
sunLight.position.set(-1, 0.25, 0); scene.add(sunLight);   // lit from the Sun (−x)
addGrid(scene, { size: 220, divisions: 36, y: -40 });       // faint ecliptic plane
addSun(scene, { scale: 96, position: [-300, 36, 0] });      // the wind's source, sunward

// ---------- constants ----------
const Rp = 3.5;            // planet radius
const EPS3 = 6.0;          // softening added to r³ so B stays finite near r→0
const DT = 0.015, SUB = 3; // fixed Boris substep + substeps/frame (frame-rate stable)
const OUTER = 62;          // beyond this a particle has left the system
const SRC_X = -48, TAIL_X = 54, RBEAM = 22;  // solar-wind source column
const AUR_R = 8.2, AUR_LAT = 0.74;           // auroral zone: low altitude + high latitude
const BELT_V = 8.5;        // belt particle speed (kept independent of wind so belts persist)

// live, slider/preset-driven
let N = 4500;
let fieldM = 2200;         // dipole moment magnitude (0 → no shield)
let windSpeed = 14;        // solar-wind bulk speed
let tiltDeg = 11;          // magnetic-axis tilt
let auroraBoost = 1.0;     // brightness multiplier for the aurora
let swFraction = 0.55;     // fraction of particles that are solar wind (rest = belts)
let presetName = "quiet sun";

// magnetic axis (world), recomputed when tilt changes
const axis = new THREE.Vector3(0, 1, 0);
function setAxis() {
  const t = tiltDeg * Math.PI / 180;     // rotate +y about z by t
  axis.set(-Math.sin(t), Math.cos(t), 0).normalize();
}

// ---------- particle state (flat typed arrays) ----------
let px, py, pz, vx, vy, vz, qm, kind;   // kind: 0 = solar wind, 1 = belt
function alloc() {
  px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
  vx = new Float32Array(N); vy = new Float32Array(N); vz = new Float32Array(N);
  qm = new Float32Array(N); kind = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    qm[i] = (Math.random() < 0.5 ? -1 : 1) * (0.85 + Math.random() * 0.3); // ion / electron
    kind[i] = Math.random() < swFraction ? 0 : 1;
    if (kind[i] === 0) spawnWind(i); else spawnBelt(i);
  }
}

const rand = (a, b) => a + Math.random() * (b - a);

// solar wind: a broad column flowing +x from the sunward side
function spawnWind(i) {
  const a = Math.random() * Math.PI * 2, rr = Math.sqrt(Math.random()) * RBEAM;
  px[i] = SRC_X - Math.random() * 18;           // staggered in depth → continuous stream
  py[i] = Math.cos(a) * rr;
  pz[i] = Math.sin(a) * rr;
  vx[i] = windSpeed * rand(0.92, 1.0);
  vy[i] = rand(-0.6, 0.6); vz[i] = rand(-0.6, 0.6);
}

// belt: seed on a dipole field line near the equator, with a pitch angle so it
// mirrors (bounces) — small pitch precipitates to the pole (aurora), large pitch
// stays as the equatorial torus. Two L-bands = inner + outer radiation belts.
const _B = new THREE.Vector3(), _p = new THREE.Vector3(), _bh = new THREE.Vector3(), _e1 = new THREE.Vector3();
function spawnBelt(i) {
  const inner = Math.random() < 0.5;
  const L = inner ? rand(5.5, 8.0) : rand(9.5, 15.0);
  const th = Math.PI / 2 + rand(-0.18, 0.18);   // near magnetic equator
  const lon = Math.random() * Math.PI * 2;
  const r = L * Math.sin(th) * Math.sin(th);
  const rho = r * Math.sin(th);
  // axis-aligned, then rotate into the tilted frame (rotate about z by tilt)
  let x = rho * Math.cos(lon), y = r * Math.cos(th), z = rho * Math.sin(lon);
  const t = tiltDeg * Math.PI / 180, ct = Math.cos(t), st = Math.sin(t);
  px[i] = x * ct - y * st; py[i] = x * st + y * ct; pz[i] = z;

  _p.set(px[i], py[i], pz[i]);
  bfield(_p, _B);
  if (_B.lengthSq() < 1e-10) {                   // field off → just drift outward
    vx[i] = rand(-1, 1); vy[i] = rand(-1, 1); vz[i] = rand(-1, 1);
    const m = Math.hypot(vx[i], vy[i], vz[i]) || 1; const s = BELT_V / m;
    vx[i] *= s; vy[i] *= s; vz[i] *= s; return;
  }
  _bh.copy(_B).normalize();
  _e1.copy(_bh).cross(axis); if (_e1.lengthSq() < 1e-6) _e1.set(1, 0, 0); _e1.normalize();
  const a = rand(25, 85) * Math.PI / 180;        // pitch angle
  const ca = Math.cos(a), sa = Math.sin(a) * (Math.random() < 0.5 ? -1 : 1);
  vx[i] = BELT_V * (ca * _bh.x + sa * _e1.x);
  vy[i] = BELT_V * (ca * _bh.y + sa * _e1.y);
  vz[i] = BELT_V * (ca * _bh.z + sa * _e1.z);
}

// ---------- dipole field + Boris pusher ----------
// B(r) = M·[3 r̂(â·r̂) − â] / (r³ + ε),  â = magnetic axis (unit), M = fieldM.
function bfield(p, out) {
  const x = p.x, y = p.y, z = p.z;
  const r2 = x * x + y * y + z * z, r = Math.sqrt(r2), rinv = 1 / (r + 1e-6);
  const rhx = x * rinv, rhy = y * rinv, rhz = z * rinv;
  const mdr = fieldM * (axis.x * rhx + axis.y * rhy + axis.z * rhz);
  const inv = 1 / (r2 * r + EPS3);
  out.set(
    (3 * rhx * mdr - fieldM * axis.x) * inv,
    (3 * rhy * mdr - fieldM * axis.y) * inv,
    (3 * rhz * mdr - fieldM * axis.z) * inv
  );
}

// Boris: half-rotate v by t = (q/m)·B·dt/2, conserves |v| exactly (energy-stable).
function pushAll() {
  for (let s = 0; s < SUB; s++) {
    for (let i = 0; i < N; i++) {
      _p.set(px[i], py[i], pz[i]);
      bfield(_p, _B);
      const f = qm[i] * 0.5 * DT;
      const tx = _B.x * f, ty = _B.y * f, tz = _B.z * f;
      const sf = 2 / (1 + tx * tx + ty * ty + tz * tz);
      const sx = tx * sf, sy = ty * sf, sz = tz * sf;
      let X = vx[i], Y = vy[i], Z = vz[i];
      // v' = v + v×t
      const px2 = X + (Y * tz - Z * ty);
      const py2 = Y + (Z * tx - X * tz);
      const pz2 = Z + (X * ty - Y * tx);
      // v⁺ = v + v'×s
      X += (py2 * sz - pz2 * sy);
      Y += (pz2 * sx - px2 * sz);
      Z += (px2 * sy - py2 * sx);
      vx[i] = X; vy[i] = Y; vz[i] = Z;
      px[i] += X * DT; py[i] += Y * DT; pz[i] += Z * DT;
    }
  }
}

// ---------- recycle + counters ----------
let nTrapped = 0, nAurora = 0;
function recycle() {
  nTrapped = 0; nAurora = 0;
  for (let i = 0; i < N; i++) {
    const x = px[i], y = py[i], z = pz[i];
    const r = Math.hypot(x, y, z) || 1e-6;
    const cosLat = Math.abs((x * axis.x + y * axis.y + z * axis.z) / r);
    if (r < AUR_R && cosLat > AUR_LAT) nAurora++;           // in the auroral funnel
    if (r < Rp + 0.4) { kind[i] === 0 ? spawnWind(i) : spawnBelt(i); continue; } // precipitated
    if (r > OUTER || x > TAIL_X) { kind[i] === 0 ? spawnWind(i) : spawnBelt(i); continue; } // left
    if (kind[i] === 1) nTrapped++;
    if (!Number.isFinite(r)) spawnWind(i);                  // NaN guard
  }
}

// ---------- one THREE.Points (position + color buffers, additive glow) ----------
let geo, pts, posAttr, colAttr;
function buildPoints() {
  if (pts) { scene.remove(pts); geo.dispose(); pts.material.dispose(); }
  geo = new THREE.BufferGeometry();
  posAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
  colAttr = new THREE.BufferAttribute(new Float32Array(N * 3), 3).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("position", posAttr);
  geo.setAttribute("color", colAttr);
  const mat = new THREE.PointsMaterial({
    size: 0.62, map: makeSprite(), vertexColors: true, transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
  });
  pts = new THREE.Points(geo, mat); pts.frustumCulled = false; scene.add(pts);
}
function makeSprite() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.35, "rgba(255,255,255,0.55)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad; g.beginPath(); g.arc(32, 32, 32, 0, Math.PI * 2); g.fill();
  return new THREE.CanvasTexture(c);
}

// region palette (cyan / amber / green / magenta — no purple)
const SW_COOL = [0.20, 0.78, 1.00], SW_HOT = [1.00, 0.62, 0.18];
const BELT_OUT = [0.30, 1.00, 0.55], BELT_IN = [1.00, 0.30, 0.78];
function draw() {
  const pos = posAttr.array, col = colAttr.array;
  for (let i = 0; i < N; i++) {
    const x = px[i], y = py[i], z = pz[i];
    pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    const r = Math.hypot(x, y, z) || 1e-6;
    const cosLat = Math.abs((x * axis.x + y * axis.y + z * axis.z) / r);
    let cr, cg, cb;
    if (r < AUR_R && cosLat > AUR_LAT) {
      // auroral curtain: green-gold, boosted above 1 for bloom
      const f = (cosLat - AUR_LAT) / (1 - AUR_LAT);
      const b = auroraBoost * (0.9 + f * 0.9);
      cr = 0.85 * b; cg = 1.45 * b; cb = 0.55 * b;
    } else if (kind[i] === 0) {
      // solar wind: cyan far out, amber where it piles against the shield
      const heat = Math.min(Math.max(1 - (r - Rp) / 26, 0), 1);
      cr = SW_COOL[0] + (SW_HOT[0] - SW_COOL[0]) * heat;
      cg = SW_COOL[1] + (SW_HOT[1] - SW_COOL[1]) * heat;
      cb = SW_COOL[2] + (SW_HOT[2] - SW_COOL[2]) * heat;
    } else {
      // trapped belt: magenta inner, green outer, brightening toward mirror points
      const t = Math.min(Math.max((r - 6) / 9, 0), 1);
      const lat = Math.min(cosLat / 0.7, 1) * 0.4;
      cr = (BELT_IN[0] + (BELT_OUT[0] - BELT_IN[0]) * t) * (1 + lat);
      cg = (BELT_IN[1] + (BELT_OUT[1] - BELT_IN[1]) * t) * (1 + lat);
      cb = (BELT_IN[2] + (BELT_OUT[2] - BELT_IN[2]) * t) * (1 + lat);
    }
    col[i * 3] = cr; col[i * 3 + 1] = cg; col[i * 3 + 2] = cb;
  }
  posAttr.needsUpdate = true; colAttr.needsUpdate = true;
}

// ---------- planet + atmosphere + axis + field lines ----------
const planet = new THREE.Mesh(
  new THREE.SphereGeometry(Rp, 40, 28),
  new THREE.MeshStandardMaterial({ color: 0x0b1a2e, emissive: 0x040d18, roughness: 0.92, metalness: 0.0 })
);
scene.add(planet);
const atmo = new THREE.Mesh(
  new THREE.SphereGeometry(Rp * 1.16, 40, 28),
  new THREE.MeshBasicMaterial({ color: 0x2bd6ff, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, side: THREE.BackSide, depthWrite: false })
);
scene.add(atmo);

let axisLine;
function buildAxisLine() {
  if (axisLine) { scene.remove(axisLine); axisLine.geometry.dispose(); axisLine.material.dispose(); }
  const a = axis.clone().multiplyScalar(Rp * 2.6), b = a.clone().negate();
  const g = new THREE.BufferGeometry().setFromPoints([a, b]);
  axisLine = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x9ad8ff, transparent: true, opacity: 0.3, depthWrite: false }));
  scene.add(axisLine);
}

let fieldLines;
const FL_L = [6, 8, 11, 15, 20], FL_LON = 8;
function buildFieldLines() {
  if (fieldLines) { scene.remove(fieldLines); fieldLines.geometry.dispose(); fieldLines.material.dispose(); }
  const segs = [];
  const t = tiltDeg * Math.PI / 180, ct = Math.cos(t), st = Math.sin(t);
  for (const L of FL_L) {
    const thMin = Math.asin(Math.sqrt(Math.min(Rp / L, 0.999)));   // start at the surface
    for (let li = 0; li < FL_LON; li++) {
      const lon = (li / FL_LON) * Math.PI * 2;
      const cl = Math.cos(lon), sl = Math.sin(lon);
      let prev = null;
      const STEPS = 60;
      for (let s = 0; s <= STEPS; s++) {
        const th = thMin + (Math.PI - 2 * thMin) * (s / STEPS);
        const r = L * Math.sin(th) * Math.sin(th);
        const rho = r * Math.sin(th);
        const x0 = rho * cl, y0 = r * Math.cos(th), z0 = rho * sl;     // axis-aligned
        const X = x0 * ct - y0 * st, Y = x0 * st + y0 * ct, Z = z0;    // tilt about z
        const cur = [X, Y, Z];
        if (prev) segs.push(prev[0], prev[1], prev[2], cur[0], cur[1], cur[2]);
        prev = cur;
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
  fieldLines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x3a6cff, transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, depthWrite: false }));
  fieldLines.frustumCulled = false;
  scene.add(fieldLines);
}

function rebuildStatics() { setAxis(); buildAxisLine(); buildFieldLines(); }
function reseed() { setAxis(); alloc(); buildPoints(); }

// ---------- presets (↑↓) ----------
const PRESETS = [
  ["quiet sun",     { name: "quiet sun",     M: 2200, wind: 14, tilt: 11, aur: 1.0, sw: 0.55 }],
  ["solar storm",   { name: "solar storm",   M: 1500, wind: 25, tilt: 11, aur: 1.8, sw: 0.70 }],
  ["weak field",    { name: "weak field",    M: 260,  wind: 16, tilt: 11, aur: 1.2, sw: 0.62 }],
  ["tilted dipole", { name: "tilted dipole", M: 2400, wind: 14, tilt: 32, aur: 1.1, sw: 0.5 }],
];
function applyPreset(p) {
  presetName = p.name; fieldM = p.M; windSpeed = p.wind; tiltDeg = p.tilt;
  auroraBoost = p.aur; swFraction = p.sw;
  // reflect on sliders if present
  syncSlider("field", fieldM); syncSlider("wind", windSpeed); syncSlider("count", N);
  rebuildStatics(); reseed();
}
function syncSlider(id, v) {
  const el = document.getElementById(id); if (!el) return;
  el.value = v; const out = document.querySelector(`[data-val="${id}"]`);
  if (out) out.textContent = el.dataset.fmt === "k" ? `${(v / 1000).toFixed(1)}k` : v;
}

// ---------- panel wiring (everything above exists before bindRange fires) ----------
const wrap = document.getElementById("presets");
let pIdx = 0;
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { pIdx = i; applyPreset(PRESETS[i][1]); chips.forEach((c, j) => c.classList.toggle("active", j === i)); });
  wrap.appendChild(b);
  return b;
});
const nameEl = document.getElementById("pname");
bindRange("wind", (v) => { windSpeed = v; }, (v) => Math.round(v));
bindRange("field", (v) => { fieldM = Math.round(v); }, (v) => Math.round(v));
bindRange("count", (v) => { N = Math.round(v); reseed(); }, (v) => `${(v / 1000).toFixed(1)}k`);
document.getElementById("reset").addEventListener("click", () => applyPreset(PRESETS[pIdx][1]));
setVariantCycler((d) => {
  pIdx = (pIdx + d + PRESETS.length) % PRESETS.length;
  applyPreset(PRESETS[pIdx][1]);
  chips.forEach((c, j) => c.classList.toggle("active", j === pIdx));
  return PRESETS[pIdx][0];
});

// diagnostics for headless verification
window.__diag = () => JSON.stringify({ preset: presetName, particles: N, trapped: nTrapped, aurora: nAurora });

// ---------- boot ----------
rebuildStatics();
alloc();
buildPoints();
if (nameEl) nameEl.textContent = presetName;
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const auEl = document.getElementById("au");
loop((dt) => {
  meter(dt);
  pushAll();
  recycle();
  draw();
  if (nameEl) nameEl.textContent = presetName;
  if (auEl) auEl.textContent = nAurora;
  controls.update();
  renderer.render(scene, camera);
});
