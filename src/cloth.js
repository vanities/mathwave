// cloth.js — Mass-spring cloth via Verlet integration + constraint relaxation.
// A grid of N×N point masses connected by springs. Each frame:
//   (1) VERLET INTEGRATE — store no explicit velocity; instead derive it from the
//       last two positions: x_new = x + (x - x_prev)·damping + a·dt²  (a = gravity
//       + wind). This is position-based / "Störmer-Verlet" and is unconditionally
//       stable for stiff springs in a way explicit Euler is not.
//   (2) SATISFY CONSTRAINTS — relax distance constraints several iterations: for
//       each spring, push the two endpoints back toward the rest length (half each,
//       unless one end is pinned). Structural springs join 4-neighbors; shear
//       springs join diagonals; bend springs skip one (resist folding). More
//       iterations ⇒ a stiffer sheet.
//   (3) PIN — hold some particles fixed (top row, two corners, …) so the cloth
//       hangs and drapes; everything else falls and billows under WIND.
// The THREE mesh's position attribute is rewritten from the particle array every
// frame and normals recomputed so the folds catch the light. Rendered double-sided.
//
// Refs: Provot, "Deformation Constraints in a Mass-Spring Model…" (1995);
//       Jakobsen, "Advanced Character Physics" (GDC 2001).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.02);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 500);
camera.position.set(0, 2, 34);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.45;
controls.minDistance = 12; controls.maxDistance = 120;
controls.target.set(0, 0, 0);

// ---------- lighting (so the folds read) ----------
scene.add(new THREE.AmbientLight(0x2a2050, 0.7));
const key = new THREE.DirectionalLight(0xfff1dd, 1.2); key.position.set(10, 16, 12); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.55); rim.position.set(-12, 4, -10); scene.add(rim);
const fill = new THREE.PointLight(0xff2e97, 0.5, 120); fill.position.set(0, -10, 14); scene.add(fill);
addGrid(scene, { size: 140, divisions: 28, y: -26 });
addSun(scene, { scale: 80, position: [0, 24, -150] });

// ---------- cloth state ----------
const N = 40;                         // grid is N×N particles
const CLOTH = 22;                     // world-space side length
const SPACING = CLOTH / (N - 1);      // structural rest length
let stiffness = 4;                    // constraint relaxation iterations (slider)
let windStrength = 1.0;               // wind multiplier (slider)
let pinsReleased = false;             // "release pins" toggle

// particle arrays (current + previous position); pinned flags
let px, py, pz, ox, oy, oz, pin;

// constraint list: pairs (i, j) with a rest length
let constraints = null;

// the optional obstacle (a sphere the cloth can drape over)
let sphere = null, sphereR = 0, sphereC = new THREE.Vector3();

let preset = "flag";

const idx = (x, y) => y * N + x;

function allocParticles() {
  const n = N * N;
  px = new Float32Array(n); py = new Float32Array(n); pz = new Float32Array(n);
  ox = new Float32Array(n); oy = new Float32Array(n); oz = new Float32Array(n);
  pin = new Uint8Array(n);
}

// Build structural (4-neighbor), shear (diagonal), and bend (skip-one) springs.
function buildConstraints() {
  const c = [];
  const add = (ax, ay, bx, by) => {
    if (bx < 0 || by < 0 || bx >= N || by >= N) return;
    const a = idx(ax, ay), b = idx(bx, by);
    const dx = px[a] - px[b], dy = py[a] - py[b], dz = pz[a] - pz[b];
    c.push(a, b, Math.sqrt(dx * dx + dy * dy + dz * dz));
  };
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      add(x, y, x + 1, y);          // structural →
      add(x, y, x, y + 1);          // structural ↓
      add(x, y, x + 1, y + 1);      // shear ↘
      add(x, y, x + 1, y - 1);      // shear ↗
      add(x, y, x + 2, y);          // bend →→
      add(x, y, x, y + 2);          // bend ↓↓
    }
  }
  // Float32Array packs [i, j, restLen] triples
  constraints = new Float32Array(c);
}

// ---------- presets: layout + pinning + obstacle ----------
const PRESETS = ["flag", "hanging banner", "drape on sphere", "free fall"];

function seed(name) {
  preset = name;
  pinsReleased = false;
  if (releaseBtn) releaseBtn.classList.toggle("active", false);
  allocParticles();
  if (sphere) { scene.remove(sphere); sphere.geometry.dispose(); sphere.material.dispose(); sphere = null; }
  sphereR = 0;

  // base layout: a flat sheet in the XY plane, centered, hanging in +Y..-Y
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      const i = idx(x, y);
      const wx = (x / (N - 1) - 0.5) * CLOTH;
      const wy = (0.5 - y / (N - 1)) * CLOTH;   // row 0 is the TOP
      px[i] = wx; py[i] = wy; pz[i] = (Math.random() - 0.5) * 0.05; // tiny jitter breaks symmetry
      ox[i] = px[i]; oy[i] = py[i]; oz[i] = pz[i];
    }
  }

  if (name === "flag") {
    // pin the LEFT edge — a flag streaming in the wind
    for (let y = 0; y < N; y++) pin[idx(0, y)] = 1;
  } else if (name === "hanging banner") {
    // pin the WHOLE top row — a banner that sags and ripples
    for (let x = 0; x < N; x++) pin[idx(x, 0)] = 1;
  } else if (name === "drape on sphere") {
    // pin nothing; lay the sheet flat above a sphere and let it fall over it
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = idx(x, y);
        const wx = (x / (N - 1) - 0.5) * CLOTH;
        const wz = (y / (N - 1) - 0.5) * CLOTH;
        px[i] = wx; py[i] = 7.5; pz[i] = wz;      // flat, held high, horizontal
        ox[i] = px[i]; oy[i] = py[i]; oz[i] = pz[i];
      }
    }
    sphereR = 6.2;
    sphereC.set(0, -1.5, 0);
    const g = new THREE.SphereGeometry(sphereR * 0.985, 40, 28); // slightly under collision R
    sphere = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x161a2e, roughness: 0.85, metalness: 0.1 }));
    sphere.position.copy(sphereC);
    scene.add(sphere);
  } else { // "free fall" — pin two top corners briefly; you release with the toggle
    pin[idx(0, 0)] = 1;
    pin[idx(N - 1, 0)] = 1;
  }

  buildConstraints();
  buildMesh();
}

// ---------- the THREE mesh (geometry rewritten each frame) ----------
let mesh = null, posAttr = null;
function buildMesh() {
  if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  const geo = new THREE.BufferGeometry();
  const verts = new Float32Array(N * N * 3);
  posAttr = new THREE.BufferAttribute(verts, 3).setUsage(THREE.DynamicDrawUsage);
  geo.setAttribute("position", posAttr);
  // triangulate the grid (two tris per quad)
  const tris = [];
  for (let y = 0; y < N - 1; y++) {
    for (let x = 0; x < N - 1; x++) {
      const a = idx(x, y), b = idx(x + 1, y), c = idx(x, y + 1), d = idx(x + 1, y + 1);
      tris.push(a, c, b, b, c, d);
    }
  }
  geo.setIndex(tris);
  const uvs = new Float32Array(N * N * 2);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const i = idx(x, y); uvs[i * 2] = x / (N - 1); uvs[i * 2 + 1] = y / (N - 1); }
  geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  const mat = new THREE.MeshStandardMaterial({
    color: 0x18c2a8, roughness: 0.55, metalness: 0.2,
    side: THREE.DoubleSide, emissive: 0x06201c, emissiveIntensity: 0.6,
    flatShading: false,
  });
  mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  scene.add(mesh);
  syncGeometry();
}

// ---------- physics ----------
const GRAVITY = -9.0;
function integrate(dt) {
  const damping = 0.985;
  const dt2 = dt * dt;
  // wind: a gusting force that varies in space and time (billows the sheet)
  const t = performance.now() * 0.001;
  const n = N * N;
  for (let i = 0; i < n; i++) {
    if (pin[i] && !pinsReleased) { ox[i] = px[i]; oy[i] = py[i]; oz[i] = pz[i]; continue; }
    // per-particle wind, phase-shifted by height so the sheet ripples
    const gust = (0.55 + 0.45 * Math.sin(t * 1.7 + py[i] * 0.35 + px[i] * 0.12));
    const ax = windStrength * 6.5 * gust;
    const az = windStrength * 3.0 * Math.sin(t * 0.9 + px[i] * 0.2);
    const ay = GRAVITY;
    const cx = px[i], cy = py[i], cz = pz[i];
    px[i] = cx + (cx - ox[i]) * damping + ax * dt2;
    py[i] = cy + (cy - oy[i]) * damping + ay * dt2;
    pz[i] = cz + (cz - oz[i]) * damping + az * dt2;
    ox[i] = cx; oy[i] = cy; oz[i] = cz;
  }
}

function satisfyConstraints() {
  const iters = Math.max(1, Math.round(stiffness));
  const m = constraints.length;
  for (let k = 0; k < iters; k++) {
    for (let s = 0; s < m; s += 3) {
      const a = constraints[s] | 0, b = constraints[s + 1] | 0, rest = constraints[s + 2];
      let dx = px[b] - px[a], dy = py[b] - py[a], dz = pz[b] - pz[a];
      let d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < 1e-6) d = 1e-6;
      const diff = (d - rest) / d * 0.5;     // half-correction toward each end
      const mx = dx * diff, my = dy * diff, mz = dz * diff;
      const pa = pin[a] && !pinsReleased, pb = pin[b] && !pinsReleased;
      if (pa && pb) continue;
      if (!pa && !pb) {
        px[a] += mx; py[a] += my; pz[a] += mz;
        px[b] -= mx; py[b] -= my; pz[b] -= mz;
      } else if (pa) {                        // a fixed → move b the full amount
        px[b] -= mx * 2; py[b] -= my * 2; pz[b] -= mz * 2;
      } else {                                // b fixed → move a the full amount
        px[a] += mx * 2; py[a] += my * 2; pz[a] += mz * 2;
      }
    }
    if (sphereR > 0) collideSphere();         // keep cloth outside the obstacle
  }
}

function collideSphere() {
  const r = sphereR, n = N * N;
  for (let i = 0; i < n; i++) {
    const dx = px[i] - sphereC.x, dy = py[i] - sphereC.y, dz = pz[i] - sphereC.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < r * r) {
      const d = Math.sqrt(d2) || 1e-6;
      const k = r / d;
      px[i] = sphereC.x + dx * k; py[i] = sphereC.y + dy * k; pz[i] = sphereC.z + dz * k;
    }
  }
}

// ---------- write particle positions into the mesh ----------
function syncGeometry() {
  const arr = posAttr.array, n = N * N;
  for (let i = 0; i < n; i++) { arr[i * 3] = px[i]; arr[i * 3 + 1] = py[i]; arr[i * 3 + 2] = pz[i]; }
  posAttr.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
  mesh.geometry.computeBoundingSphere();
}

// ---------- panel ----------
const wrap = document.getElementById("presets");
let pIdx = 0;
const chips = PRESETS.map((label, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { pIdx = i; seed(label); chips.forEach((c, j) => c.classList.toggle("active", j === i)); });
  wrap.appendChild(b);
  return b;
});

bindRange("wind", (v) => { windStrength = v; }, (v) => v.toFixed(2) + "×");
bindRange("stiff", (v) => { stiffness = Math.round(v); }, (v) => `${Math.round(v)}`);

const releaseBtn = document.getElementById("release");
releaseBtn.addEventListener("click", () => { pinsReleased = !pinsReleased; releaseBtn.classList.toggle("active", pinsReleased); });
document.getElementById("reset").addEventListener("click", () => { seed(PRESETS[pIdx]); chips.forEach((c, j) => c.classList.toggle("active", j === pIdx)); });

setVariantCycler((d) => {
  pIdx = (pIdx + d + PRESETS.length) % PRESETS.length;
  seed(PRESETS[pIdx]);
  chips.forEach((c, j) => c.classList.toggle("active", j === pIdx));
  return PRESETS[pIdx];
});

// diag for headless checks
window.__diag = () => JSON.stringify({ preset });

// ---------- boot ----------
seed("flag");
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const pEl = document.getElementById("pn");

loop((dt) => {
  meter(dt);
  // clamp dt so a stalled tab can't blow up the springs
  const h = Math.min(dt, 1 / 30);
  integrate(h);
  satisfyConstraints();
  syncGeometry();
  if (pEl) pEl.textContent = preset;
  controls.update();
  renderer.render(scene, camera);
});
