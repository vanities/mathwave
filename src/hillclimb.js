// hillclimb.js — local-search optimization on a bumpy fitness landscape.
// Four climbers MAXIMIZE a 2D fitness function f(x,y), drawn as 3D terrain
// (a sum of Gaussian PEAKS of differing height → several LOCAL maxima, one
// GLOBAL maximum). Each runs a different strategy so the local-vs-global
// optimum problem is visible. ↑↓ switches landscapes. AI search × 3D calculus.
//
// THE ALGORITHMS (Russell & Norvig, AIMA §4.1; SA: Kirkpatrick et al. 1983):
//   1. Greedy / steepest-ascent — sample a ring of neighbors, move to the
//      strictly-best uphill one; HALT when none is higher → stuck on a local peak.
//   2. Stochastic hill climbing — pick a RANDOM improving neighbor, weighted by
//      steepness (Δf). Wanders more, still local-only.
//   3. Random-restart — greedy, but TELEPORT to a fresh random start when stuck
//      and retry; over many restarts it finds the global peak (visible jump).
//   4. Simulated annealing — propose a random neighbor, accept improvements
//      always, accept a worse move with prob exp(ΔE/T) where (MAXIMIZING)
//      ΔE = f_new − f_cur; T cools over time → escapes shallow local maxima.
// Fitness and its gradient are analytic; the neighbor ring is fixed; trail
// buffers are preallocated — no per-frame allocation in the hot loop.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(15, 13, 17);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 7; controls.maxDistance = 80;
controls.target.set(0, 1, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.8));
const keyL = new THREE.DirectionalLight(0xfff1dd, 1.2); keyL.position.set(8, 16, 6); scene.add(keyL);
const rimL = new THREE.DirectionalLight(0xff7ad8, 0.7); rimL.position.set(-10, 7, -8); scene.add(rimL);
const fillL = new THREE.DirectionalLight(0x2be4ff, 0.5); fillL.position.set(4, -4, 9); scene.add(fillL);
addGrid(scene, { size: 60, divisions: 30, y: -7 });
addSun(scene, { scale: 44, position: [0, 11, -66] });

// ---------- landscapes: arrays of Gaussian PEAKS over a gentle inverted bowl ----------
// Each peak = [cx, cy, height, sigma]. The tallest peak is the GLOBAL maximum.
const D = 8;          // domain half-width
const HSCALE = 1.7;   // vertical exaggeration for the terrain
const LANDSCAPES = [
  ["few peaks", [[2.4, 1.8, 4.2, 2.0], [-3.2, -1.2, 2.6, 1.9], [-1.4, 3.6, 2.0, 1.6]]],
  ["many / needle", [
    [1.0, -3.4, 5.0, 0.95], [-4.0, 2.8, 2.4, 1.8], [4.0, 3.0, 2.3, 1.7],
    [-3.6, -3.2, 2.1, 1.6], [3.8, -0.6, 2.0, 1.6], [0.2, 3.9, 1.9, 1.5],
  ]],
  ["ridge", [
    [-4.2, -4.2, 2.6, 1.7], [-2.1, -2.1, 2.9, 1.7], [0, 0, 3.2, 1.7],
    [2.1, 2.1, 3.6, 1.7], [4.2, 4.2, 4.4, 1.7],
  ]],
  ["deceptive", [[-2.2, -1.4, 3.4, 3.4], [4.4, 4.0, 4.8, 1.0], [3.2, -3.6, 1.8, 1.6]]],
];
let landIdx = 0;
let peaks = LANDSCAPES[0][1];

// f(x,y) = Σ hᵢ exp(−rᵢ²/(2sᵢ²)) − 0.04(x²+y²)   (maximize; higher = better)
function f(x, y) {
  let z = -0.04 * (x * x + y * y);
  for (const [cx, cy, h, s] of peaks) {
    const r2 = (x - cx) ** 2 + (y - cy) ** 2;
    z += h * Math.exp(-r2 / (2 * s * s));
  }
  return z;
}
// +∇f (uphill direction) for the gradient-following fallback / global polish.
function grad(x, y, out) {
  let gx = -0.08 * x, gy = -0.08 * y;
  for (const [cx, cy, h, s] of peaks) {
    const r2 = (x - cx) ** 2 + (y - cy) ** 2;
    const e = h * Math.exp(-r2 / (2 * s * s)) / (s * s);
    gx += -e * (x - cx); gy += -e * (y - cy);
  }
  out.x = gx; out.y = gy;
}
const zworld = (x, y) => f(x, y) * HSCALE; // terrain height in world units

// locate the global maximum on a fine grid, then polish with gradient ascent
const _gp = { x: 0, y: 0 };
let globalMax = { x: 0, y: 0, f: 0 };
function findGlobalMax() {
  let bx = 0, by = 0, bf = -Infinity;
  const M = 110;
  for (let i = 0; i <= M; i++) {
    const x = -D + (2 * D * i) / M;
    for (let j = 0; j <= M; j++) {
      const y = -D + (2 * D * j) / M;
      const v = f(x, y);
      if (v > bf) { bf = v; bx = x; by = y; }
    }
  }
  let x = bx, y = by;
  for (let k = 0; k < 40; k++) { grad(x, y, _gp); x += 0.06 * _gp.x; y += 0.06 * _gp.y; }
  globalMax = { x, y, f: f(x, y) };
}
findGlobalMax();

// ---------- surface mesh (rebuilt in-place on landscape change) ----------
const N = 120;
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(N * N * 3);
const col = new Float32Array(N * N * 3);
const idx = [];
for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) {
  const a = i * N + j; idx.push(a, a + 1, a + N, a + 1, a + N + 1, a + N);
}
geo.setIndex(idx);
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
const surface = new THREE.Mesh(
  geo,
  new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.72, metalness: 0.12, side: THREE.DoubleSide, transparent: true, opacity: 0.94 }),
);
scene.add(surface);
const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x2be4ff, wireframe: true, transparent: true, opacity: 0.07 }));
scene.add(wire);

function buildSurface() {
  let lo = 1e9, hi = -1e9;
  const step = (2 * D) / (N - 1);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = -D + i * step, y = -D + j * step, z = zworld(x, y);
    const p = (i * N + j) * 3;
    pos[p] = x; pos[p + 1] = z; pos[p + 2] = y;
    if (z < lo) lo = z; if (z > hi) hi = z;
  }
  for (let i = 0; i < N * N; i++) {
    const z = pos[i * 3 + 1];
    const c = ramp((z - lo) / (hi - lo + 1e-6));
    col[i * 3] = c[0]; col[i * 3 + 1] = c[1]; col[i * 3 + 2] = c[2];
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
}

// ---------- global-maximum beacon (a bright pillar + pulsing ring) ----------
const beacon = new THREE.Group();
const beaconMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85 });
const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 5, 10), beaconMat);
pillar.position.set(0, 2.5, 0); beacon.add(pillar);
const ring = new THREE.Mesh(new THREE.TorusGeometry(0.55, 0.06, 8, 28),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 }));
ring.rotation.x = Math.PI / 2; ring.position.y = 5; beacon.add(ring);
const tip = new THREE.Mesh(new THREE.SphereGeometry(0.18, 14, 10), new THREE.MeshBasicMaterial({ color: 0xffffff }));
tip.position.y = 5; beacon.add(tip);
scene.add(beacon);
function placeBeacon() { beacon.position.set(globalMax.x, zworld(globalMax.x, globalMax.y), globalMax.y); }
placeBeacon();

// ---------- climbers (one per local-search strategy) ----------
const METHODS = [
  { key: "greedy",     name: "steepest-ascent", color: 0x2be4ff },
  { key: "stochastic", name: "stochastic",      color: 0x62ffb3 },
  { key: "restart",    name: "random-restart",  color: 0xffc24b },
  { key: "anneal",     name: "annealing",       color: 0xff4fd8 },
];
const TRAIL = 240;
const STEP = 0.16;        // neighbor probe / move distance
const NEIGHBORS = 12;     // ring of candidate neighbors per step

// fixed unit ring of probe directions (hoisted; no per-frame alloc)
const RING = [];
for (let k = 0; k < NEIGHBORS; k++) {
  const a = (k / NEIGHBORS) * Math.PI * 2;
  RING.push({ dx: Math.cos(a), dy: Math.sin(a) });
}

function makeClimber(m) {
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.3, 20, 16),
    new THREE.MeshStandardMaterial({ color: m.color, emissive: m.color, emissiveIntensity: 0.7, roughness: 0.3 }));
  scene.add(ball);
  // "stuck" flag ring that fades in when greedy/stochastic halt
  const flag = new THREE.Mesh(new THREE.RingGeometry(0.42, 0.6, 22),
    new THREE.MeshBasicMaterial({ color: m.color, transparent: true, opacity: 0, side: THREE.DoubleSide }));
  flag.rotation.x = -Math.PI / 2;
  scene.add(flag);
  const tgeo = new THREE.BufferGeometry();
  const tpos = new Float32Array(TRAIL * 3);
  tgeo.setAttribute("position", new THREE.BufferAttribute(tpos, 3).setUsage(THREE.DynamicDrawUsage));
  const trail = new THREE.Line(tgeo, new THREE.LineBasicMaterial({ color: m.color, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(trail);
  return { m, ball, flag, trail, tpos, n: 0, x: 0, y: 0, t: 0, cur: -Infinity, best: -Infinity, stuck: false, restarts: 0 };
}
const climbers = METHODS.map(makeClimber);

function randStart(seed) {
  const ang = seed * 2.39996;                 // golden-angle scatter
  const rad = D * (0.45 + (seed % 3) * 0.16);
  return {
    x: Math.max(-D, Math.min(D, Math.cos(ang) * rad + (Math.random() - 0.5) * 1.4)),
    y: Math.max(-D, Math.min(D, Math.sin(ang) * rad + (Math.random() - 0.5) * 1.4)),
  };
}
function resetClimber(c, seed) {
  const s = randStart(seed);
  c.x = s.x; c.y = s.y; c.t = 0;
  c.cur = f(c.x, c.y); c.best = c.cur;
  c.stuck = false; c.restarts = 0; c.n = 0;
  c.trail.geometry.setDrawRange(0, 0);
  c.flag.material.opacity = 0;
  c.ball.material.emissiveIntensity = 0.7;
}
function cutTrail(c) { c.n = 0; c.trail.geometry.setDrawRange(0, 0); } // visible jump on teleport
function pushTrail(c) {
  const z = zworld(c.x, c.y) + 0.3;
  if (c.n < TRAIL) { c.tpos.set([c.x, z, c.y], c.n * 3); c.n++; }
  else { c.tpos.copyWithin(0, 3); c.tpos.set([c.x, z, c.y], (TRAIL - 1) * 3); }
  c.trail.geometry.setDrawRange(0, c.n);
  c.trail.geometry.attributes.position.needsUpdate = true;
}

// sample the fixed neighbor ring → improving candidates in hoisted scratch
const _cand = [];
for (let i = 0; i < NEIGHBORS; i++) _cand.push({ x: 0, y: 0, f: 0, d: 0 });
function gatherNeighbors(c) {
  let n = 0, bestF = c.cur, bestIdx = -1;
  for (let k = 0; k < NEIGHBORS; k++) {
    const nx = c.x + RING[k].dx * STEP, ny = c.y + RING[k].dy * STEP;
    if (nx < -D || nx > D || ny < -D || ny > D) continue;
    const fv = f(nx, ny), d = fv - c.cur;
    if (d > 1e-5) {
      const e = _cand[n]; e.x = nx; e.y = ny; e.f = fv; e.d = d; n++;
      if (fv > bestF) { bestF = fv; bestIdx = n - 1; }
    }
  }
  return { n, bestIdx };
}

function stepClimber(c) {
  c.t++;
  const k = c.m.key;
  if (k === "greedy" || k === "restart") {
    const { n, bestIdx } = gatherNeighbors(c);
    if (n === 0 || bestIdx < 0) {
      if (k === "restart") {                       // teleport to a fresh start
        c.restarts++;
        const s = randStart(c.restarts * 7 + 3 + ((Math.random() * 997) | 0));
        c.x = s.x; c.y = s.y; c.cur = f(c.x, c.y);
        cutTrail(c);
      } else { c.stuck = true; }                   // greedy halts for good
    } else { const e = _cand[bestIdx]; c.x = e.x; c.y = e.y; c.cur = e.f; }
  } else if (k === "stochastic") {
    const { n } = gatherNeighbors(c);
    if (n === 0) { c.stuck = true; }
    else {                                          // random improving move, weighted by Δf
      let total = 0; for (let i = 0; i < n; i++) total += _cand[i].d;
      let r = Math.random() * total, pick = 0;
      for (let i = 0; i < n; i++) { r -= _cand[i].d; if (r <= 0) { pick = i; break; } }
      const e = _cand[pick]; c.x = e.x; c.y = e.y; c.cur = e.f;
    }
  } else {                                          // simulated annealing (MAXIMIZE: ΔE = f_new − f_cur)
    const T = Math.max(0.03, 2.0 * Math.exp(-c.t * 0.0016));
    const a = Math.random() * Math.PI * 2, r = STEP * (0.7 + Math.random() * 1.6);
    let nx = Math.max(-D, Math.min(D, c.x + Math.cos(a) * r));
    let ny = Math.max(-D, Math.min(D, c.y + Math.sin(a) * r));
    const fn = f(nx, ny), dE = fn - c.cur;
    if (dE >= 0 || Math.random() < Math.exp(dE / T)) { c.x = nx; c.y = ny; c.cur = fn; }
  }
  if (c.cur > c.best) c.best = c.cur;

  const z = zworld(c.x, c.y) + 0.3;
  c.ball.position.set(c.x, z, c.y);
  if (c.stuck) {                                    // dim + flag when halted
    c.ball.material.emissiveIntensity = 0.12;
    c.flag.position.set(c.x, zworld(c.x, c.y) + 0.05, c.y);
    c.flag.material.opacity = 0.55;
  } else {
    c.ball.material.emissiveIntensity = 0.7;
    c.flag.material.opacity = 0;
  }
}

// ---------- panel ----------
const wrap = document.getElementById("landscapes");
const chips = LANDSCAPES.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => selectLandscape(i));
  wrap.appendChild(b);
  return b;
});

// legend: color swatch → method (built from METHODS so colors stay in sync)
const legendEl = document.getElementById("legend");
if (legendEl) {
  legendEl.innerHTML = METHODS.map((m) => {
    const hex = "#" + m.color.toString(16).padStart(6, "0");
    return `<span style="color:${hex}">■</span> ${m.name}`;
  }).join(" &nbsp; ");
}

let speed = 1, seedBase = 1;
function resetRun() { seedBase += METHODS.length + 1; climbers.forEach((c, i) => resetClimber(c, seedBase + i)); }
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
const restartBtn = document.getElementById("restart");
if (restartBtn) restartBtn.addEventListener("click", resetRun);
const home = camera.position.clone();
const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", () => { camera.position.copy(home); controls.target.set(0, 1, 0); });

function selectLandscape(i) {
  landIdx = (i + LANDSCAPES.length) % LANDSCAPES.length;
  peaks = LANDSCAPES[landIdx][1];
  findGlobalMax();
  buildSurface();
  placeBeacon();
  resetRun();
  chips.forEach((c, k) => c.classList.toggle("active", k === landIdx));
  return LANDSCAPES[landIdx][0];
}
setVariantCycler((d) => selectLandscape(landIdx + d));

// ---------- boot ----------
buildSurface();
climbers.forEach((c, i) => resetClimber(c, i + 1));
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const bestEl = document.getElementById("best");

const MAX_STEPS = 1500;          // hard cap before a forced restart
let acc = 0, settle = 0;
loop((dt, t) => {
  meter(dt);
  if (!reducedMotion) {          // pulse the beacon
    beaconMat.opacity = 0.6 + 0.25 * (Math.sin(t * 3) * 0.5 + 0.5);
    ring.scale.setScalar(1 + 0.12 * Math.sin(t * 3));
  }

  // advance climbers (cap sub-steps/frame for 60fps)
  acc += dt * speed * 24;
  let budget = Math.min(acc | 0, 6); acc -= budget;
  while (budget-- > 0) {
    for (const c of climbers) {
      if ((c.m.key === "greedy" || c.m.key === "stochastic") && c.stuck) continue;
      if (c.t >= MAX_STEPS && c.m.key !== "restart") continue;
      stepClimber(c);
    }
  }
  for (const c of climbers) pushTrail(c);

  // readout: each method's best fitness (★ = reached the global peak)
  if (bestEl) {
    let html = "";
    for (const c of climbers) {
      const hex = "#" + c.m.color.toString(16).padStart(6, "0");
      const atGlobal = (globalMax.f - c.best) < 0.06 + 0.03 * Math.abs(globalMax.f);
      const tag = atGlobal ? " ★" : (c.stuck ? " ·stuck" : "");
      html += `<div><span style="color:${hex}">${c.m.name}</span> ${c.best.toFixed(2)}${tag}</div>`;
    }
    html += `<div style="opacity:.7">global ${globalMax.f.toFixed(2)}</div>`;
    bestEl.innerHTML = html;
  }

  // auto-restart once everyone has settled (greedy+stochastic stuck, or step cap)
  const settled = climbers.every((c) => c.stuck || c.t >= MAX_STEPS || c.m.key === "restart");
  if (settled) { settle += dt; if (settle > 3.0) { settle = 0; resetRun(); } } else settle = 0;

  controls.update();
  renderer.render(scene, camera);
});

window.__diag = () => JSON.stringify({
  landscape: LANDSCAPES[landIdx][0],
  best: {
    greedy: climbers[0].best,
    stochastic: climbers[1].best,
    restart: climbers[2].best,
    anneal: climbers[3].best,
  },
});
