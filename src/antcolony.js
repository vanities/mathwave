// antcolony.js — Ant Colony Optimization (Dorigo, 1992) on a Traveling-Salesman
// graph. M cities sit in 3D; every edge (i,j) carries a PHEROMONE level τ_ij.
// Each iteration a batch of ANTS each build a tour, at every step choosing the
// next unvisited city j with probability
//        P_ij ∝ τ_ij^α · η_ij^β ,   η_ij = 1/distance   (the heuristic)
// — exploit strong trails (α) while still favouring short hops (β). When all
// ants finish we EVAPORATE every edge (τ ← (1−ρ)·τ) and DEPOSIT on the edges
// each ant used, more for shorter tours (Δτ ∝ Q/tourLength). Over iterations the
// colony reinforces the edges of the near-optimal tour and lets the rest fade —
// the strong edges light up as the answer, and the best tour length shrinks.
//
// Visual: cities as glowing nodes; ALL M² edges drawn as ONE LineSegments whose
// per-vertex colour/brightness encodes τ (dim teal → bright amber/cyan as a trail
// strengthens). The current best tour is a bright magenta/white loop; ant dots
// crawl their chosen edges. After convergence (or N iterations) we reseed cities.
//
// Ref: Dorigo, Maniezzo & Colorni, "Ant System: optimization by a colony of
// cooperating agents," IEEE Trans. SMC-B (1996); Dorigo & Stützle, "Ant Colony
// Optimization" (MIT Press, 2004).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x01060a, 0.011);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 1200);
camera.position.set(0, 34, 96);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.3;
controls.minDistance = 20; controls.maxDistance = 400;

scene.add(new THREE.AmbientLight(0x16303a, 0.9));
const key = new THREE.PointLight(0xbfeaff, 0.7, 0, 0); key.position.set(20, 40, 30); scene.add(key);
addGrid(scene, { size: 200, divisions: 24, y: -42 });

// ---------- ACO parameters ----------
const SPREAD = 44;            // half-extent cities are scattered over
let M = 22;                   // number of cities
let ANTS = 24;                // ants released per iteration
let ALPHA = 1.0;              // pheromone exponent  (exploit)
let BETA = 3.5;              // heuristic exponent  (greed toward short edges)
let RHO = 0.12;             // evaporation rate
const Q = 4.0;               // deposit scale
const TAU0 = 0.4;           // initial pheromone
let layout = "circle";       // city placement
const maxIters = 240;        // reseed after this many iterations

// ---------- state ----------
let cities;                   // Float32Array(M*3)
let tau;                      // Float32Array(M*M)  pheromone, symmetric
let dist;                     // Float32Array(M*M)  edge length
let eta;                      // Float32Array(M*M)  1/dist heuristic
let bestTour = null;          // Int32Array(M) order of cities
let bestLen = Infinity;
let iter = 0;
let tauMax = TAU0;            // running max for colour normalisation

function rand(a, b) { return a + Math.random() * (b - a); }

function placeCities() {
  cities = new Float32Array(M * 3);
  if (layout === "circle") {
    // a noisy ring — the optimum is close to the ring order, satisfying to watch resolve
    for (let i = 0; i < M; i++) {
      const a = (i / M) * Math.PI * 2 + rand(-0.18, 0.18);
      const r = SPREAD * (0.78 + rand(-0.12, 0.12));
      cities[i*3]   = Math.cos(a) * r;
      cities[i*3+1] = rand(-6, 6);
      cities[i*3+2] = Math.sin(a) * r;
    }
  } else if (layout === "clustered") {
    // a handful of tight clusters — ACO has to find inter-cluster bridges
    const nc = 4;
    const cx = [], cy = [], cz = [];
    for (let k = 0; k < nc; k++) { cx.push(rand(-SPREAD, SPREAD)); cy.push(rand(-10, 10)); cz.push(rand(-SPREAD, SPREAD)); }
    for (let i = 0; i < M; i++) {
      const k = i % nc;
      cities[i*3]   = cx[k] + rand(-9, 9);
      cities[i*3+1] = cy[k] + rand(-5, 5);
      cities[i*3+2] = cz[k] + rand(-9, 9);
    }
  } else { // "random" cloud
    for (let i = 0; i < M; i++) {
      cities[i*3]   = rand(-SPREAD, SPREAD);
      cities[i*3+1] = rand(-14, 14);
      cities[i*3+2] = rand(-SPREAD, SPREAD);
    }
  }
}

function edgeLen(i, j) {
  const ax = cities[i*3], ay = cities[i*3+1], az = cities[i*3+2];
  const bx = cities[j*3], by = cities[j*3+1], bz = cities[j*3+2];
  const dx = ax-bx, dy = ay-by, dz = az-bz;
  return Math.sqrt(dx*dx + dy*dy + dz*dz);
}

function tourLength(order) {
  let L = 0;
  for (let k = 0; k < M; k++) L += dist[order[k]*M + order[(k+1) % M]];
  return L;
}

function initColony() {
  placeCities();
  tau  = new Float32Array(M * M);
  dist = new Float32Array(M * M);
  eta  = new Float32Array(M * M);
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < M; j++) {
      if (i === j) continue;
      const d = edgeLen(i, j);
      dist[i*M + j] = d;
      eta[i*M + j]  = 1 / (d + 1e-3);
      tau[i*M + j]  = TAU0;
    }
  }
  bestTour = null; bestLen = Infinity; iter = 0; tauMax = TAU0;
  buildGeometry();
}

// ---------- one ACO iteration (a batch of ant tours) ----------
let _visited = new Uint8Array(M);   // reusable visited-flags, resized with M

function buildTour(order) {
  if (_visited.length !== M) _visited = new Uint8Array(M);
  _visited.fill(0);
  let cur = (Math.random() * M) | 0;
  order[0] = cur; _visited[cur] = 1;
  for (let step = 1; step < M; step++) {
    // weighted choice over unvisited cities: w = τ^α · η^β
    let sum = 0;
    for (let j = 0; j < M; j++) {
      if (_visited[j]) continue;
      sum += Math.pow(tau[cur*M + j], ALPHA) * Math.pow(eta[cur*M + j], BETA);
    }
    let next = -1;
    if (sum > 0) {
      let r = Math.random() * sum, acc = 0;
      for (let j = 0; j < M; j++) {
        if (_visited[j]) continue;
        acc += Math.pow(tau[cur*M + j], ALPHA) * Math.pow(eta[cur*M + j], BETA);
        if (acc >= r) { next = j; break; }
      }
    }
    if (next < 0) { // numeric fallback: first unvisited
      for (let j = 0; j < M; j++) if (!_visited[j]) { next = j; break; }
    }
    order[step] = next; _visited[next] = 1; cur = next;
  }
}

let _order = new Int32Array(M);
function iterate() {
  if (_order.length !== M) _order = new Int32Array(M);
  const order = _order;
  // evaporate every edge
  const keep = 1 - RHO;
  for (let e = 0; e < M * M; e++) tau[e] *= keep;
  // release the batch of ants; each builds a tour and deposits
  let iterBest = Infinity, iterBestOrder = null;
  for (let a = 0; a < ANTS; a++) {
    buildTour(order);
    const L = tourLength(order);
    const dep = Q / L;
    for (let k = 0; k < M; k++) {
      const i = order[k], j = order[(k+1) % M];
      tau[i*M + j] += dep; tau[j*M + i] += dep;   // symmetric deposit
    }
    if (L < iterBest) { iterBest = L; iterBestOrder = order.slice(); }
  }
  if (iterBestOrder && iterBest < bestLen) { bestLen = iterBest; bestTour = iterBestOrder; }
  // track max pheromone for colour scaling
  let mx = TAU0;
  for (let e = 0; e < M * M; e++) if (tau[e] > mx) mx = tau[e];
  tauMax = mx;
  iter++;
}

// ---------- geometry ----------
let nodeMesh;                 // InstancedMesh of city spheres
let edgeSeg, edgePos, edgeCol;// LineSegments of ALL edges (colour = pheromone)
let bestLine, bestPos;        // bright loop of the best tour
let antMesh, antEdges;        // dots crawling chosen edges
const ANT_DOTS = 40;

// teal (cold/weak) → cyan → amber (hot/strong) ramp for pheromone strength
function pheroColor(t, out) {
  // t in 0..1 ; keep weak edges dim, strong edges bright + warm. NOT purple.
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  let r, g, b;
  if (t < 0.5) {            // dim teal → bright cyan
    const u = t / 0.5;
    r = 0.02 + 0.10 * u;
    g = 0.28 + 0.62 * u;
    b = 0.34 + 0.62 * u;
  } else {                  // cyan → warm amber
    const u = (t - 0.5) / 0.5;
    r = 0.12 + 0.88 * u;
    g = 0.90 - 0.18 * u;
    b = 0.96 - 0.84 * u;
  }
  out[0] = r; out[1] = g; out[2] = b;
  return out;
}

function buildGeometry() {
  // --- city nodes ---
  if (nodeMesh) { scene.remove(nodeMesh); nodeMesh.geometry.dispose(); nodeMesh.material.dispose(); }
  const ng = new THREE.SphereGeometry(1, 14, 12);
  nodeMesh = new THREE.InstancedMesh(ng, new THREE.MeshStandardMaterial({ color: 0xfff0cf, emissive: 0x402a08, roughness: 0.35, metalness: 0.2 }), M);
  nodeMesh.frustumCulled = false;
  const dummy = new THREE.Object3D();
  for (let i = 0; i < M; i++) {
    dummy.position.set(cities[i*3], cities[i*3+1], cities[i*3+2]);
    dummy.scale.setScalar(1.5);
    dummy.updateMatrix(); nodeMesh.setMatrixAt(i, dummy.matrix);
  }
  nodeMesh.instanceMatrix.needsUpdate = true;
  scene.add(nodeMesh);

  // --- ALL edges as one LineSegments; two vertices per undirected edge i<j ---
  if (edgeSeg) { scene.remove(edgeSeg); edgeSeg.geometry.dispose(); edgeSeg.material.dispose(); }
  const pairs = (M * (M - 1)) / 2;
  edgePos = new Float32Array(pairs * 2 * 3);
  edgeCol = new Float32Array(pairs * 2 * 3);
  let p = 0;
  for (let i = 0; i < M; i++) {
    for (let j = i + 1; j < M; j++) {
      edgePos[p*6]   = cities[i*3];   edgePos[p*6+1] = cities[i*3+1]; edgePos[p*6+2] = cities[i*3+2];
      edgePos[p*6+3] = cities[j*3];   edgePos[p*6+4] = cities[j*3+1]; edgePos[p*6+5] = cities[j*3+2];
      p++;
    }
  }
  const eg = new THREE.BufferGeometry();
  eg.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));
  eg.setAttribute("color", new THREE.BufferAttribute(edgeCol, 3).setUsage(THREE.DynamicDrawUsage));
  edgeSeg = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.9,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  edgeSeg.frustumCulled = false;
  scene.add(edgeSeg);

  // --- best-tour loop (M+1 points, looped) ---
  if (bestLine) { scene.remove(bestLine); bestLine.geometry.dispose(); bestLine.material.dispose(); }
  bestPos = new Float32Array((M + 1) * 3);
  const bg = new THREE.BufferGeometry();
  bg.setAttribute("position", new THREE.BufferAttribute(bestPos, 3).setUsage(THREE.DynamicDrawUsage));
  bestLine = new THREE.Line(bg, new THREE.LineBasicMaterial({
    color: 0xff4fd8, transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false,
  }));
  bestLine.frustumCulled = false;
  bestLine.visible = false;
  scene.add(bestLine);

  // --- crawling ant dots ---
  if (antMesh) { scene.remove(antMesh); antMesh.geometry.dispose(); antMesh.material.dispose(); }
  antMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.7, 8, 6),
    new THREE.MeshBasicMaterial({ color: 0xfff6e0, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false }),
    ANT_DOTS
  );
  antMesh.frustumCulled = false;
  scene.add(antMesh);
  // each ant dot is assigned a best-tour leg; resampled when a best tour exists
  antEdges = new Int32Array(ANT_DOTS * 2);
}

// ---------- per-frame visual update ----------
const _c = [0, 0, 0];
function paintEdges() {
  const inv = 1 / (tauMax - TAU0 * 0.5 + 1e-6);
  let p = 0;
  for (let i = 0; i < M; i++) {
    for (let j = i + 1; j < M; j++) {
      // normalise this edge's pheromone into 0..1 (with a gentle gamma so the
      // winning trails really pop while the field stays faint)
      let t = (tau[i*M + j] - TAU0 * 0.5) * inv;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const tg = Math.pow(t, 0.6);
      pheroColor(tg, _c);
      // fade very weak edges toward black so the graph isn't a solid mesh
      const a = 0.10 + 0.90 * tg;
      edgeCol[p*6]   = _c[0]*a; edgeCol[p*6+1] = _c[1]*a; edgeCol[p*6+2] = _c[2]*a;
      edgeCol[p*6+3] = _c[0]*a; edgeCol[p*6+4] = _c[1]*a; edgeCol[p*6+5] = _c[2]*a;
      p++;
    }
  }
  edgeSeg.geometry.attributes.color.needsUpdate = true;
}

function paintBest() {
  if (!bestTour) { bestLine.visible = false; return; }
  bestLine.visible = true;
  for (let k = 0; k < M; k++) {
    const c = bestTour[k];
    bestPos[k*3] = cities[c*3]; bestPos[k*3+1] = cities[c*3+1]; bestPos[k*3+2] = cities[c*3+2];
  }
  const c0 = bestTour[0];
  bestPos[M*3] = cities[c0*3]; bestPos[M*3+1] = cities[c0*3+1]; bestPos[M*3+2] = cities[c0*3+2];
  bestLine.geometry.attributes.position.needsUpdate = true;
}

// assign each ant dot a leg of the best tour to crawl along
function assignAnts() {
  if (!bestTour) return;
  for (let a = 0; a < ANT_DOTS; a++) {
    const k = (Math.random() * M) | 0;
    antEdges[a*2] = bestTour[k];
    antEdges[a*2+1] = bestTour[(k+1) % M];
  }
}

const _ad = new THREE.Object3D();
function paintAnts(elapsed) {
  if (!bestTour) { antMesh.count = 0; antMesh.instanceMatrix.needsUpdate = true; return; }
  antMesh.count = ANT_DOTS;
  for (let a = 0; a < ANT_DOTS; a++) {
    const i = antEdges[a*2], j = antEdges[a*2+1];
    // phase per dot so they're spread along the legs
    const f = (elapsed * 0.35 + a * 0.137) % 1;
    const x = cities[i*3]   + (cities[j*3]   - cities[i*3])   * f;
    const y = cities[i*3+1] + (cities[j*3+1] - cities[i*3+1]) * f;
    const z = cities[i*3+2] + (cities[j*3+2] - cities[i*3+2]) * f;
    _ad.position.set(x, y, z);
    _ad.scale.setScalar(0.9);
    _ad.updateMatrix(); antMesh.setMatrixAt(a, _ad.matrix);
  }
  antMesh.instanceMatrix.needsUpdate = true;
}

// ---------- presets (α/β/ρ "explore" vs "exploit") + layout/size ----------
const PRESETS = [
  // label,               M,  ants, α,   β,   ρ,    layout
  ["circle · balanced",   22, 24, 1.0, 3.5, 0.12, "circle"],
  ["random · explore",    26, 28, 1.0, 2.0, 0.05, "random"],
  ["clustered · exploit", 28, 28, 1.6, 4.5, 0.20, "clustered"],
  ["dense · 36 cities",   36, 32, 1.0, 3.0, 0.12, "circle"],
  ["sparse · 15 cities",  15, 18, 1.0, 4.0, 0.10, "random"],
];
let pi = 0;
function applyPreset(p) {
  M = p[1]; ANTS = p[2]; ALPHA = p[3]; BETA = p[4]; RHO = p[5]; layout = p[6];
  initColony();
  // keep the sliders in sync with the preset
  const rEl = document.getElementById("rho"); if (rEl) { rEl.value = RHO; const o = document.querySelector('[data-val="rho"]'); if (o) o.textContent = RHO.toFixed(2); }
  const bEl = document.getElementById("beta"); if (bEl) { bEl.value = BETA; const o = document.querySelector('[data-val="beta"]'); if (o) o.textContent = BETA.toFixed(1); }
}

const wrap = document.getElementById("presets");
let chips = [];
function buildChips() {
  chips = PRESETS.map((cfg, i) => {
    const b = document.createElement("button");
    b.className = "chip" + (i === 0 ? " active" : "");
    b.textContent = cfg[0];
    b.addEventListener("click", () => { pi = i; applyPreset(PRESETS[i]); setChips(i); });
    wrap.appendChild(b);
    return b;
  });
}
function setChips(i) { chips.forEach((c, k) => c.classList.toggle("active", k === i)); }
buildChips();

bindRange("beta", (v) => { BETA = v; }, (v) => v.toFixed(1));
bindRange("rho", (v) => { RHO = v; }, (v) => v.toFixed(2));
document.getElementById("reseed").addEventListener("click", () => { applyPreset(PRESETS[pi]); });

setVariantCycler((d) => {
  pi = (pi + d + PRESETS.length) % PRESETS.length;
  applyPreset(PRESETS[pi]);
  setChips(pi);
  return PRESETS[pi][0];
});

// ---------- diagnostics ----------
window.__diag = () => JSON.stringify({ cities: M, iter, bestLen: Number.isFinite(bestLen) ? +bestLen.toFixed(2) : null });

// ---------- boot ----------
applyPreset(PRESETS[0]);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const iterEl = document.getElementById("iterc");
const lenEl = document.getElementById("blen");
const cityEl = document.getElementById("mc");

let acc = 0;                  // accumulator to pace ACO iterations
let antTimer = 0;
loop((dt, elapsed) => {
  meter(dt);

  // run a bounded number of ACO iterations per frame for smooth 60fps
  acc += dt;
  let budget = 2;            // at most 2 full ant batches per frame
  while (acc >= 0.06 && budget-- > 0) {
    acc -= 0.06;
    iterate();
    if (iter >= maxIters) { applyPreset(PRESETS[pi]); break; }  // converged → reseed
  }

  // refresh ant-dot leg assignment periodically so they sample fresh best legs
  antTimer += dt;
  if (antTimer > 1.2) { antTimer = 0; assignAnts(); }

  paintEdges();
  paintBest();
  paintAnts(elapsed);

  if (iterEl) iterEl.textContent = iter;
  if (lenEl)  lenEl.textContent = Number.isFinite(bestLen) ? bestLen.toFixed(1) : "–";
  if (cityEl) cityEl.textContent = M;

  controls.update();
  renderer.render(scene, camera);
});
