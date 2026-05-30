// som.js — Kohonen Self-Organizing Map (SOM): an unsupervised competitive-
// learning neural net, trained live on the CPU. A GxG lattice of neurons, each
// holding a weight vector in 3D (its position in data space), learns the shape
// of a target point cloud. Each iteration:
//   (1) draw a random sample x from the target distribution
//   (2) find the Best Matching Unit (BMU): the neuron whose weight w is
//       nearest x in Euclidean distance
//   (3) pull the BMU and its lattice-neighbors toward x:
//           w_i ← w_i + α · h(i, BMU) · (x − w_i)
//       with a Gaussian neighborhood over GRID distance d_grid:
//           h = exp( −d_grid² / (2 σ²) )
//   (4) decay the learning rate α and neighborhood radius σ over time
// Starting from a flat random sheet, the lattice UNFOLDS to wrap the cloud,
// preserving neighborhood topology — a 2D manifold draped over 3D data. The
// sheet is colored by its (u,v) grid coordinate so the fold reads.
//
// Ref: Kohonen, "Self-Organized Formation of Topologically Correct Feature
//      Maps", Biological Cybernetics 43 (1982); Kohonen, "The Self-Organizing
//      Map", Proc. IEEE 78 (1990).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 600);
camera.position.set(16, 12, 22);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.4;
controls.minDistance = 6;
controls.maxDistance = 120;

addGrid(scene, { size: 60, divisions: 30, y: -14 });
addSun(scene, { scale: 50, position: [0, 18, -80] });

// ---------- SOM lattice ----------
const G = 24;            // grid is GxG neurons
const NEUR = G * G;
const SPAN = 9;          // data lives roughly within [-SPAN, SPAN]
const weights = new Float32Array(NEUR * 3); // each neuron's 3D position

// learning schedule — α and σ are seeded each (re)start and decay toward the
// floor over DECAY_ITERS iterations of training.
let alpha0 = 0.3;        // initial learning rate (slider)
let sigma0 = G * 0.5;    // initial neighborhood radius in grid cells (slider)
const ALPHA_MIN = 0.01;
const SIGMA_MIN = 0.6;
const DECAY_ITERS = 60000;
let iter = 0;

function idx(gx, gy) { return gy * G + gx; }

function randomizeSheet() {
  // start as a small, nearly-flat random cloud near the origin so the unfold
  // is visible; a little jitter breaks symmetry.
  for (let i = 0; i < NEUR; i++) {
    weights[i * 3] = (Math.random() - 0.5) * 2;
    weights[i * 3 + 1] = (Math.random() - 0.5) * 2;
    weights[i * 3 + 2] = (Math.random() - 0.5) * 2;
  }
  iter = 0;
}
randomizeSheet();

// ---------- target distributions: each writes a sample into out3 ----------
const out3 = [0, 0, 0];

function gauss() {
  // Box–Muller standard normal
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const DISTS = {
  // uniform on a SPHERE surface
  sphere: () => {
    let x, y, z, m;
    do {
      x = Math.random() * 2 - 1; y = Math.random() * 2 - 1; z = Math.random() * 2 - 1;
      m = x * x + y * y + z * z;
    } while (m > 1 || m < 1e-6);
    const k = (SPAN * 0.8) / Math.sqrt(m);
    out3[0] = x * k; out3[1] = y * k; out3[2] = z * k;
    return out3;
  },
  // a TORUS surface
  torus: () => {
    const R = SPAN * 0.55, r = SPAN * 0.22;
    const u = Math.random() * Math.PI * 2, v = Math.random() * Math.PI * 2;
    out3[0] = (R + r * Math.cos(v)) * Math.cos(u);
    out3[1] = r * Math.sin(v);
    out3[2] = (R + r * Math.cos(v)) * Math.sin(u);
    return out3;
  },
  // a filled CUBE volume
  cube: () => {
    out3[0] = (Math.random() - 0.5) * 2 * SPAN * 0.7;
    out3[1] = (Math.random() - 0.5) * 2 * SPAN * 0.7;
    out3[2] = (Math.random() - 0.5) * 2 * SPAN * 0.7;
    return out3;
  },
  // two Gaussian BLOBS separated along x
  blobs: () => {
    const c = Math.random() < 0.5 ? -1 : 1, s = SPAN * 0.18;
    out3[0] = c * SPAN * 0.45 + gauss() * s;
    out3[1] = gauss() * s;
    out3[2] = gauss() * s;
    return out3;
  },
  // SWISS roll — a 2D sheet rolled into 3D, the SOM's favorite manifold
  swiss: () => {
    const t = 1.5 * Math.PI * (1 + 2 * Math.random());
    const h = (Math.random() - 0.5) * 2 * SPAN * 0.6, k = SPAN * 0.16;
    out3[0] = k * t * Math.cos(t);
    out3[1] = h;
    out3[2] = k * t * Math.sin(t);
    return out3;
  },
};
const DIST_NAMES = Object.keys(DISTS);
let distIdx = 0;
let distName = DIST_NAMES[0];
let sampleFn = DISTS[distName];

// ---------- a static faint cloud of the current target ----------
const CLOUD_N = 1600;
const cloudPos = new Float32Array(CLOUD_N * 3);
const cloudGeom = new THREE.BufferGeometry();
cloudGeom.setAttribute("position", new THREE.BufferAttribute(cloudPos, 3).setUsage(THREE.DynamicDrawUsage));
const cloud = new THREE.Points(cloudGeom, new THREE.PointsMaterial({
  size: 0.13, color: 0x3a5a78, transparent: true, opacity: 0.35, depthWrite: false,
}));
scene.add(cloud);

function rebuildCloud() {
  for (let i = 0; i < CLOUD_N; i++) {
    const s = sampleFn();
    cloudPos[i * 3] = s[0]; cloudPos[i * 3 + 1] = s[1]; cloudPos[i * 3 + 2] = s[2];
  }
  cloudGeom.attributes.position.needsUpdate = true;
}

// ---------- training step ----------
function bmu(x, y, z) {
  // index of nearest neuron by Euclidean distance in data space
  let best = 0, bestD = Infinity;
  for (let n = 0; n < NEUR; n++) {
    const i = n * 3;
    const dx = weights[i] - x, dy = weights[i + 1] - y, dz = weights[i + 2] - z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) { bestD = d; best = n; }
  }
  return best;
}

function trainOnce() {
  const decay = Math.min(iter / DECAY_ITERS, 1);
  const alpha = alpha0 * (1 - decay) + ALPHA_MIN * decay;
  const sigma = sigma0 * (1 - decay) + SIGMA_MIN * decay;
  const twoSig2 = 2 * sigma * sigma;
  const win = Math.max(1, Math.ceil(sigma * 3)); // neighbors past ~3σ ≈ 0; clamp window

  const s = sampleFn();
  const sx = s[0], sy = s[1], sz = s[2];
  const b = bmu(sx, sy, sz);
  const bx = b % G, by = (b / G) | 0;

  const gx0 = Math.max(0, bx - win), gx1 = Math.min(G - 1, bx + win);
  const gy0 = Math.max(0, by - win), gy1 = Math.min(G - 1, by + win);

  for (let gy = gy0; gy <= gy1; gy++) {
    const ddy = gy - by;
    for (let gx = gx0; gx <= gx1; gx++) {
      const ddx = gx - bx;
      const h = Math.exp(-(ddx * ddx + ddy * ddy) / twoSig2); // Gaussian neighborhood
      const lr = alpha * h;
      const i = idx(gx, gy) * 3;
      weights[i] += lr * (sx - weights[i]);
      weights[i + 1] += lr * (sy - weights[i + 1]);
      weights[i + 2] += lr * (sz - weights[i + 2]);
    }
  }
  iter++;
}

// ---------- geometry: SOM sheet (lines) + neuron points ----------
// lattice edges connect grid-neighbors: horizontal + vertical segments.
const EDGE_COUNT = 2 * G * (G - 1);
const linePos = new Float32Array(EDGE_COUNT * 2 * 3);
const lineCol = new Float32Array(EDGE_COUNT * 2 * 3);
const edgeA = new Int32Array(EDGE_COUNT); // neuron index of endpoint A
const edgeB = new Int32Array(EDGE_COUNT); // neuron index of endpoint B
{
  let e = 0;
  for (let gy = 0; gy < G; gy++)
    for (let gx = 0; gx < G - 1; gx++) { edgeA[e] = idx(gx, gy); edgeB[e] = idx(gx + 1, gy); e++; }
  for (let gy = 0; gy < G - 1; gy++)
    for (let gx = 0; gx < G; gx++) { edgeA[e] = idx(gx, gy); edgeB[e] = idx(gx, gy + 1); e++; }
}

const tmpColor = new THREE.Color();
function gridColor(n, target) {
  // 2D grid coord -> vivid color (NOT purple-dominant). Hue sweeps the
  // cyan→green→amber arc with u; v lifts lightness so the fold reads.
  const gx = n % G, gy = (n / G) | 0;
  const u = gx / (G - 1), v = gy / (G - 1);
  target.setHSL(0.5 - 0.42 * u, 0.72, 0.42 + 0.22 * v);
}

// static line colors (each endpoint = the color of its neuron)
for (let e = 0; e < EDGE_COUNT; e++) {
  const a = e * 6;
  gridColor(edgeA[e], tmpColor);
  lineCol[a] = tmpColor.r; lineCol[a + 1] = tmpColor.g; lineCol[a + 2] = tmpColor.b;
  gridColor(edgeB[e], tmpColor);
  lineCol[a + 3] = tmpColor.r; lineCol[a + 4] = tmpColor.g; lineCol[a + 5] = tmpColor.b;
}

const lineGeom = new THREE.BufferGeometry();
lineGeom.setAttribute("position", new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage));
lineGeom.setAttribute("color", new THREE.BufferAttribute(lineCol, 3));
const sheet = new THREE.LineSegments(lineGeom, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.9,
}));
sheet.frustumCulled = false;
scene.add(sheet);

// neuron points (one per neuron), colored by grid coord
const ptPos = new Float32Array(NEUR * 3);
const ptCol = new Float32Array(NEUR * 3);
for (let n = 0; n < NEUR; n++) {
  gridColor(n, tmpColor);
  ptCol[n * 3] = tmpColor.r; ptCol[n * 3 + 1] = tmpColor.g; ptCol[n * 3 + 2] = tmpColor.b;
}
const ptGeom = new THREE.BufferGeometry();
ptGeom.setAttribute("position", new THREE.BufferAttribute(ptPos, 3).setUsage(THREE.DynamicDrawUsage));
ptGeom.setAttribute("color", new THREE.BufferAttribute(ptCol, 3));
const neurons = new THREE.Points(ptGeom, new THREE.PointsMaterial({
  size: 0.34, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
}));
neurons.frustumCulled = false;
scene.add(neurons);

rebuildCloud();

// push current weights into the line + point geometries each frame
function syncGeometry() {
  for (let n = 0; n < NEUR; n++) {
    ptPos[n * 3] = weights[n * 3];
    ptPos[n * 3 + 1] = weights[n * 3 + 1];
    ptPos[n * 3 + 2] = weights[n * 3 + 2];
  }
  ptGeom.attributes.position.needsUpdate = true;
  for (let e = 0; e < EDGE_COUNT; e++) {
    const a = edgeA[e] * 3, b = edgeB[e] * 3, o = e * 6;
    linePos[o] = weights[a]; linePos[o + 1] = weights[a + 1]; linePos[o + 2] = weights[a + 2];
    linePos[o + 3] = weights[b]; linePos[o + 4] = weights[b + 1]; linePos[o + 5] = weights[b + 2];
  }
  lineGeom.attributes.position.needsUpdate = true;
}

// ---------- panel ----------
function setDist(i) {
  distIdx = (i % DIST_NAMES.length + DIST_NAMES.length) % DIST_NAMES.length;
  distName = DIST_NAMES[distIdx];
  sampleFn = DISTS[distName];
  rebuildCloud();
  randomizeSheet();              // restart the unfold for the new target
  if (nameEl) nameEl.textContent = distName;
  return distName;
}

const wrap = document.getElementById("dists");
const nameEl = document.getElementById("dname");
const chips = DIST_NAMES.map((name, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = name;
  b.addEventListener("click", () => { setDist(i); chips.forEach((c, k) => c.classList.toggle("active", k === distIdx)); });
  if (wrap) wrap.appendChild(b);
  return b;
});

bindRange("alpha", (v) => { alpha0 = v; }, (v) => v.toFixed(2));
bindRange("sigma", (v) => { sigma0 = v; }, (v) => v.toFixed(1));
document.getElementById("reset").addEventListener("click", () => randomizeSheet());

setVariantCycler((d) => {
  const label = setDist(distIdx + d);
  chips.forEach((c, k) => c.classList.toggle("active", k === distIdx));
  return label;
});

window.__diag = () => JSON.stringify({ dist: distName, iter });

// ---------- boot ----------
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const iterEl = document.getElementById("iter");

// several training iterations per frame so the unfold animates smoothly;
// bounded for perf, and trimmed under reduced-motion.
const ITERS_PER_FRAME = reducedMotion ? 36 : 90;

loop((dt) => {
  meter(dt);
  for (let k = 0; k < ITERS_PER_FRAME; k++) trainOnce();
  syncGeometry();
  if (iterEl) iterEl.textContent = iter.toLocaleString();
  controls.update();
  renderer.render(scene, camera);
});
