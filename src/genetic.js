// genetic.js — Genetic Algorithm / 遺伝
// ----------------------------------------------------------------------------
// A POPULATION of N candidate solutions (genomes), each scored by a fitness
// function, evolved toward an optimum. Every generation:
//   (1) EVALUATE  — score every genome with fit(genome).
//   (2) SELECT    — pick parents, favouring fitter genomes. Two schemes:
//                     • roulette (fitness-proportional) selection, or
//                     • tournament selection (best of k random contenders).
//   (3) CROSSOVER — recombine pairs of parents into offspring. Genomes here are
//                   2-vectors (x,y); uniform crossover takes each gene from
//                   either parent at 50/50.
//   (4) MUTATE    — perturb offspring genes by a small Gaussian step
//                   (mutation rate / size from the slider).
//   (5) ELITISM   — carry the best K genomes through unchanged (optional).
// Repeat → the population climbs toward the global optimum and its spread
// shrinks as it converges. When converged (or after G generations) we reseed a
// fresh random population and restart.
//
// Refs: Holland, "Adaptation in Natural and Artificial Systems" (1975);
// Goldberg, "Genetic Algorithms in Search, Optimization & ML" (1989).
//
// SHIPPED FITNESS TASK — (A) function optimization in 3D. The genome (x,y)
// indexes a bumpy landscape z = f(x,y) = Σ Gaussian peaks (one clearly dominant
// global peak + several lower local optima). Each genome is a glowing point ON
// the surface; watch the whole cloud migrate and CONVERGE onto the global peak
// over generations — the spread shrinking IS the convergence. The best genome
// is highlighted white-hot, and a sparkline tracks best + average fitness.
// Sibling of the gradient / hill-climbing optimization rooms.
//
// CPU evolution in plain Float32Arrays. The population is ONE THREE.Points draw
// call (buffers mutated in place), plus a small Points sprite for the current
// best, plus one Points cloud for the (static) landscape surface.
// ----------------------------------------------------------------------------

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil,
  bindRange, reducedMotion, setVariantCycler, addGrid, addSun,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070b);
scene.fog = new THREE.FogExp2(0x05070b, 0.0042);

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 96, 168);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; // reducedMotion is a const boolean
controls.autoRotateSpeed = 0.32;
controls.minDistance = 60;
controls.maxDistance = 520;
controls.target.set(0, 28, 0);

scene.add(new THREE.AmbientLight(0x2a3344, 0.9));
addGrid(scene, { size: 320, divisions: 32, y: -2 });
addSun(scene, { scale: 120, position: [0, 70, -300] });

// ---- a soft round sprite so points read as glowing balls -------------------
function makeSprite() {
  const s = 64;
  const cv = document.createElement("canvas");
  cv.width = cv.height = s;
  const g = cv.getContext("2d");
  const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.3, "rgba(255,255,255,0.85)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, s, s);
  return new THREE.CanvasTexture(cv);
}
const sprite = makeSprite();

// ---- cyan → green → amber → magenta ramp (NOT purple-dominant) --------------
function ramp(t) {
  const stops = [
    [0.30, 0.80, 0.90], // cyan
    [0.40, 0.95, 0.55], // green
    [0.98, 0.78, 0.30], // amber
    [0.95, 0.35, 0.85], // magenta
  ];
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const x = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// ---- fitness landscape: z = Σ Gaussian bumps (genome (x,y) → height) -------
const EXTENT = 120; // world units across (x,z span −EXTENT/2 … +EXTENT/2)
const GRID = 96;    // surface resolution per axis
const HSCALE = 34;  // vertical world scale for height
let bumps = [];
let landN = 6;      // number of Gaussian peaks in the current landscape

function makeBumps(n) {
  bumps = [];
  // One clearly dominant global peak + several lower local optima, so
  // convergence onto "the" optimum reads cleanly on screen.
  for (let i = 0; i < n; i++) {
    bumps.push({
      x: (Math.random() * 2 - 1) * 0.8,
      y: (Math.random() * 2 - 1) * 0.8,
      h: i === 0 ? 1.6 + Math.random() * 0.4 : 0.45 + Math.random() * 0.65,
      s: 0.05 + Math.random() * 0.1,
    });
  }
}

// fitness in normalized coords (x,y ∈ −1…1) → height ~0…2 (higher = fitter)
function fit(x, y) {
  let z = 0;
  for (const b of bumps) {
    const dx = x - b.x, dy = y - b.y;
    z += b.h * Math.exp(-(dx * dx + dy * dy) / (2 * b.s));
  }
  return z;
}

function worldX(nx) { return nx * EXTENT * 0.5; }
function worldZ(ny) { return ny * EXTENT * 0.5; }

// ---- landscape surface as one Points cloud ---------------------------------
const surfN = GRID * GRID;
const surfPos = new Float32Array(surfN * 3);
const surfCol = new Float32Array(surfN * 3);
const surfGeom = new THREE.BufferGeometry();
surfGeom.setAttribute("position", new THREE.BufferAttribute(surfPos, 3));
surfGeom.setAttribute("color", new THREE.BufferAttribute(surfCol, 3));
const surf = new THREE.Points(
  surfGeom,
  new THREE.PointsMaterial({
    size: 1.4, vertexColors: true, transparent: true, opacity: 0.45,
    depthWrite: false, blending: THREE.AdditiveBlending,
  }),
);
scene.add(surf);

function buildSurface() {
  let p = 0;
  for (let iy = 0; iy < GRID; iy++) {
    for (let ix = 0; ix < GRID; ix++) {
      const nx = (ix / (GRID - 1)) * 2 - 1;
      const ny = (iy / (GRID - 1)) * 2 - 1;
      const z = fit(nx, ny);
      surfPos[p * 3] = worldX(nx);
      surfPos[p * 3 + 1] = z * HSCALE;
      surfPos[p * 3 + 2] = worldZ(ny);
      const t = Math.min(1, z / 2);
      const c = ramp(t);
      // surface kept muted so the live population reads as the foreground
      surfCol[p * 3] = c[0] * 0.55;
      surfCol[p * 3 + 1] = c[1] * 0.55;
      surfCol[p * 3 + 2] = c[2] * 0.55;
      p++;
    }
  }
  surfGeom.attributes.position.needsUpdate = true;
  surfGeom.attributes.color.needsUpdate = true;
}

// ---- population state (genomes are 2-vectors in normalized space) ----------
const POP = 140;        // population size N
const ELITE = 4;        // genomes carried unchanged (elitism)
const TOURN_K = 3;      // tournament size
const G_MAX = 90;       // generations before a forced reseed
const CONVERGE = 0.012; // mean spread below this ⇒ converged, reseed

// gene[i*2], gene[i*2+1] are (x,y) ∈ [-1,1]; score[i] is fitness.
let gene = new Float32Array(POP * 2);
let next = new Float32Array(POP * 2);
const score = new Float32Array(POP);
const order = new Int32Array(POP); // indices sorted best→worst

let gen = 0;
let bestIdx = 0;
let bestFit = 0;
let avgFit = 0;
let spread = 1;

let mutRate = 0.06; // Gaussian mutation step size (slider)
let selMode = 0;    // 0 roulette · 1 tournament · 2 tournament + heavy elitism
const SEL_NAMES = ["roulette", "tournament", "tourn+elite"];

function clamp(v) { return v < -1 ? -1 : v > 1 ? 1 : v; }

// cheap Gaussian (sum of uniforms, ~unit variance) for mutation noise
function gauss() {
  return (Math.random() + Math.random() + Math.random() - 1.5) * 0.9428;
}

function seedPopulation() {
  for (let i = 0; i < POP; i++) {
    gene[i * 2] = Math.random() * 2 - 1;
    gene[i * 2 + 1] = Math.random() * 2 - 1;
  }
  gen = 0;
}

// ---- geometry: population as ONE Points cloud, best as a sprite ------------
const popPos = new Float32Array(POP * 3);
const popCol = new Float32Array(POP * 3);
const popGeom = new THREE.BufferGeometry();
popGeom.setAttribute("position", new THREE.BufferAttribute(popPos, 3));
popGeom.setAttribute("color", new THREE.BufferAttribute(popCol, 3));
const population = new THREE.Points(
  popGeom,
  new THREE.PointsMaterial({
    size: 4.5, vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, map: sprite,
  }),
);
population.frustumCulled = false;
scene.add(population);

// the current best genome, drawn larger and white-hot
const bestPos = new Float32Array(3);
const bestGeom = new THREE.BufferGeometry();
bestGeom.setAttribute("position", new THREE.BufferAttribute(bestPos, 3));
const bestMarker = new THREE.Points(
  bestGeom,
  new THREE.PointsMaterial({
    size: 13, color: 0xffffff, transparent: true, opacity: 0.95,
    depthWrite: false, blending: THREE.AdditiveBlending, map: sprite,
  }),
);
bestMarker.frustumCulled = false;
scene.add(bestMarker);

// ---- evaluate + selection helpers -----------------------------------------
function evaluate() {
  let sum = 0;
  bestFit = -Infinity;
  for (let i = 0; i < POP; i++) {
    const f = fit(gene[i * 2], gene[i * 2 + 1]);
    score[i] = f;
    sum += f;
    if (f > bestFit) { bestFit = f; bestIdx = i; }
  }
  avgFit = sum / POP;

  // sort indices best→worst (POP is small; insertion sort is plenty)
  for (let i = 0; i < POP; i++) order[i] = i;
  for (let i = 1; i < POP; i++) {
    const v = order[i];
    let j = i - 1;
    while (j >= 0 && score[order[j]] < score[v]) { order[j + 1] = order[j]; j--; }
    order[j + 1] = v;
  }

  // population spread = mean distance from the centroid (convergence gauge)
  let cx = 0, cy = 0;
  for (let i = 0; i < POP; i++) { cx += gene[i * 2]; cy += gene[i * 2 + 1]; }
  cx /= POP; cy /= POP;
  let d = 0;
  for (let i = 0; i < POP; i++) {
    const ddx = gene[i * 2] - cx, ddy = gene[i * 2 + 1] - cy;
    d += Math.sqrt(ddx * ddx + ddy * ddy);
  }
  spread = d / POP;
}

// roulette (fitness-proportional) selection — returns a genome index.
let rouletteTotal = 0;
function prepRoulette() {
  // shift so the worst genome still has a small (non-zero) chance
  const floor = score[order[POP - 1]];
  rouletteTotal = 0;
  for (let i = 0; i < POP; i++) rouletteTotal += score[i] - floor + 0.05;
}
function pickRoulette() {
  let r = Math.random() * rouletteTotal;
  const floor = score[order[POP - 1]];
  for (let i = 0; i < POP; i++) {
    r -= score[i] - floor + 0.05;
    if (r <= 0) return i;
  }
  return order[0];
}

// tournament selection — best of TOURN_K random contenders.
function pickTournament() {
  let best = (Math.random() * POP) | 0;
  for (let t = 1; t < TOURN_K; t++) {
    const c = (Math.random() * POP) | 0;
    if (score[c] > score[best]) best = c;
  }
  return best;
}

function pickParent() {
  return selMode === 0 ? pickRoulette() : pickTournament();
}

// ---- one generation: select → crossover → mutate → elitism ----------------
function generation() {
  evaluate();

  if (selMode === 0) prepRoulette();
  const eliteCount = selMode === 2 ? ELITE * 2 : ELITE; // "tourn+elite" keeps more

  // elitism: copy the top genomes through unchanged
  for (let e = 0; e < eliteCount; e++) {
    const src = order[e];
    next[e * 2] = gene[src * 2];
    next[e * 2 + 1] = gene[src * 2 + 1];
  }

  // fill the rest with crossover + mutation of selected parents
  for (let i = eliteCount; i < POP; i++) {
    const a = pickParent();
    const b = pickParent();
    // uniform crossover: each gene from either parent
    const gx = Math.random() < 0.5 ? gene[a * 2] : gene[b * 2];
    const gy = Math.random() < 0.5 ? gene[a * 2 + 1] : gene[b * 2 + 1];
    // Gaussian mutation
    next[i * 2] = clamp(gx + gauss() * mutRate);
    next[i * 2 + 1] = clamp(gy + gauss() * mutRate);
  }

  // swap buffers
  const tmp = gene; gene = next; next = tmp;
  gen++;

  pushHistory(bestFit, avgFit);

  // restart when converged or after G_MAX generations
  if (gen >= G_MAX || (spread < CONVERGE && gen > 8)) {
    if (Math.random() < 0.5) { makeBumps(landN); buildSurface(); }
    seedPopulation();
    resetHistory();
  }
}

// ---- push genomes onto the surface for rendering ---------------------------
function updatePoints() {
  for (let i = 0; i < POP; i++) {
    const x = gene[i * 2], y = gene[i * 2 + 1];
    const z = fit(x, y);
    popPos[i * 3] = worldX(x);
    popPos[i * 3 + 1] = z * HSCALE + 1.5;
    popPos[i * 3 + 2] = worldZ(y);
    // color by fitness: dim (low) → bright (high)
    const t = Math.min(1, z / 2);
    const c = ramp(t);
    const glow = 0.35 + 0.65 * t; // dim low-fitness genomes
    popCol[i * 3] = c[0] * glow;
    popCol[i * 3 + 1] = c[1] * glow;
    popCol[i * 3 + 2] = c[2] * glow;
  }
  popGeom.attributes.position.needsUpdate = true;
  popGeom.attributes.color.needsUpdate = true;

  const bx = gene[bestIdx * 2], by = gene[bestIdx * 2 + 1];
  bestPos[0] = worldX(bx);
  bestPos[1] = fit(bx, by) * HSCALE + 3.0;
  bestPos[2] = worldZ(by);
  bestGeom.attributes.position.needsUpdate = true;
}

// ---- fitness sparkline (best + avg over recent generations) ----------------
const SPARK = 120;
const histBest = new Float32Array(SPARK);
const histAvg = new Float32Array(SPARK);
let histLen = 0;
const sparkCanvas = document.getElementById("spark");
const sparkCtx = sparkCanvas ? sparkCanvas.getContext("2d") : null;

function resetHistory() { histLen = 0; }
function pushHistory(b, a) {
  if (histLen < SPARK) {
    histBest[histLen] = b; histAvg[histLen] = a; histLen++;
  } else {
    histBest.copyWithin(0, 1); histAvg.copyWithin(0, 1);
    histBest[SPARK - 1] = b; histAvg[SPARK - 1] = a;
  }
}
function drawSpark() {
  if (!sparkCtx) return;
  const W = sparkCanvas.width, H = sparkCanvas.height;
  sparkCtx.clearRect(0, 0, W, H);
  if (histLen < 2) return;
  const lo = 0, hi = 2; // fixed range matches the 0..2 landscape height
  const xAt = (i) => (i / (SPARK - 1)) * W;
  const yAt = (v) => H - ((v - lo) / (hi - lo)) * (H - 4) - 2;
  const drawLine = (arr, col) => {
    sparkCtx.beginPath();
    for (let i = 0; i < histLen; i++) {
      const x = xAt(i), y = yAt(arr[i]);
      if (i === 0) sparkCtx.moveTo(x, y); else sparkCtx.lineTo(x, y);
    }
    sparkCtx.strokeStyle = col;
    sparkCtx.lineWidth = 1.5;
    sparkCtx.stroke();
  };
  drawLine(histAvg, "rgba(90,180,210,0.75)");  // avg — cool cyan
  drawLine(histBest, "rgba(250,200,80,0.95)"); // best — amber
}

// ---- UI --------------------------------------------------------------------
let simSpeed = 1.0; // generations per frame multiplier
let genAccum = 0;

bindRange("speed", (v) => { simSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("mutate", (v) => { mutRate = v; }, (v) => v.toFixed(3));

// ↑/↓ cycles the selection method so its effect on convergence is visible.
// Returns the new variant label for the kiosk toast.
setVariantCycler((dir) => {
  selMode = (selMode + dir + SEL_NAMES.length) % SEL_NAMES.length;
  seedPopulation();
  resetHistory();
  if (selEl) selEl.textContent = SEL_NAMES[selMode];
  return SEL_NAMES[selMode];
});

const selEl = document.getElementById("sel-name");
const genEl = document.getElementById("gen");
const bestEl = document.getElementById("best");
if (selEl) selEl.textContent = SEL_NAMES[selMode];

// ---- boot ------------------------------------------------------------------
makeBumps(landN);
buildSurface();
seedPopulation();
evaluate();

const meter = fpsMeter(document.getElementById("fps"));
liftVeil();
onResize(renderer, camera);

loop((dt) => {
  meter(dt);
  controls.update();

  // advance whole generations; cap per-frame for 60fps
  genAccum += simSpeed;
  let steps = 0;
  while (genAccum >= 1 && steps < 3) {
    generation();
    genAccum -= 1;
    steps++;
  }
  if (steps === 0) evaluate(); // keep best/avg fresh when paused/slow

  updatePoints();
  drawSpark();

  if (genEl) genEl.textContent = gen;
  if (bestEl) bestEl.textContent = bestFit.toFixed(2);

  renderer.render(scene, camera);
});

window.__diag = () =>
  JSON.stringify({
    gen,
    best: Number(bestFit.toFixed(4)),
    avg: Number(avgFit.toFixed(4)),
    pop: POP,
  });
