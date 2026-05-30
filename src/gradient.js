// gradient.js — gradient descent on a 3D loss landscape. Three optimizers
// (SGD, Momentum, Adam) start from the same point and roll downhill, leaving
// glowing light trails. The surface is a sum of Gaussian wells (analytic
// gradient), colored by height. ↑↓ switches landscapes. 3D calculus × ML.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.014);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(14, 12, 16);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.45;
controls.minDistance = 6; controls.maxDistance = 80;
controls.target.set(0, -1, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.85));
const key = new THREE.DirectionalLight(0xfff1dd, 1.2); key.position.set(8, 14, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0xff2e97, 0.9); rim.position.set(-10, 6, -8); scene.add(rim);
const fill = new THREE.DirectionalLight(0x2be4ff, 0.5); fill.position.set(4, -6, 8); scene.add(fill);
addGrid(scene, { size: 60, divisions: 30, y: -8 });
addSun(scene, { scale: 44, position: [0, 10, -64] });

// ---------- landscapes: arrays of Gaussian wells over a gentle bowl ----------
const D = 8;            // domain half-width
const HSCALE = 1.0;
const LANDSCAPES = [
  ["twin basins", [[-3, -1, 3.2, 1.8], [3.5, 1.5, 3.8, 2.0]]],
  ["four corners", [[-4, -4, 3, 1.7], [4, -4, 3, 1.7], [-4, 4, 3, 1.7], [4, 4, 3.4, 1.8]]],
  ["deep & shallow", [[-3.5, 0, 1.6, 1.2], [3.5, 0, 4.5, 2.3], [0, 4, 2.2, 1.6]]],
  ["ridge", [[0, -4, 3.6, 2.0], [-4, 3, 2.8, 1.5], [4, 3, 2.8, 1.5]]],
];
let wells = LANDSCAPES[0][1];

function f(x, y) {
  let z = 0.05 * (x * x + y * y);
  for (const [wx, wy, depth, s] of wells) {
    const r2 = (x - wx) ** 2 + (y - wy) ** 2;
    z -= depth * Math.exp(-r2 / (2 * s * s));
  }
  return z * HSCALE;
}
function grad(x, y) {
  let gx = 0.1 * x, gy = 0.1 * y;
  for (const [wx, wy, depth, s] of wells) {
    const r2 = (x - wx) ** 2 + (y - wy) ** 2;
    const e = depth * Math.exp(-r2 / (2 * s * s)) / (s * s);
    gx += e * (x - wx); gy += e * (y - wy);
  }
  return [gx * HSCALE, gy * HSCALE];
}

// ---------- surface mesh ----------
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
  new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.7, metalness: 0.1, side: THREE.DoubleSide, transparent: true, opacity: 0.92, wireframe: false })
);
scene.add(surface);
const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0x2be4ff, wireframe: true, transparent: true, opacity: 0.08 }));
scene.add(wire);

function buildSurface() {
  let lo = 1e9, hi = -1e9;
  const step = (2 * D) / (N - 1);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x = -D + i * step, y = -D + j * step, z = f(x, y);
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

// ---------- optimizers ----------
const TRAIL = 260;
function makeBall(color) {
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.28, 20, 16),
    new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.6, roughness: 0.3 }));
  scene.add(ball);
  const tgeo = new THREE.BufferGeometry();
  const tpos = new Float32Array(TRAIL * 3);
  tgeo.setAttribute("position", new THREE.BufferAttribute(tpos, 3).setUsage(THREE.DynamicDrawUsage));
  const trail = new THREE.Line(tgeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
  scene.add(trail);
  return { ball, trail, tpos, n: 0 };
}
const OPT = {
  sgd:      { color: 0x2be4ff, vis: makeBall(0x2be4ff), lr: 0.55 },
  momentum: { color: 0xff2e97, vis: makeBall(0xff2e97), lr: 0.35, mu: 0.85 },
  adam:     { color: 0x62ffb3, vis: makeBall(0x62ffb3), lr: 0.45, b1: 0.9, b2: 0.999 },
  anneal:   { color: 0xb06bff, vis: makeBall(0xb06bff), T0: 4.0, cool: 0.0009 }, // simulated annealing
};

let x = {}, y = {}, st = {};
let speed = 1;
let steps = 0;

function resetRun() {
  const sx = (Math.random() * 2 - 1) * D * 0.8;
  const sy = (Math.random() * 2 - 1) * D * 0.8;
  for (const k of Object.keys(OPT)) {
    x[k] = sx; y[k] = sy;
    st[k] = { vx: 0, vy: 0, mx: 0, my: 0, t: 0, T: OPT[k].T0 || 0, e: f(sx, sy) };
    OPT[k].vis.n = 0;
  }
  steps = 0;
}

function stepOpt(k) {
  const o = OPT[k], s = st[k];
  const [gx, gy] = grad(x[k], y[k]);
  if (k === "anneal") {
    // simulated annealing: random hop, accept worse moves with prob e^(-ΔE/T).
    // Lets it tunnel OUT of local minima the gradient methods fall into.
    s.T = Math.max(0.02, s.T - o.cool);
    const span = 0.3 + s.T * 0.9;          // explore wider while hot
    const nx = Math.max(-D, Math.min(D, x[k] + (Math.random() * 2 - 1) * span));
    const ny = Math.max(-D, Math.min(D, y[k] + (Math.random() * 2 - 1) * span));
    const ne = f(nx, ny);
    const dE = ne - s.e;
    if (dE < 0 || Math.random() < Math.exp(-dE / s.T)) { x[k] = nx; y[k] = ny; s.e = ne; }
    return;
  }
  if (k === "sgd") { x[k] -= o.lr * gx; y[k] -= o.lr * gy; }
  else if (k === "momentum") {
    s.vx = o.mu * s.vx - o.lr * gx; s.vy = o.mu * s.vy - o.lr * gy;
    x[k] += s.vx; y[k] += s.vy;
  } else { // adam
    s.t++;
    s.vx = o.b1 * s.vx + (1 - o.b1) * gx; s.vy = o.b1 * s.vy + (1 - o.b1) * gy;
    s.mx = o.b2 * s.mx + (1 - o.b2) * gx * gx; s.my = o.b2 * s.my + (1 - o.b2) * gy * gy;
    const cx = s.vx / (1 - o.b1 ** s.t), cy = s.vy / (1 - o.b1 ** s.t);
    const dx = s.mx / (1 - o.b2 ** s.t), dy = s.my / (1 - o.b2 ** s.t);
    x[k] -= o.lr * cx / (Math.sqrt(dx) + 1e-8);
    y[k] -= o.lr * cy / (Math.sqrt(dy) + 1e-8);
  }
  x[k] = Math.max(-D, Math.min(D, x[k]));
  y[k] = Math.max(-D, Math.min(D, y[k]));
}

function place() {
  let moving = false;
  for (const k of Object.keys(OPT)) {
    const z = f(x[k], y[k]) + 0.28;
    const v = OPT[k].vis;
    v.ball.position.set(x[k], z, y[k]);
    // push trail point
    if (v.n < TRAIL) { v.tpos.set([x[k], z, y[k]], v.n * 3); v.n++; }
    else { v.tpos.copyWithin(0, 3); v.tpos.set([x[k], z, y[k]], (TRAIL - 1) * 3); }
    v.trail.geometry.setDrawRange(0, v.n);
    v.trail.geometry.attributes.position.needsUpdate = true;
    // annealing keeps "moving" until it has cooled; others until the gradient flattens
    if (k === "anneal") { if (st[k].T > 0.06) moving = true; }
    else { const [gx, gy] = grad(x[k], y[k]); if (Math.hypot(gx, gy) > 0.02) moving = true; }
  }
  return moving;
}

// ---------- panel ----------
const wrap = document.getElementById("landscapes");
let landIdx = 0;
const chips = LANDSCAPES.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { landIdx = i; wells = LANDSCAPES[i][1]; buildSurface(); resetRun(); chips.forEach((c, k) => c.classList.toggle("active", k === i)); });
  wrap.appendChild(b);
  return b;
});
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
document.getElementById("restart").addEventListener("click", resetRun);
const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => { camera.position.copy(home); controls.target.set(0, -1, 0); });

setVariantCycler((d) => {
  landIdx = (landIdx + d + LANDSCAPES.length) % LANDSCAPES.length;
  wells = LANDSCAPES[landIdx][1]; buildSurface(); resetRun();
  chips.forEach((c, k) => c.classList.toggle("active", k === landIdx));
  return LANDSCAPES[landIdx][0];
});

// ---------- boot ----------
buildSurface();
resetRun();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const stepsEl = document.getElementById("steps");

let acc = 0, settle = 0;
loop((dt) => {
  meter(dt);
  acc += dt * speed * 30;
  let budget = Math.min(acc | 0, 6);
  acc -= budget;
  let moving = true;
  while (budget-- > 0) { for (const k of Object.keys(OPT)) stepOpt(k); steps++; }
  moving = place();
  if (!moving) { settle += dt; if (settle > 1.6) { settle = 0; resetRun(); } } else settle = 0;
  stepsEl.textContent = steps;
  controls.update();
  renderer.render(scene, camera);
});
