// attractor.js — strange attractors integrated on the CPU, drawn as a
// progressively-revealed glowing ribbon (additive blending over a dark field).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 6, 30);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.6;
controls.minDistance = 6;
controls.maxDistance = 120;

addGrid(scene, { size: 90, divisions: 45, y: -14 });
addSun(scene, { scale: 60, position: [0, 24, -100] });

// ---------- the attractors ----------
// each: deriv(p, k) writes dx/dy/dz into k; dt; start; one-line blurb.
const SYSTEMS = {
  lorenz: {
    about: "Edward Lorenz, 1963. The butterfly. A toy weather model whose two lobes a trajectory hops between, forever, never repeating.",
    dt: 0.005, start: { x: 0.1, y: 0, z: 0 },
    deriv: (p, k) => { const s = 10, r = 28, b = 8 / 3;
      k.x = s * (p.y - p.x); k.y = p.x * (r - p.z) - p.y; k.z = p.x * p.y - b * p.z; },
  },
  aizawa: {
    about: "The Aizawa attractor — a spherical shell pierced by a spiralling spindle. Six parameters of pure ornament.",
    dt: 0.01, start: { x: 0.1, y: 0, z: 0 },
    deriv: (p, k) => { const a = 0.95, b = 0.7, c = 0.6, d = 3.5, e = 0.25, f = 0.1;
      k.x = (p.z - b) * p.x - d * p.y;
      k.y = d * p.x + (p.z - b) * p.y;
      k.z = c + a * p.z - (p.z ** 3) / 3 - (p.x * p.x + p.y * p.y) * (1 + e * p.z) + f * p.z * (p.x ** 3); },
  },
  thomas: {
    about: "René Thomas' cyclically-symmetric attractor. A particle wandering a lattice of its own sine-wave feedback.",
    dt: 0.018, start: { x: 1.1, y: 1.1, z: -0.5 },
    deriv: (p, k) => { const b = 0.208186;
      k.x = Math.sin(p.y) - b * p.x;
      k.y = Math.sin(p.z) - b * p.y;
      k.z = Math.sin(p.x) - b * p.z; },
  },
  halvorsen: {
    about: "The Halvorsen attractor — three-fold symmetry folding in on itself like a knotted ribbon.",
    dt: 0.005, start: { x: -5, y: 0, z: 0 },
    deriv: (p, k) => { const a = 1.4;
      k.x = -a * p.x - 4 * p.y - 4 * p.z - p.y * p.y;
      k.y = -a * p.y - 4 * p.z - 4 * p.x - p.z * p.z;
      k.z = -a * p.z - 4 * p.x - 4 * p.y - p.x * p.x; },
  },
  dadras: {
    about: "The Dadras–Momeni system. A four-winged sweep, wide and calligraphic.",
    dt: 0.004, start: { x: 1.1, y: 2.1, z: -2 },
    deriv: (p, k) => { const a = 3, b = 2.7, c = 1.7, d = 2, e = 9;
      k.x = p.y - a * p.x + b * p.y * p.z;
      k.y = c * p.y - p.x * p.z + p.z;
      k.z = d * p.x * p.y - e * p.z; },
  },
};

const N = 130000; // trajectory length
const positions = new Float32Array(N * 3);
const colors = new Float32Array(N * 3);

const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

const material = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.62,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});
const ribbon = new THREE.Line(geometry, material);
scene.add(ribbon);

// RK4 integrate a system into the position buffer, then center + scale to fit.
const k1 = {}, k2 = {}, k3 = {}, k4 = {}, tmp = {};
function integrate(name) {
  const sys = SYSTEMS[name];
  const { dt, deriv } = sys;
  let { x, y, z } = sys.start;
  let minX = 1e9, minY = 1e9, minZ = 1e9, maxX = -1e9, maxY = -1e9, maxZ = -1e9;

  for (let i = 0; i < N; i++) {
    positions[i * 3] = x; positions[i * 3 + 1] = y; positions[i * 3 + 2] = z;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;

    // RK4 step
    deriv({ x, y, z }, k1);
    tmp.x = x + k1.x * dt / 2; tmp.y = y + k1.y * dt / 2; tmp.z = z + k1.z * dt / 2; deriv(tmp, k2);
    tmp.x = x + k2.x * dt / 2; tmp.y = y + k2.y * dt / 2; tmp.z = z + k2.z * dt / 2; deriv(tmp, k3);
    tmp.x = x + k3.x * dt;     tmp.y = y + k3.y * dt;     tmp.z = z + k3.z * dt;     deriv(tmp, k4);
    x += (k1.x + 2 * k2.x + 2 * k3.x + k4.x) * dt / 6;
    y += (k1.y + 2 * k2.y + 2 * k3.y + k4.y) * dt / 6;
    z += (k1.z + 2 * k2.z + 2 * k3.z + k4.z) * dt / 6;
  }

  // center + uniform scale to a pleasant size
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
  const s = 22 / span;
  for (let i = 0; i < N; i++) {
    positions[i * 3]     = (positions[i * 3] - cx) * s;
    positions[i * 3 + 1] = (positions[i * 3 + 1] - cy) * s;
    positions[i * 3 + 2] = (positions[i * 3 + 2] - cz) * s;
    const c = ramp(i / N);
    colors[i * 3] = c[0]; colors[i * 3 + 1] = c[1]; colors[i * 3 + 2] = c[2];
  }
  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
  geometry.computeBoundingSphere();
}

// ---------- reveal state ----------
let drawn = 0;          // points currently shown
let revealRate = N / 7; // points per second at speed 1
let speed = 1;
let current = "lorenz";

function load(name) {
  current = name;
  integrate(name);
  drawn = reducedMotion ? N : 0;
  geometry.setDrawRange(0, drawn);
  document.getElementById("name").textContent = name;
  document.getElementById("about").textContent = SYSTEMS[name].about;
  systemsWrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.k === name));
}

// ---------- panel ----------
const systemsWrap = document.getElementById("systems");
Object.keys(SYSTEMS).forEach((name) => {
  const b = document.createElement("button");
  b.className = "chip";
  b.dataset.k = name;
  b.textContent = name;
  b.addEventListener("click", () => load(name));
  systemsWrap.appendChild(b);
});

bindRange("speed", (v) => { speed = v; controls.autoRotateSpeed = 0.6 * Math.max(v, 0.15); }, (v) => v.toFixed(2) + "×");

let spinning = !reducedMotion;
const spinBtn = document.getElementById("spin");
spinBtn.addEventListener("click", () => {
  spinning = !spinning;
  controls.autoRotate = spinning;
  spinBtn.classList.toggle("active", spinning);
});

document.getElementById("replay").addEventListener("click", () => { drawn = 0; });

const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => {
  camera.position.copy(home);
  controls.target.set(0, 0, 0);
});

// ---------- boot ----------
load("lorenz");
liftVeil();

onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const ptsEl = document.getElementById("pts");

loop((dt) => {
  meter(dt);
  if (drawn < N) {
    drawn = Math.min(N, drawn + revealRate * Math.max(speed, 0.05) * dt);
    geometry.setDrawRange(0, Math.floor(drawn));
  }
  ptsEl.textContent = Math.floor(drawn).toLocaleString();
  controls.update();
  renderer.render(scene, camera);
});
