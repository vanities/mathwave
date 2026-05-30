// parametric.js — a height-field playground.
// z = f(x, y, t), evaluated per-vertex with math.js, colored by altitude.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun,
} from "./common.js";

const math = window.math;

// ---------- scene ----------
const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.02);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 500);
camera.position.set(9, 8, 11);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 3;
controls.maxDistance = 60;
controls.target.set(0, 0, 0);

// lights — warm key, hot-pink rim, cyan fill (vaporwave)
scene.add(new THREE.AmbientLight(0x3a2466, 0.85));
const key = new THREE.DirectionalLight(0xfff1dd, 1.3);
key.position.set(6, 12, 4);
scene.add(key);
const rim = new THREE.DirectionalLight(0xff2e97, 1.05);
rim.position.set(-8, 4, -6);
scene.add(rim);
const fill = new THREE.DirectionalLight(0x2be4ff, 0.6);
fill.position.set(2, -6, 8);
scene.add(fill);

addGrid(scene, { size: 80, divisions: 40, y: -8 });
addSun(scene, { scale: 46, position: [0, 16, -70] });

// ---------- the surface mesh ----------
const material = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 0.62,
  metalness: 0.08,
  side: THREE.DoubleSide,
  wireframe: true,
  flatShading: false,
});
let mesh = null;
let geo = null;

// state
let N = 120;            // grid resolution (N x N)
let D = 6.5;            // domain half-width
let speed = 1;          // time multiplier
let compiled = null;    // math.js compiled expr
let usesTime = true;    // does the expression reference t?
let needsRebuild = true;

const statusEl = document.getElementById("status");
const vertsEl = document.getElementById("verts");

// cached scope object to avoid per-vertex allocation
const scope = { x: 0, y: 0, t: 0 };

function compile(src) {
  try {
    const node = math.parse(src);
    const c = node.compile();
    // smoke-test it returns a finite number
    const test = c.evaluate({ x: 0.3, y: -0.4, t: 0.5 });
    if (typeof test !== "number" || !isFinite(test)) throw new Error("not a number");
    compiled = c;
    usesTime = /\bt\b/.test(src);
    statusEl.textContent = "ok";
    statusEl.style.color = "";
    document.getElementById("eqn").classList.remove("invalid");
    needsRebuild = true;
    return true;
  } catch (e) {
    statusEl.textContent = "✗ " + (e.message || "parse error").slice(0, 22);
    statusEl.style.color = "#e69a7d";
    document.getElementById("eqn").classList.add("invalid");
    return false;
  }
}

// (re)allocate geometry buffers for the current resolution
function allocate() {
  if (geo) geo.dispose();
  geo = new THREE.BufferGeometry();
  const count = N * N;
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  const idx = [];
  for (let i = 0; i < N - 1; i++) {
    for (let j = 0; j < N - 1; j++) {
      const a = i * N + j;
      idx.push(a, a + 1, a + N, a + 1, a + N + 1, a + N);
    }
  }
  geo.setIndex(idx);

  if (mesh) scene.remove(mesh);
  mesh = new THREE.Mesh(geo, material);
  scene.add(mesh);
  vertsEl.textContent = count.toLocaleString();
}

// evaluate f over the grid into the buffers
function evaluate(t) {
  if (!compiled) return;
  const pos = geo.attributes.position.array;
  const col = geo.attributes.color.array;
  const step = (2 * D) / (N - 1);
  scope.t = t;
  let p = 0;
  for (let i = 0; i < N; i++) {
    const x = -D + i * step;
    scope.x = x;
    for (let j = 0; j < N; j++) {
      const y = -D + j * step;
      scope.y = y;
      let z = compiled.evaluate(scope);
      if (!isFinite(z)) z = 0;
      // map XZ to ground plane, height up Y
      pos[p] = x; pos[p + 1] = z; pos[p + 2] = y;
      // color by a bounded function of height
      const h = 0.5 + 0.5 * Math.tanh(z * 0.55);
      const c = ramp(h);
      col[p] = c[0]; col[p + 1] = c[1]; col[p + 2] = c[2];
      p += 3;
    }
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
}

// ---------- controls / panel ----------
const PRESETS = [
  ["ripple", "sin(hypot(x,y)*3 - t*2) / (1 + hypot(x,y)*0.5)"],
  ["waves", "sin(x + t) * cos(y + t*0.7)"],
  ["interference", "(sin(hypot(x-2,y)*3 - t*2) + sin(hypot(x+2,y)*3 - t*2)) * 0.6"],
  ["saddle", "(x^2 - y^2) / 6"],
  ["monkey", "(x^3 - 3*x*y^2) / 14"],
  ["peaks", "3*(1-x/3)^2*exp(-(x/3)^2-(y/3+1)^2) - 10*(x/15-(x/3)^3-(y/3)^5)*exp(-(x/3)^2-(y/3)^2)"],
  ["vortex", "sin(atan2(y,x)*5 + hypot(x,y)*2 - t*3) * 0.8"],
  ["egg", "sin(x*1.5)*sin(y*1.5)"],
];
const presetWrap = document.getElementById("presets");
const eqnInput = document.getElementById("eqn");
PRESETS.forEach(([name, src]) => {
  const b = document.createElement("button");
  b.className = "chip";
  b.textContent = name;
  b.addEventListener("click", () => {
    eqnInput.value = src;
    if (compile(src)) markActiveChip(b);
  });
  presetWrap.appendChild(b);
});
function markActiveChip(active) {
  presetWrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === active));
}
// mark the first preset (matches the default equation) active
markActiveChip(presetWrap.firstChild);

eqnInput.addEventListener("input", () => {
  compile(eqnInput.value);
  markActiveChip(null);
});

bindRange("res", (v) => { N = Math.round(v); allocate(); needsRebuild = true; }, (v) => `${Math.round(v)}²`);
bindRange("domain", (v) => { D = v; needsRebuild = true; }, (v) => v.toFixed(1));
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");

const wireBtn = document.getElementById("wire");
wireBtn.addEventListener("click", () => {
  material.wireframe = !material.wireframe;
  wireBtn.classList.toggle("active", material.wireframe);
});

let spinning = false;
const spinBtn = document.getElementById("spin");
spinBtn.addEventListener("click", () => {
  spinning = !spinning;
  controls.autoRotate = spinning;
  controls.autoRotateSpeed = 0.8;
  spinBtn.classList.toggle("active", spinning);
});

const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => {
  camera.position.copy(home);
  controls.target.set(0, 0, 0);
});

// ---------- boot ----------
compile(eqnInput.value);
allocate();
evaluate(0);
liftVeil();

onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt, elapsed) => {
  meter(dt);
  const t = elapsed * speed;
  // rebuild when the geometry params changed, or every frame if animated
  if (needsRebuild || (usesTime && speed > 0 && !reducedMotion)) {
    evaluate(t);
    needsRebuild = false;
  }
  controls.update();
  renderer.render(scene, camera);
});
