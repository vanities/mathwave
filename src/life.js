// life.js — a TRUE 3D cellular automaton (3D Game of Life).
// Each cell has 26 Moore neighbors; rules are Carter Bays' survive/born ranges.
// Rendered as hollow neon voxel shells (interior cells culled) so you can read
// the structure, orbiting slowly over the grid. Built to be filmed.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

// ---------- grid ----------
const N = 30;                 // N×N×N cells
const COUNT = N * N * N;
const CC = (N - 1) / 2;       // center coordinate
const CELL = 1.0;
const MAXR = Math.hypot(CC, CC, CC);

let alive = new Uint8Array(COUNT);
let next = new Uint8Array(COUNT);
const age = new Float32Array(COUNT);
const sEase = new Float32Array(COUNT);   // eased render scale (smooth birth/death)

const idx = (x, y, z) => (x * N + y) * N + z;

// ---------- rule: survive Smin..Smax, born Bmin..Bmax (26-neighborhood) ----------
let SMIN = 5, SMAX = 7, BMIN = 6, BMAX = 6;
const RULES = {
  "5766":        [5, 7, 6, 6],     // Bays' 3D Life — has gliders
  "4555":        [4, 5, 5, 5],     // Bays — stable blobs
  "pyroclastic": [4, 7, 6, 8],     // roiling smoke
  "clouds":      [13, 26, 14, 19], // slow billowing clouds
  "crystal":     [5, 8, 5, 6],     // growing lattice
};
let ruleName = "5766";
function setRule(name) {
  ruleName = name;
  [SMIN, SMAX, BMIN, BMAX] = RULES[name];
}

// ---------- scene ----------
const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.011);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, N * 0.38, N * 1.05);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.55;
controls.minDistance = 8;
controls.maxDistance = N * 2.6;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xff2e97, 1.15); key.position.set(14, 22, 10); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.9); rim.position.set(-16, 8, -12); scene.add(rim);
const top = new THREE.DirectionalLight(0xfff1dd, 0.5); top.position.set(0, 26, 2); scene.add(top);

addGrid(scene, { size: N * 1.4, divisions: N, y: -N * 0.5 - 2 });
addSun(scene, { scale: N * 0.95, position: [0, 2, -N * 1.5] });

// ---------- instanced voxels ----------
const geo = new THREE.BoxGeometry(CELL * 0.92, CELL * 0.92, CELL * 0.92);
const mat = new THREE.MeshStandardMaterial({ roughness: 0.34, metalness: 0.25 });
const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
mesh.count = 0;
scene.add(mesh);
const dummy = new THREE.Object3D();
const color = new THREE.Color();

// ---------- the rule step ----------
let gen = 0;
function step() {
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      for (let z = 0; z < N; z++) {
        let n = 0;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= N) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const yy = y + dy; if (yy < 0 || yy >= N) continue;
            const base = (xx * N + yy) * N;
            for (let dz = -1; dz <= 1; dz++) {
              if (dx === 0 && dy === 0 && dz === 0) continue;
              const zz = z + dz; if (zz < 0 || zz >= N) continue;
              n += alive[base + zz];
            }
          }
        }
        const i = idx(x, y, z);
        next[i] = alive[i] ? (n >= SMIN && n <= SMAX ? 1 : 0) : (n >= BMIN && n <= BMAX ? 1 : 0);
      }
    }
  }
  for (let i = 0; i < COUNT; i++) age[i] = next[i] ? (alive[i] ? age[i] + 1 : 1) : 0;
  const t = alive; alive = next; next = t;
  gen++;
}

// a cell shows only if it's on the surface of a solid mass (hollow shells)
function isSurface(x, y, z) {
  if (x === 0 || x === N - 1 || y === 0 || y === N - 1 || z === 0 || z === N - 1) return true;
  return !alive[idx(x - 1, y, z)] || !alive[idx(x + 1, y, z)] ||
         !alive[idx(x, y - 1, z)] || !alive[idx(x, y + 1, z)] ||
         !alive[idx(x, y, z - 1)] || !alive[idx(x, y, z + 1)];
}

// ---------- seeding ----------
function clear() { alive.fill(0); next.fill(0); age.fill(0); gen = 0; }
function reseed(radius = 6, p = 0.34) {
  clear();
  const c = Math.round(CC);
  for (let x = -radius; x <= radius; x++)
    for (let y = -radius; y <= radius; y++)
      for (let z = -radius; z <= radius; z++) {
        if (Math.random() < p) alive[idx(c + x, c + y, c + z)] = 1;
      }
  if (!reducedMotion) sEase.fill(0);   // grow in from nothing
}

// ---------- render ----------
let aliveCount = 0;
function draw(elapsed) {
  let n = 0;
  aliveCount = 0;
  const ease = reducedMotion ? 1 : 0.2;
  const flow = elapsed * 0.04;
  for (let i = 0; i < COUNT; i++) {
    const a = alive[i];
    if (a) aliveCount++;
    let target = 0;
    if (a) {
      const x = (i / (N * N)) | 0, rem = i % (N * N), y = (rem / N) | 0, z = rem % N;
      target = isSurface(x, y, z) ? 1 : 0;     // interior cells stay hidden
    }
    const s = (sEase[i] += (target - sEase[i]) * ease);
    if (s > 0.04) {
      const x = (i / (N * N)) | 0, rem = i % (N * N), y = (rem / N) | 0, z = rem % N;
      const dx = x - CC, dy = y - CC, dz = z - CC;
      dummy.position.set(dx * CELL, dy * CELL, dz * CELL);
      dummy.scale.setScalar(s * 0.92);
      dummy.updateMatrix();
      mesh.setMatrixAt(n, dummy.matrix);
      const r = Math.hypot(dx, dy, dz) / MAXR;          // shell gradient...
      const c = ramp((r * 0.7 + flow) % 1);             // ...flowing through the ramp
      color.setRGB(c[0], c[1], c[2]);
      mesh.setColorAt(n, color);
      n++;
    }
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------- panel ----------
const wrap = document.getElementById("patterns");
Object.keys(RULES).forEach((name, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = name;
  b.addEventListener("click", () => {
    setRule(name);
    reseed();
    wrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === b));
  });
  wrap.appendChild(b);
});

let gps = 6;
bindRange("speed", (v) => { gps = v; }, (v) => `${Math.round(v)}`);

let playing = true;
const playBtn = document.getElementById("play");
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
  playBtn.classList.toggle("active", playing);
});
document.getElementById("step").addEventListener("click", () => step());
document.getElementById("clear").addEventListener("click", () => reseed());

// ↑/↓ cycle the rules
const _ruleNames = Object.keys(RULES);
let _variantIdx = 0;
setVariantCycler((d) => {
  _variantIdx = (_variantIdx + d + _ruleNames.length) % _ruleNames.length;
  setRule(_ruleNames[_variantIdx]); reseed();
  wrap.querySelectorAll(".chip").forEach((c, k) => c.classList.toggle("active", k === _variantIdx));
  return _ruleNames[_variantIdx];
});

// ---------- boot ----------
setRule("5766");
reseed();
draw(0);
liftVeil();

onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const genEl = document.getElementById("gen");
const aliveEl = document.getElementById("alive");

let acc = 0;
loop((dt, elapsed) => {
  meter(dt);
  if (playing) {
    acc += dt;
    const stepEvery = 1 / gps;
    let budget = 3;
    while (acc >= stepEvery && budget-- > 0) { step(); acc -= stepEvery; }
    if (acc > stepEvery) acc = 0;
  }
  draw(elapsed);
  genEl.textContent = gen;
  aliveEl.textContent = aliveCount.toLocaleString();
  controls.update();
  renderer.render(scene, camera);
});
