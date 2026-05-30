// orbitals.js — atomic orbitals via REAL spherical harmonics Yₗᵐ(θ,φ).
// The shape you see is the angular probability lobe: each direction (θ,φ) is
// pushed out to radius |Yₗᵐ| and two-tone colored by the SIGN of Yₗᵐ (the wave-
// function phase) — cyan for +, magenta for −. These are the genuine closed-form
// real harmonics, so s / p / d / f orbitals come out exactly right. The crossover
// of physics (the Schrödinger angular solutions) and chemistry (orbital shapes).
// ↑↓ steps through the orbitals.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(7, 5, 9);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
controls.minDistance = 4; controls.maxDistance = 40;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.8); key.position.set(6, 10, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.5); rim.position.set(-8, 4, -6); scene.add(rim);
addGrid(scene, { size: 40, divisions: 20, y: -5 });
addSun(scene, { scale: 30, position: [0, 12, -60] });

// ---------- real spherical harmonics (closed form, n=θ polar, φ azimuth) ----------
// returns signed value; we use |·| for radius and sign for color.
const SQRT = Math.sqrt;
const ORBITALS = [
  ["1s",     (t, p) => 0.2821],
  ["2p_z",   (t, p) => 0.4886 * Math.cos(t)],
  ["2p_x",   (t, p) => 0.4886 * Math.sin(t) * Math.cos(p)],
  ["2p_y",   (t, p) => 0.4886 * Math.sin(t) * Math.sin(p)],
  ["3d_z²",  (t, p) => 0.3153 * (3 * Math.cos(t) ** 2 - 1)],
  ["3d_xz",  (t, p) => 1.0925 * Math.sin(t) * Math.cos(t) * Math.cos(p)],
  ["3d_x²-y²", (t, p) => 0.5463 * Math.sin(t) ** 2 * Math.cos(2 * p)],
  ["3d_xy",  (t, p) => 0.5463 * Math.sin(t) ** 2 * Math.sin(2 * p)],
  ["4f_z³",  (t, p) => 0.3732 * (5 * Math.cos(t) ** 3 - 3 * Math.cos(t))],
  ["4f_xz²", (t, p) => 0.4570 * Math.sin(t) * (5 * Math.cos(t) ** 2 - 1) * Math.cos(p)],
  ["4f_z(x²-y²)", (t, p) => 1.4453 * Math.sin(t) ** 2 * Math.cos(t) * Math.cos(2 * p)],
  ["4f_x³",  (t, p) => 0.5901 * Math.sin(t) ** 3 * Math.cos(3 * p)],
];
let idx = 1; // start on a p orbital (more interesting than the sphere)

// ---------- build the lobe surface ----------
const ROWS = 96, COLS = 160;     // (θ, φ)
const SCALE = 7.0;
const vcount = (ROWS + 1) * (COLS + 1);
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(vcount * 3);
const col = new Float32Array(vcount * 3);
const indices = [];
const stride = COLS + 1;
for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
  const a = r * stride + c; indices.push(a, a + stride, a + 1, a + 1, a + stride, a + stride + 1);
}
geo.setIndex(indices);
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.45, metalness: 0.15, side: THREE.DoubleSide, flatShading: false }));
scene.add(mesh);
const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 }));
scene.add(wire);

const POS_C = [0.16, 0.82, 0.96];  // cyan  (+)
const NEG_C = [1.00, 0.18, 0.60];  // magenta (−)
function build() {
  const Y = ORBITALS[idx][1];
  let p = 0;
  for (let r = 0; r <= ROWS; r++) {
    const theta = (r / ROWS) * Math.PI;
    const st = Math.sin(theta), ct = Math.cos(theta);
    for (let c = 0; c <= COLS; c++) {
      const phi = (c / COLS) * Math.PI * 2;
      const v = Y(theta, phi);
      const rad = Math.abs(v) * SCALE + 0.02;
      // direction (physics convention: z up)
      const x = rad * st * Math.cos(phi);
      const y = rad * ct;             // z → world-up (y)
      const z = rad * st * Math.sin(phi);
      pos[p] = x; pos[p + 1] = y; pos[p + 2] = z;
      const cc = v >= 0 ? POS_C : NEG_C;
      const shade = 0.55 + 0.45 * Math.min(Math.abs(v) * SCALE / 4, 1);
      col[p] = cc[0] * shade; col[p + 1] = cc[1] * shade; col[p + 2] = cc[2] * shade;
      p += 3;
    }
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingSphere();
  if (nameEl) nameEl.textContent = ORBITALS[idx][0];
  chips && chips.forEach((b, k) => b.classList.toggle("active", k === idx));
}

// ---------- panel ----------
const wrap = document.getElementById("orbs");
const nameEl = document.getElementById("orbname");
const chips = ORBITALS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === idx ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { idx = i; build(); });
  wrap.appendChild(b);
  return b;
});
const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => { camera.position.copy(home); controls.target.set(0, 0, 0); });
setVariantCycler((d) => { idx = (idx + d + ORBITALS.length) % ORBITALS.length; build(); return ORBITALS[idx][0]; });

// ---------- boot ----------
build();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); controls.update(); renderer.render(scene, camera); });
