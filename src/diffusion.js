// diffusion.js — how image generators (DDPM-style) actually work, in 3D.
// FORWARD process: start from a clean shape (points arranged as a target) and
// add Gaussian noise step by step until it's pure static — q(xₜ|x₀)=N(√ᾱₜ x₀,(1−ᾱₜ)I).
// REVERSE process: a generator learns to DENOISE, walking the cloud back from
// chaos to the shape. Here the reverse uses the known target (a teaching demo),
// pulling each point toward its destination as the noise schedule βₜ cools —
// exactly the noise→image trajectory, made spatial. ↑↓ changes the target shape.
//
// Ref: Ho et al. 2020, "Denoising Diffusion Probabilistic Models".

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.014);

const camera = new THREE.PerspectiveCamera(54, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 4, 22);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 8; controls.maxDistance = 70;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
addGrid(scene, { size: 50, divisions: 25, y: -10 });
addSun(scene, { scale: 40, position: [0, 12, -70] });

// ---------- the point cloud ----------
const N = 9000;
const target = new Float32Array(N * 3);   // x₀ — the clean shape
const cur = new Float32Array(N * 3);       // xₜ — current noised state
const noise = new Float32Array(N * 3);     // fixed per-point noise direction
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(N * 3);
const col = new Float32Array(N * 3);
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
geo.setAttribute("color", new THREE.BufferAttribute(col, 3).setUsage(THREE.DynamicDrawUsage));
const points = new THREE.Points(geo, new THREE.PointsMaterial({ vertexColors: true, size: 0.12, sizeAttenuation: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false }));
points.frustumCulled = false;
scene.add(points);

function gauss() { return Math.sqrt(-2 * Math.log(Math.random() + 1e-9)) * Math.cos(2 * Math.PI * Math.random()); }

// ---------- target shapes ----------
const SHAPES = ["sphere", "torus", "galaxy", "heart", "cube"];
function setTarget(kind) {
  for (let i = 0; i < N; i++) {
    let x, y, z;
    if (kind === "sphere") {
      const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, r = Math.sqrt(1 - u * u);
      x = r * Math.cos(a); y = u; z = r * Math.sin(a); const R = 6; x *= R; y *= R; z *= R;
    } else if (kind === "torus") {
      const a = Math.random() * Math.PI * 2, b = Math.random() * Math.PI * 2, R = 5, rr = 2;
      x = (R + rr * Math.cos(b)) * Math.cos(a); y = rr * Math.sin(b); z = (R + rr * Math.cos(b)) * Math.sin(a);
    } else if (kind === "galaxy") {
      const arm = (i % 3) * (Math.PI * 2 / 3), t = Math.random() * 4, r = t * 1.6;
      const a = arm + t * 1.1 + gauss() * 0.12; x = Math.cos(a) * r; z = Math.sin(a) * r; y = gauss() * (0.6 + r * 0.04);
    } else if (kind === "heart") {
      const t = Math.random() * Math.PI * 2, s = 0.42;
      x = 16 * Math.sin(t) ** 3 * s; y = (13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t)) * s; z = gauss() * 0.8;
    } else { // cube shell
      const f = (Math.random() * 6) | 0, u = Math.random() * 2 - 1, v = Math.random() * 2 - 1, S = 5;
      const c = [[1,u,v],[-1,u,v],[u,1,v],[u,-1,v],[u,v,1],[u,v,-1]][f]; x = c[0]*S; y = c[1]*S; z = c[2]*S;
    }
    target[i*3] = x; target[i*3+1] = y; target[i*3+2] = z;
    noise[i*3] = gauss() * 9; noise[i*3+1] = gauss() * 9; noise[i*3+2] = gauss() * 9;
  }
}

// ---------- diffusion schedule ----------
// t in [0,1]: 0 = clean (x₀), 1 = pure noise (x_T). ᾱ(t) cosine schedule.
const alphaBar = (t) => Math.cos((t * 0.5 + 0.008) / 1.008 * Math.PI / 2) ** 2;
let t = 1;            // start as pure noise, then denoise
let dir = -1;         // -1 = reverse (denoise), +1 = forward (add noise)
let speed = 1;
let auto = true;

function render() {
  const ab = Math.max(0, Math.min(1, alphaBar(t)));
  const sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);   // xₜ = √ᾱ·x₀ + √(1−ᾱ)·ε
  for (let i = 0; i < N; i++) {
    const x = sa * target[i*3]   + sn * noise[i*3];
    const y = sa * target[i*3+1] + sn * noise[i*3+1];
    const z = sa * target[i*3+2] + sn * noise[i*3+2];
    pos[i*3]=x; pos[i*3+1]=y; pos[i*3+2]=z;
    // color: noisy = magenta/cyan static; clean = the neon ramp by height
    const cleanC = ramp((target[i*3+1] + 6) / 12);
    const noiseC = ramp((i % 100) / 100);
    const k = 1 - t;  // how "denoised"
    col[i*3]   = noiseC[0]*(1-k) + cleanC[0]*k;
    col[i*3+1] = noiseC[1]*(1-k) + cleanC[1]*k;
    col[i*3+2] = noiseC[2]*(1-k) + cleanC[2]*k;
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
}

// ---------- panel ----------
const wrap = document.getElementById("shapes");
let shapeIdx = 0;
const chips = SHAPES.map((label, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { shapeIdx = i; setTarget(label); chips.forEach((c, k) => c.classList.toggle("active", k === i)); });
  wrap.appendChild(b);
  return b;
});
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
const autoBtn = document.getElementById("auto");
autoBtn.classList.toggle("active", auto);
autoBtn.addEventListener("click", () => { auto = !auto; autoBtn.classList.toggle("active", auto); });
const scrub = document.getElementById("scrub");
scrub.addEventListener("input", () => { auto = false; autoBtn.classList.remove("active"); t = 1 - parseFloat(scrub.value); });
setVariantCycler((d) => { shapeIdx = (shapeIdx + d + SHAPES.length) % SHAPES.length; setTarget(SHAPES[shapeIdx]); chips.forEach((c, k) => c.classList.toggle("active", k === shapeIdx)); return SHAPES[shapeIdx]; });

// ---------- boot ----------
setTarget("sphere");
render();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const stEl = document.getElementById("stage");

loop((dt) => {
  meter(dt);
  if (auto) {
    t += dir * dt * 0.18 * speed;
    if (t <= 0) { t = 0; dir = 1; }          // fully formed → start re-noising
    else if (t >= 1) { t = 1; dir = -1; setTarget(SHAPES[shapeIdx]); } // pure noise → denoise to shape
    scrub.value = 1 - t;
  }
  render();
  if (stEl) stEl.textContent = t > 0.92 ? "pure noise xₜ" : t < 0.08 ? "image x₀" : (dir < 0 ? "denoising ↓" : "noising ↑");
  controls.update();
  renderer.render(scene, camera);
});
