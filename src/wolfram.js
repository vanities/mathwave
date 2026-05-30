// wolfram.js — elementary cellular automata (Wolfram). A 1D row of cells, each
// 0/1, updates by a RULE: the new state of a cell is a function of its own value
// and its two neighbors — 3 bits → 8 possible neighborhoods → one output bit each
// → 2⁸ = 256 rules, numbered by the 8-bit output (Wolfram code). We grow the
// history downward into a 3D ribbon so you see the whole spacetime pattern:
//   rule 90 = Sierpiński triangle, rule 30 = chaos (used as a PRNG), rule 110 =
//   Turing-complete (proved universal), rule 184 = traffic flow. ↑↓ cycles rules.
//
// Ref: Wolfram, "A New Kind of Science"; rule 110 universality (Cook 2004).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.01);

const W = 121;          // cells per row (odd → centered single seed)
const ROWS = 121;       // generations of history kept
const CELL = 0.34;

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 4, 34);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.35;
controls.minDistance = 12; controls.maxDistance = 90;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xff2e97, 0.9); key.position.set(10, 18, 12); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.7); rim.position.set(-12, 8, -8); scene.add(rim);
addGrid(scene, { size: 60, divisions: 30, y: -ROWS * CELL * 0.5 - 2 });
addSun(scene, { scale: 40, position: [0, ROWS * CELL * 0.4, -70] });

// ---------- automaton ----------
let RULE = 110;
let ruleBits = [];
function setRuleBits(r) { ruleBits = []; for (let i = 0; i < 8; i++) ruleBits[i] = (r >> i) & 1; }
function nextRow(row) {
  const out = new Uint8Array(W);
  for (let i = 0; i < W; i++) {
    const l = row[(i - 1 + W) % W], c = row[i], rr = row[(i + 1) % W];
    out[i] = ruleBits[(l << 2) | (c << 1) | rr];
  }
  return out;
}

// history grid
let history = [];
function reseed(kind) {
  history = [];
  let row = new Uint8Array(W);
  if (kind === "random") { for (let i = 0; i < W; i++) row[i] = Math.random() < 0.5 ? 1 : 0; }
  else row[(W - 1) >> 1] = 1;     // single centered seed
  history.push(row);
  for (let r = 1; r < ROWS; r++) history.push(nextRow(history[r - 1]));
}

// ---------- instanced cubes ----------
const geo = new THREE.BoxGeometry(CELL * 0.9, CELL * 0.9, CELL * 0.9);
const mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.2 }), W * ROWS);
mesh.frustumCulled = false; scene.add(mesh);
const dummy = new THREE.Object3D(); const color = new THREE.Color();
const halfW = (W - 1) / 2, halfR = (ROWS - 1) / 2;
function draw() {
  let n = 0;
  for (let r = 0; r < ROWS; r++) {
    for (let i = 0; i < W; i++) {
      if (!history[r][i]) continue;
      dummy.position.set((i - halfW) * CELL, (halfR - r) * CELL, 0); dummy.updateMatrix();
      mesh.setMatrixAt(n, dummy.matrix);
      const c = ramp(0.2 + 0.8 * (1 - r / ROWS));   // newer rows brighter
      color.setRGB(c[0], c[1], c[2]); mesh.setColorAt(n, color);
      n++;
    }
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

function load(rule, seed = "single") { RULE = rule; setRuleBits(rule); reseed(seed); draw(); if (ruleEl) ruleEl.textContent = "rule " + rule; chips.forEach((b) => b.classList.toggle("active", +b.dataset.r === rule)); }

// ---------- panel ----------
const RULES = [[30,"chaos"],[90,"Sierpiński"],[110,"universal"],[184,"traffic"],[150,"XOR"],[54,"class 4"]];
const wrap = document.getElementById("rules");
const ruleEl = document.getElementById("rulename");
const chips = RULES.map(([r, label]) => {
  const b = document.createElement("button"); b.className = "chip"; b.dataset.r = r; b.textContent = r + " · " + label;
  b.addEventListener("click", () => load(r, seedMode)); wrap.appendChild(b); return b;
});
let seedMode = "single";
const seedBtn = document.getElementById("seed");
seedBtn.addEventListener("click", () => { seedMode = seedMode === "single" ? "random" : "single"; seedBtn.textContent = "seed: " + seedMode; load(RULE, seedMode); });
setVariantCycler((d) => { const i = (RULES.findIndex(x => x[0] === RULE) + d + RULES.length) % RULES.length; load(RULES[i][0], seedMode); return "rule " + RULES[i][0]; });

// ---------- boot ----------
load(110, "single");
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); controls.update(); renderer.render(scene, camera); });
