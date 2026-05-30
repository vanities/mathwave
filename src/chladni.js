// chladni.js — Chladni plates / cymatics. A plate driven at a resonant frequency
// forms STANDING WAVES; sand sprinkled on it flees the moving antinodes and
// collects along the NODAL LINES where the plate is still. For a square plate
// with free edges the mode shape is
//   s(x,y) = cos(nπx)cos(mπy) − cos(mπx)cos(nπy)
// and the sand settles where s(x,y) ≈ 0. We render the plate as a vibrating 3D
// surface (height = s) AND scatter thousands of "sand" grains that random-walk
// downhill in |s| until they pile onto the nodal curves — the real physics of
// how the figure forms. ↑↓ steps the (m,n) modes.
//
// Ref: Chladni's law; s(x,y)=cos(nπx)cos(mπy)−cos(mπx)cos(nπy) (Paul Bourke / Wikipedia).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.014);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 9, 11);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 6; controls.maxDistance = 40;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.85));
const key = new THREE.DirectionalLight(0xfff1dd, 1.0); key.position.set(6, 12, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.6); rim.position.set(-8, 6, -6); scene.add(rim);
addGrid(scene, { size: 30, divisions: 15, y: -3 });
addSun(scene, { scale: 28, position: [0, 9, -54] });

// ---------- the mode function ----------
let m = 3, n = 2, amp = 1.0;
const PLATE = 7;                  // world size of the plate
const s = (x, y) => Math.cos(n * Math.PI * x) * Math.cos(m * Math.PI * y) - Math.cos(m * Math.PI * x) * Math.cos(n * Math.PI * y);

// ---------- plate surface ----------
const N = 140;
const geo = new THREE.BufferGeometry();
const pos = new Float32Array(N * N * 3);
const col = new Float32Array(N * N * 3);
const idx = [];
for (let i = 0; i < N - 1; i++) for (let j = 0; j < N - 1; j++) { const a = i * N + j; idx.push(a, a + 1, a + N, a + 1, a + N + 1, a + N); }
geo.setIndex(idx);
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.DynamicDrawUsage));
geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
const plate = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, metalness: 0.1, side: THREE.DoubleSide }));
scene.add(plate);

function buildPlate(phase) {
  let p = 0;
  for (let i = 0; i < N; i++) {
    const u = i / (N - 1), x = (u - 0.5) * 2;            // x in [-1,1]
    for (let j = 0; j < N; j++) {
      const v = j / (N - 1), y = (v - 0.5) * 2;
      const val = s(x, y) * amp * phase;
      pos[p] = (u - 0.5) * PLATE; pos[p + 1] = val * 0.9; pos[p + 2] = (v - 0.5) * PLATE;
      // color: near a node (|s|→0) glow mint; antinodes magenta/cyan by sign
      const nodeness = 1 - Math.min(Math.abs(s(x, y)), 1);
      const base = s(x, y) >= 0 ? [1.0, 0.18, 0.6] : [0.16, 0.82, 0.96];
      col[p] = base[0] * (1 - nodeness) + 0.55 * nodeness;
      col[p + 1] = base[1] * (1 - nodeness) + 1.0 * nodeness;
      col[p + 2] = base[2] * (1 - nodeness) + 0.78 * nodeness;
      p += 3;
    }
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
  geo.computeVertexNormals();
}

// ---------- sand grains ----------
const GRAINS = 14000;
const gx = new Float32Array(GRAINS), gy = new Float32Array(GRAINS);
const sandGeo = new THREE.BufferGeometry();
const sandPos = new Float32Array(GRAINS * 3);
sandGeo.setAttribute("position", new THREE.BufferAttribute(sandPos, 3).setUsage(THREE.DynamicDrawUsage));
const sand = new THREE.Points(sandGeo, new THREE.PointsMaterial({ color: 0xfff3d8, size: 0.06, sizeAttenuation: true }));
sand.frustumCulled = false;
scene.add(sand);
function scatterSand() { for (let i = 0; i < GRAINS; i++) { gx[i] = Math.random() * 2 - 1; gy[i] = Math.random() * 2 - 1; } }

function stepSand(vibration) {
  for (let i = 0; i < GRAINS; i++) {
    const x = gx[i], y = gy[i];
    const here = Math.abs(s(x, y));
    // sample a few random nearby moves; hop toward lower |s| (sand leaves antinodes)
    const jump = 0.02 + vibration * 0.05;
    let bx = x, by = y, best = here;
    for (let k = 0; k < 4; k++) {
      const nx = x + (Math.random() * 2 - 1) * jump, ny = y + (Math.random() * 2 - 1) * jump;
      if (nx < -1 || nx > 1 || ny < -1 || ny > 1) continue;
      const val = Math.abs(s(nx, ny));
      if (val < best) { best = val; bx = nx; by = ny; }
    }
    // small thermal jitter scaled by how much the plate is shaking here
    gx[i] = bx + (Math.random() * 2 - 1) * here * vibration * 0.04;
    gy[i] = by + (Math.random() * 2 - 1) * here * vibration * 0.04;
    gx[i] = Math.max(-1, Math.min(1, gx[i])); gy[i] = Math.max(-1, Math.min(1, gy[i]));
    sandPos[i * 3] = (gx[i] * 0.5) * PLATE; sandPos[i * 3 + 1] = 0.05; sandPos[i * 3 + 2] = (gy[i] * 0.5) * PLATE;
  }
  sandGeo.attributes.position.needsUpdate = true;
}

// ---------- panel ----------
const MODES = [[1,2],[2,3],[3,2],[3,4],[4,5],[5,3],[6,7],[7,4]];
const wrap = document.getElementById("modes");
let mi = 1;
const chips = MODES.map(([mm, nn], i) => {
  const b = document.createElement("button"); b.className = "chip" + (i === mi ? " active" : ""); b.textContent = `${mm},${nn}`;
  b.addEventListener("click", () => { mi = i; m = mm; n = nn; scatterSand(); chips.forEach((c,k)=>c.classList.toggle("active",k===i)); modeEl.textContent = `m=${m} n=${n}`; });
  wrap.appendChild(b); return b;
});
const modeEl = document.getElementById("mode");
bindRange("amp", (v) => { amp = v; }, (v) => v.toFixed(2));
document.getElementById("shake").addEventListener("click", scatterSand);
setVariantCycler((d) => { mi = (mi + d + MODES.length) % MODES.length; [m, n] = MODES[mi]; scatterSand(); chips.forEach((c,k)=>c.classList.toggle("active",k===mi)); modeEl.textContent = `m=${m} n=${n}`; return `${m},${n}`; });

// ---------- boot ----------
m = MODES[mi][0]; n = MODES[mi][1]; modeEl.textContent = `m=${m} n=${n}`;
scatterSand();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

let t = 0;
loop((dt) => {
  meter(dt);
  t += dt;
  const phase = reducedMotion ? 1 : Math.sin(t * 8) * 0.5 + 0.5 + 0.2;  // plate vibrates
  buildPlate(reducedMotion ? 1 : Math.sin(t * 8));
  stepSand(0.6 + 0.4 * Math.abs(Math.sin(t * 8)));
  controls.update();
  renderer.render(scene, camera);
});
