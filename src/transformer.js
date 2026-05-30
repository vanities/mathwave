// transformer.js — a REAL toy transformer forward pass, rendered as a HUGE 3D
// monument you fly around. Every number comes from actual matrix math:
//
//   x   = embed(token) + positional_encoding(pos)        (sinusoidal PE)
//   Qh  = x·Wq_h   Kh = x·Wk_h   Vh = x·Wv_h             (per head)
//   S   = Qh·Khᵀ / √dk    (+ causal mask: j > i → −∞)
//   A   = softmax(S)                                      (rows sum to 1)
//   z   = concat_h(A·Vh)·Wo ;  x ← x + z                 (residual)
//   x  ← x + W2·relu(W1·x)                                (FFN + residual)
//   repeat × LAYERS ;  logits = x·Wuᵀ  (logit lens at every layer)
//
// 3D LAYOUT (this is the part that's now actually 3D, not a flat diagram):
//   • X = token position; Y = LAYER (the residual stream flows UP the stack);
//     so every token is a vertical COLUMN and the model is a tall lattice.
//   • Attention arcs fan out in DEPTH (+Z), one z-plane per head, colored by
//     head, thickness/opacity ∝ weight — a 3D web in front of each layer.
//   • The T×T attention map is a real 3D BAR CHART (bar height = weight), so the
//     causal lower-triangle is a literal 3D staircase.
//   • Residual-stream cloud (PCA→3D) with the live ABLITERATION toggle.
//   A glowing compute-plane rises through the layers = the forward pass.
//
// Refs: Vaswani et al. 2017; logit lens (nostalgebraist); abliteration
//   (arditi et al. / mlabonne) — "refusal ≈ one residual-stream direction".

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.0045);

// ---------- model dims (toy but real) ----------
const TOKENS = ["the", "cat", "sat", "on", "the", "mat", "and", "then", "it", "purrs"];
const T = TOKENS.length;          // 10 tokens
const VOCAB = TOKENS;
const Dm = 16;                    // model dim
let HEADS = 4;
const LAYERS = 6;                 // a taller stack
let DK = Dm / HEADS;
const HEAD_COLORS = [0xff2e97, 0x2be4ff, 0x62ffb3, 0xb06bff, 0xffd166, 0xff7a5a];

// ---------- HUGE 3D layout ----------
const COL_W = 5.0;                // token spacing (x)
const LAYER_H = 8.0;             // layer spacing (y) — tall
const BASE_Y = 3.0;
const xOf = (i) => (i - (T - 1) / 2) * COL_W;
const yOfLayer = (L) => BASE_Y + L * LAYER_H;
const TOP_Y = yOfLayer(LAYERS - 1);
const MIDY = (BASE_Y + TOP_Y) / 2;

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, MIDY, 95);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.45;
controls.minDistance = 25; controls.maxDistance = 320;
controls.target.set(0, MIDY, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.95));
const key = new THREE.DirectionalLight(0xfff1dd, 0.9); key.position.set(20, 40, 30); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.6); rim.position.set(-30, 10, -20); scene.add(rim);
addGrid(scene, { size: 160, divisions: 32, y: -1 });
addSun(scene, { scale: 60, position: [0, MIDY, -180] });

// ---------- deterministic RNG ----------
let seedSalt = 7;
function rng(seed) { let s = (seed * 9301 + 49297) % 233280 || 1; return () => (s = (s * 9301 + 49297) % 233280) / 233280; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }
function mat(rows, cols, r, scale = 1) { return Array.from({ length: rows }, () => Array.from({ length: cols }, () => gauss(r) * scale)); }

// ---------- linear algebra ----------
const matmul = (A, B) => {
  const n = A.length, k = B.length, m = B[0].length, out = [];
  for (let i = 0; i < n; i++) { const row = new Array(m).fill(0); for (let p = 0; p < k; p++) { const a = A[i][p]; const Bp = B[p]; for (let j = 0; j < m; j++) row[j] += a * Bp[j]; } out.push(row); }
  return out;
};
const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const relu = (A) => A.map((r) => r.map((v) => Math.max(0, v)));
const concatCols = (parts) => parts[0].map((_, i) => parts.flatMap((P) => P[i]));

// ---------- weights ----------
let WE, PE, WQ, WK, WV, WO, W1, W2, WU;
function sinusoidalPE() {
  const pe = [];
  for (let pos = 0; pos < T; pos++) {
    const row = [];
    for (let i = 0; i < Dm; i++) { const k = Math.floor(i / 2), denom = Math.pow(10000, (2 * k) / Dm); row.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom)); }
    pe.push(row);
  }
  return pe;
}
function initWeights() {
  DK = Dm / HEADS;
  const r = rng(1234 + seedSalt);
  WE = mat(T, Dm, r, 1.0); PE = sinusoidalPE();
  WQ = []; WK = []; WV = [];
  for (let h = 0; h < HEADS; h++) { WQ.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); WK.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); WV.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); }
  WO = mat(Dm, Dm, r, 1 / Math.sqrt(Dm));
  W1 = mat(Dm, Dm * 2, r, 1 / Math.sqrt(Dm));
  W2 = mat(Dm * 2, Dm, r, 1 / Math.sqrt(Dm * 2));
  WU = mat(T, Dm, r, 1 / Math.sqrt(Dm));
}
function softmaxRowCausal(s, i) {
  let mx = -1e9; for (let j = 0; j <= i; j++) mx = Math.max(mx, s[j]);
  let sum = 0; const out = new Array(s.length).fill(0);
  for (let j = 0; j <= i; j++) { out[j] = Math.exp(s[j] - mx); sum += out[j]; }
  for (let j = 0; j <= i; j++) out[j] /= sum;
  return out;
}

let attnByLayer, hiddenByLayer, refusal;
function forward() {
  attnByLayer = []; hiddenByLayer = [];
  let x = addM(WE, PE);
  for (let L = 0; L < LAYERS; L++) {
    const heads = [], headOuts = [];
    for (let h = 0; h < HEADS; h++) {
      const Q = matmul(x, WQ[h]), K = matmul(x, WK[h]), V = matmul(x, WV[h]);
      const A = [];
      for (let i = 0; i < T; i++) {
        const s = new Array(T).fill(-1e9);
        for (let j = 0; j <= i; j++) { let d = 0; for (let p = 0; p < DK; p++) d += Q[i][p] * K[j][p]; s[j] = d / Math.sqrt(DK); }
        A.push(softmaxRowCausal(s, i));
      }
      heads.push(A);
      const ctx = A.map((arow) => { const o = new Array(DK).fill(0); for (let j = 0; j < T; j++) for (let p = 0; p < DK; p++) o[p] += arow[j] * V[j][p]; return o; });
      headOuts.push(ctx);
    }
    const concat = concatCols(headOuts);
    x = addM(x, matmul(concat, WO));
    x = addM(x, matmul(relu(matmul(x, W1)), W2));
    attnByLayer.push(heads);
    hiddenByLayer.push(x.map((r) => r.slice()));
  }
  const mean = new Array(Dm).fill(0);
  for (const row of hiddenByLayer[LAYERS - 1]) for (let d = 0; d < Dm; d++) mean[d] += row[d] / T;
  const nrm = Math.hypot(...mean) || 1; refusal = mean.map((v) => v / nrm);
}
function predict(hiddenRow) {
  let best = 0, bestv = -1e9;
  for (let t = 0; t < VOCAB.length; t++) { let d = 0; for (let p = 0; p < Dm; p++) d += hiddenRow[p] * WU[t][p]; if (d > bestv) { bestv = d; best = t; } }
  return VOCAB[best];
}

// ============================================================
// 3D SCENE
// ============================================================
const stack = new THREE.Group(); scene.add(stack);     // the lattice (nodes + columns)
const arcGroup = new THREE.Group(); scene.add(arcGroup); // attention webs per layer
const clearGroup = (g) => { while (g.children.length) { const c = g.children.pop(); c.geometry && c.geometry.dispose(); c.material && c.material.dispose && c.material.dispose(); g.remove(c); } };
const at = (sp, p, s) => { sp.position.copy(p); if (s) sp.scale.copy(s); return sp; };

function makeLabel(text, color = "#f6e9ff", glow = "#2be4ff") {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 44px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 12; g.fillText(text, 128, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sp.scale.set(3.4, 0.85, 1); return sp;
}

// node spheres for every (token, layer) — an instanced lattice
let lattice, latColor;
function buildStack() {
  clearGroup(stack);
  const count = T * LAYERS;
  lattice = new THREE.InstancedMesh(new THREE.SphereGeometry(0.42, 18, 14),
    new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.2, emissive: 0x10042a }), count);
  lattice.frustumCulled = false;
  const d = new THREE.Object3D(); const col = new THREE.Color();
  for (let L = 0; L < LAYERS; L++) for (let i = 0; i < T; i++) {
    d.position.set(xOf(i), yOfLayer(L), 0); d.updateMatrix();
    lattice.setMatrixAt(L * T + i, d.matrix);
    const c = ramp(0.1 + 0.8 * (L / (LAYERS - 1))); col.setRGB(c[0], c[1], c[2]);
    lattice.setColorAt(L * T + i, col);
  }
  stack.add(lattice);
  // residual columns: one vertical line per token, base→top
  for (let i = 0; i < T; i++) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xOf(i), BASE_Y, 0), new THREE.Vector3(xOf(i), TOP_Y, 0)]);
    stack.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x4a1f7a, transparent: true, opacity: 0.4 })));
    // token label at the base
    stack.add(at(makeLabel(TOKENS[i], "#c0a3e8", "#b06bff"), new THREE.Vector3(xOf(i), BASE_Y - 1.6, 0)));
  }
  // "layer" tags up the left side
  for (let L = 0; L < LAYERS; L++) stack.add(at(makeLabel("L" + L, "#8366b8", "#4a1f7a"), new THREE.Vector3(xOf(0) - 3.6, yOfLayer(L), 0), new THREE.Vector3(2.0, 0.9, 1)));
}

// attention arcs for layer L: 3D fan, each head on its own depth plane
function buildArcs(L) {
  const yb = yOfLayer(L);
  for (let h = 0; h < HEADS; h++) {
    const A = attnByLayer[L][h], color = HEAD_COLORS[h % HEAD_COLORS.length];
    const zPlane = (h - (HEADS - 1) / 2) * 2.4;     // heads separate in DEPTH → real 3D
    for (let i = 0; i < T; i++) for (let j = 0; j < i; j++) {
      const w = A[i][j]; if (w < 0.12) continue;
      const x0 = xOf(j), x1 = xOf(i);
      const mx = (x0 + x1) / 2;
      const bow = 2.0 + w * 5.0;                     // stronger weight bows further out
      const my = yb + bow * 0.5, mz = zPlane + bow;  // arc bulges up AND toward +z
      const pts = [];
      for (let s = 0; s <= 20; s++) {
        const t = s / 20, u = 1 - t;
        pts.push(new THREE.Vector3(
          u*u*x0 + 2*u*t*mx + t*t*x1,
          u*u*yb + 2*u*t*my + t*t*yb,
          u*u*zPlane + 2*u*t*mz + t*t*zPlane,
        ));
      }
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(0.12 + w * 1.4, 0.95), blending: THREE.AdditiveBlending, depthWrite: false }));
      line.userData.layer = L; line.userData.baseOp = Math.min(0.12 + w * 1.4, 0.95);
      arcGroup.add(line);
    }
  }
}

// ---------- 3D BAR-CHART heatmap (right of the stack) ----------
const HM = new THREE.Group(); HM.position.set(xOf(T - 1) + 14, MIDY - 6, 0); scene.add(HM);
const HMW = 1.3;                  // cell footprint
let hmMesh;
function buildHeatmap() {
  clearGroup(HM);
  // base plate
  const plate = new THREE.Mesh(new THREE.BoxGeometry(T * HMW + 1, 0.3, T * HMW + 1), new THREE.MeshStandardMaterial({ color: 0x14062b, roughness: 0.8, transparent: true, opacity: 0.6 }));
  plate.position.y = -0.15; HM.add(plate);
  hmMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(HMW * 0.8, 1, HMW * 0.8).translate(0, 0.5, 0),
    new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.2 }), T * T);
  hmMesh.frustumCulled = false; HM.add(hmMesh);
  HM.add(at(makeLabel("attention Aᵢⱼ · causal", "#2be4ff", "#2be4ff"), new THREE.Vector3(0, T * HMW * 0.5 + 4, 0), new THREE.Vector3(6, 1.3, 1)));
}
const hmColor = new THREE.Color(); const hmDummy = new THREE.Object3D();
function paintHeatmap(L, head) {
  const A = attnByLayer[L][head];
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    const masked = j > i;
    const w = masked ? 0 : A[i][j];
    const hgt = masked ? 0.04 : 0.1 + w * 9.0;       // bar height = attention weight (3D!)
    hmDummy.position.set((j - (T - 1) / 2) * HMW, 0, (i - (T - 1) / 2) * HMW);
    hmDummy.scale.set(1, hgt, 1); hmDummy.updateMatrix();
    hmMesh.setMatrixAt(i * T + j, hmDummy.matrix);
    if (masked) hmColor.setRGB(0.05, 0.03, 0.1);
    else { const c = ramp(0.15 + 0.85 * w); hmColor.setRGB(c[0], c[1], c[2]); }
    hmMesh.setColorAt(i * T + j, hmColor);
  }
  hmMesh.instanceMatrix.needsUpdate = true;
  if (hmMesh.instanceColor) hmMesh.instanceColor.needsUpdate = true;
}

// ---------- residual cloud (left of the stack) + abliteration ----------
const RS = new THREE.Group(); RS.position.set(xOf(0) - 16, MIDY, 0); scene.add(RS);
let dots = [], refLine = null, basis, RSCALE = 1, rb3 = [0, 0, 1];
let abliterate = false;
function topBasis(rows) {
  const varc = new Array(Dm).fill(0), mean = new Array(Dm).fill(0);
  for (const r of rows) for (let d = 0; d < Dm; d++) mean[d] += r[d] / rows.length;
  for (const r of rows) for (let d = 0; d < Dm; d++) varc[d] += (r[d] - mean[d]) ** 2;
  const idx = varc.map((v, d) => [v, d]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => x[1]);
  return { idx, mean };
}
function buildResidual() {
  clearGroup(RS); dots = [];
  for (let i = 0; i < T; i++) { const m = new THREE.Mesh(new THREE.SphereGeometry(0.5, 16, 12), new THREE.MeshStandardMaterial({ color: 0x62ffb3, emissive: 0x163c2c, roughness: 0.4 })); RS.add(m); dots.push(m); }
  RS.add(at(makeLabel("residual stream", "#62ffb3", "#62ffb3"), new THREE.Vector3(0, 7, 0), new THREE.Vector3(6, 1.4, 1)));
  basis = topBasis(hiddenByLayer[LAYERS - 1]);
  const { idx, mean } = basis;
  let spread = 1e-6;
  for (const r of hiddenByLayer[LAYERS - 1]) for (let k = 0; k < 3; k++) spread = Math.max(spread, Math.abs(r[idx[k]] - mean[idx[k]]));
  RSCALE = 5.0 / spread;
  rb3 = [refusal[idx[0]], refusal[idx[1]], refusal[idx[2]]];
  const rn = Math.hypot(rb3[0], rb3[1], rb3[2]) || 1; rb3 = rb3.map((v) => v / rn);
  const a = new THREE.Vector3(rb3[0], rb3[1], rb3[2]).multiplyScalar(6.0);
  const lg = new THREE.BufferGeometry().setFromPoints([a.clone().negate(), a]);
  refLine = new THREE.Line(lg, new THREE.LineDashedMaterial({ color: 0xff2e97, dashSize: 0.5, gapSize: 0.35, transparent: true, opacity: 0.85 })); refLine.computeLineDistances(); RS.add(refLine);
}
function placeResidual(L) {
  const rows = hiddenByLayer[L], { idx, mean } = basis;
  for (let i = 0; i < T; i++) {
    let p = [(rows[i][idx[0]] - mean[idx[0]]) * RSCALE, (rows[i][idx[1]] - mean[idx[1]]) * RSCALE, (rows[i][idx[2]] - mean[idx[2]]) * RSCALE];
    if (abliterate) { const dt = p[0]*rb3[0]+p[1]*rb3[1]+p[2]*rb3[2]; p = [p[0]-dt*rb3[0], p[1]-dt*rb3[1], p[2]-dt*rb3[2]]; }
    dots[i].position.lerp(new THREE.Vector3(p[0], p[1], p[2]), 0.15);
    dots[i].material.color.setHex(abliterate ? 0xff2e97 : 0x62ffb3);
  }
  if (refLine) refLine.material.opacity = abliterate ? 0.95 : 0.4;
}

// ---------- a glowing compute-plane that rises through the stack ----------
const plane = new THREE.Mesh(
  new THREE.PlaneGeometry(T * COL_W + 6, 16).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x2be4ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false })
);
scene.add(plane);

// ---------- logit-lens readout (top of the stack) ----------
let lensSprite = null;
function showLens(L) {
  if (lensSprite) { scene.remove(lensSprite); lensSprite.material.map.dispose(); lensSprite.material.dispose(); }
  lensSprite = makeLabel("predict → " + predict(hiddenByLayer[L][T - 1]), "#ffd166", "#ff9f5a");
  lensSprite.scale.set(8, 2, 1); lensSprite.position.set(0, TOP_Y + 5, 0);
  scene.add(lensSprite);
}

// ---------- animation state (declared before rebuild(), which runs at boot) ----------
let curLayer = 0, layerT = 0, planeY = BASE_Y;

// ---------- (re)build everything ----------
function rebuild() { initWeights(); forward(); buildStack(); buildHeatmap(); buildResidual(); clearGroup(arcGroup); curLayer = 0; buildArcs(0); paintHeatmap(0, activeHead); showLens(0); }

// ---------- panel ----------
let activeHead = 0;
const ablBtn = document.getElementById("abliterate");
ablBtn.addEventListener("click", () => { abliterate = !abliterate; ablBtn.classList.toggle("active", abliterate); ablBtn.textContent = abliterate ? "abliterated ●" : "abliterate"; document.getElementById("mode").textContent = abliterate ? "refusal removed" : "intact"; });
let layerSpeed = 1; bindRange("speed", (v) => { layerSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("heads", (v) => { HEADS = Math.round(v); activeHead = Math.min(activeHead, HEADS - 1); rebuild(); }, (v) => `${Math.round(v)}`);
const headBtn = document.getElementById("head");
headBtn.addEventListener("click", () => { activeHead = (activeHead + 1) % HEADS; headBtn.textContent = "head " + (activeHead + 1); paintHeatmap(curLayer, activeHead); });
document.getElementById("resample").addEventListener("click", () => { seedSalt += 13; rebuild(); });
setVariantCycler((d) => { ablBtn.click(); return abliterate ? "abliterated" : "intact"; });

// ---------- boot ----------
rebuild();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const layerEl = document.getElementById("layer");

window.__diag = () => JSON.stringify({ T, HEADS, LAYERS, DK, curLayer, arcs: arcGroup.children.length, attnRowSum: attnByLayer[0][0][T-1].reduce((a,b)=>a+b,0).toFixed(3), cloudR: Math.max(...dots.map(d => d.position.length())).toFixed(2) });

loop((dt) => {
  meter(dt);
  layerT += dt * layerSpeed;
  if (layerT > 1.1) {                       // ascend one layer (the forward pass)
    layerT = 0; curLayer = (curLayer + 1) % LAYERS;
    if (curLayer === 0) { clearGroup(arcGroup); }   // looped → start a fresh ascent
    buildArcs(curLayer); paintHeatmap(curLayer, activeHead); showLens(curLayer);
    layerEl.textContent = `${curLayer + 1}/${LAYERS}`;
  }
  // the compute-plane glides up to the active layer
  planeY += (yOfLayer(curLayer) - planeY) * Math.min(1, dt * 4);
  plane.position.y = planeY;
  // pulse the ACTIVE layer's arcs brighter; older revealed layers stay dim
  const k = 0.6 + 0.4 * Math.sin(performance.now() * 0.005);
  arcGroup.children.forEach((a) => { const active = a.userData.layer === curLayer; a.material.opacity = a.userData.baseOp * (active ? (0.7 + 0.5 * k) : 0.32); });
  placeResidual(curLayer);
  HM.lookAt(camera.position.x, HM.position.y, camera.position.z);  // bars face you, stay upright
  RS.quaternion.copy(camera.quaternion);
  controls.update();
  renderer.render(scene, camera);
});
