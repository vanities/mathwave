// transformer.js — a REAL toy transformer forward pass, rendered.
// Not decorative: every number on screen comes from actual matrix math.
//
//   x   = embed(token) + positional_encoding(pos)        (sinusoidal PE)
//   Qh  = x·Wq_h   Kh = x·Wk_h   Vh = x·Wv_h             (per head)
//   S   = Qh·Khᵀ / √dk    (+ causal mask: j > i → −∞)
//   A   = softmax(S)                                      (rows sum to 1)
//   z   = concat_h(A·Vh)·Wo ;  x ← x + z                 (residual)
//   x  ← x + W2·relu(W1·x)                                (FFN + residual)
//   repeat × LAYERS ;  logits = x·Wuᵀ  (logit lens at every layer)
//
// Shown: token row, the canonical T×T attention HEATMAP for the active
// head/layer, attention arcs (causal, opacity = weight, color = head), a
// logit-lens readout of the forming prediction, and the residual-stream cloud
// with a live ABLITERATION toggle (project off the refusal direction).
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
scene.fog = new THREE.FogExp2(0x0a0118, 0.01);

const camera = new THREE.PerspectiveCamera(54, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 7, 30);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.32;
controls.minDistance = 12; controls.maxDistance = 90;
controls.target.set(0, 6, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.85); key.position.set(8, 16, 12); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.6); rim.position.set(-12, 8, -8); scene.add(rim);
addGrid(scene, { size: 60, divisions: 30, y: -1 });
addSun(scene, { scale: 26, position: [0, 16, -78] });

// ---------- model dims (toy but real) ----------
const TOKENS = ["the", "cat", "sat", "on", "the", "mat", "and", "purr"];
const T = TOKENS.length;
const VOCAB = TOKENS;            // tied: predict from the same small vocab
const Dm = 16;                   // model dim
let HEADS = 4;
const LAYERS = 4;
let DK = Dm / HEADS;             // per-head dim
const HEAD_COLORS = [0xff2e97, 0x2be4ff, 0x62ffb3, 0xb06bff, 0xffd166, 0xff7a5a];

// ---------- deterministic RNG + gaussian (seedable, no Math.random at top level) ----------
let seedSalt = 7;
function rng(seed) { let s = (seed * 9301 + 49297) % 233280 || 1; return () => (s = (s * 9301 + 49297) % 233280) / 233280; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }
function mat(rows, cols, r, scale = 1) { return Array.from({ length: rows }, () => Array.from({ length: cols }, () => gauss(r) * scale)); }

// ---------- linear algebra ----------
const matmul = (A, B) => { // [n×k]·[k×m] → [n×m]
  const n = A.length, k = B.length, m = B[0].length, out = [];
  for (let i = 0; i < n; i++) { const row = new Array(m).fill(0); for (let p = 0; p < k; p++) { const a = A[i][p]; const Bp = B[p]; for (let j = 0; j < m; j++) row[j] += a * Bp[j]; } out.push(row); }
  return out;
};
const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const relu = (A) => A.map((r) => r.map((v) => Math.max(0, v)));
const slice = (A, c0, c1) => A.map((r) => r.slice(c0, c1));
const concatCols = (parts) => parts[0].map((_, i) => parts.flatMap((P) => P[i]));

// ---------- weights ----------
let WE, PE, WQ, WK, WV, WO, W1, W2, WU;
function sinusoidalPE() {
  const pe = [];
  for (let pos = 0; pos < T; pos++) {
    const row = [];
    for (let i = 0; i < Dm; i++) {
      const k = Math.floor(i / 2), denom = Math.pow(10000, (2 * k) / Dm);
      row.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom));
    }
    pe.push(row);
  }
  return pe;
}
function initWeights() {
  DK = Dm / HEADS;
  const r = rng(1234 + seedSalt);
  WE = mat(T, Dm, r, 1.0);                 // token embeddings (one row per token slot)
  PE = sinusoidalPE();
  WQ = []; WK = []; WV = [];
  for (let h = 0; h < HEADS; h++) { WQ.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); WK.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); WV.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); }
  WO = mat(Dm, Dm, r, 1 / Math.sqrt(Dm));
  W1 = mat(Dm, Dm * 2, r, 1 / Math.sqrt(Dm));
  W2 = mat(Dm * 2, Dm, r, 1 / Math.sqrt(Dm * 2));
  WU = mat(T, Dm, r, 1 / Math.sqrt(Dm));   // unembed → logit per vocab token
}

function softmaxRowCausal(s, i) {
  let mx = -1e9; for (let j = 0; j <= i; j++) mx = Math.max(mx, s[j]);
  let sum = 0; const out = new Array(s.length).fill(0);
  for (let j = 0; j <= i; j++) { out[j] = Math.exp(s[j] - mx); sum += out[j]; }
  for (let j = 0; j <= i; j++) out[j] /= sum;
  return out;
}

// run the full forward pass; capture per-layer attention (per head) and hidden states
let attnByLayer;   // [layer][head] = T×T matrix
let hiddenByLayer; // [layer] = T×Dm   (residual stream after the layer)
let refusal;       // unit vector in Dm (the "refusal direction")
function forward() {
  attnByLayer = []; hiddenByLayer = [];
  let x = addM(WE, PE);                     // [T×Dm]
  for (let L = 0; L < LAYERS; L++) {
    const heads = [], headOuts = [];
    for (let h = 0; h < HEADS; h++) {
      const Q = matmul(x, WQ[h]), K = matmul(x, WK[h]), V = matmul(x, WV[h]); // [T×DK]
      const A = [];
      for (let i = 0; i < T; i++) {
        const s = new Array(T).fill(-1e9);
        for (let j = 0; j <= i; j++) { let d = 0; for (let p = 0; p < DK; p++) d += Q[i][p] * K[j][p]; s[j] = d / Math.sqrt(DK); }
        A.push(softmaxRowCausal(s, i));
      }
      heads.push(A);
      // context = A·V
      const ctx = A.map((arow) => { const o = new Array(DK).fill(0); for (let j = 0; j < T; j++) for (let p = 0; p < DK; p++) o[p] += arow[j] * V[j][p]; return o; });
      headOuts.push(ctx);
    }
    const concat = concatCols(headOuts);    // [T×Dm]
    x = addM(x, matmul(concat, WO));        // attention residual
    x = addM(x, matmul(relu(matmul(x, W1)), W2)); // FFN residual
    attnByLayer.push(heads);
    hiddenByLayer.push(x.map((r) => r.slice()));
  }
  // refusal direction = mean hidden state (a real direction in residual space)
  const mean = new Array(Dm).fill(0);
  for (const row of hiddenByLayer[LAYERS - 1]) for (let d = 0; d < Dm; d++) mean[d] += row[d] / T;
  const nrm = Math.hypot(...mean) || 1; refusal = mean.map((v) => v / nrm);
}

// logit lens: from a hidden state, which vocab token wins?
function predict(hiddenRow) {
  let best = 0, bestv = -1e9;
  for (let t = 0; t < VOCAB.length; t++) { let d = 0; for (let p = 0; p < Dm; p++) d += hiddenRow[p] * WU[t][p]; if (d > bestv) { bestv = d; best = t; } }
  return VOCAB[best];
}

// ---------- scene scaffolding ----------
const COL_W = 3.0, LAYER_H = 2.0;
const xOf = (i) => (i - (T - 1) / 2) * COL_W;
const yOfLayer = (L) => 1 + L * LAYER_H;

const nodeGroup = new THREE.Group(); scene.add(nodeGroup);
const arcGroup = new THREE.Group(); scene.add(arcGroup);
const clearGroup = (g) => { while (g.children.length) { const c = g.children.pop(); c.geometry && c.geometry.dispose(); c.material && c.material.dispose && c.material.dispose(); g.remove(c); } };

function makeLabel(text, color = "#f6e9ff", glow = "#2be4ff") {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 42px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 12; g.fillText(text, 128, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sp.scale.set(2.8, 0.7, 1); return sp;
}
// place a sprite: Object3D.position/scale are read-only refs, so copy (can't Object.assign)
const at = (sp, p, s) => { sp.position.copy(p); if (s) sp.scale.copy(s); return sp; };

// token nodes + columns through the layers
function buildScaffold() {
  clearGroup(nodeGroup);
  for (let i = 0; i < T; i++) {
    const node = new THREE.Mesh(new THREE.SphereGeometry(0.28, 18, 14), new THREE.MeshStandardMaterial({ color: 0x2be4ff, emissive: 0x0b3a44, roughness: 0.3 }));
    node.position.set(xOf(i), 1, 0); nodeGroup.add(node);
    nodeGroup.add(at(makeLabel(TOKENS[i]), new THREE.Vector3(xOf(i), 0.1, 0)));
    const col = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(xOf(i), 1, 0), new THREE.Vector3(xOf(i), yOfLayer(LAYERS - 1), 0)]);
    nodeGroup.add(new THREE.Line(col, new THREE.LineBasicMaterial({ color: 0x4a1f7a, transparent: true, opacity: 0.45 })));
  }
}

// arcs for one layer (all heads, causal, opacity = weight)
function buildArcs(L) {
  clearGroup(arcGroup);
  const y = yOfLayer(L);
  for (let h = 0; h < HEADS; h++) {
    const A = attnByLayer[L][h], color = HEAD_COLORS[h % HEAD_COLORS.length];
    for (let i = 0; i < T; i++) for (let j = 0; j < i; j++) {
      const w = A[i][j]; if (w < 0.14) continue;
      const pts = []; const mx = (xOf(i) + xOf(j)) / 2, my = y + 0.8 + w * 2.0 + h * 0.12;
      for (let s = 0; s <= 16; s++) { const t = s / 16; pts.push(new THREE.Vector3((1 - t) ** 2 * xOf(i) + 2 * (1 - t) * t * mx + t * t * xOf(j), (1 - t) ** 2 * y + 2 * (1 - t) * t * my + t * t * y, 0)); }
      const g = new THREE.BufferGeometry().setFromPoints(pts);
      arcGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(0.15 + w * 1.3, 0.95), blending: THREE.AdditiveBlending, depthWrite: false })));
    }
  }
}

// ---------- the canonical T×T attention HEATMAP (floating grid, right side) ----------
const HM = new THREE.Group(); HM.position.set(xOf(T - 1) + 5.5, 6, 0); scene.add(HM);
let hmCells, hmGeo, hmMesh, hmLabels = [];
const CELLW = 0.7;
function buildHeatmap() {
  clearGroup(HM); hmLabels = [];
  hmGeo = new THREE.PlaneGeometry(CELLW * 0.92, CELLW * 0.92);
  hmMesh = new THREE.InstancedMesh(hmGeo, new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), T * T);
  hmMesh.frustumCulled = false;
  const d = new THREE.Object3D();
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    d.position.set((j - (T - 1) / 2) * CELLW, ((T - 1) / 2 - i) * CELLW, 0); d.updateMatrix();
    hmMesh.setMatrixAt(i * T + j, d.matrix);
  }
  HM.add(hmMesh);
  HM.add(at(makeLabel("attention  AᵢⱼL", "#2be4ff", "#2be4ff"), new THREE.Vector3(0, (T / 2) * CELLW + 0.7, 0), new THREE.Vector3(4.2, 1.0, 1)));
}
const hmColor = new THREE.Color();
function paintHeatmap(L, head) {
  const A = attnByLayer[L][head];
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    const c = ramp(0.05 + 0.95 * A[i][j]); hmColor.setRGB(c[0], c[1], c[2]);
    hmMesh.setColorAt(i * T + j, hmColor);
  }
  hmMesh.instanceColor.needsUpdate = true;
}

// ---------- residual stream cloud (left side) + abliteration (PCA→3D) ----------
const RS = new THREE.Group(); RS.position.set(xOf(0) - 5.5, 6, 0); scene.add(RS);
let dots = [], refLine = null, basis;
let abliterate = false;
function topBasis(rows) {                  // crude: pick 3 most-spread coordinate axes for a stable 3D view
  const varc = new Array(Dm).fill(0); const mean = new Array(Dm).fill(0);
  for (const r of rows) for (let d = 0; d < Dm; d++) mean[d] += r[d] / rows.length;
  for (const r of rows) for (let d = 0; d < Dm; d++) varc[d] += (r[d] - mean[d]) ** 2;
  const idx = varc.map((v, d) => [v, d]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => x[1]);
  return { idx, mean };
}
function buildResidual() {
  clearGroup(RS); dots = [];
  for (let i = 0; i < T; i++) { const m = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12), new THREE.MeshStandardMaterial({ color: 0x62ffb3, emissive: 0x163c2c, roughness: 0.4 })); RS.add(m); dots.push(m); }
  RS.add(at(makeLabel("residual stream", "#62ffb3", "#62ffb3"), new THREE.Vector3(0, 4.2, 0), new THREE.Vector3(4.6, 1.0, 1)));
  basis = topBasis(hiddenByLayer[LAYERS - 1]);
  const a = new THREE.Vector3(refusal[basis.idx[0]], refusal[basis.idx[1]], refusal[basis.idx[2]]).normalize().multiplyScalar(3.4);
  const lg = new THREE.BufferGeometry().setFromPoints([a.clone().negate(), a]);
  refLine = new THREE.Line(lg, new THREE.LineDashedMaterial({ color: 0xff2e97, dashSize: 0.3, gapSize: 0.2, transparent: true, opacity: 0.85 })); refLine.computeLineDistances(); RS.add(refLine);
}
function placeResidual(L) {
  const rows = hiddenByLayer[L], { idx, mean } = basis, sc = 1.1;
  const rb = [refusal[idx[0]], refusal[idx[1]], refusal[idx[2]]];
  for (let i = 0; i < T; i++) {
    let p = [rows[i][idx[0]] - mean[idx[0]], rows[i][idx[1]] - mean[idx[1]], rows[i][idx[2]] - mean[idx[2]]];
    if (abliterate) { const dt = p[0] * rb[0] + p[1] * rb[1] + p[2] * rb[2]; p = [p[0] - dt * rb[0], p[1] - dt * rb[1], p[2] - dt * rb[2]]; }
    dots[i].position.lerp(new THREE.Vector3(p[0] * sc, p[1] * sc, p[2] * sc), 0.15);
    dots[i].material.color.setHex(abliterate ? 0xff2e97 : 0x62ffb3);
  }
  if (refLine) refLine.material.opacity = abliterate ? 0.95 : 0.4;
}

// ---------- logit-lens readout ----------
let lensSprite = null;
function showLens(L) {
  if (lensSprite) { nodeGroup.remove(lensSprite); lensSprite.material.map.dispose(); lensSprite.material.dispose(); }
  const tok = predict(hiddenByLayer[L][T - 1]);   // prediction for the LAST token
  lensSprite = makeLabel("predict → " + tok, "#ffd166", "#ff9f5a");
  lensSprite.scale.set(5.5, 1.2, 1);
  lensSprite.position.set(0, yOfLayer(LAYERS - 1) + 1.6, 0);
  nodeGroup.add(lensSprite);
}

// ---------- (re)build all ----------
function rebuild() { initWeights(); forward(); buildScaffold(); buildHeatmap(); buildResidual(); buildArcs(0); paintHeatmap(0, activeHead); showLens(0); }

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

let curLayer = 0, layerT = 0;
window.__diag = () => JSON.stringify({ T, HEADS, LAYERS, DK, curLayer, attnRowSum: attnByLayer[0][0][T-1].reduce((a,b)=>a+b,0).toFixed(3) });

loop((dt) => {
  meter(dt);
  layerT += dt * layerSpeed;
  if (layerT > 1.1) {                       // advance the forward pass one layer
    layerT = 0; curLayer = (curLayer + 1) % LAYERS;
    buildArcs(curLayer); paintHeatmap(curLayer, activeHead); showLens(curLayer);
    layerEl.textContent = `${curLayer + 1}/${LAYERS}`;
  }
  // subtle emphasis: active layer's arcs gently brighten (NO geometry bounce)
  const k = 0.7 + 0.3 * Math.sin(performance.now() * 0.004);
  arcGroup.children.forEach((a) => { a.material.opacity = Math.min(a.material.opacity, 0.95) * (0.85 + 0.15 * k); });
  placeResidual(curLayer);
  HM.lookAt(camera.position);               // keep the heatmap facing you
  RS.quaternion.copy(camera.quaternion);    // billboard the label group lightly
  controls.update();
  renderer.render(scene, camera);
});
