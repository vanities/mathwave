// transformer.js — a GPT-style transformer as a TOWER OF TENSORS, in the spirit
// of Brendan Bycroft's bbycroft.net/llm: every tensor is a solid 3D block made
// of small cubes, each cube COLORED BY ITS ACTUAL VALUE from a real forward pass.
// Not a wireframe diagram — a volumetric structure you fly up.
//
// The real math (unchanged, verified: softmax rows sum to 1.000):
//   x  = embed(token) + sinusoidal_PE(pos)
//   Qh = x·Wq_h   Kh = x·Wk_h   Vh = x·Wv_h
//   A  = softmax(Qh·Khᵀ/√dk + causal_mask)
//   x ← x + concat_h(A·Vh)·Wo                (attention residual)
//   x ← x + W2·relu(W1·x)                    (FFN residual)
//   ×LAYERS ; logits = x·Wuᵀ (logit lens)
//
// The 3D TOWER (data flows UP):
//   • bottom slab  = token embedding matrix  (T tokens × Dm channels) of cubes
//   • each layer, rising:
//       – Q,K,V  : three slabs of cubes on the LEFT (the projections)
//       – ATTENTION : a T×T field of BARS on the RIGHT (bar height = weight),
//                     so the causal mask is a literal 3D staircase
//       – RESIDUAL : the hidden-state slab (T×Dm) at the center — the stream
//   • ABLITERATE recolors the residual slabs with the refusal direction projected
//     out (you watch the upper layers shift); HEAD cycles which head's bars show.
//
// Refs: bbycroft.net/llm (tensor-block layout); Vaswani 2017; logit lens;
//   abliteration (arditi et al. / mlabonne).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.0035);

// ---------- model dims ----------
const TOKENS = ["the", "cat", "sat", "on", "the", "mat", "and", "then", "it", "purrs"];
const T = TOKENS.length, VOCAB = TOKENS, Dm = 16;
let HEADS = 4; const LAYERS = 6; let DK = Dm / HEADS;

// ---------- PIPELINE geometry (horizontal: input → output along +X) ----------
// A transformer reads left→right; data flows through a connected residual stream.
//   X = stage (embed, then each layer) — the direction of compute
//   Y = token position (rows stack up)
//   Z = channel / feature dimension (depth)
const CW = 1.0;                   // cube cell spacing
const STAGE_X = 26.0;            // horizontal gap between stages
const EMB_X = 0.0;               // embeddings at the left
const xOfStage = (s) => EMB_X + s * STAGE_X;   // s=0 embed, s=L+1 → layer L output
const xOfLayer = (L) => xOfStage(L + 1);
const RES_CY = 8.0;              // residual slab vertical center (tokens stack around it)
const END_X = xOfLayer(LAYERS - 1);
const MIDX = END_X / 2;

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(MIDX, RES_CY + 22, 70);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 14; controls.maxDistance = 400;
controls.target.set(MIDX, RES_CY, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 1.0));
const key = new THREE.DirectionalLight(0xfff1dd, 1.1); key.position.set(25, 50, 35); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.7); rim.position.set(-30, 20, -25); scene.add(rim);
const fill = new THREE.DirectionalLight(0xff2e97, 0.45); fill.position.set(20, -10, 20); scene.add(fill);
addGrid(scene, { size: 200, divisions: 40, y: -1 });
addSun(scene, { scale: 70, position: [MIDX, RES_CY + 30, -220] });

// ---------- deterministic RNG + linear algebra ----------
let seedSalt = 7;
function rng(seed) { let s = (seed * 9301 + 49297) % 233280 || 1; return () => (s = (s * 9301 + 49297) % 233280) / 233280; }
function gauss(r) { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }
function mat(rows, cols, r, sc = 1) { return Array.from({ length: rows }, () => Array.from({ length: cols }, () => gauss(r) * sc)); }
const matmul = (A, B) => { const n = A.length, k = B.length, m = B[0].length, o = []; for (let i = 0; i < n; i++) { const row = new Array(m).fill(0); for (let p = 0; p < k; p++) { const a = A[i][p], Bp = B[p]; for (let j = 0; j < m; j++) row[j] += a * Bp[j]; } o.push(row); } return o; };
const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
const relu = (A) => A.map((r) => r.map((v) => Math.max(0, v)));
const concatCols = (parts) => parts[0].map((_, i) => parts.flatMap((P) => P[i]));

// ---------- weights + forward pass ----------
let WE, PE, WQ, WK, WV, WO, W1, W2, WU;
function sinusoidalPE() { const pe = []; for (let pos = 0; pos < T; pos++) { const row = []; for (let i = 0; i < Dm; i++) { const k = Math.floor(i / 2), d = Math.pow(10000, (2 * k) / Dm); row.push(i % 2 ? Math.cos(pos / d) : Math.sin(pos / d)); } pe.push(row); } return pe; }
function initWeights() {
  DK = Dm / HEADS; const r = rng(1234 + seedSalt);
  WE = mat(T, Dm, r, 1); PE = sinusoidalPE();
  WQ = []; WK = []; WV = [];
  for (let h = 0; h < HEADS; h++) { WQ.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); WK.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); WV.push(mat(Dm, DK, r, 1 / Math.sqrt(Dm))); }
  WO = mat(Dm, Dm, r, 1 / Math.sqrt(Dm)); W1 = mat(Dm, Dm * 2, r, 1 / Math.sqrt(Dm)); W2 = mat(Dm * 2, Dm, r, 1 / Math.sqrt(Dm * 2)); WU = mat(T, Dm, r, 1 / Math.sqrt(Dm));
}
function softmaxCausal(s, i) { let mx = -1e9; for (let j = 0; j <= i; j++) mx = Math.max(mx, s[j]); let sm = 0; const o = new Array(s.length).fill(0); for (let j = 0; j <= i; j++) { o[j] = Math.exp(s[j] - mx); sm += o[j]; } for (let j = 0; j <= i; j++) o[j] /= sm; return o; }
let embed, qByLayer, kByLayer, vByLayer, attnByLayer, hiddenByLayer, refusal;
function forward() {
  qByLayer = []; kByLayer = []; vByLayer = []; attnByLayer = []; hiddenByLayer = [];
  let x = addM(WE, PE); embed = x.map((r) => r.slice());
  for (let L = 0; L < LAYERS; L++) {
    const heads = [], headOuts = [], Qh = [], Kh = [], Vh = [];
    for (let h = 0; h < HEADS; h++) {
      const Q = matmul(x, WQ[h]), K = matmul(x, WK[h]), V = matmul(x, WV[h]);
      Qh.push(Q); Kh.push(K); Vh.push(V);
      const A = [];
      for (let i = 0; i < T; i++) { const s = new Array(T).fill(-1e9); for (let j = 0; j <= i; j++) { let d = 0; for (let p = 0; p < DK; p++) d += Q[i][p] * K[j][p]; s[j] = d / Math.sqrt(DK); } A.push(softmaxCausal(s, i)); }
      heads.push(A);
      headOuts.push(A.map((ar) => { const o = new Array(DK).fill(0); for (let j = 0; j < T; j++) for (let p = 0; p < DK; p++) o[p] += ar[j] * V[j][p]; return o; }));
    }
    x = addM(x, matmul(concatCols(headOuts), WO));
    x = addM(x, matmul(relu(matmul(x, W1)), W2));
    qByLayer.push(Qh); kByLayer.push(Kh); vByLayer.push(Vh);
    attnByLayer.push(heads); hiddenByLayer.push(x.map((r) => r.slice()));
  }
  const mean = new Array(Dm).fill(0);
  for (const row of hiddenByLayer[LAYERS - 1]) for (let d = 0; d < Dm; d++) mean[d] += row[d] / T;
  const nrm = Math.hypot(...mean) || 1; refusal = mean.map((v) => v / nrm);
}
function predict(row) { let best = 0, bv = -1e9; for (let t = 0; t < VOCAB.length; t++) { let d = 0; for (let p = 0; p < Dm; p++) d += row[p] * WU[t][p]; if (d > bv) { bv = d; best = t; } } return VOCAB[best]; }

// ---------- label sprites ----------
function makeLabel(text, color = "#f6e9ff", glow = "#2be4ff") {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64; const g = c.getContext("2d");
  g.font = "bold 42px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 12; g.fillText(text, 128, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sp.scale.set(3.2, 0.8, 1); return sp;
}
const labelGroup = new THREE.Group(); scene.add(labelGroup);
const clearGroup = (G) => { while (G.children.length) { const c = G.children.pop(); c.geometry && c.geometry.dispose(); c.material && (c.material.map && c.material.map.dispose(), c.material.dispose()); G.remove(c); } };

// ---------- the voxel tensors ----------
// two instanced meshes: unit value-cubes (embed/QKV/residual) and attention bars
const cubeGeo = new THREE.BoxGeometry(CW * 0.82, CW * 0.82, CW * 0.82);
const cubeMat = new THREE.MeshStandardMaterial({ roughness: 0.38, metalness: 0.18, vertexColors: false });
const barGeo = new THREE.BoxGeometry(CW * 0.8, 1, CW * 0.8).translate(0, 0.5, 0);
const barMat = new THREE.MeshStandardMaterial({ roughness: 0.32, metalness: 0.22 });
let valueMesh = null, barMesh = null;
const dummy = new THREE.Object3D(); const col = new THREE.Color();

const vnorm = (v) => 0.5 + 0.5 * Math.tanh(v * 0.6);   // value → 0..1 for coloring

// a (T × C) matrix → a vertical SLAB of cubes standing in the Y-Z plane at x=cx:
//   token i → Y (rows stack up), channel c → Z (depth). This is one "tensor" the
//   data becomes at a stage; slabs sit in a row along +X = the pipeline.
function slabSpecs(M, cx, specs) {
  const rows = M.length, cols = M[0].length;
  for (let i = 0; i < rows; i++) for (let c = 0; c < cols; c++)
    specs.push({ x: cx, y: RES_CY + (i - (rows - 1) / 2) * CW, z: (c - (cols - 1) / 2) * CW, t: vnorm(M[i][c]) });
}
function ablate(H) {
  if (!abliterate) return H;
  return H.map((row) => { let dt = 0; for (let d = 0; d < Dm; d++) dt += row[d] * refusal[d]; return row.map((v, d) => v - dt * refusal[d]); });
}
function buildValueCubes() {
  const specs = [];
  slabSpecs(embed, EMB_X, specs);                       // embeddings at the far left
  for (let L = 0; L < LAYERS; L++) slabSpecs(ablate(hiddenByLayer[L]), xOfLayer(L), specs);  // residual after each layer
  if (valueMesh) { scene.remove(valueMesh); valueMesh.dispose(); }
  valueMesh = new THREE.InstancedMesh(cubeGeo, cubeMat, specs.length);
  valueMesh.frustumCulled = false;
  for (let n = 0; n < specs.length; n++) {
    const s = specs[n];
    dummy.position.set(s.x, s.y, s.z); dummy.scale.setScalar(1); dummy.updateMatrix();
    valueMesh.setMatrixAt(n, dummy.matrix);
    const c = ramp(s.t); col.setRGB(c[0], c[1], c[2]); valueMesh.setColorAt(n, col);
  }
  scene.add(valueMesh);
  valueMesh.instanceMatrix.needsUpdate = true; if (valueMesh.instanceColor) valueMesh.instanceColor.needsUpdate = true;
  buildFlow();   // the connectors that make the stream continuous
  return specs.length;
}

// ---------- the CONNECTED residual stream: lines linking each token's slab to
//   the same token in the next slab, so data visibly flows through every stage.
const flowGroup = new THREE.Group(); scene.add(flowGroup);
function buildFlow() {
  clearGroup(flowGroup);
  const stages = [embed, ...hiddenByLayer.map(ablate)];   // embed → L0 → … → L5
  const zMid = 0;
  for (let s = 0; s < stages.length - 1; s++) {
    const x0 = xOfStage(s), x1 = xOfStage(s + 1);
    const pos = [], colr = [];
    for (let i = 0; i < T; i++) {
      const y = RES_CY + (i - (T - 1) / 2) * CW;
      // a faint per-token wire from this slab's front face to the next
      pos.push(x0 + 0.5, y, zMid, x1 - 0.5, y, zMid);
      const c = ramp(0.3 + 0.5 * (i / T));
      colr.push(c[0], c[1], c[2], c[0], c[1], c[2]);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(colr, 3));
    flowGroup.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false })));
  }
  // a bright spine down the middle of the whole stream (the residual highway)
  const sp = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(EMB_X - 1, RES_CY, 0), new THREE.Vector3(END_X + 1, RES_CY, 0)]);
  flowGroup.add(new THREE.Line(sp, new THREE.LineBasicMaterial({ color: 0x62ffb3, transparent: true, opacity: 0.35 })));
}

// attention bars: a T×T field floating ABOVE each layer's stage (between the
// previous residual slab and this one — that's where attention happens). Bar
// height = weight; the causal mask is the empty upper triangle (a 3D staircase).
function buildBars() {
  const specs = [];   // {x,y,z,h,t,masked}
  for (let L = 0; L < LAYERS; L++) {
    const cx = (xOfStage(L) + xOfStage(L + 1)) / 2;     // midway across the layer
    const A = attnByLayer[L][activeHead];
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const masked = j > i, w = masked ? 0 : A[i][j];
      specs.push({
        x: cx + (j - (T - 1) / 2) * CW,                 // key index → X (across the stage)
        y: RES_CY + 13,                                  // hovering above the stream
        z: (i - (T - 1) / 2) * CW,                       // query index → Z (depth)
        h: masked ? 0.05 : 0.15 + w * 5.5, t: w, masked,
      });
    }
  }
  if (barMesh) { scene.remove(barMesh); barMesh.dispose(); }
  barMesh = new THREE.InstancedMesh(barGeo, barMat, specs.length);
  barMesh.frustumCulled = false;
  for (let n = 0; n < specs.length; n++) {
    const s = specs[n];
    dummy.position.set(s.x, s.y, s.z); dummy.scale.set(1, s.h, 1); dummy.updateMatrix();
    barMesh.setMatrixAt(n, dummy.matrix);
    if (s.masked) col.setRGB(0.05, 0.03, 0.1); else { const c = ramp(0.15 + 0.85 * s.t); col.setRGB(c[0], c[1], c[2]); }
    barMesh.setColorAt(n, col);
  }
  barMesh.instanceMatrix.needsUpdate = true; if (barMesh.instanceColor) barMesh.instanceColor.needsUpdate = true;
  scene.add(barMesh);
}

// labels under each stage of the horizontal pipeline
function buildLabels() {
  clearGroup(labelGroup);
  const lowY = RES_CY - (T / 2) * CW - 2.2;
  const at = (sp, x, y, z, sx) => { sp.position.set(x, y, z); if (sx) sp.scale.set(sx, sx * 0.25, 1); return sp; };
  labelGroup.add(at(makeLabel("token embeddings", "#c0a3e8", "#b06bff"), EMB_X, lowY, 0, 7));
  for (let L = 0; L < LAYERS; L++) {
    labelGroup.add(at(makeLabel("layer " + L, "#8366b8", "#4a1f7a"), xOfLayer(L), lowY, 0, 5));
    labelGroup.add(at(makeLabel("attention Aᵢⱼ", "#ff2e97", "#ff2e97"), (xOfStage(L) + xOfStage(L + 1)) / 2, RES_CY + 20, 0, 6));
  }
  labelGroup.add(at(makeLabel("input →", "#62ffb3", "#62ffb3"), EMB_X - 9, RES_CY, 0, 5));
  labelGroup.add(at(makeLabel("→ output", "#ffd166", "#ff9f5a"), END_X + 9, RES_CY, 0, 5));
}

// ---------- compute-plane that sweeps ALONG the pipeline (a vertical wall that
//   travels left→right through the stages = the forward pass moving through) ----
const plane = new THREE.Mesh(new THREE.PlaneGeometry(28, 34).rotateY(Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x2be4ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false }));
plane.position.set(EMB_X, RES_CY, 0);
scene.add(plane);

// ---------- logit lens (at the output end of the pipeline) ----------
const lensGroup = new THREE.Group(); scene.add(lensGroup);
function showLens(L) {
  clearGroup(lensGroup);
  const sp = makeLabel("predict → " + predict(hiddenByLayer[L][T - 1]), "#ffd166", "#ff9f5a");
  sp.scale.set(9, 2.2, 1); sp.position.set(END_X + 9, RES_CY + 4, 0); lensGroup.add(sp);
}

// ---------- (re)build all ----------
let abliterate = false, activeHead = 0, curLayer = 0, layerT = 0, planeX = EMB_X, valCount = 0;
function rebuild() { initWeights(); forward(); valCount = buildValueCubes(); buildBars(); buildLabels(); showLens(LAYERS - 1); }

// ---------- panel ----------
const ablBtn = document.getElementById("abliterate");
ablBtn.addEventListener("click", () => { abliterate = !abliterate; ablBtn.classList.toggle("active", abliterate); ablBtn.textContent = abliterate ? "abliterated ●" : "abliterate"; document.getElementById("mode").textContent = abliterate ? "refusal removed" : "intact"; buildValueCubes(); });
let layerSpeed = 1; bindRange("speed", (v) => { layerSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("heads", (v) => { HEADS = Math.round(v); activeHead = Math.min(activeHead, HEADS - 1); rebuild(); }, (v) => `${Math.round(v)}`);
const headBtn = document.getElementById("head");
headBtn.addEventListener("click", () => { activeHead = (activeHead + 1) % HEADS; headBtn.textContent = "head " + (activeHead + 1); buildValueCubes(); buildBars(); });
document.getElementById("resample").addEventListener("click", () => { seedSalt += 13; rebuild(); });
setVariantCycler((d) => { ablBtn.click(); return abliterate ? "abliterated" : "intact"; });

// ---------- boot ----------
rebuild();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const layerEl = document.getElementById("layer");
window.__diag = () => JSON.stringify({ T, HEADS, LAYERS, DK, curLayer, valueCubes: valCount, bars: barMesh ? barMesh.count : 0, attnRowSum: attnByLayer[0][0][T - 1].reduce((a, b) => a + b, 0).toFixed(3) });

loop((dt) => {
  meter(dt);
  layerT += dt * layerSpeed;
  if (layerT > 1.0) { layerT = 0; curLayer = (curLayer + 1) % LAYERS; showLens(curLayer); layerEl.textContent = `${curLayer + 1}/${LAYERS}`; }
  planeX += (xOfLayer(curLayer) - planeX) * Math.min(1, dt * 4); plane.position.x = planeX;  // sweep along the pipeline
  controls.update();
  renderer.render(scene, camera);
});
