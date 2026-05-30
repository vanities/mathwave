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
const CW = 2.6;                   // token spacing → each stage is a tall, substantial column
const STAGE_X = 26.0;            // horizontal gap between stages
const EMB_X = 0.0;               // embeddings at the left
const xOfStage = (s) => EMB_X + s * STAGE_X;   // s=0 embed, s=L+1 → layer L output
const xOfLayer = (L) => xOfStage(L + 1);
const RES_CY = 14.0;             // vertical center (tokens stack around it)
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

// ---------- ONE structure: token NODES, wired together by ATTENTION ----------
// Not two competing things. Each token at each stage is a small glowing node
// (sphere), colored by its residual-state magnitude. Everything else is the
// attention wiring between nodes (built in buildFlow). The nodes are the neurons;
// the lines are the attention. That's the whole object.
const nodeGeo = new THREE.SphereGeometry(1.15, 20, 14);
const nodeMat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.25, emissive: 0x12042a });
let valueMesh = null;
const dummy = new THREE.Object3D(); const col = new THREE.Color();
const vmag = (row) => { let s = 0; for (let d = 0; d < row.length; d++) s += row[d] * row[d]; return 0.5 + 0.5 * Math.tanh(Math.sqrt(s / row.length) * 0.9); };

function ablate(H) {
  if (!abliterate) return H;
  return H.map((row) => { let dt = 0; for (let d = 0; d < Dm; d++) dt += row[d] * refusal[d]; return row.map((v, d) => v - dt * refusal[d]); });
}
// one node per (stage, token); stages: embed, then each layer output
function buildValueCubes() {
  const stages = [embed, ...hiddenByLayer.map(ablate)];
  const count = stages.length * T;
  if (valueMesh) { scene.remove(valueMesh); valueMesh.dispose(); }
  valueMesh = new THREE.InstancedMesh(nodeGeo, nodeMat, count);
  valueMesh.frustumCulled = false;
  let n = 0;
  for (let s = 0; s < stages.length; s++) {
    for (let i = 0; i < T; i++) {
      dummy.position.set(xOfStage(s), tokY(i), 0); dummy.updateMatrix();
      valueMesh.setMatrixAt(n, dummy.matrix);
      const c = ramp(vmag(stages[s][i])); col.setRGB(c[0], c[1], c[2]); valueMesh.setColorAt(n, col);
      n++;
    }
  }
  scene.add(valueMesh);
  valueMesh.instanceMatrix.needsUpdate = true; if (valueMesh.instanceColor) valueMesh.instanceColor.needsUpdate = true;
  buildFlow();   // the attention wiring between the nodes
  return count;
}

// ---------- ATTENTION lives IN the connections ----------
// Each layer's wiring isn't plain — it IS the attention pattern: a curve from
// key token j (at the input slab) to query token i (at the output slab), with
// brightness = A[i][j]. So you literally see token i reaching back and pulling
// on the tokens it attends to. Causal ⇒ i only connects to j ≤ i. Colored by the
// active head (cycle with the head button). A faint identity wire keeps the
// stream readable where attention is weak.
const flowGroup = new THREE.Group(); scene.add(flowGroup);
const tokY = (i) => RES_CY + (i - (T - 1) / 2) * CW;
const hc = new THREE.Color();
let attnEdges = 0;
function buildFlow() {
  clearGroup(flowGroup);
  hc.setHex(HEAD_COLORS[activeHead % HEAD_COLORS.length]);
  const pos = [], colr = [];
  attnEdges = 0;
  // faint identity residual: token i → token i, very dim (the stream skeleton)
  const idPos = [], idCol = [];
  for (let s = 0; s < LAYERS + 1; s++) {
    if (s >= LAYERS) break;
    const x0 = xOfStage(s) + 0.6, x1 = xOfStage(s + 1) - 0.6;
    for (let i = 0; i < T; i++) { const y = tokY(i); idPos.push(x0, y, 0, x1, y, 0); const c = ramp(0.25 + 0.5 * (i / T)); for (let k = 0; k < 2; k++) idCol.push(c[0] * 0.25, c[1] * 0.25, c[2] * 0.25); }
  }
  const idG = new THREE.BufferGeometry();
  idG.setAttribute("position", new THREE.Float32BufferAttribute(idPos, 3));
  idG.setAttribute("color", new THREE.Float32BufferAttribute(idCol, 3));
  flowGroup.add(new THREE.LineSegments(idG, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false })));

  // attention connections for the active head, per layer
  const SEG = 14;
  for (let L = 0; L < LAYERS; L++) {
    const A = attnByLayer[L][activeHead];
    const x0 = xOfStage(L) + 0.6, x1 = xOfStage(L + 1) - 0.6;
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
      const w = A[i][j]; if (w < 0.06) continue;               // declutter weak links
      attnEdges++;
      const yj = tokY(j), yi = tokY(i);
      // quadratic bezier bowing forward in +Z (toward viewer) by strength, so the
      // strongest attention arcs reach out the most and don't all overlap
      const mx = (x0 + x1) / 2, my = (yj + yi) / 2, mz = 3.0 + w * 16.0 + (i - j) * 0.4;
      const b = 0.28 + w * 2.3;   // brightness = attention weight — strong links blaze                                 // brightness = weight
      let px = x0, py = yj, pz = 0;
      for (let s = 1; s <= SEG; s++) {
        const t = s / SEG, u = 1 - t;
        const nx = u*u*x0 + 2*u*t*mx + t*t*x1;
        const ny = u*u*yj + 2*u*t*my + t*t*yi;
        const nz = u*u*0  + 2*u*t*mz + t*t*0;
        pos.push(px, py, pz, nx, ny, nz);
        colr.push(hc.r*b, hc.g*b, hc.b*b, hc.r*b, hc.g*b, hc.b*b);
        px = nx; py = ny; pz = nz;
      }
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(colr, 3));
  flowGroup.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false })));

  // bright spine = the residual highway running the length of the stream
  const sp = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(EMB_X - 1, RES_CY, 0), new THREE.Vector3(END_X + 1, RES_CY, 0)]);
  flowGroup.add(new THREE.Line(sp, new THREE.LineBasicMaterial({ color: 0x62ffb3, transparent: true, opacity: 0.3 })));
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
function rebuild() { initWeights(); forward(); valCount = buildValueCubes(); buildLabels(); showLens(LAYERS - 1); }

// ---------- panel ----------
const ablBtn = document.getElementById("abliterate");
ablBtn.addEventListener("click", () => { abliterate = !abliterate; ablBtn.classList.toggle("active", abliterate); ablBtn.textContent = abliterate ? "abliterated ●" : "abliterate"; document.getElementById("mode").textContent = abliterate ? "refusal removed" : "intact"; buildValueCubes(); });
let layerSpeed = 1; bindRange("speed", (v) => { layerSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("heads", (v) => { HEADS = Math.round(v); activeHead = Math.min(activeHead, HEADS - 1); rebuild(); }, (v) => `${Math.round(v)}`);
const headBtn = document.getElementById("head");
headBtn.addEventListener("click", () => { activeHead = (activeHead + 1) % HEADS; headBtn.textContent = "head " + (activeHead + 1); buildFlow(); });
document.getElementById("resample").addEventListener("click", () => { seedSalt += 13; rebuild(); });
setVariantCycler((d) => { ablBtn.click(); return abliterate ? "abliterated" : "intact"; });

// ---------- boot ----------
rebuild();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const layerEl = document.getElementById("layer");
window.__diag = () => JSON.stringify({ T, HEADS, LAYERS, DK, curLayer, valueCubes: valCount, attnEdges, activeHead, attnRowSum: attnByLayer[0][0][T - 1].reduce((a, b) => a + b, 0).toFixed(3) });

loop((dt) => {
  meter(dt);
  layerT += dt * layerSpeed;
  if (layerT > 1.0) { layerT = 0; curLayer = (curLayer + 1) % LAYERS; showLens(curLayer); layerEl.textContent = `${curLayer + 1}/${LAYERS}`; }
  planeX += (xOfLayer(curLayer) - planeX) * Math.min(1, dt * 4); plane.position.x = planeX;  // sweep along the pipeline
  controls.update();
  renderer.render(scene, camera);
});
