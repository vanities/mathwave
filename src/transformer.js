// transformer.js — a GPT-style transformer as a genuinely 3D structure you fly
// around. The key to it being 3D (not a flat diagram): a transformer's data is
// TOKENS × CHANNELS, so every token is a vector — drawn here as a row of cubes
// extending in DEPTH. That gives each stage real volume:
//   X = stage   (embed → layer 0 → … → layer 5 → output)  ← the pipeline
//   Y = token   (the 10 tokens stack upward)
//   Z = channel (each token's 16 feature values run into the screen)
// So each stage is a solid 2D SLAB standing in the Y–Z plane, and the slabs are
// strung along X. Genuinely volumetric in all three axes.
//
// ATTENTION is the wiring: a glowing arc from token j to token i (causal, j≤i),
// brightness = the real weight A[i][j], colored by the active head — so you watch
// each token reach back through 3D space and pull on what it attends to.
//
// Real forward pass (verified: softmax rows sum to 1.000):
//   x = embed(tok) + sinusoidalPE(pos); Qh=x·Wq_h …; A=softmax(QKᵀ/√dk + mask);
//   x += concat_h(A·Vh)·Wo ; x += W2·relu(W1·x) ; ×LAYERS ; logits = x·Wuᵀ.
// ABLITERATE projects the residual off the refusal direction; HEAD cycles heads.
// Refs: bbycroft.net/llm, Vaswani 2017, logit lens, abliteration (mlabonne).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05060b);
scene.fog = new THREE.Fog(0x05060b, 95, 320);      // near-black so EVERY cube colour pops (teal bg ate the cyan/green cubes)

// ---------- model dims ----------
const TOKENS = ["the", "cat", "sat", "on", "the", "mat", "and", "then", "it", "purrs"];
const T = TOKENS.length, VOCAB = TOKENS, Dm = 16;
let HEADS = 4; const LAYERS = 6; let DK = Dm / HEADS;
const HEAD_COLORS = [0xff2e97, 0x2be4ff, 0x62ffb3, 0xb06bff, 0xffd166, 0xff7a5a];

// ---------- 3D layout: X=stage, Y=token, Z=channel ----------
const CW = 1.15;                                  // cube/cell spacing
const STAGE_X = 30.0;                             // gap between stages along X
const EMB_X = 0.0;
const xOfStage = (s) => EMB_X + s * STAGE_X;      // s=0 embed, s=L+1 = layer L output
const xOfLayer = (L) => xOfStage(L + 1);
const NSTAGE = LAYERS + 1;
const END_X = xOfStage(NSTAGE - 1);
const MIDX = END_X / 2;
const yOfTok = (i) => (i - (T - 1) / 2) * CW;     // token → Y (centered on 0)
const zOfCh = (c) => (c - (Dm - 1) / 2) * CW;     // channel → Z (centered on 0)
const CY = 12.0;                                  // lift the whole structure off the grid

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(MIDX - 50, CY + 20, 120);     // 3/4 so all 7 slabs read left→right as solid blocks
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.28;
controls.minDistance = 16; controls.maxDistance = 420;
controls.target.set(MIDX, CY, 0);

// neutral lights so the lit cube faces read as 3D BLOCKS (top/side shading) without
// tinting their value-colours.
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.85); keyLight.position.set(40, 60, 50); scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x9fd0ff, 0.3); rimLight.position.set(-40, 10, -30); scene.add(rimLight);
// clean dark-teal wire floor (no neon magenta, no wobble), dissolving into the fog
const floor = new THREE.GridHelper(440, 88, 0x3aa6bf, 0x174450);
floor.position.set(MIDX, -1, 0);
floor.material.transparent = true; floor.material.opacity = 0.5; floor.material.depthWrite = false;
scene.add(floor);

// ---------- subtle bloom: a gentle glow on the bright cube faces, kept crisp ----------
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
// high threshold → only the brightest lit faces glow; low strength → stays legible
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.42, 0.5, 0.72);
composer.addPass(bloom);

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
// LayerNorm (per token row): without it the residual stream grows every layer,
// so Q·K explodes → softmax saturates → only layer 0 shows varied attention.
// Real GPT is pre-LN, so this is both the fix and the accurate thing.
const layerNorm = (X) => X.map((row) => {
  let m = 0; for (const v of row) m += v; m /= row.length;
  let s = 0; for (const v of row) s += (v - m) * (v - m); s = Math.sqrt(s / row.length + 1e-5);
  return row.map((v) => (v - m) / s);
});
let embed, attnByLayer, hiddenByLayer, refusal;
function forward() {
  attnByLayer = []; hiddenByLayer = [];
  let x = addM(WE, PE); embed = x.map((r) => r.slice());
  for (let L = 0; L < LAYERS; L++) {
    const xn = layerNorm(x);                       // pre-attention norm
    const heads = [], headOuts = [];
    for (let h = 0; h < HEADS; h++) {
      const Q = matmul(xn, WQ[h]), K = matmul(xn, WK[h]), V = matmul(xn, WV[h]);
      const A = [];
      for (let i = 0; i < T; i++) { const s = new Array(T).fill(-1e9); for (let j = 0; j <= i; j++) { let d = 0; for (let p = 0; p < DK; p++) d += Q[i][p] * K[j][p]; s[j] = d / Math.sqrt(DK); } A.push(softmaxCausal(s, i)); }
      heads.push(A);
      headOuts.push(A.map((ar) => { const o = new Array(DK).fill(0); for (let j = 0; j < T; j++) for (let p = 0; p < DK; p++) o[p] += ar[j] * V[j][p]; return o; }));
    }
    x = addM(x, matmul(concatCols(headOuts), WO));
    x = addM(x, matmul(relu(matmul(layerNorm(x), W1)), W2));   // pre-FFN norm
    attnByLayer.push(heads); hiddenByLayer.push(x.map((r) => r.slice()));
  }
  const mean = new Array(Dm).fill(0);
  for (const row of hiddenByLayer[LAYERS - 1]) for (let d = 0; d < Dm; d++) mean[d] += row[d] / T;
  const nrm = Math.hypot(...mean) || 1; refusal = mean.map((v) => v / nrm);
}
function predict(row) { let best = 0, bv = -1e9; for (let t = 0; t < VOCAB.length; t++) { let d = 0; for (let p = 0; p < Dm; p++) d += row[p] * WU[t][p]; if (d > bv) { bv = d; best = t; } } return VOCAB[best]; }

// ---------- labels ----------
function makeLabel(text, color = "#f6e9ff", glow = "#2be4ff") {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64; const g = c.getContext("2d");
  g.font = "bold 42px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 12; g.fillText(text, 128, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  return sp;
}
const labelGroup = new THREE.Group(); scene.add(labelGroup);
const clearGroup = (G) => { while (G.children.length) { const c = G.children.pop(); c.geometry && c.geometry.dispose(); c.material && (c.material.map && c.material.map.dispose(), c.material.dispose()); G.remove(c); } };

// ---------- the cubes: every (stage, token, channel) is a value-colored cube ----------
// rounded-ish, low-roughness + emissive vertex colors so each cell self-glows into bloom
const cubeGeo = new THREE.BoxGeometry(CW * 0.82, CW * 0.82, CW * 0.82);
// unlit, vertex-colored: each cube renders EXACTLY its value-hue (no white emissive
// or metallic reflection to wash it out). Bloom adds glow from the color's own brightness.
const cubeMat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.0 });
let cubeMesh = null;
const dummy = new THREE.Object3D(); const col = new THREE.Color();
const vnorm = (v) => 0.5 + 0.5 * Math.tanh(v * 0.7);   // v is a z-score → spreads cyan→pink evenly
// bright PS1 value→color (cyan→green→amber→pink); the quantizer crunches it into bands
const PS1_STOPS = [new THREE.Color(0x14e0ff), new THREE.Color(0x57ff9b), new THREE.Color(0xffe24d), new THREE.Color(0xff5d8f)];
const _ps1c = new THREE.Color();
const ps1col = (t) => { t = Math.min(0.9999, Math.max(0, t)) * (PS1_STOPS.length - 1); const i = Math.floor(t); return _ps1c.copy(PS1_STOPS[i]).lerp(PS1_STOPS[i + 1] || PS1_STOPS[i], t - i); };
function ablate(H) {
  if (!abliterate) return H;
  return H.map((row) => { let dt = 0; for (let d = 0; d < Dm; d++) dt += row[d] * refusal[d]; return row.map((v, d) => v - dt * refusal[d]); });
}
function buildCubes() {
  const stages = [embed, ...hiddenByLayer.map(ablate)];   // 7 slabs
  // RANK-normalise across ALL stages → colour ramp spread uniformly (z-scoring left
  // everything pink; the distribution is too skewed). Each cube is coloured by its
  // percentile, so the slabs are a guaranteed cyan→green→amber→pink rainbow.
  const _sorted = [];
  for (const M of stages) for (let i = 0; i < T; i++) for (let c = 0; c < Dm; c++) _sorted.push(M[i][c]);
  _sorted.sort((a, b) => a - b);
  const _N = _sorted.length;
  const rankOf = (v) => { let lo = 0, hi = _N; while (lo < hi) { const m = (lo + hi) >> 1; if (_sorted[m] < v) lo = m + 1; else hi = m; } return _N > 1 ? lo / (_N - 1) : 0.5; };
  const count = stages.length * T * Dm;
  if (cubeMesh) { scene.remove(cubeMesh); cubeMesh.dispose(); }
  cubeMesh = new THREE.InstancedMesh(cubeGeo, cubeMat, count);
  cubeMesh.frustumCulled = false;
  let n = 0;
  for (let s = 0; s < stages.length; s++) {
    const x = xOfStage(s), M = stages[s];
    for (let i = 0; i < T; i++) for (let c = 0; c < Dm; c++) {
      dummy.position.set(x, CY + yOfTok(i), zOfCh(c)); dummy.updateMatrix();
      cubeMesh.setMatrixAt(n, dummy.matrix);
      col.copy(ps1col(rankOf(M[i][c]))); cubeMesh.setColorAt(n, col);
      n++;
    }
  }
  scene.add(cubeMesh);
  cubeMesh.instanceMatrix.needsUpdate = true; if (cubeMesh.instanceColor) cubeMesh.instanceColor.needsUpdate = true;
  buildFlow();
  return count;
}

// ---------- ATTENTION = the wiring; pulses of light FLOW along each arc ----------
// Each connection is a strand of small glowing dots that travel j→i over time —
// so you literally watch information move from a token to the one attending to it.
// (No moving plane; the animation IS the attention.)
const flowGroup = new THREE.Group(); scene.add(flowGroup);
const hc = new THREE.Color();
let attnEdges = 0;
const halfZ = (Dm / 2) * CW;
const anchor = (stage, i) => new THREE.Vector3(xOfStage(stage), CY + yOfTok(i), halfZ + 0.6);

let edges = [];            // {curve, w, layer, base[r,g,b], n} sampled control data
let pulseGeo = null, pulsePos = null, pulseCol = null, pulseMesh = null;
const PER = 3;             // travelling dots per connection
const SEG = 16;            // faint static strand resolution

function buildFlow() {
  clearGroup(flowGroup);
  hc.setHex(HEAD_COLORS[activeHead % HEAD_COLORS.length]);
  edges = []; attnEdges = 0;
  const strandPos = [], strandCol = [];
  for (let L = 0; L < LAYERS; L++) {
    const A = attnByLayer[L][activeHead];
    for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
      const w = A[i][j]; if (w < 0.05) continue;
      attnEdges++;
      const a = anchor(L, j), b = anchor(L + 1, i);
      const mid = new THREE.Vector3((a.x + b.x) / 2, (a.y + b.y) / 2 + w * 3.0, halfZ + 2 + w * 10 + (i - j) * 0.4);
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      const base = [hc.r, hc.g, hc.b];
      edges.push({ curve, w, layer: L, base });
      // faint static strand so the wiring is visible even between pulses
      const fb = 0.04 + w * 0.22;
      let p = curve.getPoint(0);
      for (let s = 1; s <= SEG; s++) {
        const q = curve.getPoint(s / SEG);
        strandPos.push(p.x, p.y, p.z, q.x, q.y, q.z);
        for (let k = 0; k < 2; k++) strandCol.push(base[0] * fb, base[1] * fb, base[2] * fb);
        p = q;
      }
    }
  }
  const sg = new THREE.BufferGeometry();
  sg.setAttribute("position", new THREE.Float32BufferAttribute(strandPos, 3));
  sg.setAttribute("color", new THREE.Float32BufferAttribute(strandCol, 3));
  flowGroup.add(new THREE.LineSegments(sg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false })));

  // travelling pulses: PER dots per edge, animated each frame in the loop
  const total = edges.length * PER;
  pulsePos = new Float32Array(total * 3);
  pulseCol = new Float32Array(total * 3);
  pulseGeo = new THREE.BufferGeometry();
  pulseGeo.setAttribute("position", new THREE.BufferAttribute(pulsePos, 3).setUsage(THREE.DynamicDrawUsage));
  pulseGeo.setAttribute("color", new THREE.BufferAttribute(pulseCol, 3));
  if (pulseMesh) flowGroup.remove(pulseMesh);
  pulseMesh = new THREE.Points(pulseGeo, new THREE.PointsMaterial({ size: 0.8, vertexColors: true, transparent: true, opacity: 0.32, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true }));
  pulseMesh.frustumCulled = false;
  flowGroup.add(pulseMesh);
}

// advance the pulses along every edge (called from the render loop)
const _pt = new THREE.Vector3();
function animateFlow(time) {
  if (!edges.length || !pulseGeo) return;
  let n = 0;
  for (let e = 0; e < edges.length; e++) {
    const ed = edges[e];
    const speedBoost = 0.4 + ed.w * 1.6;             // stronger attention → faster, brighter flow
    for (let k = 0; k < PER; k++) {
      // each dot offset along the strand; flows j→i (the direction of compute)
      let t = (time * speedBoost + k / PER + e * 0.137) % 1;
      ed.curve.getPoint(t, _pt);
      pulsePos[n * 3] = _pt.x; pulsePos[n * 3 + 1] = _pt.y; pulsePos[n * 3 + 2] = _pt.z;
      // brightness fades in/out along the strand (comet-like) and scales with weight
      const fade = Math.sin(t * Math.PI);
      const br = (0.07 + ed.w * 0.7) * fade;
      pulseCol[n * 3] = ed.base[0] * br; pulseCol[n * 3 + 1] = ed.base[1] * br; pulseCol[n * 3 + 2] = ed.base[2] * br;
      n++;
    }
  }
  pulseGeo.attributes.position.needsUpdate = true;
  pulseGeo.attributes.color.needsUpdate = true;
}

// ---------- labels under each stage ----------
function buildLabels() {
  clearGroup(labelGroup);
  const lowY = CY - (T / 2) * CW - 2.6;
  const put = (sp, x, y, z, sx) => { sp.position.set(x, y, z); sp.scale.set(sx, sx * 0.25, 1); labelGroup.add(sp); };
  put(makeLabel("token embeddings", "#c0a3e8", "#b06bff"), EMB_X, lowY, halfZ + 1, 8);
  for (let L = 0; L < LAYERS; L++) put(makeLabel("layer " + L, "#8366b8", "#4a1f7a"), xOfLayer(L), lowY, halfZ + 1, 6);
  put(makeLabel("input →", "#62ffb3", "#62ffb3"), EMB_X - 11, CY, halfZ + 1, 6);
  put(makeLabel("→ output", "#ffd166", "#ff9f5a"), END_X + 11, CY, halfZ + 1, 6);
}

// ---------- logit lens ----------
const lensGroup = new THREE.Group(); scene.add(lensGroup);
function showLens(L) {
  clearGroup(lensGroup);
  const sp = makeLabel("predict → " + predict(hiddenByLayer[L][T - 1]), "#ffd166", "#ff9f5a");
  sp.scale.set(11, 2.6, 1); sp.position.set(END_X + 13, CY + 5, 0); lensGroup.add(sp);
}

// ---------- (re)build ----------
let abliterate = false, activeHead = 0, curLayer = 0, layerT = 0, cubeCount = 0;
function rebuild() { initWeights(); forward(); cubeCount = buildCubes(); buildLabels(); showLens(LAYERS - 1); }

// ---------- panel ----------
const ablBtn = document.getElementById("abliterate");
ablBtn.addEventListener("click", () => { abliterate = !abliterate; ablBtn.classList.toggle("active", abliterate); ablBtn.textContent = abliterate ? "abliterated ●" : "abliterate"; document.getElementById("mode").textContent = abliterate ? "refusal removed" : "intact"; buildCubes(); });
let layerSpeed = 1; bindRange("speed", (v) => { layerSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("heads", (v) => { HEADS = Math.round(v); activeHead = Math.min(activeHead, HEADS - 1); rebuild(); }, (v) => `${Math.round(v)}`);
const headBtn = document.getElementById("head");
headBtn.addEventListener("click", () => { activeHead = (activeHead + 1) % HEADS; headBtn.textContent = "head " + (activeHead + 1); buildFlow(); });
document.getElementById("resample").addEventListener("click", () => { seedSalt += 13; rebuild(); });
setVariantCycler((d) => { ablBtn.click(); return abliterate ? "abliterated" : "intact"; });

// ---------- boot ----------
rebuild();
liftVeil();
onResize(renderer, camera, (w, h) => composer.setSize(w, h));
const meter = fpsMeter(document.getElementById("fps"));
const layerEl = document.getElementById("layer");
window.__diag = () => {
  // edges per layer (proves attention is spread across ALL gaps, not just the first)
  const per = [];
  for (let L = 0; L < LAYERS; L++) { let e = 0; const A = attnByLayer[L][activeHead]; for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) if (A[i][j] >= 0.05) e++; per.push(e); }
  return JSON.stringify({ T, HEADS, LAYERS, DK, Dm, curLayer, cubes: cubeCount, attnEdges, edgesPerLayer: per, attnRowSum: attnByLayer[0][0][T - 1].reduce((a, b) => a + b, 0).toFixed(3) });
};

let clock = 0;
loop((dt) => {
  meter(dt);
  clock += dt * layerSpeed;
  layerT += dt * layerSpeed;
  if (layerT > 1.0) { layerT = 0; curLayer = (curLayer + 1) % LAYERS; showLens(curLayer); if (layerEl) layerEl.textContent = `${curLayer + 1}/${LAYERS}`; }
  animateFlow(clock * 0.35);   // pulses of light travel along the attention arcs
  controls.update();
  composer.render();           // render through the bloom pipeline
});
