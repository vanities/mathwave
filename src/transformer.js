// transformer.js — "Attention Is All You Need", made visible, plus a live
// ABLITERATION toggle.
//
// Tokens sit in a row as glowing nodes. Above them, several stacked LAYERS light
// up in sequence; within each layer, multi-head self-attention is drawn as arcs
// from every token to every other, one COLOR per head, thickness ∝ attention
// weight. The weights come from a toy scaled-dot-product softmax over random
// per-token Q/K vectors, so the pattern is different every load but behaves like
// real attention (a token attends most to a few others).
//
// To the side floats the RESIDUAL STREAM: each token's hidden state as a point
// in a 3D cloud. Abliteration = find the "refusal direction" r and project every
// hidden state onto the plane orthogonal to r (h ← h − (h·r)r). Toggle it and
// watch the cloud collapse off that axis — the geometric act of removing a
// behavior direction from the residual stream.
//
// Refs: Vaswani et al. 2017 (attention); arditi et al. / mlabonne (abliteration:
//   "refusal is mediated by a single direction in the residual stream").

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.011);

const camera = new THREE.PerspectiveCamera(54, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 6, 26);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 10; controls.maxDistance = 90;
controls.target.set(0, 4, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.9); key.position.set(8, 16, 10); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.7); rim.position.set(-12, 8, -8); scene.add(rim);
addGrid(scene, { size: 60, divisions: 30, y: -1 });
addSun(scene, { scale: 46, position: [0, 12, -70] });

// ---------- config ----------
const TOKENS = ["the", "cat", "sat", "on", "the", "mat", "and", "purred"];
let T = TOKENS.length;
let HEADS = 4;
let LAYERS = 5;
const DK = 8;                       // toy head dimension
const HEAD_COLORS = [0xff2e97, 0x2be4ff, 0x62ffb3, 0xb06bff, 0xffd166, 0xff7a5a];

const COL_W = 3.2;                  // spacing between tokens
const LAYER_H = 1.7;               // vertical spacing between layers
const xOf = (i) => (i - (T - 1) / 2) * COL_W;

// ---------- token nodes (bottom row) + sprites ----------
const nodeGroup = new THREE.Group();
scene.add(nodeGroup);
const tokSprites = [];

function makeLabel(text) {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(0,0,0,0)"; g.fillRect(0, 0, 256, 64);
  g.font = "bold 40px 'VT323', monospace";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = "#f6e9ff"; g.shadowColor = "#2be4ff"; g.shadowBlur = 10;
  g.fillText(text, 128, 34);
  const tex = new THREE.CanvasTexture(c);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sp.scale.set(2.6, 0.65, 1);
  return sp;
}

// ---------- random per-token Q/K/V and a refusal direction ----------
let Q, K, V, refusal;
let seedSalt = 0;
function rng(seed) { let s = seed * 9301 + 49297; return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; }; }
function gaussian(r) { return Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); }

function initWeights() {
  const r = rng(1234 + seedSalt);
  Q = []; K = []; V = [];
  for (let h = 0; h < HEADS; h++) {
    const q = [], k = [];
    for (let i = 0; i < T; i++) {
      const qi = [], ki = [];
      for (let d = 0; d < DK; d++) { qi.push(gaussian(r)); ki.push(gaussian(r)); }
      q.push(qi); k.push(ki);
    }
    Q.push(q); K.push(k);
  }
  // residual hidden states (3D for display)
  V = [];
  for (let i = 0; i < T; i++) V.push([gaussian(r) * 3, 4 + Math.abs(gaussian(r)) * 2.2, gaussian(r) * 3]);
  // a unit "refusal direction"
  refusal = [gaussian(r), gaussian(r) * 0.4, gaussian(r)];
  const n = Math.hypot(...refusal); refusal = refusal.map((c) => c / n);
}

// attention weights for head h: softmax_j (q_i·k_j / sqrt(dk))
function attention(h) {
  const W = [];
  for (let i = 0; i < T; i++) {
    const logits = [];
    let mx = -1e9;
    for (let j = 0; j < T; j++) {
      let dot = 0; for (let d = 0; d < DK; d++) dot += Q[h][i][d] * K[h][j][d];
      dot /= Math.sqrt(DK);
      logits.push(dot); if (dot > mx) mx = dot;
    }
    let sum = 0; for (let j = 0; j < T; j++) { logits[j] = Math.exp(logits[j] - mx); sum += logits[j]; }
    for (let j = 0; j < T; j++) logits[j] /= sum;
    W.push(logits);
  }
  return W;
}

// ---------- attention arc lines (rebuilt per layer reveal) ----------
const arcGroup = new THREE.Group();
scene.add(arcGroup);

function clearGroup(g) { while (g.children.length) { const c = g.children.pop(); c.geometry && c.geometry.dispose(); g.remove(c); } }

function arc(x1, y1, x2, y2, lift, color, weight) {
  const pts = [];
  const seg = 18;
  const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + lift;
  for (let s = 0; s <= seg; s++) {
    const t = s / seg;
    // quadratic bezier
    const x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mx + t * t * x2;
    const y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * my + t * t * y2;
    pts.push(new THREE.Vector3(x, y, 0));
  }
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(0.1 + weight * 1.4, 0.95), blending: THREE.AdditiveBlending, depthWrite: false });
  return new THREE.Line(g, m);
}

function buildLayer(layer) {
  clearGroup(arcGroup);
  const yBase = 1 + layer * LAYER_H;
  for (let h = 0; h < HEADS; h++) {
    const W = attention(h);
    const color = HEAD_COLORS[h % HEAD_COLORS.length];
    const zoff = (h - (HEADS - 1) / 2) * 0.0; // keep planar; color separates heads
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        if (i === j) continue;
        const w = W[i][j];
        if (w < 0.12) continue;                 // declutter: only strong links
        const a = arc(xOf(i), yBase, xOf(j), yBase, 0.9 + w * 2.2 + h * 0.15, color, w);
        arcGroup.add(a);
      }
    }
  }
}

// ---------- residual stream cloud (right side) ----------
const cloudOrigin = new THREE.Vector3(0, 4, 0);
const residualGroup = new THREE.Group();
residualGroup.position.set(0, 0, 0);
scene.add(residualGroup);

let dots = [];
let refLine = null;
let abliterate = false;

function buildResidual() {
  clearGroup(residualGroup);
  dots = [];
  for (let i = 0; i < T; i++) {
    const d = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0x62ffb3, emissive: 0x163c2c, roughness: 0.4 }));
    residualGroup.add(d); dots.push(d);
  }
  // the refusal-direction axis through the cloud center
  const c = new THREE.Vector3(...meanV());
  const r = new THREE.Vector3(...refusal).multiplyScalar(6);
  const lg = new THREE.BufferGeometry().setFromPoints([c.clone().sub(r), c.clone().add(r)]);
  refLine = new THREE.Line(lg, new THREE.LineDashedMaterial({ color: 0xff2e97, dashSize: 0.4, gapSize: 0.25, transparent: true, opacity: 0.85 }));
  refLine.computeLineDistances();
  residualGroup.add(refLine);
}
function meanV() {
  const m = [0, 0, 0];
  for (const h of V) { m[0] += h[0]; m[1] += h[1]; m[2] += h[2]; }
  return m.map((c) => c / V.length);
}

function placeResidual() {
  const c = meanV();
  for (let i = 0; i < T; i++) {
    let h = V[i].slice();
    if (abliterate) {
      // project onto plane orthogonal to refusal: h ← h − ((h−c)·r) r
      const rel = [h[0] - c[0], h[1] - c[1], h[2] - c[2]];
      const dot = rel[0] * refusal[0] + rel[1] * refusal[1] + rel[2] * refusal[2];
      h = [h[0] - dot * refusal[0], h[1] - dot * refusal[1], h[2] - dot * refusal[2]];
    }
    // ease toward target for a smooth collapse
    const d = dots[i];
    d.position.lerp(new THREE.Vector3(h[0], h[1], h[2]), 0.12);
    d.material.color.setHex(abliterate ? 0xff2e97 : 0x62ffb3);
    d.material.emissive.setHex(abliterate ? 0x3c1426 : 0x163c2c);
  }
  if (refLine) refLine.material.opacity = abliterate ? 0.95 : 0.4;
}

// ---------- (re)build everything for current T/HEADS/LAYERS ----------
function rebuild() {
  clearGroup(nodeGroup);
  tokSprites.length = 0;
  for (let i = 0; i < T; i++) {
    const node = new THREE.Mesh(new THREE.SphereGeometry(0.3, 18, 14),
      new THREE.MeshStandardMaterial({ color: 0x2be4ff, emissive: 0x0b3a44, roughness: 0.3 }));
    node.position.set(xOf(i), 1, 0);
    nodeGroup.add(node);
    const sp = makeLabel(TOKENS[i % TOKENS.length]);
    sp.position.set(xOf(i), 0.1, 0);
    nodeGroup.add(sp); tokSprites.push(sp);
    // faint column up through the layers
    const colGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(xOf(i), 1, 0), new THREE.Vector3(xOf(i), 1 + (LAYERS - 1) * LAYER_H, 0),
    ]);
    nodeGroup.add(new THREE.Line(colGeo, new THREE.LineBasicMaterial({ color: 0x4a1f7a, transparent: true, opacity: 0.5 })));
  }
  initWeights();
  buildResidual();
}

// ---------- panel ----------
let ablBtn = document.getElementById("abliterate");
ablBtn.addEventListener("click", () => {
  abliterate = !abliterate;
  ablBtn.classList.toggle("active", abliterate);
  ablBtn.textContent = abliterate ? "abliterated ●" : "abliterate";
  document.getElementById("mode").textContent = abliterate ? "refusal removed" : "intact";
});

let layerSpeed = 1;
bindRange("speed", (v) => { layerSpeed = v; }, (v) => v.toFixed(2) + "×");

bindRange("heads", (v) => { HEADS = Math.round(v); rebuild(); }, (v) => `${Math.round(v)}`);

document.getElementById("resample").addEventListener("click", () => { seedSalt++; rebuild(); });

setVariantCycler((d) => {
  // ↑↓ toggles abliteration (the headline interaction)
  ablBtn.click();
  return abliterate ? "abliterated" : "intact";
});

// ---------- boot ----------
rebuild();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const layerEl = document.getElementById("layer");

let curLayer = 0, layerT = 0;
buildLayer(0);
loop((dt) => {
  meter(dt);
  layerT += dt * layerSpeed;
  if (layerT > 0.9) {                 // advance the active layer (the "forward pass")
    layerT = 0;
    curLayer = (curLayer + 1) % LAYERS;
    buildLayer(curLayer);
    layerEl.textContent = `${curLayer + 1}/${LAYERS}`;
  }
  // pulse the active layer's arcs
  const pulse = 0.6 + 0.4 * Math.sin(performance.now() * 0.006);
  arcGroup.children.forEach((a) => { a.material.opacity = Math.min(a.material.opacity, 1) * 1; a.scale.y = pulse; });
  placeResidual();
  controls.update();
  renderer.render(scene, camera);
});
