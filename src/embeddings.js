// embeddings.js — semantic vector arithmetic, the thing that made word2vec
// famous:  king − man + woman ≈ queen.  Words become vectors; meaning becomes
// GEOMETRY, so analogies are parallelograms in the space. We use a small
// hand-built embedding where interpretable axes (gender, royalty, age, place…)
// are near-orthogonal, so the arithmetic lands exactly — the same structure real
// embeddings discover, made legible. The analogy parallelogram is drawn live.
// ↑↓ steps through famous analogies.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(54, innerWidth / innerHeight, 0.1, 600);
camera.position.set(7, 5, 11);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.35;
controls.minDistance = 5; controls.maxDistance = 40;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.7); key.position.set(5, 10, 7); scene.add(key);
addGrid(scene, { size: 36, divisions: 18, y: -6 });
addSun(scene, { scale: 28, position: [0, 10, -56] });

// ---------- a tiny interpretable embedding (axes: x=gender, y=royalty, z=place/other) ----------
// [x, y, z]; we scatter for display. Built so analogies are exact parallelograms.
const SCALE = 3.2;
const WORDS = {
  man:    [-1, -1,  0], woman:  [ 1, -1,  0],
  king:   [-1,  1,  0], queen:  [ 1,  1,  0],
  prince: [-1,  0.4, 0], princess: [1, 0.4, 0],
  boy:    [-1, -1.6, 0.2], girl: [1, -1.6, 0.2],
  uncle:  [-1, -0.6, -1.2], aunt: [1, -0.6, -1.2],
  france:[ 0.2, -0.2, 1.6], paris:[ -0.2, 0.5, 1.6],
  japan: [ 1.4, -0.2, 1.6], tokyo:[ 1.0, 0.5, 1.6],
  italy: [-1.4, -0.2, 1.6], rome: [-1.0, 0.5, 1.6],
};
const ANALOGIES = [
  ["king − man + woman = ?", "king", "man", "woman", "queen"],
  ["queen − woman + man = ?", "queen", "woman", "man", "king"],
  ["paris − france + japan = ?", "paris", "france", "japan", "tokyo"],
  ["paris − france + italy = ?", "paris", "france", "italy", "rome"],
  ["king − prince + princess = ?", "king", "prince", "princess", "queen"],
  ["uncle − man + woman = ?", "uncle", "man", "woman", "aunt"],
];
let idx = 0;

const V = (w) => new THREE.Vector3(WORDS[w][0], WORDS[w][1], WORDS[w][2]).multiplyScalar(SCALE);

function makeLabel(text, color = "#f6e9ff", glow = "#2be4ff", s = 1) {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 38px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 10; g.fillText(text, 128, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sp.scale.set(2.6 * s, 0.66 * s, 1); return sp;
}
const at = (sp, p) => { sp.position.copy(p); return sp; };  // position is a read-only ref
const clearGroup = (gr) => { while (gr.children.length) { const c = gr.children.pop(); c.geometry && c.geometry.dispose(); c.material && c.material.dispose && c.material.dispose(); gr.remove(c); } };

const cloud = new THREE.Group(); scene.add(cloud);
const overlay = new THREE.Group(); scene.add(overlay);

function buildCloud() {
  clearGroup(cloud);
  for (const w of Object.keys(WORDS)) {
    const p = V(w);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.16, 14, 10), new THREE.MeshStandardMaterial({ color: 0x6a3fae, emissive: 0x1a0b30, roughness: 0.4 }));
    dot.position.copy(p); cloud.add(dot);
    const lab = makeLabel(w, "#c0a3e8", "#b06bff", 0.85); lab.position.copy(p).add(new THREE.Vector3(0, 0.5, 0)); cloud.add(lab);
  }
}

function arrow(from, to, color) {
  const dir = to.clone().sub(from); const len = dir.length();
  const a = new THREE.ArrowHelper(dir.clone().normalize(), from, len, color, Math.min(0.6, len * 0.25), 0.3);
  a.line.material.transparent = true; a.line.material.opacity = 0.95;
  return a;
}

function showAnalogy(i) {
  clearGroup(overlay);
  const [, A, B, C, expect] = ANALOGIES[i];
  const va = V(A), vb = V(B), vc = V(C);
  const result = va.clone().sub(vb).add(vc);     // A − B + C
  // parallelogram arrows: B→A (the relation) and C→result (same relation)
  overlay.add(arrow(vb, va, 0x2be4ff));          // man→king
  overlay.add(arrow(vc, result, 0x62ffb3));      // woman→(result)
  overlay.add(arrow(va, result, 0xff2e97));      // king→result  (the +C−B step)
  // highlight the four words
  for (const w of [A, B, C]) { const h = new THREE.Mesh(new THREE.SphereGeometry(0.28, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 })); h.position.copy(V(w)); overlay.add(h); }
  // result marker + nearest-word readout
  const rm = new THREE.Mesh(new THREE.SphereGeometry(0.32, 18, 14), new THREE.MeshBasicMaterial({ color: 0xffffff })); rm.position.copy(result); overlay.add(rm);
  overlay.add(at(makeLabel("= " + expect, "#ffd166", "#ff9f5a", 1.1), result.clone().add(new THREE.Vector3(0, 0.7, 0))));
  // nearest neighbor (proof it lands on the right word)
  let best = "", bd = 1e9;
  for (const w of Object.keys(WORDS)) { const d = V(w).distanceTo(result); if (d < bd) { bd = d; best = w; } }
  if (eqEl) eqEl.textContent = ANALOGIES[i][0].replace("?", best);
  if (nnEl) nnEl.textContent = best + (best === expect ? " ✓" : "");
}

// ---------- panel ----------
const wrap = document.getElementById("analogies");
const eqEl = document.getElementById("eq");
const nnEl = document.getElementById("nn");
const chips = ANALOGIES.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label.replace(" = ?", "");
  b.addEventListener("click", () => { idx = i; showAnalogy(i); chips.forEach((c, k) => c.classList.toggle("active", k === i)); });
  wrap.appendChild(b);
  return b;
});
const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => { camera.position.copy(home); controls.target.set(0, 0, 0); });
setVariantCycler((d) => { idx = (idx + d + ANALOGIES.length) % ANALOGIES.length; showAnalogy(idx); chips.forEach((c, k) => c.classList.toggle("active", k === idx)); return ANALOGIES[idx][0].replace(" = ?", ""); });

// ---------- boot ----------
buildCloud();
showAnalogy(0);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); controls.update(); renderer.render(scene, camera); });
