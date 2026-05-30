// grokking.js — how a transformer ACTUALLY learns modular addition.
// When a small net groks "a + b mod p", mechanistic-interpretability work
// (Nanda et al. 2023) found it doesn't memorize — it places each number k on a
// CIRCLE at angle 2πk/p (a Fourier feature), and computes a+b by COMPOSING
// ROTATIONS: angle(a) + angle(b) = angle((a+b) mod p). Addition becomes rotation.
//
// Here: a ring of p numbered nodes (the learned embedding circle). Two arms show
// a and b; their angles add; the sum snaps onto node (a+b)%p — exact, every time.
// Stacked behind it: the same numbers embedded at several Fourier frequencies ω,
// which is what the real network superposes. ↑↓ changes the modulus p.
//
// Ref: "Progress measures for grokking via mechanistic interpretability", ICLR'23.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 4, 16);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.3;
controls.minDistance = 7; controls.maxDistance = 50;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.7); key.position.set(4, 10, 8); scene.add(key);
addGrid(scene, { size: 40, divisions: 20, y: -6 });
addSun(scene, { scale: 30, position: [0, 10, -60] });

let P = 11;            // modulus (prime, like the paper's 113 but small enough to read)
const RING_R = 5;
const FREQS = [1, 2, 3];   // Fourier frequencies the network superposes

const mainGroup = new THREE.Group(); scene.add(mainGroup);
const freqGroup = new THREE.Group(); scene.add(freqGroup);

function makeLabel(text, color = "#f6e9ff", glow = "#2be4ff", scale = 1) {
  const c = document.createElement("canvas"); c.width = 128; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 44px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = glow; g.shadowBlur = 10; g.fillText(text, 64, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sp.scale.set(1.1 * scale, 0.55 * scale, 1); return sp;
}
const at = (sp, p) => { sp.position.copy(p); return sp; };  // position is a read-only ref
const clearGroup = (gr) => { while (gr.children.length) { const c = gr.children.pop(); c.geometry && c.geometry.dispose(); c.material && c.material.dispose && c.material.dispose(); gr.remove(c); } };

let nodeMeshes = [], nodeLabels = [];
const angleOf = (k) => (2 * Math.PI * k) / P - Math.PI / 2;   // k=0 at top
const posOf = (k, r = RING_R) => new THREE.Vector3(Math.cos(angleOf(k)) * r, Math.sin(angleOf(k)) * r, 0);

let armA, armB, armSum, sumNode;
function build() {
  clearGroup(mainGroup); clearGroup(freqGroup); nodeMeshes = []; nodeLabels = [];

  // main embedding circle: ring of p numbered nodes
  const ringPts = [];
  for (let k = 0; k <= P; k++) ringPts.push(posOf(k % P));
  mainGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ringPts), new THREE.LineBasicMaterial({ color: 0x4a1f7a, transparent: true, opacity: 0.6 })));
  for (let k = 0; k < P; k++) {
    const c = ramp(k / P);
    const node = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12), new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), emissive: new THREE.Color(c[0] * 0.3, c[1] * 0.3, c[2] * 0.3), roughness: 0.35 }));
    node.position.copy(posOf(k)); mainGroup.add(node); nodeMeshes.push(node);
    const lab = makeLabel(String(k)); lab.position.copy(posOf(k, RING_R + 0.7)); mainGroup.add(lab); nodeLabels.push(lab);
  }

  // arms: a (cyan), b stacked onto a (mint), result (white)
  const mk = (color) => { const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(1,0,0)]); return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending })); };
  armA = mk(0x2be4ff); armB = mk(0x62ffb3); armSum = mk(0xffffff);
  mainGroup.add(armA, armB, armSum);
  sumNode = new THREE.Mesh(new THREE.SphereGeometry(0.34, 18, 14), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  mainGroup.add(sumNode);

  // behind it: the same numbers embedded at several Fourier frequencies ω
  FREQS.forEach((w, fi) => {
    const z = -3 - fi * 2.4, r = 2.6 - fi * 0.2;
    const pts = [];
    for (let k = 0; k <= P; k++) { const a = (2 * Math.PI * w * (k % P)) / P; pts.push(new THREE.Vector3(Math.cos(a) * r, Math.sin(a) * r, z)); }
    const cc = [0x2be4ff, 0xff2e97, 0x62ffb3][fi % 3];
    freqGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: cc, transparent: true, opacity: 0.3 })));
    for (let k = 0; k < P; k++) { const a = (2 * Math.PI * w * k) / P; const d = new THREE.Mesh(new THREE.SphereGeometry(0.07, 8, 6), new THREE.MeshBasicMaterial({ color: cc, transparent: true, opacity: 0.5 })); d.position.set(Math.cos(a) * r, Math.sin(a) * r, z); freqGroup.add(d); }
    freqGroup.add(at(makeLabel("ω=" + w, "#c0a3e8", "#b06bff", 1.4), new THREE.Vector3(0, r + 0.8, z)));
  });

  if (pEl) pEl.textContent = P;
}

function setArm(line, k, frac = 1) {
  const end = posOf(k).multiplyScalar(frac);
  line.geometry.setFromPoints([new THREE.Vector3(), end]);
  line.geometry.attributes.position.needsUpdate = true;
}

// ---------- panel ----------
const pEl = document.getElementById("pval");
const eqEl = document.getElementById("eq");
let speed = 1;
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
const PRIMES = [7, 11, 13, 17, 23];
bindRange("mod", (v) => { P = PRIMES[Math.round(v)]; build(); }, (v) => `${PRIMES[Math.round(v)]}`);
setVariantCycler((d) => { const i = Math.max(0, PRIMES.indexOf(P)); const ni = (i + d + PRIMES.length) % PRIMES.length; P = PRIMES[ni]; const el = document.getElementById("mod"); el.value = ni; build(); return "p = " + P; });

// ---------- boot ----------
build();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

let t = 0, a = 0, b = 0;
loop((dt) => {
  meter(dt);
  t += dt * speed * 0.5;
  // every ~2s pick a new (a,b); animate b sweeping, sum landing on (a+b)%p
  const cyc = t % 1;
  if (cyc < dt * speed * 0.5) { a = (Math.random() * P) | 0; b = (Math.random() * P) | 0; }
  const sum = (a + b) % P;
  // highlight a and the running sum
  nodeMeshes.forEach((n, k) => { const on = (k === a || k === sum); n.scale.setScalar(on ? 1.8 : 1); });
  setArm(armA, a);
  // armB: from a's node, sweep toward sum (visualizing the +b rotation)
  const sweep = Math.min(cyc * 2.2, 1);
  const aAng = angleOf(a), addAng = (2 * Math.PI * b) / P;
  const curAng = aAng + addAng * sweep;
  const bEnd = new THREE.Vector3(Math.cos(curAng) * RING_R, Math.sin(curAng) * RING_R, 0);
  armB.geometry.setFromPoints([posOf(a), bEnd]); armB.geometry.attributes.position.needsUpdate = true;
  setArm(armSum, sum, sweep);
  sumNode.position.copy(posOf(sum)); sumNode.visible = sweep > 0.98;
  if (eqEl) eqEl.textContent = `${a} + ${b} ≡ ${sum} (mod ${P})`;
  controls.update();
  renderer.render(scene, camera);
});
