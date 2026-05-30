// kuramoto.js — the Kuramoto model: how synchrony emerges from chaos. N
// oscillators, each with its own natural frequency ωᵢ, nudge each other:
//   dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ − θᵢ)
// Below a critical coupling Kc they drift independently; raise K past Kc and they
// spontaneously lock into a common rhythm — the math behind fireflies flashing in
// unison and metronomes syncing on a table. We show them two ways at once: dots
// orbiting a ring at their phase angle (they bunch up when synced), and a central
// arrow = the ORDER PARAMETER r·e^{iψ} (length r: 0=chaos, 1=total sync). The big
// 3D field of "fireflies" pulses brightness with each phase. ↑↓ steps coupling K.
//
// Ref: Kuramoto 1975; r·e^{iψ}=(1/N)Σe^{iθⱼ} (Strogatz review).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.013);

const camera = new THREE.PerspectiveCamera(54, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 8, 22);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.35;
controls.minDistance = 8; controls.maxDistance = 70;
controls.target.set(0, 2, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.7));
addGrid(scene, { size: 60, divisions: 30, y: -8 });
addSun(scene, { scale: 36, position: [0, 14, -70] });

// ---------- oscillators ----------
let N = 600;
let K = 1.5;
let theta, omega;
function alloc() {
  theta = new Float32Array(N); omega = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    theta[i] = Math.random() * Math.PI * 2;
    // natural frequencies ~ Gaussian around 1
    omega[i] = 1 + (Math.sqrt(-2 * Math.log(Math.random() + 1e-9)) * Math.cos(2 * Math.PI * Math.random())) * 0.35;
  }
}

// fireflies: a 3D grid of glowing sprites, brightness = (1+cos θ)/2
let fireflies, ffColor;
const FF_R = 11;
function buildFireflies() {
  if (fireflies) scene.remove(fireflies);
  const geo = new THREE.SphereGeometry(0.16, 8, 6);
  fireflies = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({}), N);
  fireflies.frustumCulled = false;
  // fixed random positions in a ball
  const d = new THREE.Object3D();
  fireflies.userData.home = [];
  for (let i = 0; i < N; i++) {
    const r = Math.cbrt(Math.random()) * FF_R, a = Math.random() * Math.PI * 2, e = Math.acos(Math.random() * 2 - 1);
    const x = r * Math.sin(e) * Math.cos(a), y = 2 + r * Math.cos(e) * 0.6, z = r * Math.sin(e) * Math.sin(a);
    fireflies.userData.home.push([x, y, z]);
    d.position.set(x, y, z); d.updateMatrix(); fireflies.setMatrixAt(i, d.matrix);
  }
  scene.add(fireflies);
  ffColor = new THREE.Color();
}

// phase ring (top): dots at angle θ on a circle; bunch = synced
const RING_R = 5, RING_Y = 9;
let ring;
function buildRing() {
  if (ring) scene.remove(ring);
  ring = new THREE.InstancedMesh(new THREE.SphereGeometry(0.13, 8, 6), new THREE.MeshBasicMaterial({}), N);
  ring.frustumCulled = false; scene.add(ring);
  const pts = []; for (let k = 0; k <= 64; k++) { const a = k/64*Math.PI*2; pts.push(new THREE.Vector3(Math.cos(a)*RING_R, RING_Y, Math.sin(a)*RING_R)); }
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x4a1f7a, transparent: true, opacity: 0.5 })));
}
// order-parameter arrow at ring center
const orderArrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0, RING_Y, 0), RING_R, 0xffffff, 0.6, 0.4);
scene.add(orderArrow);

// ---------- step ----------
function step(dt) {
  // order parameter r e^{iψ}
  let sx = 0, sy = 0;
  for (let i = 0; i < N; i++) { sx += Math.cos(theta[i]); sy += Math.sin(theta[i]); }
  sx /= N; sy /= N;
  const r = Math.hypot(sx, sy), psi = Math.atan2(sy, sx);
  // mean-field update: dθ = ω + K r sin(ψ − θ)
  const h = Math.min(dt, 0.05) * 2.2;
  for (let i = 0; i < N; i++) theta[i] += (omega[i] + K * r * Math.sin(psi - theta[i])) * h;
  return { r, psi };
}

const dummy = new THREE.Object3D(); const col = new THREE.Color();
function draw(order) {
  // fireflies brightness by phase
  for (let i = 0; i < N; i++) {
    const b = (1 + Math.cos(theta[i])) / 2;
    const c = ramp(0.2 + 0.6 * ((theta[i] % (Math.PI*2)) / (Math.PI*2)));
    ffColor.setRGB(c[0] * b + 0.02, c[1] * b + 0.02, c[2] * b + 0.04);
    fireflies.setColorAt(i, ffColor);
    // ring dot at phase angle
    dummy.position.set(Math.cos(theta[i]) * RING_R, RING_Y, Math.sin(theta[i]) * RING_R);
    dummy.updateMatrix(); ring.setMatrixAt(i, dummy.matrix);
    ring.setColorAt(i, ffColor);
  }
  fireflies.instanceColor.needsUpdate = true;
  ring.instanceColor.needsUpdate = true;
  ring.instanceMatrix.needsUpdate = true;
  // order arrow
  orderArrow.setDirection(new THREE.Vector3(Math.cos(order.psi), 0, Math.sin(order.psi)));
  orderArrow.setLength(Math.max(0.05, order.r) * RING_R, 0.6, 0.4);
}

// ---------- panel ----------
const rEl = document.getElementById("rval");
const kEl = document.getElementById("kval");
bindRange("coupling", (v) => { K = v; kEl.textContent = v.toFixed(2); }, (v) => v.toFixed(2));
document.getElementById("scramble").addEventListener("click", () => { for (let i = 0; i < N; i++) theta[i] = Math.random() * Math.PI * 2; });
const KS = [0, 0.5, 1.0, 1.6, 2.5, 4.0];
let ki = 3;
setVariantCycler((d) => { ki = (ki + d + KS.length) % KS.length; K = KS[ki]; const el = document.getElementById("coupling"); el.value = K; el.dispatchEvent(new Event("input")); return "K = " + K.toFixed(1); });

// ---------- boot ----------
alloc(); buildFireflies(); buildRing();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  const order = step(dt);
  draw(order);
  if (rEl) rEl.textContent = order.r.toFixed(3);
  controls.update();
  renderer.render(scene, camera);
});
