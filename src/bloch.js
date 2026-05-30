// bloch.js — a qubit on the Bloch sphere. A pure single-qubit state is
//   |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
// which maps bijectively to a point on a unit sphere at polar angle θ, azimuth φ.
// |0⟩ is the north pole, |1⟩ the south, and the equator is equal superposition
// with the phase φ running around it. Quantum GATES are rotations of this vector:
//   X,Y,Z = π rotations about those axes; H swaps poles↔equator (X+Z); S,T are
//   φ phase shifts of π/2, π/4. We apply each gate as a real SU(2) rotation of
//   the Bloch vector and animate the arc. ↑↓ cycles a preset gate sequence.
//
// Ref: Nielsen & Chuang, "Quantum Computation and Quantum Information", Bloch sphere.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(6, 4, 7);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 4; controls.maxDistance = 30;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.6); key.position.set(5, 8, 6); scene.add(key);
addGrid(scene, { size: 30, divisions: 15, y: -4 });
addSun(scene, { scale: 26, position: [0, 9, -50] });

const R = 3;
// translucent sphere
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R, 48, 32), new THREE.MeshStandardMaterial({ color: 0x3a1f6e, transparent: true, opacity: 0.12, roughness: 0.6, depthWrite: false })));
scene.add(new THREE.Mesh(new THREE.SphereGeometry(R, 24, 16), new THREE.MeshBasicMaterial({ color: 0x6a3fae, wireframe: true, transparent: true, opacity: 0.12 })));

// axes X(cyan) Y(mint) Z(magenta) + equator
function axis(dir, color) {
  const a = new THREE.ArrowHelper(dir, dir.clone().multiplyScalar(-R), R * 2, color, 0.4, 0.24);
  a.line.material.transparent = true; a.line.material.opacity = 0.6; scene.add(a);
}
axis(new THREE.Vector3(1,0,0), 0x2be4ff); axis(new THREE.Vector3(0,1,0), 0x62ffb3); axis(new THREE.Vector3(0,0,1), 0xff2e97);
const eqPts = []; for (let k = 0; k <= 96; k++) { const a = k/96*Math.PI*2; eqPts.push(new THREE.Vector3(Math.cos(a)*R, 0, Math.sin(a)*R)); }
scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(eqPts), new THREE.LineBasicMaterial({ color: 0x6a3fae, transparent: true, opacity: 0.4 })));

function makeLabel(text, color) {
  const c = document.createElement("canvas"); c.width = 128; c.height = 64;
  const g = c.getContext("2d"); g.font = "bold 44px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 8; g.fillText(text, 64, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false })); sp.scale.set(1, 0.5, 1); return sp;
}
const l0 = makeLabel("|0⟩", "#ff2e97"); l0.position.set(0, R + 0.6, 0); scene.add(l0);
const l1 = makeLabel("|1⟩", "#2be4ff"); l1.position.set(0, -R - 0.6, 0); scene.add(l1);

// ---------- the state vector (Bloch vector) ----------
// store as unit 3-vec; map: north(+Y)=|0⟩, so θ from +Y, φ in XZ plane.
let vec = new THREE.Vector3(0, R, 0);       // start at |0⟩
let target = vec.clone();
const stateArrow = new THREE.ArrowHelper(vec.clone().normalize(), new THREE.Vector3(), R, 0xffffff, 0.5, 0.3);
stateArrow.line.material.linewidth = 3; scene.add(stateArrow);
const tip = new THREE.Mesh(new THREE.SphereGeometry(0.18, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })); scene.add(tip);

// gate = rotation of the Bloch vector. axis in world (Y is |0>-|1> i.e. Z-axis of qubit).
// We treat: qubit Z-axis ↔ world Y; qubit X ↔ world X; qubit Y ↔ world Z.
function rotAxisAngle(v, axis, ang) { return v.clone().applyAxisAngle(axis, ang); }
const QX = new THREE.Vector3(1,0,0), QY = new THREE.Vector3(0,0,1), QZ = new THREE.Vector3(0,1,0);
function applyGate(g) {
  if (g === "X") target = rotAxisAngle(vec, QX, Math.PI);
  else if (g === "Y") target = rotAxisAngle(vec, QY, Math.PI);
  else if (g === "Z") target = rotAxisAngle(vec, QZ, Math.PI);
  else if (g === "H") target = rotAxisAngle(vec, new THREE.Vector3(1,1,0).normalize(), Math.PI); // (X+Z)/√2
  else if (g === "S") target = rotAxisAngle(vec, QZ, Math.PI/2);
  else if (g === "T") target = rotAxisAngle(vec, QZ, Math.PI/4);
  else if (g === "√X") target = rotAxisAngle(vec, QX, Math.PI/2);
  if (gateEl) gateEl.textContent = g;
}

// ---------- panel ----------
const wrap = document.getElementById("gates");
const gateEl = document.getElementById("gate");
const stateEl = document.getElementById("qstate");
["X","Y","Z","H","S","T","√X"].forEach((g) => {
  const b = document.createElement("button"); b.className = "chip"; b.textContent = g;
  b.addEventListener("click", () => applyGate(g)); wrap.appendChild(b);
});
document.getElementById("reset").addEventListener("click", () => { target = new THREE.Vector3(0, R, 0); if (gateEl) gateEl.textContent = "reset"; });
// ↑↓ runs through a canonical gate tour
const TOUR = ["H","T","H","S","X","H","Z"]; let ti = 0;
setVariantCycler((d) => { ti = (ti + d + TOUR.length) % TOUR.length; applyGate(TOUR[ti]); return "gate " + TOUR[ti]; });

// ---------- boot ----------
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  vec.lerp(target, 0.1); vec.setLength(R);
  stateArrow.setDirection(vec.clone().normalize());
  tip.position.copy(vec);
  // readout: θ from +Y, φ in XZ
  const theta = Math.acos(Math.max(-1, Math.min(1, vec.y / R)));
  const phi = Math.atan2(vec.z, vec.x);
  const c0 = Math.cos(theta/2), c1 = Math.sin(theta/2);
  if (stateEl) stateEl.textContent = `${c0.toFixed(2)}|0⟩ + ${c1.toFixed(2)}e^{i${(phi).toFixed(2)}}|1⟩`;
  controls.update();
  renderer.render(scene, camera);
});
