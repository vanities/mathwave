// nbody.js — Newtonian N-body gravity. Every body pulls on every other by
// F = G·mᵢmⱼ / (r² + ε²)  (ε = softening, so close passes don't explode), and we
// integrate with velocity-Verlet (leapfrog) — symplectic, so it conserves energy
// far better than Euler and the dance stays stable. Bodies leave neon trails
// colored by speed; mass sets size. Chaos you can watch (and film).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 60, 130);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.35;
controls.minDistance = 20; controls.maxDistance = 600;

scene.add(new THREE.AmbientLight(0x3a2466, 0.8));
const key = new THREE.PointLight(0xfff1dd, 0.6, 0, 0); key.position.set(0, 0, 0); scene.add(key);
addGrid(scene, { size: 300, divisions: 30, y: -80 });
addSun(scene, { scale: 90, position: [0, 40, -260] });

// ---------- bodies ----------
const G = 1.2, EPS2 = 4.0;
let N = 80;
let px, py, pz, vx, vy, vz, m;
const TRAIL = 90;
let mesh, trailLines, trailPos, trailHead;

function alloc() {
  px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
  vx = new Float32Array(N); vy = new Float32Array(N); vz = new Float32Array(N); m = new Float32Array(N);
}

function rand(a, b) { return a + Math.random() * (b - a); }

// preset configs
function seed(kind) {
  alloc();
  if (kind === "disk") {
    // a heavy central star + a rotating disk (mini solar system)
    m[0] = 400; px[0]=py[0]=pz[0]=vx[0]=vy[0]=vz[0]=0;
    for (let i = 1; i < N; i++) {
      const r = rand(18, 90), a = rand(0, Math.PI * 2);
      px[i] = Math.cos(a) * r; pz[i] = Math.sin(a) * r; py[i] = rand(-3, 3);
      const v = Math.sqrt(G * m[0] / r);            // circular orbital velocity
      vx[i] = -Math.sin(a) * v; vz[i] = Math.cos(a) * v; vy[i] = 0;
      m[i] = rand(0.5, 3);
    }
  } else if (kind === "collision") {
    // two clusters thrown at each other
    for (let i = 0; i < N; i++) {
      const side = i < N / 2 ? -1 : 1;
      px[i] = rand(-12, 12) + side * 55; py[i] = rand(-12, 12); pz[i] = rand(-12, 12);
      vx[i] = -side * 6 + rand(-1, 1); vy[i] = rand(-1, 1); vz[i] = rand(-1, 1);
      m[i] = rand(1, 4);
    }
  } else if (kind === "cloud") {
    // cold collapse — a random cloud falling into itself
    for (let i = 0; i < N; i++) {
      const r = rand(0, 70), a = rand(0, Math.PI * 2), e = Math.acos(rand(-1, 1));
      px[i] = r * Math.sin(e) * Math.cos(a); py[i] = r * Math.cos(e); pz[i] = r * Math.sin(e) * Math.sin(a);
      vx[i] = vy[i] = vz[i] = 0; m[i] = rand(1, 3);
    }
  } else { // figure-8-ish three plus extras
    for (let i = 0; i < N; i++) { px[i]=rand(-60,60); py[i]=rand(-30,30); pz[i]=rand(-60,60); vx[i]=rand(-2,2); vy[i]=rand(-2,2); vz[i]=rand(-2,2); m[i]=rand(1,5); }
  }
  rebuildMeshes();
}

function rebuildMeshes() {
  if (mesh) scene.remove(mesh);
  const geo = new THREE.SphereGeometry(1, 12, 10);
  mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.3, emissive: 0x120318 }), N);
  mesh.frustumCulled = false;
  scene.add(mesh);
  if (trailLines) scene.remove(trailLines);
  trailPos = new Float32Array(N * TRAIL * 3);
  trailHead = 0;
  const tgeo = new THREE.BufferGeometry();
  tgeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3).setUsage(THREE.DynamicDrawUsage));
  trailLines = new THREE.LineSegments(tgeo, new THREE.LineBasicMaterial({ color: 0x2be4ff, transparent: true, opacity: 0.28, blending: THREE.AdditiveBlending, depthWrite: false }));
  trailLines.frustumCulled = false;
  scene.add(trailLines);
}

// ---------- physics (velocity-Verlet) ----------
let AX = new Float32Array(0), AY = new Float32Array(0), AZ = new Float32Array(0);
function accel() {
  if (AX.length !== N) { AX = new Float32Array(N); AY = new Float32Array(N); AZ = new Float32Array(N); }
  AX.fill(0); AY.fill(0); AZ.fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = px[j] - px[i], dy = py[j] - py[i], dz = pz[j] - pz[i];
      const r2 = dx*dx + dy*dy + dz*dz + EPS2;
      const inv = 1 / Math.sqrt(r2);
      const f = G * inv * inv * inv;       // G / r³, multiply by mass·component below
      AX[i] += f * m[j] * dx; AY[i] += f * m[j] * dy; AZ[i] += f * m[j] * dz;
      AX[j] -= f * m[i] * dx; AY[j] -= f * m[i] * dy; AZ[j] -= f * m[i] * dz;
    }
  }
}
let speed = 1;
function stepPhysics(dt) {
  const h = dt * 6 * speed;
  accel();
  for (let i = 0; i < N; i++) { vx[i] += 0.5 * AX[i] * h; vy[i] += 0.5 * AY[i] * h; vz[i] += 0.5 * AZ[i] * h;
    px[i] += vx[i] * h; py[i] += vy[i] * h; pz[i] += vz[i] * h; }
  accel();
  for (let i = 0; i < N; i++) { vx[i] += 0.5 * AX[i] * h; vy[i] += 0.5 * AY[i] * h; vz[i] += 0.5 * AZ[i] * h; }
}

// ---------- render ----------
const dummy = new THREE.Object3D(); const color = new THREE.Color();
function draw() {
  for (let i = 0; i < N; i++) {
    dummy.position.set(px[i], py[i], pz[i]);
    const s = 0.6 + Math.cbrt(m[i]) * 0.9;
    dummy.scale.setScalar(s); dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    const spd = Math.hypot(vx[i], vy[i], vz[i]);
    const c = ramp(Math.min(spd / 14, 1));
    color.setRGB(c[0], c[1], c[2]); mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  // trail: store current head, draw segments head→prev
  for (let i = 0; i < N; i++) {
    const base = (i * TRAIL + trailHead) * 3;
    trailPos[base] = px[i]; trailPos[base + 1] = py[i]; trailPos[base + 2] = pz[i];
  }
  // build line segments connecting consecutive trail points per body
  // (cheap: rewrite the whole buffer as pairs each frame is too much; use gl_LINES over ring)
  trailHead = (trailHead + 1) % TRAIL;
}

// We render trails as a separate dynamic geometry: connect ring points in order.
function buildTrailSegments() {
  // create index pairs once
  const idx = [];
  for (let i = 0; i < N; i++) for (let k = 0; k < TRAIL - 1; k++) {
    idx.push(i * TRAIL + k, i * TRAIL + k + 1);
  }
  trailLines.geometry.setIndex(idx);
}

// ---------- panel ----------
const wrap = document.getElementById("systems");
const SEEDS = [["disk", "disk"], ["collision", "collision"], ["cold cloud", "cloud"], ["random", "random"]];
let seedIdx = 0;
const chips = SEEDS.map(([label, k], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { seedIdx = i; seed(k); buildTrailSegments(); chips.forEach((c, j) => c.classList.toggle("active", j === i)); });
  wrap.appendChild(b);
  return b;
});
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
bindRange("bodies", (v) => { N = Math.round(v); seed(SEEDS[seedIdx][1]); buildTrailSegments(); }, (v) => `${Math.round(v)}`);
document.getElementById("reseed").addEventListener("click", () => { seed(SEEDS[seedIdx][1]); buildTrailSegments(); });
setVariantCycler((d) => { seedIdx = (seedIdx + d + SEEDS.length) % SEEDS.length; seed(SEEDS[seedIdx][1]); buildTrailSegments(); chips.forEach((c, j) => c.classList.toggle("active", j === seedIdx)); return SEEDS[seedIdx][0]; });

// ---------- boot ----------
seed("disk");
buildTrailSegments();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const nEl = document.getElementById("nb");

loop((dt) => {
  meter(dt);
  stepPhysics(dt);
  draw();
  trailLines.geometry.attributes.position.needsUpdate = true;
  if (nEl) nEl.textContent = N;
  controls.update();
  renderer.render(scene, camera);
});
