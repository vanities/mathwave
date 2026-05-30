// dla.js — Diffusion-Limited Aggregation: how lightning, frost, coral, copper
// dendrites and mineral veins grow. Start with one stuck seed. Release a walker
// that random-walks (Brownian motion) until it touches the cluster, where it
// STICKS — then release another. Particles can't reach deep into the cluster
// (they hit the tips first), so the structure grows fractal, feathery branches
// (fractal dimension ≈ 1.71 in 2D). We grow it in 3D, coloring by arrival time
// so you read the growth front. ↑↓ changes the seed geometry.
//
// Ref: Witten & Sander 1981, "Diffusion-Limited Aggregation".

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 8, 30);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
controls.minDistance = 8; controls.maxDistance = 90;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xff2e97, 0.9); key.position.set(12, 18, 10); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.8); rim.position.set(-12, 8, -8); scene.add(rim);
addGrid(scene, { size: 70, divisions: 35, y: -18 });
addSun(scene, { scale: 44, position: [0, 14, -80] });

// ---------- voxel grid for fast neighbor checks ----------
const G = 96;                    // grid resolution
const half = G / 2;
let occ = new Uint8Array(G * G * G);
const key3 = (x, y, z) => (x * G + y) * G + z;
const inb = (x, y, z) => x >= 0 && x < G && y >= 0 && y < G && z >= 0 && z < G;

const MAX = 16000;
let stuck = [];                  // [x,y,z,t] grid coords + arrival order
let maxR = 1;

const geo = new THREE.BoxGeometry(0.42, 0.42, 0.42);
let mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.25 }), MAX);
mesh.frustumCulled = false; mesh.count = 0; scene.add(mesh);
const dummy = new THREE.Object3D(); const color = new THREE.Color();

let seedMode = "point";
function reset() {
  occ = new Uint8Array(G * G * G);
  stuck = [];
  maxR = 1;
  const c = half;
  if (seedMode === "point") { add(c, c, c); }
  else if (seedMode === "line") { for (let x = c - 20; x <= c + 20; x++) add(x, c, c); }
  else if (seedMode === "plane") { for (let x = c - 18; x <= c + 18; x += 2) for (let z = c - 18; z <= c + 18; z += 2) add(x, 4, z); }
  mesh.count = 0;
}
function add(x, y, z) { if (!inb(x, y, z) || occ[key3(x, y, z)]) return; occ[key3(x, y, z)] = 1; stuck.push([x, y, z, stuck.length]); const r = Math.hypot(x - half, y - half, z - half); if (r > maxR) maxR = r; }

function touching(x, y, z) {
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
    if (!dx && !dy && !dz) continue;
    const xx = x + dx, yy = y + dy, zz = z + dz;
    if (inb(xx, yy, zz) && occ[key3(xx, yy, zz)]) return true;
  }
  return false;
}

// grow a batch of walkers per frame
let speed = 400;
function grow(n) {
  for (let w = 0; w < n && stuck.length < MAX; w++) {
    // spawn on a sphere just outside the current cluster
    const spawnR = Math.min(maxR + 6, half - 2);
    let a = Math.random() * Math.PI * 2, e = Math.acos(Math.random() * 2 - 1);
    let x = Math.round(half + spawnR * Math.sin(e) * Math.cos(a));
    let y = Math.round(half + spawnR * Math.cos(e));
    let z = Math.round(half + spawnR * Math.sin(e) * Math.sin(a));
    const killR = spawnR + 10;
    let steps = 0;
    while (steps++ < 4000) {
      if (touching(x, y, z)) { add(x, y, z); break; }
      // random walk
      const d = (Math.random() * 6) | 0;
      if (d === 0) x++; else if (d === 1) x--; else if (d === 2) y++; else if (d === 3) y--; else if (d === 4) z++; else z--;
      // if it wanders too far, respawn (don't waste steps)
      const rr = Math.hypot(x - half, y - half, z - half);
      if (rr > killR || !inb(x, y, z)) { a = Math.random()*Math.PI*2; e = Math.acos(Math.random()*2-1); x = Math.round(half+spawnR*Math.sin(e)*Math.cos(a)); y = Math.round(half+spawnR*Math.cos(e)); z = Math.round(half+spawnR*Math.sin(e)*Math.sin(a)); }
    }
  }
}

function draw() {
  const n = stuck.length;
  for (let i = mesh.count; i < n; i++) {
    const [x, y, z] = stuck[i];
    dummy.position.set((x - half) * 0.42, (y - half) * 0.42, (z - half) * 0.42); dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    const c = ramp(i / MAX * 2.2);   // color by arrival order (growth front)
    color.setRGB(c[0], c[1], c[2]); mesh.setColorAt(i, color);
  }
  mesh.count = n;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------- panel ----------
const wrap = document.getElementById("seeds");
const SEEDS = [["point","point"],["line","line"],["plane","plane"]];
let si = 0;
const chips = SEEDS.map(([label,k],i) => { const b=document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=label; b.addEventListener("click",()=>{ si=i; seedMode=k; reset(); chips.forEach((c,j)=>c.classList.toggle("active",j===i)); }); wrap.appendChild(b); return b; });
bindRange("speed", (v) => { speed = Math.round(v); }, (v) => `${Math.round(v)}/f`);
document.getElementById("reset").addEventListener("click", reset);
setVariantCycler((d) => { si=(si+d+SEEDS.length)%SEEDS.length; seedMode=SEEDS[si][1]; reset(); chips.forEach((c,j)=>c.classList.toggle("active",j===si)); return SEEDS[si][0]; });

// ---------- boot ----------
reset();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const cntEl = document.getElementById("cnt");
loop((dt) => {
  meter(dt);
  if (stuck.length < MAX) grow(speed);
  draw();
  if (cntEl) cntEl.textContent = stuck.length.toLocaleString();
  controls.update();
  renderer.render(scene, camera);
});
