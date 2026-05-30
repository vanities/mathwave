// boids.js — Craig Reynolds' Boids (1986): emergent flocking from three local
// steering rules, no leader. For each boid, over neighbors within a perception
// radius:
//   SEPARATION — steer away from crowding (short-range repulsion)
//   ALIGNMENT  — steer toward neighbors' average heading
//   COHESION   — steer toward neighbors' center of mass
// The three are weighted, summed into an acceleration, clamped to a max force,
// and velocity is clamped to a max speed. A soft spherical boundary turns them
// back. Colored by heading; faint trails. ↑↓ swaps behavior presets.
//
// Ref: Reynolds, "Flocks, Herds, and Schools" (SIGGRAPH '87).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 800);
camera.position.set(0, 20, 70);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.3;
controls.minDistance = 15; controls.maxDistance = 250;

scene.add(new THREE.AmbientLight(0x3a2466, 0.85));
const key = new THREE.DirectionalLight(0xfff1dd, 0.7); key.position.set(8, 16, 10); scene.add(key);
addGrid(scene, { size: 160, divisions: 32, y: -45 });
addSun(scene, { scale: 70, position: [0, 30, -180] });

// ---------- params ----------
const BOUND = 40;
let N = 400;
let perception = 9, sepW = 1.6, aliW = 1.0, cohW = 1.0, maxSpeed = 0.9, maxForce = 0.04;

let px, py, pz, vx, vy, vz;
function alloc() {
  px = new Float32Array(N); py = new Float32Array(N); pz = new Float32Array(N);
  vx = new Float32Array(N); vy = new Float32Array(N); vz = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    px[i] = (Math.random()*2-1)*BOUND*0.6; py[i] = (Math.random()*2-1)*BOUND*0.6; pz[i] = (Math.random()*2-1)*BOUND*0.6;
    vx[i] = Math.random()*2-1; vy[i] = Math.random()*2-1; vz[i] = Math.random()*2-1;
  }
}

let mesh;
const geoBase = new THREE.ConeGeometry(0.5, 1.6, 6).rotateX(Math.PI / 2); // points along +z
function buildMesh() {
  if (mesh) scene.remove(mesh);
  mesh = new THREE.InstancedMesh(geoBase, new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.2 }), N);
  mesh.frustumCulled = false; scene.add(mesh);
}

// ---------- the flocking step (O(N²) — fine for a few hundred) ----------
function step() {
  for (let i = 0; i < N; i++) {
    let sx=0,sy=0,sz=0, ax=0,ay=0,az=0, cx=0,cy=0,cz=0, n=0, ns=0;
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      const dx = px[j]-px[i], dy = py[j]-py[i], dz = pz[j]-pz[i];
      const d2 = dx*dx+dy*dy+dz*dz;
      if (d2 > perception*perception) continue;
      const d = Math.sqrt(d2) + 1e-6;
      // separation: weight by inverse distance, away from neighbor
      if (d < perception*0.5) { sx -= dx/d; sy -= dy/d; sz -= dz/d; ns++; }
      ax += vx[j]; ay += vy[j]; az += vz[j];   // alignment
      cx += px[j]; cy += py[j]; cz += pz[j];    // cohesion
      n++;
    }
    let fx=0, fy=0, fz=0;
    if (n > 0) {
      // alignment toward avg heading
      ax/=n; ay/=n; az/=n; fx += (ax-vx[i])*aliW; fy += (ay-vy[i])*aliW; fz += (az-vz[i])*aliW;
      // cohesion toward center of mass
      cx=cx/n-px[i]; cy=cy/n-py[i]; cz=cz/n-pz[i]; fx += cx*0.02*cohW; fy += cy*0.02*cohW; fz += cz*0.02*cohW;
    }
    if (ns > 0) { fx += sx*sepW; fy += sy*sepW; fz += sz*sepW; }
    // soft boundary
    const r = Math.sqrt(px[i]*px[i]+py[i]*py[i]+pz[i]*pz[i]);
    if (r > BOUND) { const k = (r-BOUND)*0.01; fx -= px[i]/r*k*10; fy -= py[i]/r*k*10; fz -= pz[i]/r*k*10; }
    // clamp force
    const fm = Math.hypot(fx,fy,fz); if (fm > maxForce) { fx=fx/fm*maxForce; fy=fy/fm*maxForce; fz=fz/fm*maxForce; }
    vx[i]+=fx; vy[i]+=fy; vz[i]+=fz;
    const sp = Math.hypot(vx[i],vy[i],vz[i]); if (sp > maxSpeed) { vx[i]=vx[i]/sp*maxSpeed; vy[i]=vy[i]/sp*maxSpeed; vz[i]=vz[i]/sp*maxSpeed; }
    px[i]+=vx[i]; py[i]+=vy[i]; pz[i]+=vz[i];
  }
}

const dummy = new THREE.Object3D(); const color = new THREE.Color(); const up = new THREE.Vector3(0,1,0);
function draw() {
  for (let i = 0; i < N; i++) {
    dummy.position.set(px[i], py[i], pz[i]);
    const dir = new THREE.Vector3(vx[i], vy[i], vz[i]);
    if (dir.lengthSq() > 1e-6) dummy.lookAt(dummy.position.clone().add(dir));
    dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
    const c = ramp((dir.normalize().y + 1) / 2);
    color.setRGB(c[0], c[1], c[2]); mesh.setColorAt(i, color);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------- presets ----------
const PRESETS = [
  ["flock",   { p: 9,  s: 1.6, a: 1.0, c: 1.0, ms: 0.9 }],
  ["swarm",   { p: 14, s: 0.8, a: 0.4, c: 1.8, ms: 0.7 }],
  ["scatter", { p: 7,  s: 2.6, a: 0.5, c: 0.3, ms: 1.2 }],
  ["school",  { p: 11, s: 1.2, a: 1.8, c: 0.9, ms: 1.0 }],
];
function applyPreset(p) { perception=p.p; sepW=p.s; aliW=p.a; cohW=p.c; maxSpeed=p.ms; }

const wrap = document.getElementById("presets");
let pi = 0;
const chips = PRESETS.map(([label, cfg], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { pi = i; applyPreset(cfg); nameEl.textContent = label; chips.forEach((c,k)=>c.classList.toggle("active", k===i)); });
  wrap.appendChild(b);
  return b;
});
const nameEl = document.getElementById("pname");
bindRange("count", (v) => { N = Math.round(v); alloc(); buildMesh(); }, (v) => `${Math.round(v)}`);
setVariantCycler((d) => { pi = (pi + d + PRESETS.length) % PRESETS.length; applyPreset(PRESETS[pi][1]); nameEl.textContent = PRESETS[pi][0]; chips.forEach((c,k)=>c.classList.toggle("active", k===pi)); return PRESETS[pi][0]; });

// ---------- boot ----------
applyPreset(PRESETS[0][1]);
alloc(); buildMesh();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const nEl = document.getElementById("nb");
loop((dt) => { meter(dt); step(); draw(); if (nEl) nEl.textContent = N; controls.update(); renderer.render(scene, camera); });
