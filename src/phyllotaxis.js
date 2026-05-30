// phyllotaxis.js — how plants pack seeds. Vogel's model (1979): floret n sits at
//   angle θ = n · 137.507°   (the GOLDEN ANGLE, 360°·(2−φ))
//   radius r = c·√n
// Because the golden angle is the "most irrational" turn, no two seeds ever line
// up, so they pack with zero gaps — the sunflower spiral. The interlocking arms
// you see are Fibonacci-numbered (parastichies). Crucially: nudge the angle even
// 0.1° off 137.5° and the packing collapses into ugly radial spokes — proof that
// the golden angle is special. We arrange the seeds on a gentle 3D dome and color
// by index. ↑↓ snaps the divergence angle between revealing presets.
//
// Ref: Vogel 1979; golden angle = 360°(1 − 1/φ) ≈ 137.50776°.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.013);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 12, 14);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.45;
controls.minDistance = 6; controls.maxDistance = 50;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.85));
const key = new THREE.DirectionalLight(0xfff1dd, 0.9); key.position.set(6, 14, 6); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.5); rim.position.set(-8, 6, -6); scene.add(rim);
addGrid(scene, { size: 40, divisions: 20, y: -4 });
addSun(scene, { scale: 30, position: [0, 10, -58] });

const GOLDEN = 137.50776405;     // degrees
let N = 1600;
let angleDeg = GOLDEN;
let dome = 0.5;                  // how domed (0=flat disk, 1=tall)

const geo = new THREE.SphereGeometry(0.14, 10, 8);
let mesh = new THREE.InstancedMesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.4, metalness: 0.2 }), 4000);
mesh.frustumCulled = false; scene.add(mesh);
const dummy = new THREE.Object3D(); const color = new THREE.Color();

function build() {
  const a = angleDeg * Math.PI / 180, c = 0.32;
  for (let i = 0; i < N; i++) {
    const r = c * Math.sqrt(i) * 1.0;
    const th = i * a;
    const x = Math.cos(th) * r, z = Math.sin(th) * r;
    const y = dome * (Math.sqrt(N) * c - r) * 0.6;     // dome: center rises
    const s = 0.5 + 0.5 * (i / N);                      // outer seeds bigger
    dummy.position.set(x, y, z); dummy.scale.setScalar(s); dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    const col = ramp(i / N); color.setRGB(col[0], col[1], col[2]); mesh.setColorAt(i, color);
  }
  mesh.count = N;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------- panel ----------
const PRESETS = [["137.5° golden", GOLDEN], ["137.3° (off)", 137.3], ["90° square", 90], ["137.6° (off)", 137.6], ["120° trimerous", 120], ["99.5° (Fib 1/4)", 99.5]];
const wrap = document.getElementById("angles");
let pi = 0;
const nameEl = document.getElementById("angname");
const chips = PRESETS.map(([label, v], i) => { const b = document.createElement("button"); b.className="chip"+(i===0?" active":""); b.textContent=label; b.addEventListener("click",()=>{ pi=i; angleDeg=v; build(); chips.forEach((c,k)=>c.classList.toggle("active",k===i)); nameEl.textContent=label; }); wrap.appendChild(b); return b; });
bindRange("count", (v) => { N = Math.round(v); build(); }, (v) => `${Math.round(v)}`);
bindRange("fine", (v) => { angleDeg = v; build(); chips.forEach((c)=>c.classList.remove("active")); nameEl.textContent = v.toFixed(2)+"°"; }, (v) => v.toFixed(2)+"°");
bindRange("dome", (v) => { dome = v; build(); }, (v) => v.toFixed(2));
setVariantCycler((d) => { pi=(pi+d+PRESETS.length)%PRESETS.length; angleDeg=PRESETS[pi][1]; build(); chips.forEach((c,k)=>c.classList.toggle("active",k===pi)); nameEl.textContent=PRESETS[pi][0]; const f=document.getElementById("fine"); if(f) f.value=angleDeg; return PRESETS[pi][0]; });

// ---------- boot ----------
build();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); controls.update(); renderer.render(scene, camera); });
