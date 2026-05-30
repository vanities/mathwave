// sorting3d.js — sorting in 3D. A cube of voxels is laid out along a Hilbert-
// like space-filling curve so that "sorted index → 3D position" fills the cube
// continuously. We shuffle the values, then sort with a chosen algorithm; each
// voxel's COLOR is its value, so a finished sort is a smooth 3D color gradient
// flowing along the curve. Answers: yes, sorting a 3D array works — you sort the
// linear ordering the curve imposes. ↑↓ switches algorithm.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(14, 12, 16);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
controls.minDistance = 8; controls.maxDistance = 70;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xff2e97, 1.05); key.position.set(12, 20, 10); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.9); rim.position.set(-14, 8, -10); scene.add(rim);
const top = new THREE.DirectionalLight(0xfff1dd, 0.45); top.position.set(0, 26, 4); scene.add(top);
addGrid(scene, { size: 40, divisions: 20, y: -9 });
addSun(scene, { scale: 24, position: [0, 14, -74] });   // smaller/higher so it doesn't swamp the cube

// ---------- the cube + space-filling curve ----------
// Exploded on purpose: GAP > VOX leaves air between voxels so you can see INTO
// the cube and watch the interior reorder (a solid packed cube hides the sort).
const ORDER = 3;             // 2^3 = 8 per side → 512 cells (watchable + see-through)
const S = 1 << ORDER;
const COUNT = S * S * S;
const GAP = 1.55;            // spacing between cells
const VOX = 0.7;            // voxel size (< GAP → visible gaps)
const CELL = GAP;            // (curve layout uses spacing)

// Space-filling layout: Morton (Z-order) curve — interleave the bits of the
// linear index across x/y/z. Continuity of sorted index along Z-order still
// yields pleasing 3D color gradients, and it's robust and fast to compute.
function mortonToXYZ(m) {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < ORDER; i++) {
    x |= ((m >> (3 * i + 0)) & 1) << i;
    y |= ((m >> (3 * i + 1)) & 1) << i;
    z |= ((m >> (3 * i + 2)) & 1) << i;
  }
  return [x, y, z];
}

// precompute each linear position's world coordinate
const cellXYZ = new Array(COUNT);
const half = (S - 1) / 2;
for (let m = 0; m < COUNT; m++) {
  const [x, y, z] = mortonToXYZ(m);
  cellXYZ[m] = [(x - half) * CELL, (y - half) * CELL, (z - half) * CELL];
}

let arr = new Int32Array(COUNT);
function shuffle() {
  for (let i = 0; i < COUNT; i++) arr[i] = i;
  for (let i = COUNT - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; const t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
}

// ---------- instanced voxels ----------
// Unlit material so each voxel's COLOR is exactly its value, readable from any
// angle (lit interior faces would otherwise go dark and hide the data).
const geo = new THREE.BoxGeometry(VOX, VOX, VOX);
const mat = new THREE.MeshBasicMaterial({});
const mesh = new THREE.InstancedMesh(geo, mat, COUNT);
mesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
mesh.frustumCulled = false;   // bounding sphere is the unit box at origin; instances span ±5 → don't cull
mesh.count = COUNT;
scene.add(mesh);
const dummy = new THREE.Object3D();
const color = new THREE.Color();

// positions are fixed (the curve); only colors change as we sort
for (let m = 0; m < COUNT; m++) {
  dummy.position.set(...cellXYZ[m]);
  dummy.updateMatrix();
  mesh.setMatrixAt(m, dummy.matrix);
}
mesh.instanceMatrix.needsUpdate = true;

let touch = -1;
function paint() {
  for (let m = 0; m < COUNT; m++) {
    // keep values off the darkest end of the ramp so low voxels still glow
    const c = ramp(0.16 + 0.84 * (arr[m] / (COUNT - 1)));
    if (m === touch) color.setRGB(1, 1, 1);
    else color.setRGB(c[0], c[1], c[2]);
    mesh.setColorAt(m, color);
  }
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------- algorithms as generators over the linear array ----------
function* bubble(a) { const n=a.length; for(let i=0;i<n-1;i++){let sw=false; for(let j=0;j<n-1-i;j++){ if(a[j]>a[j+1]){const t=a[j];a[j]=a[j+1];a[j+1]=t;sw=true;} yield j;} if(!sw)break;} }
function* odd_even(a){ const n=a.length; let sorted=false; while(!sorted){ sorted=true; for(let p=0;p<2;p++){ for(let i=p;i<n-1;i+=2){ if(a[i]>a[i+1]){const t=a[i];a[i]=a[i+1];a[i+1]=t;sorted=false;} yield i; } } } }
function* quick(a){ const st=[[0,a.length-1]]; while(st.length){ const [lo,hi]=st.pop(); if(lo>=hi)continue; const pv=a[hi]; let i=lo; for(let j=lo;j<hi;j++){ if(a[j]<pv){const t=a[i];a[i]=a[j];a[j]=t;i++;} yield j;} const t=a[i];a[i]=a[hi];a[hi]=t; yield i; st.push([lo,i-1],[i+1,hi]); } }
function* heap(a){ const n=a.length; function* sift(s,e){ let r=s; while(true){ let c=2*r+1; if(c>e)break; if(c+1<=e&&a[c]<a[c+1])c++; if(a[r]<a[c]){const t=a[r];a[r]=a[c];a[c]=t; yield c; r=c;} else break; } } for(let s=(n>>1)-1;s>=0;s--) yield* sift(s,n-1); for(let e=n-1;e>0;e--){const t=a[0];a[0]=a[e];a[e]=t; yield e; yield* sift(0,e-1);} }
function* shell(a){ const n=a.length; for(let g=n>>1;g>0;g>>=1){ for(let i=g;i<n;i++){ let j=i; while(j>=g&&a[j-g]>a[j]){const t=a[j-g];a[j-g]=a[j];a[j]=t; yield j; j-=g;} } } }

const ALGOS = [["bubble",bubble],["odd-even",odd_even],["shell",shell],["quick",quick],["heap",heap]];
let algoIdx = 3; // quick by default
let gen = null;
let phase = "shuffled", hold = 0;   // start in a visible shuffled hold, THEN sort

function start(i) {
  algoIdx = i; shuffle(); gen = ALGOS[i][1](arr); phase = "shuffled"; hold = 0; touch = -1;
  if (nameEl) nameEl.textContent = ALGOS[i][0];
  if (chips) chips.forEach((c, k) => c.classList.toggle("active", k === i));
  paint();
}

// ---------- panel ----------
const wrap = document.getElementById("algos");
const nameEl = document.getElementById("algoname");
const chips = ALGOS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === algoIdx ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => start(i));
  wrap.appendChild(b);
  return b;
});
let speed = 45;
bindRange("speed", (v) => { speed = v; }, (v) => `${Math.round(v)}/f`);
document.getElementById("shuffle").addEventListener("click", () => start(algoIdx));

setVariantCycler((d) => { const n = ALGOS.length; start(((algoIdx + d) % n + n) % n); return ALGOS[algoIdx][0]; });

// ---------- boot ----------
start(algoIdx);
liftVeil();

// debug hook (read via agent-browser eval)
window.__diag = () => JSON.stringify({
  count: mesh.count, COUNT,
  hasColor: !!mesh.instanceColor,
  frustum: mesh.frustumCulled,
  inScene: scene.children.includes(mesh),
  m0: cellXYZ[0], mLast: cellXYZ[COUNT - 1],
  cam: [camera.position.x.toFixed(1), camera.position.y.toFixed(1), camera.position.z.toFixed(1)],
  bound: mesh.geometry.boundingSphere ? mesh.geometry.boundingSphere.radius : null,
});
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const doneEl = document.getElementById("done");

loop((dt) => {
  meter(dt);
  if (phase === "shuffled") {           // hold on the chaos so you SEE it shuffled
    hold += dt;
    doneEl.textContent = "shuffled";
    if (hold > 1.2) { phase = "sort"; hold = 0; }
  } else if (phase === "sort") {
    let budget = Math.round(speed);
    let r;
    while (budget-- > 0) { r = gen.next(); if (r.done) { phase = "hold"; touch = -1; break; } touch = r.value; }
    paint();
    doneEl.textContent = phase === "hold" ? "sorted ✓" : "sorting…";
  } else {
    hold += dt;
    if (hold > 2.2) start(algoIdx);     // reshuffle and run again
  }
  controls.update();
  renderer.render(scene, camera);
});
