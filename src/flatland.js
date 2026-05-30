// flatland.js — dimensions, after Abbott's "Flatland". A 4D hypercube
// (tesseract) rotates through the XW/YW planes and is projected 4D→3D→screen,
// so you watch the "impossible" cube-within-a-cube turn inside-out. A horizontal
// slice plane shows the 3D cross-section — what a lower-D being would perceive as
// the shape passes through their world. ↑↓ steps the dimension ladder 1→2→3→4.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.013);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(7, 5, 9);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.4;
controls.minDistance = 4; controls.maxDistance = 40;

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.8); key.position.set(6, 10, 6); scene.add(key);
addGrid(scene, { size: 40, divisions: 20, y: -4 });
addSun(scene, { scale: 36, position: [0, 10, -56] });

// ---------- the dimension ladder ----------
// dim 1: a line; 2: a square; 3: a cube; 4: a tesseract. Each is a set of 2^d
// vertices in {-1,1}^d and edges between vertices differing in one coordinate.
function hypercube(d) {
  const verts = [];
  for (let i = 0; i < (1 << d); i++) {
    const v = [];
    for (let b = 0; b < d; b++) v.push((i >> b) & 1 ? 1 : -1);
    while (v.length < 4) v.push(0); // pad to 4D
    verts.push(v);
  }
  const edges = [];
  for (let i = 0; i < verts.length; i++)
    for (let j = i + 1; j < verts.length; j++) {
      let diff = 0;
      for (let k = 0; k < d; k++) if (verts[i][k] !== verts[j][k]) diff++;
      if (diff === 1) edges.push([i, j]);
    }
  return { verts, edges, d };
}

let DIM = 4;
let shape = hypercube(DIM);

// ---------- rotation in 4D + projection ----------
function rot(v, a, b, ang) { // rotate in the (a,b) plane
  const c = Math.cos(ang), s = Math.sin(ang);
  const va = v[a], vb = v[b];
  v[a] = va * c - vb * s; v[b] = va * s + vb * c;
}
function project4to3(v) {     // perspective projection along W
  const wDist = 2.6;
  const k = 1 / (wDist - v[3] * 0.9);
  return new THREE.Vector3(v[0] * k * 2.4, v[1] * k * 2.4, v[2] * k * 2.4);
}

// ---------- geometry: edges as one LineSegments, vertices as points ----------
const edgeGeo = new THREE.BufferGeometry();
let edgePos = new Float32Array(0);
const edgeMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false });
const edges = new THREE.LineSegments(edgeGeo, edgeMat);
scene.add(edges);

const ptGeo = new THREE.BufferGeometry();
let ptPos = new Float32Array(0);
const points = new THREE.Points(ptGeo, new THREE.PointsMaterial({ color: 0xffffff, size: 0.14, sizeAttenuation: true }));
scene.add(points);

// cross-section: where edges cross the y=sliceY plane, drop a marker
const sliceMat = new THREE.MeshBasicMaterial({ color: 0x2be4ff, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false });
const slicePlane = new THREE.Mesh(new THREE.PlaneGeometry(12, 12).rotateX(-Math.PI / 2), sliceMat);
scene.add(slicePlane);
const crossGeo = new THREE.BufferGeometry();
let crossPos = new Float32Array(0);
const cross = new THREE.Points(crossGeo, new THREE.PointsMaterial({ color: 0x62ffb3, size: 0.26, sizeAttenuation: true }));
scene.add(cross);

function allocate() {
  shape = hypercube(DIM);
  edgePos = new Float32Array(shape.edges.length * 2 * 3);
  const ecol = new Float32Array(shape.edges.length * 2 * 3);
  // color edges by which axis they span (x cyan, y magenta, z mint, w amber)
  const AX = [[0.16,0.82,0.96],[1.0,0.18,0.6],[0.55,1.0,0.78],[1.0,0.7,0.3]];
  shape.edges.forEach(([i, j], e) => {
    let axis = 0;
    for (let k = 0; k < 4; k++) if (shape.verts[i][k] !== shape.verts[j][k]) axis = k;
    const c = AX[axis];
    for (const o of [0, 3]) { ecol[e*6+o] = c[0]; ecol[e*6+o+1] = c[1]; ecol[e*6+o+2] = c[2]; }
  });
  edgeGeo.setAttribute("position", new THREE.BufferAttribute(edgePos, 3).setUsage(THREE.DynamicDrawUsage));
  edgeGeo.setAttribute("color", new THREE.BufferAttribute(ecol, 3));
  ptPos = new Float32Array(shape.verts.length * 3);
  ptGeo.setAttribute("position", new THREE.BufferAttribute(ptPos, 3).setUsage(THREE.DynamicDrawUsage));
  crossPos = new Float32Array(shape.edges.length * 3);
  crossGeo.setAttribute("position", new THREE.BufferAttribute(crossPos, 3).setUsage(THREE.DynamicDrawUsage));
  document.getElementById("dimname").textContent = ["point","line","square","cube","tesseract"][DIM];
}

let sliceY = 0;
let sliceDir = 1;
const projected = [];

function update(elapsed) {
  // rotate a working copy in the planes that involve the higher dims
  const pr = [];
  for (const v of shape.verts) {
    const w = v.slice();
    if (DIM >= 2) rot(w, 0, 1, elapsed * 0.3);
    if (DIM >= 3) rot(w, 1, 2, elapsed * 0.24);
    if (DIM >= 4) { rot(w, 0, 3, elapsed * 0.4); rot(w, 2, 3, elapsed * 0.33); }
    pr.push(project4to3(w));
  }
  projected.length = 0; projected.push(...pr);

  for (let i = 0; i < pr.length; i++) { ptPos[i*3] = pr[i].x; ptPos[i*3+1] = pr[i].y; ptPos[i*3+2] = pr[i].z; }
  ptGeo.attributes.position.needsUpdate = true;

  shape.edges.forEach(([i, j], e) => {
    const a = pr[i], b = pr[j];
    edgePos[e*6] = a.x; edgePos[e*6+1] = a.y; edgePos[e*6+2] = a.z;
    edgePos[e*6+3] = b.x; edgePos[e*6+4] = b.y; edgePos[e*6+5] = b.z;
  });
  edgeGeo.attributes.position.needsUpdate = true;

  // cross-section markers: edge intersections with the moving y-plane
  let n = 0;
  shape.edges.forEach(([i, j]) => {
    const a = pr[i], b = pr[j];
    const dy = b.y - a.y;
    if (Math.abs(dy) < 1e-5) return;
    const t = (sliceY - a.y) / dy;
    if (t < 0 || t > 1) return;
    crossPos[n*3] = a.x + (b.x - a.x) * t;
    crossPos[n*3+1] = sliceY;
    crossPos[n*3+2] = a.z + (b.z - a.z) * t;
    n++;
  });
  cross.geometry.setDrawRange(0, n);
  crossGeo.attributes.position.needsUpdate = true;
  slicePlane.position.y = sliceY;
}

// ---------- panel ----------
const wrap = document.getElementById("dims");
const DIMS = [["1D line",1],["2D square",2],["3D cube",3],["4D tesseract",4]];
const chips = DIMS.map(([label, d], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (d === DIM ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { DIM = d; allocate(); chips.forEach((c,k)=>c.classList.toggle("active", k===i)); });
  wrap.appendChild(b);
  return b;
});

let sliceOn = true;
const sliceBtn = document.getElementById("slice");
sliceBtn.classList.toggle("active", sliceOn);
sliceBtn.addEventListener("click", () => {
  sliceOn = !sliceOn;
  sliceBtn.classList.toggle("active", sliceOn);
  slicePlane.visible = sliceOn; cross.visible = sliceOn;
});
let speed = 1;
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");

setVariantCycler((d) => {
  const i = (DIMS.findIndex(x => x[1] === DIM) + d + DIMS.length) % DIMS.length;
  DIM = DIMS[i][1]; allocate();
  chips.forEach((c,k)=>c.classList.toggle("active", k===i));
  return DIMS[i][0];
});

// ---------- boot ----------
allocate();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

let clock = 0;
loop((dt) => {
  meter(dt);
  clock += dt * speed;
  if (sliceOn) { sliceY += sliceDir * dt * 0.7; if (sliceY > 2) sliceDir = -1; else if (sliceY < -2) sliceDir = 1; }
  update(clock);
  controls.update();
  renderer.render(scene, camera);
});
