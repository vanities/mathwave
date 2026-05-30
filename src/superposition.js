// superposition.js — Anthropic's "Toy Models of Superposition" result, in 3D.
// A network with only m dimensions can store n > m features by packing them as
// directions that aren't quite orthogonal — and when it does, those directions
// snap into regular GEOMETRY: antipodal pairs, triangles, squares, pentagons,
// the vertices of uniform polytopes. Which shape appears is governed by feature
// SPARSITY: dense → it can only keep m features (orthogonal); sparse → it crams
// them all in as a polygon with a little tolerable interference.
//
// Here: n unit vectors in a 2D/3D "activation space", arranged as the optimal
// polytope for the current count, with the interference matrix WᵀW shown as a
// grid. ↑↓ changes how many features are squeezed in.
//
// Ref: Elhage et al. 2022, transformer-circuits.pub/2022/toy_model.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(52, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 3, 13);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.5;
controls.minDistance = 6; controls.maxDistance = 40;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
const key = new THREE.DirectionalLight(0xfff1dd, 0.7); key.position.set(5, 10, 7); scene.add(key);
addGrid(scene, { size: 30, divisions: 15, y: -5 });
addSun(scene, { scale: 26, position: [0, 9, -50] });

// ---------- feature directions: the optimal arrangement for n features in the plane ----------
// (Toy models: at high sparsity, n features in 2D pack as a regular n-gon.)
const R = 4;
let N = 5;
const featGroup = new THREE.Group(); scene.add(featGroup);
const gridGroup = new THREE.Group(); gridGroup.position.set(7.5, 2.5, 0); scene.add(gridGroup);

let W = [];  // n×2 feature vectors
function buildVectors() {
  W = [];
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2 - Math.PI / 2;
    W.push([Math.cos(a), Math.sin(a)]);   // unit vectors → regular n-gon
  }
}

function makeLabel(text, color = "#f6e9ff") {
  const c = document.createElement("canvas"); c.width = 128; c.height = 64;
  const g = c.getContext("2d");
  g.font = "bold 40px 'VT323', monospace"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillStyle = color; g.shadowColor = color; g.shadowBlur = 8; g.fillText(text, 64, 34);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  sp.scale.set(0.9, 0.45, 1); return sp;
}
const clearGroup = (gr) => { while (gr.children.length) { const c = gr.children.pop(); c.geometry && c.geometry.dispose(); c.material && c.material.dispose && c.material.dispose(); gr.remove(c); } };

const SHAPE_NAMES = { 1: "single", 2: "antipodal pair", 3: "triangle", 4: "square", 5: "pentagon", 6: "hexagon", 7: "heptagon", 8: "octagon" };

function build() {
  buildVectors();
  clearGroup(featGroup);
  // unit circle for reference
  const circPts = [];
  for (let k = 0; k <= 64; k++) { const a = k / 64 * Math.PI * 2; circPts.push(new THREE.Vector3(Math.cos(a) * R, Math.sin(a) * R, 0)); }
  featGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(circPts), new THREE.LineBasicMaterial({ color: 0x4a1f7a, transparent: true, opacity: 0.5 })));
  // each feature as an arrow from origin + a dot + label
  for (let i = 0; i < N; i++) {
    const c = ramp(i / N);
    const end = new THREE.Vector3(W[i][0] * R, W[i][1] * R, 0);
    const arr = new THREE.ArrowHelper(end.clone().normalize(), new THREE.Vector3(), R, new THREE.Color(c[0], c[1], c[2]).getHex(), 0.5, 0.28);
    arr.line.material.linewidth = 2; featGroup.add(arr);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.18, 14, 10), new THREE.MeshStandardMaterial({ color: new THREE.Color(c[0], c[1], c[2]), emissive: new THREE.Color(c[0]*0.3, c[1]*0.3, c[2]*0.3) }));
    dot.position.copy(end); featGroup.add(dot);
    const lab = makeLabel("f" + i, "#c0a3e8"); lab.position.copy(end.clone().multiplyScalar(1.18)); featGroup.add(lab);
  }
  // polygon connecting the tips (the polytope)
  const poly = [];
  for (let i = 0; i <= N; i++) { const w = W[i % N]; poly.push(new THREE.Vector3(w[0] * R, w[1] * R, 0)); }
  featGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(poly), new THREE.LineBasicMaterial({ color: 0x2be4ff, transparent: true, opacity: 0.55 })));

  buildGrid();
  if (nameEl) nameEl.textContent = SHAPE_NAMES[N] || (N + "-gon");
  if (countEl) countEl.textContent = N;
}

// interference matrix WWᵀ as a colored grid (off-diagonal = how much features collide)
let gridMesh;
function buildGrid() {
  clearGroup(gridGroup);
  const cw = 0.6;
  gridMesh = new THREE.InstancedMesh(new THREE.PlaneGeometry(cw * 0.92, cw * 0.92), new THREE.MeshBasicMaterial({ side: THREE.DoubleSide }), N * N);
  gridMesh.frustumCulled = false;
  const d = new THREE.Object3D(), col = new THREE.Color();
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    d.position.set((j - (N-1)/2) * cw, ((N-1)/2 - i) * cw, 0); d.updateMatrix();
    gridMesh.setMatrixAt(i*N+j, d.matrix);
    const dot = W[i][0]*W[j][0] + W[i][1]*W[j][1];   // cosine similarity
    const c = ramp((dot + 1) / 2);
    col.setRGB(c[0], c[1], c[2]); gridMesh.setColorAt(i*N+j, col);
  }
  gridGroup.add(gridMesh);
  const lbl = makeLabel("WWᵀ", "#2be4ff"); lbl.position.set(0, (N/2)*cw + 0.5, 0); lbl.scale.set(1.6, 0.7, 1); gridGroup.add(lbl);
}

// ---------- panel ----------
const wrap = document.getElementById("counts");
const nameEl = document.getElementById("shapename");
const countEl = document.getElementById("nfeat");
const COUNTS = [2,3,4,5,6,7,8];
const chips = COUNTS.map((n, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (n === N ? " active" : "");
  b.textContent = n + "";
  b.addEventListener("click", () => { N = n; build(); chips.forEach((c, k) => c.classList.toggle("active", k === i)); });
  wrap.appendChild(b);
  return b;
});
document.getElementById("reset").addEventListener("click", () => { camera.position.set(0,3,13); controls.target.set(0,0,0); });
setVariantCycler((d) => { const i = Math.max(0, COUNTS.indexOf(N)); const ni = (i + d + COUNTS.length) % COUNTS.length; N = COUNTS[ni]; build(); chips.forEach((c, k) => c.classList.toggle("active", k === ni)); return (SHAPE_NAMES[N] || N + "-gon"); });

// ---------- boot ----------
build();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
loop((dt) => { meter(dt); controls.update(); renderer.render(scene, camera); });
