// sorting.js — sorting algorithms as a 3D neon bar landscape.
// Each algorithm is a generator that yields the indices it's currently touching,
// so we can animate it step by step. When a sort finishes, a confirm-sweep runs,
// then it reshuffles and moves to the next algorithm. Loops forever — built to film.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.012);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 26, 52);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.4;
controls.minDistance = 16;
controls.maxDistance = 160;
controls.target.set(0, 8, 0);

scene.add(new THREE.AmbientLight(0x2a1860, 0.9));
const key = new THREE.DirectionalLight(0xff2e97, 1.05); key.position.set(10, 24, 14); scene.add(key);
const rim = new THREE.DirectionalLight(0x2be4ff, 0.9); rim.position.set(-16, 14, -8); scene.add(rim);
const top = new THREE.DirectionalLight(0xfff1dd, 0.45); top.position.set(0, 30, 4); scene.add(top);

addGrid(scene, { size: 100, divisions: 50, y: 0 });
addSun(scene, { scale: 60, position: [0, 26, -90] });

// ---------- the bars ----------
let N = 80;
let arr = [];
const BAR_W = 0.7;
const SPACING = 0.92;
const HSCALE = 0.42;          // value → world height

const geo = new THREE.BoxGeometry(BAR_W, 1, BAR_W).translate(0, 0.5, 0); // sit on floor
const mat = new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.25 });
let mesh = new THREE.InstancedMesh(geo, mat, 200);
mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(mesh);
const dummy = new THREE.Object3D();
const color = new THREE.Color();

function shuffle() {
  arr = Array.from({ length: N }, (_, i) => i + 1);
  for (let i = N - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
}

// ---------- algorithms as generators (yield touched indices) ----------
function* bubble(a) {
  const n = a.length;
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - 1 - i; j++) {
      yield [j, j + 1];
      if (a[j] > a[j + 1]) { const t = a[j]; a[j] = a[j + 1]; a[j + 1] = t; swapped = true; yield [j, j + 1]; }
    }
    if (!swapped) break;
  }
}
function* insertion(a) {
  const n = a.length;
  for (let i = 1; i < n; i++) {
    let j = i;
    while (j > 0) {
      yield [j - 1, j];
      if (a[j - 1] > a[j]) { const t = a[j - 1]; a[j - 1] = a[j]; a[j] = t; yield [j - 1, j]; j--; }
      else break;
    }
  }
}
function* selection(a) {
  const n = a.length;
  for (let i = 0; i < n; i++) {
    let m = i;
    for (let j = i + 1; j < n; j++) { yield [m, j]; if (a[j] < a[m]) m = j; }
    if (m !== i) { const t = a[i]; a[i] = a[m]; a[m] = t; yield [i, m]; }
  }
}
function* shell(a) {
  const n = a.length;
  for (let gap = n >> 1; gap > 0; gap >>= 1) {
    for (let i = gap; i < n; i++) {
      let j = i;
      while (j >= gap) {
        yield [j - gap, j];
        if (a[j - gap] > a[j]) { const t = a[j - gap]; a[j - gap] = a[j]; a[j] = t; yield [j - gap, j]; j -= gap; }
        else break;
      }
    }
  }
}
function* quick(a) {
  const stack = [[0, a.length - 1]];
  while (stack.length) {
    const [lo, hi] = stack.pop();
    if (lo >= hi) continue;
    const pivot = a[hi];
    let i = lo;
    for (let j = lo; j < hi; j++) {
      yield [j, hi];
      if (a[j] < pivot) { const t = a[i]; a[i] = a[j]; a[j] = t; if (i !== j) yield [i, j]; i++; }
    }
    const t = a[i]; a[i] = a[hi]; a[hi] = t; yield [i, hi];
    stack.push([lo, i - 1], [i + 1, hi]);
  }
}
function* heap(a) {
  const n = a.length;
  function* sift(s, e) {
    let root = s;
    while (true) {
      let child = 2 * root + 1;
      if (child > e) break;
      if (child + 1 <= e) { yield [child, child + 1]; if (a[child] < a[child + 1]) child++; }
      yield [root, child];
      if (a[root] < a[child]) { const t = a[root]; a[root] = a[child]; a[child] = t; yield [root, child]; root = child; }
      else break;
    }
  }
  for (let s = (n >> 1) - 1; s >= 0; s--) yield* sift(s, n - 1);
  for (let e = n - 1; e > 0; e--) { const t = a[0]; a[0] = a[e]; a[e] = t; yield [0, e]; yield* sift(0, e - 1); }
}

const ALGOS = [
  ["bubble", bubble], ["insertion", insertion], ["selection", selection],
  ["shell", shell], ["quick", quick], ["heap", heap],
];

// ---------- driver / phases ----------
let algoIdx = 0;
let gen = null;
let touch = [-1, -1];
let phase = "sort";      // sort → sweep → hold → (next)
let sweep = 0;
let holdT = 0;
let autoCycle = true;
let speed = 90;          // generator steps per second

function startAlgo(i) {
  algoIdx = i;
  shuffle();
  gen = ALGOS[i][1](arr);
  phase = "sort"; sweep = 0; holdT = 0; touch = [-1, -1];
  nameEl.textContent = ALGOS[i][0];
  chips.forEach((c, k) => c.classList.toggle("active", k === i));
}

// ---------- render the bars ----------
function draw() {
  const x0 = -((N - 1) * SPACING) / 2;
  for (let i = 0; i < N; i++) {
    const v = arr[i];
    const hh = v * HSCALE;
    dummy.position.set(x0 + i * SPACING, 0, 0);
    dummy.scale.set(1, hh, 1);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);

    let c;
    if (i === touch[0] || i === touch[1]) {
      c = [1, 1, 1];                                   // active comparison → white
    } else if (phase === "sweep" && i < sweep) {
      c = [0.55, 1.0, 0.78];                           // confirmed → mint
    } else if (phase === "hold") {
      const m = ramp(v / N);
      c = [m[0] * 0.4 + 0.33, m[1] * 0.5 + 0.5, m[2] * 0.4 + 0.47]; // all mint-tinted
    } else {
      c = ramp(v / N);
    }
    color.setRGB(c[0], c[1], c[2]);
    mesh.setColorAt(i, color);
  }
  mesh.count = N;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// ---------- panel ----------
const wrap = document.getElementById("algos");
const chips = ALGOS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => startAlgo(i));
  wrap.appendChild(b);
  return b;
});

const nameEl = document.getElementById("algoname");
bindRange("speed", (v) => { speed = v; }, (v) => `${Math.round(v)}/s`);
bindRange("size", (v) => { N = Math.round(v); startAlgo(algoIdx); }, (v) => `${Math.round(v)}`);

const cycleBtn = document.getElementById("cycle");
cycleBtn.classList.toggle("active", autoCycle);
cycleBtn.addEventListener("click", () => { autoCycle = !autoCycle; cycleBtn.classList.toggle("active", autoCycle); });
document.getElementById("shuffle").addEventListener("click", () => startAlgo(algoIdx));

// ↑/↓ cycle the algorithms
setVariantCycler((d) => {
  const n = ALGOS.length;
  startAlgo(((algoIdx + d) % n + n) % n);
  return ALGOS[algoIdx][0];
});

// ---------- boot ----------
startAlgo(0);
draw();
liftVeil();

onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const opsEl = document.getElementById("ops");
let ops = 0;

let acc = 0;
loop((dt) => {
  meter(dt);

  if (phase === "sort") {
    acc += dt * speed;
    let budget = Math.min(acc | 0, 4000);
    acc -= budget;
    while (budget-- > 0) {
      const r = gen.next();
      if (r.done) { phase = "sweep"; sweep = 0; break; }
      touch = r.value; ops++;
    }
  } else if (phase === "sweep") {
    touch = [-1, -1];
    sweep += dt * Math.max(N * 1.6, 60);
    if (sweep >= N) { phase = "hold"; holdT = 0; }
  } else if (phase === "hold") {
    holdT += dt;
    if (holdT > 1.3) {
      ops = 0;
      startAlgo(autoCycle ? (algoIdx + 1) % ALGOS.length : algoIdx);
    }
  }

  opsEl.textContent = ops.toLocaleString();
  draw();
  controls.update();
  renderer.render(scene, camera);
});
