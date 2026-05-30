// spiking.js — a Spiking Neural Network firing in 3D. Each neuron is an
// Izhikevich (2003) point neuron: a 2-D system in membrane potential v and a
// slow recovery variable u,
//   dv/dt = 0.04 v² + 5 v + 140 − u + I
//   du/dt = a (b v − u)
// with the spike-reset rule: when v ≥ +30 mV, emit a SPIKE, then v ← c, u ← u + d.
// The four constants (a,b,c,d) pick the firing class — regular-spiking,
// chattering, fast-spiking, etc. Neurons are wired by sparse random synapses
// (mostly excitatory, some inhibitory); when a neuron spikes it injects its
// synaptic weight as current into every postsynaptic target, so firing
// cascades/avalanches ripple across the net. A little background drive keeps it
// alive. Neon flash on spike, decaying back to a dim idle; faint synapse lines,
// with a bright pulse riding a synapse as it transmits.
//
// Integration: explicit Euler with two half-steps for v at dt ≈ 0.5 ms (the
// standard Izhikevich substepping) so the quadratic term can't blow up.
//
// Ref: E. M. Izhikevich, "Simple Model of Spiking Neurons," IEEE Trans. Neural
// Networks 14(6):1569–1572, 2003.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050010, 0.011);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 24, 96);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.32;
controls.minDistance = 28; controls.maxDistance = 420;

scene.add(new THREE.AmbientLight(0x1a2c44, 0.7));
const key = new THREE.PointLight(0x66e0ff, 0.5, 0, 0); key.position.set(0, 40, 60); scene.add(key);
addGrid(scene, { size: 260, divisions: 26, y: -72 });
addSun(scene, { scale: 80, position: [0, 34, -240] });

// ---------- neuron classes (Izhikevich a,b,c,d) ----------
// excitatory pool draws from RS↔CH, inhibitory from FS, per Izhikevich (2003).
const RS = { a: 0.02, b: 0.2, c: -65, d: 8 };   // regular spiking
const CH = { a: 0.02, b: 0.2, c: -50, d: 2 };   // chattering (bursty)
const FS = { a: 0.1,  b: 0.2, c: -65, d: 2 };   // fast spiking (inhibitory)

// ---------- params ----------
const SPAN = 46;                 // world half-extent for layouts
let N = 700;                     // neuron count
let drive = 5.0;                 // background input current (excitability)
let density = 0.012;             // synapse probability per ordered pair
let topology = "ball";

// per-neuron state
let v, u, a, b, c, d, inhib, I, spiked, flash;
// flat synapse arrays (CSR-ish: post-target + weight, grouped by presynaptic src)
let synStart, synTarget, synWeight;
let synN = 0;

function rand(lo, hi) { return lo + Math.random() * (hi - lo); }

function allocNeurons() {
  v = new Float32Array(N); u = new Float32Array(N);
  a = new Float32Array(N); b = new Float32Array(N);
  c = new Float32Array(N); d = new Float32Array(N);
  inhib = new Uint8Array(N);
  I = new Float32Array(N);
  spiked = new Uint8Array(N);
  flash = new Float32Array(N);    // 0..1 visual excitation, decays each frame
}

// ---------- layouts: where the neurons sit in 3D ----------
let posX, posY, posZ;
function layout(kind) {
  posX = new Float32Array(N); posY = new Float32Array(N); posZ = new Float32Array(N);
  if (kind === "sheet") {
    // a gently rippled square sheet (cortical patch)
    const side = Math.max(2, Math.round(Math.sqrt(N)));
    for (let i = 0; i < N; i++) {
      const gx = i % side, gz = Math.floor(i / side);
      const fx = (gx / (side - 1) - 0.5), fz = (gz / (side - 1) - 0.5);
      posX[i] = fx * SPAN * 2;
      posZ[i] = fz * SPAN * 2;
      posY[i] = Math.sin(fx * 5) * Math.cos(fz * 5) * 6 + rand(-1.5, 1.5);
    }
  } else if (kind === "columns") {
    // layered cortical columns — a grid of vertical stacks
    const cols = Math.max(2, Math.round(Math.sqrt(N / 8)));
    for (let i = 0; i < N; i++) {
      const col = i % (cols * cols);
      const cx = col % cols, cz = Math.floor(col / cols);
      const layer = Math.floor(i / (cols * cols));
      posX[i] = (cx / (cols - 1) - 0.5) * SPAN * 2 + rand(-1.2, 1.2);
      posZ[i] = (cz / (cols - 1) - 0.5) * SPAN * 2 + rand(-1.2, 1.2);
      posY[i] = (layer * 4.2) - 26 + rand(-1, 1);
    }
  } else { // ball — uniform-ish solid sphere
    for (let i = 0; i < N; i++) {
      const r = Math.cbrt(Math.random()) * SPAN;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      posX[i] = r * Math.sin(ph) * Math.cos(th);
      posY[i] = r * Math.cos(ph);
      posZ[i] = r * Math.sin(ph) * Math.sin(th);
    }
  }
}

// ---------- wiring: sparse random synapses, distance-biased ----------
// ~20% of neurons are inhibitory (FS); the rest excitatory (RS/CH mix).
// Nearer neurons are likelier to connect, so cascades have spatial structure.
function assignTypes() {
  for (let i = 0; i < N; i++) {
    const isInhib = Math.random() < 0.2;
    inhib[i] = isInhib ? 1 : 0;
    const re = Math.random();   // per-neuron heterogeneity (Izhikevich's r)
    if (isInhib) {
      a[i] = FS.a + 0.08 * re; b[i] = FS.b - 0.05 * re; c[i] = FS.c; d[i] = FS.d;
    } else {
      const t = re * re;        // bias toward RS, occasional CH
      a[i] = RS.a; b[i] = RS.b;
      c[i] = RS.c + (CH.c - RS.c) * t;     // -65 → -50
      d[i] = RS.d + (CH.d - RS.d) * t;     // 8 → 2
    }
    v[i] = -65 + rand(-3, 3);
    u[i] = b[i] * v[i];
    flash[i] = 0;
  }
}

function buildSynapses() {
  // cap the edge budget so a dense slider can't tank the framerate
  const MAX_SYN = 16000;
  const targets = [];
  const weights = [];
  synStart = new Int32Array(N + 1);
  const reach2 = (SPAN * 0.85) * (SPAN * 0.85);
  let budgetLeft = MAX_SYN;
  for (let i = 0; i < N; i++) {
    synStart[i] = targets.length;
    // expected out-degree from density, but distance-gated below
    for (let j = 0; j < N; j++) {
      if (j === i) continue;
      if (budgetLeft <= 0) break;
      const dx = posX[j] - posX[i], dy = posY[j] - posY[i], dz = posZ[j] - posZ[i];
      const dd = dx * dx + dy * dy + dz * dz;
      const prox = Math.max(0, 1 - dd / reach2);          // 0..1 nearer = bigger
      if (Math.random() < density * (0.25 + 1.5 * prox)) {
        targets.push(j);
        // excitatory positive, inhibitory negative & a bit stronger
        weights.push(inhib[i] ? -rand(3.5, 6.5) : rand(3.0, 7.0));
        budgetLeft--;
      }
    }
  }
  synStart[N] = targets.length;
  synN = targets.length;
  synTarget = Int32Array.from(targets);
  synWeight = Float32Array.from(weights);
}

// ---------- meshes ----------
let mesh, lineSeg, linePos, lineCol, pulseGeoPos, pulseCol, pulseMesh;
const NEURON_GEO = new THREE.IcosahedronGeometry(1, 1);
const idleColor = new THREE.Color(0x16b6cf);   // cyan idle (excitatory)
const idleInhib = new THREE.Color(0x8a3bd6);   // dim violet idle (inhibitory)
const flashExc = new THREE.Color(0xfff4d6);    // white/amber excitatory flash
const flashInh = new THREE.Color(0xff3bc4);    // magenta inhibitory flash

function buildMeshes() {
  if (mesh) { scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); }
  mesh = new THREE.InstancedMesh(
    NEURON_GEO,
    new THREE.MeshBasicMaterial({ toneMapped: false }),
    N
  );
  mesh.frustumCulled = false;
  // seed instance colors so instanceColor exists before first draw
  const tmp = new THREE.Color();
  for (let i = 0; i < N; i++) { tmp.copy(inhib[i] ? idleInhib : idleColor); mesh.setColorAt(i, tmp); }
  scene.add(mesh);

  // synapse lines — one segment (2 verts) per synapse, additively blended
  if (lineSeg) { scene.remove(lineSeg); lineSeg.geometry.dispose(); lineSeg.material.dispose(); }
  linePos = new Float32Array(Math.max(1, synN) * 2 * 3);
  lineCol = new Float32Array(Math.max(1, synN) * 2 * 3);
  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3).setUsage(THREE.DynamicDrawUsage));
  lgeo.setAttribute("color", new THREE.BufferAttribute(lineCol, 3).setUsage(THREE.DynamicDrawUsage));
  lineSeg = new THREE.LineSegments(
    lgeo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false })
  );
  lineSeg.frustumCulled = false;
  scene.add(lineSeg);
  refreshLineGeometry();

  // travelling pulses — a Points cloud, one point per synapse, hidden until firing
  if (pulseMesh) { scene.remove(pulseMesh); pulseMesh.geometry.dispose(); pulseMesh.material.dispose(); }
  pulseGeoPos = new Float32Array(Math.max(1, synN) * 3);
  pulseCol = new Float32Array(Math.max(1, synN) * 3);
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute("position", new THREE.BufferAttribute(pulseGeoPos, 3).setUsage(THREE.DynamicDrawUsage));
  pgeo.setAttribute("color", new THREE.BufferAttribute(pulseCol, 3).setUsage(THREE.DynamicDrawUsage));
  pulseMesh = new THREE.Points(
    pgeo,
    new THREE.PointsMaterial({ size: 1.7, vertexColors: true, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true })
  );
  pulseMesh.frustumCulled = false;
  scene.add(pulseMesh);
}

// map each synapse index → its presynaptic source neuron (for line/pulse origins)
let synSrc;
function refreshLineGeometry() {
  synSrc = new Int32Array(synN);
  for (let i = 0; i < N; i++) {
    for (let s = synStart[i]; s < synStart[i + 1]; s++) synSrc[s] = i;
  }
  for (let s = 0; s < synN; s++) {
    const i = synSrc[s], j = synTarget[s];
    const o = s * 6;
    linePos[o]     = posX[i]; linePos[o + 1] = posY[i]; linePos[o + 2] = posZ[i];
    linePos[o + 3] = posX[j]; linePos[o + 4] = posY[j]; linePos[o + 5] = posZ[j];
  }
  lineSeg.geometry.attributes.position.needsUpdate = true;
}

// transmission animation: each spike spawns a pulse that rides src→dst
// store (synapse index, progress) in a compact ring; cheap and bounded.
const MAX_PULSES = 1400;
let pulseSyn = new Int32Array(MAX_PULSES);
let pulseT = new Float32Array(MAX_PULSES);
let pulseAlive = new Uint8Array(MAX_PULSES);
let pulseHead = 0;
function spawnPulse(synIdx) {
  pulseSyn[pulseHead] = synIdx;
  pulseT[pulseHead] = 0;
  pulseAlive[pulseHead] = 1;
  pulseHead = (pulseHead + 1) % MAX_PULSES;
}

// ---------- the simulation step ----------
let firingThisStep = 0;
function stepSim(dt) {
  // real-time-ish: advance ~ a few ms of model time per rendered frame.
  // two 0.5 ms half-steps per ms of model time keeps v stable.
  const modelMs = Math.min(dt, 1 / 30) * 1000 * 0.5;   // ms of neural time this frame
  const sub = Math.max(1, Math.round(modelMs / 0.5));
  const halfDt = (modelMs / sub) * 0.5;                 // half-step in ms

  firingThisStep = 0;
  // background drive: thalamic noise so the net never goes silent
  for (let i = 0; i < N; i++) {
    I[i] = drive * (inhib[i] ? 0.6 : 1.0) * (Math.random() - 0.5) * 2;
  }

  for (let s = 0; s < sub; s++) {
    // integrate v with two half-steps, u once (Izhikevich's recipe)
    for (let i = 0; i < N; i++) {
      let vi = v[i];
      const ui = u[i], Ii = I[i];
      vi += halfDt * (0.04 * vi * vi + 5 * vi + 140 - ui + Ii);
      vi += halfDt * (0.04 * vi * vi + 5 * vi + 140 - ui + Ii);
      u[i] = ui + (halfDt * 2) * (a[i] * (b[i] * vi - ui));
      v[i] = vi;
    }
    // detect spikes, reset, and inject synaptic current into targets
    for (let i = 0; i < N; i++) {
      if (v[i] >= 30) {
        v[i] = c[i];
        u[i] += d[i];
        spiked[i] = 1;
        flash[i] = 1;
        firingThisStep++;
        const end = synStart[i + 1];
        for (let sy = synStart[i]; sy < end; sy++) {
          I[synTarget[sy]] += synWeight[sy];
          if (Math.random() < 0.5) spawnPulse(sy);   // visualise a subset
        }
      } else {
        spiked[i] = 0;
      }
    }
  }
}

// ---------- render ----------
const dummy = new THREE.Object3D();
const col = new THREE.Color();
function draw(dt) {
  // neurons: scale + color flash, decaying back to idle
  const decay = Math.exp(-dt * 6.5);     // flash half-life ~0.1s
  for (let i = 0; i < N; i++) {
    // a fresh spike pins the flash to 1; otherwise it decays toward dim idle
    const f = spiked[i] ? 1 : flash[i] * decay;
    flash[i] = f;
    const s = 0.85 + f * 2.4;
    dummy.position.set(posX[i], posY[i], posZ[i]);
    dummy.scale.setScalar(s);
    dummy.updateMatrix();
    mesh.setMatrixAt(i, dummy.matrix);
    if (inhib[i]) col.copy(idleInhib).lerp(flashInh, f);
    else col.copy(idleColor).lerp(flashExc, f);
    // lift brightness with flash so it reads as a neon pop
    col.multiplyScalar(0.55 + f * 1.6);
    mesh.setColorAt(i, col);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

  // synapse lines: tint by endpoint excitation so active wires glow
  for (let s = 0; s < synN; s++) {
    const i = synSrc[s], j = synTarget[s];
    const o = s * 6;
    const fi = flash[i], fj = flash[j];
    const inh = synWeight[s] < 0;
    // base faint, brighten with the source's recent firing
    const base = 0.06;
    if (inh) {
      lineCol[o]   = base + fi * 0.9; lineCol[o + 1] = base * 0.4; lineCol[o + 2] = base + fi * 0.7;
      lineCol[o + 3] = base + fj * 0.9; lineCol[o + 4] = base * 0.4; lineCol[o + 5] = base + fj * 0.7;
    } else {
      lineCol[o]   = base + fi * 0.5; lineCol[o + 1] = base + fi * 0.8; lineCol[o + 2] = base + fi * 0.9;
      lineCol[o + 3] = base + fj * 0.5; lineCol[o + 4] = base + fj * 0.8; lineCol[o + 5] = base + fj * 0.9;
    }
  }
  lineSeg.geometry.attributes.color.needsUpdate = true;

  // travelling pulses: advance along their synapse, fade out at the far end
  let pi = 0;
  const speed = dt * 3.2;
  for (let p = 0; p < MAX_PULSES; p++) {
    if (!pulseAlive[p]) continue;
    pulseT[p] += speed;
    if (pulseT[p] >= 1) { pulseAlive[p] = 0; continue; }
    const s = pulseSyn[p];
    const i = synSrc[s], j = synTarget[s];
    const t = pulseT[p];
    const o = pi * 3;
    pulseGeoPos[o]     = posX[i] + (posX[j] - posX[i]) * t;
    pulseGeoPos[o + 1] = posY[i] + (posY[j] - posY[i]) * t;
    pulseGeoPos[o + 2] = posZ[i] + (posZ[j] - posZ[i]) * t;
    const inh = synWeight[s] < 0;
    const fade = 1 - t;
    if (inh) { pulseCol[o] = fade; pulseCol[o + 1] = 0.18 * fade; pulseCol[o + 2] = 0.85 * fade; }
    else { pulseCol[o] = 0.7 * fade; pulseCol[o + 1] = 0.95 * fade; pulseCol[o + 2] = fade; }
    pi++;
    if (pi >= synN) break;
  }
  // park unused points far away (size 0 isn't possible per-point cheaply)
  for (let k = pi; k < synN; k++) {
    const o = k * 3;
    pulseGeoPos[o] = 0; pulseGeoPos[o + 1] = 1e6; pulseGeoPos[o + 2] = 0;
    pulseCol[o] = pulseCol[o + 1] = pulseCol[o + 2] = 0;
  }
  pulseMesh.geometry.attributes.position.needsUpdate = true;
  pulseMesh.geometry.attributes.color.needsUpdate = true;
  pulseMesh.geometry.setDrawRange(0, Math.max(1, synN));
}

// ---------- (re)build the whole network ----------
function rebuild() {
  allocNeurons();
  layout(topology);
  assignTypes();
  buildSynapses();
  buildMeshes();
}

// ---------- panel ----------
const wrap = document.getElementById("topos");
const TOPOS = [["ball", "ball"], ["sheet", "sheet"], ["columns", "columns"]];
let topoIdx = 0;
const chips = TOPOS.map(([label, k], i) => {
  const btn = document.createElement("button");
  btn.className = "chip" + (i === 0 ? " active" : "");
  btn.textContent = label;
  btn.addEventListener("click", () => { topoIdx = i; topology = k; rebuild(); chips.forEach((cc, j) => cc.classList.toggle("active", j === i)); });
  wrap.appendChild(btn);
  return btn;
});

bindRange("drive", (val) => { drive = val; }, (val) => val.toFixed(1));
bindRange("density", (val) => { density = val; rebuild(); }, (val) => `${(val * 100).toFixed(1)}%`);
document.getElementById("reset").addEventListener("click", () => rebuild());

setVariantCycler((dir) => {
  topoIdx = (topoIdx + dir + TOPOS.length) % TOPOS.length;
  topology = TOPOS[topoIdx][1];
  rebuild();
  chips.forEach((cc, j) => cc.classList.toggle("active", j === topoIdx));
  return TOPOS[topoIdx][0];
});

// ---------- boot ----------
rebuild();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const nEl = document.getElementById("nb");
const fireEl = document.getElementById("fire");

window.__diag = () => JSON.stringify({ neurons: N, firing: firingThisStep });

loop((dt) => {
  meter(dt);
  stepSim(dt);
  draw(dt);
  if (nEl) nEl.textContent = N;
  if (fireEl) fireEl.textContent = firingThisStep;
  controls.update();
  renderer.render(scene, camera);
});
