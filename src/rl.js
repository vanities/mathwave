// rl.js — an agent learning to balance a pole (CartPole) in real time, in front
// of you. Room: 学習 / RL CartPole. CPU physics + CPU learning, real 3-D geometry.
//
// PROBLEM — the classic CartPole control task (Barto, Sutton & Anderson 1983,
//   "Neuronlike adaptive elements that can solve difficult learning control
//   problems", IEEE Trans. SMC). A pole is hinged on a cart that slides on a 1-D
//   track; the agent pushes the cart left/right to keep the pole upright.
//
// PHYSICS — standard cartpole dynamics (the Gym / Florian formulation):
//   state s = (x, ẋ, θ, θ̇),  action a ∈ {−1,+1} → force F = a·F_mag
//   constants: g = 9.8, cart mass m_c = 1.0, pole mass m_p = 0.1,
//              pole half-length l = 0.5, F_mag = 10.
//     temp = (F + m_p·l·θ̇²·sinθ) / (m_c + m_p)
//     θ̈   = (g·sinθ − cosθ·temp) / ( l·(4/3 − m_p·cos²θ/(m_c+m_p)) )
//     ẍ   = temp − m_p·l·θ̈·cosθ / (m_c + m_p)
//   Semi-implicit Euler at dt = 0.02 s. An episode ends (and resets) when
//   |θ| > 12° (0.2095 rad) or |x| > 2.4; reward +1 per surviving step, capped
//   at 500 steps ("solved").
//
// LEARNING — SHIPPED: (A) the Cross-Entropy Method (CEM), an evolutionary search
//   over a LINEAR policy  a = sign(w·s + b). Each generation samples a population
//   of weight vectors from a Gaussian N(μ, σ²), scores each by the survival
//   length of a headless rollout, keeps the top "elites", and refits μ, σ to
//   them (with a small noise floor). It reliably balances within a few seconds,
//   no gradients, no divergence. The DISPLAYED cart always plays the current
//   best (μ) policy at real time, so you watch it go from flailing to rock-steady.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05060a, 0.022);

const camera = new THREE.PerspectiveCamera(56, innerWidth / innerHeight, 0.1, 400);
camera.position.set(0, 2.4, 8.5);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.minDistance = 4; controls.maxDistance = 30;
controls.target.set(0, 1.0, 0);

scene.add(new THREE.AmbientLight(0x33405c, 1.3));
const key = new THREE.PointLight(0xffffff, 1.4, 0, 0); key.position.set(5, 9, 7); scene.add(key);
const rim = new THREE.PointLight(0x2be4ff, 0.7, 0, 0); rim.position.set(-6, 4, -4); scene.add(rim);
addGrid(scene, { size: 24, divisions: 24, y: 0 });

// ---------- physics constants (cartpole) ----------
const GRAV = 9.8;
const M_CART = 1.0, M_POLE = 0.1, M_TOTAL = M_CART + M_POLE;
const L_HALF = 0.5;                 // pole half-length (physics)
const POLE_LEN = L_HALF * 2;        // full length, for rendering the rod
const PML = M_POLE * L_HALF;        // pole mass · half length
let F_MAG = 10.0;                   // push force magnitude (variant-tunable)
const DT = 0.02;                    // fixed physics step
const THETA_LIMIT = 12 * Math.PI / 180; // ~0.2095 rad
const X_LIMIT = 2.4;
const MAX_STEPS = 500;              // "solved" cap

// one physics step; mutates st = {x, xd, th, thd}
function physics(st, action) {
  const force = action > 0 ? F_MAG : -F_MAG;
  const ct = Math.cos(st.th), s = Math.sin(st.th);
  const temp = (force + PML * st.thd * st.thd * s) / M_TOTAL;
  const thacc = (GRAV * s - ct * temp) / (L_HALF * (4 / 3 - (M_POLE * ct * ct) / M_TOTAL));
  const xacc = temp - (PML * thacc * ct) / M_TOTAL;
  // semi-implicit Euler (Gym default): velocities first
  st.x += DT * st.xd;  st.xd += DT * xacc;
  st.th += DT * st.thd; st.thd += DT * thacc;
}

function failed(st) { return Math.abs(st.th) > THETA_LIMIT || Math.abs(st.x) > X_LIMIT; }

function randStart() {
  return {
    x:  (Math.random() - 0.5) * 0.1,
    xd: (Math.random() - 0.5) * 0.1,
    th: (Math.random() - 0.5) * 0.1,
    thd:(Math.random() - 0.5) * 0.1,
  };
}

// ---------- linear policy: a = sign(w·s + b) ----------
// weight layout: [w_x, w_xd, w_th, w_thd, b]  (length 5)
const DIM = 5;
function policyAction(w, st) {
  const v = w[0] * st.x + w[1] * st.xd + w[2] * st.th + w[3] * st.thd + w[4];
  return v >= 0 ? 1 : -1;
}

// headless rollout: survival length (steps) for weights w
function rollout(w, maxSteps) {
  const st = randStart();
  let steps = 0;
  while (steps < maxSteps) {
    physics(st, policyAction(w, st));
    steps++;
    if (failed(st)) break;
  }
  return steps;
}

// ---------- Cross-Entropy Method ----------
const POP = 40;            // population per generation
const ELITE = 8;           // top performers kept
const SIGMA0 = 1.0;        // initial spread
const NOISE_FLOOR = 0.02;  // keep exploring; prevents premature collapse
const EVAL_ROLLOUTS = 3;   // average a few noisy rollouts per candidate
const CURVE_MAX = 90;

let mu, sigma;             // distribution over weight vectors
let generation = 0;
let best = 0;              // best survival length seen (steps)
let bestW;                 // best weights ever (for display / __diag)
let recent = 0;           // mean-elite score of the last generation
const rewardCurve = [];   // recent mean-elite scores for the sparkline

function gauss() {        // Box–Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function resetLearning() {
  mu = new Float64Array(DIM);
  sigma = new Float64Array(DIM).fill(SIGMA0);
  bestW = mu.slice();
  generation = 0; best = 0; recent = 0;
  rewardCurve.length = 0;
}

// run one CEM generation (all headless); refits mu, sigma to the elites
function trainGeneration() {
  const cand = [];
  for (let p = 0; p < POP; p++) {
    const w = new Float64Array(DIM);
    for (let d = 0; d < DIM; d++) w[d] = mu[d] + sigma[d] * gauss();
    let score = 0;
    for (let r = 0; r < EVAL_ROLLOUTS; r++) score += rollout(w, MAX_STEPS);
    score /= EVAL_ROLLOUTS;
    cand.push({ w, score });
    if (score > best) { best = score; bestW = w.slice(); }
  }
  cand.sort((a, b) => b.score - a.score);

  const newMu = new Float64Array(DIM);
  const newSigma = new Float64Array(DIM);
  for (let e = 0; e < ELITE; e++) {
    const w = cand[e].w;
    for (let d = 0; d < DIM; d++) newMu[d] += w[d];
  }
  for (let d = 0; d < DIM; d++) newMu[d] /= ELITE;
  for (let e = 0; e < ELITE; e++) {
    const w = cand[e].w;
    for (let d = 0; d < DIM; d++) { const diff = w[d] - newMu[d]; newSigma[d] += diff * diff; }
  }
  for (let d = 0; d < DIM; d++) newSigma[d] = Math.sqrt(newSigma[d] / ELITE) + NOISE_FLOOR;
  mu = newMu; sigma = newSigma;

  generation++;
  recent = 0;
  for (let e = 0; e < ELITE; e++) recent += cand[e].score;
  recent /= ELITE;
  rewardCurve.push(recent);
  if (rewardCurve.length > CURVE_MAX) rewardCurve.shift();
}

// ---------- geometry ----------
const RAIL_LEN = X_LIMIT * 2 + 1.2;
const CART_Y = 0.45;

const rail = new THREE.Mesh(
  new THREE.BoxGeometry(RAIL_LEN, 0.12, 0.5),
  new THREE.MeshStandardMaterial({ color: 0x1d4a5c, emissive: 0x06161c, roughness: 0.6, metalness: 0.2 })
);
rail.position.set(0, CART_Y - 0.26, 0); scene.add(rail);

const stopMat = new THREE.MeshStandardMaterial({ color: 0xff2e97, emissive: 0x3a0a16, roughness: 0.4 });
for (const sx of [-X_LIMIT, X_LIMIT]) {
  const post = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.6, 0.6), stopMat);
  post.position.set(sx, CART_Y, 0); scene.add(post);
}

const cart = new THREE.Mesh(
  new THREE.BoxGeometry(0.7, 0.4, 0.5),
  new THREE.MeshStandardMaterial({ color: 0x2be4ff, emissive: 0x0a2a3a, roughness: 0.35, metalness: 0.15 })
);
cart.position.set(0, CART_Y, 0); scene.add(cart);

// the pole hinges on the cart top; a group lets us rotate cleanly about Z
const poleGroup = new THREE.Group();
poleGroup.position.set(0, CART_Y + 0.2, 0); scene.add(poleGroup);

const poleMat = new THREE.MeshStandardMaterial({ color: 0x62ffb3, emissive: 0x0a3a28, roughness: 0.3, metalness: 0.1 });
let pole = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, POLE_LEN, 14), poleMat);
pole.position.set(0, POLE_LEN / 2, 0);   // base at the hinge, tip up
poleGroup.add(pole);

const bob = new THREE.Mesh(
  new THREE.SphereGeometry(0.12, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0xfff3a8, emissive: 0x3a2a06, roughness: 0.3 })
);
bob.position.set(0, POLE_LEN, 0); poleGroup.add(bob);

const hinge = new THREE.Mesh(
  new THREE.SphereGeometry(0.07, 12, 12),
  new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x223344, roughness: 0.4 })
);
hinge.position.set(0, CART_Y + 0.2, 0); scene.add(hinge);

// ---------- displayed (real-time) episode ----------
// The shown cart plays the CURRENT BEST policy at real time using its own state,
// so we never render a training rollout. It resets on failure.
let demo = randStart();
let demoSteps = 0;
let demoTimescale = 1.0;   // sim-speed multiplier (slider)
let genPerFrame = 1;       // headless CEM generations per frame (slider)

function demoStep() {
  const w = best > 0 ? bestW : mu;   // before any success, use μ
  physics(demo, policyAction(w, demo));
  demoSteps++;
  if (failed(demo) || demoSteps >= MAX_STEPS) { demo = randStart(); demoSteps = 0; }
}

function syncGeometry() {
  // position read-only writes from the sim state
  cart.position.x = demo.x;
  poleGroup.position.x = demo.x;
  hinge.position.x = demo.x;
  // θ from vertical: a positive θ leans the pole toward +x → rotate -θ about Z
  poleGroup.rotation.z = -demo.th;
}

// ---------- panel ----------
// slider maps to both the displayed timescale and the background training budget
bindRange("speed", (v) => {
  demoTimescale = Math.max(0, v);
  genPerFrame = Math.max(0, Math.round(v * 1.5));   // 0..~5 generations/frame
}, (v) => v.toFixed(2) + "×");

document.getElementById("reset").addEventListener("click", () => {
  resetLearning(); demo = randStart(); demoSteps = 0;
});

// ↑/↓ presets — difficulty. The physics half-length stays at the textbook
// L_HALF=0.5 for stable dynamics; the *visual* rod length communicates the
// preset and the push force is the real difficulty lever. Retrain each time.
const VARIANTS = [
  { name: "standard",            len: 1.0, force: 10 },
  { name: "long pole (easier)",  len: 1.6, force: 10 },
  { name: "short pole (harder)", len: 0.6, force: 10 },
  { name: "weak motor (harder)", len: 1.0, force: 6  },
];
let variantIdx = 0;
function applyVariant() {
  const v = VARIANTS[variantIdx];
  F_MAG = v.force;
  pole.geometry.dispose();
  pole.geometry = new THREE.CylinderGeometry(0.05, 0.05, v.len, 14);
  pole.position.set(0, v.len / 2, 0);
  bob.position.set(0, v.len, 0);
  resetLearning(); demo = randStart(); demoSteps = 0;
}
setVariantCycler((d) => {
  variantIdx = (variantIdx + d + VARIANTS.length) % VARIANTS.length;
  applyVariant();
  return VARIANTS[variantIdx].name;
});

// ---------- HUD readouts ----------
const epEl = document.getElementById("ep");
const bestEl = document.getElementById("best");
const recentEl = document.getElementById("recent");
const curveCanvas = document.getElementById("curve");
const cctx = curveCanvas ? curveCanvas.getContext("2d") : null;

function drawCurve() {
  if (!cctx) return;
  const W = curveCanvas.width, H = curveCanvas.height;
  cctx.clearRect(0, 0, W, H);
  cctx.fillStyle = "rgba(8,12,18,0.55)";
  cctx.fillRect(0, 0, W, H);
  if (rewardCurve.length < 2) return;
  cctx.strokeStyle = "#62ffb3";
  cctx.lineWidth = 1.5;
  cctx.beginPath();
  for (let i = 0; i < rewardCurve.length; i++) {
    const px = (i / (CURVE_MAX - 1)) * W;
    const py = H - (rewardCurve[i] / MAX_STEPS) * (H - 2) - 1;
    if (i === 0) cctx.moveTo(px, py); else cctx.lineTo(px, py);
  }
  cctx.stroke();
}

let hudAcc = 0;
function updateHud(dt) {
  hudAcc += dt;
  if (hudAcc < 0.12) return;
  hudAcc = 0;
  if (epEl) epEl.textContent = generation;
  if (bestEl) bestEl.textContent = Math.round(best);
  if (recentEl) recentEl.textContent = Math.round(recent);
  drawCurve();
}

// ---------- boot ----------
resetLearning();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
window.__diag = () => JSON.stringify({ episode: generation, best: Math.round(best) });

let demoAccum = 0;
loop((dt) => {
  meter(dt);
  if (!reducedMotion) {
    // 1) background training — a few fast headless CEM generations per frame
    for (let g = 0; g < genPerFrame; g++) {
      if (best >= MAX_STEPS && generation > 4) break;  // solved; stop churning
      trainGeneration();
    }
    // 2) displayed cart — advance the best policy at ~real time (fixed dt)
    demoAccum += dt * demoTimescale;
    let guard = 0;
    while (demoAccum >= DT && guard < 60) { demoStep(); demoAccum -= DT; guard++; }
  }
  syncGeometry();
  updateHud(dt);
  controls.update();
  renderer.render(scene, camera);
});
