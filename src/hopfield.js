// hopfield.js — Hopfield associative-memory network ("連想 / Hopfield").
//
// MATH (Hopfield 1982, "Neural networks and physical systems with emergent
// collective computational abilities", PNAS 79:2554): a recurrent network of
// N two-state neurons s_i ∈ {-1,+1}. A handful of patterns ξ^μ are stored in a
// symmetric weight matrix by the Hebbian outer-product rule
//     W_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ ,   W_ii = 0 .
// To recall, start from a corrupted version of a stored pattern and repeatedly
// update neurons by the sign of their local field
//     s_i ← sign( Σ_j W_ij s_j )            (async = one random i at a time,
//                                            sync = all i from the old state).
// Because W is symmetric with zero diagonal, every async update is monotone in
// the Lyapunov "energy"
//     E = -1/2 Σ_ij W_ij s_i s_j ,
// so the state slides downhill and settles in a fixed point — the nearest
// stored memory (reliable for ≤ ~0.14·N patterns; here N=256 so we keep 3).
//
// Visual: the N=G×G neurons are uploaded as a single-channel DataTexture and
// drawn on one fullscreen quad (the fractal.js pattern). +1 cells glow cyan,
// warming to amber as the state settles; -1 cells sit dark magenta on near-
// black. A HUD bar shows the energy falling each frame.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
// Fullscreen-quad pattern: the quad is already in clip space, so this camera's
// transform is unused — it just satisfies renderer.render(scene, camera).
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const G = 16;            // grid side
const N = G * G;         // neurons (256) — capacity ≈ 0.14·N, so store only 3

// --- stored memories (ξ^μ) as 16×16 bitmaps -------------------------------
// '#' = +1 (on), '.' = -1 (off). Three well-separated glyphs recall cleanly.
const GLYPHS = [
  { name: "glyph A", rows: [
    "................",
    "................",
    ".......##.......",
    "......####......",
    "......#..#......",
    ".....##..##.....",
    ".....#....#.....",
    ".....#....#.....",
    ".....######.....",
    ".....#....#.....",
    "....##....##....",
    "....#......#....",
    "...##......##...",
    "................",
    "................",
    "................",
  ]},
  { name: "glyph T", rows: [
    "................",
    "................",
    "...##########...",
    "...##########...",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    "................",
    "................",
    "................",
  ]},
  { name: "cross +", rows: [
    "................",
    "................",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    "..############..",
    "..############..",
    "..############..",
    "..############..",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    ".......##.......",
    "................",
    "................",
  ]},
];

// Decode each glyph to a bipolar Int8Array (±1), length N.
const patterns = GLYPHS.map(({ rows }) => {
  const p = new Int8Array(N);
  for (let y = 0; y < G; y++){
    const line = rows[y] || "";
    for (let x = 0; x < G; x++){
      p[y * G + x] = line.charCodeAt(x) === 35 /* '#' */ ? 1 : -1;
    }
  }
  return p;
});

// --- Hebbian weights: W_ij = (1/N) Σ_μ ξ_i ξ_j, W_ii = 0 ------------------
// Symmetric N×N (256×256 = 65536 floats — cheap). Built once.
const W = new Float32Array(N * N);
(function buildWeights(){
  for (let i = 0; i < N; i++){
    const rowOff = i * N;
    for (let j = i + 1; j < N; j++){
      let sum = 0;
      for (let m = 0; m < patterns.length; m++){
        sum += patterns[m][i] * patterns[m][j];
      }
      const w = sum / N;
      W[rowOff + j] = w;
      W[j * N + i] = w;   // symmetric
    }
    // W_ii stays 0
  }
})();

// --- network state --------------------------------------------------------
const state = new Int8Array(N);     // current neuron states (±1)
const scratch = new Int8Array(N);   // old-state copy for synchronous updates
let target = 0;                     // which memory we are recalling
let sync = false;                   // false = asynchronous, true = synchronous
let curEnergy = 0;

// Local field h_i = Σ_j W_ij s_j  (zero diagonal makes the self-term vanish).
function field(i, src){
  const rowOff = i * N;
  let h = 0;
  for (let j = 0; j < N; j++) h += W[rowOff + j] * src[j];
  return h;
}

// Energy E = -1/2 Σ_ij W_ij s_i s_j (full double sum; N small so it's fine).
function energy(){
  let e = 0;
  for (let i = 0; i < N; i++){
    e += field(i, state) * state[i];
  }
  return -0.5 * e;
}

// Corrupt a stored pattern by flipping a fraction `noise` of its bits.
function corruptAndRecall(){
  const base = patterns[target];
  for (let i = 0; i < N; i++) state[i] = base[i];
  const flips = Math.round(noise * N);
  for (let k = 0; k < flips; k++){
    const idx = (Math.random() * N) | 0;
    state[idx] = -state[idx];
  }
  curEnergy = energy();
  uploadState();
}

// One sweep: async updates `count` random neurons in place; sync updates all
// neurons at once from a frozen copy of the previous state.
function updateNeurons(count){
  if (sync){
    scratch.set(state);
    for (let i = 0; i < N; i++){
      state[i] = field(i, scratch) >= 0 ? 1 : -1;
    }
  } else {
    for (let k = 0; k < count; k++){
      const i = (Math.random() * N) | 0;
      state[i] = field(i, state) >= 0 ? 1 : -1;
    }
  }
  curEnergy = energy();
  uploadState();
}

// --- GPU texture: one byte per neuron, 255 = +1, 0 = -1 -------------------
const cells = new Uint8Array(N);
const tex = new THREE.DataTexture(cells, G, G, THREE.RedFormat, THREE.UnsignedByteType);
tex.needsUpdate = true;

function uploadState(){
  for (let i = 0; i < N; i++) cells[i] = state[i] > 0 ? 255 : 0;
  tex.needsUpdate = true;   // re-upload the neuron grid every change
}

// Display shader: bright cyan (warming to amber when settled) for +1, dark
// magenta for -1, with a soft dark gutter between cells so the lattice reads
// as discrete neurons.
const material = new THREE.ShaderMaterial({
  uniforms: {
    uTex: { value: tex },
    uTime: { value: 0 },
    uGrid: { value: G },
    uHue: { value: 0.5 },       // base +1 hue: 0.5 = cyan
    uSettle: { value: 0 },      // 0 while recalling → 1 once converged (amber)
    uAspect: { value: 1 },      // viewport aspect, to keep the grid square
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTex;
    uniform float uTime, uGrid, uHue, uSettle, uAspect;
    vec3 hsv2rgb(vec3 c){
      vec3 p = abs(fract(c.xxx + vec3(0.0,2.0/3.0,1.0/3.0))*6.0 - 3.0);
      return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
    }
    void main(){
      // letterbox the grid into a centred square regardless of viewport aspect
      vec2 uv = vUv - 0.5;
      if (uAspect > 1.0) uv.x *= uAspect; else uv.y /= uAspect;
      if (abs(uv.x) > 0.5 || abs(uv.y) > 0.5){ gl_FragColor = vec4(0.01,0.01,0.02,1.0); return; }
      uv += 0.5;

      float s = texture2D(uTex, uv).r;          // 1.0 = on (+1), 0.0 = off
      // cell-local coords for a thin dark gutter between neurons
      vec2 g = fract(uv * uGrid);
      float gut = smoothstep(0.0, 0.07, g.x) * smoothstep(0.0, 0.07, g.y)
                * smoothstep(0.0, 0.07, 1.0 - g.x) * smoothstep(0.0, 0.07, 1.0 - g.y);
      // ON: cyan while recalling, warming to amber as it settles.
      vec3 onCol = mix(hsv2rgb(vec3(uHue, 0.85, 1.0)),
                       hsv2rgb(vec3(0.10, 0.95, 1.0)), uSettle);
      // OFF: deep magenta, near-black.
      vec3 offCol = hsv2rgb(vec3(0.88, 0.85, 0.13));
      vec3 col = mix(offCol, onCol, s);
      // subtle pulse so on-cells feel alive
      col *= 0.85 + 0.15 * (0.5 + 0.5 * sin(uTime * 2.0 + uv.x * 6.0));
      col *= mix(0.22, 1.0, gut);               // gutter darkening
      gl_FragColor = vec4(col, 1.0);
    }
  `,
});
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// --- controls -------------------------------------------------------------
let noise = 0.25;   // fraction of bits flipped on corruption (0..1)
let speed = 30;     // neuron updates per second (async sweep size)

// slider 0..60 reads as a percentage → 0..0.6 corruption.
bindRange("noise", (v) => { noise = v / 100; }, (v) => Math.round(v) + "%");
bindRange("speed", (v) => { speed = v; }, (v) => Math.round(v));

const recallBtn = document.getElementById("recall");
if (recallBtn) recallBtn.addEventListener("click", () => corruptAndRecall());

// presets cycled with ↑/↓ — pick which memory to recall and the update mode.
const PRESETS = [
  { target: 0, sync: false, name: "recall A · async" },
  { target: 1, sync: false, name: "recall T · async" },
  { target: 2, sync: false, name: "recall + · async" },
  { target: 0, sync: true,  name: "recall A · sync" },
  { target: 2, sync: true,  name: "recall + · sync" },
];
let presetIdx = 0;
setVariantCycler((dir) => {
  presetIdx = (presetIdx + dir + PRESETS.length) % PRESETS.length;
  const P = PRESETS[presetIdx];
  target = P.target;
  sync = P.sync;
  corruptAndRecall();
  if (modeEl) modeEl.textContent = sync ? "sync" : "async";
  return P.name;
});

// --- HUD: energy value + bar, recall mode --------------------------------
const eValEl = document.getElementById("eval");
const eBarEl = document.getElementById("ebar");
const modeEl = document.getElementById("mode");
// Track the energy range seen so the bar has something to normalise against.
let eMin = 0, eMax = 0, eSeen = false;
function paintEnergy(){
  if (!eSeen){ eMin = eMax = curEnergy; eSeen = true; }
  if (curEnergy < eMin) eMin = curEnergy;
  if (curEnergy > eMax) eMax = curEnergy;
  if (eValEl) eValEl.textContent = curEnergy.toFixed(1);
  if (eBarEl){
    const span = eMax - eMin;
    // bar fills as energy approaches its running minimum (deeper recall).
    const frac = span > 1e-6 ? 1 - (curEnergy - eMin) / span : 1;
    eBarEl.style.width = (5 + frac * 95).toFixed(1) + "%";
  }
}

// --- boot + main loop -----------------------------------------------------
corruptAndRecall();                 // start from a corrupted first memory
if (modeEl) modeEl.textContent = sync ? "sync" : "async";

onResize(renderer, camera, (w, h) => { material.uniforms.uAspect.value = w / h; });
material.uniforms.uAspect.value = innerWidth / innerHeight;

const meter = fpsMeter(document.getElementById("fps"));
let acc = 0;                        // accumulated fractional update credits

loop((dt, elapsed) => {
  meter(dt);
  material.uniforms.uTime.value = elapsed;

  // accumulate fractional updates; the slider throttles the recall speed.
  acc += dt * speed;
  if (sync){
    // one synchronous sweep per (N/8) credits so the slider still gates sweeps.
    const per = Math.max(1, N / 8);
    while (acc >= per){ updateNeurons(0); acc -= per; }
  } else {
    const batch = Math.floor(acc);
    if (batch > 0){ updateNeurons(batch); acc -= batch; }
  }

  // ease the "settled" colour toward amber as the energy nears its minimum.
  const span = eMax - eMin;
  const settled = (eSeen && span > 1e-6) ? 1 - Math.min(1, (curEnergy - eMin) / span) : 1;
  const u = material.uniforms.uSettle;
  u.value += (settled - u.value) * (reducedMotion ? 1 : 0.08);

  paintEnergy();
  renderer.render(scene, camera);
});

liftVeil();

// expose state for the harness diagnostics
window.__diag = () => JSON.stringify({ energy: curEnergy, pattern: target });
