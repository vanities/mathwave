// nca.js — 細胞 · Neural CA — a living, self-healing organism on the GPU.
//
// The honest target is a *Neural* Cellular Automaton in the lineage of
// Mordvintsev, Randazzo, Niklasson & Levin, "Growing Neural Cellular Automata",
// Distill (2020): a grid of cells whose state is updated by a tiny per-cell
// network from a Sobel/identity perception of its neighbours, with a stochastic
// update mask, an alive-mask, and the signature ability to REGENERATE after
// damage. That self-healing only emerges from *trained* weights, which we can't
// fit in-browser — random weights just decay or explode.
//
// So rather than ship something dead, we run **Lenia** (Bert Chan, "Lenia —
// Biology of Artificial Life", 2019; arXiv:1812.05433): a continuous CA in the
// same morphogenetic family that *always* self-organises into lifeforms. Each
// step convolves the field A with a smooth radial multi-ring kernel K, applies a
// Gaussian growth mapping  G(u) = 2·exp(−½((u−μ)/σ)²) − 1,  and integrates
//   A' = clamp( A + dt·G(K∗A), 0, 1 ).
// Lenia grows gliders (the Orbium), colonies, and reefs — and is genuinely
// self-healing: carve a hole and the organism knits itself back together.
//
// Everything below is GPU ping-pong over a float texture: convolve+grow+integrate
// into the back buffer, swap, display as neon life on near-black. Click / drag
// the canvas to DAMAGE the organism and watch it regenerate.
//
// SHIPPED MODEL: Lenia (continuous CA) — chosen over NCA because untrained NCA
// weights don't produce stable, self-healing life, while Lenia reliably does.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.PlaneGeometry(2, 2);
const VS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// ---- simulation grid (square, fixed) ----
const SIM = 512;

// Largest kernel radius we ever sample. The convolution loops a fixed
// [-RMAX, RMAX] window (GLSL ES 1.0 needs a constant loop bound); cells beyond
// the active species radius are skipped by `continue`.
const RMAX = 18;

// ---- ping-pong render targets (float, exactly like reaction.js) ----
// FloatType when EXT_color_buffer_float is present, else HalfFloat; Nearest
// filtering (we sample with integer texel offsets); toroidal wrap so gliders
// that fly off one edge reappear on the other.
function makeRT() {
  let type = THREE.FloatType;
  const gl = renderer.getContext();
  if (!gl.getExtension("EXT_color_buffer_float")) type = THREE.HalfFloatType;
  return new THREE.WebGLRenderTarget(SIM, SIM, {
    type, format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping,
    depthBuffer: false, stencilBuffer: false,
  });
}
let rtA = makeRT();
let rtB = makeRT();

// ---- kernel LUT (baked on the CPU) ----
// Radial kernel K(rn) for normalised radius rn in [0,1], packed into a small
// RGBA float texture (.r holds the weight). RGBA keeps us on the same broadly
// supported path as the rest of the room. K is a sum of smooth shells.
const KERN_N = 256;
function bell(x, m, s) { const d = (x - m) / s; return Math.exp(-0.5 * d * d); }

function buildKernel(peaks, R) {
  const data = new Float32Array(KERN_N * 4);
  const shells = peaks.length;
  for (let i = 0; i < KERN_N; i++) {
    const rn = (i + 0.5) / KERN_N;
    let k = 0;
    for (let b = 0; b < shells; b++) k += peaks[b] * bell(rn * shells, b + 0.5, 0.15);
    data[i * 4 + 0] = k;
    data[i * 4 + 3] = 1.0;
  }
  const tex = new THREE.DataTexture(data, KERN_N, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.minFilter = THREE.NearestFilter; tex.magFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return { tex, R };
}

// ---- compute pass: convolve, grow, integrate ----
const simU = {
  uTex: { value: null },
  uKernel: { value: null },
  uTexel: { value: new THREE.Vector2(1 / SIM, 1 / SIM) },
  uR: { value: 13.0 },        // active kernel radius (pixels)
  uMu: { value: 0.15 },       // growth centre
  uSigma: { value: 0.017 },   // growth width
  uDt: { value: 0.10 },       // integration step (1/T)
};
const simScene = new THREE.Scene();
simScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: simU, vertexShader: VS,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex, uKernel;
    uniform vec2 uTexel;
    uniform float uR, uMu, uSigma, uDt;
    const int RMAX = ${RMAX};
    void main(){
      // kernel-weighted neighbourhood mean  U = (K*A)/sum(K)
      float sum = 0.0, wsum = 0.0;
      for (int dy = -RMAX; dy <= RMAX; dy++){
        for (int dx = -RMAX; dx <= RMAX; dx++){
          float r = length(vec2(float(dx), float(dy)));
          if (r > uR) continue;               // outside this species' radius
          float w = texture2D(uKernel, vec2(r / uR, 0.5)).r;
          vec2 uv = fract(vUv + uTexel * vec2(float(dx), float(dy)));  // toroidal
          sum  += w * texture2D(uTex, uv).r;
          wsum += w;
        }
      }
      float u = (wsum > 0.0) ? sum / wsum : 0.0;

      // Gaussian growth G(u) in [-1, 1]  (x*x, never pow: GLSL pow undefined for x<0)
      float z = (u - uMu) / uSigma;
      float g = 2.0 * exp(-0.5 * z * z) - 1.0;

      float a0 = texture2D(uTex, vUv).r;
      float a1 = clamp(a0 + uDt * g, 0.0, 1.0);
      if (!(a1 == a1)) a1 = 0.0;              // NaN guard -> dead

      // slow maturity trace in G (age of living mass), drives the bloom
      float age = clamp(mix(texture2D(uTex, vUv).g, a1, 0.04), 0.0, 1.0);
      gl_FragColor = vec4(a1, age, 0.0, 1.0);
    }`,
})));

// ---- stamp pass: write a circular hole (damage) or living blob (seed) ----
const stampU = {
  uTex: { value: null },
  uCenter: { value: new THREE.Vector2(0.5, 0.5) }, // uv space
  uRadius: { value: 0.06 },                         // uv space
  uMode: { value: 0.0 },                            // 0 = erase, 1 = seed
};
const stampScene = new THREE.Scene();
stampScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: stampU, vertexShader: VS,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform vec2 uCenter; uniform float uRadius, uMode;
    void main(){
      vec4 s = texture2D(uTex, vUv);
      float dist = length(vUv - uCenter);    // SIM is square -> no aspect fix
      if (uMode < 0.5) {
        // ERASE: soft-edged circular hole -> a regeneration target
        float k = smoothstep(uRadius, uRadius * 0.6, dist);
        s.r *= (1.0 - k); s.g *= (1.0 - k);
      } else {
        // SEED: stamp noisy living mass inside the disc
        if (dist < uRadius) {
          float n = fract(sin(dot(vUv, vec2(91.7, 47.3))) * 43758.5453);
          float fall = 1.0 - smoothstep(0.0, uRadius, dist);
          s.r = clamp(s.r + (0.45 + 0.55 * n) * fall, 0.0, 1.0);
          s.g = max(s.g, s.r * 0.5);
        }
      }
      gl_FragColor = s;
    }`,
})));

// Apply a stamp: read rtA, write rtB, swap (ping-pong, same as the sim step).
function stamp(cx, cy, radius, mode) {
  stampU.uTex.value = rtA.texture;
  stampU.uCenter.value.set(cx, cy);
  stampU.uRadius.value = radius;
  stampU.uMode.value = mode;
  renderer.setRenderTarget(rtB);
  renderer.render(stampScene, cam);
  const t = rtA; rtA = rtB; rtB = t;
}

// ---- display pass: neon life on near-black ----
const dispU = {
  uTex: { value: null },
  uCyan: { value: new THREE.Color(0x18f0e6) },
  uMagenta: { value: new THREE.Color(0xff2bb0) },
  uAmber: { value: new THREE.Color(0xffc23b) },
  uTexel: { value: new THREE.Vector2(1 / SIM, 1 / SIM) },
};
const dispScene = new THREE.Scene();
dispScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: dispU, vertexShader: VS,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform vec3 uCyan, uMagenta, uAmber; uniform vec2 uTexel;
    void main(){
      vec4 s = texture2D(uTex, vUv);
      float a = s.r;     // life value
      float age = s.g;   // maturity trace

      // three-stop neon ramp: teal core -> magenta mantle -> amber rim
      vec3 col = vec3(0.012, 0.018, 0.028);              // near-black
      col = mix(col, uCyan,    smoothstep(0.04, 0.45, a));
      col = mix(col, uMagenta, smoothstep(0.35, 0.78, a));
      col = mix(col, uAmber,   smoothstep(0.72, 0.98, a));

      // bright living edges: gradient magnitude as a thin neon outline
      float ax = abs(texture2D(uTex, vUv + vec2(uTexel.x,0.0)).r
                   - texture2D(uTex, vUv - vec2(uTexel.x,0.0)).r);
      float ay = abs(texture2D(uTex, vUv + vec2(0.0,uTexel.y)).r
                   - texture2D(uTex, vUv - vec2(0.0,uTexel.y)).r);
      col += uAmber * clamp((ax + ay) * 5.0, 0.0, 1.0) * 0.6;

      // soft glow from maturity + a gentle vignette to seat it in the dark
      col += uCyan * age * 0.10;
      col *= mix(0.78, 1.0, smoothstep(0.95, 0.35, length(vUv - 0.5)));
      gl_FragColor = vec4(col, 1.0);
    }`,
})));

// ---- copy pass: blit a CPU seed texture into rtA (like reaction.js) ----
const copyU = { uTex: { value: null } };
const copyScene = new THREE.Scene();
copyScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: copyU, vertexShader: VS,
  fragmentShader: `precision highp float; varying vec2 vUv; uniform sampler2D uTex;
    void main(){ gl_FragColor = texture2D(uTex, vUv); }`,
})));
function blit(data) {
  const tex = new THREE.DataTexture(data, SIM, SIM, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  copyU.uTex.value = tex;
  // write to BOTH buffers so a swap can never reveal stale state
  for (const rt of [rtA, rtB]) { renderer.setRenderTarget(rt); renderer.render(copyScene, cam); }
  renderer.setRenderTarget(null);
  tex.dispose();
}

// ---- species presets (each a tuned Lenia rule) ----
// μ/σ/R/dt sit on or near Bert Chan's published Lenia parameters so the
// patterns reliably self-organise. Orbium is the iconic glider.
const SPECIES = [
  { name: "orbium",   peaks: [1.0],             R: 13, mu: 0.15, sigma: 0.017, dt: 0.10, seed: "glider" },
  { name: "geminium", peaks: [0.5, 1.0, 0.667], R: 18, mu: 0.26, sigma: 0.036, dt: 0.12, seed: "blob"   },
  { name: "hydra",    peaks: [1.0, 0.6, 0.3],   R: 16, mu: 0.20, sigma: 0.028, dt: 0.12, seed: "blob"   },
  { name: "pulsar",   peaks: [1.0, 0.25],       R: 12, mu: 0.16, sigma: 0.020, dt: 0.10, seed: "blob"   },
  { name: "coral",    peaks: [0.3, 1.0],        R: 18, mu: 0.30, sigma: 0.046, dt: 0.14, seed: "blob"   },
];
for (const sp of SPECIES) sp.kernel = buildKernel(sp.peaks, sp.R);

// ---- seeding ----
function emptyField() {
  const d = new Float32Array(SIM * SIM * 4);
  for (let i = 0; i < SIM * SIM; i++) d[i * 4 + 3] = 1.0;  // opaque, dead
  return d;
}

// Asymmetric oriented "glider" cell for the Orbium: an off-centre ring of mass
// with a directional bias so it starts travelling. Built on the CPU because
// Orbium needs a specific structured seed (a noisy blob won't become a glider).
function seedGlider() {
  const data = emptyField();
  const cx = SIM * 0.5, cy = SIM * 0.5, R = 13;
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const r = Math.hypot(dx, dy);
      if (r > R) continue;
      const x = (cx + dx) | 0, y = (cy + dy) | 0;
      const ang = Math.atan2(dy, dx);
      const ring = Math.exp(-((r - R * 0.55) / (R * 0.30)) ** 2);
      const bias = 0.5 + 0.5 * Math.cos(ang - 0.6);
      const v = Math.min(1, ring * (0.55 + 0.7 * bias));
      const i = y * SIM + x;
      data[i * 4 + 0] = v;
      data[i * 4 + 1] = v * 0.5;
    }
  }
  blit(data);
}

// Apply a species: set kernel + growth uniforms, then lay down its seed.
let speciesIdx = 0;
function applySpecies(idx) {
  speciesIdx = (idx + SPECIES.length) % SPECIES.length;
  const sp = SPECIES[speciesIdx];
  simU.uKernel.value = sp.kernel.tex;
  simU.uR.value = sp.kernel.R;
  simU.uMu.value = sp.mu;
  simU.uSigma.value = sp.sigma * growthScale;
  simU.uDt.value = sp.dt;
  if (sp.seed === "glider") {
    seedGlider();
  } else {
    blit(emptyField());                      // clear both buffers first
    stamp(0.50, 0.50, 0.10, 1.0);            // a few overlapping noisy blobs
    stamp(0.42, 0.55, 0.06, 1.0);            // -> the colony self-organises
    stamp(0.58, 0.45, 0.06, 1.0);
  }
  if (nameEl) nameEl.textContent = sp.name;
  return sp.name;
}

// ---- panel / controls ----
const nameEl = document.getElementById("species");

let speed = 4; // sim steps per frame
bindRange("speed", (v) => { speed = Math.round(v); }, (v) => `${Math.round(v)}×`);

// growth-width nudge: scale σ around the species default. growthScale is
// declared ABOVE applySpecies so neither callback hits a temporal-dead-zone.
let growthScale = 1.0;
bindRange("growth", (v) => {
  growthScale = v;
  simU.uSigma.value = SPECIES[speciesIdx].sigma * growthScale;
}, (v) => `${v.toFixed(2)}×`);

setVariantCycler((d) => applySpecies(speciesIdx + d));

const damageBtn = document.getElementById("damage");
if (damageBtn) damageBtn.addEventListener("click", () => {
  // carve a big random hole to show off regeneration
  stamp(0.3 + Math.random() * 0.4, 0.3 + Math.random() * 0.4, 0.14, 0.0);
});
const reseedBtn = document.getElementById("reseed");
if (reseedBtn) reseedBtn.addEventListener("click", () => applySpecies(speciesIdx));

// ---- pointer: click / drag to DAMAGE (erase) the organism ----
// Map client coords to texture UV. Canvas is full-bleed; UV origin is
// bottom-left, so flip Y.
let dragging = false;
function damageAt(e) {
  const rect = canvas.getBoundingClientRect();
  const u = (e.clientX - rect.left) / rect.width;
  const v = 1.0 - (e.clientY - rect.top) / rect.height;
  stamp(u, v, 0.05, 0.0);
}
canvas.addEventListener("pointerdown", (e) => {
  dragging = true;
  if (canvas.setPointerCapture) canvas.setPointerCapture(e.pointerId);
  damageAt(e);
});
canvas.addEventListener("pointermove", (e) => { if (dragging) damageAt(e); });
window.addEventListener("pointerup", () => { dragging = false; });

// ---- resize: SIM grid is fixed; just keep the canvas full-bleed ----
onResize(renderer, null);

// ---- boot ----
applySpecies(0);
liftVeil();
const meter = fpsMeter(document.getElementById("fps"));

// ---- main loop (signature is (dt, elapsed); reducedMotion is a boolean) ----
let stepCount = 0;
const stepsPerFrame = () => (reducedMotion ? Math.min(speed, 2) : speed);
loop((dt) => {
  meter(dt);
  const n = stepsPerFrame();
  for (let s = 0; s < n; s++) {
    simU.uTex.value = rtA.texture;
    renderer.setRenderTarget(rtB);
    renderer.render(simScene, cam);
    const t = rtA; rtA = rtB; rtB = t;
    stepCount++;
  }
  dispU.uTex.value = rtA.texture;
  renderer.setRenderTarget(null);
  renderer.render(dispScene, cam);
});

// ---- diagnostics ----
window.__diag = () => JSON.stringify({ model: "Lenia", step: stepCount, species: SPECIES[speciesIdx].name });
