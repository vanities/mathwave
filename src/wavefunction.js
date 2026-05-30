// wavefunction.js — 2D time-dependent Schrödinger equation on the GPU (ping-pong FBOs).
// Solves  i ∂ψ/∂t = (-½∇² + V)ψ  for a complex field ψ = R + iI.
//
// Integrator: Visscher's staggered leapfrog (P.B. Visscher, "A fast explicit
// algorithm for the time-dependent Schrödinger equation", Computers in Physics
// 5, 596 (1991)). Real and imaginary parts live on half-steps so the scheme is
// explicitly stable and norm-conserving:
//     R(t+dt) = R(t) + dt·H·I(t)            [H·ψ = -½∇²ψ + Vψ]
//     I(t+dt) = I(t) - dt·H·R(t+dt)         [uses the freshly-updated R]
// Concretely, with H = -½∇² + V:
//     R_new = R - dt·(-½·lap(I) + V·I)
//     I_new = I + dt·(-½·lap(R_new) + V·R_new)
// lap is the 5-point Laplacian (×4 neighbours − 4·centre) on the texel grid.
//
// State texture channels: R (real), I (imag), V (potential), |ψ|² cached in A.
// Initial ψ is a Gaussian wavepacket  exp(-(r-r0)²/2σ²)·exp(i k·r)  with
// momentum k. Potentials: free space, a double-slit barrier, a tunnelling
// barrier, and a harmonic well. Probability density |ψ|² is mapped to a neon
// cyan→magenta→amber ramp on near-black, faintly tinted by the local phase.
//
// Ping-pong: read from one render target, write the other, then swap — the sim
// never reads and writes the same texture in a single pass.

import * as THREE from "three";
import { makeRenderer, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

// --- simulation grid (square; capped for perf) ---
const SIM = 512;
const SW = SIM, SH = SIM;

const RT_OPTS = {
  type: THREE.HalfFloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
  depthBuffer: false, stencilBuffer: false,
};
let rtA = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS);
let rtB = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS);

const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.PlaneGeometry(2, 2);
const VSHADER = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// --- simulation step shader (one Visscher leapfrog update) ---
const simU = {
  uTex: { value: null },
  uTexel: { value: new THREE.Vector2(1 / SW, 1 / SH) },
  uDt: { value: 0.14 },
};
const simScene = new THREE.Scene();
simScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: simU, vertexShader: VSHADER,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform vec2 uTexel; uniform float uDt;

    // sample (R,I,V) — clamp at the walls so the boundary reflects (ψ=0 outside)
    vec3 samp(vec2 uv){
      if(uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) return vec3(0.0);
      return texture2D(uTex, uv).xyz;
    }
    void main(){
      vec3 c  = texture2D(uTex, vUv).xyz;   // R, I, V at this cell
      float V = c.z;

      // hard wall: keep ψ pinned to zero on the absolute border
      if(vUv.x <= uTexel.x || vUv.x >= 1.0 - uTexel.x ||
         vUv.y <= uTexel.y || vUv.y >= 1.0 - uTexel.y){
        gl_FragColor = vec4(0.0, 0.0, V, 0.0);
        return;
      }

      vec3 l = samp(vUv + vec2(-uTexel.x, 0.0));
      vec3 r = samp(vUv + vec2( uTexel.x, 0.0));
      vec3 d = samp(vUv + vec2( 0.0,-uTexel.y));
      vec3 u = samp(vUv + vec2( 0.0, uTexel.y));

      // 5-point Laplacian (grid spacing folded into uDt's units)
      float lapI = l.y + r.y + d.y + u.y - 4.0 * c.y;
      // R update uses old I:  R_new = R - dt*(-0.5*lapI + V*I)
      float Rn = c.x - uDt * (-0.5 * lapI + V * c.y);

      float lapR = l.x + r.x + d.x + u.x - 4.0 * Rn;   // Laplacian of the NEW R
      // I update uses fresh R: I_new = I + dt*(-0.5*lapR + V*R_new)
      float In = c.y + uDt * (-0.5 * lapR + V * Rn);

      float dens = Rn * Rn + In * In;                  // |ψ|² cached for display
      gl_FragColor = vec4(Rn, In, V, dens);
    }`,
})));

// --- display shader: |ψ|² → neon ramp, phase as a faint tint ---
const dispU = { uTex: { value: null }, uTime: { value: 0 }, uGain: { value: 1.0 } };
const dispScene = new THREE.Scene();
dispScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: dispU, vertexShader: VSHADER,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform float uTime; uniform float uGain;
    // near-black → cyan → magenta → amber (NOT purple-dominant)
    vec3 pal(float t){
      vec3 a = vec3(0.02, 0.03, 0.06);   // near-black
      vec3 b = vec3(0.10, 0.85, 0.98);   // cyan
      vec3 c = vec3(1.00, 0.20, 0.62);   // magenta
      vec3 e = vec3(1.00, 0.74, 0.24);   // amber
      t = clamp(t, 0.0, 1.0) * 3.0;
      if(t < 1.0) return mix(a, b, t);
      if(t < 2.0) return mix(b, c, t - 1.0);
      return mix(c, e, t - 2.0);
    }
    void main(){
      vec4 s = texture2D(uTex, vUv);
      float R = s.x, I = s.y, V = s.z, dens = s.w;
      float d = clamp(dens * 26.0 * uGain, 0.0, 1.0);   // density → ramp position
      float t = pow(d, 0.55);
      vec3 col = pal(t);

      // faint phase tint (hue wobble from arg(ψ)), only where there's amplitude
      float ph = atan(I, R);
      col += 0.10 * d * vec3(cos(ph), cos(ph + 2.094), cos(ph + 4.188));

      col *= 0.30 + 1.15 * d;                            // glow with density

      // ghost the potential walls so barriers/wells are visible on the dark field
      float wall = clamp(V * 0.10, 0.0, 1.0);
      col += wall * vec3(0.06, 0.10, 0.16);

      // gentle palette breathing
      col *= 0.96 + 0.04 * sin(uTime * 0.4);
      gl_FragColor = vec4(col, 1.0);
    }`,
})));

// --- copy shader: blit a CPU seed texture into a render target ---
const copyU = { uTex: { value: null } };
const copyScene = new THREE.Scene();
copyScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: copyU, vertexShader: VSHADER,
  fragmentShader: `precision highp float; varying vec2 vUv; uniform sampler2D uTex;
    void main(){ gl_FragColor = texture2D(uTex, vUv); }`,
})));

// --- potential presets: build V(x,y) on the grid in normalized [0,1] coords ---
const V_WALL = 12.0;     // barrier height (also exposed via the slider)
const PRESETS = [
  ["free packet",   { type: "free",   k: 7.0,  x0: 0.28, y0: 0.50 }],
  ["double slit",   { type: "slit",   k: 8.0,  x0: 0.24, y0: 0.50 }],
  ["tunnel barrier",{ type: "tunnel", k: 8.5,  x0: 0.26, y0: 0.50 }],
  ["harmonic well", { type: "well",   k: 0.0,  x0: 0.34, y0: 0.50 }],
];
let presetIdx = 0;
let barrierH = V_WALL;   // current wall height (slider-driven)

function buildPotential(cfg, out) {
  // out is a Float32Array(SW*SH*4); fills the V channel (index +2)
  const wallH = barrierH;
  for (let y = 0; y < SH; y++) {
    const fy = y / (SH - 1);
    for (let x = 0; x < SW; x++) {
      const fx = x / (SW - 1);
      let V = 0.0;
      if (cfg.type === "slit") {
        // vertical wall near the middle with two gaps
        const wallX = 0.52, halfW = 0.012;
        if (Math.abs(fx - wallX) < halfW) {
          const gapHalf = 0.035, slitSep = 0.12;
          const gapA = 0.5 - slitSep, gapB = 0.5 + slitSep;
          const inA = Math.abs(fy - gapA) < gapHalf;
          const inB = Math.abs(fy - gapB) < gapHalf;
          if (!inA && !inB) V = wallH;
        }
      } else if (cfg.type === "tunnel") {
        // one thin solid wall the packet must tunnel through
        const wallX = 0.55, halfW = 0.010;
        if (Math.abs(fx - wallX) < halfW) V = wallH * 0.55;
      } else if (cfg.type === "well") {
        // 2D harmonic oscillator centred on the grid
        const dx = fx - 0.5, dy = fy - 0.5;
        V = 90.0 * (dx * dx + dy * dy);
      }
      out[(y * SW + x) * 4 + 2] = V;
    }
  }
}

// --- seed: write a Gaussian wavepacket (R,I) + the chosen potential (V) ---
function seed() {
  const cfg = PRESETS[presetIdx][1];
  const data = new Float32Array(SW * SH * 4);
  buildPotential(cfg, data);

  const sigma = 0.055;                 // packet width (normalized units)
  const kx = cfg.k, ky = 0.0;          // momentum, mostly along +x
  const kGrid = 2.0 * Math.PI * 6.0;   // convert k → per-grid phase winding
  const inv2s2 = 1.0 / (2.0 * sigma * sigma);
  // harmonic well looks best with a packet that's offset so it sloshes
  const x0 = cfg.x0, y0 = cfg.y0;

  for (let y = 0; y < SH; y++) {
    const fy = y / (SH - 1);
    for (let x = 0; x < SW; x++) {
      const fx = x / (SW - 1);
      const dx = fx - x0, dy = fy - y0;
      const r2 = dx * dx + dy * dy;
      const env = Math.exp(-r2 * inv2s2);
      const phase = kGrid * (kx * dx + ky * dy);
      const i = (y * SW + x) * 4;
      data[i]     = env * Math.cos(phase);   // R
      data[i + 1] = env * Math.sin(phase);   // I
      // data[i+2] already holds V from buildPotential
      data[i + 3] = data[i] * data[i] + data[i + 1] * data[i + 1]; // |ψ|²
    }
  }

  const tex = new THREE.DataTexture(data, SW, SH, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  copyU.uTex.value = tex;
  for (const rt of [rtA, rtB]) { renderer.setRenderTarget(rt); renderer.render(copyScene, cam); }
  renderer.setRenderTarget(null);
  tex.dispose();
}

// --- panel: preset chips ---
const wrap = document.getElementById("presets");
const nameEl = document.getElementById("presetname");
function applyPreset(i, reseed = true) {
  presetIdx = ((i % PRESETS.length) + PRESETS.length) % PRESETS.length;
  nameEl.textContent = PRESETS[presetIdx][0];
  chips.forEach((c, k) => c.classList.toggle("active", k === presetIdx));
  if (reseed) seed();
}
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => applyPreset(i));
  wrap.appendChild(b);
  return b;
});

// --- sliders ---
// "momentum" scales how fast the packet is launched (re-emits on change)
bindRange("momentum", (v) => {
  for (const [, cfg] of PRESETS) cfg.k = cfg.type === "well" ? 0.0 : v;
  seed();
}, (v) => v.toFixed(1));

// "barrier" sets the wall height for slit/tunnel presets
bindRange("barrier", (v) => {
  barrierH = v;
  const t = PRESETS[presetIdx][1].type;
  if (t === "slit" || t === "tunnel") seed();
}, (v) => Math.round(v));

// sim speed: how many leapfrog steps per frame
let iters = 8;
bindRange("rate", (v) => { iters = Math.round(v); }, (v) => `${Math.round(v)}×`);

let playing = !reducedMotion;
const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.textContent = playing ? "pause" : "play";
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
  playBtn.classList.toggle("active", playing);
});

document.getElementById("reemit").addEventListener("click", seed);

setVariantCycler((d) => {
  applyPreset(presetIdx + d);
  return PRESETS[presetIdx][0];
});

// --- diagnostics for the harness ---
window.__diag = () => JSON.stringify({ preset: PRESETS[presetIdx][0] });

// --- boot ---
applyPreset(0);
liftVeil();
const meter = fpsMeter(document.getElementById("fps"));

let t = 0;
loop((dt) => {
  meter(dt);
  if (playing) {
    for (let i = 0; i < iters; i++) {
      simU.uTex.value = rtA.texture;
      renderer.setRenderTarget(rtB);
      renderer.render(simScene, cam);
      const tmp = rtA; rtA = rtB; rtB = tmp;   // read A, wrote B, now swap
    }
  }
  t += dt;
  dispU.uTex.value = rtA.texture;
  dispU.uTime.value = t;
  renderer.setRenderTarget(null);
  renderer.render(dispScene, cam);
});
