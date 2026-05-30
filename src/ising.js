// ising.js — the 2D Ising model: a lattice of spins (±1) at temperature T.
// Each spin wants to align with its 4 neighbors (ferromagnetism). We run the
// Metropolis Monte-Carlo rule: flip a spin if it lowers energy, or with
// probability e^(−ΔE/T) if it raises it. Below the Curie point Tc = 2/ln(1+√2)
// ≈ 2.269 the lattice spontaneously magnetizes into domains; above it, thermal
// noise wins and it's a fizzing static. Drag T through Tc and watch the phase
// transition happen in real time. Statistical mechanics you can film.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// ---------- lattice ----------
const L = 256;                       // L×L spins
let spin = new Int8Array(L * L);
const TC = 2 / Math.log(1 + Math.SQRT2);   // ≈ 2.269
let T = 2.0;                          // start just below Tc → domains form
let J = 1;

function randomize() { for (let i = 0; i < L * L; i++) spin[i] = Math.random() < 0.5 ? 1 : -1; }
function align() { spin.fill(1); }

// ---------- Metropolis sweep ----------
// precompute the few possible acceptance probabilities for speed
let expTable = {};
function rebuildExp() { expTable = {}; for (const dE of [4, 8]) expTable[dE] = Math.exp(-dE / T); }
function step(flips) {
  for (let n = 0; n < flips; n++) {
    const x = (Math.random() * L) | 0, y = (Math.random() * L) | 0;
    const i = y * L + x;
    const up = spin[((y - 1 + L) % L) * L + x];
    const dn = spin[((y + 1) % L) * L + x];
    const lf = spin[y * L + ((x - 1 + L) % L)];
    const rt = spin[y * L + ((x + 1) % L)];
    const s = spin[i];
    const dE = 2 * J * s * (up + dn + lf + rt);   // energy change if flipped
    if (dE <= 0 || Math.random() < (expTable[dE] || Math.exp(-dE / T))) spin[i] = -s;
  }
}

// ---------- render: spins → a DataTexture on a fullscreen quad ----------
const tex = new THREE.DataTexture(new Uint8Array(L * L * 4), L, L, THREE.RGBAFormat);
tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
const uniforms = { uTex: { value: tex }, uTime: { value: 0 } };
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
  uniforms,
  vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy,0.,1.); }`,
  fragmentShader: `
    precision highp float; varying vec2 vUv; uniform sampler2D uTex; uniform float uTime;
    void main(){
      // letterbox the square lattice into the viewport, keep aspect square-ish
      float s = texture2D(uTex, vUv).r;          // 0 = down, 1 = up
      vec3 upC = vec3(1.0, 0.18, 0.60);          // magenta domains
      vec3 dnC = vec3(0.10, 0.55, 0.95);         // cyan domains
      vec3 col = mix(dnC, upC, s);
      col *= 0.55 + 0.5 * s + 0.0*uTime;
      // subtle neon edge glow between domains via local contrast is skipped for speed
      gl_FragColor = vec4(col, 1.0);
    }`,
})));

const buf = tex.image.data;
function paint() {
  for (let i = 0; i < L * L; i++) {
    const v = spin[i] > 0 ? 255 : 0;
    buf[i*4] = v; buf[i*4+1] = v; buf[i*4+2] = v; buf[i*4+3] = 255;
  }
  tex.needsUpdate = true;
}

function magnetization() { let s = 0; for (let i = 0; i < L * L; i++) s += spin[i]; return s / (L * L); }

// ---------- panel ----------
const tEl = document.getElementById("tval");
const phaseEl = document.getElementById("phase");
const magEl = document.getElementById("mag");
bindRange("temp", (v) => { T = v; rebuildExp(); tEl.textContent = v.toFixed(2); phaseEl.textContent = v < TC ? "ferromagnetic" : "paramagnetic"; }, (v) => v.toFixed(2));
let sweepsPerFrame = 6;
bindRange("rate", (v) => { sweepsPerFrame = Math.round(v); }, (v) => `${Math.round(v)}×`);

let playing = !reducedMotion;
const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "pause" : "play"; playBtn.classList.toggle("active", playing); });
document.getElementById("randomize").addEventListener("click", randomize);

// presets that snap T relative to Tc — ↑↓ steps them
const TPRESETS = [["cold T=1.0", 1.0], ["domains T=2.0", 2.0], ["critical Tc≈2.27", TC], ["hot T=3.2", 3.2]];
let tp = 1;
function applyT(i) { tp = i; T = TPRESETS[i][1]; rebuildExp();
  const el = document.getElementById("temp"); el.value = T; el.dispatchEvent(new Event("input")); }
setVariantCycler((d) => { tp = (tp + d + TPRESETS.length) % TPRESETS.length; applyT(tp); return TPRESETS[tp][0]; });

// ---------- boot ----------
randomize(); rebuildExp(); paint();
liftVeil();
onResize(renderer, cam);
const meter = fpsMeter(document.getElementById("fps"));
const FLIPS = (L * L) >> 0;   // one "sweep" ≈ L² attempts

let acc = 0;
loop((dt) => {
  meter(dt);
  if (playing) { for (let s = 0; s < sweepsPerFrame; s++) step(FLIPS); paint(); }
  if (magEl) magEl.textContent = magnetization().toFixed(3);
  uniforms.uTime.value += dt;
  renderer.render(scene, cam);
});
