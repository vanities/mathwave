// ripple.js — a 2-D ripple tank: the discrete WAVE EQUATION solved on a GPU
// ping-pong float grid. Two oscillating point sources throw out circular
// wavefronts that overlap into the classic two-source INTERFERENCE pattern —
// bright antinodal lines where crests meet crests, dead nodal lines where a
// crest meets a trough. A barrier with one or two gaps turns the same engine
// into DIFFRACTION (Huygens: each slit re-radiates a circular wave).
//
// Math — leapfrog (Verlet) integration of  ∂²u/∂t² = c²∇²u :
//   u_next = 2u − u_prev + (c·dt/dx)²·∇²u − damping·(u − u_prev)
// with ∇²u from the 5-point Laplacian stencil
//   ∇²u ≈ u(x+1)+u(x−1)+u(y+1)+u(y−1) − 4·u(x,y).
// Stored two-in-one: channel R = u (current), channel G = u_prev (previous);
// each step reads one render target and writes the other, then we swap.
// Stability (CFL): a wave may cross at most ~one cell per step, so we keep
// C = c·dt/dx ≤ 0.7 (clamped); past 1/√2 ≈ 0.707 the leapfrog blows up to NaN.
// Sources are Dirichlet-driven: u := A·sin(ω·t) inside a small disc each step.
//
// Display: signed height → a diverging ramp on near-black (deep teal troughs,
// black at rest, warm amber → magenta crests) plus cheap shaded relief from the
// height gradient, so the wavefronts read as embossed ridges. NOT purple.
//
// Tech mirrors reaction.js: HalfFloat RGBA targets, NearestFilter, an ortho
// camera over a [-1,1] quad, ping-pong FBOs. Pure Three.js r0.169, no build.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

// --- simulation grid (capped for perf; matches viewport aspect) ---
let SW = 0, SH = 0;
const SIM_MAX = 600;
function simSize() {
  const ar = innerWidth / innerHeight;
  if (ar >= 1) { SW = SIM_MAX; SH = Math.max(2, Math.round(SIM_MAX / ar)); }
  else { SH = SIM_MAX; SW = Math.max(2, Math.round(SIM_MAX * ar)); }
}
simSize();

// CLAMP edges (not Repeat) so the boundary behaves like a wall, not a torus.
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

// --- simulation parameters (shared by the step shader) ---
// uMode: 0 two sources · 1 single source · 2 double slit · 3 single slit
// uSrc holds two source positions in UV space (.xy and .zw).
const simU = {
  uTex:     { value: null },
  uTexel:   { value: new THREE.Vector2(1 / SW, 1 / SH) },
  uC2:      { value: 0.45 },          // (c·dt/dx)² — CFL: keep c·dt/dx ≤ 0.7
  uDamp:    { value: 0.0006 },        // gentle loss so reflections don't pile up forever
  uTime:    { value: 0 },
  uOmega:   { value: 2.0 },           // angular frequency of the drivers
  uAmp:     { value: 1.0 },           // source amplitude
  uMode:    { value: 0 },
  uSrc:     { value: new THREE.Vector4(0.5, 0.32, 0.5, 0.68) },
  uSrcR:    { value: 0.012 },         // source disc radius (UV)
  uBarrier: { value: 0.0 },           // 1 → barrier present (slit modes)
  uSlitX:   { value: 0.5 },           // barrier column (UV x)
  uSlitGap: { value: 0.05 },          // half-width of each slit opening (UV)
  uSlitSep: { value: 0.16 },          // half-separation of the two slits (UV)
};

const simScene = new THREE.Scene();
simScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: simU, vertexShader: VSHADER,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform vec2 uTexel;
    uniform float uC2, uDamp, uTime, uOmega, uAmp, uSrcR;
    uniform int uMode;
    uniform vec4 uSrc;
    uniform float uBarrier, uSlitX, uSlitGap, uSlitSep;

    // is this cell part of an opaque wall? (vertical barrier with gaps)
    bool wall(vec2 uv){
      if (uBarrier < 0.5) return false;
      // a vertical band ~3 cells thick at uSlitX
      if (abs(uv.x - uSlitX) > uTexel.x * 1.5) return false;
      if (uMode == 2) {            // double slit: two openings
        float d1 = abs(uv.y - (0.5 - uSlitSep));
        float d2 = abs(uv.y - (0.5 + uSlitSep));
        return !(d1 < uSlitGap || d2 < uSlitGap);
      }
      // single slit: one central opening
      return abs(uv.y - 0.5) > uSlitGap;
    }

    void main(){
      // walls stay clamped at zero height — perfect reflectors.
      if (wall(vUv)) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }

      vec2 s  = texture2D(uTex, vUv).rg;   // r = u (current), g = u_prev
      float u = s.r, up = s.g;

      // 5-point Laplacian; treat neighbouring walls as u = 0 (reflecting).
      float uL = wall(vUv + vec2(-uTexel.x, 0.0)) ? 0.0 : texture2D(uTex, vUv + vec2(-uTexel.x, 0.0)).r;
      float uR = wall(vUv + vec2( uTexel.x, 0.0)) ? 0.0 : texture2D(uTex, vUv + vec2( uTexel.x, 0.0)).r;
      float uD = wall(vUv + vec2( 0.0,-uTexel.y)) ? 0.0 : texture2D(uTex, vUv + vec2( 0.0,-uTexel.y)).r;
      float uU = wall(vUv + vec2( 0.0, uTexel.y)) ? 0.0 : texture2D(uTex, vUv + vec2( 0.0, uTexel.y)).r;
      float lap = uL + uR + uD + uU - 4.0 * u;

      // leapfrog wave update with light damping
      float un = 2.0 * u - up + uC2 * lap - uDamp * (u - up);

      // drive the source disc(s): Dirichlet u = A·sin(ω·t)
      float drive = uAmp * sin(uOmega * uTime);
      float aspect = uTexel.y / uTexel.x;               // correct for non-square cells
      vec2 d0 = (vUv - uSrc.xy); d0.x *= aspect;
      if (length(d0) < uSrcR) un = drive;
      if (uMode == 0 || uMode == 2) {                   // second source for 2-src & double-slit feed
        vec2 d1 = (vUv - uSrc.zw); d1.x *= aspect;
        if (length(d1) < uSrcR) un = drive;
      }

      // hard clamp guards against any stray blow-up → never NaN on screen
      un = clamp(un, -4.0, 4.0);
      gl_FragColor = vec4(un, u, 0.0, 1.0);             // new current, new prev
    }`,
})));

// --- display shader: signed height → diverging ramp + shaded relief ---
const dispU = {
  uTex:     { value: null },
  uTexel:   { value: new THREE.Vector2(1 / SW, 1 / SH) },
  uGain:    { value: 0.9 },
  uBarrier: { value: 0.0 }, uSlitX: { value: 0.5 }, uSlitGap: { value: 0.05 },
  uSlitSep: { value: 0.16 }, uMode: { value: 0 },
};
const dispScene = new THREE.Scene();
dispScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: dispU, vertexShader: VSHADER,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform vec2 uTexel; uniform float uGain;
    uniform float uBarrier, uSlitX, uSlitGap, uSlitSep; uniform int uMode;

    bool wall(vec2 uv){
      if (uBarrier < 0.5) return false;
      if (abs(uv.x - uSlitX) > uTexel.x * 1.5) return false;
      if (uMode == 2) {
        float d1 = abs(uv.y - (0.5 - uSlitSep));
        float d2 = abs(uv.y - (0.5 + uSlitSep));
        return !(d1 < uSlitGap || d2 < uSlitGap);
      }
      return abs(uv.y - 0.5) > uSlitGap;
    }

    // diverging ramp: teal (trough) → near-black (rest) → amber → magenta (crest)
    vec3 ramp(float h){
      vec3 trough = vec3(0.10, 0.78, 0.82);   // deep teal
      vec3 zero   = vec3(0.02, 0.03, 0.06);   // near-black at rest
      vec3 warm   = vec3(1.00, 0.62, 0.18);   // amber
      vec3 crest  = vec3(1.00, 0.20, 0.62);   // magenta
      if (h < 0.0) return mix(zero, trough, clamp(-h, 0.0, 1.0));
      vec3 hot = mix(warm, crest, clamp(h - 1.0, 0.0, 1.0)); // push toward magenta at big crests
      return mix(zero, hot, clamp(h, 0.0, 1.0));
    }

    void main(){
      if (wall(vUv)) { gl_FragColor = vec4(0.06, 0.05, 0.10, 1.0); return; }  // dim wall
      float h = texture2D(uTex, vUv).r * uGain;

      // fake lighting from the height gradient → embossed wavefronts
      float hx = texture2D(uTex, vUv + vec2(uTexel.x, 0.0)).r
               - texture2D(uTex, vUv - vec2(uTexel.x, 0.0)).r;
      float hy = texture2D(uTex, vUv + vec2(0.0, uTexel.y)).r
               - texture2D(uTex, vUv - vec2(0.0, uTexel.y)).r;
      vec3 n = normalize(vec3(-hx * uGain, -hy * uGain, 0.12));
      float lit = clamp(dot(n, normalize(vec3(0.6, 0.7, 0.6))), 0.0, 1.0);

      vec3 col = ramp(h);
      col *= 0.65 + 0.7 * lit;                 // relief shading
      col += 0.12 * abs(h);                    // crests/troughs glow a touch
      gl_FragColor = vec4(col, 1.0);
    }`,
})));

// --- copy shader, to blit a CPU seed texture into a render target ---
const copyU = { uTex: { value: null } };
const copyScene = new THREE.Scene();
copyScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: copyU, vertexShader: VSHADER,
  fragmentShader: `precision highp float; varying vec2 vUv; uniform sampler2D uTex;
    void main(){ gl_FragColor = texture2D(uTex, vUv); }`,
})));

// --- seed: a flat, still tank (u = u_prev = 0 everywhere) ---
function seed() {
  const data = new Float32Array(SW * SH * 4);   // all zeros = at rest; set alpha for cleanliness
  for (let i = 0; i < SW * SH; i++) data[i * 4 + 3] = 1;
  const tex = new THREE.DataTexture(data, SW, SH, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  copyU.uTex.value = tex;
  for (const rt of [rtA, rtB]) { renderer.setRenderTarget(rt); renderer.render(copyScene, cam); }
  renderer.setRenderTarget(null);
  tex.dispose();
}

function reallocate() {
  simSize();
  rtA.setSize(SW, SH); rtB.setSize(SW, SH);
  simU.uTexel.value.set(1 / SW, 1 / SH);
  dispU.uTexel.value.set(1 / SW, 1 / SH);
  seed();
}

// --- presets: name, mode, barrier-on, source layout (Vector4) ---
// mode: 0 two sources · 1 single source · 2 double slit · 3 single slit
const PRESETS = [
  ["two sources",   0, 0, new THREE.Vector4(0.5, 0.32, 0.5, 0.68)],
  ["single source", 1, 0, new THREE.Vector4(0.5, 0.5,  0.5, 0.5 )],
  ["double slit",   2, 1, new THREE.Vector4(0.22, 0.5, 0.22, 0.5)],
  ["single slit",   3, 1, new THREE.Vector4(0.22, 0.5, 0.22, 0.5)],
];
let presetIdx = 0;

function applyPreset(i) {
  const [label, mode, barrier, src] = PRESETS[i];
  simU.uMode.value = mode;
  simU.uBarrier.value = barrier;
  simU.uSrc.value.copy(src);          // read-only preset vector: copy, never alias
  dispU.uMode.value = mode;
  dispU.uBarrier.value = barrier;
  nameEl.textContent = label;
  chips.forEach((c, k) => c.classList.toggle("active", k === i));
  seed();                             // re-drop: still tank, then the drivers refill it
}

// --- panel ---
const wrap = document.getElementById("presets");
const nameEl = document.getElementById("presetname");
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { presetIdx = i; applyPreset(i); });
  wrap.appendChild(b);
  return b;
});

// frequency slider drives uOmega — sets the wavelength of the wavefronts
bindRange("freq", (v) => { simU.uOmega.value = v; }, (v) => v.toFixed(1));
// separation slider moves the two sources apart (and tracks the twin-slit spacing)
bindRange("sep", (v) => {
  const half = v * 0.5;
  const ylo = Math.min(0.85, Math.max(0.15, 0.5 - half));
  const yhi = Math.min(0.85, Math.max(0.15, 0.5 + half));
  simU.uSrc.value.set(0.5, ylo, 0.5, yhi);
  const sep = Math.min(0.3, half);
  simU.uSlitSep.value = sep;
  dispU.uSlitSep.value = sep;
}, (v) => v.toFixed(2));

let playing = !reducedMotion;
const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.textContent = playing ? "pause" : "play";
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
  playBtn.classList.toggle("active", playing);
});

// reset re-drops the tank to a flat, still surface
document.getElementById("reset").addEventListener("click", seed);

// click/tap the canvas to move the primary source there (a fresh drop point)
canvas.addEventListener("pointerdown", (e) => {
  const r = canvas.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  const y = 1.0 - (e.clientY - r.top) / r.height;     // flip: screen y is top-down
  const cur = simU.uSrc.value;                         // keep the secondary source put
  simU.uSrc.value.set(
    Math.min(0.97, Math.max(0.03, x)),
    Math.min(0.97, Math.max(0.03, y)),
    cur.z, cur.w
  );
});

setVariantCycler((d) => {
  presetIdx = (presetIdx + d + PRESETS.length) % PRESETS.length;
  applyPreset(presetIdx);
  return PRESETS[presetIdx][0];
});

// --- diagnostics hook (kiosk/QA reads this) ---
window.__diag = () => JSON.stringify({
  preset: PRESETS[presetIdx][0],
  mode: simU.uMode.value,
  grid: `${SW}x${SH}`,
  cfl: Math.sqrt(simU.uC2.value).toFixed(3),   // c·dt/dx, must stay < ~0.707
  omega: +simU.uOmega.value.toFixed(2),
});

// --- boot ---
applyPreset(0);
liftVeil();
window.addEventListener("resize", reallocate);   // re-fit the grid to the new viewport
const meter = fpsMeter(document.getElementById("fps"));

// run a few sim sub-steps per frame so the wavefronts travel at a lively pace
const SUBSTEPS = 3;
let t = 0;
loop((dt) => {
  meter(dt);
  if (playing) {
    // fixed sim dt keeps the leapfrog (and the CFL bound) stable at any frame rate
    const simDt = 0.18;
    for (let i = 0; i < SUBSTEPS; i++) {
      t += simDt;
      simU.uTime.value = t;
      simU.uTex.value = rtA.texture;         // read A
      renderer.setRenderTarget(rtB);         // write B
      renderer.render(simScene, cam);
      const tmp = rtA; rtA = rtB; rtB = tmp;  // swap → A is newest again
    }
  }
  dispU.uTex.value = rtA.texture;
  renderer.setRenderTarget(null);
  renderer.render(dispScene, cam);
});
