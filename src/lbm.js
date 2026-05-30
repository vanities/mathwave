// lbm.js — Lattice Fluid: Lattice-Boltzmann D2Q9 on the GPU (ping-pong FBOs).
//
// A mesoscopic fluid solver. Instead of integrating Navier–Stokes directly we
// evolve nine discrete particle-distribution functions f_i streaming along the
// D2Q9 stencil, then relax them toward local equilibrium (BGK collision):
//
//   discrete velocities e_i and weights w_i:
//     i=0 ( 0, 0) w=4/9    (rest)
//     i=1 ( 1, 0) w=1/9    i=2 (0, 1)  i=3 (-1,0) i=4 (0,-1)    (axial,   ×4)
//     i=5 ( 1, 1) w=1/36   i=6 (-1,1)  i=7 (-1,-1) i=8 (1,-1)   (diagonal,×4)
//
//   macroscopics:  ρ = Σ f_i ,   u = (Σ f_i e_i) / ρ
//   equilibrium :  f_i^eq = w_i ρ [ 1 + 3 (e_i·u) + 4.5 (e_i·u)² − 1.5 u·u ]
//   collision   :  f_i ← f_i − (1/τ)(f_i − f_i^eq)            (BGK, single τ)
//   streaming   :  pull — node x reads neighbour at x − e_i
//
// A constant inlet velocity is imposed on the left edge (equilibrium with fixed
// u_x); a solid obstacle uses mid-grid bounce-back; the right edge is a
// zero-gradient outflow. Past a critical Reynolds number the wake destabilises
// into a periodic von Kármán vortex street. ν = c_s²(τ − 1/2), c_s² = 1/3.
//
// Storage: a texture has 4 channels but D2Q9 needs 9 f_i, so we keep THREE
// ping-pong RGBA float targets — A(f0..f3), B(f4..f7), C(f8 in .r). One fused
// compute pass streams+collides all nine (emitting one slab per draw); a display
// shader maps vorticity (curl of u) to a cyan→white→magenta ramp on near-black.

import * as THREE from "three";
import { makeRenderer, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

// ---- simulation grid (fixed; wide so the wake has room downstream) ----
const SIM_W = 512;
const SIM_H = 256;

const RT_OPTS = {
  type: THREE.FloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping,
  depthBuffer: false, stencilBuffer: false,
};
// three distribution targets, each ping-ponged
let aA = new THREE.WebGLRenderTarget(SIM_W, SIM_H, RT_OPTS); // f0..f3
let aB = new THREE.WebGLRenderTarget(SIM_W, SIM_H, RT_OPTS);
let bA = new THREE.WebGLRenderTarget(SIM_W, SIM_H, RT_OPTS); // f4..f7
let bB = new THREE.WebGLRenderTarget(SIM_W, SIM_H, RT_OPTS);
let cA = new THREE.WebGLRenderTarget(SIM_W, SIM_H, RT_OPTS); // f8 in .r
let cB = new THREE.WebGLRenderTarget(SIM_W, SIM_H, RT_OPTS);

const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.PlaneGeometry(2, 2);
const VSHADER = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// ---- obstacle (cylinder) parameters ----
// Centre in normalised UV; radius in UV.x units. Shape cycles via ↑↓ presets.
const OBS = { cx: 0.22, cy: 0.5, r: 0.055, shape: 0 }; // 0 circle, 1 square, 2 wedge

// shared GLSL: D2Q9 lattice + equilibrium + the obstacle test.
const LATTICE = `
  precision highp float;
  uniform vec2 uTexel;      // 1/SIM in uv
  uniform vec2 uSim;        // (SIM_W, SIM_H)
  uniform float uTau;       // relaxation time
  uniform float uIn;        // inlet x-velocity (lattice units)
  uniform vec3 uObs;        // cx, cy, r (uv)
  uniform float uShape;     // 0 circle, 1 square, 2 wedge

  const float w0 = 4.0/9.0;
  const float ws = 1.0/9.0;   // axial
  const float wd = 1.0/36.0;  // diagonal

  vec2 ei(int i){
    if(i==0) return vec2( 0.0, 0.0);
    if(i==1) return vec2( 1.0, 0.0);
    if(i==2) return vec2( 0.0, 1.0);
    if(i==3) return vec2(-1.0, 0.0);
    if(i==4) return vec2( 0.0,-1.0);
    if(i==5) return vec2( 1.0, 1.0);
    if(i==6) return vec2(-1.0, 1.0);
    if(i==7) return vec2(-1.0,-1.0);
    return vec2( 1.0,-1.0); // i==8
  }
  float wi(int i){
    if(i==0) return w0;
    if(i<5)  return ws;
    return wd;
  }
  float feq(int i, float rho, vec2 u){
    vec2 e = ei(i);
    float eu = dot(e, u);
    float uu = dot(u, u);
    return wi(i) * rho * (1.0 + 3.0*eu + 4.5*eu*eu - 1.5*uu);
  }

  // is this uv inside the solid obstacle? aspect-corrected so a circle is round.
  bool isSolid(vec2 uv){
    float ar = uSim.x / uSim.y;                       // width/height
    vec2 d = vec2(uv.x - uObs.x, (uv.y - uObs.y) / ar); // -> x-units
    if(uShape < 0.5){
      return dot(d, d) < uObs.z * uObs.z;             // circle
    } else if(uShape < 1.5){
      return abs(d.x) < uObs.z && abs(d.y) < uObs.z;  // square
    } else {
      float h = uObs.z * 1.4;                          // upstream wedge
      if(d.x < -h || d.x > h) return false;
      float tloc = (d.x + h) / (2.0*h);               // 0 apex .. 1 base
      return abs(d.y) < uObs.z * tloc;
    }
  }
`;

// ---- fused stream + collide pass ----
const simU = {
  tA: { value: null }, tB: { value: null }, tC: { value: null },
  uTexel: { value: new THREE.Vector2(1 / SIM_W, 1 / SIM_H) },
  uSim: { value: new THREE.Vector2(SIM_W, SIM_H) },
  uTau: { value: 0.58 },
  uIn: { value: 0.10 },
  uObs: { value: new THREE.Vector3(OBS.cx, OBS.cy, OBS.r) },
  uShape: { value: OBS.shape },
  uTarget: { value: 0 }, // which slab to emit: 0->f0..3, 1->f4..7, 2->f8
};
const simScene = new THREE.Scene();
simScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: simU, vertexShader: VSHADER,
  fragmentShader: LATTICE + `
    varying vec2 vUv;
    uniform sampler2D tA, tB, tC;
    uniform float uTarget;

    // gather the streamed (pre-collision) populations: node pulls from x - e_i.
    void streamAll(vec2 uv, out float f[9]){
      for(int i=0;i<9;i++){
        vec2 src = clamp(uv - ei(i)*uTexel, vec2(0.0), vec2(1.0));
        vec4 A = texture2D(tA, src); // f0,f1,f2,f3
        vec4 B = texture2D(tB, src); // f4,f5,f6,f7
        float C = texture2D(tC, src).r; // f8
        if(i==0) f[0]=A.x;
        else if(i==1) f[1]=A.y;
        else if(i==2) f[2]=A.z;
        else if(i==3) f[3]=A.w;
        else if(i==4) f[4]=B.x;
        else if(i==5) f[5]=B.y;
        else if(i==6) f[6]=B.z;
        else if(i==7) f[7]=B.w;
        else f[8]=C;
      }
    }
    void emit(float f[9]){
      if(uTarget < 0.5)      gl_FragColor = vec4(f[0],f[1],f[2],f[3]);
      else if(uTarget < 1.5) gl_FragColor = vec4(f[4],f[5],f[6],f[7]);
      else                   gl_FragColor = vec4(f[8],0.0,0.0,1.0);
    }

    void main(){
      vec2 uv = vUv;
      float f[9];
      streamAll(uv, f);

      // --- solid: mid-grid bounce-back, swap opposite pairs ---
      if(isSolid(uv)){
        float g[9];
        g[0]=f[0];
        g[1]=f[3]; g[3]=f[1];
        g[2]=f[4]; g[4]=f[2];
        g[5]=f[7]; g[7]=f[5];
        g[6]=f[8]; g[8]=f[6];
        emit(g);
        return;
      }

      // --- inlet (left edge): impose equilibrium with u = (uIn, 0) ---
      if(uv.x < uTexel.x*1.5){
        float rho = 1.0;
        vec2 u = vec2(uIn, 0.0);
        for(int i=0;i<9;i++) f[i] = feq(i, rho, u);
        emit(f);
        return;
      }

      // --- outflow (right edge): copy the neighbour column (zero gradient) ---
      if(uv.x > 1.0 - uTexel.x*1.5){
        vec2 src = vec2(uv.x - uTexel.x, uv.y);
        if(uTarget < 0.5)      gl_FragColor = texture2D(tA, src);
        else if(uTarget < 1.5) gl_FragColor = texture2D(tB, src);
        else                   gl_FragColor = vec4(texture2D(tC, src).r, 0.0, 0.0, 1.0);
        return;
      }

      // --- macroscopics + BGK collision toward equilibrium ---
      float rho = 0.0;
      vec2 mom = vec2(0.0);
      for(int i=0;i<9;i++){ rho += f[i]; mom += f[i]*ei(i); }
      rho = max(rho, 1e-4);
      vec2 u = mom / rho;
      float omega = 1.0 / uTau;
      for(int i=0;i<9;i++) f[i] += omega * (feq(i, rho, u) - f[i]);
      emit(f);
    }
  `,
})));

// ---- display pass (vorticity / speed colouring) ----
const dispU = {
  tA: { value: null }, tB: { value: null }, tC: { value: null },
  uTexel: { value: new THREE.Vector2(1 / SIM_W, 1 / SIM_H) },
  uSim: { value: new THREE.Vector2(SIM_W, SIM_H) },
  uTau: { value: 0.58 },
  uIn: { value: 0.10 },
  uObs: { value: new THREE.Vector3(OBS.cx, OBS.cy, OBS.r) },
  uShape: { value: OBS.shape },
  uVort: { value: 28.0 },
  uMode: { value: 0 }, // 0 vorticity, 1 speed
};
const dispScene = new THREE.Scene();
dispScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: dispU, vertexShader: VSHADER,
  fragmentShader: LATTICE + `
    varying vec2 vUv;
    uniform sampler2D tA, tB, tC;
    uniform float uVort, uMode;

    vec2 velAt(vec2 uv){
      vec4 A = texture2D(tA, uv);
      vec4 B = texture2D(tB, uv);
      float C = texture2D(tC, uv).r;
      float f[9];
      f[0]=A.x; f[1]=A.y; f[2]=A.z; f[3]=A.w;
      f[4]=B.x; f[5]=B.y; f[6]=B.z; f[7]=B.w; f[8]=C;
      float rho = 0.0; vec2 mom = vec2(0.0);
      for(int i=0;i<9;i++){ rho += f[i]; mom += f[i]*ei(i); }
      rho = max(rho, 1e-4);
      return mom / rho;
    }

    void main(){
      vec2 uv = vUv;
      if(isSolid(uv)){ gl_FragColor = vec4(0.10, 0.12, 0.16, 1.0); return; } // dim slate block

      vec2 uC = velAt(uv);
      float speed = length(uC);

      // vorticity = ∂uy/∂x − ∂ux/∂y, central differences
      vec2 uxp = velAt(uv + vec2(uTexel.x, 0.0));
      vec2 uxm = velAt(uv - vec2(uTexel.x, 0.0));
      vec2 uyp = velAt(uv + vec2(0.0, uTexel.y));
      vec2 uym = velAt(uv - vec2(0.0, uTexel.y));
      float curl = (uxp.y - uxm.y) - (uyp.x - uym.x);

      vec3 col;
      vec3 base = vec3(0.015, 0.02, 0.03);
      if(uMode < 0.5){
        // signed vorticity: magenta(−) ↔ near-black(0) ↔ cyan(+), white at extremes
        float s = clamp(curl * uVort, -1.0, 1.0);
        vec3 cyan = vec3(0.20, 0.95, 1.00);
        vec3 mag  = vec3(1.00, 0.20, 0.85);
        vec3 pos = mix(base, cyan, smoothstep(0.0, 0.7, s));
        vec3 neg = mix(base, mag,  smoothstep(0.0, 0.7, -s));
        col = (s >= 0.0) ? pos : neg;
        col = mix(col, vec3(1.0), smoothstep(0.75, 1.0, abs(s)));
      } else {
        float t = clamp(speed * 6.0, 0.0, 1.0);
        col = mix(base, vec3(0.20, 0.95, 1.00), smoothstep(0.0, 0.7, t));
        col = mix(col, vec3(1.0), smoothstep(0.7, 1.0, t));
      }
      gl_FragColor = vec4(col, 1.0);
    }
  `,
})));

// ---- copy shader: blit a CPU seed DataTexture into a render target ----
const copyU = { uTex: { value: null } };
const copyScene = new THREE.Scene();
copyScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: copyU, vertexShader: VSHADER,
  fragmentShader: `precision highp float; varying vec2 vUv; uniform sampler2D uTex;
    void main(){ gl_FragColor = texture2D(uTex, vUv); }`,
})));

// ---- seed: uniform flow at equilibrium f_i^eq(ρ=1, u=(uIn,0)) ----
function d2q9eq(rho, ux, uy) {
  const ex = [0, 1, 0, -1, 0, 1, -1, -1, 1];
  const ey = [0, 0, 1, 0, -1, 1, 1, -1, -1];
  const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
  const uu = ux * ux + uy * uy;
  const f = new Array(9);
  for (let i = 0; i < 9; i++) {
    const eu = ex[i] * ux + ey[i] * uy;
    f[i] = w[i] * rho * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uu);
  }
  return f;
}

function seed() {
  const uIn = simU.uIn.value;
  const N = SIM_W * SIM_H;
  const dA = new Float32Array(N * 4);
  const dB = new Float32Array(N * 4);
  const dC = new Float32Array(N * 4);
  const f = d2q9eq(1.0, uIn, 0.0);
  for (let p = 0; p < N; p++) {
    dA[p * 4 + 0] = f[0]; dA[p * 4 + 1] = f[1]; dA[p * 4 + 2] = f[2]; dA[p * 4 + 3] = f[3];
    dB[p * 4 + 0] = f[4]; dB[p * 4 + 1] = f[5]; dB[p * 4 + 2] = f[6]; dB[p * 4 + 3] = f[7];
    dC[p * 4 + 0] = f[8]; dC[p * 4 + 3] = 1.0;
  }
  const mk = (d) => { const t = new THREE.DataTexture(d, SIM_W, SIM_H, THREE.RGBAFormat, THREE.FloatType); t.needsUpdate = true; return t; };
  const sA = mk(dA), sB = mk(dB), sC = mk(dC);
  const blit = (tex, ...targets) => {
    copyU.uTex.value = tex;
    for (const rt of targets) { renderer.setRenderTarget(rt); renderer.render(copyScene, cam); }
  };
  blit(sA, aA, aB);
  blit(sB, bA, bB);
  blit(sC, cA, cB);
  renderer.setRenderTarget(null);
  sA.dispose(); sB.dispose(); sC.dispose();
}

seed();

// ---- one full LBM step: write the three "B" targets, then swap ----
function stepOnce() {
  simU.tA.value = aA.texture;
  simU.tB.value = bA.texture;
  simU.tC.value = cA.texture;

  simU.uTarget.value = 0; renderer.setRenderTarget(aB); renderer.render(simScene, cam);
  simU.uTarget.value = 1; renderer.setRenderTarget(bB); renderer.render(simScene, cam);
  simU.uTarget.value = 2; renderer.setRenderTarget(cB); renderer.render(simScene, cam);
  renderer.setRenderTarget(null);

  let t;
  t = aA; aA = aB; aB = t;
  t = bA; bA = bB; bB = t;
  t = cA; cA = cB; cB = t;
}

// ---- controls ----
const stepsPerFrame = 6;

function setInlet(v) { simU.uIn.value = v; dispU.uIn.value = v; }
function setTau(v) { simU.uTau.value = v; dispU.uTau.value = v; }
function setShape(s) { simU.uShape.value = s; dispU.uShape.value = s; }

bindRange("inlet", (v) => { setInlet(v); }, (v) => v.toFixed(3));
bindRange("tau", (v) => { setTau(v); }, (v) => v.toFixed(3));

// reset re-seeds uniform flow
const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", seed);

// ---- presets via ↑/↓ (Reynolds levels + obstacle shapes) ----
// Re ≈ uIn·(2r·SIM_W)/ν, ν = (τ − 1/2)/3. uIn & τ tuned per level.
const PRESETS = [
  ["Re·lo circle",  0.060, 0.62, 0],
  ["Re·mid circle", 0.100, 0.58, 0],
  ["Re·hi circle",  0.140, 0.55, 0],
  ["Re·mid square", 0.100, 0.57, 1],
  ["Re·hi wedge",   0.130, 0.55, 2],
];
let presetIdx = 1; // Re·mid circle — matches default slider values

function applyPreset(i) {
  const [, uIn, tau, shape] = PRESETS[i];
  setInlet(uIn); setTau(tau); setShape(shape);
  // reflect into the sliders + their .val mirrors, and the chip row
  const si = document.getElementById("inlet");
  const st = document.getElementById("tau");
  const ov = document.querySelector('[data-val="inlet"]');
  const tv = document.querySelector('[data-val="tau"]');
  if (si) si.value = String(uIn);
  if (st) st.value = String(tau);
  if (ov) ov.textContent = uIn.toFixed(3);
  if (tv) tv.textContent = tau.toFixed(3);
  chips.forEach((c, k) => c.classList.toggle("active", k === i));
  if (nameEl) nameEl.textContent = PRESETS[i][0];
}

// build the preset chip row + readout name
const wrap = document.getElementById("presets");
const nameEl = document.getElementById("presetname");
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === presetIdx ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { presetIdx = i; applyPreset(i); seed(); });
  if (wrap) wrap.appendChild(b);
  return b;
});

setVariantCycler((d) => {
  presetIdx = (presetIdx + d + PRESETS.length) % PRESETS.length;
  applyPreset(presetIdx); seed();
  return PRESETS[presetIdx][0];
});

// space toggles colouring (vorticity <-> speed)
window.addEventListener("keydown", (e) => {
  if (e.key === " ") { dispU.uMode.value = dispU.uMode.value > 0.5 ? 0 : 1; e.preventDefault(); }
});

// start on the default preset (keeps sliders + shape consistent)
applyPreset(presetIdx);

// ---- diagnostics hook ----
window.__diag = () => JSON.stringify({
  mode: "LBM",
  grid: [SIM_W, SIM_H],
  tau: simU.uTau.value,
  inlet: simU.uIn.value,
  preset: PRESETS[presetIdx][0],
});

// ---- boot ----
liftVeil();
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  const n = reducedMotion ? 2 : stepsPerFrame;
  for (let i = 0; i < n; i++) stepOnce();

  dispU.tA.value = aA.texture;
  dispU.tB.value = bA.texture;
  dispU.tC.value = cA.texture;
  renderer.setRenderTarget(null);
  renderer.render(dispScene, cam);
});
