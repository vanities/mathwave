// reaction.js — Gray-Scott reaction-diffusion on the GPU (ping-pong FBOs).
// Two chemicals A,B obey:  A' = Da∇²A - AB² + F(1-A);  B' = Db∇²B + AB² - (F+k)B
// Different (F,k) give coral, mitosis, spots, mazes — alien Turing patterns that
// look alive. Toroidal wrap, neon-ramped. Extremely weird; doubles as RPG texture.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

// --- simulation grid (capped for perf; matches viewport aspect) ---
let SW = 0, SH = 0;
const SIM_MAX = 640;
function simSize() {
  const ar = innerWidth / innerHeight;
  if (ar >= 1) { SW = SIM_MAX; SH = Math.max(2, Math.round(SIM_MAX / ar)); }
  else { SH = SIM_MAX; SW = Math.max(2, Math.round(SIM_MAX * ar)); }
}
simSize();

const RT_OPTS = {
  type: THREE.HalfFloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  wrapS: THREE.RepeatWrapping, wrapT: THREE.RepeatWrapping,
  depthBuffer: false, stencilBuffer: false,
};
let rtA = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS);
let rtB = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS);

const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quad = new THREE.PlaneGeometry(2, 2);
const VSHADER = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// --- simulation step shader ---
const simU = {
  uTex: { value: null }, uTexel: { value: new THREE.Vector2(1 / SW, 1 / SH) },
  uF: { value: 0.0545 }, uK: { value: 0.062 }, uDA: { value: 1.0 }, uDB: { value: 0.5 }, uDt: { value: 1.0 },
};
const simScene = new THREE.Scene();
simScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: simU, vertexShader: VSHADER,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform vec2 uTexel;
    uniform float uF, uK, uDA, uDB, uDt;
    void main(){
      vec2 c = texture2D(uTex, vUv).xy;
      vec2 lap = vec2(0.0);
      lap += texture2D(uTex, vUv + vec2(-uTexel.x, 0.0)).xy * 0.2;
      lap += texture2D(uTex, vUv + vec2( uTexel.x, 0.0)).xy * 0.2;
      lap += texture2D(uTex, vUv + vec2( 0.0,-uTexel.y)).xy * 0.2;
      lap += texture2D(uTex, vUv + vec2( 0.0, uTexel.y)).xy * 0.2;
      lap += texture2D(uTex, vUv + vec2(-uTexel.x,-uTexel.y)).xy * 0.05;
      lap += texture2D(uTex, vUv + vec2( uTexel.x,-uTexel.y)).xy * 0.05;
      lap += texture2D(uTex, vUv + vec2(-uTexel.x, uTexel.y)).xy * 0.05;
      lap += texture2D(uTex, vUv + vec2( uTexel.x, uTexel.y)).xy * 0.05;
      lap -= c;
      float a = c.x, b = c.y, abb = a * b * b;
      a += (uDA * lap.x - abb + uF * (1.0 - a)) * uDt;
      b += (uDB * lap.y + abb - (uF + uK) * b) * uDt;
      gl_FragColor = vec4(clamp(a,0.0,1.0), clamp(b,0.0,1.0), 0.0, 1.0);
    }`,
})));

// --- display shader (color B with the neon ramp) ---
const dispU = { uTex: { value: null }, uTime: { value: 0 } };
const dispScene = new THREE.Scene();
dispScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({
  uniforms: dispU, vertexShader: VSHADER,
  fragmentShader: `
    precision highp float; varying vec2 vUv;
    uniform sampler2D uTex; uniform float uTime;
    vec3 pal(float t){
      vec3 a=vec3(0.10,0.02,0.24), b=vec3(0.45,0.10,0.62), c=vec3(1.0,0.18,0.60),
           d=vec3(0.16,0.82,0.96), e=vec3(0.55,1.0,0.78);
      t=clamp(t,0.0,1.0)*4.0;
      if(t<1.0) return mix(a,b,t);
      if(t<2.0) return mix(b,c,t-1.0);
      if(t<3.0) return mix(c,d,t-2.0);
      return mix(d,e,t-3.0);
    }
    void main(){
      float b = texture2D(uTex, vUv).y;
      float t = smoothstep(0.05, 0.4, b);
      vec3 col = pal(fract(t + uTime*0.02));   // gentle palette cycle
      col *= 0.35 + 1.0*b;
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

// --- seed: A=1 everywhere, B=1 in random splats ---
function seed() {
  const data = new Float32Array(SW * SH * 4);
  for (let i = 0; i < SW * SH; i++) { data[i*4] = 1; data[i*4+1] = 0; data[i*4+3] = 1; }
  const splats = 26;
  for (let s = 0; s < splats; s++) {
    const cx = (Math.random()*SW)|0, cy = (Math.random()*SH)|0, r = (5 + Math.random()*9)|0;
    for (let y=-r; y<=r; y++) for (let x=-r; x<=r; x++) {
      if (x*x + y*y > r*r) continue;
      const px = ((cx+x)%SW+SW)%SW, py = ((cy+y)%SH+SH)%SH;
      data[(py*SW+px)*4 + 1] = 1.0;
    }
  }
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
  seed();
}

// --- presets (F, k) ---
const PRESETS = [
  ["coral",   0.0545, 0.0620],
  ["mitosis", 0.0367, 0.0649],
  ["spots",   0.0300, 0.0620],
  ["worms",   0.0580, 0.0650],
  ["maze",    0.0290, 0.0570],
  ["bubbles", 0.0120, 0.0500],
  ["u-skate", 0.0620, 0.0610],
];
function applyPreset(i) {
  simU.uF.value = PRESETS[i][1];
  simU.uK.value = PRESETS[i][2];
  nameEl.textContent = PRESETS[i][0];
  chips.forEach((c, k) => c.classList.toggle("active", k === i));
}

// --- panel ---
const wrap = document.getElementById("presets");
const nameEl = document.getElementById("presetname");
let presetIdx = 0;
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { presetIdx = i; applyPreset(i); seed(); });
  wrap.appendChild(b);
  return b;
});

let iters = 12;
bindRange("rate", (v) => { iters = Math.round(v); }, (v) => `${Math.round(v)}×`);

let playing = !reducedMotion;
const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "pause" : "play"; playBtn.classList.toggle("active", playing); });
document.getElementById("reseed").addEventListener("click", seed);

setVariantCycler((d) => {
  presetIdx = (presetIdx + d + PRESETS.length) % PRESETS.length;
  applyPreset(presetIdx); seed();
  return PRESETS[presetIdx][0];
});

// --- boot ---
applyPreset(0);
seed();
liftVeil();
window.addEventListener("resize", reallocate);
const meter = fpsMeter(document.getElementById("fps"));

let t = 0;
loop((dt) => {
  meter(dt);
  if (playing) {
    for (let i = 0; i < iters; i++) {
      simU.uTex.value = rtA.texture;
      renderer.setRenderTarget(rtB);
      renderer.render(simScene, cam);
      const tmp = rtA; rtA = rtB; rtB = tmp;
    }
  }
  t += dt;
  dispU.uTex.value = rtA.texture;
  dispU.uTime.value = t;
  renderer.setRenderTarget(null);
  renderer.render(dispScene, cam);
});
