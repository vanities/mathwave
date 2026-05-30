// menger.js — a Menger Sponge fractal raymarched on the GPU.
// A single fullscreen quad; all the geometry lives in the fragment shader.
//
// ALGORITHM — Menger sponge SDF via iterated cross-subtraction (the classic
// Inigo Quilez approach, https://iquilezles.org/articles/menger/):
//   1. Start with d = the signed distance to a cube (box SDF).
//   2. Repeat N times: fold space into a 3×3×3 grid with a = mod(p*s,2)-1,
//      then s *= 3. Carve out the central cross of three perpendicular square
//      tubes — distance to that cross is
//         cross = min(max(|a.x|,|a.y|), min(max(|a.y|,|a.z|), max(|a.z|,|a.x|))) - 1
//      Scale it back into world space as c = cross / s and subtract it from the
//      solid with d = max(d, c). Each pass punches the holes-within-holes that
//      give the sponge its self-similar lacework.
// Raymarched with the distance estimate, a cheap finite-difference normal, and
// ambient occlusion derived from the marching step count. Neon cyan/amber on
// near-black (the gallery has moved off vaporwave purple).

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion } from "./common.js";
import { makePS1Pipeline } from "./ps1.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.3)); // raymarch is heavy — cap harder

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // unused transform; quad is in clip space

const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uCamPos: { value: new THREE.Vector3(0, 0, 3.0) },
  uIter: { value: 5 },
  uSteps: { value: 120 },
};

const vert = /* glsl */ `
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const frag = /* glsl */ `
  precision highp float;

  uniform vec2  uResolution;
  uniform float uTime;
  uniform vec3  uCamPos;
  uniform int   uIter;
  uniform int   uSteps;

  const int MAX_ITER  = 7;
  const int MAX_STEPS = 200;
  const float FAR = 12.0;

  // gallery palette: deep teal -> cyan -> warm amber -> near-white (no purple)
  vec3 palette(float t) {
    vec3 a = vec3(0.02, 0.10, 0.14);
    vec3 b = vec3(0.05, 0.55, 0.62);
    vec3 c = vec3(0.16, 0.85, 0.95);
    vec3 d = vec3(1.00, 0.66, 0.22);
    vec3 e = vec3(1.00, 0.93, 0.80);
    t = clamp(t, 0.0, 1.0) * 4.0;
    if (t < 1.0) return mix(a, b, t);
    if (t < 2.0) return mix(b, c, t - 1.0);
    if (t < 3.0) return mix(c, d, t - 2.0);
    return mix(d, e, t - 3.0);
  }

  // signed distance to a box of half-extents b
  float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
  }

  // Menger sponge distance estimator. 'trap' = how deep the carving went,
  // used as a coloring orbit trap.
  float mengerDE(vec3 p, out float trap) {
    float d = sdBox(p, vec3(1.0));   // start: solid cube
    float s = 1.0;
    trap = 1.0;
    for (int i = 0; i < MAX_ITER; i++) {
      if (i >= uIter) break;
      // fold into a 3x3x3 cell, centered
      vec3 a = mod(p * s, 2.0) - 1.0;
      s *= 3.0;
      vec3 r = abs(1.0 - 3.0 * abs(a));
      // distance to the central cross of three square tubes
      float da = max(r.x, r.y);
      float db = max(r.y, r.z);
      float dc = max(r.z, r.x);
      float cross = (min(da, min(db, dc)) - 1.0) / s;
      if (cross > d) { d = cross; trap = float(i) + 1.0; }  // d = max(d, cross)
    }
    return d;
  }

  vec3 calcNormal(vec3 p) {
    float h = 0.0009;
    vec2 k = vec2(1.0, -1.0);
    float t;
    return normalize(
      k.xyy * mengerDE(p + k.xyy * h, t) +
      k.yyx * mengerDE(p + k.yyx * h, t) +
      k.yxy * mengerDE(p + k.yxy * h, t) +
      k.xxx * mengerDE(p + k.xxx * h, t)
    );
  }

  void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - uResolution) / uResolution.y;

    // camera basis, always looking at the origin
    vec3 ro = uCamPos;
    vec3 ww = normalize(-ro);
    vec3 uu = normalize(cross(vec3(0.0, 1.0, 0.0), ww));
    vec3 vv = cross(ww, uu);
    vec3 rd = normalize(uv.x * uu + uv.y * vv + 1.7 * ww);

    // march
    float t = 0.0;
    float trap = 0.0;
    float glow = 0.0;
    bool hit = false;
    int usedSteps = 0;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (i >= uSteps) break;
      usedSteps = i;
      vec3 p = ro + rd * t;
      float tr;
      float d = mengerDE(p, tr);
      glow += 0.010 / (1.0 + d * d * 50.0);  // halo from near-misses
      if (d < 0.0006 * t) { hit = true; trap = tr; break; }
      t += d * 0.9;
      if (t > FAR) break;
    }

    // background: near-black, faint cool gradient, cyan bloom from near-misses
    float bgGrad = smoothstep(-0.7, 0.9, rd.y);
    vec3 col = mix(vec3(0.008, 0.012, 0.018), vec3(0.015, 0.035, 0.05), bgGrad);
    col += vec3(0.10, 0.40, 0.52) * glow;   // cool cyan halo

    if (hit) {
      vec3 p = ro + rd * t;
      vec3 n = calcNormal(p);

      vec3 lig = normalize(vec3(0.65, 0.8, 0.55));
      float dif = clamp(dot(n, lig), 0.0, 1.0);
      float amb = 0.32 + 0.42 * (n.y * 0.5 + 0.5);
      float ao  = 1.0 - float(usedSteps) / float(uSteps); // cheap occlusion
      ao = clamp(ao * 1.25, 0.12, 1.0);

      vec3 h = normalize(lig - rd);
      float spec = pow(clamp(dot(n, h), 0.0, 1.0), 30.0);

      vec3 base = palette(trap / float(max(uIter, 1)) * 0.85 + 0.1);
      col = base * (amb + dif * 0.9) * ao;
      col += vec3(1.0, 0.78, 0.42) * spec * 0.7;   // warm amber glint
      // depth haze toward the background
      col = mix(col, vec3(0.012, 0.025, 0.035), smoothstep(3.0, FAR, t));
    }

    // gentle vignette + tonemap
    float vig = 1.0 - 0.28 * dot(uv * 0.5, uv * 0.5);
    col *= vig;
    col = col / (col + vec3(1.0));            // Reinhard
    col = pow(col, vec3(0.4545));             // gamma

    gl_FragColor = vec4(col, 1.0);
  }
`;

const material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// PS1 pass: render the raymarch into a 1/4-res target → nearest upscale + 4×4 Bayer
// dither = chunky PS1 sponge. srgb:false because this shader already gamma-corrects
// its own output. Bonus: marching 1/4 of the pixels is much cheaper, so it runs faster.
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();

// ---------- camera orbit (custom; the "camera" is just uCamPos) ----------
let yaw = 0.7, pitch = 0.45, dist = 3.0;
let targetYaw = yaw, targetPitch = pitch, targetDist = dist;
let dragging = false, lastX = 0, lastY = 0;

function updateCam() {
  yaw += (targetYaw - yaw) * 0.12;
  pitch += (targetPitch - pitch) * 0.12;
  dist += (targetDist - dist) * 0.12;
  const cp = Math.cos(pitch);
  uniforms.uCamPos.value.set(
    dist * cp * Math.sin(yaw),
    dist * Math.sin(pitch),
    dist * cp * Math.cos(yaw)
  );
  distEl.textContent = dist.toFixed(2);
}

canvas.addEventListener("pointerdown", (e) => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  spinning = false; spinBtn.classList.remove("active");
  zooming = false; zoomBtn.classList.remove("active");
});
canvas.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  targetYaw -= (e.clientX - lastX) * 0.006;
  targetPitch += (e.clientY - lastY) * 0.006;
  targetPitch = Math.max(-1.45, Math.min(1.45, targetPitch));
  lastX = e.clientX; lastY = e.clientY;
});
canvas.addEventListener("pointerup", () => (dragging = false));
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  targetDist = Math.max(1.15, Math.min(7.0, targetDist + e.deltaY * 0.0016));
}, { passive: false });

// ---------- panel ----------
const distEl = document.getElementById("dist");
const itEl = document.getElementById("it");

let curIter = 5;
bindRange("iter", (v) => { curIter = Math.round(v); uniforms.uIter.value = curIter; if (itEl) itEl.textContent = curIter; }, (v) => Math.round(v));
bindRange("steps", (v) => { uniforms.uSteps.value = Math.round(v); }, (v) => Math.round(v));

let spinning = !reducedMotion;
const spinBtn = document.getElementById("spin");
spinBtn.classList.toggle("active", spinning);
spinBtn.addEventListener("click", () => {
  spinning = !spinning;
  spinBtn.classList.toggle("active", spinning);
});

let zooming = false;
const zoomBtn = document.getElementById("zoom");
zoomBtn.addEventListener("click", () => {
  zooming = !zooming;
  zoomBtn.classList.toggle("active", zooming);
});

document.getElementById("reset").addEventListener("click", () => {
  targetYaw = 0.7; targetPitch = 0.45; targetDist = 3.0;
});

// expose diagnostics for the harness
window.__diag = () => JSON.stringify({ iters: curIter });

// ---------- resize / loop ----------
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });

const meter = fpsMeter(document.getElementById("fps"));
let booted = false;

loop((dt, elapsed) => {
  meter(dt);
  uniforms.uTime.value = elapsed;
  if (spinning && !reducedMotion) targetYaw += dt * 0.22;
  if (zooming && !reducedMotion) {
    // slow oscillating dive into the structure, then back out
    targetDist = 3.7 - 2.4 * (0.5 - 0.5 * Math.cos(elapsed * 0.18));
  }
  updateCam();
  ps1.render();
  if (!booted) { booted = true; liftVeil(); }
});
