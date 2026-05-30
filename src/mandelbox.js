// mandelbox.js — the Mandelbox, a raymarched escape-time fractal on the GPU.
// One fullscreen quad; all the geometry lives in the fragment shader, lit by a
// cheap distance-estimator gradient and rendered through the PS1 pipeline.
//
// ALGORITHM — Mandelbox distance estimator (Tom Lowe / Tglad, 2010).
// Iterate v from the sample point p, accumulating a running derivative dz:
//   1) boxFold:    v = clamp(v, -1.0, 1.0) * 2.0 - v
//                  (component-wise: if v>1 -> 2-v, if v<-1 -> -2-v)
//   2) sphereFold: r2 = dot(v, v);
//                  if r2 < minRadius2  -> scale v and dz by fixedRadius2/minRadius2
//                  else if r2 < fixedRadius2 -> scale v and dz by fixedRadius2/r2
//   3) scaleStep:  v = scale*v + p;  dz = dz*abs(scale) + 1.0
//   DE = length(v) / abs(dz)
// minRadius2 ~= 0.25, fixedRadius2 = 1.0, scale ~= -1.5..3.0 (exposed via slider).
// Negative scales fold the box inside-out into the iconic cubic-shell lattice.
// Refs: Tglad's fractalforums.com thread; Mikael Hvidtfeldt Christensen's
// "Distance Estimated 3D Fractals" series; Inigo Quilez raymarching write-ups.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";
import { makePS1Pipeline } from "./ps1.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.3)); // raymarch is heavy — cap hard

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // unused transform; quad is in clip space

const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uCamPos: { value: new THREE.Vector3(0, 0, 7.0) },
  uScale: { value: -1.7 },
  uIter: { value: 12 },
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
  uniform float uScale;
  uniform int   uIter;
  uniform int   uSteps;

  const int   MAX_ITER  = 18;
  const int   MAX_STEPS = 200;
  const float FAR        = 24.0;

  // Mandelbox constants
  const float MIN_RADIUS2   = 0.25;
  const float FIXED_RADIUS2 = 1.0;
  const float FOLD_LIMIT    = 1.0;

  // neon palette on near-black: cyan -> magenta -> amber -> warm white
  vec3 palette(float t) {
    vec3 a = vec3(0.04, 0.07, 0.10);
    vec3 b = vec3(0.10, 0.82, 0.96);   // cyan
    vec3 c = vec3(1.00, 0.20, 0.62);   // magenta
    vec3 d = vec3(1.00, 0.70, 0.20);   // amber
    vec3 e = vec3(0.45, 1.00, 0.80);   // mint (NOT white — white end-stop washed the whole fractal out)
    t = clamp(t, 0.0, 1.0) * 4.0;
    if (t < 1.0) return mix(a, b, t);
    if (t < 2.0) return mix(b, c, t - 1.0);
    if (t < 3.0) return mix(c, d, t - 2.0);
    return mix(d, e, t - 3.0);
  }

  // Mandelbox distance estimator. 'trap' = orbit trap (min |v|^2) for coloring.
  float boxDE(vec3 pos, out float trap) {
    vec3 v = pos;
    float dz = 1.0;                 // running derivative magnitude
    trap = 1e10;
    for (int i = 0; i < MAX_ITER; i++) {
      if (i >= uIter) break;

      // 1) box fold — reflect components that escape the [-1,1] box
      v = clamp(v, -FOLD_LIMIT, FOLD_LIMIT) * 2.0 - v;

      // 2) sphere fold — invert points inside the inner shell, expand the middle
      float r2 = dot(v, v);
      if (r2 < MIN_RADIUS2) {
        float f = FIXED_RADIUS2 / MIN_RADIUS2;
        v *= f; dz *= f;
      } else if (r2 < FIXED_RADIUS2) {
        float f = FIXED_RADIUS2 / r2;
        v *= f; dz *= f;
      }

      // 3) scale + translate back toward the seed point
      v = uScale * v + pos;
      dz = dz * abs(uScale) + 1.0;

      trap = min(trap, dot(v, v));
    }
    return length(v) / abs(dz);
  }

  vec3 calcNormal(vec3 p) {
    float h = 0.0009;
    vec2 k = vec2(1.0, -1.0);
    float t;
    return normalize(
      k.xyy * boxDE(p + k.xyy * h, t) +
      k.yyx * boxDE(p + k.yyx * h, t) +
      k.yxy * boxDE(p + k.yxy * h, t) +
      k.xxx * boxDE(p + k.xxx * h, t)
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
      float d = boxDE(p, tr);
      glow += 0.004 / (1.0 + d * d * 60.0); // halo from near-misses
      if (d < 0.0009 * t) { hit = true; trap = tr; break; }
      t += d * 0.9;
      if (t > FAR) break;
    }

    // background: near-black with a cool cyan bloom from the near-misses
    float bgGrad = smoothstep(-0.7, 0.9, rd.y);
    vec3 col = mix(vec3(0.012, 0.016, 0.024), vec3(0.02, 0.04, 0.06), bgGrad);
    col += vec3(0.10, 0.26, 0.36) * glow;   // cyan haze

    if (hit) {
      vec3 p = ro + rd * t;
      vec3 n = calcNormal(p);

      vec3 lig = normalize(vec3(0.7, 0.85, 0.5));
      float dif = clamp(dot(n, lig), 0.0, 1.0);
      float amb = 0.12 + 0.10 * (n.y * 0.5 + 0.5);
      float ao  = 1.0 - float(usedSteps) / float(uSteps); // cheap occlusion
      ao = clamp(ao * 1.2, 0.12, 1.0);

      vec3 h = normalize(lig - rd);
      float spec = pow(clamp(dot(n, h), 0.0, 1.0), 32.0);

      // orbit trap saturated to ~1 on nearly every pixel (→ solid white). Drive the
      // hue with a smooth sin of trap + surface position so it ALWAYS sweeps the full
      // cyan→magenta→amber→mint ramp and adjacent folds get different colours.
      float tt = 0.5 + 0.5 * sin(trap * 5.0 + p.x * 1.3 + p.y * 1.1 + p.z * 0.7);
      vec3 base = palette(tt);
      col = base * (amb + dif * 0.95) * ao;
      col += vec3(0.6, 0.85, 1.0) * spec * 0.45;   // neon glint
      // depth haze toward the background
      col = mix(col, vec3(0.015, 0.03, 0.05), smoothstep(8.0, FAR, t));
    }

    // gentle vignette + tonemap
    float vig = 1.0 - 0.28 * dot(uv * 0.5, uv * 0.5);
    col *= vig;
    col = col / (col + vec3(1.0));            // Reinhard
    col = pow(col, vec3(0.4545));             // gamma (PS1 pass runs srgb:false)

    gl_FragColor = vec4(col, 1.0);
  }
`;

const material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// PS1 pass: render the raymarch into a 1/4-res target -> nearest upscale + 4x4
// Bayer dither = chunky PS1 box. srgb:false because this shader already gamma-
// corrects its output. Bonus: marching 1/4 of the pixels is cheaper, so it runs
// faster. IMPORTANT: the shader reads uResolution from gl_FragCoord, so it MUST
// equal the low-res render-target size (syncRes), not the window size.
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();

// ---------- camera orbit (custom; the "camera" is just uCamPos) ----------
let yaw = 0.6, pitch = 0.45, dist = 7.0;
let targetYaw = yaw, targetPitch = pitch, targetDist = dist;
let dragging = false, lastX = 0, lastY = 0;

// panel readout nodes (declared before updateCam / bindRange use them)
const scEl = document.getElementById("sc");
const distEl = document.getElementById("dist");

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
  if (distEl) distEl.textContent = dist.toFixed(2);
}

canvas.addEventListener("pointerdown", (e) => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  spinning = false; spinBtn.classList.remove("active");
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
  targetDist = Math.max(2.4, Math.min(16.0, targetDist + e.deltaY * 0.004));
}, { passive: false });

// ---------- panel ----------
bindRange("scale", (v) => { uniforms.uScale.value = v; if (scEl) scEl.textContent = v.toFixed(2); }, (v) => v.toFixed(2));
bindRange("iter", (v) => { uniforms.uIter.value = Math.round(v); }, (v) => Math.round(v));
bindRange("steps", (v) => { uniforms.uSteps.value = Math.round(v); }, (v) => Math.round(v));

let morphing = false;
const morphBtn = document.getElementById("morph");
morphBtn.addEventListener("click", () => {
  morphing = !morphing;
  morphBtn.classList.toggle("active", morphing);
});

let spinning = !reducedMotion;
const spinBtn = document.getElementById("spin");
spinBtn.classList.toggle("active", spinning);
spinBtn.addEventListener("click", () => {
  spinning = !spinning;
  spinBtn.classList.toggle("active", spinning);
});

document.getElementById("reset").addEventListener("click", () => {
  targetYaw = 0.6; targetPitch = 0.45; targetDist = 7.0;
});

// variant presets (up/down arrows): characteristic fold-scale values
const SCALES = [-2.0, -1.7, -1.5, 2.0, 2.5, 3.0];
let scaleIdx = 1;
setVariantCycler((dir) => {
  scaleIdx = (scaleIdx + dir + SCALES.length) % SCALES.length;
  const v = SCALES[scaleIdx];
  uniforms.uScale.value = v;
  const sl = document.getElementById("scale"); if (sl) sl.value = String(v);
  const out = document.querySelector('[data-val="scale"]'); if (out) out.textContent = v.toFixed(2);
  if (scEl) scEl.textContent = v.toFixed(2);
  return "scale " + v.toFixed(1);
});

// ---------- resize / loop ----------
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });

const meter = fpsMeter(document.getElementById("fps"));
let booted = false;

window.__diag = () => JSON.stringify({ fps: "see #fps", scale: uniforms.uScale.value });

loop((dt, elapsed) => {
  meter(dt);
  uniforms.uTime.value = elapsed;
  if (spinning && !reducedMotion) targetYaw += dt * 0.22;
  // gently breathe the fold scale around the slider value while morphing
  if (morphing && !reducedMotion) {
    const base = parseFloat(document.getElementById("scale").value);
    uniforms.uScale.value = base + Math.sin(elapsed * 0.3) * 0.18;
    if (scEl) scEl.textContent = uniforms.uScale.value.toFixed(2);
  }
  updateCam();
  ps1.render();
  if (!booted) { booted = true; liftVeil(); }
});
