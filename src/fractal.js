// fractal.js — a Mandelbulb raymarched on the GPU.
// A single fullscreen quad; all the geometry lives in the fragment shader.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion } from "./common.js";
import { makePS1Pipeline } from "./ps1.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.3)); // fractal is heavy — cap harder

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // unused transform; quad is in clip space

const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uCamPos: { value: new THREE.Vector3(0, 0, 2.6) },
  uPower: { value: 8.0 },
  uIter: { value: 8 },
  uSteps: { value: 110 },
};

const vert = /* glsl */ `
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const frag = /* glsl */ `
  precision highp float;

  uniform vec2  uResolution;
  uniform float uTime;
  uniform vec3  uCamPos;
  uniform float uPower;
  uniform int   uIter;
  uniform int   uSteps;

  const int MAX_ITER  = 12;
  const int MAX_STEPS = 180;
  const float FAR = 8.0;

  // gallery palette: indigo -> teal -> amber -> warm white
  vec3 palette(float t) {
    vec3 a = vec3(0.10, 0.02, 0.24);
    vec3 b = vec3(0.45, 0.10, 0.62);
    vec3 c = vec3(1.00, 0.18, 0.60);
    vec3 d = vec3(0.16, 0.82, 0.96);
    vec3 e = vec3(0.55, 1.00, 0.78);
    t = clamp(t, 0.0, 1.0) * 4.0;
    if (t < 1.0) return mix(a, b, t);
    if (t < 2.0) return mix(b, c, t - 1.0);
    if (t < 3.0) return mix(c, d, t - 2.0);
    return mix(d, e, t - 3.0);
  }

  // Mandelbulb distance estimator. 'trap' = orbit trap for coloring.
  float bulbDE(vec3 pos, out float trap) {
    vec3 z = pos;
    float dr = 1.0;
    float r = 0.0;
    trap = 1e10;
    for (int i = 0; i < MAX_ITER; i++) {
      if (i >= uIter) break;
      r = length(z);
      if (r > 2.0) break;
      float theta = acos(z.z / r);
      float phi   = atan(z.y, z.x);
      dr = pow(r, uPower - 1.0) * uPower * dr + 1.0;
      float zr = pow(r, uPower);
      theta *= uPower;
      phi   *= uPower;
      z = zr * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)) + pos;
      trap = min(trap, r);
    }
    return 0.5 * log(r) * r / dr;
  }

  vec3 calcNormal(vec3 p) {
    float h = 0.0008;
    vec2 k = vec2(1.0, -1.0);
    float t;
    return normalize(
      k.xyy * bulbDE(p + k.xyy * h, t) +
      k.yyx * bulbDE(p + k.yyx * h, t) +
      k.yxy * bulbDE(p + k.yxy * h, t) +
      k.xxx * bulbDE(p + k.xxx * h, t)
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
      float d = bulbDE(p, tr);
      glow += 0.012 / (1.0 + d * d * 40.0); // halo from near-misses
      if (d < 0.0006 * t) { hit = true; trap = tr; break; }
      t += d * 0.85;
      if (t > FAR) break;
    }

    // background: deep purple night + hot-pink bloom from near-misses
    float bgGrad = smoothstep(-0.7, 0.9, rd.y);
    vec3 col = mix(vec3(0.01, 0.015, 0.03), vec3(0.02, 0.05, 0.08), bgGrad);   // near-black, not purple
    col += vec3(0.18, 0.45, 0.60) * glow;   // cool cyan halo (was hot-pink)

    if (hit) {
      vec3 p = ro + rd * t;
      vec3 n = calcNormal(p);

      vec3 lig = normalize(vec3(0.7, 0.85, 0.5));
      float dif = clamp(dot(n, lig), 0.0, 1.0);
      float amb = 0.35 + 0.4 * (n.y * 0.5 + 0.5);
      float ao  = 1.0 - float(usedSteps) / float(uSteps); // cheap occlusion
      ao = clamp(ao * 1.2, 0.15, 1.0);

      vec3 h = normalize(lig - rd);
      float spec = pow(clamp(dot(n, h), 0.0, 1.0), 28.0);

      vec3 base = palette(trap * 0.5 + 0.15);
      col = base * (amb + dif * 0.85) * ao;
      col += vec3(0.7, 0.95, 1.0) * spec * 0.85;   // neon cyan glint
      // depth haze toward the purple background
      col = mix(col, vec3(0.015, 0.03, 0.05), smoothstep(2.5, FAR, t));
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
// dither = chunky PS1 bulb. srgb:false because this shader already gamma-corrects its
// own output. Bonus: marching 1/4 of the pixels is much cheaper, so it runs faster.
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();

// ---------- camera orbit (custom; the "camera" is just uCamPos) ----------
let yaw = 0.6, pitch = 0.5, dist = 2.6;
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
  targetDist = Math.max(1.25, Math.min(6.0, targetDist + e.deltaY * 0.0016));
}, { passive: false });

// ---------- panel ----------
const pwEl = document.getElementById("pw");
const distEl = document.getElementById("dist");

let basePower = 8.0;
bindRange("power", (v) => { basePower = v; }, (v) => v.toFixed(1));
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
  targetYaw = 0.6; targetPitch = 0.5; targetDist = 2.6;
});

// ---------- resize / loop ----------
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });

const meter = fpsMeter(document.getElementById("fps"));
let booted = false;

loop((dt, elapsed) => {
  meter(dt);
  uniforms.uTime.value = elapsed;
  if (spinning && !reducedMotion) targetYaw += dt * 0.25;
  const power = morphing && !reducedMotion ? basePower + Math.sin(elapsed * 0.35) * 1.2 : basePower;
  uniforms.uPower.value = power;
  pwEl.textContent = power.toFixed(1);
  updateCam();
  ps1.render();
  if (!booted) { booted = true; liftVeil(); }
});
