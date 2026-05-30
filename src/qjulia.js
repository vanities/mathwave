// qjulia.js — a Quaternion Julia set: a TRUE 4D fractal sliced into 3D and
// raymarched on the GPU.
//
// The map is the same z → z² + c that draws the Mandelbrot/Julia sets, but z
// and c live in the QUATERNIONS (4D, basis 1,i,j,k) instead of the complex
// plane. The filled Julia set of a quaternion c is a 4D solid; we can only see
// 3D, so we fix one of the four coordinates to a constant "slice" value and
// render the resulting cross-section. Animating that slice sweeps the camera
// through the 4th dimension — the signature move of this room.
//
// Distance estimator (Hart, Sandin & Kauffman 1989, "Ray Tracing Deterministic
// 3-D Fractals"): iterate q = q² + c while carrying the derivative qp, updated
// by qp = 2·q·qp (chain rule on the squaring map). For an escaped point,
//   DE = 0.5 · |q| · log(|q|) / |qp|
// is a lower bound on the distance to the set — exactly the Koebe-style estimate
// used for the Mandelbulb, specialised to the analytic quaternion square.
//
// Quaternion product q=(x,i,j,k): (Hamilton)
//   (a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
//    a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
//    a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
//    a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x)
// The square q² follows from qmul(q,q); we keep the general qmul so qp = 2·q·qp
// stays correct.
//
// Same skeleton as fractal.js / menger.js: one fullscreen quad, DE raymarch,
// cheap-gradient normal, orbit camera, PS1 pipeline. srgb:false because the
// shader gamma-corrects its own output.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";
import { makePS1Pipeline } from "./ps1.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.3)); // heavy raymarch — cap DPR

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // unused; quad is in clip space

// c presets — each a distinct quaternion Julia set (classic qjulia constants)
const C_PRESETS = [
  [-0.450, -0.447, 0.181, 0.306],
  [-0.200, 0.400, -0.400, -0.400],
  [-0.125, -0.256, 0.847, 0.0895],
  [-0.213, -0.041, -0.563, -0.560],
  [-0.291, -0.399, 0.339, 0.437],
  [0.180, 0.880, 0.000, 0.000],
];

const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uCamPos: { value: new THREE.Vector3(0, 0, 2.8) },
  uC: { value: new THREE.Vector4(...C_PRESETS[0]) },
  uSlice: { value: 0.0 },   // the fixed 4th quaternion coordinate (the 4D slice)
  uIter: { value: 9 },
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
  uniform vec4  uC;
  uniform float uSlice;
  uniform int   uIter;
  uniform int   uSteps;

  const int MAX_ITER  = 14;
  const int MAX_STEPS = 160;
  const float FAR = 8.0;
  const float ESCAPE = 16.0;   // |q|² escape radius²

  // gallery palette: deep blue -> cyan -> magenta -> amber -> warm white
  // (neon on near-black; cyan→magenta→amber, NOT purple-dominant)
  vec3 palette(float t) {
    vec3 a = vec3(0.03, 0.06, 0.18);
    vec3 b = vec3(0.10, 0.85, 0.98);
    vec3 c = vec3(1.00, 0.20, 0.62);
    vec3 d = vec3(1.00, 0.66, 0.18);
    vec3 e = vec3(1.00, 0.96, 0.82);
    t = clamp(t, 0.0, 1.0) * 4.0;
    if (t < 1.0) return mix(a, b, t);
    if (t < 2.0) return mix(b, c, t - 1.0);
    if (t < 3.0) return mix(c, d, t - 2.0);
    return mix(d, e, t - 3.0);
  }

  // Hamilton quaternion product (basis 1,i,j,k stored as x,y,z,w)
  vec4 qmul(vec4 a, vec4 b) {
    return vec4(
      a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
      a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
      a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
      a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
  }

  // Quaternion Julia distance estimator (Hart 1989).
  // The 3D sample point fills x,y,z; uSlice fills the 4th (w) coordinate.
  // trap = closest approach to the origin, reused for coloring.
  float qjuliaDE(vec3 pos, out float trap) {
    vec4 q  = vec4(pos, uSlice);   // 3D point + fixed 4D slice
    vec4 qp = vec4(1.0, 0.0, 0.0, 0.0); // derivative, starts at 1
    float md = 1e10;
    trap = 1e10;
    for (int i = 0; i < MAX_ITER; i++) {
      if (i >= uIter) break;
      qp = 2.0 * qmul(q, qp);      // chain rule: d/dq (q² + c) = 2q
      q  = qmul(q, q) + uC;        // z -> z² + c
      float m = dot(q, q);
      trap = min(trap, m);
      md = m;
      if (m > ESCAPE) break;
    }
    float r = sqrt(md);
    // DE = 0.5 * |q| * log(|q|) / |qp|  (guard the log + division)
    return 0.5 * r * log(max(r, 1.0001)) / max(length(qp), 1e-6);
  }

  vec3 calcNormal(vec3 p) {
    vec2 e = vec2(1.0, -1.0) * 0.0009;
    float t;
    return normalize(
      e.xyy * qjuliaDE(p + e.xyy, t) +
      e.yyx * qjuliaDE(p + e.yyx, t) +
      e.yxy * qjuliaDE(p + e.yxy, t) +
      e.xxx * qjuliaDE(p + e.xxx, t)
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
      float d = qjuliaDE(p, tr);
      glow += 0.011 / (1.0 + d * d * 44.0);   // halo from near-misses
      if (d < 0.0006 * t) { hit = true; trap = tr; break; }
      t += d * 0.85;
      if (t > FAR) break;
    }

    // background: near-black night + cool cyan bloom from near-misses
    float bgGrad = smoothstep(-0.7, 0.9, rd.y);
    vec3 col = mix(vec3(0.008, 0.012, 0.025), vec3(0.012, 0.03, 0.055), bgGrad);
    col += vec3(0.14, 0.42, 0.58) * glow;

    if (hit) {
      vec3 p = ro + rd * t;
      vec3 n = calcNormal(p);

      vec3 lig = normalize(vec3(0.7, 0.85, 0.5));
      float dif = clamp(dot(n, lig), 0.0, 1.0);
      float amb = 0.35 + 0.4 * (n.y * 0.5 + 0.5);
      float ao  = 1.0 - float(usedSteps) / float(uSteps); // cheap occlusion
      ao = clamp(ao * 1.2, 0.15, 1.0);

      vec3 h = normalize(lig - rd);
      float spec = pow(clamp(dot(n, h), 0.0, 1.0), 30.0);

      // orbit-trap coloring: closeness to origin + a normal-based tint
      float tc = clamp(sqrt(trap) * 0.9 + n.y * 0.12 + 0.08, 0.0, 1.0);
      vec3 base = palette(tc);
      col = base * (amb + dif * 0.85) * ao;
      col += vec3(0.7, 0.95, 1.0) * spec * 0.85;   // neon cyan glint
      col = mix(col, vec3(0.012, 0.025, 0.05), smoothstep(2.5, FAR, t)); // depth haze
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

// PS1 pass: render the raymarch into a 1/4-res target → nearest upscale + 4×4
// Bayer dither. srgb:false because this shader already gamma-corrects its output.
// Bonus: marching 1/4 of the pixels is much cheaper, so it runs faster.
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();

// ---------- camera orbit (the "camera" is just uCamPos) ----------
let yaw = 0.6, pitch = 0.45, dist = 2.8;
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
  targetDist = Math.max(1.4, Math.min(6.0, targetDist + e.deltaY * 0.0016));
}, { passive: false });

// ---------- panel ----------
const distEl = document.getElementById("dist");
const slcEl = document.getElementById("slc");

// the 4D slice value — drag to sweep through the 4th dimension by hand
let baseSlice = 0.0;
bindRange("slice", (v) => { baseSlice = v; }, (v) => v.toFixed(2));
// c.y component — the most visually expressive of the constant quaternion
bindRange("cy", (v) => { uniforms.uC.value.y = v; }, (v) => v.toFixed(2));
bindRange("steps", (v) => { uniforms.uSteps.value = Math.round(v); }, (v) => Math.round(v));

let morphing = !reducedMotion;   // auto-sweep the 4D slice (the signature move)
const morphBtn = document.getElementById("morph");
morphBtn.classList.toggle("active", morphing);
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
  targetYaw = 0.6; targetPitch = 0.45; targetDist = 2.8;
});

// ↑/↓ — cycle c presets (each a distinct quaternion Julia set)
let presetIdx = 0;
setVariantCycler((dir) => {
  presetIdx = (presetIdx + dir + C_PRESETS.length) % C_PRESETS.length;
  const c = C_PRESETS[presetIdx];
  uniforms.uC.value.set(c[0], c[1], c[2], c[3]);
  const inp = document.getElementById("cy");
  if (inp) inp.value = c[1];      // keep the c.y slider in sync with the preset
  return "c #" + (presetIdx + 1);
});

// ---------- resize / loop ----------
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });

const meter = fpsMeter(document.getElementById("fps"));
let booted = false;

loop((dt, elapsed) => {
  meter(dt);
  uniforms.uTime.value = elapsed;
  if (spinning && !reducedMotion) targetYaw += dt * 0.24;
  // sweep the 4D slice: morph adds a slow sine drift on top of the base value
  const slice = morphing && !reducedMotion ? baseSlice + Math.sin(elapsed * 0.32) * 0.55 : baseSlice;
  uniforms.uSlice.value = slice;
  slcEl.textContent = slice.toFixed(2);
  updateCam();
  ps1.render();
  if (!booted) { booted = true; liftVeil(); }
  // diag hook for headless checks
  if (!window.__diag) window.__diag = () => JSON.stringify({ slice: uniforms.uSlice.value });
});
