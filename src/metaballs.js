// metaballs.js — raymarched metaballs (room 45 · 雫 · "Metaballs").
//
// THE ALGORITHM — smooth-union of sphere SDFs, raymarched on the GPU.
//   • We keep N (5–8) moving sphere centers. Each rides an independent
//     Lissajous/sin path over time, so they drift, merge, and split like
//     a lava lamp. Centers + radii are pushed to the shader as a fixed-length
//     uniform array `uBalls[MAX_BALLS]` (xyz = center, w = radius).
//   • The scene SDF is the iterated polynomial smooth-minimum of the per-sphere
//     distances. Classic Quílez smin:
//         smin(a,b,k): h = clamp(0.5 + 0.5*(b-a)/k, 0, 1);
//                      return mix(b, a, h) - k*h*(1.0 - h);
//     Folding all spheres through smin gives one continuous blobby surface —
//     this is the metaball/"blobby" implicit surface, the SDF cousin of the
//     classic Blinn 1982 sum-of-Gaussians metaball field.
//   • Standard sphere-tracing march; surface normal via the 4-tap tetrahedron
//     gradient of the SDF. Shade with diffuse + Fresnel rim + a warm
//     subsurface-ish core glow, colored from an iridescent neon ramp on
//     near-black (NOT purple).
//   • "blend k" slider = the smin smoothing radius (how eagerly blobs fuse).
//     "speed" slider = how fast the centers drift. Ball count cycles via ↑/↓.
//
// Refs: Inigo Quílez, "smooth minimum" + "distance functions" articles;
//       Jim Blinn, "A Generalization of Algebraic Surface Drawing" (1982).
//
// One fullscreen quad; all geometry lives in the fragment shader (like fractal.js).

import * as THREE from "three";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil,
  bindRange, reducedMotion, setVariantCycler,
} from "./common.js";
import { makePS1Pipeline } from "./ps1.js";

const MAX_BALLS = 8; // fixed GLSL array length — must match the shader declaration

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // raymarch is heavy — cap DPR

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // unused; quad lives in clip space

// flat array of MAX_BALLS Vector4s (xyz center, w radius); updated each frame
const balls = [];
for (let i = 0; i < MAX_BALLS; i++) balls.push(new THREE.Vector4(0, 0, 0, 0.5));

const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime:       { value: 0 },
  uCamPos:     { value: new THREE.Vector3(0, 0, 4.6) },
  uBalls:      { value: balls },
  uCount:      { value: 6 },
  uK:          { value: 0.55 }, // smooth-union blend radius
};

const vert = /* glsl */ `
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const frag = /* glsl */ `
  precision highp float;

  uniform vec2  uResolution;
  uniform float uTime;
  uniform vec3  uCamPos;
  uniform vec4  uBalls[${MAX_BALLS}];   // xyz = center, w = radius
  uniform int   uCount;
  uniform float uK;

  const int   MAX_BALLS = ${MAX_BALLS};
  const int   MAX_STEPS = 96;
  const float FAR = 12.0;

  // iridescent neon ramp on near-black: teal -> cyan -> magenta -> warm amber
  vec3 palette(float t) {
    vec3 a = vec3(0.05, 0.55, 0.62);
    vec3 b = vec3(0.16, 0.82, 0.96);
    vec3 c = vec3(0.95, 0.30, 0.72);
    vec3 d = vec3(1.00, 0.62, 0.28);
    t = clamp(t, 0.0, 1.0) * 3.0;
    if (t < 1.0) return mix(a, b, t);
    if (t < 2.0) return mix(b, c, t - 1.0);
    return mix(c, d, t - 2.0);
  }

  // polynomial smooth-min (Quilez). k controls how softly the two merge.
  float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
  }

  // scene SDF: iterated smooth-union of the active sphere SDFs.
  float mapScene(vec3 p) {
    float d = FAR;                       // start far; first sphere replaces it
    for (int i = 0; i < MAX_BALLS; i++) {
      if (i >= uCount) break;
      vec4 b = uBalls[i];
      float ds = length(p - b.xyz) - b.w;
      d = (i == 0) ? ds : smin(d, ds, uK);
    }
    return d;
  }

  // surface normal via the tetrahedron 4-tap SDF gradient
  vec3 calcNormal(vec3 p) {
    const float h = 0.0015;
    vec2 k = vec2(1.0, -1.0);
    return normalize(
      k.xyy * mapScene(p + k.xyy * h) +
      k.yyx * mapScene(p + k.yyx * h) +
      k.yxy * mapScene(p + k.yxy * h) +
      k.xxx * mapScene(p + k.xxx * h)
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

    // sphere-trace
    float t = 0.0;
    float glow = 0.0;
    bool hit = false;
    for (int i = 0; i < MAX_STEPS; i++) {
      vec3 p = ro + rd * t;
      float d = mapScene(p);
      glow += 0.015 / (1.0 + d * d * 28.0);   // soft halo from near-misses
      if (d < 0.0009 * t) { hit = true; break; }
      t += d * 0.9;
      if (t > FAR) break;
    }

    // background: near-black with a faint cool vertical wash (NOT purple)
    float bgGrad = smoothstep(-0.8, 0.9, rd.y);
    vec3 col = mix(vec3(0.012, 0.018, 0.028), vec3(0.02, 0.05, 0.07), bgGrad);
    col += vec3(0.10, 0.40, 0.52) * glow;       // cyan bloom

    if (hit) {
      vec3 p = ro + rd * t;
      vec3 n = calcNormal(p);

      vec3 lig = normalize(vec3(0.6, 0.8, 0.5));
      float dif = clamp(dot(n, lig), 0.0, 1.0);
      float amb = 0.30 + 0.35 * (n.y * 0.5 + 0.5);

      // Fresnel rim — bright neon edge as the surface turns away
      float fres = pow(1.0 - clamp(dot(n, -rd), 0.0, 1.0), 3.0);

      vec3 h = normalize(lig - rd);
      float spec = pow(clamp(dot(n, h), 0.0, 1.0), 36.0);

      // iridescent base: color shifts with view/normal so blobs glisten
      float hue = 0.5 + 0.5 * sin(uTime * 0.15 + p.x * 0.7 + n.y * 1.3);
      vec3 base = palette(hue);

      col  = base * (amb + dif * 0.85);
      col += palette(fres) * fres * 0.9;                 // iridescent rim
      col += vec3(0.85, 0.95, 1.0) * spec * 0.8;         // neon glint

      // warm subsurface-ish core: deeper regions of the blob glow from within
      float core = exp(-mapScene(p + n * 0.18) * 6.0);
      col += vec3(1.0, 0.45, 0.30) * core * 0.30;

      // depth haze toward the background
      col = mix(col, vec3(0.015, 0.03, 0.05), smoothstep(5.0, FAR, t));
    }

    // vignette + tonemap + gamma (this shader outputs display colour → ps1 srgb:false)
    float vig = 1.0 - 0.26 * dot(uv * 0.5, uv * 0.5);
    col *= vig;
    col = col / (col + vec3(1.0));   // Reinhard
    col = pow(col, vec3(0.4545));    // gamma

    gl_FragColor = vec4(col, 1.0);
  }
`;

const material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// PS1 pass: render into a 1/4-res target → nearest upscale + 4×4 Bayer dither.
// srgb:false because the shader already gamma-corrects. Marching 1/4 the pixels
// is also far cheaper, so the blobs run smooth.
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();

// ---------- per-ball Lissajous parameters (gives each blob its own orbit) ----------
// Each center = sum-of-sines on x/y/z with distinct frequencies + phases.
const seeds = [];
for (let i = 0; i < MAX_BALLS; i++) {
  const a = i * 1.7;
  seeds.push({
    fx: 0.7 + i * 0.13, fy: 0.5 + i * 0.17, fz: 0.6 + i * 0.11,
    px: a, py: a * 1.3 + 0.6, pz: a * 0.7 + 1.1,
    ax: 1.4 + (i % 3) * 0.25, ay: 1.1 + (i % 2) * 0.35, az: 1.3 + (i % 4) * 0.2,
    r:  0.62 - (i % 3) * 0.08,
  });
}

function updateBalls(elapsed, speed) {
  const tt = elapsed * speed;
  for (let i = 0; i < MAX_BALLS; i++) {
    const s = seeds[i];
    balls[i].set(
      Math.sin(tt * s.fx + s.px) * s.ax,
      Math.sin(tt * s.fy + s.py) * s.ay,
      Math.cos(tt * s.fz + s.pz) * s.az,
      s.r
    );
  }
}
updateBalls(0, 1); // seed initial positions so frame 0 isn't all-at-origin

// ---------- camera orbit (the "camera" is just uCamPos) ----------
let yaw = 0.5, pitch = 0.35, dist = 4.6;
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
  if (distEl) distEl.textContent = dist.toFixed(2);
}

canvas.addEventListener("pointerdown", (e) => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
  spinning = false; if (spinBtn) spinBtn.classList.remove("active");
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
  targetDist = Math.max(2.5, Math.min(9.0, targetDist + e.deltaY * 0.0022));
}, { passive: false });

// ---------- panel ----------
const distEl = document.getElementById("dist");
const kEl = document.getElementById("kval");
const countEl = document.getElementById("count");

bindRange("blend", (v) => { uniforms.uK.value = v; }, (v) => v.toFixed(2));
let speed = 1.0;
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2));

function applyCount(n) {
  uniforms.uCount.value = Math.max(2, Math.min(MAX_BALLS, Math.round(n)));
  if (countEl) countEl.textContent = uniforms.uCount.value;
}
applyCount(6);

// ↑/↓ cycles ball count in-place (kiosk variation)
setVariantCycler((dir) => {
  applyCount(uniforms.uCount.value + dir);
  return `${uniforms.uCount.value} balls`;
});

let spinning = !reducedMotion;
const spinBtn = document.getElementById("spin");
if (spinBtn) {
  spinBtn.classList.toggle("active", spinning);
  spinBtn.addEventListener("click", () => {
    spinning = !spinning;
    spinBtn.classList.toggle("active", spinning);
  });
}

const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", () => {
  targetYaw = 0.5; targetPitch = 0.35; targetDist = 4.6;
});

// diagnostics hook (used by the gallery's smoke tests)
window.__diag = () => JSON.stringify({ balls: uniforms.uCount.value });

// ---------- resize / loop ----------
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });

const meter = fpsMeter(document.getElementById("fps"));
let booted = false;

loop((dt, elapsed) => {
  meter(dt);
  uniforms.uTime.value = elapsed;
  if (spinning && !reducedMotion) targetYaw += dt * 0.18;
  // freeze the drift under reduced-motion, but keep a still, readable blob field
  updateBalls(reducedMotion ? 0 : elapsed, speed);
  if (kEl) kEl.textContent = uniforms.uK.value.toFixed(2);
  updateCam();
  ps1.render();
  if (!booted) { booted = true; liftVeil(); }
});
