// nebula.js — a domain-warped FBM nebula: volumetric flowing noise clouds.
//
// Algorithm (accurate):
//   • gradient 3D noise (Inigo Quilez hash-gradient) → fbm: sum of 5 octaves,
//     lacunarity 2.0, gain 0.5.
//   • DOMAIN WARPING (IQ, iquilezles.org/articles/warp):
//        q = fbm(p + t);  r = fbm(p + 1.7*q);  density = fbm(p + 1.5*r)
//     warping the input coordinates by noise-of-noise gives the long, curdled,
//     flowing filaments a nebula has.
//   • VOLUMETRIC RAYMARCH: step a ray through the warped noise field and
//     accumulate emission/absorption front-to-back (premultiplied alpha), so the
//     clouds read as glowing 3D volume with depth, not a flat texture.
//   • Palette: deep blue → magenta → warm highlights on near-black.
//   • Slow time drift; camera creeps through the field.
//
// HEAVY shader → rendered through the PS1 1/4-res pipeline (chunky pixels + Bayer
// dither) which also makes it cheap, and STEPS is capped low. Honors reduced-motion.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";
import { makePS1Pipeline } from "./ps1.js";

const STEPS = 56; // raymarch samples per pixel — modest; this shader is expensive

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
renderer.setPixelRatio(1); // volumetric raymarch is heavy — cap DPR hard (PS1 RT shrinks it further)

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1); // quad lives in clip space

const uniforms = {
  uResolution: { value: new THREE.Vector2(1, 1) }, // == PS1 low-res RT size (see syncRes)
  uTime:    { value: 0 },
  uDensity: { value: 1.4 }, // cloud opacity / glow
  uWarp:    { value: 1.0 }, // domain-warp strength
};

const vert = /* glsl */ `
  void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const frag = /* glsl */ `
  precision highp float;

  uniform vec2  uResolution;
  uniform float uTime;
  uniform float uDensity;
  uniform float uWarp;

  #define STEPS ${STEPS}

  // ---- gradient 3D noise (IQ) ----
  vec3 hash3(vec3 p) {
    p = vec3(dot(p, vec3(127.1, 311.7,  74.7)),
             dot(p, vec3(269.5, 183.3, 246.1)),
             dot(p, vec3(113.5, 271.9, 124.6)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
  }
  float gnoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f); // smoothstep interpolant
    return mix(mix(mix(dot(hash3(i + vec3(0,0,0)), f - vec3(0,0,0)),
                       dot(hash3(i + vec3(1,0,0)), f - vec3(1,0,0)), u.x),
                   mix(dot(hash3(i + vec3(0,1,0)), f - vec3(0,1,0)),
                       dot(hash3(i + vec3(1,1,0)), f - vec3(1,1,0)), u.x), u.y),
               mix(mix(dot(hash3(i + vec3(0,0,1)), f - vec3(0,0,1)),
                       dot(hash3(i + vec3(1,0,1)), f - vec3(1,0,1)), u.x),
                   mix(dot(hash3(i + vec3(0,1,1)), f - vec3(0,1,1)),
                       dot(hash3(i + vec3(1,1,1)), f - vec3(1,1,1)), u.x), u.y), u.z);
  }

  // fbm: 5 octaves, lacunarity 2.0, gain 0.5
  float fbm(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
      v += a * gnoise(p);
      p *= 2.0;
      a *= 0.5;
    }
    return v;
  }

  // domain-warped fbm (IQ): q = fbm(p+t); r = fbm(p+1.7q); d = fbm(p+1.5r)
  float warpedDensity(vec3 p) {
    float t = uTime * 0.08;
    vec3 q = vec3(
      fbm(p + vec3(0.0, 0.0, t)),
      fbm(p + vec3(5.2, 1.3, t)),
      fbm(p + vec3(1.7, 9.2, t))
    ) * uWarp;
    vec3 r = vec3(
      fbm(p + 1.7 * q + vec3(8.3, 2.8, 0.0)),
      fbm(p + 1.7 * q + vec3(1.2, 6.9, 0.0)),
      fbm(p + 1.7 * q + vec3(4.1, 3.3, 0.0))
    ) * uWarp;
    return fbm(p + 1.5 * r);
  }

  // rich palette: deep blue → magenta → warm highlights.
  // Hand-tuned ramp (not a single cosine) so low x is a saturated blue, the
  // midrange swings through magenta/violet, and the top blooms warm white-gold.
  vec3 palette(float x) {
    x = clamp(x, 0.0, 1.0);
    // NO purple — navy → teal → warm pink → gold (deep void to hot core)
    vec3 navy = vec3(0.03, 0.10, 0.28); // cool void edges
    vec3 teal = vec3(0.06, 0.42, 0.62); // body
    vec3 pink = vec3(0.95, 0.38, 0.50); // dense filaments
    vec3 gold = vec3(1.00, 0.78, 0.42); // hot cores
    vec3 col = mix(navy, teal, smoothstep(0.00, 0.45, x));
    col      = mix(col,  pink, smoothstep(0.45, 0.78, x));
    col      = mix(col,  gold, smoothstep(0.78, 1.00, x));
    return col;
  }

  void main() {
    // map fragment to centered, aspect-correct screen coords using the LOW-RES
    // resolution uniform (== PS1 RT size), so uv matches what we actually render.
    vec2 uv = (gl_FragCoord.xy * 2.0 - uResolution) / uResolution.y;

    // ray: camera slowly drifts + creeps forward through the field
    float t = uTime * 0.05;
    vec3 ro = vec3(sin(t * 0.7) * 0.6, cos(t * 0.5) * 0.3, uTime * 0.12);
    vec3 rd = normalize(vec3(uv, 1.4));

    // front-to-back emission/absorption compositing (premultiplied alpha)
    vec3  col   = vec3(0.0);
    float alpha = 0.0;
    float tt    = 0.0;
    const float stepLen = 1.7 / float(STEPS);

    for (int i = 0; i < STEPS; i++) {
      vec3 p = ro + rd * tt;
      float d = warpedDensity(p * 1.5);
      // fbm sits in a narrow band centred near 0, so remap with a GENEROUS
      // window: most of the field now emits (clouds actually form) while the
      // deepest voids stay dark for depth. Bias up + wide smoothstep.
      // contrasty window so voids go truly dark (structure, not a flat fill wall)
      float dRaw = smoothstep(0.02, 0.42, d);
      float dens = dRaw * uDensity;
      if (dens > 0.001) {
        // hue tracks cloud THICKNESS: thin edges→navy/teal, filaments→pink, cores→gold
        vec3 c = palette(clamp(dRaw * 1.25, 0.0, 1.0));
        c *= dens * 1.4;                                 // emission strength
        float a = clamp(dRaw * stepLen * 6.0, 0.0, 1.0); // absorption per step
        col   += (1.0 - alpha) * c * a;
        alpha += (1.0 - alpha) * a;
      }
      if (alpha > 0.99) break;
      tt += stepLen;
    }

    // composite over a near-black, faintly cool background
    vec3 bg   = vec3(0.02, 0.03, 0.06);
    vec3 outc = bg + col;
    outc = outc / (outc + vec3(1.0));      // Reinhard tonemap
    outc = pow(outc, vec3(0.4545));        // single gamma (RT is srgb:false)
    gl_FragColor = vec4(outc, 1.0);
  }
`;

const material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

// PS1 pass: render the raymarch into a 1/4-res target → nearest upscale + Bayer
// dither. srgb:false because this shader already outputs display-space colour.
// Bonus: marching 1/4 of the pixels keeps this heavy volume affordable.
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });

// ---------- panel ----------
bindRange("density", (v) => { uniforms.uDensity.value = v; }, (v) => v.toFixed(2));
bindRange("warp",    (v) => { uniforms.uWarp.value = v;    }, (v) => v.toFixed(2));

// ↑/↓ presets: calm wisps → roiling storm
const VARIANTS = [
  { name: "drift",  density: 1.1, warp: 0.7 },
  { name: "bloom",  density: 1.4, warp: 1.0 },
  { name: "storm",  density: 2.0, warp: 1.5 },
  { name: "abyss",  density: 0.8, warp: 1.9 },
];
let vi = 1;
function applyVariant(i) {
  vi = (i % VARIANTS.length + VARIANTS.length) % VARIANTS.length;
  const v = VARIANTS[vi];
  uniforms.uDensity.value = v.density;
  uniforms.uWarp.value = v.warp;
  const ds = document.getElementById("density"); if (ds) ds.value = String(v.density);
  const ws = document.getElementById("warp");    if (ws) ws.value = String(v.warp);
  const dv = document.querySelector('[data-val="density"]'); if (dv) dv.textContent = v.density.toFixed(2);
  const wv = document.querySelector('[data-val="warp"]');    if (wv) wv.textContent = v.warp.toFixed(2);
  return v.name;
}
setVariantCycler((dir) => applyVariant(vi + dir));

// ---------- loop ----------
const meter = fpsMeter(document.getElementById("fps"));
let booted = false;

loop((dt, elapsed) => {
  meter(dt);
  // honor reduced-motion by slowing the drift way down
  uniforms.uTime.value = reducedMotion ? elapsed * 0.25 : elapsed;
  ps1.render();
  if (!booted) { booted = true; liftVeil(); }
});

window.__diag = () => JSON.stringify({ steps: STEPS });
