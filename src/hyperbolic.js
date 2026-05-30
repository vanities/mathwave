// hyperbolic.js — {p,q} regular tiling of the hyperbolic plane, Poincaré disk model.
//
// Algorithm (accurate): the (2,p,q) triangle reflection group, rendered by the
// standard "fold into the fundamental domain" technique on a fullscreen quad.
//
//   The {p,q} tiling's fundamental domain is a hyperbolic right triangle with
//   angles  pi/p (at the polygon centre), pi/q (at a polygon vertex) and pi/2
//   (at an edge midpoint).  Its three sides are geodesics of the disk:
//     - two are DIAMETERS through the origin (Euclidean line mirrors), giving
//       the p-fold dihedral symmetry around the centre;
//     - one is a circular arc orthogonal to the unit circle — the polygon edge.
//
//   For each pixel z inside the unit disk we repeatedly:
//     1. fold its angle into the wedge [0, pi/p]  (the two diameter mirrors);
//     2. if it lies on the far side of the edge geodesic (i.e. inside that
//        orthogonal mirror circle), invert it back across the circle,
//   counting reflections, until the point rests in the central tile.  Colour by
//   reflection parity / count + distance to the boundary -> the infinite tiling
//   with cells shrinking toward the bounding circle.
//
//   Edge geodesic geometry: the hyperbolic inradius r_i (centre -> edge mid)
//   obeys  cosh(r_i) = cos(pi/q) / sin(pi/p)  (hyperbolic right-triangle law).
//   A hyperbolic distance s sits at Euclidean radius tanh(s/2) in the disk, so
//   the edge midpoint is at me = tanh(r_i/2).  The mirror circle through (me,0)
//   orthogonal to the unit circle has centre (cd,0), radius cr with cd^2=1+cr^2,
//   giving cd=(1+me^2)/(2 me), cr=(1-me^2)/(2 me).  (Hyperbolic iff
//   (p-2)(q-2) > 4, i.e. cosh r_i > 1.)
//
//   Gliding: a hyperbolic translation is a Möbius map of the disk
//   z -> (z - b)/(1 - conj(b) z) with b drifting, so the tiling slides past.
//
// Rendered as a single fullscreen quad + fragment shader (all work on the GPU).

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
// Fullscreen quad uses an orthographic camera; geometry covers clip space [-1,1].
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const uniforms = {
  uRes: { value: new THREE.Vector2(1, 1) },
  uTime: { value: 0 },
  uP: { value: 7 },                          // Schläfli p (polygon sides)
  uQ: { value: 3 },                          // Schläfli q (polygons per vertex)
  uGlide: { value: new THREE.Vector2(0, 0) }, // current Möbius translation b
  uZoom: { value: 1.05 },                    // disk fits in view with a margin
};

const vert = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const frag = `
  precision highp float;
  varying vec2 vUv;
  uniform vec2 uRes;
  uniform float uTime;
  uniform float uP;
  uniform float uQ;
  uniform vec2 uGlide;
  uniform float uZoom;

  #define PI 3.14159265359

  // Complex helpers.
  vec2 cmul(vec2 a, vec2 b) { return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
  vec2 cdiv(vec2 a, vec2 b) { float d = dot(b, b) + 1e-9; return vec2(a.x*b.x + a.y*b.y, a.y*b.x - a.x*b.y) / d; }

  // Hyperbolic translation of the disk: z -> (z - b) / (1 - conj(b) z).
  vec2 mobius(vec2 z, vec2 b) {
    vec2 num = z - b;
    vec2 den = vec2(1.0, 0.0) - cmul(vec2(b.x, -b.y), z);
    return cdiv(num, den);
  }

  void main() {
    // Map pixel to the unit-disk plane, aspect-corrected via uRes; the disk
    // (radius 1) sits centred with a small margin set by uZoom.
    vec2 z = (vUv - 0.5) * 2.0 * uZoom;
    z.x *= uRes.x / uRes.y;

    float rr = dot(z, z);
    if (rr >= 1.0) {
      // Outside the Poincaré disk: dark field, crisp bounding circle.
      float edge = smoothstep(1.0 + 0.02, 1.0, sqrt(rr));
      vec3 ring = vec3(0.25, 0.85, 0.95) * edge * 0.6;
      gl_FragColor = vec4(ring + vec3(0.012, 0.012, 0.02), 1.0);
      return;
    }

    // Glide the whole tiling past the viewer.
    z = mobius(z, uGlide);
    // Re-clamp after the Möbius map to stay strictly inside the disk.
    float zr2 = dot(z, z);
    if (zr2 >= 1.0) z *= 0.99999 / sqrt(zr2);

    // Schläfli numbers (kept as floats; integer steps come from the sliders).
    float p = max(uP, 3.0);
    float q = max(uQ, 3.0);
    float wedge = PI / p;                  // half-angle of one polygon sector

    // Edge geodesic (the orthogonal mirror circle). cosh(r_i)=cos(pi/q)/sin(pi/p).
    float coshri = cos(PI / q) / max(sin(wedge), 1e-4);
    // Hyperbolic requires coshri > 1; clamp degenerate (Euclidean/spherical)
    // settings so the math stays finite instead of NaN-ing out.
    coshri = max(coshri, 1.0001);
    float ri = acoshP(coshri);             // renamed: tanh/acosh are WebGL2 built-ins
    float me = tanhP(ri * 0.5);            // edge midpoint, Euclidean radius
    me = clamp(me, 1e-3, 0.999);
    float cd = (1.0 + me * me) / (2.0 * me);   // mirror-circle centre (cd,0)
    float cr = (1.0 - me * me) / (2.0 * me);   // mirror-circle radius
    vec2 Cc = vec2(cd, 0.0);
    float cr2 = cr * cr;

    // Fold into the fundamental triangle, counting reflections.
    float reflections = 0.0;
    const int MAXIT = 80;
    for (int i = 0; i < MAXIT; i++) {
      // (1) Dihedral fold of the angle into the wedge [0, pi/p].
      float rad = length(z);
      float ang = atan(z.y, z.x);
      ang = mod(ang, 2.0 * wedge);                 // p-fold rotational symmetry
      if (ang > wedge) { ang = 2.0 * wedge - ang; reflections += 1.0; } // mirror
      z = rad * vec2(cos(ang), sin(ang));

      // (2) Reflect across the polygon edge geodesic if we are on its far side
      // (inside the orthogonal mirror circle == outside the central tile).
      vec2 d = z - Cc;
      float dd = dot(d, d);
      if (dd < cr2) {
        z = Cc + d * (cr2 / max(dd, 1e-7));         // inversion in the circle
        reflections += 1.0;
      } else {
        break;                                       // rests in the central tile
      }
    }

    // ---- Colour ----
    // Angle of the folded point inside the fundamental triangle (for walls).
    float angF = atan(z.y, z.x);

    // Per-tile hue: cycle cyan / magenta / amber by reflection count (NOT purple).
    vec3 palette[3];
    palette[0] = vec3(0.20, 0.95, 0.98);   // cyan
    palette[1] = vec3(0.98, 0.22, 0.62);   // magenta/rose
    palette[2] = vec3(1.00, 0.74, 0.18);   // amber
    int ci = int(mod(reflections, 3.0));
    vec3 base = palette[0];
    if (ci == 1) base = palette[1];
    else if (ci == 2) base = palette[2];

    // Parity darkening for a checkerboard read across adjacent tiles.
    float parity = mod(reflections, 2.0);
    base *= mix(1.0, 0.62, parity);

    // Crisp cell walls: brighten near the three mirrors of the domain.
    float wallA = min(angF, wedge - angF);             // dist to the two diameters
    float wallE = abs(length(z - Cc) - cr);            // dist to the edge geodesic
    float wall = min(min(wallA, abs(angF)), wallE);
    float line = smoothstep(0.05, 0.0, wall);          // glowing outline
    vec3 col = mix(base * 0.85, base + vec3(0.35), line);

    // Cell-centre glow so polygons read as solid neon panels on near-black.
    float fill = smoothstep(wedge * 0.95, 0.0, wallA) * 0.5;
    col += base * fill * 0.25;

    // Subtle drift in brightness so the glide feels alive.
    col *= 0.78 + 0.22 * (0.5 + 0.5 * sin(uTime * 0.6 + reflections));

    // Fade toward the bounding circle (cells crowd + dim near |z|->1 of input).
    float boundary = smoothstep(1.0, 0.86, sqrt(rr));
    col *= mix(0.18, 1.0, boundary);

    // Crisp bright bounding circle just inside the rim.
    float rim = smoothstep(0.012, 0.0, abs(sqrt(rr) - 0.992));
    col = mix(col, vec3(0.55, 0.97, 1.0), rim * 0.9);

    // Lift blacks slightly toward deep blue, then tone.
    col += vec3(0.01, 0.012, 0.022);
    col = col / (col + vec3(0.5));          // gentle Reinhard so neons don't clip
    col = pow(col, vec3(0.95));

    gl_FragColor = vec4(col, 1.0);
  }
`;

// acosh/tanh are BUILT-INS on WebGL2 (GLSL ES 3.0) → redefining them is a compile
// error (black screen). Use uniquely-named helpers so it compiles on any version.
const polyfills = `
  precision highp float;
  float acoshP(float x) { return log(x + sqrt(max(x*x - 1.0, 0.0))); }
  float tanhP(float x) { float e = exp(2.0 * x); return (e - 1.0) / (e + 1.0); }
`;

const material = new THREE.ShaderMaterial({
  uniforms,
  vertexShader: vert,
  // Inject polyfills after the precision line of our frag source.
  fragmentShader: frag.replace("precision highp float;", polyfills),
});

const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
scene.add(quad);

const fps = fpsMeter(document.getElementById("fps"));

// Mirror the current {p,q} into the readout.
const pvEl = document.getElementById("pv");
const qvEl = document.getElementById("qv");
function showPQ() {
  if (pvEl) pvEl.textContent = String(uniforms.uP.value);
  if (qvEl) qvEl.textContent = String(uniforms.uQ.value);
}

// Resolution uniform must track the actual drawing-buffer size (matches fractal.js).
onResize(renderer, camera, (w, h) => {
  uniforms.uRes.value.set(w, h);
});

// reducedMotion is an exported boolean (matches common.js); pause the drift
// when the user prefers reduced motion. The tiling itself still renders.
let paused = reducedMotion;
let speed = 0.18;                 // glide speed (radians-ish of drift per sec)
let phase = 0;                    // glide path parameter

bindRange("p", v => { uniforms.uP.value = Math.round(v); showPQ(); }, v => Math.round(v));
bindRange("q", v => { uniforms.uQ.value = Math.round(v); showPQ(); }, v => Math.round(v));
bindRange("speed", v => { speed = v; }, v => v.toFixed(2));

// Reset glide back to the centred tiling. Button only — common.js already
// claims the 'r' key for the global video recorder, so we don't bind it here.
function resetGlide() {
  phase = 0;
  uniforms.uGlide.value.set(0, 0);
}
const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", resetGlide);

loop(() => {
  if (!paused) {
    uniforms.uTime.value += 0.016;
    phase += 0.016 * speed;
    // Drift b along a gentle Lissajous loop, magnitude < 1 so the Möbius map
    // stays a valid disk automorphism (guard well clear of the boundary).
    const mag = 0.62;
    uniforms.uGlide.value.set(
      mag * Math.sin(phase * 0.9),
      mag * Math.sin(phase * 0.6 + 1.7)
    );
  }
  fps();
  renderer.render(scene, camera);
});

liftVeil();

// ↑/↓ cycle through {p,q} presets (all hyperbolic: (p-2)(q-2) > 4).
const PRESETS = [[7, 3], [5, 4], [8, 3], [6, 4]];
let presetIdx = 0;
function setPQ(p, q) {
  uniforms.uP.value = p;
  uniforms.uQ.value = q;
  const ip = document.getElementById("p");
  const iq = document.getElementById("q");
  if (ip) { ip.value = String(p); const o = document.querySelector('[data-val="p"]'); if (o) o.textContent = String(p); }
  if (iq) { iq.value = String(q); const o = document.querySelector('[data-val="q"]'); if (o) o.textContent = String(q); }
  showPQ();
}
// setVariantCycler receives a direction (-1 / +1) from ↑/↓ and returns a label.
setVariantCycler(dir => {
  presetIdx = (presetIdx + dir + PRESETS.length) % PRESETS.length;
  const [p, q] = PRESETS[presetIdx];
  setPQ(p, q);
  return `{${p},${q}}`;
});

showPQ();

window.__diag = () => JSON.stringify({ p: uniforms.uP.value, q: uniforms.uQ.value });
