---
name: raymarched-shader-room
description: Build a mathwave room that is a GPU fragment-shader raymarcher/SDF — fractals (mandelbulb, mandelbox, menger, quaternion Julia), volumetrics (nebula), Truchet/metaballs. Single fullscreen quad, all geometry in the fragment shader, custom uCamPos orbit. Template — src/fractal.js.
---

# Raymarched shader room

**Canonical template: `src/fractal.js`** (the Mandelbulb). Copy it. Working siblings:
`mandelbox.js`, `menger.js`, `qjulia.js`, `nebula.js`, `metaballs.js`, `apollonian.js`.

## Shape

A single `THREE.PlaneGeometry(2,2)` covering clip space + an `OrthographicCamera(-1,1,1,-1,0,1)`.
All the 3D lives in the fragment shader:

```js
const uniforms = {
  uResolution: { value: new THREE.Vector2(innerWidth, innerHeight) },
  uTime: { value: 0 },
  uCamPos: { value: new THREE.Vector3(0,0,2.6) },
  /* + your fractal params */
};
const material = new THREE.ShaderMaterial({ uniforms, vertexShader: vert, fragmentShader: frag });
scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), material));
```

Vertex shader is trivial: `void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }`.

## The fragment shader

1. `vec2 uv = (gl_FragCoord.xy*2.0 - uResolution)/uResolution.y;` — aspect-correct.
2. Build a camera basis looking at the origin from `uCamPos`; form the ray `rd`.
3. **Distance estimator**: write `float DE(vec3 p)` for your surface (e.g. Mandelbox
   box-fold+sphere-fold; Menger cross-subtraction; quaternion Julia Hart DE). Cite the
   formula in a header comment — accuracy matters.
4. **March**: step `t += DE(p)*0.85` for ~`uSteps` iterations; hit when `DE < ε*t`.
5. **Normal** by the DE gradient (4-tap tetrahedron). Shade: diffuse + a little ambient
   + Fresnel/spec; cheap AO from step count.
6. **Tonemap then gamma, exactly once**: `col = col/(col+1.0); col = pow(col, vec3(0.4545));`

## Camera orbit

fractal.js drives `uCamPos` from yaw/pitch/dist with pointer drag + wheel zoom + slow
auto-spin (gated on `reducedMotion`). Reuse it verbatim.

## Aesthetic gotchas (learned the hard way)

- **Don't end on a white palette stop** — if the orbit-trap/value saturates to ~1 on
  most pixels, `palette(1.0)` paints the whole thing white. Drive hue with a smooth
  `sin(trap + p.x + …)` sweep, or rank/normalize, so the full ramp is used.
- **Near-black background, not purple.** Cool cyan halo from near-misses, not hot pink.
- **One gamma only.** If you later route through the PS1 pass, pass `srgb:false` (the
  shader already gamma-corrects) — see **ps1-pipeline**.

## Performance

Raymarching is heavy: `renderer.setPixelRatio(Math.min(devicePixelRatio, 1.3))`, cap
`uSteps`, and `const int MAX_STEPS` as a hard loop bound (GLSL needs constant bounds).
Routing through the PS1 pipeline at `scale:4` also makes it ~4× cheaper (quarter-res march).

## Wire-up

`window.__err` handler in the HTML before the module script; set
`window.__diag = () => JSON.stringify({ /* live params */ })`. On resize, update
`uResolution` (or the PS1 RT size — see ps1-pipeline). Then **verify-on-screen**.
