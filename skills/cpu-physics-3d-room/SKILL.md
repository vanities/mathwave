---
name: cpu-physics-3d-room
description: Build a mathwave room that runs CPU physics/simulation each frame and renders real Three.js 3D geometry — N-body, boids, cloth (Verlet), spiking neural nets, Kohonen SOM, L-systems, RL cartpole, double pendulum. PerspectiveCamera + OrbitControls. Template — src/nbody.js.
---

# CPU-physics 3D room

**Canonical template: `src/nbody.js`**. Working siblings: `boids.js`, `cloth.js`,
`spiking.js`, `som.js`, `lsystem.js`, `rl.js`, `pendulum.js`.

## Shape

A real 3D scene with a perspective camera you can orbit:

```js
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 2000);
const controls = new OrbitControls(camera, canvas);   // from three/addons
controls.enableDamping = true;
controls.autoRotate = !reducedMotion;
// lights so form reads; addGrid(scene, {...}) for a ground plane
```

Each frame: **(1)** advance the simulation on the CPU, **(2)** push results into
geometry, **(3)** `controls.update()` + render.

## Updating geometry cheaply

Allocate the geometry once; mutate its buffers in place — never rebuild per frame:

- **Points / lines** (particles, neurons, SOM sheet, L-system segments):
  write into `geometry.attributes.position.array` and set `.needsUpdate = true`.
  For per-element color, use a `color` BufferAttribute (+ `vertexColors: true`).
- **InstancedMesh** (N spheres/cubes — boids, spiking neurons): `setMatrixAt(i, m)` with
  a shared `dummy` Object3D (`dummy.position.set(...); dummy.updateMatrix()`), then
  `instanceMatrix.needsUpdate = true`. Per-instance color via `setColorAt` +
  `instanceColor.needsUpdate`. Set `frustumCulled = false` if instances spread beyond
  the base geometry's bounding sphere.
- **Deforming mesh** (cloth): rewrite the position attribute, then
  `geometry.computeVertexNormals()` so lighting follows the folds; render `DoubleSide`.

## Footguns

- `THREE.Object3D.position` / `.scale` / `.quaternion` are **read-only references** —
  use `.set()` / `.copy()`, never `obj.position = v`.
- Integrate with a small fixed `dt` (clamp `dt ≤ 1/30`); sub-step stiff systems
  (Izhikevich neurons, cloth constraints) to avoid blow-up.
- Cap work per frame (particle/synapse/constraint counts, iterations) so the tab holds 60fps.
- No temporal-dead-zone: don't reference a `const`/`let` inside a `bindRange` callback
  that runs (synchronously, on bind) before the declaration.

## Learning/agent rooms

For "it learns in front of you" (RL), run many fast training steps **headless** in the
background and render the current best policy at real-time speed — don't render every
training rollout. (CartPole: Cross-Entropy Method over a linear policy is stable and
solves in seconds.)

## Wire-up

`setVariantCycler` for ↑/↓ presets (topology, target distribution, species).
`window.__err` handler + `window.__diag`. `onResize(renderer, camera)`. Then **verify-on-screen**.
