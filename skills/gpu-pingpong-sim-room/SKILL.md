---
name: gpu-pingpong-sim-room
description: Build a mathwave room that is a GPU field simulation on swapped float render targets (ping-pong) — reaction-diffusion, wave/ripple equations, Schrödinger wavefunctions, Lattice-Boltzmann fluids, Lenia/neural cellular automata, Physarum. Template — src/reaction.js.
---

# GPU ping-pong simulation room

**Canonical template: `src/reaction.js`** (Gray-Scott). Working siblings: `ripple.js`,
`wavefunction.js`, `lbm.js`, `nca.js`, `physarum.js`.

## The pattern

State lives in a **float texture**. Each frame you render a **compute shader** that
reads the current state texture and writes the next state into a *second* render
target, then **swap**. Display is a separate shader that colors the latest state.

```js
const RT_OPTS = { type: THREE.HalfFloatType, format: THREE.RGBAFormat,
  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
  depthBuffer: false, stencilBuffer: false };
let rtA = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS);
let rtB = new THREE.WebGLRenderTarget(SW, SH, RT_OPTS);
// step: read rtA → write rtB → swap
function step(){
  simU.uTex.value = rtA.texture;
  renderer.setRenderTarget(rtB); renderer.render(simScene, cam);
  renderer.setRenderTarget(null);
  [rtA, rtB] = [rtB, rtA];
}
```

**Iron rule:** never read and write the same texture in one pass. Read one RT, write
the other, swap. If your state needs more than 4 channels (e.g. LBM's 9 distributions,
NCA's 16), use several RGBA RTs and **swap them all together**.

## Channels

Pack your fields into RGBA. Examples in-repo: reaction = (A,B); wavefunction =
(R, I, V, |ψ|²); LBM = 9 distributions across 3 RTs.

## Seeding

Build the initial state CPU-side into a `THREE.DataTexture` (FloatType), then **blit
it into both RTs** via a copy shader so the first read is valid.

## Stability — the #1 failure mode

Explicit schemes blow up if the timestep is too large → the field saturates → the
display pins to one color (e.g. solid yellow). Two real cases from this repo:
- **Wavefunction** (Visscher leapfrog): with the bare 5-point Laplacian `lap = l+r+d+u-4c`
  (no `/dx²`), stability needs `dt·(4 + V_max) ≲ 1`. A V≈90 well forced `dt≈0.015`.
- **Ripple/wave equation**: keep the CFL number `c·dt/dx ≤ ~0.7` or it NaNs.

**Verify numerically without a browser:** write a throwaway Node script that
reimplements one row / a small grid of your update and check the conserved quantity
(Σ|ψ|², total mass) stays bounded over thousands of steps. Pick `dt` from that, then
delete the script. Raise sub-steps-per-frame to keep motion watchable at small `dt`.
Add a `clamp()` in the shader as a safety net (not as the thing shaping the image).

## Display

A normalized field's peak is small — tune exposure (`1 - exp(-k*density)`) so a typical
value lands mid-ramp, not at 1.0. Near-black bg, cyan→magenta→amber (not purple).

## Wire-up

Run N sub-steps per frame in the loop, then render the display shader to screen.
`window.__err` handler + `window.__diag`. On resize, reallocate RTs (mirror reaction.js;
do NOT pass a callback into `onResize`'s camera slot). Then **verify-on-screen**.
