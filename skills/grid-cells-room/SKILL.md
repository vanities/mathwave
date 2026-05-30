---
name: grid-cells-room
description: Build a mathwave room that simulates a CPU integer/cell grid and displays it via a DataTexture + fullscreen display shader — Wolfram CA, Abelian sandpile, Hopfield network, Conway-style automata. Template — src/wolfram.js (+ the fullscreen-quad display pattern from src/fractal.js).
---

# Grid-cells room (CPU grid → DataTexture → shader)

**Templates: `src/wolfram.js`** for the grid/stepping pattern, **`src/fractal.js`** for
the fullscreen-quad display shader. Working siblings: `sandpile.js`, `hopfield.js`.

## The pattern

Keep the simulation as a **typed array on the CPU**, upload it to a `DataTexture` each
frame, and draw it with one fullscreen display shader that maps cell values to colors.
Use this (instead of GPU ping-pong) when the rule is integer/branchy/global and easier
to express in JS — sandpile toppling, Hopfield `sign(Σ W·s)`, CA lookups.

```js
const W = 257, H = 257;
const grid = new Int32Array(W*H);              // your state
const pix  = new Uint8Array(W*H*4);            // RGBA upload buffer
const tex  = new THREE.DataTexture(pix, W, H, THREE.RGBAFormat); // UnsignedByteType
tex.minFilter = tex.magFilter = THREE.NearestFilter;            // crisp cells

// each frame: step the grid, write colors into pix, then:
tex.needsUpdate = true;
```

Display it on a `PlaneGeometry(2,2)` + `OrthographicCamera`, sampling `tex`. Either
write final colors into `pix` in JS, or upload raw values and map to a palette in the
shader (more flexible). Letterbox to a square in the shader so the grid isn't stretched.

## Keep it bounded (don't hang the tab)

- Cap work per frame: sandpile topple **sweeps per frame** (e.g. 24/12/6 by grid size),
  with an early-out when stable. Re-allocate + dispose buffers on grid-size change.
- For animated growth, add a *batch* of work per frame (drop K grains, run a few sweeps)
  rather than running to completion in one frame — it should *crystallize* on screen.

## Model notes (be accurate; cite in header)

- **Sandpile** (Bak–Tang–Wiesenfeld): cell ≥4 topples — subtract 4, +1 to each
  orthogonal neighbor; boundary grains are lost. The stable field is a 0/1/2/3 fractal.
- **Hopfield**: Hebbian `W_ij = (1/N)Σ_μ ξ_i^μ ξ_j^μ`, `W_ii=0`; recall by
  `s_i ← sign(Σ_j W_ij s_j)`; energy `E = -½ Σ W s s` descends. Capacity ≈ 0.14N, so
  store few patterns or recall fails.

## Color

Distinct vivid colors per cell value on near-black (deep blue / teal / amber / magenta),
**not purple-dominant**.

## Wire-up

`bindRange` for growth/noise/speed; `setVariantCycler` for grid size / pattern / scheme;
a reset button. `window.__err` handler + `window.__diag`. Then **verify-on-screen**.
