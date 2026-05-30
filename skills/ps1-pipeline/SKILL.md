---
name: ps1-pipeline
description: Apply the PlayStation-1 rendering aesthetic to a mathwave room — low-res framebuffer + nearest upscale + 4x4 Bayer ordered dither + 15-bit color crunch, plus optional vertex snapping. Reusable module src/ps1.js. Best on rooms with real 3D FORM (fractals), not abstract data clouds.
---

# PS1 pipeline

**Module: `src/ps1.js`** — already built, color-correct, reusable. Two exports.

## When to use it

It looks great on rooms with real 3D **form** — the mandelbulb, mandelbox, quaternion
Julia, apollonian. It turns abstract data clouds (the transformer's 1,120 cubes) into
mush — Adam rejected it there. Use judgment; default rooms stay clean (no PS1).

## makePS1Pipeline — the post-process

Renders the scene into a 1/`scale`-res target, upscales NEAREST (chunky pixels), then a
fullscreen pass does 4×4 Bayer ordered dithering + quantization to `levels` (≈15-bit).
Shaped like an `EffectComposer` (`.render()` / `.setSize()`), so it drops in where you'd
call `renderer.render`.

```js
import { makePS1Pipeline } from "./ps1.js";
const ps1 = makePS1Pipeline(renderer, scene, camera, { scale: 4, levels: 32, srgb: false });
// loop: ps1.render();   // instead of renderer.render(scene, camera)
// resize: ps1.setSize();
```

### The `srgb` flag — read this or everything turns red

The low-res target is **linear-light**; a raw `ShaderMaterial` skips Three's output
encoding. The pass therefore applies linear→sRGB by default (`srgb:true`).

- Lit `MeshStandard`/normal Three scenes → leave `srgb: true`.
- A fragment shader that **already gamma-corrects its own output** (every raymarch room
  in this repo ends with `pow(col, vec3(0.4545))`) → pass **`srgb: false`** or you'll
  double-gamma and the image goes red/dark.

### Bonus: raymarch rooms get ~4× cheaper

At `scale:4` you raymarch a quarter of the pixels. **Sync your shader's resolution
uniform to the low-res RT**, or the uv is wrong:

```js
const syncRes = () => uniforms.uResolution.value.set(ps1.renderTarget.width, ps1.renderTarget.height);
syncRes();
onResize(renderer, camera, () => { ps1.setSize(); syncRes(); });
```

## ps1ify — vertex snapping (the wobble)

For lit/mesh scenes, `ps1ify(material, { snap: 220 })` injects clip-space vertex
quantization via `onBeforeCompile` → the iconic PS1 vertex jitter. Works on any
material (cubes, grid lines). Smaller `snap` = chunkier wobble.

## Heavy-distance fog

PS1 hid draw distance with heavy fog. Pair the pass with `scene.fog` in a near-black /
murky tone (not purple) for the full effect.

Always **verify-on-screen** after wiring — and remember a cache-bust (`?v=`) on the
module script tag so the browser reloads your edits.
