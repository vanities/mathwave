---
name: mathwave-overview
description: Conventions for adding a room to the mathwave Three.js gallery — file layout, the buildless ES-module/import-map setup, kiosk navigation, the shared common.js engine, and the ROOMS registry. Use this first whenever creating or modifying any mathwave room.
---

# mathwave — room conventions

mathwave is a **buildless** Three.js gallery: ES modules + import maps, Three.js
r0.169 + math.js from CDN, **no build step**. Serve over HTTP (`python3 -m http.server`),
never `file://`. It's a **kiosk**, not a website — `index.html` redirects straight
into the first room; there is no landing page or menu.

## Every room is two files

```
pieces/<room>.html   the page (HUD chrome + importmap + the module <script>)
src/<room>.js        the room logic (imports from ./common.js)
```

Copy an existing pair as your template — pick by room type:
- fragment-shader SDF/raymarch → see **raymarched-shader-room** (`src/fractal.js`)
- GPU field sim on float textures → see **gpu-pingpong-sim-room** (`src/reaction.js`)
- CPU physics + real 3D geometry → see **cpu-physics-3d-room** (`src/nbody.js`)
- CPU integer/cell grid → see **grid-cells-room** (`src/wolfram.js`)

## Hard conventions (copy them exactly — subagents drift here)

- Canvas is `<canvas id="scene">`. `common.js`'s renderer, recorder, FPS meter and
  kiosk nav all key off `#scene`. A different id silently breaks all of them.
- CSS is `../assets/base.css` (there is **no** `../base.css`).
- Import map (in every piece `<head>`):
  ```html
  <script type="importmap">
  { "imports": { "three": "https://unpkg.com/three@0.169.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.169.0/examples/jsm/" } }
  </script>
  ```
- Loading veil: `<div class="veil">…</div>` + call `liftVeil()` once the first frame
  is up. Fonts: Monoton, DotGothic16, VT323 (preconnect + the Google Fonts link).
- HUD structure: `.hud`/`.hud-top`, `.panel` + `.field` + `[data-val="<id>"]`,
  `.readout` (with `<b id="fps">–</b>`), `.hint`. Back-link `<a class="hud-back" href="parametric.html">`.

## What you import from common.js

`makeRenderer(canvas)`, `onResize(renderer, camera, extra?)`, `loop(fn)` where
`fn(dt, elapsed)`, `fpsMeter(node)` → `meter(dt)`, `liftVeil()`,
`bindRange(id, onInput, fmt)` (3rd arg is a **formatter fn**, not a default),
`setVariantCycler(fn)` where `fn(dir)` applies the ↑/↓ change and returns a label
string, `reducedMotion` (a boolean **const**), and the vaporwave set-dressing
helpers `ramp`, `addGrid(scene, {...})`, `addSun(scene, {...})`.

## Register the room in the kiosk

The single source of truth for nav order + the batch recorder is the `ROOMS` array
near the top of `src/common.js`. **Add `"<room>.html"`** there or the room won't
appear in ← → navigation or "record all". (Subagents are told NOT to touch this —
the orchestrator wires it after the room is built.)

## Controls every room should expose

- ↑/↓ cycles an in-room variation via `setVariantCycler`.
- Sliders via `bindRange` (mirror the value into `[data-val="<id>"]`).
- `R` records the canvas (handled globally by common.js — don't bind `R` yourself).
- Respect `reducedMotion` (slow or freeze auto-motion).

## Aesthetic

Adam rejected vaporwave **purple** and a full **PS1** treatment of abstract rooms.
Default: **near-black background**, vivid **cyan→green→amber→magenta** accents, clean.
The PS1 look (`src/ps1.js`) is for rooms with real 3D *form* (fractals), not abstract
data clouds — see **ps1-pipeline**.

## Always finish with verification

Never claim a room works from `node --check` alone — see **verify-on-screen**.
