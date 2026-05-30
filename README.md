# ＭＡＴＨＷＡＶＥ 数学波

> a vaporwave gallery of mathematical art

Four neon-soaked rooms, each carved from a single formula and a GPU.
No framework, no build step — just serve the folder and open it.

| Room | What it is | Tech |
|------|-----------|------|
| **01 · 曲面 Graphed Surfaces** | Type `z = f(x, y, t)` and it blooms into a living, neon-shaded surface over a grid floor. Use `t` to animate. | Three.js + [math.js] |
| **02 · 無限 The Mandelbulb** | A 3D fractal raymarched entirely on the GPU. Orbit/zoom into infinite detail, painted in pink + cyan. | Three.js + raw GLSL |
| **03 · 混沌 Strange Attractors** | Lorenz, Aizawa, Thomas, Halvorsen, Dadras — chaotic ODEs drawn as glowing ribbons. | Three.js + RK4 |
| **04 · 故障 Pixel Sort** | A generative neon field torn apart and re-ordered pixel by pixel — Kim Asendorf's glitch technique. | Canvas2D pixel sorting |

The whole thing wears a CRT scanline + grain overlay, a sliced retro sun, and an
outrun grid horizon. Type is three retro voices: **Monoton** (the neon sign),
**DotGothic16** (pixel headings + 日本語), and **VT323** (the terminal).

## Run it

It uses ES-module import maps, so it must be **served over HTTP** (opening files via
`file://` won't work — browsers block module imports there).

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

Any static server works (`npx serve .`, `php -S localhost:8000`, …). Three.js r169 and
math.js load from a CDN via the import map in each HTML file — nothing to install.

## How it's wired

```
index.html              gallery landing (neon outrun-terrain hero)
assets/base.css         the whole vaporwave design system + CRT overlay
src/common.js           shared helpers: renderer, loop, FPS, neon ramp,
                        grid floor, sliced sun, and the CRT injector
src/hero.js             landing-page hero terrain
pieces/parametric.html ─┐  src/parametric.js   room 01
pieces/fractal.html    ─┤  src/fractal.js      room 02  (Mandelbulb DE in GLSL)
pieces/attractor.html  ─┤  src/attractor.js    room 03
pieces/pixelsort.html  ─┘  src/pixelsort.js    room 04  (after kim asendorf)
```

### The honest tech note

Three.js isn't a "math library" — it's the **renderer** (a WebGL wrapper). The
mathematics comes from three places stacked on top of it:

1. **You / math.js** — evaluate the functions (room 01 parses arbitrary expressions).
2. **Three.js** — turn the numbers into meshes, lines, and points (rooms 01 & 03).
3. **GLSL fragment shaders** — the heavy artillery: room 02's fractal is computed
   per-pixel on the GPU with a signed-distance raymarcher.

Room 04 leaves 3D entirely: it's a CPU image operation (pixel sorting) on a `<canvas>`,
the same family of technique behind Kim Asendorf's *ASDF Pixel Sort* and *Mountain Tour*.

## Ideas to take it further

- Room 01: add true `u,v` parametric mode (Klein bottles, supershapes), not just height fields.
- Room 02: swap the Mandelbulb DE for Julia sets, Menger sponges, or Apollonian gaskets.
- Room 03: render the attractor as a moving "comet" trail instead of a full ribbon.
- Room 04: feed a real photo into the sorter; add datamosh / RGB-shift passes.
- Ship it: graduate to a Vite + TypeScript project and deploy to Vercel.

[math.js]: https://mathjs.org

---

Made with Three.js, raw GLSL, and a CRT. 故障 forever.
