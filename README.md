# ＭＡＴＨＷＡＶＥ 数学波

> a vaporwave gallery of mathematical art — built to be filmed

Fifteen neon-soaked rooms, each carved from a single formula and a GPU.
No framework, no build step — just serve the folder and open it.

It's a **kiosk**, not a website. You boot straight into a room; there's no menu.

```
←  →    walk between rooms
↑  ↓    cycle the variation inside the room (preset / rule / system / algorithm)
M       show / hide the interface (hidden by default)
R       record the live canvas to a video  ·  1–5 record a timed clip (10/15/20/30s)
A       record ALL rooms hands-free — walks every room, downloads a clip each
Esc     jump back to the first room
```

**Record everything at once:** press **A** (or the **⏺ ALL** button). It picks the
current duration (∞ → 12s), then visits every room in turn — warming up, recording a
clip, downloading it as `mathwave-NN-room.webm/mp4`, and advancing automatically. A
progress HUD shows `REC ALL · 7/29 · life`. The browser asks once to "allow multiple
downloads" — say yes and walk away; ~7 minutes later you have a clip of all 29 rooms.
(Esc cancels.)

The **R** recorder prefers **MP4** so clips upload straight to X / Instagram (Safari makes
real `.mp4`; Chrome/Firefox fall back to `.webm`). The duration picker auto-stops and
downloads. That's the whole point — hit record, let the math run, walk away with a clip.

## The rooms

| # | Room | What it is | Tech |
|---|------|-----------|------|
| 01 · 曲面 | **Graphed Surfaces** | Type `z = f(x,y,t)`; it blooms into a living surface. | Three.js + [math.js] |
| 02 · 無限 | **The Mandelbulb** | A 3D fractal raymarched per-pixel on the GPU. | raw GLSL |
| 03 · 混沌 | **Strange Attractors** | Lorenz, Aizawa, Thomas… chaotic ODEs as glowing ribbons. | RK4 |
| 04 · 位相 | **Hamiltonian Phase Space** | Energy surface + trajectories + phase-portrait shadow; q and p evolve together. | RK4 |
| 05 · 流れ | **3D Vector Field** | Thousands of particles advected through `V(x,y,z)` as speed-colored streaks. | particles |
| 06 · 勾配 | **Gradient Descent** | SGD, momentum, Adam **and simulated annealing** race down a 3D loss landscape — watch annealing tunnel out of local minima. | optimizers |
| 07 · 生命 | **3D Game of Life** | Conway in 3D — Carter Bays' 26-neighbor rules as hollow voxel shells. | InstancedMesh |
| 08 · 反応 | **Reaction-Diffusion** | Gray-Scott Turing patterns — coral, mitosis, mazes — alive on the GPU. | GLSL ping-pong |
| 09 · 次元 | **Flatland** | A 4D tesseract turning through 3-space, with a 2D cross-section slice. | 4D projection |
| 10 · 整列 | **Sorting** | Bubble/quick/heap… racing across a neon bar landscape, looping forever. | generators |
| 11 · 立体整列 | **3D Sorting** | A cube of voxels on a Morton curve, shuffled then sorted into a 3D gradient. | InstancedMesh |
| 12 · 反転 | **Sphere Eversion** | Turning a sphere inside-out; front/back colored so you see inside become outside. | front-face shader |
| 13 · 幻覚 | **EarthBound Backgrounds** | MOTHER 2 battle BGs — HDMA sine-row distortion + palette cycling, auto-randomizing. | GLSL |
| 14 · 故障 | **Pixel Sort** | A neon field torn apart and re-ordered pixel by pixel. | Canvas2D |
| 15 · 注意 | **The Transformer** | *Attention Is All You Need* — multi-head attention arcs pulsing layer by layer, plus a live **abliteration** toggle that removes the refusal direction from the residual stream. | toy QKV |

The whole thing wears a CRT scanline + grain overlay, a sliced retro sun, and an outrun
grid horizon. Type is three retro voices: **Monoton** (the neon sign), **DotGothic16**
(pixel headings + 日本語), and **VT323** (the terminal).

## Run it

ES-module import maps, so it must be **served over HTTP** (`file://` is blocked):

```bash
python3 -m http.server 8000
# open http://localhost:8000  → it redirects straight into room 01
```

Three.js r169 and math.js load from CDN via the import map in each page — nothing to install.

## Notes on the math

- **Three.js isn't a math library** — it's the renderer (a WebGL wrapper). The math is what
  you stack on top: you/math.js evaluate functions, Three.js turns numbers into meshes/lines/
  points, and **GLSL fragment shaders** do the per-pixel heavy lifting (fractal, reaction-
  diffusion, EarthBound).
- **3D Game of Life**: 2D Conway is *B3/S23* (8 neighbors). In 3D each cell has **26**
  neighbors, so rules are `survive/born` ranges; Carter Bays' **5766** is the canonical 3D
  analogue and even has gliders.
- **3D sorting**: yes, you can sort a 3D array — you sort the linear ordering a space-filling
  (Morton) curve imposes on the cube, so sorted-index continuity becomes a smooth 3D gradient.
- **Abliteration** (room 15): refusal in an LLM is mediated by ~one direction in the residual
  stream; abliteration projects hidden states onto the plane orthogonal to it. Here it's a toy
  3D cloud you watch collapse off the refusal axis — the geometry of the real technique.

## How it's wired

```
index.html              redirects into pieces/parametric.html (no landing page)
assets/base.css         vaporwave design system + CRT overlay + REC + kiosk-nav styles
src/common.js           renderer, loop, FPS, neon ramp, grid, sun, CRT, recorder,
                        kiosk navigation (← → rooms, ↑ ↓ variation, M, R), setVariantCycler
pieces/<room>.html  ─┐
src/<room>.js       ─┘  one pair per room (15 rooms)
```

## Roadmap

- GPU the 3D automaton (3D texture) to push past 30³.
- Real-image input for the pixel sorter; datamosh / RGB-shift passes.
- A "cinematic auto-tour" mode that visits every room hands-free for long recordings.
- Graduate to Vite + TypeScript and deploy to Vercel.

[math.js]: https://mathjs.org

---

Made with Three.js, raw GLSL, and a CRT. 故障 forever.
