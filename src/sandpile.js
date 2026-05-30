// sandpile.js — the Abelian sandpile model (Bak, Tang & Wiesenfeld 1987,
// "Self-organized criticality: An explanation of the 1/f noise", PRL 59, 381).
// A 2D integer grid of "grains". Grains are dropped at the center; any cell
// holding >= 4 grains TOPPLES: it loses 4 grains and gives 1 to each of its 4
// orthogonal neighbors. Grains that fall off the boundary are lost. Toppling
// iterates until stable (no cell >= 4). The model is "Abelian": the final stable
// configuration is independent of the ORDER in which topplings happen, so a
// batched parallel relaxation reaches the same picture as any other order. The
// stable pile from dropping N grains at one point is a striking self-similar
// fractal of 0/1/2/3-grain cells.
//
// Implementation (CPU, as recommended): an Int32Array grid. Each frame we add a
// batch of grains at the center and run a bounded number of parallel relaxation
// sweeps (every eligible cell topples once per sweep), so the fractal visibly
// crystallizes outward instead of appearing frozen. The grain count is uploaded
// to a DataTexture (UnsignedByte data + needsUpdate each frame — the safe path,
// no float-texture extension needed); a display shader maps count 0/1/2/3 to
// four vivid colors on near-black.

import * as THREE from "three";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
// Fullscreen quad in clip space; the pile lives entirely in a texture.
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

// Grid presets (odd so there is one exact center cell). 257 is the default.
const SIZES = [129, 257, 511];
let sizeIdx = 1;
let N = SIZES[sizeIdx];

// Integer grain grid, a scratch buffer for parallel toppling, the display texture.
let grid = new Int32Array(N * N);
let scratch = new Int32Array(N * N);
// Grain counts (0..3 once stable) ride in the texture's red channel as unsigned
// bytes; the shader recovers the integer and picks a color.
let tex = makeTex(N);
let totalAdded = 0;   // grains dropped so far (for __diag)

function makeTex(n) {
  const t = new THREE.DataTexture(new Uint8Array(n * n * 4), n, n, THREE.RGBAFormat);
  t.minFilter = THREE.NearestFilter;
  t.magFilter = THREE.NearestFilter;
  t.needsUpdate = true;
  return t;
}

// Four-color palettes for grain counts 0,1,2,3. Index 0 is the near-black
// background. Deliberately vivid and NOT purple-dominant.
const SCHEMES = [
  ["deep",   [[0.02, 0.03, 0.06], [0.10, 0.32, 0.85], [0.05, 0.78, 0.74], [1.00, 0.72, 0.18]]], // deep blue / teal / amber
  ["ember",  [[0.03, 0.02, 0.03], [0.85, 0.16, 0.28], [1.00, 0.48, 0.12], [1.00, 0.86, 0.30]]],                      // crimson / orange / gold
  ["aurora", [[0.02, 0.04, 0.05], [0.16, 0.30, 0.92], [0.20, 0.90, 0.45], [0.30, 0.95, 0.95]]],                      // indigo / spring-green / cyan
  ["candy",  [[0.04, 0.02, 0.05], [0.95, 0.22, 0.62], [1.00, 0.78, 0.20], [0.10, 0.80, 0.78]]],                      // magenta / amber / teal
];
let scheme = 0;

const material = new THREE.ShaderMaterial({
  uniforms: {
    uTex: { value: tex },
    uAspect: { value: 1 },     // viewport aspect so the square grid is not stretched
    uC0: { value: new THREE.Color() },
    uC1: { value: new THREE.Color() },
    uC2: { value: new THREE.Color() },
    uC3: { value: new THREE.Color() },
  },
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
  `,
  fragmentShader: `
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTex;
    uniform float uAspect;
    uniform vec3 uC0, uC1, uC2, uC3;
    void main(){
      // Fit the square pile centered in the viewport, preserving aspect ratio.
      vec2 p = vUv - 0.5;
      if (uAspect >= 1.0) p.x *= uAspect; else p.y /= uAspect;
      vec2 uv = p + 0.5;
      if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        gl_FragColor = vec4(uC0, 1.0);   // background outside the grid
        return;
      }
      // Recover the integer grain count (0..3) packed in the red channel.
      float n = floor(texture2D(uTex, uv).r * 255.0 + 0.5);
      vec3 col = uC0;
      if (n > 2.5) col = uC3;
      else if (n > 1.5) col = uC2;
      else if (n > 0.5) col = uC1;
      gl_FragColor = vec4(col, 1.0);
    }
  `,
});

scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));

function applyScheme() {
  const s = SCHEMES[scheme][1];
  material.uniforms.uC0.value.setRGB(s[0][0], s[0][1], s[0][2]);
  material.uniforms.uC1.value.setRGB(s[1][0], s[1][1], s[1][2]);
  material.uniforms.uC2.value.setRGB(s[2][0], s[2][1], s[2][2]);
  material.uniforms.uC3.value.setRGB(s[3][0], s[3][1], s[3][2]);
}
applyScheme();

// Drop `count` grains onto the single center cell.
function addGrains(count) {
  const c = (((N >> 1) * N) + (N >> 1)) | 0;
  grid[c] += count;
  totalAdded += count;
}

// One parallel relaxation sweep: every cell with >= 4 grains topples exactly
// once, losing 4 and giving 1 to each orthogonal neighbor (off-grid grains are
// discarded). Returns true if any cell toppled, so we can stop early once stable.
function sweep() {
  scratch.set(grid);
  let toppled = false;
  for (let y = 0; y < N; y++) {
    const row = y * N;
    for (let x = 0; x < N; x++) {
      const i = row + x;
      if (grid[i] >= 4) {
        toppled = true;
        scratch[i] -= 4;
        if (x > 0) scratch[i - 1] += 1;
        if (x < N - 1) scratch[i + 1] += 1;
        if (y > 0) scratch[i - N] += 1;
        if (y < N - 1) scratch[i + N] += 1;
      }
    }
  }
  const tmp = grid; grid = scratch; scratch = tmp;
  return toppled;
}

// Repaint the texture's red channel from the (possibly mid-toppling) counts,
// clamped to 3 so cells momentarily holding >= 4 still read as the "full" color.
function paint() {
  const d = tex.image.data;
  for (let i = 0; i < N * N; i++) {
    let v = grid[i];
    if (v > 3) v = 3; else if (v < 0) v = 0;
    d[i * 4] = v;        // grain count in red channel; shader picks the color
    d[i * 4 + 3] = 255;  // opaque
  }
  tex.needsUpdate = true;
}

function reset() {
  grid.fill(0);
  scratch.fill(0);
  totalAdded = 0;
  paint();
}

// Swap to a different grid size: reallocate buffers + texture, then start fresh.
function rebuildGrid() {
  N = SIZES[sizeIdx];
  grid = new Int32Array(N * N);
  scratch = new Int32Array(N * N);
  tex.dispose();
  tex = makeTex(N);
  material.uniforms.uTex.value = tex;
  totalAdded = 0;
  paint();
}

function resize(w, h) {
  material.uniforms.uAspect.value = w / h;
}
onResize(renderer, null, resize);
resize(window.innerWidth, window.innerHeight);

// Growth speed: grains added per frame. Reduced-motion users start gentler.
let grainsPerFrame = reducedMotion ? 20 : 120;
// Cap toppling work per frame so a heavy pile can never hang the tab. Larger
// grids get fewer sweeps since each sweep scans more cells.
function sweepCap() { return N >= 511 ? 6 : (N >= 257 ? 12 : 24); }

bindRange("grains", (v) => { grainsPerFrame = Math.round(v); }, (v) => `${Math.round(v)}/f`);

// ↑/↓ cycle the color scheme; return the new label for the kiosk toast.
setVariantCycler((d) => {
  scheme = (scheme + d + SCHEMES.length) % SCHEMES.length;
  applyScheme();
  return SCHEMES[scheme][0];
});

// Size button cycles the grid-size preset.
const sizeBtn = document.getElementById("size");
function syncSizeLabel() { if (sizeBtn) sizeBtn.textContent = `grid: ${N}`; }
if (sizeBtn) sizeBtn.addEventListener("click", () => {
  sizeIdx = (sizeIdx + 1) % SIZES.length;
  rebuildGrid();
  syncSizeLabel();
});
syncSizeLabel();

const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", reset);

// ---------- boot ----------
reset();
liftVeil();

const meter = fpsMeter(document.getElementById("fps"));
const grainsEl = document.getElementById("grainCount");

loop((dt) => {
  meter(dt);
  addGrains(grainsPerFrame);
  // Bounded sweeps; bail out as soon as the pile is stable to save work.
  const cap = sweepCap();
  for (let s = 0; s < cap; s++) {
    if (!sweep()) break;
  }
  paint();
  if (grainsEl) grainsEl.textContent = totalAdded.toLocaleString();
  renderer.render(scene, camera);
});

window.__diag = () => JSON.stringify({ grains: totalAdded });
