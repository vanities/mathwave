// pixelsort.js — a homage to Kim Asendorf's ASDF Pixel Sort.
// Generate a neon plasma field, then re-order runs of pixels by brightness
// wherever the brightness crosses a threshold. Chunky on purpose (CRT pixels).

import { loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion } from "./common.js";

const canvas = document.getElementById("scene");
canvas.style.imageRendering = "pixelated";
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const SCALE = 3; // internal pixels are 3× chunky → fast + on-aesthetic
let W = 0, H = 0;
let source = null;     // pristine ImageData
let work = null;       // sorted copy we draw each frame
let bright = null;     // Float32Array of source luma (0..1)

// ---------- state ----------
let threshold = 0.5;
let orientation = "vertical"; // vertical | horizontal
let order = "asc";            // asc | desc
let sortKey = "luma";         // luma | hue | sat
let animating = !reducedMotion;
let seed = 1.7;

const luma = (r, g, b) => (0.299 * r + 0.587 * g + 0.114 * b) / 255;

function keyOf(r, g, b) {
  if (sortKey === "luma") return luma(r, g, b);
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  if (sortKey === "sat") return mx === 0 ? 0 : d / mx;
  // hue
  if (d === 0) return 0;
  let h;
  if (mx === r) h = ((g - b) / d) % 6;
  else if (mx === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  h *= 60; if (h < 0) h += 360;
  return h / 360;
}

// ---------- generate the neon field ----------
function generate() {
  const data = source.data;
  let p = 0, q = 0;
  for (let y = 0; y < H; y++) {
    const ny = y / H;
    for (let x = 0; x < W; x++) {
      const nx = x / W;
      let v =
        Math.sin(nx * 5.0 + seed) +
        Math.sin(ny * 7.0 - seed * 0.6) +
        Math.sin((nx + ny) * 6.0 + seed * 1.3) +
        Math.sin(Math.hypot(nx - 0.5, ny - 0.5) * 16.0 - seed * 1.1);
      v = (v + 4) / 8;                         // → 0..1
      v += 0.06 * Math.sin(ny * 130.0 + seed); // fine grain for sortable structure
      v = Math.min(Math.max(v, 0), 1);
      const c = ramp(v);
      const r = (c[0] * 255) | 0, g = (c[1] * 255) | 0, b = (c[2] * 255) | 0;
      data[p] = r; data[p + 1] = g; data[p + 2] = b; data[p + 3] = 255;
      bright[q] = luma(r, g, b);
      p += 4; q += 1;
    }
  }
}

// sort one run of pixels (array of base byte-offsets) in `work` by key
function sortRun(d, offsets) {
  const n = offsets.length;
  const items = new Array(n);
  for (let i = 0; i < n; i++) {
    const o = offsets[i];
    items[i] = { k: keyOf(d[o], d[o + 1], d[o + 2]), r: d[o], g: d[o + 1], b: d[o + 2] };
  }
  items.sort(order === "asc" ? (a, b) => a.k - b.k : (a, b) => b.k - a.k);
  for (let i = 0; i < n; i++) {
    const o = offsets[i], it = items[i];
    d[o] = it.r; d[o + 1] = it.g; d[o + 2] = it.b;
  }
}

// run a full sort pass into `work`, threshold may wave across the image
function sortPass(time) {
  work.data.set(source.data);
  const d = work.data;
  const amp = animating ? 0.12 : 0;

  if (orientation === "vertical") {
    for (let x = 0; x < W; x++) {
      const thr = threshold + Math.sin(time * 0.8 + x * 0.03) * amp;
      let y = 0;
      while (y < H) {
        while (y < H && bright[y * W + x] < thr) y++;
        const start = y;
        while (y < H && bright[y * W + x] >= thr) y++;
        if (y - start > 1) {
          const offs = [];
          for (let yy = start; yy < y; yy++) offs.push((yy * W + x) * 4);
          sortRun(d, offs);
        }
      }
    }
  } else {
    for (let y = 0; y < H; y++) {
      const thr = threshold + Math.sin(time * 0.8 + y * 0.03) * amp;
      const row = y * W;
      let x = 0;
      while (x < W) {
        while (x < W && bright[row + x] < thr) x++;
        const start = x;
        while (x < W && bright[row + x] >= thr) x++;
        if (x - start > 1) {
          const offs = [];
          for (let xx = start; xx < x; xx++) offs.push((row + xx) * 4);
          sortRun(d, offs);
        }
      }
    }
  }
  ctx.putImageData(work, 0, 0);
}

// ---------- sizing ----------
function resize() {
  W = Math.max(2, Math.ceil(window.innerWidth / SCALE));
  H = Math.max(2, Math.ceil(window.innerHeight / SCALE));
  canvas.width = W; canvas.height = H;
  source = ctx.createImageData(W, H);
  work = ctx.createImageData(W, H);
  bright = new Float32Array(W * H);
  generate();
  document.getElementById("res").textContent = `${W}×${H}`;
}
window.addEventListener("resize", resize);

// ---------- panel ----------
const DIRS = [["↓", "vertical", "asc"], ["↑", "vertical", "desc"], ["→", "horizontal", "asc"], ["←", "horizontal", "desc"]];
const dirsWrap = document.getElementById("dirs");
DIRS.forEach(([label, o, ord], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => {
    orientation = o; order = ord;
    dirsWrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === b));
  });
  dirsWrap.appendChild(b);
});

const KEYS = [["luma", "luma"], ["hue", "hue"], ["sat", "sat"]];
const keysWrap = document.getElementById("keys");
KEYS.forEach(([label, k], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => {
    sortKey = k;
    keysWrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === b));
  });
  keysWrap.appendChild(b);
});

const thrval = document.getElementById("thrval");
bindRange("thr", (v) => { threshold = v; thrval.textContent = v.toFixed(2); }, (v) => v.toFixed(2));

const animBtn = document.getElementById("animate");
animBtn.classList.toggle("active", animating);
animBtn.addEventListener("click", () => { animating = !animating; animBtn.classList.toggle("active", animating); });

document.getElementById("regen").addEventListener("click", () => { seed += 1.37; generate(); });

// ---------- boot ----------
resize();
liftVeil();
const meter = fpsMeter(document.getElementById("fps"));

loop((dt, elapsed) => {
  meter(dt);
  // when paused, only re-sort if something might have changed — but cheap enough to always run
  sortPass(animating ? elapsed : 0);
});
