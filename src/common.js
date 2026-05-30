// common.js — shared helpers for every room in ＭＡＴＨＷＡＶＥ.
// Renderer, resize, rAF loop, FPS, the neon palette, plus the
// vaporwave set-dressing (grid floor + sliced sun) and the CRT overlay.

import * as THREE from "three";

export const reducedMotion =
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// --- drop the CRT scanline/grain overlay onto any page that imports us ---
(function injectCRT() {
  const add = () => {
    if (document.querySelector(".crt")) return;
    const d = document.createElement("div");
    d.className = "crt";
    d.setAttribute("aria-hidden", "true");
    document.body.appendChild(d);
  };
  if (document.body) add();
  else window.addEventListener("DOMContentLoaded", add);
})();

// --- renderer wired to a #scene canvas, DPR-capped for perf ---
export function makeRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x0a0118, 1);
  return renderer;
}

// --- keep camera + renderer synced to the viewport ---
export function onResize(renderer, camera, extra) {
  const handler = () => {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h);
    if (camera && camera.isPerspectiveCamera) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    if (extra) extra(w, h);
  };
  window.addEventListener("resize", handler);
  return handler;
}

// --- clean rAF loop with delta + elapsed time ---
export function loop(fn) {
  let last = performance.now(), elapsed = 0, raf = 0;
  const tick = (now) => {
    const dt = Math.min((now - last) / 1000, 1 / 20);
    last = now; elapsed += dt;
    fn(dt, elapsed);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}

// --- FPS meter that writes into a node ---
export function fpsMeter(node) {
  let frames = 0, acc = 0, fps = 0;
  return (dt) => {
    frames++; acc += dt;
    if (acc >= 0.5) { fps = Math.round(frames / acc); frames = 0; acc = 0; if (node) node.textContent = fps; }
    return fps;
  };
}

// --- fade out the loading veil ---
export function liftVeil() {
  const veil = document.querySelector(".veil");
  if (!veil) return;
  requestAnimationFrame(() => {
    veil.classList.add("gone");
    setTimeout(() => veil.remove(), 700);
  });
}

// --- bind a range input to a callback, mirroring its value into .val ---
export function bindRange(id, onInput, fmt = (v) => v) {
  const el = document.getElementById(id);
  if (!el) return null;
  const out = document.querySelector(`[data-val="${id}"]`);
  const sync = () => {
    const v = parseFloat(el.value);
    if (out) out.textContent = fmt(v);
    onInput(v);
  };
  el.addEventListener("input", sync);
  sync();
  return el;
}

// Vaporwave color ramp: indigo → purple → magenta → cyan → mint.
// Returns [r,g,b] in 0..1 for a 0..1 input.
const RAMP = [
  [0.10, 0.02, 0.24],
  [0.45, 0.10, 0.62],
  [1.00, 0.18, 0.60],
  [0.16, 0.82, 0.96],
  [0.55, 1.00, 0.78],
];
export function ramp(t) {
  t = Math.min(Math.max(t, 0), 1) * (RAMP.length - 1);
  const i = Math.floor(t), f = t - i;
  const a = RAMP[i], b = RAMP[Math.min(i + 1, RAMP.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// --- neon perspective grid floor (the outrun motif) ---
export function addGrid(scene, { size = 120, divisions = 60, y = -7 } = {}) {
  const grid = new THREE.GridHelper(size, divisions, 0x2be4ff, 0xff2e97);
  grid.position.y = y;
  grid.material.transparent = true;
  grid.material.opacity = 0.4;
  grid.material.depthWrite = false;
  scene.add(grid);
  return grid;
}

// --- a glowing sliced sun as a far-away additive sprite ---
export function addSun(scene, { scale = 60, position = [0, 14, -90] } = {}) {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(128, 128, 8, 128, 128, 128);
  grad.addColorStop(0, "rgba(255,243,168,1)");
  grad.addColorStop(0.35, "rgba(255,159,90,0.95)");
  grad.addColorStop(0.7, "rgba(255,46,151,0.55)");
  grad.addColorStop(1, "rgba(255,46,151,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  // slice it
  g.globalCompositeOperation = "destination-out";
  for (let i = 0; i < 6; i++) {
    const yy = 150 + i * 16;
    g.fillRect(0, yy, 256, 4 + i * 1.5);
  }
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })
  );
  sprite.scale.set(scale, scale, 1);
  sprite.position.set(...position);
  scene.add(sprite);
  return sprite;
}
