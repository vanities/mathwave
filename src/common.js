// common.js — shared helpers for every room in ＭＡＴＨＷＡＶＥ.
// Renderer, resize, rAF loop, FPS, the neon palette, the vaporwave
// set-dressing (grid floor + sliced sun), the CRT overlay, the .webm
// video recorder, and kiosk navigation (arrow keys + M to toggle UI).

import * as THREE from "three";

export const reducedMotion =
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// true when the user is typing in a field — so hotkeys (R, M, arrows) don't fire
export function isTyping() {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

// rooms register a variant cycler so ↑/↓ flips presets in-place.
// fn(dir) should apply the change and return the new variant's label (a string).
let variantCycler = null;
export function setVariantCycler(fn) { variantCycler = fn; }

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
    preserveDrawingBuffer: true, // lets captureStream/screenshot read the buffer
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

// the canonical room order — shared by the kiosk nav and the batch recorder
const ROOMS = [
  "parametric.html", "fractal.html", "attractor.html", "hamiltonian.html",
  "vectorfield.html", "gradient.html", "life.html", "reaction.html",
  "flatland.html", "sorting.html", "sorting3d.html", "eversion.html",
  "earthbound.html", "pixelsort.html", "transformer.html",
  "orbitals.html", "nbody.html", "ising.html",
  "grokking.html", "embeddings.html",
  "pendulum.html", "diffusion.html", "superposition.html",
  "boids.html", "epicycles.html", "hopf.html",
  "bloch.html", "wolfram.html", "physarum.html",
  "chladni.html", "kuramoto.html", "newton.html", "mandelbrot.html",
  "blackhole.html", "phyllotaxis.html", "dla.html",
  "attractorsong.html", "gpuflow.html", "machineelves.html",
  "mandelbox.html", "menger.html", "nebula.html", "metaballs.html",
  "wavefunction.html", "lbm.html", "sandpile.html", "ripple.html", "cloth.html",
  "lsystem.html", "hyperbolic.html", "apollonian.html", "qjulia.html", "primes.html",
  "nca.html", "som.html", "spiking.html", "hopfield.html", "rl.html",
  "hillclimb.html", "minimax.html", "genetic.html", "antcolony.html", "astar.html",
  "magnetosphere.html", "flood.html",
];

// ============================================================
// VIDEO RECORDER — capture the "plays" to a downloadable clip.
// MP4 is preferred so the file uploads straight to X / Instagram
// (Safari produces real .mp4; Chrome/Firefox fall back to .webm).
// A duration picker (∞ / 10 / 15 / 20 / 30s) auto-stops + downloads.
// Click the REC pill, press R, or press 1–5 to start a timed clip.
// ============================================================
(function attachRecorder() {
  const setup = () => {
    const canvas = document.getElementById("scene");
    if (!canvas || !("MediaRecorder" in window) || !canvas.captureStream) return;

    const wrapEl = document.createElement("div");
    wrapEl.className = "rec-wrap";

    const btn = document.createElement("button");
    btn.className = "rec";
    btn.innerHTML = '<span class="rec-dot"></span><span class="rec-label">REC</span>';
    btn.title = "Record this play (R) · pick a length on the right";
    wrapEl.appendChild(btn);

    // duration chips — ∞ means record until you press R/click again
    const DURATIONS = [["∞", 0], ["10s", 10], ["15s", 15], ["20s", 20], ["30s", 30]];
    let durSec = 0;
    const durEls = DURATIONS.map(([lab, s], i) => {
      const d = document.createElement("button");
      d.className = "rec-dur" + (i === 0 ? " active" : "");
      d.textContent = lab;
      d.title = s ? `Record a ${s}s clip` : "Record until stopped";
      d.addEventListener("click", () => {
        durSec = s;
        durEls.forEach((e) => e.classList.toggle("active", e === d));
      });
      wrapEl.appendChild(d);
      return d;
    });

    // "⏺ ALL" — record a clip of every room, hands-free
    const allBtn = document.createElement("button");
    allBtn.className = "rec-all";
    allBtn.textContent = "⏺ ALL";
    allBtn.title = "Record a clip of EVERY room, hands-free (A)";
    allBtn.addEventListener("click", () => beginBatch());
    wrapEl.appendChild(allBtn);

    document.body.appendChild(wrapEl);

    let recorder = null, chunks = [], t0 = 0, timer = 0, autostop = 0;
    const label = () => btn.querySelector(".rec-label");

    // ext + mime: prefer mp4 (X-friendly), else webm
    const pickMime = () => {
      const types = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
      return types.find((t) => { try { return MediaRecorder.isTypeSupported(t); } catch { return false; } }) || "";
    };

    let pendingName = null, pendingThen = null;
    const start = (opts = {}) => {
      const dur = opts.dur != null ? opts.dur : durSec;
      pendingName = opts.name || null;
      pendingThen = opts.then || null;
      const stream = canvas.captureStream(60);
      // if a sonified room has audio running, fold its sound into the recording
      try {
        if (window.__mwAudioStream) { const a = window.__mwAudioStream(); if (a) a.getAudioTracks().forEach((t) => stream.addTrack(t)); }
      } catch (e) {}
      const mime = pickMime();
      try {
        recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 12_000_000 } : undefined);
      } catch (e) { recorder = new MediaRecorder(stream); }
      chunks = [];
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        const type = (chunks[0] && chunks[0].type) || mime || "video/webm";
        const ext = type.includes("mp4") ? "mp4" : "webm";
        const blob = new Blob(chunks, { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        const base = (document.title.split("—")[0] || "mathwave").trim().replace(/[^\w]+/g, "-").toLowerCase() || "mathwave";
        const fname = pendingName ? `mathwave-${pendingName}` : `${base}-${stamp}`;
        a.href = url; a.download = `${fname}.${ext}`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 4000);
        const cb = pendingThen; pendingThen = null; pendingName = null;
        if (cb) setTimeout(cb, 600);   // let the download settle before advancing
      };
      recorder.start();
      t0 = performance.now();
      btn.classList.add("on");
      document.body.classList.add("recording");   // keep REC visible even with chrome hidden
      timer = setInterval(() => {
        const el = (performance.now() - t0) / 1000;
        label().textContent = dur ? `${Math.max(0, dur - el).toFixed(1)}s` : `${el.toFixed(1)}s`;
      }, 100);
      if (dur) autostop = setTimeout(stop, dur * 1000);
    };

    const stop = () => {
      clearTimeout(autostop);
      if (recorder && recorder.state !== "inactive") recorder.stop();
      clearInterval(timer);
      btn.classList.remove("on");
      document.body.classList.remove("recording");
      label().textContent = "REC";
    };

    const toggle = () => (recorder && recorder.state === "recording" ? stop() : start());
    btn.addEventListener("click", toggle);

    // ---- batch: record EVERY room hands-free ----
    // Stores a queue in sessionStorage, then each room auto-records for `dur`
    // seconds, downloads its clip (NN-room.webm/mp4), and navigates to the next.
    const pad = (n) => String(n).padStart(2, "0");
    function beginBatch() {
      const dur = durSec || 12;                       // ∞ → default 12s per room
      sessionStorage.setItem("mw_batch", JSON.stringify({ dur }));
      document.body.classList.add("warping");
      setTimeout(() => { location.href = ROOMS[0]; }, 150);
    }
    function batchHUD(html) {
      let h = document.getElementById("mw-batch-hud");
      if (!h) { h = document.createElement("div"); h.id = "mw-batch-hud"; h.className = "batch-hud"; document.body.appendChild(h); }
      h.innerHTML = html;
      return h;
    }
    function cancelBatch() {
      sessionStorage.removeItem("mw_batch");
      const h = document.getElementById("mw-batch-hud"); if (h) h.remove();
    }
    (function maybeRunBatch() {
      const raw = sessionStorage.getItem("mw_batch");
      if (!raw) return;
      let b; try { b = JSON.parse(raw); } catch { cancelBatch(); return; }
      const curFile = location.pathname.split("/").pop();
      const idx = ROOMS.indexOf(curFile);
      if (idx < 0) { cancelBatch(); return; }
      const dur = b.dur || 12;
      const stem = curFile.replace(".html", "");
      batchHUD(`◉ <b>REC ALL</b> &nbsp; ${idx + 1}/${ROOMS.length} &nbsp; <span style="color:var(--accent-2)">${stem}</span> &nbsp;·&nbsp; <span style="opacity:.7">Esc cancels · allow multiple downloads if asked</span>`);
      window.addEventListener("keydown", (e) => { if (e.key === "Escape") cancelBatch(); });
      setTimeout(() => {                                // let the scene warm up first
        if (!sessionStorage.getItem("mw_batch")) return; // cancelled during the wait
        start({ dur, name: `${pad(idx + 1)}-${stem}`, then: () => {
          if (!sessionStorage.getItem("mw_batch")) return;
          if (idx + 1 < ROOMS.length) { document.body.classList.add("warping"); setTimeout(() => (location.href = ROOMS[idx + 1]), 250); }
          else { sessionStorage.removeItem("mw_batch"); batchHUD(`✓ <b>done</b> — ${ROOMS.length} clips downloaded`); setTimeout(cancelBatch, 5000); }
        }});
      }, 1700);
    })();

    window.addEventListener("keydown", (e) => {
      if (isTyping() || e.metaKey || e.ctrlKey) return;
      if (e.key === "r" || e.key === "R") { e.preventDefault(); toggle(); }
      else if (e.key === "a" || e.key === "A") { e.preventDefault(); beginBatch(); }
      // 1–5 pick a duration and immediately start a timed clip
      else if ("12345".includes(e.key)) {
        e.preventDefault();
        durEls["12345".indexOf(e.key)].click();
        if (!(recorder && recorder.state === "recording")) start();
      }
    });
  };
  if (document.readyState !== "loading") setup();
  else window.addEventListener("DOMContentLoaded", setup);
})();

// ---- veil failsafe: never let a loading veil get stuck ----
window.addEventListener("load", () => {
  setTimeout(() => {
    document.querySelectorAll(".veil:not(.gone)").forEach((v) => {
      v.classList.add("gone");
      setTimeout(() => v.remove(), 700);
    });
  }, 2500);
});

// ============================================================
// KIOSK NAV — boot straight into the rooms; walk them with ← →.
// ↑ / ↓ cycle the variation within the current room. The interface
// stays hidden until you press M. R records. Esc → first room.
// (No landing page — the gallery IS the rooms.)
// ============================================================
(function kioskNav() {
  // uses the module-scope ROOMS (single source of truth, shared with the recorder)
  const curFile = location.pathname.split("/").pop();
  const curIdx = () => { const i = ROOMS.indexOf(curFile); return i < 0 ? 0 : i; };

  let navigating = false;
  function goRoom(delta) {
    if (navigating) return;
    const n = ROOMS.length;
    const next = ROOMS[((curIdx() + delta) % n + n) % n];
    navigating = true;
    document.body.classList.add("warping");      // quick fade-out; veil covers the load
    setTimeout(() => { location.href = next; }, 170);  // rooms are siblings in /pieces/
  }

  let toastEl = null, toastT = 0;
  function flashVariant(text) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.className = "kiosk-toast variant";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = text;
    toastEl.classList.remove("fade");
    clearTimeout(toastT);
    toastT = setTimeout(() => toastEl.classList.add("fade"), 950);
  }

  function setup() {
    const onScene = !!document.getElementById("scene");
    if (onScene) {
      document.body.classList.add("chrome-hidden");   // rooms start clean
      const toast = document.createElement("div");
      toast.className = "kiosk-toast";
      toast.innerHTML = '<b>&larr; &rarr;</b> rooms &nbsp;·&nbsp; <b>&uarr; &darr;</b> variation &nbsp;·&nbsp; <b>M</b> menu &nbsp;·&nbsp; <b>R</b> rec &nbsp;·&nbsp; <b>A</b> rec all';
      document.body.appendChild(toast);
      setTimeout(() => toast.classList.add("fade"), 3600);
      setTimeout(() => toast.remove(), 4500);
    }

    window.addEventListener("keydown", (e) => {
      if (isTyping() || e.metaKey || e.ctrlKey || e.altKey) return;
      switch (e.key) {
        case "ArrowRight": case "PageDown": e.preventDefault(); goRoom(1); break;
        case "ArrowLeft":  case "PageUp":   e.preventDefault(); goRoom(-1); break;
        case "ArrowUp":   e.preventDefault(); { const l = variantCycler && variantCycler(-1); if (l) flashVariant(l); } break;
        case "ArrowDown": e.preventDefault(); { const l = variantCycler && variantCycler(1);  if (l) flashVariant(l); } break;
        case "m": case "M": e.preventDefault(); document.body.classList.toggle("chrome-hidden"); break;
        case "Escape": e.preventDefault(); goRoom(-curIdx()); break;   // back to room 01
      }
    });
  }
  if (document.readyState !== "loading") setup();
  else window.addEventListener("DOMContentLoaded", setup);
})();
