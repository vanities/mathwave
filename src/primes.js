// primes.js — Prime Spiral & Riemann Zeta. The distribution of the primes,
// shown three ways (↑/↓ cycles the mode):
//
//   (0) SACKS spiral — place integer n at polar angle 2π√n and radius √n.
//       The perfect squares land on the positive x-axis; primes light up,
//       and the prime-rich quadratic polynomials (e.g. Euler's n²+n+41)
//       trace the gentle "product curves." Ref: Robert Sacks, the number
//       spiral (1994).
//   (1) ULAM spiral — write the integers on a square spiral grid (Ulam,
//       1963) and light the primes. They fall along diagonals because each
//       diagonal/anti-diagonal is a quadratic 4n²+bn+c, some of them
//       unusually prime-dense — the famous striations.
//   (2) ZETA — the Riemann zeta function on its critical line. We plot
//       |ζ(½ + it)| as a 3D ribbon climbing in t, computed from the
//       Dirichlet eta (alternating, convergent) series:
//           η(s) = Σ_{n≥1} (-1)^(n-1) n^(-s),   ζ(s) = η(s) / (1 - 2^(1-s)).
//       The curve dips to ~0 exactly at the nontrivial zeros; the first ~50
//       imaginary parts t_k are overlaid as bright markers. Via the explicit
//       formula, those t_k *are* the hidden music of the primes — so this
//       room ties the spiral (left brain) to the zeros (right brain).
//
// Buildless: ES modules + import map, Three.js r0.169 from CDN.
// Real geometry — THREE.Points (BufferGeometry, position+color) and
// THREE.Line — never a fullscreen shader.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x05070a, 0.0017);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 6000);
camera.position.set(0, 90, 360);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.08;
controls.minDistance = 20; controls.maxDistance = 2500;

scene.add(new THREE.AmbientLight(0xffffff, 0.7));

// ---------------------------------------------------------------------------
// Number theory
// ---------------------------------------------------------------------------

// Sieve of Eratosthenes -> Uint8Array isPrime[0..N]. Built once per N, cached,
// so we never re-sieve per frame (or when only the mode changes).
let sieveCache = { N: -1, arr: null };
function sieve(N) {
  if (sieveCache.N >= N && sieveCache.arr) return sieveCache.arr;
  const isPrime = new Uint8Array(N + 1).fill(1);
  isPrime[0] = 0; if (N >= 1) isPrime[1] = 0;
  for (let p = 2; p * p <= N; p++) {
    if (isPrime[p]) for (let m = p * p; m <= N; m += p) isPrime[m] = 0;
  }
  sieveCache = { N, arr: isPrime };
  return isPrime;
}

// First 50 nontrivial zeros of ζ(s): imaginary parts on Re(s)=1/2.
// Standard table (Odlyzko), to 6 dp.
const ZETA_ZEROS = [
  14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
  37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
  52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
  67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
  79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
  92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
  103.725539, 105.446623, 107.168611, 111.029536, 111.874659,
  114.320221, 116.226680, 118.790783, 121.370125, 122.946829,
  124.256819, 127.516684, 129.578704, 131.087689, 133.497737,
  134.756510, 138.116042, 139.736209, 141.123707, 143.111846,
];

// |ζ(1/2 + i t)| via the Dirichlet eta partial sum.
//   n^{-s} = n^{-1/2} e^{-i t ln n};  ζ = η / (1 - 2^{1-s}).
function zetaMagOnCritical(t, terms) {
  let reEta = 0, imEta = 0;
  for (let n = 1; n <= terms; n++) {
    const amp = 1 / Math.sqrt(n);     // n^{-1/2}
    const ang = -t * Math.log(n);     // phase of n^{-i t}
    const sign = (n & 1) ? 1 : -1;    // (-1)^{n-1}
    reEta += sign * amp * Math.cos(ang);
    imEta += sign * amp * Math.sin(ang);
  }
  // 1/(1 - 2^{1-s}),  2^{1-s} = 2^{1/2} 2^{-i t}
  const p = Math.SQRT2;
  const aLn2 = -t * Math.LN2;
  const fRe = 1 - p * Math.cos(aLn2);
  const fIm = -p * Math.sin(aLn2);
  const fMag = Math.hypot(fRe, fIm);
  if (fMag < 1e-9) return 0;
  return Math.hypot(reEta, imEta) / fMag;
}

// ---------------------------------------------------------------------------
// Palette — neon on near-black; cyan / amber / magenta (NOT purple-dominant)
// ---------------------------------------------------------------------------
const COL_DIM    = new THREE.Color(0x16323a); // composite integers (dim teal)
const COL_PRIME  = new THREE.Color(0x29e0ff); // primes (cyan)
const COL_PRIME2 = new THREE.Color(0xffb13b); // prime accent (amber)
const COL_ZETA   = new THREE.Color(0x29e0ff); // zeta ribbon (cyan)
const COL_ZERO   = new THREE.Color(0xff3bd0); // zeta zeros (magenta)
const COL_AXIS   = new THREE.Color(0x39555f);

// A soft round sprite so points read as glowing dots, not squares.
function makeDiscTexture() {
  const s = 64, cv = document.createElement("canvas"); cv.width = cv.height = s;
  const ctx = cv.getContext("2d");
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0.0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.85)");
  g.addColorStop(1.0, "rgba(255,255,255,0)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(cv); tex.needsUpdate = true; return tex;
}
const discTex = makeDiscTexture();

function pointsMaterial(size, opacity) {
  return new THREE.PointsMaterial({
    size, map: discTex, vertexColors: true, transparent: true, opacity,
    alphaTest: 0.02, depthWrite: false, blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
  });
}

// ---------------------------------------------------------------------------
// State + groups (one per mode; geometry rebuilt only when its inputs change)
// ---------------------------------------------------------------------------
const MODE = { SACKS: 0, ULAM: 1, ZETA: 2 };
const MODE_NAMES = ["Sacks", "Ulam", "Zeta"];

const state = { mode: MODE.SACKS, N: 30000 };

const gSacks = new THREE.Group();
const gUlam  = new THREE.Group();
const gZeta  = new THREE.Group();
scene.add(gSacks, gUlam, gZeta);

let primeCountLast = 0;   // for the readout

function clearGroup(g) {
  for (let i = g.children.length - 1; i >= 0; i--) {
    const o = g.children[i];
    g.remove(o);
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  }
}

// ---- (0) Sacks spiral: n -> radius √n, angle 2π√n. Slight z by n for 3D. ----
function buildSacks(N) {
  clearGroup(gSacks);
  const isPrime = sieve(N);
  const SCALE = 6.0;
  const positions = new Float32Array((N + 1) * 3);
  const colors = new Float32Array((N + 1) * 3);
  const tmp = new THREE.Color();
  let primes = 0;
  for (let n = 0; n <= N; n++) {
    const r = Math.sqrt(n);
    const ang = 2 * Math.PI * r;          // 2π√n
    const i3 = n * 3;
    positions[i3]     = Math.cos(ang) * r * SCALE;   // spiral plane -> X
    positions[i3 + 1] = (n / N) * 60 - 30;           // gentle extrusion -> height
    positions[i3 + 2] = Math.sin(ang) * r * SCALE;   // spiral plane -> Z
    if (isPrime[n]) { primes++; tmp.copy(primes & 1 ? COL_PRIME : COL_PRIME2); }
    else tmp.copy(COL_DIM);
    colors[i3] = tmp.r; colors[i3 + 1] = tmp.g; colors[i3 + 2] = tmp.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeBoundingSphere();
  gSacks.add(new THREE.Points(geo, pointsMaterial(3.2, 0.95)));
  return primes;
}

// ---- (1) Ulam spiral: integers on a square spiral; primes lit. ----
function buildUlam(N) {
  clearGroup(gUlam);
  const isPrime = sieve(N);
  const STEP = 5.0;
  const positions = new Float32Array((N + 1) * 3);
  const colors = new Float32Array((N + 1) * 3);
  const tmp = new THREE.Color();
  let primes = 0;
  // square-spiral walk starting at origin (n=1), turning CCW
  let x = 0, z = 0, dx = 1, dz = 0, segLen = 1, segPassed = 0, turns = 0;
  for (let n = 1; n <= N; n++) {
    const i3 = n * 3;
    positions[i3]     = x * STEP;
    positions[i3 + 1] = (n / N) * 30 - 15;   // faint extrusion for 3D read
    positions[i3 + 2] = z * STEP;
    if (isPrime[n]) { primes++; tmp.copy(primes & 1 ? COL_PRIME : COL_PRIME2); }
    else tmp.copy(COL_DIM);
    colors[i3] = tmp.r; colors[i3 + 1] = tmp.g; colors[i3 + 2] = tmp.b;
    // advance
    x += dx; z += dz; segPassed++;
    if (segPassed === segLen) {
      segPassed = 0;
      const ndx = -dz, ndz = dx;   // rotate (dx,dz) CCW
      dx = ndx; dz = ndz; turns++;
      if (turns % 2 === 0) segLen++;
    }
  }
  colors[0] = COL_DIM.r; colors[1] = COL_DIM.g; colors[2] = COL_DIM.b; // unused n=0
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeBoundingSphere();
  gUlam.add(new THREE.Points(geo, pointsMaterial(2.8, 0.95)));
  return primes;
}

// ---- (2) Zeta: |ζ(½+it)| ribbon climbing in t, plus zero markers. ----
let zetaTMax = -1;
function buildZeta(tMax) {
  clearGroup(gZeta);
  zetaTMax = tMax;
  const samples = 1600;
  const terms = 4000;        // eta-series terms (capped; accurate for t up to ~150)
  const T_SCALE = 4.0;       // world units per unit t (height)
  const MAG_SCALE = 70.0;    // world units per unit normalized |ζ|
  const half = tMax * T_SCALE * 0.5;

  const mags = new Float32Array(samples);
  let maxMag = 1e-4;
  for (let i = 0; i < samples; i++) {
    const t = (i / (samples - 1)) * tMax;
    const m = zetaMagOnCritical(t, terms);
    mags[i] = m; if (m > maxMag) maxMag = m;
  }

  const positions = new Float32Array(samples * 3);
  const colors = new Float32Array(samples * 3);
  const tmp = new THREE.Color();
  for (let i = 0; i < samples; i++) {
    const t = (i / (samples - 1)) * tMax;
    const norm = mags[i] / maxMag;        // 0..1
    const i3 = i * 3;
    positions[i3]     = norm * MAG_SCALE;  // |ζ| out from the line -> X
    positions[i3 + 1] = t * T_SCALE - half; // t -> height
    positions[i3 + 2] = 0;                 // critical line Re(s)=1/2 plane
    tmp.copy(COL_ZETA).lerp(COL_ZERO, 1 - Math.min(1, norm * 1.6)); // dip -> magenta
    colors[i3] = tmp.r; colors[i3 + 1] = tmp.g; colors[i3 + 2] = tmp.b;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  gZeta.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false,
  })));

  // the critical line Re(s)=1/2 itself, as a faint axis
  const axisGeo = new THREE.BufferGeometry();
  axisGeo.setAttribute("position", new THREE.BufferAttribute(
    new Float32Array([0, -half, 0, 0, half, 0]), 3));
  gZeta.add(new THREE.Line(axisGeo, new THREE.LineBasicMaterial({
    color: COL_AXIS, transparent: true, opacity: 0.5 })));

  // nontrivial-zero markers: bright points at (x≈0, t=t_k)
  const zeros = ZETA_ZEROS.filter((t) => t <= tMax);
  const zpos = new Float32Array(zeros.length * 3);
  const zcol = new Float32Array(zeros.length * 3);
  for (let k = 0; k < zeros.length; k++) {
    const k3 = k * 3;
    zpos[k3] = 0; zpos[k3 + 1] = zeros[k] * T_SCALE - half; zpos[k3 + 2] = 0;
    zcol[k3] = COL_ZERO.r; zcol[k3 + 1] = COL_ZERO.g; zcol[k3 + 2] = COL_ZERO.b;
  }
  const zgeo = new THREE.BufferGeometry();
  zgeo.setAttribute("position", new THREE.BufferAttribute(zpos, 3));
  zgeo.setAttribute("color", new THREE.BufferAttribute(zcol, 3));
  gZeta.add(new THREE.Points(zgeo, pointsMaterial(11.0, 1.0)));

  return zeros.length;
}

// Map the single N slider (5000..100000) to a zeta tMax (~20..143) so the
// control is meaningful in every mode.
function mapNtoTMax(N) {
  const f = Math.max(0, Math.min(1, (N - 5000) / (100000 - 5000)));
  return 20 + f * (143 - 20);
}

// ---------------------------------------------------------------------------
// Build orchestration + visibility + camera framing
// ---------------------------------------------------------------------------
let builtMode = -1, builtN = -1;
function applyVisibility() {
  gSacks.visible = state.mode === MODE.SACKS;
  gUlam.visible  = state.mode === MODE.ULAM;
  gZeta.visible  = state.mode === MODE.ZETA;
}
function frameCamera() {
  if (state.mode === MODE.ZETA) { camera.position.set(190, 0, 260); controls.target.set(0, 0, 0); }
  else { camera.position.set(0, 90, 360); controls.target.set(0, 0, 0); }
  controls.update();
}
function rebuild() {
  if (state.mode === MODE.SACKS) {
    if (builtMode !== MODE.SACKS || builtN !== state.N) primeCountLast = buildSacks(state.N);
  } else if (state.mode === MODE.ULAM) {
    if (builtMode !== MODE.ULAM || builtN !== state.N) primeCountLast = buildUlam(state.N);
  } else {
    const tMax = mapNtoTMax(state.N);
    if (builtMode !== MODE.ZETA || Math.abs(tMax - zetaTMax) > 0.01) primeCountLast = buildZeta(tMax);
  }
  builtMode = state.mode; builtN = state.N;
  applyVisibility();
  updateReadout();
}

// ---------------------------------------------------------------------------
// Panel / readout
// ---------------------------------------------------------------------------
const countLabelEl = document.querySelector('[data-label="count"]');
function setCountLabel() {
  if (countLabelEl) countLabelEl.textContent = state.mode === MODE.ZETA ? "t max" : "count N";
}
function updateReadout() {
  const cEl = document.getElementById("primecount");
  const lEl = document.getElementById("primelabel");
  if (state.mode === MODE.ZETA) {
    if (cEl) cEl.textContent = String(primeCountLast);
    if (lEl) lEl.textContent = "ζ zeros";
  } else {
    if (cEl) cEl.textContent = primeCountLast.toLocaleString();
    if (lEl) lEl.textContent = "primes";
  }
}

// mode chips (also wired to the same cycler the arrow keys use)
const modesWrap = document.getElementById("modes");
const modeChips = MODE_NAMES.map((label, i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => applyMode(i));
  if (modesWrap) modesWrap.appendChild(b);
  return b;
});
function syncChips() { modeChips.forEach((c, j) => c.classList.toggle("active", j === state.mode)); }

function applyMode(i) {
  state.mode = i;
  syncChips(); setCountLabel(); frameCamera(); rebuild();
}

// ↑/↓ cycler — returns the new mode's label for the kiosk flash.
setVariantCycler((d) => {
  state.mode = (state.mode + d + MODE_NAMES.length) % MODE_NAMES.length;
  syncChips(); setCountLabel(); frameCamera(); rebuild();
  return MODE_NAMES[state.mode];
});

// N / t slider — same control reinterpreted per mode.
bindRange("count", (v) => {
  state.N = Math.max(2000, Math.round(v));
  rebuild();
}, (v) => state.mode === MODE.ZETA ? `t≤${mapNtoTMax(v).toFixed(0)}` : `${Math.round(v).toLocaleString()}`);

const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", () => {
  state.mode = MODE.SACKS; state.N = 30000;
  const slider = document.getElementById("count");
  if (slider) { slider.value = String(state.N); }
  syncChips(); setCountLabel();
  // refresh the slider's mirrored value text
  const out = document.querySelector('[data-val="count"]');
  if (out) out.textContent = `${state.N.toLocaleString()}`;
  frameCamera(); rebuild();
});

// ---------------------------------------------------------------------------
// Diagnostics hook required by the harness
// ---------------------------------------------------------------------------
window.__diag = () => JSON.stringify({
  mode: MODE_NAMES[state.mode],
  n: state.mode === MODE.ZETA ? Math.round(mapNtoTMax(state.N)) : state.N,
});

// ---------------------------------------------------------------------------
// Boot + render loop (slow auto-rotate, filmable)
// ---------------------------------------------------------------------------
setCountLabel();
rebuild();
frameCamera();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

const ROT = 0.05; // rad/s
loop((dt) => {
  meter(dt);
  if (!reducedMotion) {
    const g = state.mode === MODE.SACKS ? gSacks : state.mode === MODE.ULAM ? gUlam : gZeta;
    g.rotation.y += ROT * dt;
  }
  controls.update();
  renderer.render(scene, camera);
});
