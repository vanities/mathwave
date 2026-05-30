// redsea.js — 紅海 "The Parting of the Red Sea" as a real 3D WATER-SURFACE MESH.
//
// Method — an animated height-field plane h(x,z,t) deformed on the CPU each frame,
// then computeVertexNormals() so the sea catches the dusk light. The height is the
// SUM of two terms:
//
//   (1) a living sea — a small stack of travelling sines (a cheap Gerstner-ish
//       swell): h_sea = Σ Aᵢ·sin(kᵢ·(x·cosθ + z·sinθ) − ωᵢ·t). This keeps the
//       whole surface breathing so it never looks like a flat sheet.
//
//   (2) the PARTING — a corridor mask centred on x = 0. Let g = |x| / w be the
//       distance from the centre line in units of the corridor half-width w(t):
//         · inside the corridor (g < 1) the water is pushed DOWN to the seabed
//           (a smooth floor that recedes as the path clears),
//         · just OUTSIDE the corridor (1 ≤ g ≲ 2) the displaced water PILES UP
//           into two steep standing WALLS of height H(t) — a raised-cosine ridge,
//       blended with smoothstep so the walls read as sheer cliffs of water.
//
//   The drama is animation of w(t) and H(t) through a 3-phase loop:
//     parting  — walls rise, corridor opens (w: 0 → wMax, H: 0 → Hmax)
//     crossing — hold open, walls steady, sea calmer between them
//     closing  — walls collapse and crash back together (w → 0, H → 0)
//   then it repeats. A SEABED plane (wet sand) sits underneath and is revealed in
//   the corridor; foam/whitecaps are vertex-coloured onto the wall crests, and a
//   little spray of points rides the crest tops.
//
// Look: deep blue→teal water under a RED/AMBER dusk sky (a big gradient backdrop
// sprite + a warm key light) — that reddish dusk is what reads as "Red Sea" while
// the water itself stays believably blue-green. mathwave dark backdrop, vivid
// accents, NOT vaporwave-purple.  Pure Three.js r0.169, no build.
//
// Tech mirrors cloth.js: a BufferGeometry grid whose position attribute is rewritten
// each frame + computeVertexNormals(); MeshStandardMaterial; PerspectiveCamera +
// OrbitControls looking DOWN the corridor.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x140a14);
scene.fog = new THREE.FogExp2(0x241019, 0.006);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 4000);
// look DOWN the corridor (the corridor runs along +Z): camera low and behind, gazing forward
camera.position.set(0, 26, 116);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.22;
controls.minDistance = 30; controls.maxDistance = 360;
controls.maxPolarAngle = Math.PI * 0.495;   // don't dip under the sea
controls.target.set(0, 6, -20);

// ---------- lighting: warm dusk key + cool sky fill so the sea reads blue ----------
scene.add(new THREE.AmbientLight(0x33203a, 0.7));
const dusk = new THREE.DirectionalLight(0xffb066, 1.45);  // low amber sun
dusk.position.set(20, 24, -90); scene.add(dusk);
const sky  = new THREE.DirectionalLight(0x4a7fff, 0.55);  // cool sky bounce
sky.position.set(-30, 40, 40); scene.add(sky);
const rim  = new THREE.PointLight(0xff5a4a, 0.6, 800);    // red rim from the horizon
rim.position.set(0, 14, -260); scene.add(rim);

// ---------- a red/amber dusk sky as a big gradient backdrop sprite ----------
(function addDuskSky() {
  const c = document.createElement("canvas"); c.width = 16; c.height = 256;
  const g = c.getContext("2d");
  const grad = g.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0.00, "#2a0d1e");   // deep wine zenith
  grad.addColorStop(0.45, "#6e1f33");   // dusk red
  grad.addColorStop(0.72, "#c4502f");   // amber band at the horizon
  grad.addColorStop(0.86, "#f0a962");   // amber glow at the horizon line
  grad.addColorStop(1.00, "#3a1418");   // dark sea-haze under the horizon
  g.fillStyle = grad; g.fillRect(0, 0, 16, 256);
  const tex = new THREE.CanvasTexture(c);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, depthTest: false }));
  sprite.scale.set(4000, 2400, 1);
  sprite.position.set(0, 120, -1200);
  scene.add(sprite);
  // a fat amber sun disc low on the horizon, dead ahead down the corridor
  const sc = document.createElement("canvas"); sc.width = sc.height = 256;
  const sg = sc.getContext("2d");
  const sgr = sg.createRadialGradient(128, 128, 6, 128, 128, 128);
  sgr.addColorStop(0, "rgba(255,236,180,1)");
  sgr.addColorStop(0.4, "rgba(255,150,70,0.9)");
  sgr.addColorStop(1, "rgba(255,90,60,0)");
  sg.fillStyle = sgr; sg.fillRect(0, 0, 256, 256);
  const sun = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(sc), transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
  }));
  sun.scale.set(420, 420, 1);
  sun.position.set(0, 40, -1000);
  scene.add(sun);
})();

// ---------- sea grid (capped for 60fps) ----------
const SEA = 260;                 // world-space side length of the sea plane
const NX = 150, NZ = 150;        // vertices across X and Z  (≈22.5k verts — fine on a GPU)
const idx = (ix, iz) => iz * NX + ix;

// build the water mesh once; rewrite its position buffer each frame.
const seaGeo = new THREE.BufferGeometry();
const seaPos = new Float32Array(NX * NZ * 3);
const seaCol = new Float32Array(NX * NZ * 3);   // vertex colour = foam tint on crests
const seaTris = [];
for (let iz = 0; iz < NZ - 1; iz++) {
  for (let ix = 0; ix < NX - 1; ix++) {
    const a = idx(ix, iz), b = idx(ix + 1, iz), c = idx(ix, iz + 1), d = idx(ix + 1, iz + 1);
    seaTris.push(a, c, b, b, c, d);
  }
}
seaGeo.setIndex(seaTris);
const seaPosAttr = new THREE.BufferAttribute(seaPos, 3).setUsage(THREE.DynamicDrawUsage);
seaGeo.setAttribute("position", seaPosAttr);
seaGeo.setAttribute("color", new THREE.BufferAttribute(seaCol, 3));
// seed the flat XZ layout once (x and z never change; only y is animated)
for (let iz = 0; iz < NZ; iz++) {
  for (let ix = 0; ix < NX; ix++) {
    const i = idx(ix, iz);
    seaPos[i * 3]     = (ix / (NX - 1) - 0.5) * SEA;
    seaPos[i * 3 + 1] = 0;
    seaPos[i * 3 + 2] = (iz / (NZ - 1) - 0.5) * SEA;
  }
}
const seaMat = new THREE.MeshStandardMaterial({
  vertexColors: true, roughness: 0.28, metalness: 0.45,
  side: THREE.DoubleSide, flatShading: false,
  transparent: true, opacity: 0.94, depthWrite: true,
  emissive: 0x05172a, emissiveIntensity: 0.5,
});
const seaMesh = new THREE.Mesh(seaGeo, seaMat);
seaMesh.frustumCulled = false;
scene.add(seaMesh);

// ---------- seabed: wet sand, revealed in the corridor ----------
const SEABED_Y = -16;
const bedGeo = new THREE.PlaneGeometry(SEA * 1.2, SEA * 1.2, 1, 1);
bedGeo.rotateX(-Math.PI / 2);
const bedMat = new THREE.MeshStandardMaterial({ color: 0x4a3320, roughness: 1.0, metalness: 0.0, emissive: 0x140d06, emissiveIntensity: 0.4 });
const bed = new THREE.Mesh(bedGeo, bedMat);
bed.position.set(0, SEABED_Y, 0);
scene.add(bed);
// a faint "dust path" strip down the seabed corridor (a slightly lighter, drier sand)
const pathGeo = new THREE.PlaneGeometry(1, SEA * 1.2, 1, 1);
pathGeo.rotateX(-Math.PI / 2);
const pathMat = new THREE.MeshStandardMaterial({ color: 0x6e5230, roughness: 1.0, metalness: 0.0, transparent: true, opacity: 0.85 });
const path = new THREE.Mesh(pathGeo, pathMat);
path.position.set(0, SEABED_Y + 0.05, 0);
scene.add(path);

// ---------- spray: points that ride the wall crests ----------
const SPRAY = 900;
const sprayPos = new Float32Array(SPRAY * 3);
const sprayGeo = new THREE.BufferGeometry();
sprayGeo.setAttribute("position", new THREE.BufferAttribute(sprayPos, 3).setUsage(THREE.DynamicDrawUsage));
const sprayMat = new THREE.PointsMaterial({
  color: 0xdff0ff, size: 1.5, transparent: true, opacity: 0.7,
  blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true,
});
const spray = new THREE.Points(sprayGeo, sprayMat);
spray.frustumCulled = false;
scene.add(spray);

// ---------- colours (water depth gradient + foam) ----------
const C_DEEP = new THREE.Color(0x06243f);   // deep blue
const C_MID  = new THREE.Color(0x0e6f7e);   // teal
const C_SHAL = new THREE.Color(0x27b7c0);   // bright shallow teal
const C_FOAM = new THREE.Color(0xeaf7ff);   // white foam
const _c = new THREE.Color();

// ======================= the parting model =======================
// Parameters the sliders drive.
const params = {
  partSpeed: 0.6,    // how fast the parting phases advance
  wallMax:   34,     // peak wall height (world units)
  corridorMax: 26,   // peak corridor half-width (world units)
  chop:      1.0,    // swell choppiness multiplier
  holdOpen:  false,  // "the crossing" preset freezes the corridor wide open
};

// swell components: [amplitude, wavenumber, angularFreq, directionAngle]
const SWELL = [
  [1.7, 0.045, 0.9, 0.3],
  [1.1, 0.085, 1.4, 1.9],
  [0.6, 0.150, 2.1, 3.4],
  [0.35, 0.230, 2.7, 5.1],
];

// phase clock → returns { name, w (corridor half-width), H (wall height), calm 0..1 }
// One full cycle: parting (open) → crossing (hold) → closing (crash) → tiny rest.
let phaseT = 0;            // advances with partSpeed
const T_PART = 3.2, T_HOLD = 3.0, T_CLOSE = 2.4, T_REST = 1.0;
const T_CYCLE = T_PART + T_HOLD + T_CLOSE + T_REST;
function smooth(e) { return e * e * (3 - 2 * e); }   // smoothstep on a 0..1 edge

function phase() {
  if (params.holdOpen) {
    return { name: "crossing", k: 1, calm: 0.85 };   // frozen wide open
  }
  let t = phaseT % T_CYCLE;
  if (t < T_PART) {                       // walls rise, path opens
    const e = smooth(t / T_PART);
    return { name: "parting", k: e, calm: 0.15 + 0.4 * e };
  }
  t -= T_PART;
  if (t < T_HOLD) {                       // hold open — the crossing
    return { name: "crossing", k: 1, calm: 0.85 };
  }
  t -= T_HOLD;
  if (t < T_CLOSE) {                      // walls crash back together
    const e = smooth(t / T_CLOSE);
    return { name: "closing", k: 1 - e, calm: 0.85 - 0.7 * e };
  }
  return { name: "closing", k: 0, calm: 0.1 };   // brief rest at full sea
}

let curPhase = "parting", curW = 0, curH = 0;

function updateSea(elapsed) {
  const ph = phase();
  curPhase = ph.name;
  // current corridor half-width & wall height, scaled by the sliders
  const w = Math.max(0.001, ph.k * params.corridorMax);
  const H = ph.k * params.wallMax;
  curW = w; curH = H;
  const calm = ph.calm;                         // between the walls the swell eases off
  const chop = params.chop;

  for (let iz = 0; iz < NZ; iz++) {
    for (let ix = 0; ix < NX; ix++) {
      const i = idx(ix, iz);
      const x = seaPos[i * 3];
      const z = seaPos[i * 3 + 2];

      // (1) living sea — sum of travelling sines
      let hSea = 0;
      for (let s = 0; s < SWELL.length; s++) {
        const A = SWELL[s][0], k = SWELL[s][1], om = SWELL[s][2], ang = SWELL[s][3];
        const dir = x * Math.cos(ang) + z * Math.sin(ang);
        hSea += A * Math.sin(k * dir - om * elapsed);
      }
      hSea *= chop;

      // (2) the parting — corridor mask centred on x = 0
      const g = Math.abs(x) / w;                  // 0 at centre line, 1 at corridor edge

      // dip toward the seabed inside the corridor (g<1), fading to none by g≈1.15
      const inside = 1 - smooth(Math.min(1, g / 1.0));      // 1 in corridor → 0 at edge
      const floorY = SEABED_Y + 1.2;                        // wet seabed level (just above sand)
      // wall ridge just outside the corridor: raised-cosine bump centred at g≈1.4
      const wb = Math.max(0, 1 - Math.abs(g - 1.4) / 0.95); // 0..1 triangular support
      const wallShape = smooth(wb);                         // smooth the ridge
      // foam fringe sits a touch wider than the wall, on the crest face
      const crest = wallShape;

      // base water height (calm sea between the walls, livelier outside)
      const seaLevel = hSea * (0.35 + 0.65 * (1 - inside) ) * (calm * inside + (1 - inside));
      // compose: start at the swelling sea, carve the corridor down, raise the walls
      let h = seaLevel;
      // carve down to the seabed inside the corridor (only while it's actually open)
      h = h * (1 - inside) + (floorY) * inside;
      // raise the standing walls (with a foamy, jittery crest)
      const wallTop = H * wallShape;
      const foamJitter = wallShape * Math.sin(z * 0.5 + elapsed * 4.0 + x * 0.3) * 1.4 * chop;
      h += wallTop + foamJitter;

      seaPos[i * 3 + 1] = h;

      // ---- vertex colour: depth gradient + foam on the crests & corridor lips ----
      // depth factor: deeper (more negative) → C_DEEP; near surface → teal
      const depthF = THREE.MathUtils.clamp((h - SEABED_Y) / (H + 22), 0, 1);
      _c.copy(C_DEEP).lerp(C_MID, THREE.MathUtils.clamp(depthF * 1.4, 0, 1));
      _c.lerp(C_SHAL, THREE.MathUtils.clamp((depthF - 0.55) * 2.2, 0, 1));
      // foam where the wall crest is high, or right at the corridor lip (g≈1)
      const lip = Math.max(0, 1 - Math.abs(g - 1.0) / 0.18);
      const foam = THREE.MathUtils.clamp(crest * (wallTop / (params.wallMax + 1)) * 2.4 + lip * 0.7, 0, 1);
      _c.lerp(C_FOAM, foam);
      seaCol[i * 3] = _c.r; seaCol[i * 3 + 1] = _c.g; seaCol[i * 3 + 2] = _c.b;
    }
  }
  seaPosAttr.needsUpdate = true;
  seaGeo.attributes.color.needsUpdate = true;
  seaGeo.computeVertexNormals();
  seaGeo.computeBoundingSphere();

  // dust path widens/narrows with the corridor; brighten while open
  const pw = Math.min(w * 1.1, params.corridorMax * 1.1) * 2;
  path.scale.set(Math.max(0.001, pw), 1, 1);
  pathMat.opacity = 0.25 + 0.6 * ph.k;

  // spray rides the two crests; hide when walls are low
  const crestZspan = SEA * 0.96;
  for (let p = 0; p < SPRAY; p++) {
    const side = p < SPRAY / 2 ? -1 : 1;
    const zz = (Math.sin(p * 12.9898 + elapsed * 0.6) * 0.5 + 0.5) * crestZspan - crestZspan / 2;
    const wallX = side * w * 1.4;
    const jig = Math.sin(p * 3.7 + elapsed * 5.0);
    sprayPos[p * 3]     = wallX + jig * 2.0;
    sprayPos[p * 3 + 1] = (H > 2 ? H + 1.5 + (Math.sin(p * 7.1 + elapsed * 6.0) * 0.5 + 0.5) * H * 0.35 : -999);
    sprayPos[p * 3 + 2] = zz;
  }
  sprayGeo.attributes.position.needsUpdate = true;
  sprayMat.opacity = 0.15 + 0.55 * THREE.MathUtils.clamp(H / params.wallMax, 0, 1);
}

// ======================= presets (↑↓) =======================
// Each preset tweaks the sliders + the phase behaviour for a distinct beat.
const PRESETS = [
  ["the parting",  { partSpeed: 0.6, wallMax: 34, corridorMax: 26, chop: 1.0, holdOpen: false }],
  ["the crossing", { partSpeed: 0.6, wallMax: 40, corridorMax: 30, chop: 0.45, holdOpen: true  }],
  ["the return",   { partSpeed: 1.25, wallMax: 30, corridorMax: 22, chop: 1.15, holdOpen: false }],
  ["storm sea",    { partSpeed: 0.7, wallMax: 44, corridorMax: 20, chop: 1.9, holdOpen: false }],
];
let presetIdx = 0;

function applyPreset(i) {
  const cfg = PRESETS[i][1];
  params.partSpeed = cfg.partSpeed;
  params.wallMax = cfg.wallMax;
  params.corridorMax = cfg.corridorMax;
  params.chop = cfg.chop;
  params.holdOpen = cfg.holdOpen;
  // mirror into the sliders so the UI tracks the preset
  setSlider("speed", params.partSpeed);
  setSlider("wall", params.wallMax);
  setSlider("chop", params.chop);
  nameEl.textContent = PRESETS[i][0];
  chips.forEach((c, k) => c.classList.toggle("active", k === i));
  // "the return" jumps to the crash; others start at the open
  if (PRESETS[i][0] === "the return") phaseT = (T_PART + T_HOLD) * params.partSpeed;
  else phaseT = 0;
}

function setSlider(id, v) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = String(v);
  const out = document.querySelector(`[data-val="${id}"]`);
  if (out) out.textContent = fmtFor(id, v);
}
function fmtFor(id, v) {
  if (id === "speed") return (+v).toFixed(2) + "×";
  if (id === "wall") return Math.round(v) + "u";
  if (id === "chop") return (+v).toFixed(2) + "×";
  return String(v);
}

// ---------- panel ----------
const wrap = document.getElementById("presets");
const nameEl = document.getElementById("presetname");
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { presetIdx = i; applyPreset(i); });
  wrap.appendChild(b);
  return b;
});

bindRange("speed", (v) => { params.partSpeed = v; }, (v) => v.toFixed(2) + "×");
bindRange("wall",  (v) => { params.wallMax = v; },   (v) => Math.round(v) + "u");
bindRange("chop",  (v) => { params.chop = v; },      (v) => v.toFixed(2) + "×");

const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => {
  camera.position.copy(home);
  controls.target.set(0, 6, -20);
  phaseT = 0;
});

setVariantCycler((d) => {
  presetIdx = (presetIdx + d + PRESETS.length) % PRESETS.length;
  applyPreset(presetIdx);
  return PRESETS[presetIdx][0];
});

// ---------- diagnostics hook (kiosk/QA reads this) ----------
window.__diag = () => JSON.stringify({
  phase: curPhase,
  corridor: +curW.toFixed(1),
  wall: +curH.toFixed(1),
  preset: PRESETS[presetIdx][0],
});

// ---------- boot ----------
applyPreset(0);
updateSea(0);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const phEl = document.getElementById("phase");
const corEl = document.getElementById("corridor");

loop((dt, elapsed) => {
  meter(dt);
  if (!reducedMotion) phaseT += dt * params.partSpeed;
  updateSea(elapsed);
  if (phEl) phEl.textContent = curPhase;
  if (corEl) corEl.textContent = Math.round(curW) + "u";
  controls.update();
  renderer.render(scene, camera);
});
