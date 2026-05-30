// babel.js — The Tower of Babel (Genesis 11:1-9). "And the whole earth was of
// one language… let us build us a city and a tower, whose top may reach unto
// heaven… And the LORD said, Go to, let us go down, and there confound their
// language… So the LORD scattered them abroad from thence upon the face of all
// the earth: and they left off to build the city."
//
// CONSTRUCTION METHOD — a procedurally generated stepped ZIGGURAT (after
// Bruegel's 1563 painting): TIERS stacked from the ground up, each a closed
// RING of BRICKS whose radius shrinks with height, every ring rotated by a small
// helical offset so the courses spiral. Each brick is one instance of a single
// THREE.InstancedMesh of boxes (one draw call). For every brick we precompute a
// "home" transform (position, yaw facing the tower axis, slightly jittered scale)
// ONCE, plus a per-brick reveal time (bottom bricks appear first).
//
// Three phases driven by a clock, looped forever (a filmable rise→fall cycle):
//   RISE       — bricks reveal in build order, the tower climbs toward the sky.
//   CONFUSION  — at the peak, "the LORD confused their language": a THREE.Points
//                field of glowing glyph-sparks erupts and scatters outward (the
//                scattering of peoples & tongues), while each brick is handed a
//                ballistic velocity + gravity + spin and tumbles/disperses.
//   FALLEN     — the tower lies as rubble on the plain; after a beat, reset.
//
// Per-brick fall is plain ballistics integrated into the instance matrix each
// frame (Object3D.position is a read-only ref → mutate a shared dummy then
// setMatrixAt + instanceMatrix.needsUpdate). Glyph sparks update a Points buffer
// in place. Bricks are capped for 60fps. Warm sandstone/amber on the mathwave
// dark backdrop; the confusion glyphs glow in many neon hues (many languages).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const DEG = Math.PI / 180;
const GRAV = -34;            // gravity for tumbling rubble (world units / s²)
const MAX_BRICKS = 4200;     // hard cap so the tab holds 60fps
const MAX_GLYPHS = 1400;     // scattering-tongues spark field

// --- scene -----------------------------------------------------------------
const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x120a06, 0.010);   // warm dusty haze

const camera = new THREE.PerspectiveCamera(56, innerWidth / innerHeight, 0.1, 3000);
camera.position.set(0, 60, 150);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.32;
controls.minDistance = 30; controls.maxDistance = 520;
controls.target.set(0, 42, 0);

// warm sunlit mood — amber key + cool sky fill so the steps read in 3D
scene.add(new THREE.AmbientLight(0x4a3a2a, 0.85));
const key = new THREE.DirectionalLight(0xffd9a0, 1.25); key.position.set(40, 80, 30); scene.add(key);
const fill = new THREE.DirectionalLight(0x3a4a78, 0.45); fill.position.set(-50, 30, -40); scene.add(fill);
addGrid(scene, { size: 480, divisions: 48, y: 0 });
addSun(scene, { scale: 150, position: [0, 70, -420] });

// --- presets ---------------------------------------------------------------
// pace: build speed multiplier · confusionHold/fallenHold: dwell seconds at the
// peak / on the rubble · shape: "square" ziggurat vs "round" spiral · startFallen
// drops you straight into the ruin. Sliders can still override tiers/buildSpeed.
const PRESETS = [
  ["the ascent",  { pace: 0.55, confusionHold: 4.0, fallenHold: 2.0, shape: "round",  startFallen: false }],
  ["confusion",   { pace: 2.1,  confusionHold: 3.0, fallenHold: 3.0, shape: "round",  startFallen: false }],
  ["ruin",        { pace: 1.0,  confusionHold: 3.0, fallenHold: 5.5, shape: "square", startFallen: true  }],
  ["ziggurat",    { pace: 0.9,  confusionHold: 3.2, fallenHold: 2.4, shape: "square", startFallen: false }],
];
let presetIdx = 0;
let presetName = PRESETS[0][0];

// --- tower geometry parameters (some slider-driven) ------------------------
let tiers = 16;              // number of stacked rings (slider)
let buildSpeed = 1.0;        // rise-speed multiplier (slider)
let shape = "round";         // "round" spiral | "square" ziggurat
let confusionHold = 4.0;
let fallenHold = 2.0;

const TIER_H = 4.2;          // vertical height of one tier
const BRICK_GAP = 0.12;      // fraction of brick width left as mortar
const BASE_RADIUS = 46;      // footprint radius of the lowest ring
const TOP_RADIUS = 8;        // radius of the topmost ring
const HELIX_TURN = 26 * DEG; // ring-to-ring rotation → the spiral

// --- one InstancedMesh of bricks ------------------------------------------
const brickGeo = new THREE.BoxGeometry(1, 1, 1);
const brickMat = new THREE.MeshStandardMaterial({ roughness: 0.92, metalness: 0.04 });
let bricks = new THREE.InstancedMesh(brickGeo, brickMat, MAX_BRICKS);
bricks.frustumCulled = false;                       // they fly far during the fall
bricks.count = 0;
scene.add(bricks);

// per-brick "home" transform + physics state (flat typed arrays, reused)
const homePos = new Float32Array(MAX_BRICKS * 3);   // resting position
const homeYaw = new Float32Array(MAX_BRICKS);       // facing the tower axis
const brickW = new Float32Array(MAX_BRICKS);        // box width  (tangential)
const brickD = new Float32Array(MAX_BRICKS);        // box depth  (radial)
const revealAt = new Float32Array(MAX_BRICKS);      // build-order reveal time (0..1)
const brickTier = new Int16Array(MAX_BRICKS);       // which tier (for color)
// live physics (populated when the tower falls)
const px = new Float32Array(MAX_BRICKS), py = new Float32Array(MAX_BRICKS), pz = new Float32Array(MAX_BRICKS);
const vx = new Float32Array(MAX_BRICKS), vy = new Float32Array(MAX_BRICKS), vz = new Float32Array(MAX_BRICKS);
const rqx = new Float32Array(MAX_BRICKS), rqy = new Float32Array(MAX_BRICKS), rqz = new Float32Array(MAX_BRICKS), rqw = new Float32Array(MAX_BRICKS); // live spin quaternion
const wx = new Float32Array(MAX_BRICKS), wy = new Float32Array(MAX_BRICKS), wz = new Float32Array(MAX_BRICKS);     // angular velocity
let brickCount = 0;
let towerTop = 0;           // world Y of the tower's crown (for camera + glyph spawn)

// sandstone → amber → sunlit-tip palette by tier height (warm, not purple)
const cLow = new THREE.Color(0x8a5a32);   // shadowed sandstone base
const cMid = new THREE.Color(0xd99441);   // lit amber
const cTop = new THREE.Color(0xffe6a8);   // sunstruck crown
const tmpCol = new THREE.Color();
function tierColor(tier, nTiers) {
  const t = nTiers > 1 ? tier / (nTiers - 1) : 0;
  if (t < 0.5) tmpCol.copy(cLow).lerp(cMid, t / 0.5);
  else tmpCol.copy(cMid).lerp(cTop, (t - 0.5) / 0.5);
  // subtle per-brick value jitter so the masonry isn't flat
  const j = 0.86 + Math.random() * 0.28;
  tmpCol.multiplyScalar(j);
  return tmpCol;
}

const dummy = new THREE.Object3D();
const tmpQ = new THREE.Quaternion();
const tmpQ2 = new THREE.Quaternion();
const tmpEuler = new THREE.Euler();
const SPIN_AXIS = new THREE.Vector3();

function rand(a, b) { return a + Math.random() * (b - a); }

// Build the ziggurat: lay bricks tier by tier, ground up. Records each brick's
// home transform, reveal time, color, and seeds its physics state at rest.
function buildTower() {
  brickCount = 0;
  let maxY = 0;
  for (let t = 0; t < tiers && brickCount < MAX_BRICKS; t++) {
    const f = tiers > 1 ? t / (tiers - 1) : 0;          // 0 at base → 1 at top
    const radius = BASE_RADIUS + (TOP_RADIUS - BASE_RADIUS) * f;
    const y = TIER_H * 0.5 + t * TIER_H;
    const phase = t * HELIX_TURN;                        // helical course offset

    if (shape === "round") {
      // a ring of bricks; count scales with circumference so spacing stays even
      const targetW = 5.2;                               // nominal brick width
      let n = Math.max(6, Math.round((2 * Math.PI * radius) / targetW));
      const w = (2 * Math.PI * radius) / n * (1 - BRICK_GAP);
      const depth = 5.0;
      for (let i = 0; i < n && brickCount < MAX_BRICKS; i++) {
        const a = phase + (i / n) * Math.PI * 2;
        const rr = radius + rand(-0.25, 0.25);
        placeBrick(brickCount, Math.cos(a) * rr, y + rand(-0.12, 0.12), Math.sin(a) * rr,
          -a + Math.PI / 2, w, depth, t, f);
        brickCount++;
      }
    } else {
      // square ziggurat: bricks marching along the 4 edges of a shrinking square
      const half = radius * 0.82;
      const targetW = 5.4;
      let per = Math.max(2, Math.round((2 * half) / targetW));
      const w = (2 * half) / per * (1 - BRICK_GAP);
      const depth = 5.0;
      for (let e = 0; e < 4 && brickCount < MAX_BRICKS; e++) {
        const yaw = e * (Math.PI / 2) + phase;
        for (let i = 0; i < per && brickCount < MAX_BRICKS; i++) {
          const u = -half + (i + 0.5) * (2 * half / per);
          // place along edge e of a square of side 2*half, then rotate by phase
          let lx, lz;
          if (e === 0) { lx = u; lz = half; }
          else if (e === 1) { lx = half; lz = -u; }
          else if (e === 2) { lx = -u; lz = -half; }
          else { lx = -half; lz = u; }
          const cs = Math.cos(phase), sn = Math.sin(phase);
          const wxp = lx * cs - lz * sn, wzp = lx * sn + lz * cs;
          placeBrick(brickCount, wxp + rand(-0.2, 0.2), y + rand(-0.12, 0.12), wzp + rand(-0.2, 0.2),
            yaw, w, depth, t, f);
          brickCount++;
        }
      }
    }
    maxY = y + TIER_H * 0.5;
  }
  towerTop = maxY;

  // build-order reveal: 0 at the first brick → ~0.92 at the crown (so the
  // capstone lands just before the peak). A touch of jitter keeps it organic.
  for (let i = 0; i < brickCount; i++) {
    const base = brickCount > 1 ? i / (brickCount - 1) : 0;
    revealAt[i] = Math.min(0.96, base * 0.92 + (Math.random() * 0.03));
  }

  bricks.count = brickCount;
  // upload colors once (constant through the whole cycle)
  for (let i = 0; i < brickCount; i++) {
    bricks.setColorAt(i, tierColor(brickTier[i], tiers));
  }
  if (bricks.instanceColor) bricks.instanceColor.needsUpdate = true;
  controls.target.set(0, towerTop * 0.5, 0);
}

function placeBrick(i, x, y, z, yaw, w, d, tier, f) {
  homePos[i * 3] = x; homePos[i * 3 + 1] = y; homePos[i * 3 + 2] = z;
  homeYaw[i] = yaw;
  brickW[i] = w; brickD[i] = d; brickTier[i] = tier;
  // seed physics at rest (used until the fall begins)
  px[i] = x; py[i] = y; pz[i] = z;
  vx[i] = vy[i] = vz[i] = 0;
  wx[i] = wy[i] = wz[i] = 0;
  rqx[i] = 0; rqy[i] = 0; rqz[i] = 0; rqw[i] = 1;
}

// Kick the whole tower into ballistic rubble: outward+upward burst that grows
// with height (the top scatters farthest — the peoples flung "abroad").
function shatter() {
  for (let i = 0; i < brickCount; i++) {
    const hx = homePos[i * 3], hy = homePos[i * 3 + 1], hz = homePos[i * 3 + 2];
    px[i] = hx; py[i] = hy; pz[i] = hz;
    const f = towerTop > 0 ? hy / towerTop : 0;            // 0 base → 1 crown
    let rx = hx, rz = hz; const rl = Math.hypot(rx, rz) || 1; rx /= rl; rz /= rl;
    const outward = 7 + f * 26;
    vx[i] = rx * outward + rand(-5, 5);
    vz[i] = rz * outward + rand(-5, 5);
    vy[i] = rand(6, 14) + f * 22;                          // tossed skyward, higher up
    wx[i] = rand(-6, 6); wy[i] = rand(-6, 6); wz[i] = rand(-6, 6);
    rqx[i] = 0; rqy[i] = 0; rqz[i] = 0; rqw[i] = 1;
  }
}

// --- confusion glyphs: a scattering field of glowing "tongues" -------------
// A Points cloud of tiny sprites textured with a glyph atlas; many neon hues.
let glyphTex = makeGlyphTexture();
const glyphPos = new Float32Array(MAX_GLYPHS * 3);
const glyphCol = new Float32Array(MAX_GLYPHS * 3);
const gvx = new Float32Array(MAX_GLYPHS), gvy = new Float32Array(MAX_GLYPHS), gvz = new Float32Array(MAX_GLYPHS);
const glyphLife = new Float32Array(MAX_GLYPHS);   // seconds remaining (0 = dead)
let glyphCount = 0;
const glyphGeo = new THREE.BufferGeometry();
glyphGeo.setAttribute("position", new THREE.BufferAttribute(glyphPos, 3).setUsage(THREE.DynamicDrawUsage));
glyphGeo.setAttribute("color", new THREE.BufferAttribute(glyphCol, 3).setUsage(THREE.DynamicDrawUsage));
glyphGeo.setDrawRange(0, 0);
const glyphMat = new THREE.PointsMaterial({
  size: 7.5, map: glyphTex, vertexColors: true, transparent: true,
  opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending,
  sizeAttenuation: true,
});
const glyphs = new THREE.Points(glyphGeo, glyphMat);
glyphs.frustumCulled = false;
scene.add(glyphs);

// glyph sprite atlas: scribbles evoking many alphabets, baked to one texture.
function makeGlyphTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.clearRect(0, 0, 64, 64);
  g.strokeStyle = "#fff"; g.fillStyle = "#fff";
  g.lineWidth = 5; g.lineCap = "round"; g.lineJoin = "round";
  // a few random strokes → an abstract "letter" with a soft glow center
  const cx = 32, cy = 32;
  g.beginPath();
  g.moveTo(cx + rand(-16, 16), cy + rand(-16, 16));
  for (let k = 0; k < 3; k++) g.lineTo(cx + rand(-18, 18), cy + rand(-18, 18));
  g.stroke();
  g.beginPath(); g.arc(cx, cy, 6, 0, Math.PI * 2); g.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

// many-languages palette: vivid mixed neon hues (cyan, magenta, lime, gold…)
const GLYPH_HUES = [0x2be4ff, 0xff2e97, 0x9be23a, 0xffd166, 0xb15cff, 0x4affc4, 0xff7a3d];
const gTmp = new THREE.Color();
function spawnConfusion() {
  glyphCount = MAX_GLYPHS;
  for (let i = 0; i < glyphCount; i++) {
    // erupt from around the crown of the tower
    const a = rand(0, Math.PI * 2), r = rand(0, TOP_RADIUS + 6);
    glyphPos[i * 3] = Math.cos(a) * r;
    glyphPos[i * 3 + 1] = towerTop + rand(-4, 8);
    glyphPos[i * 3 + 2] = Math.sin(a) * r;
    // scatter outward in every direction (the dispersal of tongues)
    const dir = new THREE.Vector3(rand(-1, 1), rand(-0.2, 1), rand(-1, 1)).normalize();
    const sp = rand(14, 40);
    gvx[i] = dir.x * sp; gvy[i] = dir.y * sp + rand(2, 10); gvz[i] = dir.z * sp;
    glyphLife[i] = rand(2.2, 5.0);
    gTmp.set(GLYPH_HUES[(Math.random() * GLYPH_HUES.length) | 0]);
    glyphCol[i * 3] = gTmp.r; glyphCol[i * 3 + 1] = gTmp.g; glyphCol[i * 3 + 2] = gTmp.b;
  }
  glyphGeo.setDrawRange(0, glyphCount);
  glyphGeo.attributes.position.needsUpdate = true;
  glyphGeo.attributes.color.needsUpdate = true;
}
function clearGlyphs() { glyphCount = 0; glyphGeo.setDrawRange(0, 0); }

// --- phase state machine ---------------------------------------------------
// rising → confusion → fallen → (reset) rising …
const PHASE = { RISING: "rising", CONFUSION: "confusion", FALLEN: "fallen" };
let phase = PHASE.RISING;
let phaseT = 0;             // seconds in current phase
let revealFrac = 0;        // 0..1 build progress during RISING

function resetCycle(startFallen) {
  buildTower();
  clearGlyphs();
  if (startFallen) {
    shatter();
    // fast-forward the rubble so it's already settled on the plain
    for (let s = 0; s < 90; s++) stepRubble(1 / 60);
    phase = PHASE.FALLEN; phaseT = 0; revealFrac = 1;
  } else {
    phase = PHASE.RISING; phaseT = 0; revealFrac = 0;
  }
  syncReadout();
}

// integrate one physics step for all bricks (ballistic + spin + floor bounce)
function stepRubble(dt) {
  for (let i = 0; i < brickCount; i++) {
    vy[i] += GRAV * dt;
    px[i] += vx[i] * dt; py[i] += vy[i] * dt; pz[i] += vz[i] * dt;
    // floor at ~brick half-height; settle with damping (rubble piling up)
    const floor = 0.8;                      // bricks are thin slabs (scaleY≈1.6)
    if (py[i] < floor) {
      py[i] = floor;
      vy[i] *= -0.28;                       // weak bounce
      vx[i] *= 0.62; vz[i] *= 0.62;         // ground friction
      wx[i] *= 0.5; wy[i] *= 0.5; wz[i] *= 0.5;
      if (Math.abs(vy[i]) < 1.2) vy[i] = 0;
    }
    // integrate spin: dq = quat(ω·dt) ⊗ q
    const ang = Math.hypot(wx[i], wy[i], wz[i]) * dt;
    if (ang > 1e-5) {
      const inv = 1 / (Math.hypot(wx[i], wy[i], wz[i]) || 1);
      SPIN_AXIS.set(wx[i] * inv, wy[i] * inv, wz[i] * inv);
      tmpQ.setFromAxisAngle(SPIN_AXIS, ang);
      tmpQ2.set(rqx[i], rqy[i], rqz[i], rqw[i]);
      tmpQ.multiply(tmpQ2);
      rqx[i] = tmpQ.x; rqy[i] = tmpQ.y; rqz[i] = tmpQ.z; rqw[i] = tmpQ.w;
    }
  }
}

// write all brick matrices for the current phase
function drawBricks() {
  const revealing = phase === PHASE.RISING;
  const falling = phase === PHASE.CONFUSION || phase === PHASE.FALLEN;
  let shown = 0;
  for (let i = 0; i < brickCount; i++) {
    if (revealing) {
      if (revealAt[i] > revealFrac) {
        // not yet placed — collapse to zero scale (invisible) but keep matrix valid
        dummy.position.set(homePos[i * 3], homePos[i * 3 + 1], homePos[i * 3 + 2]);
        dummy.quaternion.set(0, 0, 0, 1);
        dummy.scale.set(0.0001, 0.0001, 0.0001);
        dummy.updateMatrix();
        bricks.setMatrixAt(i, dummy.matrix);
        continue;
      }
      // pop-in ease: bricks settle from a hair above their slot
      const age = Math.min(1, (revealFrac - revealAt[i]) * 14);
      const ease = 1 - (1 - age) * (1 - age);
      const drop = (1 - ease) * 3.0;
      dummy.position.set(homePos[i * 3], homePos[i * 3 + 1] + drop, homePos[i * 3 + 2]);
      dummy.rotation.set(0, homeYaw[i], 0);
      dummy.scale.set(brickW[i] * ease, 1.6 * ease, brickD[i] * ease);
      dummy.updateMatrix();
      bricks.setMatrixAt(i, dummy.matrix);
      shown++;
    } else if (falling) {
      dummy.position.set(px[i], py[i], pz[i]);
      // combine the brick's facing yaw with its tumbling spin (no per-frame alloc)
      tmpEuler.set(0, homeYaw[i], 0);
      tmpQ.setFromEuler(tmpEuler);
      tmpQ2.set(rqx[i], rqy[i], rqz[i], rqw[i]);
      tmpQ2.multiply(tmpQ);
      dummy.quaternion.copy(tmpQ2);
      dummy.scale.set(brickW[i], 1.6, brickD[i]);
      dummy.updateMatrix();
      bricks.setMatrixAt(i, dummy.matrix);
      shown++;
    }
  }
  bricks.instanceMatrix.needsUpdate = true;
  return shown;
}

// --- panel + nav -----------------------------------------------------------
const wrap = document.getElementById("acts");
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => setPreset(i));
  wrap.appendChild(b);
  return b;
});
const nameEl = document.getElementById("act-name");
const phaseEl = document.getElementById("phase");
const brickEl = document.getElementById("bricks");

function syncReadout() {
  if (phaseEl) phaseEl.textContent = phase;
  if (brickEl) brickEl.textContent = brickCount;
  if (nameEl) nameEl.textContent = presetName;
}

function applyPreset(p) {
  buildSpeed = p.pace;
  confusionHold = p.confusionHold;
  fallenHold = p.fallenHold;
  shape = p.shape;
  // mirror buildSpeed into the slider so the UI agrees
  const sp = document.getElementById("speed");
  if (sp) {
    sp.value = String(buildSpeed);
    const out = document.querySelector('[data-val="speed"]');
    if (out) out.textContent = buildSpeed.toFixed(2) + "×";
  }
}

function setPreset(i) {
  presetIdx = ((i % PRESETS.length) + PRESETS.length) % PRESETS.length;
  presetName = PRESETS[presetIdx][0];
  const p = PRESETS[presetIdx][1];
  applyPreset(p);
  chips.forEach((c, j) => c.classList.toggle("active", j === presetIdx));
  resetCycle(p.startFallen);
}

bindRange("tiers", (v) => { tiers = Math.round(v); resetCycle(false); }, (v) => `${Math.round(v)}`);
bindRange("speed", (v) => { buildSpeed = v; }, (v) => `${v.toFixed(2)}×`);
const resetBtn = document.getElementById("reset");
if (resetBtn) resetBtn.addEventListener("click", () => resetCycle(PRESETS[presetIdx][1].startFallen));

// ↑↓ cycles the act (ascent / confusion / ruin / ziggurat) in place.
setVariantCycler((d) => { setPreset(presetIdx + d); return presetName; });

// --- boot ------------------------------------------------------------------
setPreset(0);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  phaseT += dt;

  if (phase === PHASE.RISING) {
    // climb build progress; ~6s base reveal, scaled by buildSpeed
    revealFrac += dt * (buildSpeed / 6);
    if (revealFrac >= 1) {
      revealFrac = 1;
      phase = PHASE.CONFUSION; phaseT = 0;
      spawnConfusion();   // tongues scatter
      shatter();          // and the tower destabilizes
      syncReadout();
    }
  } else if (phase === PHASE.CONFUSION) {
    stepRubble(dt);
    if (phaseT >= confusionHold) { phase = PHASE.FALLEN; phaseT = 0; syncReadout(); }
  } else { // FALLEN
    stepRubble(dt);   // rubble keeps settling
    if (phaseT >= fallenHold) { resetCycle(false); }   // rebuild → loop
  }

  // advance + fade the glyph sparks (in-place buffer update)
  if (glyphCount > 0) {
    let anyAlive = false;
    for (let i = 0; i < glyphCount; i++) {
      if (glyphLife[i] <= 0) continue;
      glyphLife[i] -= dt;
      gvy[i] += GRAV * 0.25 * dt;          // gentle drift downward
      glyphPos[i * 3] += gvx[i] * dt;
      glyphPos[i * 3 + 1] += gvy[i] * dt;
      glyphPos[i * 3 + 2] += gvz[i] * dt;
      const fade = Math.max(0, Math.min(1, glyphLife[i] / 1.2));
      if (glyphLife[i] > 0) anyAlive = true;
      // fade dying sparks by darkening toward black (additive blend → they vanish)
      if (fade < 1) {
        const k = 0.90 + 0.10 * fade;
        glyphCol[i * 3] *= k; glyphCol[i * 3 + 1] *= k; glyphCol[i * 3 + 2] *= k;
      }
    }
    glyphGeo.attributes.position.needsUpdate = true;
    glyphGeo.attributes.color.needsUpdate = true;
    if (!anyAlive) clearGlyphs();
  }

  drawBricks();
  controls.update();
  renderer.render(scene, camera);
});

// diagnostics hook for the gallery harness.
window.__diag = () => JSON.stringify({ phase, bricks: brickCount, tiers });
