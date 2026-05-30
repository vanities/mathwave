// lsystem.js — L-systems (Aristid Lindenmayer, 1968): a parallel string-rewriting
// grammar that models the branching growth of plants. Start from an axiom string
// and apply production rules N times *in parallel* (every symbol rewritten at once),
// so the string expands exponentially. Then a 3D turtle reads the symbols and draws:
//   F / G  move forward one step, drawing a segment
//   f      move forward one step WITHOUT drawing
//   + / -  yaw   (turn left / right)  — rotate about the turtle's up axis
//   & / ^  pitch (down / up)          — rotate about the turtle's left axis
//   \ / /  roll  (left / right)       — rotate about the turtle's heading axis
//   [ / ]  push / pop turtle state (position, orientation, depth)
//
// Orientation is carried as a Quaternion and every turn multiplies it by a
// quaternion about the appropriate *local* axis — this avoids the gimbal lock
// plain Euler angles hit on pitch (cf. footguns). All segments go into ONE merged
// BufferGeometry (LineSegments) → a single draw call. Growth is animated by
// climbing the geometry's draw range so the plant draws on, then loops/regrows.
// Color ramps green→amber by branch depth on near-black (no purple dominance).
//
// Ref: Prusinkiewicz & Lindenmayer, "The Algorithmic Beauty of Plants" (1990).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const DEG = Math.PI / 180;

// Hard caps so deep iterations never freeze the tab (footgun guard).
const MAX_STRING = 200000;   // characters of expanded L-system
const MAX_SEGMENTS = 60000;  // drawn line segments (also caps the buffer)

// --- presets ---------------------------------------------------------------
// Each: axiom, rules map, base angle (deg), iteration cap, and a roll injected
// at every branch so otherwise-flat rules unfurl into three dimensions.
const PRESETS = [
  // bushy 3D plant — the classic, rolled into 3D
  ["bush", {
    axiom: "F", rules: { F: "FF-[-F+F+F]+[+F-F-F]" },
    angle: 22.5, maxIter: 5, roll: 28,
  }],
  // fractal tree — symmetric canopy that spreads in 3D
  ["tree", {
    axiom: "X", rules: { X: "F[&+X][&-X][^/X]FX", F: "FF" },
    angle: 26, maxIter: 6, roll: 90,
  }],
  // seaweed / Penrose-ish — sparse leaning fronds
  ["seaweed", {
    axiom: "F", rules: { F: "F[+F]F[-F][&F]" },
    angle: 24, maxIter: 5, roll: 64,
  }],
  // Sierpinski-ish weave — dense lattice unrolled into 3D
  ["sierpinski", {
    axiom: "F-G-G", rules: { F: "F-G+F+G-F", G: "GG" },
    angle: 120, maxIter: 6, roll: 14,
  }],
];

// --- scene -----------------------------------------------------------------
const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x050a07, 0.012);

const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 2000);
camera.position.set(0, 22, 70);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion; controls.autoRotateSpeed = 0.35;
controls.minDistance = 12; controls.maxDistance = 320;
controls.target.set(0, 22, 0);

scene.add(new THREE.AmbientLight(0x274033, 0.8));
const key = new THREE.DirectionalLight(0xeaffd6, 0.6); key.position.set(8, 20, 12); scene.add(key);
addGrid(scene, { size: 220, divisions: 44, y: 0 });
addSun(scene, { scale: 80, position: [0, 36, -240] });

// --- one merged line geometry (pre-allocated to the cap) -------------------
const positions = new Float32Array(MAX_SEGMENTS * 2 * 3);
const colors = new Float32Array(MAX_SEGMENTS * 2 * 3);
const geom = new THREE.BufferGeometry();
const posAttr = new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage);
const colAttr = new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage);
geom.setAttribute("position", posAttr);
geom.setAttribute("color", colAttr);
geom.setDrawRange(0, 0);
const lines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({
  vertexColors: true, transparent: true, opacity: 0.95,
  blending: THREE.AdditiveBlending, depthWrite: false,
}));
lines.frustumCulled = false;
scene.add(lines);

// --- state -----------------------------------------------------------------
let presetIdx = 0;
let presetName = PRESETS[0][0];
let iterations = 4;      // from slider
let angleScale = 1.0;    // multiplier on the preset's base angle (slider)
let totalSegments = 0;
let maxDepthSeen = 1;
let grown = 0;           // segments currently revealed
let growSpeed = 1;       // segments revealed per frame

// scratch — reused so we never reallocate per symbol
const seg = [];          // flat: x1,y1,z1, x2,y2,z2, depth  (×7 per segment)
const HEADING = new THREE.Vector3(0, 1, 0); // forward grows along local +Y
const UP = new THREE.Vector3(0, 0, 1);      // yaw axis
const LEFT = new THREE.Vector3(1, 0, 0);    // pitch axis (roll is about HEADING)
const tmpQ = new THREE.Quaternion();
const tmpV = new THREE.Vector3();

// expand the axiom by applying rules `iters` times, capping length.
function expand(preset, iters) {
  let s = preset.axiom;
  for (let i = 0; i < iters; i++) {
    let next = "";
    for (let c = 0; c < s.length; c++) {
      const ch = s[c];
      const r = preset.rules[ch];
      next += r !== undefined ? r : ch;
      if (next.length > MAX_STRING) return next.slice(0, MAX_STRING);
    }
    s = next;
  }
  return s;
}

// interpret the expanded string with a 3D turtle, filling `seg`.
function turtle(str, preset) {
  const angle = preset.angle * angleScale * DEG;
  const roll = preset.roll * DEG;

  const pos = new THREE.Vector3(0, 0, 0);
  const quat = new THREE.Quaternion();   // identity → heading == +Y
  const stack = [];
  let depth = 1;

  seg.length = 0;
  totalSegments = 0;
  maxDepthSeen = 1;

  const rot = (axis, ang) => { tmpQ.setFromAxisAngle(axis, ang); quat.multiply(tmpQ); };
  const forward = (draw) => {
    tmpV.copy(HEADING).applyQuaternion(quat);       // local +Y in world space
    const x1 = pos.x, y1 = pos.y, z1 = pos.z;
    pos.add(tmpV);                                   // step length == 1
    if (draw && totalSegments < MAX_SEGMENTS) {
      seg.push(x1, y1, z1, pos.x, pos.y, pos.z, depth);
      totalSegments++;
    }
  };

  for (let i = 0; i < str.length; i++) {
    switch (str[i]) {
      case "F": case "G": forward(true); break;
      case "f": forward(false); break;
      case "+": rot(UP, angle); break;
      case "-": rot(UP, -angle); break;
      case "&": rot(LEFT, angle); break;
      case "^": rot(LEFT, -angle); break;
      case "\\": rot(HEADING, roll); break;
      case "/": rot(HEADING, -roll); break;
      case "[":
        stack.push(pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w, depth);
        depth++;
        if (depth > maxDepthSeen) maxDepthSeen = depth;
        rot(HEADING, roll);     // inject roll at each branch → flat rules become 3D
        break;
      case "]": {
        if (stack.length >= 8) {
          depth = stack.pop();
          quat.set(stack[stack.length - 4], stack[stack.length - 3], stack[stack.length - 2], stack[stack.length - 1]);
          stack.length -= 4;
          pos.set(stack[stack.length - 3], stack[stack.length - 2], stack[stack.length - 1]);
          stack.length -= 3;
        }
        break;
      }
      default: break;          // X, Y and other control symbols draw nothing
    }
    if (totalSegments >= MAX_SEGMENTS) break;
  }
}

// vivid depth ramp on near-black: green base → chartreuse → amber tips.
const cBase = new THREE.Color(0x2bd46a);
const cMid = new THREE.Color(0x9be23a);
const cTip = new THREE.Color(0xffb142);
const tmpColor = new THREE.Color();
function depthColor(depth) {
  const t = maxDepthSeen > 1 ? (depth - 1) / (maxDepthSeen - 1) : 0;
  if (t < 0.5) tmpColor.copy(cBase).lerp(cMid, t / 0.5);
  else tmpColor.copy(cMid).lerp(cTip, (t - 0.5) / 0.5);
  return tmpColor;
}

// rebuild geometry from the current preset / iterations / angle, then regrow.
function rebuild() {
  const preset = PRESETS[presetIdx][1];
  const iters = Math.min(iterations, preset.maxIter);
  turtle(expand(preset, iters), preset);

  // center horizontally, sit the trunk base on the grid, normalize height.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < totalSegments; i++) {
    const o = i * 7;
    for (let k = 0; k < 2; k++) {
      const x = seg[o + k * 3], y = seg[o + k * 3 + 1], z = seg[o + k * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
  }
  let scaleF = 1, cx = 0, cz = 0, baseY = 0;
  if (totalSegments > 0) {
    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-4);
    scaleF = 44 / span;            // tallest extent → ~44 units
    cx = (minX + maxX) / 2;
    cz = (minZ + maxZ) / 2;
    baseY = minY;
  }

  for (let i = 0; i < totalSegments; i++) {
    const o = i * 7;
    const col = depthColor(seg[o + 6]);
    const vo = i * 6;
    positions[vo]     = (seg[o]     - cx) * scaleF;
    positions[vo + 1] = (seg[o + 1] - baseY) * scaleF;
    positions[vo + 2] = (seg[o + 2] - cz) * scaleF;
    positions[vo + 3] = (seg[o + 3] - cx) * scaleF;
    positions[vo + 4] = (seg[o + 4] - baseY) * scaleF;
    positions[vo + 5] = (seg[o + 5] - cz) * scaleF;
    colors[vo]     = col.r; colors[vo + 1] = col.g; colors[vo + 2] = col.b;
    colors[vo + 3] = col.r; colors[vo + 4] = col.g; colors[vo + 5] = col.b;
  }
  posAttr.needsUpdate = true;
  colAttr.needsUpdate = true;

  grown = 0;
  growSpeed = Math.max(1, Math.ceil(totalSegments / 240));   // ~4s reveal at 60fps
  if (segEl) segEl.textContent = totalSegments;
}

// --- panel + nav -----------------------------------------------------------
const wrap = document.getElementById("species");
const chips = PRESETS.map(([label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => { setPreset(i); });
  wrap.appendChild(b);
  return b;
});
const nameEl = document.getElementById("species-name");
const segEl = document.getElementById("segs");

function setPreset(i) {
  presetIdx = ((i % PRESETS.length) + PRESETS.length) % PRESETS.length;
  presetName = PRESETS[presetIdx][0];
  const cap = PRESETS[presetIdx][1].maxIter;
  // clamp the iterations slider to this preset's cap (no TDZ — iterations declared above)
  const el = document.getElementById("iters");
  if (el) {
    el.max = String(cap);
    if (Number(el.value) > cap) el.value = String(cap);
    iterations = Math.min(Math.round(Number(el.value)), cap);
    const out = document.querySelector('[data-val="iters"]');
    if (out) out.textContent = String(iterations);
  }
  chips.forEach((c, j) => c.classList.toggle("active", j === presetIdx));
  if (nameEl) nameEl.textContent = presetName;
  rebuild();
}

bindRange("iters", (v) => { iterations = Math.round(v); rebuild(); }, (v) => `${Math.round(v)}`);
bindRange("angle", (v) => { angleScale = v; rebuild(); }, (v) => `${v.toFixed(2)}×`);
const regrowBtn = document.getElementById("regrow");
if (regrowBtn) regrowBtn.addEventListener("click", () => { grown = 0; });

// ↑↓ cycles species in place; returns the new label for the kiosk toast.
setVariantCycler((d) => { setPreset(presetIdx + d); return presetName; });

// --- boot ------------------------------------------------------------------
setPreset(0);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  if (grown < totalSegments) {
    grown = Math.min(totalSegments, grown + growSpeed);
    geom.setDrawRange(0, grown * 2);     // two vertices per segment
  }
  controls.update();
  renderer.render(scene, camera);
});

// diagnostics hook for the gallery harness.
window.__diag = () => JSON.stringify({ preset: presetName, segments: totalSegments });
