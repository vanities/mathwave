// vectorfield.js — thousands of particles advected through a 3D vector field
// V(x,y,z). Each particle draws a short streak (prev → now) colored by speed,
// so you read both the flow direction and its velocity. Pure 3D calculus made
// visible: divergence-free swirls, vortices, sources/sinks, the Lorenz flow.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler,
} from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.014);

const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 600);
camera.position.set(20, 14, 24);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = !reducedMotion;
controls.autoRotateSpeed = 0.5;
controls.minDistance = 6;
controls.maxDistance = 140;

addGrid(scene, { size: 70, divisions: 35, y: -18 });
addSun(scene, { scale: 52, position: [0, 18, -80] });

// ---------- vector fields: (x,y,z) -> writes vx,vy,vz into out ----------
const FIELDS = {
  // Arnold–Beltrami–Childress flow — chaotic, divergence-free streamlines
  abc: (x, y, z, o) => {
    const s = 0.32;
    o.x = Math.sin(z * s) + Math.cos(y * s);
    o.y = Math.sin(x * s) + Math.cos(z * s);
    o.z = Math.sin(y * s) + Math.cos(x * s);
  },
  // tornado: rotate around Y, drift up where it's calm
  vortex: (x, y, z, o) => {
    const r = Math.hypot(x, z) + 0.001;
    o.x = -z / r * 2.2;
    o.y = 0.9 + Math.sin(r * 0.25) * 0.5;
    o.z = x / r * 2.2;
  },
  // the Lorenz attractor, treated as a flow field
  lorenz: (x, y, z, o) => {
    const s = 10, rr = 28, b = 8 / 3, k = 0.09;
    o.x = s * (y - x) * k;
    o.y = (x * (rr - z) - y) * k;
    o.z = (x * y - b * z) * k;
  },
  // double-well saddle: pulls toward two basins
  saddle: (x, y, z, o) => {
    o.x = (x - x * x * x * 0.04) * 0.5 - z * 0.3;
    o.y = -y * 0.8 + Math.sin(x * 0.3) * 0.6;
    o.z = (z - z * z * z * 0.04) * 0.5 + x * 0.3;
  },
  // source/sink dipole: spray out of one pole, into the other
  dipole: (x, y, z, o) => {
    const ax = x, ay = y - 6, az = z;
    const bx = x, by = y + 6, bz = z;
    const ra = Math.pow(ax * ax + ay * ay + az * az + 1, 1.5);
    const rb = Math.pow(bx * bx + by * by + bz * bz + 1, 1.5);
    o.x = (ax / ra - bx / rb) * 14;
    o.y = (ay / ra - by / rb) * 14;
    o.z = (az / ra - bz / rb) * 14;
  },
};
let fieldName = "abc";
let field = FIELDS.abc;

// ---------- particles ----------
const COUNT = 9000;
const SPAWN = 22;       // half-size of spawn cube
const BOUND = 34;       // recycle past this radius
const MAXLIFE = 7;      // seconds before forced respawn (keeps it fresh)

const cur = new Float32Array(COUNT * 3);
const life = new Float32Array(COUNT);
// two verts per particle: a streak from prev → cur
const segPos = new Float32Array(COUNT * 2 * 3);
const segCol = new Float32Array(COUNT * 2 * 3);

const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(segPos, 3).setUsage(THREE.DynamicDrawUsage));
geo.setAttribute("color", new THREE.BufferAttribute(segCol, 3).setUsage(THREE.DynamicDrawUsage));
const lines = new THREE.LineSegments(
  geo,
  new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.92, blending: THREE.AdditiveBlending, depthWrite: false })
);
scene.add(lines);

function spawn(i) {
  cur[i * 3] = (Math.random() * 2 - 1) * SPAWN;
  cur[i * 3 + 1] = (Math.random() * 2 - 1) * SPAWN;
  cur[i * 3 + 2] = (Math.random() * 2 - 1) * SPAWN;
  life[i] = Math.random() * MAXLIFE;
}
for (let i = 0; i < COUNT; i++) spawn(i);

const v = { x: 0, y: 0, z: 0 };
let flow = 1;
let playing = !reducedMotion;
const MAXSPEED = 3.5;

function update(dt) {
  const step = dt * flow;
  for (let i = 0; i < COUNT; i++) {
    const ix = i * 3;
    let x = cur[ix], y = cur[ix + 1], z = cur[ix + 2];
    field(x, y, z, v);
    const sp = Math.hypot(v.x, v.y, v.z) + 1e-5;
    const nx = x + v.x * step, ny = y + v.y * step, nz = z + v.z * step;

    // streak of a VISIBLE length along the flow (the per-step move is sub-pixel,
    // so we draw a fixed comet tail behind the head; faster flow → longer tail).
    const inv = 1 / sp;
    const L = 0.6 + Math.min(sp / MAXSPEED, 1) * 1.6;
    const s = i * 6;
    segPos[s] = nx - v.x * inv * L; segPos[s + 1] = ny - v.y * inv * L; segPos[s + 2] = nz - v.z * inv * L;
    segPos[s + 3] = nx; segPos[s + 4] = ny; segPos[s + 5] = nz;
    const c = ramp(Math.min(sp / MAXSPEED, 1));
    segCol[s] = c[0] * 0.12; segCol[s + 1] = c[1] * 0.12; segCol[s + 2] = c[2] * 0.12; // dim tail
    segCol[s + 3] = c[0]; segCol[s + 4] = c[1]; segCol[s + 5] = c[2];                  // bright head

    cur[ix] = nx; cur[ix + 1] = ny; cur[ix + 2] = nz;
    life[i] += dt;
    if (life[i] > MAXLIFE || Math.hypot(nx, ny, nz) > BOUND) spawn(i);
  }
  geo.attributes.position.needsUpdate = true;
  geo.attributes.color.needsUpdate = true;
}

// ---------- panel ----------
const FIELD_LABELS = [["abc", "abc flow"], ["vortex", "vortex"], ["lorenz", "lorenz"], ["saddle", "saddle"], ["dipole", "dipole"]];
const wrap = document.getElementById("fields");
const nameEl = document.getElementById("fieldname");
FIELD_LABELS.forEach(([key, label], i) => {
  const b = document.createElement("button");
  b.className = "chip" + (i === 0 ? " active" : "");
  b.textContent = label;
  b.addEventListener("click", () => {
    fieldName = key; field = FIELDS[key];
    nameEl.textContent = key;
    for (let j = 0; j < COUNT; j++) spawn(j);  // reseed for the new field
    wrap.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === b));
  });
  wrap.appendChild(b);
});

bindRange("speed", (val) => { flow = val; }, (val) => val.toFixed(2) + "×");

const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.addEventListener("click", () => {
  playing = !playing;
  playBtn.textContent = playing ? "pause" : "play";
  playBtn.classList.toggle("active", playing);
});
document.getElementById("respawn").addEventListener("click", () => { for (let j = 0; j < COUNT; j++) spawn(j); });

const home = camera.position.clone();
document.getElementById("reset").addEventListener("click", () => {
  camera.position.copy(home); controls.target.set(0, 0, 0);
});

// ↑/↓ cycle the fields
let _variantIdx = 0;
setVariantCycler((d) => {
  _variantIdx = (_variantIdx + d + FIELD_LABELS.length) % FIELD_LABELS.length;
  wrap.children[_variantIdx].click();
  return FIELD_LABELS[_variantIdx][1];
});

// ---------- boot ----------
update(0.016);
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));

loop((dt) => {
  meter(dt);
  if (playing) update(dt);
  controls.update();
  renderer.render(scene, camera);
});
