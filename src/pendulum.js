// pendulum.js — the double pendulum: the textbook example of deterministic chaos.
// Two rigid arms; the lower one's motion is famously unpredictable. We integrate
// the EXACT Lagrangian equations of motion (not a fake) with RK4, and run MANY
// pendulums whose start angles differ by one part in ten-thousand — they track
// together, then explosively diverge. That spreading rainbow IS the butterfly
// effect (sensitive dependence on initial conditions). Trails fade behind each
// bob. ↑↓ changes how many pendulums.
//
// EOM ref: standard double-pendulum Lagrangian (e.g. Wikipedia "Double pendulum").

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { makeRenderer, onResize, loop, fpsMeter, liftVeil, bindRange, ramp, reducedMotion, addGrid, addSun, setVariantCycler } from "./common.js";

const canvas = document.getElementById("scene");
const renderer = makeRenderer(canvas);
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0a0118, 0.02);

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 0, 16);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.dampingFactor = 0.07;
controls.minDistance = 6; controls.maxDistance = 50;
controls.autoRotate = false;
controls.target.set(0, -1, 0);

scene.add(new THREE.AmbientLight(0x3a2466, 0.9));
addGrid(scene, { size: 40, divisions: 20, y: -8 });
addSun(scene, { scale: 34, position: [0, 6, -60] });

// ---------- physics constants ----------
const g = 9.81, L1 = 2.0, L2 = 2.0, m1 = 1.0, m2 = 1.0;

// derivative of [θ1, ω1, θ2, ω2] — exact double-pendulum EOM
function deriv(s, out) {
  const [t1, w1, t2, w2] = s;
  const d = t1 - t2, sd = Math.sin(d), cd = Math.cos(d);
  const den = 2 * m1 + m2 - m2 * Math.cos(2 * d);
  out[0] = w1;
  out[1] = (-g * (2 * m1 + m2) * Math.sin(t1) - m2 * g * Math.sin(t1 - 2 * t2)
            - 2 * sd * m2 * (w2 * w2 * L2 + w1 * w1 * L1 * cd)) / (L1 * den);
  out[2] = w2;
  out[3] = (2 * sd * (w1 * w1 * L1 * (m1 + m2) + g * (m1 + m2) * Math.cos(t1)
            + w2 * w2 * L2 * m2 * cd)) / (L2 * den);
}

const k1 = [0,0,0,0], k2 = [0,0,0,0], k3 = [0,0,0,0], k4 = [0,0,0,0], tmp = [0,0,0,0];
function rk4(s, h) {
  deriv(s, k1);
  for (let i = 0; i < 4; i++) tmp[i] = s[i] + k1[i] * h / 2; deriv(tmp, k2);
  for (let i = 0; i < 4; i++) tmp[i] = s[i] + k2[i] * h / 2; deriv(tmp, k3);
  for (let i = 0; i < 4; i++) tmp[i] = s[i] + k3[i] * h;     deriv(tmp, k4);
  for (let i = 0; i < 4; i++) s[i] += (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) * h / 6;
}

// ---------- ensemble of nearly-identical pendulums ----------
let COUNT = 24;
let states = [];
let arms, bobs, trails = [];
const TRAIL = 120;

function clearAll() {
  if (arms) scene.remove(arms);
  if (bobs) scene.remove(bobs);
  trails.forEach((t) => scene.remove(t)); trails = [];
}
function build() {
  clearAll();
  states = [];
  for (let i = 0; i < COUNT; i++) {
    // all start near θ1=θ2=2.4 rad, perturbed by 1e-4·i → chaos amplifies it
    states.push([2.4 + i * 1e-4, 0, 2.4, 0]);
  }
  // arms: 2 segments × COUNT = lines
  const armGeo = new THREE.BufferGeometry();
  armGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(COUNT * 4 * 3), 3).setUsage(THREE.DynamicDrawUsage));
  armGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(COUNT * 4 * 3), 3));
  arms = new THREE.LineSegments(armGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 }));
  arms.frustumCulled = false; scene.add(arms);
  // bobs as points
  const bobGeo = new THREE.BufferGeometry();
  bobGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(COUNT * 3), 3).setUsage(THREE.DynamicDrawUsage));
  bobGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(COUNT * 3), 3));
  bobs = new THREE.Points(bobGeo, new THREE.PointsMaterial({ vertexColors: true, size: 0.28, sizeAttenuation: true }));
  bobs.frustumCulled = false; scene.add(bobs);
  // one fading trail line per pendulum (the lower bob's path)
  for (let i = 0; i < COUNT; i++) {
    const c = ramp(i / COUNT);
    const tg = new THREE.BufferGeometry();
    tg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRAIL * 3), 3).setUsage(THREE.DynamicDrawUsage));
    const tl = new THREE.Line(tg, new THREE.LineBasicMaterial({ color: new THREE.Color(c[0], c[1], c[2]), transparent: true, opacity: 0.5, blending: THREE.AdditiveBlending, depthWrite: false }));
    tl.frustumCulled = false; tl.userData = { n: 0 };
    scene.add(tl); trails.push(tl);
  }
}

let speed = 1, playing = !reducedMotion;
function update(dt) {
  const h = Math.min(dt, 0.02) * 2.2 * speed;
  const ap = arms.geometry.attributes.position.array;
  const ac = arms.geometry.attributes.color.array;
  const bp = bobs.geometry.attributes.position.array;
  const bc = bobs.geometry.attributes.color.array;
  for (let i = 0; i < COUNT; i++) {
    if (playing) { rk4(states[i], h); }
    const [t1, , t2] = states[i];
    const x1 = L1 * Math.sin(t1), y1 = -L1 * Math.cos(t1);
    const x2 = x1 + L2 * Math.sin(t2), y2 = y1 - L2 * Math.cos(t2);
    const c = ramp(i / COUNT);
    // arm segments: pivot→bob1, bob1→bob2
    const a = i * 12;
    ap[a]=0; ap[a+1]=0; ap[a+2]=0;       ap[a+3]=x1; ap[a+4]=y1; ap[a+5]=0;
    ap[a+6]=x1; ap[a+7]=y1; ap[a+8]=0;   ap[a+9]=x2; ap[a+10]=y2; ap[a+11]=0;
    for (let v = 0; v < 4; v++) { ac[a+v*3]=c[0]; ac[a+v*3+1]=c[1]; ac[a+v*3+2]=c[2]; }
    bp[i*3]=x2; bp[i*3+1]=y2; bp[i*3+2]=0;
    bc[i*3]=c[0]; bc[i*3+1]=c[1]; bc[i*3+2]=c[2];
    // trail
    const tl = trails[i], tp = tl.geometry.attributes.position.array;
    if (playing) {
      if (tl.userData.n < TRAIL) { tp.set([x2, y2, 0], tl.userData.n * 3); tl.userData.n++; }
      else { tp.copyWithin(0, 3); tp.set([x2, y2, 0], (TRAIL - 1) * 3); }
      tl.geometry.setDrawRange(0, tl.userData.n);
      tl.geometry.attributes.position.needsUpdate = true;
    }
  }
  arms.geometry.attributes.position.needsUpdate = true;
  arms.geometry.attributes.color.needsUpdate = true;
  bobs.geometry.attributes.position.needsUpdate = true;
  bobs.geometry.attributes.color.needsUpdate = true;
}

// ---------- panel ----------
bindRange("speed", (v) => { speed = v; }, (v) => v.toFixed(2) + "×");
bindRange("count", (v) => { COUNT = Math.round(v); build(); }, (v) => `${Math.round(v)}`);
const playBtn = document.getElementById("play");
playBtn.classList.toggle("active", playing);
playBtn.addEventListener("click", () => { playing = !playing; playBtn.textContent = playing ? "pause" : "play"; playBtn.classList.toggle("active", playing); });
document.getElementById("reset").addEventListener("click", build);
const COUNTS = [1, 8, 24, 60, 120];
setVariantCycler((d) => { const i = Math.max(0, COUNTS.indexOf(COUNT)); const ni = (i + d + COUNTS.length) % COUNTS.length; COUNT = COUNTS[ni]; const el = document.getElementById("count"); if (el) el.value = COUNT; build(); return COUNT + " pendulums"; });

// ---------- boot ----------
build();
liftVeil();
onResize(renderer, camera);
const meter = fpsMeter(document.getElementById("fps"));
const nEl = document.getElementById("nb");
loop((dt) => { meter(dt); update(dt); if (nEl) nEl.textContent = COUNT; controls.update(); renderer.render(scene, camera); });
